# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from collections.abc import Iterable, Sequence
import dataclasses
import functools
import math
import threading
from typing import Any

import jax
from jax import lax
from jax._src import callback
from jax._src import core as jax_core
from jax._src.lax.control_flow import for_loop
from jax._src import linear_util as lu
from jax._src.pallas.mosaic import primitives as mosaic_primitives
from jax._src.pallas.mosaic import core as mosaic_core
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src import pjit
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax._src.util import (
    safe_map,
    safe_zip,
    split_list
)
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp
import numpy as np


map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

Grid = pallas_core.Grid
TupleGrid = pallas_core.TupleGrid
GridSpec = pallas_core.GridSpec
BlockMapping = pallas_core.BlockMapping
GridMapping = pallas_core.GridMapping
BlockSpec = pallas_core.BlockSpec
BlockSpecTree = pallas_core.BlockSpecTree
NoBlockSpec = pallas_core.NoBlockSpec
no_block_spec = pallas_core.no_block_spec
ScratchShapeTree = pallas_core.ScratchShapeTree
CostEstimate = pallas_core.CostEstimate


@dataclasses.dataclass(frozen=True)
class TPUInterpretParams:
  pass


class Semaphore:
  def __init__(self):
    self.cv = threading.Condition()

    # TODO(jburnim): Make this an array.
    self.counts = collections.defaultdict(int)

  def signal(self, inc, device_id):
    with self.cv:
      self.counts[device_id] += inc
      self.cv.notify_all()

  def wait(self, value, device_id):
    with self.cv:
      while self.counts[device_id] < value:
        self.cv.wait()
      self.counts[device_id] -= value


@dataclasses.dataclass(frozen=True)
class SharedMemory:
  # (memory_space, buffer_id, device_id) -> NumPy array
  # TODO(jburnim): Handle Megacore.
  mem: dict = dataclasses.field(default_factory=dict)

  # semaphore_id -> Semaphore
  sem: dict = dataclasses.field(default_factory=dict)

  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

  next_buffer_id: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(lambda: 100))
  next_semaphore_id: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(lambda: 2000))

# TODO(jburnim): Do we want to support multiple instances of SharedMemory?
# Maybe for running multiple distinct interpreted computations in parallel?
_shared_memory = None
_shared_memory_init_lock = threading.Lock()

def _get_shared_memory() -> SharedMemory:
  global _shared_memory
  if _shared_memory is None:
    with _shared_memory_init_lock:
      if _shared_memory is None:
        _shared_memory = SharedMemory()
  return _shared_memory

def _clear_shared_memory():
  global _shared_memory
  with _shared_memory_init_lock:
    _shared_memory = None

def _allocate_buffer(device_id, memory_space, val):
  device_id = int(device_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  val = np.array(val)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    buffer_id = shared_memory.next_buffer_id[device_id]
    shared_memory.next_buffer_id[device_id] = buffer_id + 1
    shared_memory.mem[(memory_space, buffer_id, device_id)] = val

  # TODO(jburnim): Raise an error if buffer_id is too big for int16.
  return np.int16(buffer_id)

def _deallocate_buffer(device_id, memory_space, buffer_id):
  device_id = int(device_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    # TODO(jburnim): Error if buffer doesn't exist?
    shared_memory.mem.pop((memory_space, buffer_id, device_id), None)

def _allocate_semaphores(device_id, shape):
  device_id = int(device_id)
  shape = tuple(map(int, shape))
  num_semaphores = math.prod(shape)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    semaphore_id = shared_memory.next_semaphore_id[device_id]
    shared_memory.next_semaphore_id[device_id] = semaphore_id + num_semaphores
    for i in range(semaphore_id, semaphore_id + num_semaphores):
      if not i in shared_memory.sem:
        shared_memory.sem[i] = Semaphore()

  # NOTE: For now, we use a relatively uncommon datatype (int16) for
  # semaphore (and buffer) IDs, so these values are more easily identifiable
  # in kernels.
  #
  # TODO(jburnim): Raise an error if any IDs are too big for int16.
  return np.int16(
      range(semaphore_id, semaphore_id + num_semaphores)
  ).reshape(shape)


TPU_MEMORY_SPACE_IDXS : dict[mosaic_core.TPUMemorySpace | None, int] = {
    v: i for i, v in enumerate(mosaic_core.TPUMemorySpace)}
TPU_MEMORY_SPACE_NAMES = {
    i: v.value for i, v in enumerate(mosaic_core.TPUMemorySpace)}

# Default to VMEM when no memory space is specified.
TPU_MEMORY_SPACE_IDXS[None] = (
    TPU_MEMORY_SPACE_IDXS[mosaic_core.TPUMemorySpace.VMEM])

def get_barrier_semaphore(device_id, collective_id):
  del device_id
  collective_id = int(collective_id)

  # TODO(jburnim): Check/fix so that IDs for barrier semaphores do not conflict
  # with IDs for regular or DMA semaphores.  (For example, store them in a
  # different table.)
  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    semaphore_id = collective_id
    if not semaphore_id in shared_memory.sem:
      shared_memory.sem[semaphore_id] = Semaphore()

  return np.int16(semaphore_id)

def _transform_slice_or_index(slice_or_idx):
  if isinstance(slice_or_idx, int):
    return slice_or_idx
  else:
    start, size, stride = (
        int(slice_or_idx.start),
        int(slice_or_idx.size),
        int(slice_or_idx.stride))
    return slice(start, start + size * stride, stride)

def _compose_slice_or_index(slice_or_idx1, slice_or_idx2):
  ret = []
  i = 0
  j = 0
  while True:
    if i == len(slice_or_idx1):
      ret.extend(slice_or_idx2[j:])
      return tuple(ret)
    elif j == len(slice_or_idx2):
      ret.extend(slice_or_idx1[i:])
      return tuple(ret)
    elif isinstance(slice_or_idx1[i], int):
      ret.append(slice_or_idx1[i])
      i += 1
    elif isinstance(slice_or_idx2[j], int):
      ret.append(slice_or_idx1[i].start + slice_or_idx2[j] * slice_or_idx1[i].step)
      i += 1
      j += 1
    else:
      ret.append(slice(
          slice_or_idx1[i].start + slice_or_idx2[j].start * slice_or_idx1[i].step,
          slice_or_idx1[i].start + slice_or_idx2[j].stop * slice_or_idx1[i].step,
          slice_or_idx1[i].step * slice_or_idx2[j].step
      ))
      i += 1
      j += 1

def _to_range(transforms) -> tuple[slice | int, ...]:
  ret = ()
  for transform in transforms:
    # For now, assume only NDIndexer transforms.
    ret = _compose_slice_or_index(
        ret, tuple(_transform_slice_or_index(i) for i in transform.indices))
  return ret

def get(device_id, memory_space, buffer_id, transforms):
  device_id = int(device_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)
  try:
    transforms = jax.tree.map(int, transforms)
  except:
    raise ValueError('Advanced indexers are not supported on TPU')

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    read_range = _to_range(transforms)
    buffer = shared_memory.mem[(memory_space, buffer_id, device_id)]
    ret = buffer[read_range].copy()
    if transforms:
      # TODO(jburnim): Instead of using NDIndexer, do the computation ourselves
      # with buffer.shape and read_range?
      expected_shape = transforms[-1].get_indexer_shape()
      if expected_shape != ret.shape[:len(expected_shape)]:
        raise ValueError(
            f'Out-of-bounds read of ({device_id} {memory_space} {buffer_id}): '
            f'reading [{read_range}] but bufer has shape {buffer.shape} .')
    return ret

def store(device_id, memory_space, buffer_id, transforms, val):
  device_id = int(device_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)
  try:
    transforms = jax.tree.map(int, transforms)
  except:
    raise ValueError('Advanced indexers are not supported on TPU')
  val = np.array(val)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    buff = shared_memory.mem[(memory_space, buffer_id, device_id)]
    write_range = _to_range(transforms)
    # TODO(jburnim): Better error message if this raises?
    in_bounds_shape = buff[write_range].shape
    if in_bounds_shape != val.shape:
      raise ValueError(
          f'Out-of-bounds write of ({device_id} {memory_space} {buffer_id}): '
          f'writing [{write_range}] but buffer has shape {buff.shape} .')
    buff[write_range] = val

def swap(device_id, memory_space, buffer_id, transforms, val, mask):
  device_id = int(device_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)
  try:
    transforms = jax.tree.map(int, transforms)
  except:
    raise ValueError('Advanced indexers are not supported on TPU')
  val = np.array(val)
  mask = np.array(mask) if mask is not None else None
  if mask is not None:
    assert mask.shape == val.shape

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    buff = shared_memory.mem[(memory_space, buffer_id, device_id)]
    read_write_range = _to_range(transforms)
    # TODO(jburnim): Better error message if this raises?
    raw_result = buff[read_write_range]
    in_bounds_shape = raw_result.shape
    if mask is None:
      if in_bounds_shape != val.shape:
        raise ValueError(
            f'Out-of-bounds swap of ({device_id} {memory_space} {buffer_id}): '
            f'swapping [{read_write_range}] but buffer has shape {buff.shape} .')
      buff[read_write_range] = val
      return raw_result.copy()

    in_bounds_mask = np.full(mask.shape, True)
    for i in range(len(in_bounds_shape)):
      in_bounds_mask[in_bounds_shape[i]:] = False
    if (~in_bounds_mask & mask).any():
      # TODO(jburnim): Include indices of out-of-bounds locations where mask
      # is True.
      raise ValueError(
          f'Out-of-bounds masked swap of ({device_id} {memory_space} {buffer_id}): '
          f'swapping [{read_write_range}] but buffer has shape {buff.shape} . ')

    in_bounds_idx = tuple(slice(i) for i in in_bounds_shape)
    result = val.copy()
    result[in_bounds_idx] = np.where(
        mask[in_bounds_idx], raw_result, val[in_bounds_idx])
    buff[read_write_range] = np.where(
        mask[in_bounds_idx], val[in_bounds_idx], raw_result)
    return result

def execute_dma(src, dst, send_sem, recv_sem):
  # NOTE: `src` is a list of arguments for `get` (device_id, memory_space,
  # buffer_id, transforms) and `dst` is a list of arguments for `store`
  # (dst_device_id, dst_memory_space, dst_id, dst_transforms).
  #
  # TODO(jburnim): Clean this up.

  # Do the read.
  data = get(*src)
  data_size = data.itemsize * data.size

  # Signal the send semaphore.
  if send_sem is not None:
    send_sem.signal(data_size, device_id=src[0])

  # Do the write.
  store(*dst, data)

  # Signal the receive semaphore.
  recv_sem.signal(data_size, device_id=dst[0])

def print_memory(device_id):
  device_id = int(device_id)
  if all(d == 0 for d in device_id):
    shared_memory = _get_shared_memory()
    with shared_memory.lock:
      print(shared_memory.mem)

def dma_start(device_id, src_memory_space, src_id, src_transforms,
              dst_memory_space, dst_id, dst_transforms,
              dst_sem,
              src_sem,
              dst_device_id):
  device_id = int(device_id)
  src_memory_space, src_id = int(src_memory_space), int(src_id)
  src_transforms = jax.tree.map(int, src_transforms)
  dst_memory_space, dst_id = int(dst_memory_space), int(dst_id)
  dst_transforms = jax.tree.map(int, dst_transforms)
  dst_sem = int(dst_sem)
  if src_sem is not None:
    src_sem = int(src_sem)
  if dst_device_id is not None:
    dst_device_id = int(dst_device_id)
  else:
    dst_device_id = device_id

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    dst_sem = shared_memory.sem[dst_sem]
    if src_sem is not None:
      src_sem = shared_memory.sem[src_sem]

  # For now, just execute the DMA immediately.
  # TODO(jburnim): Execute DMAs asynchronously.
  execute_dma(
      (device_id, src_memory_space, src_id, src_transforms),
      (dst_device_id, dst_memory_space, dst_id, dst_transforms),
      src_sem,
      dst_sem)

def dma_wait(device_id, sem, size):
  device_id = int(device_id)
  sem = int(sem)
  size = int(size)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    sem = shared_memory.sem[sem]
  sem.wait(size, device_id)

def semaphore_signal(device_id, sem, inc, target_device_id, target_core_index):
  device_id = int(device_id)
  sem = int(sem)
  inc = int(inc)
  if target_device_id is None:
    target_device_id = device_id
  else:
    target_device_id = int(target_device_id)

  if target_core_index is not None:
    raise NotImplementedError('semaphore_signal with target_core_index')

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    sem = shared_memory.sem[sem]
  sem.signal(inc, target_device_id)

def semaphore_wait(device_id, sem, value):
  device_id = int(device_id)
  sem = int(sem)
  value = int(value)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    sem = shared_memory.sem[sem]
  sem.wait(value, device_id)

def _compute_transformed_shape_and_dtype(shape, dtype, transforms):
  for transform in transforms:
    if transform is None:
      continue
    shape = transform.transform_shape(shape)
    dtype = transform.transform_dtype(dtype)
  return shape, dtype

def _device_coords_to_logical_id(device_coords, axis_sizes):
  if not isinstance(device_coords, tuple):
    device_coords = (device_coords,)
  assert len(device_coords) == len(axis_sizes)
  sizes = list(axis_sizes.values())
  ret = 0
  for i in range(len(device_coords)):
    ret += device_coords[i] * math.prod(sizes[i+1:])
  return ret

def _device_id_to_logical(device_id, device_id_type, axis_sizes):
  if device_id is None:
    return None
  if device_id_type == mosaic_primitives.DeviceIdType.MESH:
    return _device_coords_to_logical_id(device_id, axis_sizes)
  elif device_id_type == mosaic_primitives.DeviceIdType.LOGICAL:
    return device_id
  else:
    raise ValueError(f'Unsupported device ID type: {device_id_type}')

@lu.cache
def _to_jaxpr(flat_fun, in_avals):
  new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
  new_jaxpr = jax_core.ClosedJaxpr(new_jaxpr, consts)
  return new_jaxpr

def _is_any(memory_space):
  return ((memory_space == mosaic_core.TPUMemorySpace.ANY) or
          (memory_space == pallas_core.MemorySpace.ANY))

def _interpret_jaxpr(jaxpr, *args, compiler_params):
  env = {}

  def read(var):
    if isinstance(var, jax_core.Literal):
      return var.val
    else:
      return env[var]

  def write(var, value):
    env[var] = value

  jax.util.safe_map(write, jaxpr.constvars + jaxpr.invars, args)

  # Get the device ID.
  axis_sizes = jax_core.get_axis_env().axis_sizes
  device_id = _device_coords_to_logical_id(
      tuple(lax.axis_index(s) for s in axis_sizes.keys()),
      axis_sizes)
  # TODO(jburnim): Pass the device ID around, instead of re-fetching/computing
  # it for each sub-jaxpr.

  # TODO(jburnim): Clean up and finish this evaluation loop.  For example:
  #  - Handle missing Pallas primitives, like masked_load.
  #  - Replace the big if-statement with a dictionary of rules.
  #  - Handle other higher-order primitives?
  #  - Megacore.
  for eqn in jaxpr.eqns:
    prim = eqn.primitive
    invals = jax.util.safe_map(read, eqn.invars)

    if prim is primitives.load_p:
      (ref, transforms, mask, _) = jax.tree.unflatten(
          eqn.params['args_tree'], invals)
      if mask is not None:
        raise NotImplementedError('masked load_p')
      out = callback.io_callback(
          get,
          eqn.outvars[0].aval,
          device_id,
          TPU_MEMORY_SPACE_IDXS[eqn.invars[0].aval.memory_space],
          ref,
          transforms,
          ordered=True)

    elif prim is primitives.swap_p:
      (ref, transforms, val, mask) = jax.tree.unflatten(
          eqn.params['args_tree'], invals)
      out = callback.io_callback(
          swap,
          eqn.outvars[0].aval,
          device_id,
          TPU_MEMORY_SPACE_IDXS[eqn.invars[0].aval.memory_space],
          ref,
          transforms,
          val,
          mask,
          ordered=True)

    elif prim is lax.cond_p:
      def _make_branch(jaxpr):
        return lambda *args: _interpret_jaxpr(
            jaxpr, *args, compiler_params=compiler_params)
      out = lax.switch(
          invals[0],
          [_make_branch(branch_jaxpr.jaxpr)
           for branch_jaxpr in eqn.params['branches']],
          *invals[1:])

    elif prim is lax.scan_p:
      consts, init_carry, xs = split_list(
          invals, [eqn.params['num_consts'], eqn.params['num_carry']])
      def _scan_body(c, a):
        return split_list(
            _interpret_jaxpr(eqn.params['jaxpr'].jaxpr, *consts, *c, *a,
                             compiler_params=compiler_params),
            [eqn.params['num_carry']])
      carry, out = lax.scan(_scan_body, init_carry, xs=xs,
                            length=eqn.params.get('length', None))
      out = carry + out

    elif prim is lax.while_p:
      cond_consts, body_consts, init_vals  = split_list(
          invals, [eqn.params['cond_nconsts'], eqn.params['body_nconsts']])
      out = lax.while_loop(
          lambda args: _interpret_jaxpr(eqn.params['cond_jaxpr'].jaxpr,
                                         *cond_consts, *args,
                                         compiler_params=compiler_params)[0],
          lambda args: _interpret_jaxpr(eqn.params['body_jaxpr'].jaxpr,
                                         *body_consts, *args,
                                         compiler_params=compiler_params),
          init_vals)

    elif prim is for_loop.for_p:
      raise NotImplementedError('for_p')

    elif prim is pjit.pjit_p:
      def f(*args, jaxpr):
        return _interpret_jaxpr(jaxpr.jaxpr, *jaxpr.consts, *args,
                                compiler_params=compiler_params)
      in_avals = tuple(jax_core.shaped_abstractify(i) for i in invals)
      new_jaxpr = _to_jaxpr(
          lu.wrap_init(functools.partial(f, jaxpr=eqn.params['jaxpr']),
                       debug_info=eqn.params['jaxpr'].jaxpr.debug_info),
          in_avals)
      out = pjit.pjit_p.bind(*invals, **(eqn.params | {'jaxpr': new_jaxpr}))

    elif prim is primitives.run_scoped_p:
      # Allocate a buffer or semaphore for each element of
      # eqn.params['jaxpr'].invars .
      allocs = []
      for v in eqn.params['jaxpr'].invars:
        if v.aval.memory_space == mosaic_core.TPUMemorySpace.SEMAPHORE:
          allocs.append(callback.io_callback(
              _allocate_semaphores,
              jax.ShapeDtypeStruct(v.aval.shape, jnp.int16),
              device_id,
              v.aval.shape,
              ordered=True))
        else:
          allocs.append(callback.io_callback(
              _allocate_buffer,
              jax.ShapeDtypeStruct((), jnp.int16),
              device_id,
              TPU_MEMORY_SPACE_IDXS[v.aval.memory_space],
              primitives.uninitialized_value(v.aval.shape, v.aval.dtype),
              ordered=True))

      out = _interpret_jaxpr(eqn.params['jaxpr'], *invals, *allocs,
                             compiler_params=compiler_params)

      for a in allocs:
        if isinstance(a, tuple):
          callback.io_callback(
              _deallocate_buffer,
              None,
              device_id,
              TPU_MEMORY_SPACE_IDXS[v.aval.memory_space],
              a,
              ordered=True)
        else:
          # TODO(jburnim): Delete semaphores.
          # callback.io_callback(
          #     _deallocate_semaphores,
          #     None,
          #     device_id,
          #     a,
          #     ordered=True)
          pass

    elif prim is state_primitives.get_p:
      out = callback.io_callback(
          get,
          eqn.outvars[0].aval,
          device_id,
          TPU_MEMORY_SPACE_IDXS[eqn.invars[0].aval.memory_space],
          invals[0],
          jax.tree.unflatten(eqn.params['tree'], invals[1:]),
          ordered=True)

    elif prim is state_primitives.swap_p:
      out = callback.io_callback(
          swap,
          eqn.outvars[0].aval,
          device_id,
          TPU_MEMORY_SPACE_IDXS[eqn.invars[0].aval.memory_space],
          invals[0],
          jax.tree.unflatten(eqn.params['tree'], invals[2:]),
          invals[1],
          None,
          ordered=True)

    elif prim is mosaic_primitives.dma_start_p:
      (src, src_transforms,
       dst, dst_transforms,
       dst_sem, dst_sem_transforms,
       src_sem, src_sem_transforms,
       target_device_id) = jax.tree.unflatten(eqn.params['tree'], invals)
      target_device_id = _device_id_to_logical(
          target_device_id, eqn.params['device_id_type'], axis_sizes)
      (orig_src_ref, _, orig_dst_ref, *_
       ) = jax.tree.unflatten(eqn.params['tree'], eqn.invars)
      callback.io_callback(
          dma_start,
          (),
          device_id,
          TPU_MEMORY_SPACE_IDXS[orig_src_ref.aval.memory_space],
          src, src_transforms,
          TPU_MEMORY_SPACE_IDXS[orig_dst_ref.aval.memory_space],
          dst, dst_transforms,
          state_discharge.transform_array(dst_sem, dst_sem_transforms),
          state_discharge.transform_array(src_sem, src_sem_transforms),
          target_device_id,
          ordered=True)
      out = []

    elif prim is mosaic_primitives.dma_wait_p:
      (src, src_transforms,
       dst, dst_transforms,
       dst_sem, dst_sem_transforms,
       src_sem, src_sem_transforms,
       target_device_id) = jax.tree.unflatten(eqn.params['tree'], invals)
      read_shape, read_dtype = _compute_transformed_shape_and_dtype(
          eqn.invars[0].aval.shape, eqn.invars[0].aval.dtype, src_transforms)
      callback.io_callback(
          dma_wait,
          (),
          device_id,
          state_discharge.transform_array(dst_sem, dst_sem_transforms),
          math.prod(read_shape) * read_dtype.itemsize,
          ordered=True)
      out = []

    elif prim is mosaic_primitives.get_barrier_semaphore_p:
      out = callback.io_callback(
          get_barrier_semaphore,
          jax.ShapeDtypeStruct((), jnp.int16),
          device_id,
          compiler_params['mosaic']['collective_id'],
          ordered=True)

    elif prim is mosaic_primitives.semaphore_signal_p:
      sem, sem_transforms, inc, target_device_id, core_index = (
          jax.tree.unflatten(eqn.params['args_tree'], invals))
      target_device_id = _device_id_to_logical(
          target_device_id, eqn.params['device_id_type'], axis_sizes)
      callback.io_callback(
          semaphore_signal,
          (),
          device_id,
          state_discharge.transform_array(sem, sem_transforms),
          inc,
          target_device_id,
          core_index,
          ordered=True)
      out = []

    elif prim is mosaic_primitives.semaphore_wait_p:
      sem, sem_transforms, value = (
          jax.tree.unflatten(eqn.params['args_tree'], invals))
      callback.io_callback(
          semaphore_wait,
          (),
          device_id,
          state_discharge.transform_array(sem, sem_transforms),
          value,
          ordered=True)
      out = []

    elif prim is primitives.atomic_rmw_p:
      raise NotImplementedError('atomic_rmw_p')

    elif prim is primitives.atomic_cas_p:
      raise NotImplementedError('atomic_cas_p')

    else:
      subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
      out = prim.bind(*subfuns, *invals, **bind_params)

    out = out if prim.multiple_results else [out]
    jax.util.safe_map(write, eqn.outvars, out)

  return jax.util.safe_map(read, jaxpr.outvars)

def _initialize_output_vals(
    block_mappings_output: Iterable[BlockMapping],
    input_args, input_output_aliases) -> Sequence[jax.Array]:
  oi_map = {v: k for k, v in input_output_aliases}
  output_vals = []
  for i, bm in enumerate(block_mappings_output):
    if i in oi_map:
      output_vals.append(input_args[oi_map[i]])
    else:
      output_vals.append(primitives.uninitialized_value(
          bm.array_shape_dtype.shape,
          bm.array_shape_dtype.dtype))
  return output_vals

def _compute_start_indices(block_mapping, loop_idx, *args):
    block_indices = (
        jax_core.jaxpr_as_fun(block_mapping.index_map_jaxpr)(*loop_idx, *args))
    if isinstance(block_mapping.indexing_mode, pallas_core.Blocked):
      ret = tuple(i if b is pallas_core.mapped else b * i
                  for b, i in zip(block_mapping.block_shape, block_indices))
    elif isinstance(block_mapping.indexing_mode, pallas_core.Unblocked):
      ret = block_indices
    else:
      raise RuntimeError(f"Unknown indexing mode: {block_mapping.indexing_mode}")
    return ret

def _get_next_indices(grid, indices):
  next_indices = []
  carry = True
  for dim_size, index in reversed(list(zip(grid, indices))):
    i = jnp.where(carry, index + 1, index)
    carry = dim_size == i
    next_indices.append(jnp.where(carry, 0, i))
  return tuple(reversed(next_indices))

def _maybe_dynamic_slice(start_idx, block_shape, value, is_indexing):
  start_idx = tuple(jnp.array(s, dtype=jnp.int32) for s in start_idx)
  output = lax.dynamic_slice(value, start_idx, slice_sizes=block_shape)
  squeeze_dims = tuple(np.arange(len(is_indexing))[np.array(is_indexing,
                                                            dtype=np.bool_)])
  return lax.squeeze(output, squeeze_dims)

def _pad_to_block_dimension(value, block_shape):
  """Pads values so the shape evenly divides into block dimensions.

  For example, if values has a shape of (33, 2, 5) with a block_shape of
  (32, 2, 4), this function will pad the value of shape to (64, 2, 8).

  Args:
    value: Array to be padded.
    block_shape: Block shapes to use for padding. If None, no padding will
      be performed.

  Returns:
    A padded array.
  """
  padded_shape = tuple(
      ((v - 1) // b + 1) * b for v, b in zip(value.shape, block_shape)
  )
  if padded_shape != value.shape:
    pad_width = tuple((0, a-b) for a, b in zip(padded_shape, value.shape))
    pad_value = primitives.uninitialized_value(shape=(), dtype=value.dtype)
    value = jnp.pad(value, pad_width, constant_values=pad_value)
  return value

def get_interpret_effects():
  return {callback._OrderedIOEffect}

def interpret_pallas_call(
    *args,
    jaxpr: jax_core.Jaxpr,
    name_and_src_info: pallas_core.NameAndSrcInfo,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: GridMapping,
    compiler_params: Any,
    cost_estimate: CostEstimate,
    out_avals: tuple[jax_core.AbstractValue, ...],
):
  del debug, cost_estimate, out_avals

  # args contains: *dynamic_grid_sizes, *index, *inputs.  (No consts?)
  dynamic_grid_args, scalars, input_args = split_list(
      args,
      [grid_mapping.num_dynamic_grid_bounds, grid_mapping.num_index_operands],
  )
  dynamic_grid_args_iter = iter(dynamic_grid_args)
  grid = tuple(
      a if a is not pallas_core.dynamic_grid_dim
      else next(dynamic_grid_args_iter)
      for a in grid_mapping.grid
  )
  assert next(dynamic_grid_args_iter, None) is None

  axis_sizes = jax_core.get_axis_env().axis_sizes
  device_id = _device_coords_to_logical_id(
      tuple(lax.axis_index(s) for s in axis_sizes.keys()),
      axis_sizes)

  # Pad input arguments.
  is_indexing_dim = [
      tuple(b is pallas_core.mapped for b in bm.block_shape)
      for bm in grid_mapping.block_mappings
  ]
  block_shapes = [
      tuple(1 if i else b for i, b in zip(iid, bm.block_shape))
      for iid, bm in zip(is_indexing_dim, grid_mapping.block_mappings)
  ]
  num_inputs = grid_mapping.num_inputs
  input_args = [
      _pad_to_block_dimension(a, bs)
      for a, bs in zip(input_args, block_shapes[:num_inputs])
  ]

  # Allocate buffers in HBM for outputs.
  output_buffer_ids = []
  output_buffer_shapes = []
  output_vals = _initialize_output_vals(
      grid_mapping.block_mappings_output, args, input_output_aliases)
  num_outputs = grid_mapping.num_outputs
  output_block_shapes = block_shapes[num_inputs : num_inputs + num_outputs]
  for out_val, bs in zip(output_vals, output_block_shapes):
    padded_val = _pad_to_block_dimension(out_val, bs)
    output_buffer_shapes.append(padded_val.shape)
    output_buffer_ids.append(callback.io_callback(
        _allocate_buffer,
        jax.ShapeDtypeStruct((), jnp.int16),
        device_id,
        TPU_MEMORY_SPACE_IDXS[mosaic_core.TPUMemorySpace.ANY],
        padded_val,
        ordered=True))

  # Allocate buffers for all kernel arguments (e.g., scalars, inputs,
  # outputs, scratch).
  io_alias_map = dict(input_output_aliases)
  oi_alias_map = {v: k for k, v in input_output_aliases}
  kernel_buffer_ids = []
  for var, val in zip(jaxpr.invars[grid_mapping.slice_index_ops], scalars):
    kernel_buffer_ids.append(callback.io_callback(
        _allocate_buffer,
        jax.ShapeDtypeStruct((), jnp.int16),
        device_id,
        TPU_MEMORY_SPACE_IDXS[mosaic_core.TPUMemorySpace.SMEM],
        val,
        ordered=True))
  for i, var in enumerate(jaxpr.invars[grid_mapping.num_index_operands:]):
    output_idx = i - grid_mapping.num_inputs
    is_input = i < grid_mapping.num_inputs
    is_output = (output_idx >= 0) and (output_idx < grid_mapping.num_outputs)
    if var.aval.memory_space == mosaic_core.TPUMemorySpace.SEMAPHORE:
      kernel_buffer_ids.append(callback.io_callback(
          _allocate_semaphores,
          jax.ShapeDtypeStruct(var.aval.shape, jnp.int16),
          device_id,
          var.aval.shape,
          ordered=True))
    elif is_output and _is_any(var.aval.memory_space):
      # Use the already-allocated HBM output buffer.
      #
      # TODO(jburnim): For kernel args in HBM, check that block shape is the
      # same as for the corresponding pallas_call input, and that the index_map
      # is trivial.
      kernel_buffer_ids.append(output_buffer_ids[output_idx])
    elif is_output and (output_idx in oi_alias_map):
      # Use the already-allocated (non-HBM) input buffer.
      kernel_buffer_ids.append(kernel_buffer_ids[oi_alias_map[output_idx]])
    elif is_input and (i in io_alias_map) and _is_any(var.aval.memory_space):
      # Use the already-allocated HBM output buffer.
      kernel_buffer_ids.append(output_buffer_ids[io_alias_map[i]])
    else:
      # TODO(jburnim): For kernel args in HBM, check that block shape is the
      # same as for the corresponding pallas_call input, and that the index_map
      # is trivial.
      kernel_buffer_ids.append(callback.io_callback(
          _allocate_buffer,
          jax.ShapeDtypeStruct((), jnp.int16),
          device_id,
          TPU_MEMORY_SPACE_IDXS[var.aval.memory_space],
          primitives.uninitialized_value(var.aval.shape, var.aval.dtype),
          ordered=True))

  _, input_ids, kernel_output_ids, _  = split_list(
      kernel_buffer_ids,
      [grid_mapping.num_index_operands, num_inputs, grid_mapping.num_outputs])
  input_vars, output_vars = split_list(
      jaxpr.invars[grid_mapping.slice_block_ops], [num_inputs])

  # For kernel inputs that are in HBM, we populate the buffer once before
  # any kernel invocations.
  for buffer_id, var, val in zip(input_ids, input_vars, input_args):
    if not _is_any(var.aval.memory_space):
      continue
    if val.shape != var.aval.shape:
      # TODO(jburnim): Also check that the index_map is trivial.
      raise ValueError()
    callback.io_callback(
        store,
        (),
        device_id,
        TPU_MEMORY_SPACE_IDXS[mosaic_core.TPUMemorySpace.ANY],
        buffer_id,
        (),
        val,
        ordered=True)

  scalar_ids, in_out_ids, scratch_ids = split_list(
      kernel_buffer_ids,
      [grid_mapping.num_index_operands, len(grid_mapping.block_mappings)])

  if grid:
    num_iterations = functools.reduce(jnp.multiply, grid)  # type: ignore[arg-type]
  else:
    # Base case is always one iteration when grid is ()
    num_iterations = 1

  def body(carry):
    # The loop carry: (i, loop_idx) --
    #  - i:int32 is the interation index
    #  - loop_idx: tuple[int32] are the program ids for each grid axis
    i, loop_idx = carry

    if grid_mapping.local_grid_env is not None:
      local_grid_env = grid_mapping.local_grid_env(loop_idx, grid)
    else:
      local_grid_env = tuple(
          pallas_core.GridAxis(idx, b)
          for dim, (idx, b) in enumerate(zip(loop_idx, grid))
          if dim not in grid_mapping.vmapped_dims
      )

    with pallas_core.grid_env(local_grid_env):
      # Copy slices of the input to the kernel buffers.
      #
      # TODO(jburnim): Only copy slices when the index mapping has changed?
      start_indices = [_compute_start_indices(bm, loop_idx, *scalars)
                       for bm in grid_mapping.block_mappings]
      for j, var in enumerate(input_vars):
        if _is_any(var.aval.memory_space):
          continue
        sliced_val = _maybe_dynamic_slice(start_indices[j], block_shapes[j],
                                          input_args[j], is_indexing_dim[j])
        assert(sliced_val.shape == var.aval.shape)
        callback.io_callback(
            store,
            (),
            device_id,
            TPU_MEMORY_SPACE_IDXS[var.aval.memory_space],
            input_ids[j],
            (),
            sliced_val,
            ordered=True)

      # Invoke the kernel.
      _interpret_jaxpr(jaxpr, *kernel_buffer_ids,
                       compiler_params=compiler_params)

      # Copy from the kernel buffers to slices of the output in HBM.
      #
      # TODO(jburnim): Only copy if the index mapping will change in the
      # next iteration (or if this is the last iteration)?
      for j, var in enumerate(output_vars):
        if _is_any(var.aval.memory_space):
          continue
        kernel_output_val = callback.io_callback(
            get,
            var.aval,
            device_id,
            TPU_MEMORY_SPACE_IDXS[var.aval.memory_space],
            kernel_output_ids[j],
            (),
            ordered=True)
        transform = indexing.NDIndexer(
            indices=tuple(indexing.ds(st, sz) if not iid else st
                          for st, sz, iid  in zip(start_indices[num_inputs + j],
                                                  block_shapes[num_inputs + j],
                                                  is_indexing_dim[num_inputs + j])),
            shape=output_vals[j].shape,
            int_indexer_shape=())
        callback.io_callback(
            store,
            (),
            device_id,
            TPU_MEMORY_SPACE_IDXS[mosaic_core.TPUMemorySpace.ANY],
            output_buffer_ids[j],
            (transform,),
            kernel_output_val,
            ordered=True)

      return i + 1, _get_next_indices(grid, loop_idx)

  # TODO(jburnim): Handle parallel grid dimensions + megacore.
  _ = lax.while_loop(
      lambda carry: carry[0] < num_iterations,
      body,
      (jnp.int32(0), (jnp.int32(0),) * len(grid))
  )

  # Read the output from the allocated output buffers.
  ret = [
      callback.io_callback(
          get,
          val,
          device_id,
          TPU_MEMORY_SPACE_IDXS[mosaic_core.TPUMemorySpace.ANY],
          output_buffer_id,
          (indexing.NDIndexer.from_indices_shape(
              tuple(indexing.ds(0, s) for s in val.shape),
              output_buffer_shape),),
          ordered=True)
      for val, output_buffer_id, output_buffer_shape in zip(
          output_vals, output_buffer_ids, output_buffer_shapes)
  ]

  for buffer_id in output_buffer_ids:
    callback.io_callback(
        _deallocate_buffer,
        (),
        device_id,
        TPU_MEMORY_SPACE_IDXS[mosaic_core.TPUMemorySpace.ANY],
        buffer_id,
        ordered=True)
  for buffer_id, var in zip(kernel_buffer_ids, jaxpr.invars):
    if var.aval.memory_space == mosaic_core.TPUMemorySpace.SEMAPHORE:
      pass
    else:
      callback.io_callback(
          _deallocate_buffer,
          (),
          device_id,
          TPU_MEMORY_SPACE_IDXS[var.aval.memory_space],
          buffer_id,
          ordered=True)

  return ret
