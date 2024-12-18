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
from functools import reduce
import itertools
import math
import threading
from typing import Any

import jax
from jax import lax
from jax._src import callback
from jax._src import core as jax_core
from jax._src.pallas.mosaic import primitives as mosaic_primitives
from jax._src.pallas.mosaic import core as mosaic_core
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax._src.util import (
    safe_map,
    safe_zip,
    split_list
)
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
class SharedMemory:
  # (memory_space, buffer_id, device_id) -> NumPy array
  mem: dict = dataclasses.field(default_factory=dict)

  # semaphore_id -> Semaphore
  sem: dict = dataclasses.field(default_factory=dict)

  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

  next_buffer_id: dict = dataclasses.field(
    default_factory=lambda: collections.defaultdict(lambda: 100))
  next_semaphore_id: dict = dataclasses.field(
    default_factory=lambda: collections.defaultdict(lambda: 2000))

shared_memory = SharedMemory()

class Semaphore:
  def __init__(self, num_devices=1):
    self.cv = threading.Condition()

    # TODO(jburnim): Index counts by a single integer device ID, instead of
    # mesh coordinates.
    self.counts = collections.defaultdict(int)

  def signal(self, inc, device_id):
    device_id = tuple(int(x) for x in device_id)
    with self.cv:
      self.counts[device_id] += inc
      self.cv.notify_all()

  def wait(self, value, device_id):
    device_id = tuple(int(x) for x in device_id)
    with self.cv:
      while self.counts[device_id] < value:
        self.cv.wait()
      self.counts[device_id] -= value

def _allocate_buffer(device_id, memory_space, val):
  device_id = tuple(map(int, device_id))
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  val = np.array(val)

  with shared_memory.lock:
    buffer_id = shared_memory.next_buffer_id[device_id]
    shared_memory.next_buffer_id[device_id] = buffer_id + 1
    shared_memory.mem[(memory_space, buffer_id, device_id)] = val

  return np.int16(buffer_id)

def _allocate_semaphore(device_id, shape):
  device_id = tuple(map(int, device_id))
  shape = tuple(map(int, shape))
  num_semaphores = math.prod(shape)

  with shared_memory.lock:
    semaphore_id = shared_memory.next_semaphore_id[device_id]
    shared_memory.next_semaphore_id[device_id] = semaphore_id + num_semaphores
    for i in range(semaphore_id, semaphore_id + num_semaphores):
      if not i in shared_memory.sem:
        shared_memory.sem[i] = Semaphore()

  return np.int16(
      range(semaphore_id, semaphore_id + num_semaphores)
  ).reshape(shape)


TPU_MEMORY_SPACE_IDXS = {
    v: i for i, v in enumerate(mosaic_core.TPUMemorySpace)}
TPU_MEMORY_SPACE_NAMES = {
    i: v.value for i, v in enumerate(mosaic_core.TPUMemorySpace)}

def get_barrier_semaphore(device_id, collective_id):
  device_id = tuple(map(int, device_id))
  collective_id = int(collective_id)

  # TODO(jburnim): Check/fix so that IDs for barrier semaphores do not conflict
  # with IDs for regular or DMA semaphores.  (For example, store them in a
  # different table.)
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
        slice_or_idx.start, slice_or_idx.size, slice_or_idx.stride)
    return slice(start, start + size * stride, stride)

def transform_array(x, transforms):
  for transform in transforms:
    # For now, assume only NDIndexer transforms.
    x = x[tuple(_transform_slice_or_index(i) for i in transform.indices)]
  return x

def get(device_id, memory_space, buffer_id, transforms):
  # NOTE: buffer_id is a pair of an int ID and a tuple of transforms
  # TODO(jburnim): Clean this up.
  buffer_id, transforms = buffer_id[0], buffer_id[1] + transforms

  device_id = tuple(int(x) for x in device_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)
  transforms = jax.tree.map(int, transforms)
  with shared_memory.lock:
    return transform_array(
        shared_memory.mem[(memory_space, buffer_id, device_id)], transforms
    ).copy()

def store(device_id, memory_space, buffer_id, transforms, val):
  # NOTE: buffer_id is a pair of an int ID and a tuple of transforms.
  buffer_id, transforms = buffer_id[0], buffer_id[1] + transforms

  device_id = tuple(int(x) for x in device_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)
  val = np.asarray(val)
  with shared_memory.lock:
    if transforms:
      transform_array(
          shared_memory.mem[(memory_space, buffer_id, device_id)],
          transforms
      )[:] = val
    else:
      shared_memory.mem[(memory_space, buffer_id, device_id)] = val

def swap(device_id, memory_space, buffer_id, transforms, val):
  # NOTE: buffer_id is a pair of an int ID and a tuple of transforms.
  buffer_id, transforms = buffer_id[0], buffer_id[1] + transforms

  device_id = tuple(int(x) for x in device_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)
  transforms = jax.tree.map(int, transforms)
  val = np.array(val)

  with shared_memory.lock:
    mem_val = shared_memory.mem[(memory_space, buffer_id, device_id)].copy()
  # TODO(jburnim): Do this transform with NumPy arrays?
  result, result_val = state_discharge.transform_swap_array(
      jnp.array(mem_val), transforms, jnp.array(val))
  result_val = np.array(result_val)
  with shared_memory.lock:
    shared_memory.mem[(memory_space, buffer_id, device_id)] = result_val

  return np.array(result)

def execute_dma(src, dst, send_sem, recv_sem):
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
  device_id = tuple(map(int, device_id))
  if all(d == 0 for d in device_id):
    with shared_memory.lock:
      print(shared_memory.mem)

def dma_start(device_id, src_memory_space, src_id, src_transforms,
              dst_memory_space, dst_id, dst_transforms,
              dst_sem,
              src_sem,
              dst_device_id):
  # NOTE: src_id and dst_id are pairs of int IDs and tuples of transforms.
  src_id, src_transforms = src_id[0], src_id[1] + src_transforms
  dst_id, dst_transforms = dst_id[0], dst_id[1] + dst_transforms

  device_id = tuple(int(x) for x in device_id)
  src_memory_space, src_id = int(src_memory_space), int(src_id)
  src_transforms = jax.tree.map(int, src_transforms)
  dst_memory_space, dst_id = int(dst_memory_space), int(dst_id)
  dst_transforms = jax.tree.map(int, dst_transforms)
  dst_sem = int(dst_sem)
  if src_sem is not None:
    src_sem = int(src_sem)
  if dst_device_id is not None:
    dst_device_id = tuple(int(x) for x in dst_device_id)
  else:
    dst_device_id = device_id

  with shared_memory.lock:
    dst_sem = shared_memory.sem[dst_sem]
    if src_sem is not None:
      src_sem = shared_memory.sem[src_sem]

  # For now, just execute the DMA immediately.
  # TODO(jburnim): Execute DMAs asynchronously.
  execute_dma(
      (device_id, src_memory_space, (src_id, ()), src_transforms),
      (dst_device_id, dst_memory_space, (dst_id, ()), dst_transforms),
      src_sem,
      dst_sem)

def dma_wait(device_id, sem, size):
  device_id = tuple(int(x) for x in device_id)
  sem = int(sem)
  size = int(size)

  with shared_memory.lock:
    sem = shared_memory.sem[sem]
  sem.wait(size, device_id)

def semaphore_signal(device_id, sem, inc, target_device_id, target_core_index):
  device_id = tuple(map(int, device_id))
  sem = int(sem)
  inc = int(inc)
  target_device_id = tuple(map(int, target_device_id))

  if target_core_index is not None:
    raise NotImplementedError()

  with shared_memory.lock:
    sem = shared_memory.sem[sem]
  sem.signal(inc, target_device_id)

def semaphore_wait(device_id, sem, value):
  device_id = tuple(map(int, device_id))
  sem = int(sem)
  value = int(value)

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

  # Get the mesh coordinates.
  device_coords = tuple(
    lax.axis_index(s) for s in jax_core.get_axis_env().axis_sizes)
  # TODO(jburnim): Convert to a single integer device ID?

  # TODO(jburnim): Clean up and finish this evaluation loop.  For example:
  #  - Handle missing Pallas primitives, like masked_load.
  #  - Replace the big if-statement with a dictionary of rules.
  #  - Handle other higher-order primitives?
  #  - Megacore.
  for eqn in jaxpr.eqns:
    prim = eqn.primitive
    invals = jax.util.safe_map(read, eqn.invars)

    if prim is primitives.load_p:
      raise NotImplementedError()

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
      out = lax.scan(_scan_body, init_carry, xs)

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

    elif prim is primitives.run_scoped_p:
      # Allocate a buffer or semaphore for each element of
      # eqn.params['jaxpr'].invars .
      allocs = []
      for v in eqn.params['jaxpr'].invars:
        if v.aval.memory_space.value == 'semaphore_mem':
          allocs.append(callback.io_callback(
            _allocate_semaphore,
            jax.ShapeDtypeStruct(v.aval.shape, jnp.int16),
            device_coords,
            v.aval.shape,
            ordered=True))
        else:
          allocs.append((callback.io_callback(
            _allocate_buffer,
            jax.ShapeDtypeStruct((), jnp.int16),
            device_coords,
            TPU_MEMORY_SPACE_IDXS[v.aval.memory_space],
            primitives.uninitialized_value(v.aval.shape, v.aval.dtype),
            ordered=True), ()))

      out = _interpret_jaxpr(eqn.params['jaxpr'], *invals, *allocs,
                             compiler_params=compiler_params)

      # TODO(jburnim): De-allocate.
      # for a in allocs:
      #   if isinstance(a, tuple):
      #     callback.io_callback(
      #       _deallocate_buffer,
      #       None,
      #       device_coords,
      #       TPU_MEMORY_SPACE_IDXS[v.aval.memory_space],
      #       a,
      #       ordered=True)
      #   else:
      #     callback.io_callback(
      #       _deallocate_semaphore,
      #       None,
      #       device_coords,
      #       a,
      #       ordered=True)

    elif prim is state_primitives.get_p:
      out = callback.io_callback(
        get,
        eqn.outvars[0].aval,
        device_coords,
        TPU_MEMORY_SPACE_IDXS[eqn.invars[0].aval.memory_space],
        invals[0],
        jax.tree.unflatten(eqn.params['tree'], invals[1:]),
        ordered=True)

    elif prim is state_primitives.swap_p:
      out = callback.io_callback(
        swap,
        eqn.outvars[0].aval,
        device_coords,
        TPU_MEMORY_SPACE_IDXS[eqn.invars[0].aval.memory_space],
        invals[0],
        jax.tree.unflatten(eqn.params['tree'], invals[2:]),
        invals[1],
        ordered=True)

    elif prim is mosaic_primitives.dma_start_p:
      (src, src_transforms,
       dst, dst_transforms,
       dst_sem, dst_sem_transforms,
       src_sem, src_sem_transforms,
       device_id) = jax.tree.unflatten(eqn.params['tree'], invals)
      (orig_src_ref, _, orig_dst_ref, *_
       ) = jax.tree.unflatten(eqn.params['tree'], eqn.invars)
      callback.io_callback(
        dma_start,
        (),
        device_coords,
        TPU_MEMORY_SPACE_IDXS[orig_src_ref.aval.memory_space],
        src, src_transforms,
        TPU_MEMORY_SPACE_IDXS[orig_dst_ref.aval.memory_space],
        dst, dst_transforms,
        state_discharge.transform_array(dst_sem, dst_sem_transforms),
        state_discharge.transform_array(src_sem, src_sem_transforms),
        device_id,
        ordered=True)
      out = []

    elif prim is mosaic_primitives.dma_wait_p:
      (src, src_transforms,
       dst, dst_transforms,
       dst_sem, dst_sem_transforms,
       src_sem, src_sem_transforms,
       device_id) = jax.tree.unflatten(eqn.params['tree'], invals)
      read_shape, read_dtype = _compute_transformed_shape_and_dtype(
        eqn.invars[0].aval.shape, eqn.invars[0].aval.dtype, src_transforms)
      callback.io_callback(
        dma_wait,
        (),
        device_coords,
        state_discharge.transform_array(dst_sem, dst_sem_transforms),
        math.prod(read_shape) * read_dtype.itemsize,
        ordered=True)
      out = []

    elif prim is mosaic_primitives.get_barrier_semaphore_p:
      out = callback.io_callback(
        get_barrier_semaphore,
        jax.ShapeDtypeStruct((), jnp.int16),
        device_coords,
        compiler_params['mosaic']['collective_id'],
        ordered=True)

    elif prim is mosaic_primitives.semaphore_signal_p:
      sem, sem_transforms, inc, device_id, core_index = (
        jax.tree.unflatten(eqn.params['args_tree'], invals))
      callback.io_callback(
        semaphore_signal,
        (),
        device_coords,
        state_discharge.transform_array(sem, sem_transforms),
        inc,
        device_id,
        core_index,
        ordered=True)
      out = []

    elif prim is mosaic_primitives.semaphore_wait_p:
      sem, sem_transforms, value = (
        jax.tree.unflatten(eqn.params['args_tree'], invals))
      callback.io_callback(
        semaphore_wait,
        (),
        device_coords,
        state_discharge.transform_array(sem, sem_transforms),
        value,
        ordered=True)
      out = []

    else:
      out = prim.bind(*invals, **eqn.params)

    out = out if prim.multiple_results else [out]
    jax.util.safe_map(write, eqn.outvars, out)

  return jax.util.safe_map(read, jaxpr.outvars)

def _initialize_scratch_vals(scratch_avals) -> tuple[jax.Array, ...]:
  scratch_avals = (jax_core.raise_to_shaped(x) for x in scratch_avals)
  return tuple(
      primitives.uninitialized_value(a.shape, a.dtype) for a in scratch_avals
  )

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

  dynamic_grid_args, args = split_list(  # type: ignore
      args, [grid_mapping.num_dynamic_grid_bounds]
  )
  dynamic_grid_args_iter = iter(dynamic_grid_args)
  grid = tuple(
      a if a is not pallas_core.dynamic_grid_dim
      else next(dynamic_grid_args_iter)
      for a in grid_mapping.grid
  )
  assert next(dynamic_grid_args_iter, None) is None

  scalars = args[grid_mapping.slice_index_ops]
  out = _initialize_output_vals(
      grid_mapping.block_mappings_output, args, input_output_aliases)
  block_args = args[len(scalars):]
  # block_args now contains: *consts, *inputs

  # invars: [*scalar_prefetch, *consts, *inputs, *outputs, *scratch]
  scratch_invars = jaxpr.invars[grid_mapping.slice_scratch_ops]
  scratch_avals = [v.aval for v in scratch_invars]
  scratch_values = _initialize_scratch_vals(scratch_avals)

  device_coords = tuple(
    lax.axis_index(s) for s in jax_core.get_axis_env().axis_sizes)

  # Allocate and fill buffers for all block arguments, outputs,
  # and scratch values.
  # TODO(jburnim): Handle aliasing.
  # TODO(jburnim): Handle padding.
  in_out_args = list(block_args) + list(out)
  buffer_ids = []
  for var, value in zip(jaxpr.invars[grid_mapping.num_index_operands:],
                        itertools.chain(in_out_args, scratch_values)):
    if var.aval.memory_space.value == 'semaphore_mem':
      buffer_ids.append(callback.io_callback(
        _allocate_semaphore,
        jax.ShapeDtypeStruct(var.aval.shape, jnp.int16),
        device_coords,
        var.aval.shape,
        ordered=True))
    else:
      buffer_ids.append((callback.io_callback(
        _allocate_buffer,
        jax.ShapeDtypeStruct((), jnp.int16),
        device_coords,
        TPU_MEMORY_SPACE_IDXS[var.aval.memory_space],
        value,
        ordered=True), ()))
  in_out_ids, scratch_ids = split_list(
    buffer_ids, [grid_mapping.num_inputs + grid_mapping.num_outputs])

  is_indexing_dim = [
      tuple(b is pallas_core.mapped for b in bm.block_shape)
      for bm in grid_mapping.block_mappings
  ]
  block_shapes = [
      tuple(1 if i else b for i, b in zip(iid, bm.block_shape))
      for iid, bm in zip(is_indexing_dim, grid_mapping.block_mappings)
  ]

  if grid:
    num_iterations = reduce(jnp.multiply, grid)  # type: ignore[arg-type]
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
      start_indices = [_compute_start_indices(bm, loop_idx, *scalars)
                       for bm in grid_mapping.block_mappings]
      transformed_in_out_ids = []
      for start_idx, block_shape, arg_id, arg, is_indexing in zip(
          start_indices, block_shapes, in_out_ids, in_out_args, is_indexing_dim):
        assert all(is_indexing_dim)  # TODO(jburnim): Handle non-indexing dims.
        transformed_in_out_ids.append((
          arg_id[0],
          (indexing.NDIndexer(
            indices=tuple(indexing.ds(i, s) for i, s in zip(start_idx, block_shape)),
            shape=arg.aval.shape,
            int_indexer_shape=()),)
        ))
      _interpret_jaxpr(jaxpr, *scalars, *transformed_in_out_ids, *scratch_ids,
                       compiler_params=compiler_params)

      return i + 1, _get_next_indices(grid, loop_idx)

  # TODO(jburnim): Handle parallel grid dimensions + megacore.
  _ = lax.while_loop(
    lambda carry: carry[0] < num_iterations,
    body,
    (jnp.int32(0), (jnp.int32(0),) * len(grid))
  )

  # Read the output from the allocated output buffers.
  out_buffer_ids = buffer_ids[len(block_args):len(block_args) + len(out)]
  out_invars = (
      jaxpr.invars[grid_mapping.slice_block_ops][-grid_mapping.num_outputs:])
  out_out = [
    callback.io_callback(
        get,
        var.aval,
        device_coords,
        TPU_MEMORY_SPACE_IDXS[var.aval.memory_space],
        out_buffer_id,
        (),
        ordered=True)
    for var, out_buffer_id in zip(out_invars, out_buffer_ids)
  ]

  # TODO(jburnim): De-allocate buffers and semaphores.

  return out_out
