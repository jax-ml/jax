# Copyright 2018 The JAX Authors.
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
"""Implementation of pmap and related functionality."""

from __future__ import annotations

import collections
from collections.abc import Callable, Sequence, Iterable
import dataclasses
from functools import partial
import functools
import itertools as it
import logging
import math
from typing import Any, NamedTuple, Union

import numpy as np

from jax._src import api
from jax._src import array
from jax._src import compiler
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import jaxpr_util
from jax._src import literals
from jax._src import op_shardings
from jax._src import pjit
from jax._src import profiler
from jax._src import sharding_impls
from jax._src import stages
from jax._src import tree_util
from jax._src import typing
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.abstract_arrays import array_types
from jax._src.core import ShapedArray
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import mlir
from jax._src.layout import Layout, AutoLayout, Format
from jax._src.lib import _jax
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.partition_spec import PartitionSpec
from jax._src.sharding import (Sharding as JSharding, IndivisibleError,
                               common_is_equivalent_to)
from jax._src.mesh import (AbstractMesh, Mesh, get_abstract_mesh,
                           get_concrete_mesh)
from jax._src.sharding_impls import (
    ArrayMapping, AUTO, UnspecifiedValue, SingleDeviceSharding,
    make_single_device_sharding, GSPMDSharding,
    NamedSharding, PartitionSpec as P)
from jax._src.util import (safe_map, safe_zip, partition_list,
                           unzip2, weakref_lru_cache, tuple_insert)
from jax._src.state.types import AbstractRef, RefEffect
from jax._src.typing import ArrayLike

unsafe_map, map = map, safe_map
zip, unsafe_zip = safe_zip, zip

logger = logging.getLogger(__name__)

Index = Union[int, slice, tuple[Union[int, slice], ...]]
PyTreeDef = tree_util.PyTreeDef

MeshAxisName = sharding_impls.MeshAxisName

### util

def identity(x): return x


@profiler.annotate_function
def shard_args(
    shardings: Sequence[JSharding],
    layouts: Sequence[Any | None],
    copy_semantics: Sequence[xc.ArrayCopySemantics],
    args: Sequence[Any],
    canonicalize: bool = True,
) -> Sequence[xc.ArrayImpl]:
  # Fast path for one argument.
  if len(args) == 1:
    arg = args[0]
    if canonicalize:
      arg = dtypes.canonicalize_value(arg)
    return shard_arg_handlers[type(arg)]([arg], shardings, layouts,
                                         copy_semantics)

  # type(arg) -> (list[indices], list[args], list[shardings], list[layouts],
  #               list[copy_semantics])
  batches = collections.defaultdict(lambda: ([], [], [], [], []))
  for i, (arg, sharding, layout, cs) in enumerate(
      safe_zip(args, shardings, layouts, copy_semantics)):
    if canonicalize:
      arg = dtypes.canonicalize_value(arg)
    batch = batches[type(arg)]
    batch[0].append(i)
    batch[1].append(arg)
    batch[2].append(sharding)
    batch[3].append(layout)
    batch[4].append(cs)

  # Call `shard_arg_handlers` per batch and build a flat list of arrays returned
  # from each call in the same order as `args`. Since `batches` is grouped by
  # types, we cannot simply flatten the results and we have to use the original
  # indices to put each array back to its original position.
  results: list[typing.Array | None] = [None] * len(args)
  for t, (indices, a, s, l, xcs) in batches.items():
    outs = shard_arg_handlers[t](a, s, l, xcs)
    for i, out in safe_zip(indices, outs):
      results[i] = out
  assert all(result is not None for result in results)
  return results


shard_arg_handlers: dict[
    Any,
    Callable[
      [Sequence[Any], Sequence[Any], Sequence[Any],
       Sequence[xc.ArrayCopySemantics]],
      Sequence[Any],
    ],
] = {}


@util.cache(max_size=2048, trace_context_in_key=False)
def is_default_layout(curr_layout, sharding, aval):
  if curr_layout is None or sharding is None or isinstance(sharding, UnspecifiedValue):
    return True
  if (aval is core.abstract_token or aval.dtype == dtypes.float0 or
      dtypes.issubdtype(aval.dtype, dtypes.extended)):
    return True
  if isinstance(curr_layout, AutoLayout):
    return False
  d = sharding._device_assignment[0]
  shard_shape = sharding.shard_shape(aval.shape)
  try:
    # TODO(yashkatariya): Replace this with normal `==` check once CPU supports
    # int4.
    return is_user_xla_layout_equal(
        curr_layout,
        Layout.from_pjrt_layout(
            d.client.get_default_layout(aval.dtype, shard_shape, d)))
  except _jax.JaxRuntimeError as e:
    msg, *_ = e.args
    if isinstance(msg, str) and msg.startswith("UNIMPLEMENTED"):
      return True
    else:
      raise


def _masked_array_error(xs, shardings, layouts, copy_semantics):
  raise ValueError("numpy masked arrays are not supported as direct inputs to JAX functions. "
                   "Use arr.filled() to convert the value to a standard numpy array.")
shard_arg_handlers[np.ma.MaskedArray] = _masked_array_error

def _shard_np_array(xs, shardings, layouts, copy_semantics):
  results = []
  for x, sharding, layout in safe_zip(xs, shardings, layouts):
    devices = sharding._addressable_device_assignment
    if x.dtype == dtypes.float0:
      x = np.zeros(x.shape, dtype=np.dtype(bool))
    aval = core.shaped_abstractify(x)
    if layout is not None:
      results.append(api.device_put(x, Format(layout, sharding)))
    else:
      if sharding.is_fully_replicated:
        shards = [x] * len(devices)
      else:
        indices = tuple(sharding.addressable_devices_indices_map(x.shape).values())
        shards = [x[i] for i in indices]
      results.append(batched_device_put(aval, sharding, shards, devices))
  return results
for _t in array_types:
  shard_arg_handlers[_t] = _shard_np_array

shard_arg_handlers[literals.TypedNdArray] = _shard_np_array

def _shard_python_scalar(xs, shardings, layouts, copy_semantics):
  return shard_args(shardings, layouts, copy_semantics,
                    [np.array(x) for x in xs])
for _t in dtypes.python_scalar_types:
  shard_arg_handlers[_t] = _shard_python_scalar

def _shard_typed_scalar(xs, shardings, layouts, copy_semantics):
  return _shard_np_array(
      [literals.TypedNdArray(
        np.array(x, dtype=x.dtype),
        aval=core.ShapedArray((), x.dtype, weak_type=True))
       for x in xs],
      shardings, layouts, copy_semantics
  )
for _t in literals.typed_scalar_types:
  shard_arg_handlers[_t] = _shard_typed_scalar

def _shard_mutable_array(xs, shardings, layouts, copy_semantics):
  bufs = [x._refs._buf for x in xs]
  return shard_args(shardings, layouts, copy_semantics, bufs)
shard_arg_handlers[core.Ref] = _shard_mutable_array

def batched_device_put(aval: core.ShapedArray,
                       sharding: JSharding, xs: Sequence[Any],
                       devices: Sequence[xc.Device], committed: bool = True,
                       enable_x64: bool | None = None):
  util.test_event("batched_device_put_start")
  try:
    bufs = [x for x, d in safe_zip(xs, devices)
            if (isinstance(x, array.ArrayImpl) and x.sharding.num_devices == 1
                and x.devices() == {d})]
    if len(bufs) == len(xs) > 0:
      return array.ArrayImpl(
          aval, sharding, bufs, committed=committed, _skip_checks=True)
    return xc.batched_device_put(aval, sharding, xs, list(devices), committed,
                                 enable_x64=enable_x64)
  finally:
    util.test_event("batched_device_put_end")

def global_aval_to_result_handler(
    aval: core.AbstractValue, out_sharding, committed: bool
) -> Callable[[Sequence[xc.ArrayImpl]], Any]:
  """Returns a function for handling the raw buffers of a single output aval.

  Args:
    aval: The global output AbstractValue.
    out_axis_resources: A PartitionSpec specifying the sharding of outputs.
      Used for creating GSDAs.
    global_mesh: The global device mesh that generated this output. Used
      for creating GSDAs.

  Returns:
    A function for handling the Buffers that will eventually be produced
    for this output. The function will return an object suitable for returning
    to the user, e.g. an Array.
  """
  try:
    return global_result_handlers[type(aval)](aval, out_sharding, committed)
  except KeyError as err:
    raise TypeError(
        f"No pxla_result_handler for type: {type(aval)}") from err

PxlaResultHandler = Callable[..., xc._xla.ResultHandler]
global_result_handlers: dict[type[core.AbstractValue], PxlaResultHandler] = {}


class InputsHandler:
  __slots__ = ("handler", "in_shardings", "in_layouts", "local_devices",
               "input_indices")

  def __init__(self, in_shardings, in_layouts, local_devices=None,
               input_indices=None):
    self.handler = partial(
        shard_args, in_shardings, in_layouts,
        [xc.ArrayCopySemantics.REUSE_INPUT] * len(in_shardings))
    self.in_shardings = in_shardings
    self.in_layouts = in_layouts
    self.local_devices = local_devices
    self.input_indices = input_indices

  def __call__(self, input_buffers):
    return self.handler(input_buffers)

  def __str__(self):
    return ("InputsHandler(\n"
            f"in_shardings={self.in_shardings},\n"
            f"in_layouts={self.in_layouts},\n"
            f"local_devices={self.local_devices},\n"
            f"input_indices={self.input_indices})")


class ResultsHandler:
  __slots__ = ("handlers", "out_shardings", "out_avals")

  def __init__(self, handlers, out_shardings, out_avals):
    self.handlers = handlers
    self.out_shardings = out_shardings
    self.out_avals = out_avals

  def __call__(self, out_bufs):
    return [h(bufs) for h, bufs in safe_zip(self.handlers, out_bufs)]


def global_avals_to_results_handler(
    global_out_avals: Sequence[ShapedArray],
    shardings: Sequence[JSharding],
    committed: bool) -> ResultsHandler:
  handlers = [
      global_aval_to_result_handler(global_aval, s, committed)
      for global_aval, s in safe_zip(global_out_avals, shardings)
  ]
  return ResultsHandler(handlers, shardings, global_out_avals)


class ExecuteReplicated:
  """The logic to shard inputs, execute a replicated model, returning outputs."""
  __slots__ = ['xla_executable', 'name', 'backend', 'in_handler', 'out_handler',
               'has_unordered_effects', 'ordered_effects', 'keepalive',
               'has_host_callbacks', '_local_devices', 'kept_var_idx',
               'mut', 'pgle_profiler', '__weakref__']

  def __init__(self, xla_executable, name, backend, in_handler: InputsHandler,
               out_handler: ResultsHandler,
               unordered_effects: list[core.Effect],
               ordered_effects: list[core.Effect], keepalive: Any,
               has_host_callbacks: bool, kept_var_idx: set[int],
               mut: MutationData | None,
               pgle_profiler: profiler.PGLEProfiler | None = None):
    self.xla_executable = xla_executable
    self.name = name
    self.backend = backend
    self.in_handler = in_handler
    self.out_handler = out_handler
    self.has_unordered_effects = bool(unordered_effects)
    self.ordered_effects = ordered_effects
    self._local_devices = self.xla_executable.local_devices()
    self.keepalive = keepalive
    self.has_host_callbacks = has_host_callbacks
    self.kept_var_idx = kept_var_idx
    self.mut = mut
    self.pgle_profiler = pgle_profiler

  def _add_tokens_to_inputs(self, input_bufs):
    if self.ordered_effects:
      tokens = [
          dispatch.runtime_tokens.get_token_input(eff, self._local_devices)._buf
          for eff in self.ordered_effects
      ]
      input_bufs = [*tokens, *input_bufs]
    return input_bufs

  def _handle_token_bufs(self, token_bufs, sharded_token):
    # token_bufs: Sequence[Sequence[tokenArray]], for each effect the returned
    # token buffers.
    # sharded_token: ShardedToken, containing the RuntimeTokens for each device
    for i, device in enumerate(self._local_devices):
      dispatch.runtime_tokens.set_output_runtime_token(
          device, sharded_token.get_token(i))
    for eff, token_buf in zip(self.ordered_effects, token_bufs):
      assert len(token_buf) > 0
      if len(token_buf) == 1:
        dispatch.runtime_tokens.set_token_result(eff, core.Token(token_buf[0]))
      else:
        token_devices = []
        for token in token_buf:
          assert len(token.sharding.device_set) == 1
          token_devices.append(token.sharding._device_assignment[0])
        s = NamedSharding(Mesh(token_devices, 'x'), P('x'))
        global_token_array = array.make_array_from_single_device_arrays(
            (0,), s, token_buf
        )
        dispatch.runtime_tokens.set_token_result(
            eff, core.Token(global_token_array)
        )

  @profiler.annotate_function
  def __call__(self, *args):
    if config.no_execution.value:
      raise RuntimeError(
      f"JAX tried to execute function {self.name}, but the no_execution config "
      "option is set")
    args = [x for i, x in enumerate(args) if i in self.kept_var_idx]
    if self.mut:
      args = [*args, *self.mut.in_mut]
    input_bufs = self.in_handler(args)
    with profiler.PGLEProfiler.trace(self.pgle_profiler):
      if (self.ordered_effects or self.has_unordered_effects
          or self.has_host_callbacks):
        input_bufs = self._add_tokens_to_inputs(input_bufs)
        results = self.xla_executable.execute_sharded(input_bufs, with_tokens=True)

        result_token_bufs = results.consume_with_handlers(
            [lambda xs: xs] * len(self.ordered_effects), strict=False)
        sharded_runtime_token = results.consume_token()
        self._handle_token_bufs(result_token_bufs, sharded_runtime_token)
      else:
        results = self.xla_executable.execute_sharded(input_bufs)

      handlers = self.out_handler.handlers
      if dispatch.needs_check_special():
        special_check = functools.partial(
            dispatch.check_special_array, self.name)
        handlers = [h.pre_wrap(special_check) for h in handlers]
      out = results.consume_with_handlers(handlers)

      if (self.pgle_profiler is not None and self.pgle_profiler.is_running()
          and len(out) > 0):
        out[0].block_until_ready()

    if self.mut is None:
      return out
    else:
      out_ = []
      for i, o in zip(self.mut.out_mut, out):
        if i is not None:
          try: args[i]._refs._buf._replace_with(o)
          except AttributeError: pass  # TODO(mattjj): remove float0
        else:
          out_.append(o)
      return out_


def _axis_read(axis_env, axis_name):
  try:
    return max(i for i, name in enumerate(axis_env.names) if name == axis_name)
  except ValueError:
    raise NameError(f"unbound axis name: {axis_name}") from None

def axis_groups(axis_env: sharding_impls.AxisEnv, name) -> tuple[tuple[int, ...]]:
  if not isinstance(name, (list, tuple)):
    name = (name,)
  mesh_axes = tuple(unsafe_map(partial(_axis_read, axis_env), name))
  trailing_size, ragged = divmod(axis_env.nreps, math.prod(axis_env.sizes))
  assert not ragged
  mesh_spec = axis_env.sizes + (trailing_size,)
  return _axis_groups(mesh_spec, mesh_axes)

def _axis_groups(mesh_spec, mesh_axes):
  """Computes replica group ids for a collective performed over a subset of the mesh.

  Args:
    mesh_spec: A sequence of integers representing the mesh shape.
    mesh_axes: A sequence of integers between 0 and `len(mesh_spec)` (exclusive)
      indicating over which axes the collective is performed.
  Returns:
    A tuple of replica groups (i.e. tuples containing replica ids).
  """
  iota = np.arange(math.prod(mesh_spec)).reshape(mesh_spec)
  groups = np.reshape(
      np.moveaxis(iota, mesh_axes, np.arange(len(mesh_axes))),
      (math.prod(np.take(mesh_spec, mesh_axes)), -1))
  return tuple(unsafe_map(tuple, groups.T))


def tile_aval_nd(axis_sizes, in_axes: ArrayMapping, aval):
  assert isinstance(aval, ShapedArray)
  shape = list(aval.shape)
  for name, axis in in_axes.items():
    assert shape[axis] % axis_sizes[name] == 0
    shape[axis] //= axis_sizes[name]
  return aval.update(shape=tuple(shape))

def untile_aval_nd(axis_sizes, out_axes: ArrayMapping, aval):
  assert isinstance(aval, ShapedArray)
  shape = list(aval.shape)
  for name, axis in out_axes.items():
    shape[axis] *= axis_sizes[name]
  return aval.update(shape=tuple(shape))


def mesh_local_to_global(mesh, axes: ArrayMapping, aval):
  return untile_aval_nd(mesh.shape, axes,
                        tile_aval_nd(mesh.local_mesh.shape, axes, aval))

def mesh_global_to_local(mesh, axes: ArrayMapping, aval):
  return untile_aval_nd(mesh.local_mesh.shape, axes,
                        tile_aval_nd(mesh.shape, axes, aval))


def manual_proto(
    aval: core.ShapedArray,
    manual_axes_set: frozenset[sharding_impls.MeshAxisName], mesh: Mesh):
  """Create an OpSharding proto that declares all mesh axes from `axes` as manual
  and all others as replicated.
  """
  named_mesh_shape = mesh.shape
  mesh_shape = list(named_mesh_shape.values())
  axis_order = {axis: i for i, axis in enumerate(mesh.axis_names)}

  manual_axes = sorted(manual_axes_set, key=str)
  replicated_axes = [axis for axis in mesh.axis_names
                     if axis not in manual_axes_set]

  tad_perm = ([axis_order[a] for a in replicated_axes] +
              [axis_order[a] for a in manual_axes])
  tad_shape = [1] * aval.ndim
  tad_shape.append(math.prod([named_mesh_shape[a] for a in replicated_axes]))
  tad_shape.append(math.prod([named_mesh_shape[a] for a in manual_axes]))

  proto = xc.OpSharding()
  proto.type = xc.OpSharding.Type.OTHER
  proto.tile_assignment_dimensions = tad_shape
  proto.iota_reshape_dims = mesh_shape
  proto.iota_transpose_perm = tad_perm
  proto.last_tile_dims = [xc.OpSharding.Type.REPLICATED, xc.OpSharding.Type.MANUAL]
  return proto


def check_if_any_auto(
    shardings: Iterable[(JSharding | AUTO | UnspecifiedValue)]) -> bool:
  for s in shardings:
    if isinstance(s, AUTO):
      return True
  return False


ShardingInfo = tuple[
    Union[JSharding, UnspecifiedValue, AUTO],
    stages.MismatchType,
    Union[Any, None],  # Any is dispatch.SourceInfo to avoid circular imports
]


def get_default_device() -> xc.Device:
  if isinstance(config.default_device.value, str):
    return xb.get_backend(config.default_device.value).local_devices()[0]
  else:
    return config.default_device.value or xb.local_devices()[0]


def _get_and_check_device_assignment(
    shardings: Iterable[ShardingInfo],
    ctx_mesh: Mesh | AbstractMesh,
) -> tuple[xc.Client, tuple[xc.Device, ...] | None, int]:
  first_sharding_info = None
  abstract_mesh = (
      ctx_mesh if not ctx_mesh.empty and isinstance(ctx_mesh, AbstractMesh)
      else None)
  any_concrete_sharding = (
      True if not ctx_mesh.empty and isinstance(ctx_mesh, Mesh) else False)

  for sh, s_type, source_info in shardings:
    if isinstance(sh, UnspecifiedValue):
      continue
    elif isinstance(sh, NamedSharding) and isinstance(sh.mesh, AbstractMesh):
      if (abstract_mesh is not None and not sh.mesh.empty and
          abstract_mesh.size != sh.mesh.size):
        raise ValueError("AbstractMesh should be of the same size across all "
                         f"shardings. Got {abstract_mesh} and {sh.mesh}")
      abstract_mesh = sh.mesh
    else:
      any_concrete_sharding = True
      arr_device_assignment = sh._device_assignment
      if first_sharding_info is None:
        first_sharding_info = (arr_device_assignment, s_type, source_info)
      if ctx_mesh.empty:
        if first_sharding_info[0] != arr_device_assignment:
          raise stages.DeviceAssignmentMismatchError([
              stages.DeviceAssignmentMismatch(*first_sharding_info),
              stages.DeviceAssignmentMismatch(
                  arr_device_assignment, s_type, source_info)])
      elif isinstance(ctx_mesh, AbstractMesh):
        if ctx_mesh.size != len(arr_device_assignment):
          raise stages.DeviceAssignmentMismatchError([
              stages.DeviceAssignmentMismatch(
                  ctx_mesh.size, stages.MismatchType.CONTEXT_DEVICES, None),
              stages.DeviceAssignmentMismatch(
                  arr_device_assignment, s_type, source_info)])
      else:
        if ctx_mesh._flat_devices_tuple != arr_device_assignment:
          raise stages.DeviceAssignmentMismatchError([
              stages.DeviceAssignmentMismatch(
                  ctx_mesh._flat_devices_tuple,
                  stages.MismatchType.CONTEXT_DEVICES, None),
              stages.DeviceAssignmentMismatch(
                  arr_device_assignment, s_type, source_info)])

  device_assignment: tuple[xc.Device, ...]
  if (first_sharding_info is None and not ctx_mesh.empty and
      isinstance(ctx_mesh, Mesh)):
    device_assignment = ctx_mesh._flat_devices_tuple
  elif first_sharding_info is None:
    device_assignment = (get_default_device(),)
  else:
    device_assignment = first_sharding_info[0]  # pyrefly: ignore[bad-assignment]

  backend = xb.get_device_backend(device_assignment[0])

  if (any_concrete_sharding and abstract_mesh is not None and
      len(device_assignment) != abstract_mesh.size):
    raise ValueError(
        f"AbstractMesh size: {abstract_mesh.size} does not match the"
        f" device assignment size: {len(device_assignment)}")

  if any_concrete_sharding or abstract_mesh is None:
    return backend, device_assignment, len(device_assignment)
  else:
    return backend, None, abstract_mesh.size

MaybeSharding = Union[JSharding, UnspecifiedValue]


def prune_unused_inputs(
    jaxpr: core.Jaxpr,
) -> tuple[core.Jaxpr, set[int], set[int]]:
  used_outputs = [True] * len(jaxpr.outvars)
  new_jaxpr, used_consts, used_inputs = pe.dce_jaxpr_consts(jaxpr, used_outputs)
  kept_const_idx = {i for i, b in enumerate(used_consts) if b}
  kept_var_idx = {i for i, b in enumerate(used_inputs) if b}
  return new_jaxpr, kept_const_idx, kept_var_idx


@weakref_lru_cache
def _dce_jaxpr(closed_jaxpr, keep_unused, donated_invars, auto_spmd_lowering):
  assert isinstance(closed_jaxpr, core.ClosedJaxpr)
  jaxpr = closed_jaxpr.jaxpr
  consts = closed_jaxpr.consts
  in_avals = closed_jaxpr.in_avals

  if (keep_unused or auto_spmd_lowering or
      any(hasattr(a, "shape") and not core.is_constant_shape(a.shape)
          for a in in_avals)):
    kept_var_idx = set(range(len(in_avals)))
  else:
    jaxpr, kept_const_idx, kept_var_idx = prune_unused_inputs(jaxpr)
    consts = [c for i, c in enumerate(consts) if i in kept_const_idx]
    donated_invars = tuple(x for i, x in enumerate(donated_invars) if i in kept_var_idx)
    del kept_const_idx

  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  return closed_jaxpr, donated_invars, kept_var_idx


class MutationData(NamedTuple):
  in_mut: list[core.Ref]
  # out_mut[o_idx] = i_idx, when the output[o_idx] corresponds to the
  # mutable array args[i_idx]. None when it does not correspond to a mutable array.
  out_mut: list[int | None]


@weakref_lru_cache
def _discharge_refs(
    jaxpr: core.ClosedJaxpr
) -> tuple[core.ClosedJaxpr, Sequence[int | None], MutationData]:
  from jax._src.state.discharge import discharge_state2  # pytype: disable=import-error
  jaxpr, in_mut = _move_mutable_consts(jaxpr)
  new_jaxpr = discharge_state2(jaxpr)
  count = it.count(len(jaxpr.out_avals))  # new outputs are appended to the end
  inout_map = {i: next(count) for i, a in enumerate(jaxpr.in_avals)
               if isinstance(a, AbstractRef)}
  outin_map = {j: i for i, j in inout_map.items()}
  inout_aliases = tuple(map(inout_map.get, range(len(new_jaxpr.in_avals))))
  out_mut = list(map(outin_map.get, range(len(new_jaxpr.out_avals))))
  return new_jaxpr, inout_aliases, MutationData(in_mut, out_mut)

@weakref_lru_cache
def _move_mutable_consts(
    closed_jaxpr: core.ClosedJaxpr,
) -> tuple[core.ClosedJaxpr, list[core.Ref]]:
  jaxpr = closed_jaxpr.jaxpr
  hoist = [isinstance(c, core.Ref) for c in closed_jaxpr.consts]
  consts, in_mut = partition_list(hoist, closed_jaxpr.consts)
  constvars, mutvars = partition_list(hoist, jaxpr.constvars)
  invars = (*jaxpr.invars, *mutvars)
  effects = pe.make_jaxpr_effects(constvars, invars, jaxpr.outvars, jaxpr.eqns)
  # TODO(mattjj): debug_info must be updated...
  jaxpr = closed_jaxpr.jaxpr.replace(
      constvars=constvars, invars=invars, effects=effects,
      debug_info=closed_jaxpr.debug_info.with_unknown_names())
  return core.ClosedJaxpr(jaxpr, consts), in_mut

@weakref_lru_cache
def _discharge_internal_refs(jaxpr: core.ClosedJaxpr) -> core.ClosedJaxpr:
  from jax._src.state.discharge import discharge_state  # pytype: disable=import-error
  jaxpr_, consts = discharge_state(jaxpr.jaxpr, jaxpr.consts)
  jaxpr_._debug_info = jaxpr.jaxpr._debug_info
  return core.ClosedJaxpr(jaxpr_, consts)


class SemanticallyEqualShardings:

  def __init__(self, shardings, avals):
    self.shardings = shardings
    self.avals = avals

  def __hash__(self):
    return hash(tuple(s if isinstance(s, (UnspecifiedValue, AUTO)) else
                      (s._to_xla_hlo_sharding(a.ndim), s.memory_kind)
                      for s, a in zip(self.shardings, self.avals)))

  def __eq__(self, other):
    if not isinstance(other, SemanticallyEqualShardings):
      return False
    is_ua = lambda x: isinstance(x, (UnspecifiedValue, AUTO))
    return all(common_is_equivalent_to(s, o, a.ndim, check_devices=False)
               if not is_ua(s) and not is_ua(o) else s == o
               for s, o, a in zip(self.shardings, other.shardings, self.avals))


@weakref_lru_cache
def _cached_lowering_to_hlo(
    closed_jaxpr: core.ClosedJaxpr, module_name, backend, num_const_args: int,
    in_avals, semantic_in_shardings, semantic_out_shardings,
    in_layouts, out_layouts, num_devices, device_assignment, donated_invars,
    all_default_mem_kind, inout_aliases: None | tuple[None | int, ...],
    propagated_out_mem_kinds: tuple[None | str, ...], platforms: tuple[str, ...],
    lowering_parameters: mlir.LoweringParameters,
    abstract_mesh: AbstractMesh | None):
  # in_avals, in_shardings, in_layouts include the jaxpr_const_args(jaxpr)
  out_avals = closed_jaxpr.out_avals
  jaxpr = closed_jaxpr.jaxpr
  in_shardings = semantic_in_shardings.shardings
  out_shardings = semantic_out_shardings.shardings

  log_priority = logging.WARNING if config.log_compiles.value else logging.DEBUG
  if logger.isEnabledFor(log_priority):
    logger.log(log_priority,
               "Compiling %s with global shapes and types %s. "
               "Argument mapping: %s.",
               module_name, in_avals, in_shardings)

  in_mlir_shardings = map(_to_logical_sharding, in_avals, in_shardings)
  out_mlir_shardings = map(_to_logical_sharding, out_avals, out_shardings)
  replicated_args = [False] * len(in_avals)
  axis_ctx = sharding_impls.ShardingContext(num_devices, device_assignment,
                                            abstract_mesh)

  if num_devices > 1:
    unsupported_effects = effects.ordered_effects.filter_in(closed_jaxpr.effects)
    unsupported_effects = effects.shardable_ordered_effects.filter_not_in(
        unsupported_effects)
    if len(unsupported_effects) > 0:
      raise ValueError(
        "The following ordered effects are not supported for "
        f"more than 1 device: {unsupported_effects}")
  ordered_effects = list(effects.ordered_effects.filter_in(closed_jaxpr.effects))
  arg_names = ("",) * num_const_args + jaxpr._debug_info.safe_arg_names(len(in_avals) - num_const_args)
  with dispatch.log_elapsed_time(
        "Finished jaxpr to MLIR module conversion {fun_name} in {elapsed_time:.9f} sec",
        fun_name=module_name, event=dispatch.JAXPR_TO_MLIR_MODULE_EVENT):
    lowering_result = mlir.lower_jaxpr_to_module(
        module_name,
        closed_jaxpr,
        num_const_args=num_const_args,
        ordered_effects=ordered_effects,
        backend=backend,
        platforms=platforms,
        axis_context=axis_ctx,
        in_avals=in_avals,
        donated_args=donated_invars,
        replicated_args=replicated_args,
        arg_shardings=in_mlir_shardings,
        result_shardings=out_mlir_shardings,
        in_layouts=in_layouts,
        out_layouts=out_layouts,
        arg_names=arg_names,
        result_names=jaxpr._debug_info.safe_result_paths(len(out_avals)),
        num_partitions=num_devices,
        all_default_mem_kind=all_default_mem_kind,
        input_output_aliases=inout_aliases,
        propagated_out_mem_kinds=propagated_out_mem_kinds,
        lowering_parameters=lowering_parameters)
  tuple_args = dispatch.should_tuple_args(len(in_avals), backend.platform)
  unordered_effects = list(
      effects.ordered_effects.filter_not_in(closed_jaxpr.effects))
  return (lowering_result.module, lowering_result.keepalive,
          lowering_result.host_callbacks, unordered_effects, ordered_effects,
          tuple_args, lowering_result.shape_poly_state)


@util.cache(max_size=2048, trace_context_in_key=False)
def _create_device_list_cached(device_assignment: tuple[xc.Device, ...]
                             ) -> xc.DeviceList:
  return xc.DeviceList(device_assignment)

def _create_device_list(
    device_assignment: tuple[xc.Device, ...] | xc.DeviceList | None
    ) -> xc.DeviceList | None:
  if device_assignment is None or isinstance(device_assignment, xc.DeviceList):
    return device_assignment
  return _create_device_list_cached(device_assignment)


@weakref_lru_cache
def jaxpr_transfer_mem_kinds(jaxpr: core.Jaxpr):
  out = []
  for eqn in jaxpr.eqns:
    if eqn.primitive is dispatch.device_put_p:
      out.extend(d for d in eqn.params['devices']
                 if isinstance(d, core.MemorySpace))
    elif eqn.primitive.name == 'compute_on':
      out.extend(o for o in eqn.params['out_memory_spaces'])
    elif eqn.primitive.name == 'call_exported':
      out.extend(aval.memory_space for aval in eqn.params['exported'].out_avals)

  for subjaxpr in core.subjaxprs(jaxpr):
    out.extend(jaxpr_transfer_mem_kinds(subjaxpr))
  return out


def are_all_shardings_default_mem_kind(shardings):
  for i in shardings:
    if isinstance(i, (UnspecifiedValue, AUTO)):
      continue
    mem_kind = (core.mem_space_to_kind(i) if isinstance(i, core.MemorySpace)
                else i.memory_kind)
    if mem_kind is None:
      continue
    if mem_kind != 'device':
      return False
  return True


@weakref_lru_cache
def get_out_layouts_via_propagation(closed_jaxpr: core.ClosedJaxpr
                                    ) -> tuple[None | Layout]:
  env = {}
  jaxpr = closed_jaxpr.jaxpr

  def read(var):
    if type(var) is core.Literal:
      return None
    return env[var]

  def write(var, val):
    env[var] = val

  safe_map(write, jaxpr.invars, [None] * len(jaxpr.invars))
  safe_map(write, jaxpr.constvars, [None] * len(jaxpr.constvars))

  for eqn in jaxpr.eqns:
    if eqn.primitive is pjit.sharding_constraint_p:
      out_eqn_layouts = [eqn.params['layout']]
    elif eqn.primitive is pjit.layout_constraint_p:
      out_eqn_layouts = [eqn.params['layout']]
    else:
      out_eqn_layouts = [None] * len(eqn.outvars)
    safe_map(write, eqn.outvars, out_eqn_layouts)
  return tuple(safe_map(read, jaxpr.outvars))


MaybeLayout = Sequence[Union[Layout, AutoLayout, None]]


class AllArgsInfo(NamedTuple):
  """Avals and debug_info for all arguments prior to DCE."""
  in_avals: Sequence[core.ShapedArray]
  debug_info: core.DebugInfo


def _discharge_refs_jaxpr(closed_jaxpr, in_shardings, in_layouts,
                          donated_invars, out_shardings, out_layouts):
  if (any(isinstance(e, RefEffect) for e in closed_jaxpr.effects) or
      any(isinstance(a, AbstractRef) for a in closed_jaxpr.in_avals)):
    closed_jaxpr, inout_aliases, mut = _discharge_refs(closed_jaxpr)
    in_shardings = (*in_shardings, *(
        pjit.finalize_arg_sharding(c.sharding, c.committed) for c in mut.in_mut))
    in_layouts = (*in_layouts, *(c.format.layout if hasattr(c, 'format')
                                 else None for c in mut.in_mut))
    donated_invars = (*donated_invars,) + (False,) * len(mut.in_mut)
    out_layouts_ = iter(zip(out_shardings, out_layouts))
    out_shardings, out_layouts = unzip2(
        next(out_layouts_) if i is None else (in_shardings[i], in_layouts[i])
        for i in mut.out_mut)
    assert next(out_layouts_, None) is None
  else:
    inout_aliases = mut = None
    if any(isinstance(e, core.InternalMutableArrayEffect) for e in closed_jaxpr.effects):
      closed_jaxpr = _discharge_internal_refs(closed_jaxpr)

  return (closed_jaxpr, inout_aliases, mut, in_shardings, in_layouts,
          donated_invars, out_shardings, out_layouts)


def hoist_constants_as_args(
    closed_jaxpr: core.ClosedJaxpr, global_in_avals, in_shardings, in_layouts,
    donated_invars, kept_var_idx: set[int], inout_aliases, mut,
    all_args_info: AllArgsInfo):
  const_args, const_arg_avals = unzip2(
      core.jaxpr_const_args(closed_jaxpr.jaxpr)
  )
  num_const_args = len(const_args)
  if num_const_args:
    global_in_avals = list(const_arg_avals) + global_in_avals
    ca_shardings = pjit.const_args_shardings(const_args)
    in_shardings = (*ca_shardings, *in_shardings)
    ca_layouts = pjit.const_args_layouts(const_args, const_arg_avals,
                                          ca_shardings)
    in_layouts = (*ca_layouts, *in_layouts)

    donated_invars = (False,) * num_const_args + donated_invars
    kept_var_idx = set(range(num_const_args)).union(
        {kv + num_const_args for kv in kept_var_idx})
    if inout_aliases is not None:
      inout_aliases = (None,) * num_const_args + inout_aliases
    if mut is not None:
      mut = MutationData(
          in_mut=mut.in_mut,
          out_mut=[None if i_idx is None else i_idx + num_const_args
                   for i_idx in mut.out_mut])
    if all_args_info.debug_info.arg_names is None:
      arg_names = None
    else:
      arg_names = (("",) * num_const_args + all_args_info.debug_info.arg_names)
    all_args_info = AllArgsInfo(
        [*const_arg_avals, *all_args_info.in_avals],
        all_args_info.debug_info._replace(arg_names=arg_names))

  return (const_args, global_in_avals, in_shardings, in_layouts, donated_invars,
          kept_var_idx, inout_aliases, mut, all_args_info)


@util.cache(max_size=1024, trace_context_in_key=False)
def _abstract_to_concrete_mesh(abstract_mesh, device_assignment):
  np_dev = np.vectorize(lambda i: device_assignment[i],
                        otypes=[object])(np.arange(len(device_assignment)))
  return Mesh(np_dev.reshape(abstract_mesh.axis_sizes),
              abstract_mesh.axis_names, axis_types=abstract_mesh.axis_types)

def _concretize_abstract_out_shardings(shardings, avals, device_assignment,
                                       out_mem_kinds):
  if device_assignment is None:
    return shardings

  out: list[UnspecifiedValue | JSharding] = []
  for s, a, mem_kind in zip(shardings, avals, out_mem_kinds):
    if isinstance(s, UnspecifiedValue) and isinstance(a, core.ShapedArray):
      if a.sharding.mesh.empty:
        out.append(s)
      elif a.sharding.mesh._are_all_axes_auto_or_manual:
        out.append(s)
      else:
        spec = (PartitionSpec(*[PartitionSpec.UNCONSTRAINED if sp is None else sp
                                for sp in a.sharding.spec])
                if a.sharding.mesh._any_axis_auto else a.sharding.spec)
        out.append(NamedSharding(
            _abstract_to_concrete_mesh(a.sharding.mesh, device_assignment),
            spec, memory_kind=mem_kind))
    else:
      out.append(s)
  return tuple(out)


def _get_context_mesh(context_mesh: Mesh | AbstractMesh) -> Mesh | AbstractMesh:
  if isinstance(context_mesh, AbstractMesh):
    return context_mesh
  if get_concrete_mesh().empty:
    return context_mesh
  cur_mesh = get_abstract_mesh()
  if cur_mesh.empty or context_mesh.empty:
    return context_mesh
  if cur_mesh == context_mesh.abstract_mesh:
    return context_mesh
  assert context_mesh.size == cur_mesh.size
  return Mesh(context_mesh.devices.reshape(cur_mesh.axis_sizes),
              cur_mesh.axis_names, cur_mesh.axis_types)


@profiler.annotate_function
def lower_sharding_computation(
    closed_jaxpr: core.ClosedJaxpr,
    api_name: str,
    fun_name: str,
    in_shardings: Sequence[MaybeSharding],
    out_shardings: Sequence[MaybeSharding],
    in_layouts: MaybeLayout,
    out_layouts: MaybeLayout,
    donated_invars: Sequence[bool],
    *,
    keep_unused: bool,
    context_mesh: Mesh | AbstractMesh,
    compiler_options_kvs: tuple[tuple[str, Any], ...],
    lowering_platforms: tuple[str, ...] | None,
    lowering_parameters: mlir.LoweringParameters,
    pgle_profiler: profiler.PGLEProfiler | None,
) -> MeshComputation:
  """Lowers a computation to XLA. It can take arbitrary shardings as input.

  The caller of this code can pass in a singleton UNSPECIFIED because the
  number of out_avals might not be known at that time and
  lower_sharding_computation calculates the number of out_avals so it can apply
  the singleton UNSPECIFIED to all out_avals."""
  auto_spmd_lowering = check_if_any_auto(it.chain(in_shardings, out_shardings))

  all_args_info = AllArgsInfo(closed_jaxpr.in_avals, closed_jaxpr.jaxpr._debug_info)

  closed_jaxpr, donated_invars, kept_var_idx = _dce_jaxpr(
      closed_jaxpr, keep_unused, donated_invars, auto_spmd_lowering)
  in_shardings = tuple(s for i, s in enumerate(in_shardings) if i in kept_var_idx)
  in_layouts = tuple(l for i, l in enumerate(in_layouts) if i in kept_var_idx)

  (closed_jaxpr, inout_aliases, mut, in_shardings, in_layouts,
   donated_invars, out_shardings, out_layouts) = _discharge_refs_jaxpr(
       closed_jaxpr, in_shardings, in_layouts, donated_invars, out_shardings,
       out_layouts)

  jaxpr = closed_jaxpr.jaxpr
  global_in_avals = closed_jaxpr.in_avals
  global_out_avals = closed_jaxpr.out_avals

  if lowering_parameters.hoist_constants_as_args:
    (const_args, global_in_avals, in_shardings, in_layouts, donated_invars,
     kept_var_idx, inout_aliases, mut, all_args_info) = hoist_constants_as_args(
         closed_jaxpr, global_in_avals, in_shardings, in_layouts,
         donated_invars, kept_var_idx, inout_aliases, mut, all_args_info)
  else:
    const_args = []

  # If layout is propagated, then set the out_layout in the top module to AUTO
  # so that XLA can override the entry_computation_layout. The propagated
  # layout will be set via a custom call.
  out_layouts_via_prop = get_out_layouts_via_propagation(closed_jaxpr)
  out_layouts = tuple(Layout.AUTO if p is not None else o
                      for o, p in safe_zip(out_layouts, out_layouts_via_prop))

  assert len(out_shardings) == len(out_layouts) == len(global_out_avals), (
      len(out_shardings), len(out_layouts), len(global_out_avals))

  context_mesh = _get_context_mesh(context_mesh)

  # Device assignment across all inputs, outputs and shardings inside jaxpr
  # should be the same.
  unique_intermediate_shardings = util.stable_unique(
      dispatch.get_intermediate_shardings(jaxpr))
  unique_const_shardings = util.stable_unique(in_shardings[:len(const_args)])
  unique_in_shardings = util.stable_unique(in_shardings[len(const_args):])
  unique_out_shardings = util.stable_unique(out_shardings)
  backend, device_assignment, num_devices = _get_and_check_device_assignment(
      it.chain(
          ((i, stages.MismatchType.ARG_SHARDING, None) for i in unique_in_shardings),
          ((c, stages.MismatchType.CONST_SHARDING, None) for c in unique_const_shardings),
          ((o, stages.MismatchType.OUT_SHARDING, None) for o in unique_out_shardings),
          ((js, stages.MismatchType.SHARDING_INSIDE_COMPUTATION, source_info)
           for js, source_info in unique_intermediate_shardings)),
      context_mesh)
  unique_intermediate_shardings = [js for js, _ in unique_intermediate_shardings]
  unique_in_shardings = unique_in_shardings | unique_const_shardings  # pyrefly: ignore[unsupported-operation]
  del unique_const_shardings

  prim_requires_devices = dispatch.jaxpr_has_prim_requiring_devices(jaxpr)

  if device_assignment is None:
    if lowering_platforms is None:
      raise ValueError(
          "Passing lowering_platforms via jax.export or"
          " jit(f).trace(*args).lower(lowering_platforms=...) is required when"
          " only AbstractMesh exists in a jitted computation. Got context"
          f" mesh: {context_mesh}")
    if prim_requires_devices:
      raise ValueError(
          "AbstractMesh cannot be used when jaxpr contains primitives that"
          " require devices to be present during lowering.")

  # For device_assignment == 1, this doesn't matter.
  if device_assignment is not None and len(device_assignment) > 1:
    rep_gs = GSPMDSharding.get_replicated(device_assignment)
    in_shardings = tuple(
        rep_gs if (isinstance(s, UnspecifiedValue) and
                   aval is not core.abstract_token and aval.ndim == 0)
        else s for s, aval in zip(in_shardings, global_in_avals))

  for a in global_out_avals:
    if (a is not core.abstract_token and not a.sharding.mesh.empty and
        a.sharding.mesh.are_all_axes_explicit and
        device_assignment is not None and
        len(device_assignment) != a.sharding.mesh.size):
      raise ValueError(
          f"Length of device assignment {len(device_assignment)} is not equal"
          f" to the size of the mesh {a.sharding.mesh.size} of aval"
          f" {a.str_short(True, True)}. Please enter your `jit` into a mesh"
          " context via `jax.set_mesh`.")

  # TODO(parkers): One _raw_platform has been unified with platform,
  # change this back to just read platform.
  platforms = lowering_platforms or (
      getattr(backend, "_raw_platform", backend.platform),)

  device_list = _create_device_list(device_assignment)
  transfer_mem_kind_in_jaxpr = jaxpr_transfer_mem_kinds(jaxpr)

  committed = bool(
      not context_mesh.empty
      or num_devices > 1
      or any(not isinstance(s, UnspecifiedValue) for s in it.chain(
          unique_in_shardings, unique_out_shardings,
          unique_intermediate_shardings))
      or transfer_mem_kind_in_jaxpr
  )

  all_default_mem_kind = are_all_shardings_default_mem_kind(
      it.chain(unique_in_shardings, unique_out_shardings,
               unique_intermediate_shardings, transfer_mem_kind_in_jaxpr))

  if all_default_mem_kind:
    propagated_out_mem_kinds = (None,) * len(global_out_avals)
  else:
    propagated_out_mem_kinds = tuple(
        core.mem_space_to_kind(o.memory_space) for o in closed_jaxpr.out_avals)

  out_shardings = _concretize_abstract_out_shardings(
      out_shardings, global_out_avals, device_assignment,
      propagated_out_mem_kinds)

  global_in_avals = [core.update_aval_with_sharding(a, sh)
                     if isinstance(a, core.ShapedArray) else a
                     for a, sh in zip(global_in_avals, in_shardings)]
  global_out_avals = [core.update_aval_with_sharding(a, sh)
                      if isinstance(a, core.ShapedArray) else a
                      for a, sh in zip(global_out_avals, out_shardings)]

  ############################ Build up the stableHLO ######################

  abstract_mesh = None
  if prim_requires_devices:
    assert device_list is not None
    for sharding in it.chain(unique_in_shardings, unique_out_shardings,
                             unique_intermediate_shardings):
      if isinstance(sharding, NamedSharding):
        if (abstract_mesh is not None and
            abstract_mesh != sharding.mesh.abstract_mesh):
          raise ValueError(
              "mesh should be the same across the entire program. Got mesh"
              f" shape for one sharding {abstract_mesh} and"
              f" {sharding.mesh.abstract_mesh} for another")
        abstract_mesh = sharding.mesh.abstract_mesh

  semantic_in_shardings = SemanticallyEqualShardings(
      in_shardings, global_in_avals)
  semantic_out_shardings = SemanticallyEqualShardings(
      out_shardings, global_out_avals)

  jaxpr_util.maybe_dump_jaxpr_to_file(fun_name, closed_jaxpr.jaxpr)
  module_name = util.wrap_name(api_name, fun_name)

  (module, keepalive, host_callbacks, unordered_effects, ordered_effects,
   tuple_args, shape_poly_state) = _cached_lowering_to_hlo(
       closed_jaxpr, module_name, backend,
       len(const_args), tuple(global_in_avals),
       semantic_in_shardings, semantic_out_shardings,
       in_layouts, out_layouts, num_devices,
       tuple(device_list) if prim_requires_devices else None,  # pyrefly: ignore[bad-argument-type]
       donated_invars, all_default_mem_kind, inout_aliases,
       propagated_out_mem_kinds, platforms,
       lowering_parameters=lowering_parameters,
       abstract_mesh=abstract_mesh)

  # backend and device_assignment is passed through to MeshExecutable because
  # if keep_unused=False and all in_shardings are pruned, then there is no way
  # to get the device_assignment and backend. So pass it to MeshExecutable
  # because we calculate the device_assignment and backend before in_shardings,
  # etc are pruned.
  return MeshComputation(
      module_name,
      module,
      const_args,
      donated_invars,
      platforms,
      compiler_options_kvs,
      device_list,
      global_in_avals=global_in_avals,
      global_out_avals=global_out_avals,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      tuple_args=tuple_args,
      auto_spmd_lowering=auto_spmd_lowering,
      unordered_effects=unordered_effects,
      ordered_effects=ordered_effects,
      host_callbacks=host_callbacks,
      keepalive=keepalive,
      kept_var_idx=kept_var_idx,
      mut=mut,
      backend=backend,
      num_devices=num_devices,
      committed=committed,
      in_layouts=in_layouts,
      out_layouts=out_layouts,
      shape_poly_state=shape_poly_state,
      all_args_info=all_args_info,
      pgle_profiler=pgle_profiler,
      intermediate_shardings=unique_intermediate_shardings,
      context_mesh=context_mesh)


def _to_logical_sharding(
    aval: core.AbstractValue, sharding: MaybeSharding | AUTO
) -> JSharding | AUTO | None:
  if isinstance(sharding, UnspecifiedValue):
    return None
  if isinstance(sharding, AUTO):
    return sharding
  elif isinstance(aval, (ShapedArray, AbstractRef)):
    assert isinstance(sharding, JSharding)
    return sharding
  elif isinstance(aval, core.AbstractToken):
    return None
  else:
    raise TypeError(aval)


class MeshComputation(stages.Lowering):
  _hlo: ir.Module
  _executable: MeshExecutable | None

  def __init__(self, name: str, hlo: ir.Module,
               const_args: list[ArrayLike],
               donated_invars: Sequence[bool], platforms: Sequence[str],
               compiler_options_kvs: tuple[tuple[str, Any], ...],
               device_assignment: xc.DeviceList | tuple[xc.Device, ...] | None,
               **compile_args):
    self._name = name
    self._hlo = hlo
    self.const_args = const_args
    self._donated_invars = donated_invars
    self._platforms = platforms
    self._compiler_options_kvs = compiler_options_kvs
    self._device_list = _create_device_list(device_assignment)
    self.compile_args = compile_args
    self._executable = None

  # -- stages.Lowering overrides

  def stablehlo(self) -> ir.Module:
    return self._hlo

  def compile(self, compiler_options=None, *, device_assignment=None,
              ) -> MeshExecutable:
    t_compiler_options = (() if compiler_options is None else
                          tuple(compiler_options.items()))
    compiler_options_kvs = self._compiler_options_kvs + t_compiler_options

    device_list = _create_device_list(device_assignment)
    if device_list is None:
      compilation_device_list = self._device_list
    else:
      if (self._device_list is not None and
          self._device_list != device_list):
        raise ValueError(
            "device_assignment passed to `.compile` must match the"
            " device_assignment calculated from array shardings and"
            " out_shardings. Got device ids passed to compile"
            f" {[d.id for d in device_list]} on platform"
            f" {device_list[0].platform.upper()} and devices ids"
            " calculated from array shardings and out_shardings"
            f" {[d.id for d in self._device_list]} on platform"
            f" {self._device_list[0].platform.upper()}")
      compilation_device_list = device_list
    assert isinstance(compilation_device_list, (type(None), xc.DeviceList))

    # Only cache executable into `self` if `.compile()` in AOT is called without
    # specifying compiler_options and device_assignment.
    use_cache = compiler_options is None and device_assignment is None
    if use_cache and self._executable is not None:
      return self._executable

    executable = UnloadedMeshExecutable.from_hlo(
        self._name, self._hlo, **self.compile_args,
        compiler_options_kvs=compiler_options_kvs,
        device_list=compilation_device_list)
    if use_cache:
      self._executable = executable
    return executable

  def cost_analysis(self) -> dict[str, float]:
    backend = self.compile_args["backend"]
    if xb.using_pjrt_c_api(backend):
      raise NotImplementedError(
          "Lowered.cost_analysis not implemented on platform "
          f"'{backend.platform}'. Use compile().cost_analysis() for "
          "post-compilation cost estimates.")
    return _jax.hlo_module_cost_analysis(backend, self.hlo().as_hlo_module())


def get_op_sharding_from_executable(
    executable) -> tuple[Sequence[xc.OpSharding], Sequence[xc.OpSharding]]:
  in_op_shardings: list[xc.OpSharding] = []
  parameter_shardings_from_xla = executable.get_parameter_shardings()
  if parameter_shardings_from_xla is not None:
    in_op_shardings = parameter_shardings_from_xla

  out_op_shardings: list[xc.OpSharding] = []
  output_shardings_from_xla = executable.get_output_shardings()
  if output_shardings_from_xla is not None:
    out_op_shardings = output_shardings_from_xla

  return in_op_shardings, out_op_shardings


def get_pspec_from_executable(
    executable, mesh: Mesh
) -> tuple[tuple[PartitionSpec, ...], tuple[PartitionSpec, ...]]:
  input_op_s, output_op_s = get_op_sharding_from_executable(executable)
  in_pspec: list[PartitionSpec] = []
  for s in input_op_s:
    in_pspec.extend(sharding_impls.parse_flatten_op_sharding(s, mesh))

  out_pspec: list[PartitionSpec] = []
  for s in output_op_s:
    out_pspec.extend(sharding_impls.parse_flatten_op_sharding(s, mesh))
  return tuple(in_pspec), tuple(out_pspec)


def get_out_shardings_from_executable(
    xla_executable,
    device_list: xc.DeviceList,
    num_out_avals: int,
    num_ordered_effects: int,
) -> Sequence[sharding_impls.GSPMDSharding] | None:
  assert isinstance(device_list, xc.DeviceList)

  try:
    omk = xla_executable.get_output_memory_kinds()[0]
    if num_ordered_effects > 0:
      omk = omk[num_ordered_effects:]
  except:
    omk = [None] * num_out_avals

  assert len(omk) == num_out_avals, (len(omk), num_out_avals)

  # When the device assignment only has 1 device, SPMD partitioner will not run.
  # Hence the op shardings will not be set on the `hlo_module`.
  if len(device_list) == 1:
    return [sharding_impls.GSPMDSharding.get_replicated(device_list, memory_kind=mk)
            for mk in omk]

  out_op_shardings: Sequence[xc.OpSharding]
  _, out_op_shardings = get_op_sharding_from_executable(xla_executable)
  if not out_op_shardings:
    return None

  if num_ordered_effects > 0:
    out_op_shardings = out_op_shardings[num_ordered_effects:]

  # This means that there are no outputs for JAX but for XLA there is an empty
  # tuple output which gets a replicated sharding.
  if num_out_avals == 0 and len(out_op_shardings) == 1:
    return None

  # This condition happens when all the elements in the output tuple have the
  # same sharding, so XLA decides to run the `FusionTupleDeduplicator` to
  # put the sharding on ROOT instead of the tuple.
  # TODO(b/245667823): Remove this when XLA fixes this.
  if len(out_op_shardings) == 1 and len(out_op_shardings) < num_out_avals:
    out_op_shardings = list(out_op_shardings) * num_out_avals

  assert len(out_op_shardings) == num_out_avals == len(omk), (
      len(out_op_shardings), num_out_avals, len(omk))

  return [sharding_impls.GSPMDSharding(device_list, os, memory_kind=mk)
          for os, mk in safe_zip(out_op_shardings, omk)]


def _get_in_shardings_from_xla(
    xla_executable, device_list: xc.DeviceList, num_in_avals: int,
    num_ordered_effects: int
  ) -> Sequence[GSPMDSharding] | None:
  """Returns input shardings from XLA."""
  # When the device assignment only has 1 device, SPMD partitioner will not run.
  # Hence the op shardings will not be set on the `hlo_module`.
  assert isinstance(device_list, xc.DeviceList)
  if len(device_list) == 1:
    return [GSPMDSharding.get_replicated(device_list)] * num_in_avals

  in_op_shardings, _ = get_op_sharding_from_executable(xla_executable)
  if not in_op_shardings:
    return None

  if num_ordered_effects > 0:
    in_op_shardings = in_op_shardings[num_ordered_effects:]

  assert len(in_op_shardings) == num_in_avals, (
      len(in_op_shardings), num_in_avals)

  return [GSPMDSharding(device_list, os) for os in in_op_shardings]


# TODO(yashkatariya): Remove this function after `AUTO` can return shardings
# without mesh.
def _get_mesh_pspec_shardings_from_executable(
    xla_executable, mesh: Mesh
) -> tuple[Sequence[NamedSharding], Sequence[NamedSharding]]:
  in_pspec, out_pspec = get_pspec_from_executable(xla_executable, mesh)
  return ([NamedSharding(mesh, i) for i in in_pspec],
          [NamedSharding(mesh, o) for o in out_pspec])


_orig_out_sharding_handlers: dict[Any, Any] = {}

def _gspmd_to_named_sharding(
    out_s: GSPMDSharding, out_aval, orig_in_s: NamedSharding) -> NamedSharding:
  assert isinstance(out_s, GSPMDSharding)
  assert isinstance(orig_in_s, NamedSharding)
  assert isinstance(orig_in_s.mesh, Mesh)
  if (out_aval is not None and not out_aval.sharding.mesh.empty and
      not out_aval.sharding.mesh._any_axis_manual):
    mesh = _abstract_to_concrete_mesh(
        out_aval.sharding.mesh, out_s._device_assignment)
  else:
    mesh = orig_in_s.mesh
  return sharding_impls._gspmd_to_named_sharding_via_mesh(out_s, mesh)
_orig_out_sharding_handlers[NamedSharding] = _gspmd_to_named_sharding

def _gspmd_to_single_device_sharding(
    out_s: GSPMDSharding, out_aval, orig_in_s: SingleDeviceSharding
    ) -> SingleDeviceSharding:
  assert isinstance(out_s, GSPMDSharding)
  assert isinstance(orig_in_s, SingleDeviceSharding)
  return SingleDeviceSharding(
      out_s._device_assignment[0], memory_kind=out_s.memory_kind)
_orig_out_sharding_handlers[SingleDeviceSharding] = _gspmd_to_single_device_sharding


def _get_out_sharding_from_orig_sharding(
    out_shardings, out_avals, orig_in_s, orig_aval):
  out: list[JSharding] = []
  orig_handler = _orig_out_sharding_handlers[type(orig_in_s)]
  for o, out_aval in safe_zip(out_shardings, out_avals):
    if (isinstance(o, sharding_impls.GSPMDSharding) and
        out_aval is not core.abstract_token):
      # TODO(yashkatariya): Remove this condition and ask users to drop into
      # explicit mode.
      if (orig_aval is not None and out_aval is not None
          and out_aval.ndim == orig_aval.ndim
          and isinstance(orig_in_s, NamedSharding)
          and out_aval.sharding.mesh == orig_in_s.mesh.abstract_mesh
          and o.is_equivalent_to(orig_in_s, orig_aval.ndim)):
        out.append(orig_in_s)
      else:
        try:
          out.append(orig_handler(o, out_aval, orig_in_s))
        except IndivisibleError:
          raise
        except:
          out.append(o)
    else:
      out.append(o)
  return out


def maybe_recover_user_shardings(
    old_shardings, new_shardings, old_avals, new_avals,
    intermediate_shardings=None, context_mesh: Mesh | None = None):
  if all(not isinstance(o, sharding_impls.GSPMDSharding) for o in new_shardings):
    return new_shardings

  for oi, o_aval in safe_zip(old_shardings, old_avals):
    if oi is not None and type(oi) in _orig_out_sharding_handlers:
      return _get_out_sharding_from_orig_sharding(
          new_shardings, new_avals, oi, o_aval)

  if intermediate_shardings is not None:
    for i in intermediate_shardings:
      if i is not None and type(i) in _orig_out_sharding_handlers:
        return _get_out_sharding_from_orig_sharding(
            new_shardings, [None] * len(new_shardings), i, None)

  # For nullary cases like: `jit(lambda: ..., out_shardings=(None, sharding))`
  for ns in new_shardings:
    if ns is not None and type(ns) in _orig_out_sharding_handlers:
      return _get_out_sharding_from_orig_sharding(
          new_shardings, new_avals, ns, None)

  if context_mesh is not None and not context_mesh.empty:
    return [sharding_impls._gspmd_to_named_sharding_via_mesh(n, context_mesh)
            if isinstance(n, GSPMDSharding) else n
            for n in new_shardings]

  return new_shardings

def is_user_xla_layout_equal(ul: Layout | AutoLayout,
                             xl: Layout) -> bool:
  if isinstance(ul, Layout) and not ul.tiling:
    return ul.major_to_minor == xl.major_to_minor
  else:
    return ul == xl


def _get_layouts_from_executable(
    xla_executable, in_layouts, out_layouts, num_ordered_effects
) -> tuple[Sequence[Layout | None], Sequence[Layout | None]]:
  try:
    in_layouts_xla = xla_executable.get_parameter_layouts()
    out_layouts_xla = xla_executable.get_output_layouts()
  except:
    return (None,) * len(in_layouts), (None,) * len(out_layouts)

  if num_ordered_effects > 0:
    in_layouts_xla = in_layouts_xla[num_ordered_effects:]
    out_layouts_xla = out_layouts_xla[num_ordered_effects:]

  new_in_layouts = []
  for x, l in safe_zip(in_layouts_xla, in_layouts):
    x = Layout.from_pjrt_layout(x)
    if isinstance(l, Layout) and not is_user_xla_layout_equal(l, x):
      raise AssertionError(
          f"Unexpected XLA layout override: (XLA) {x} != {l} "
          f"(User input layout)")
    # Always append the XLA layout because it has the full information
    # (tiling, etc) even if the user layout does not specify tiling.
    new_in_layouts.append(x)

  new_out_layouts = []
  for x, l in safe_zip(out_layouts_xla, out_layouts):
    x = Layout.from_pjrt_layout(x)
    if isinstance(l, Layout) and not is_user_xla_layout_equal(l, x):
      raise AssertionError(
          f"Unexpected XLA layout override: (XLA) {x} != {l} "
          f"(User output layout)")
    # Always append the XLA layout because it has the full information
    # (tiling, etc) even if the user layout does not specify tiling.
    new_out_layouts.append(x)

  assert all(isinstance(i, Layout) for i in new_in_layouts)
  assert all(isinstance(o, Layout) for o in new_out_layouts)
  return new_in_layouts, new_out_layouts


def get_logical_mesh_ids(mesh_shape):
  return np.arange(math.prod(mesh_shape)).reshape(mesh_shape)


def create_compile_options(
    computation, mesh, tuple_args, auto_spmd_lowering,
    allow_prop_to_inputs, allow_prop_to_outputs, backend,
    np_dev, compiler_options):
  num_replicas, num_partitions = 1, np_dev.size
  xla_device_assignment = np_dev.reshape((num_replicas, num_partitions))
  fdo_profile = compiler_options.pop("fdo_profile", None)
  compile_options = compiler.get_compile_options(
      num_replicas=num_replicas,
      num_partitions=num_partitions,
      device_assignment=xla_device_assignment,
      use_spmd_partitioning=True,
      use_auto_spmd_partitioning=auto_spmd_lowering,
      env_options_overrides=compiler_options,
      fdo_profile=fdo_profile,
      detailed_logging=compiler.use_detailed_logging(computation),
      backend=backend,
  )
  opts = compile_options.executable_build_options
  if auto_spmd_lowering:
    assert mesh is not None
    opts.auto_spmd_partitioning_mesh_shape = list(mesh.shape.values())
    opts.auto_spmd_partitioning_mesh_ids = (
        get_logical_mesh_ids(list(mesh.shape.values()))
        .reshape(-1))
  compile_options.parameter_is_tupled_arguments = tuple_args
  opts.allow_spmd_sharding_propagation_to_parameters = list(allow_prop_to_inputs)
  opts.allow_spmd_sharding_propagation_to_output = list(allow_prop_to_outputs)
  return compile_options


@weakref_lru_cache
def _cached_compilation(computation, name, mesh,
                        tuple_args, auto_spmd_lowering, allow_prop_to_inputs,
                        allow_prop_to_outputs, host_callbacks, backend,
                        da, compiler_options_kvs, pgle_profiler):
  # One would normally just write: dev = np.array(device_assignment)
  # The formulation below is substantially faster if there are many devices.
  dev = np.vectorize(lambda i: da[i], otypes=[object])(np.arange(len(da)))
  compiler_options = dict(compiler_options_kvs)

  compile_options = create_compile_options(
      computation, mesh, tuple_args, auto_spmd_lowering,
      allow_prop_to_inputs, allow_prop_to_outputs, backend,
      dev, compiler_options)

  with dispatch.log_elapsed_time(
      "Finished XLA compilation of {fun_name} in {elapsed_time:.9f} sec",
      fun_name=name, event=dispatch.BACKEND_COMPILE_EVENT):
    xla_executable = compiler.compile_or_get_cached(
        backend, computation, dev, compile_options, host_callbacks,
        da, pgle_profiler)
  return xla_executable


def _maybe_get_and_check_in_shardings(
    xla_executable, in_shardings, device_list, global_in_avals,
    num_ordered_effects):
  """Returns in_shardings extracted from XLA or checks and returns original
  shardings.

  If in_shardings exist on `jit` or on `jax.Array`, then this function will
  check that sharding against what XLA returns as in_shardings. If they don't
  match, an error is raised.

  If in_sharding is unspecified, then the sharding returned by XLA is returned.
  """
  in_shardings_xla = _get_in_shardings_from_xla(
      xla_executable, device_list, len(global_in_avals), num_ordered_effects)
  if in_shardings_xla is None:
    return in_shardings

  new_in_shardings = []
  for xla_s, orig, aval in safe_zip(in_shardings_xla, in_shardings,
                                    global_in_avals):
    if isinstance(orig, UnspecifiedValue):
      if (aval is not core.abstract_token and
          dtypes.issubdtype(aval.dtype, dtypes.extended)):
        xla_s = sharding_impls.logical_sharding(aval.shape, aval.dtype, xla_s)
      new_in_shardings.append(xla_s)
    else:
      xla_hlo_s = xla_s._to_xla_hlo_sharding(aval.ndim)
      orig_hlo_s = orig._to_xla_hlo_sharding(aval.ndim)
      # MANUAL HloSharding comes from other partitioning frameworks.
      if (not dtypes.issubdtype(aval.dtype, dtypes.extended) and
          not xla_hlo_s.is_manual() and
          (not op_shardings.are_hlo_shardings_equal(xla_hlo_s, orig_hlo_s))):
        raise AssertionError(
            f"Unexpected XLA sharding override: (XLA) {xla_s} != {orig} "
            "(User sharding)")
      new_in_shardings.append(orig)

  new_in_shardings = maybe_recover_user_shardings(
      in_shardings, new_in_shardings, global_in_avals, global_in_avals)

  return new_in_shardings


def _maybe_get_and_check_out_shardings(
    xla_executable, out_shardings, device_list, global_out_avals,
    num_ordered_effects
  ):
  out_shardings_xla = get_out_shardings_from_executable(
      xla_executable, device_list, len(global_out_avals),
      num_ordered_effects)
  if out_shardings_xla is None:
    return out_shardings

  new_out_shardings = []
  for xla_s, orig, aval in safe_zip(out_shardings_xla, out_shardings,
                                    global_out_avals):
    if isinstance(orig, UnspecifiedValue):
      if (aval is not core.abstract_token and
          dtypes.issubdtype(aval.dtype, dtypes.extended)):
        xla_s = sharding_impls.logical_sharding(aval.shape, aval.dtype, xla_s)
      new_out_shardings.append(xla_s)
    elif mlir.contains_unconstrained(orig):
      if (aval is not core.abstract_token and
          dtypes.issubdtype(aval.dtype, dtypes.extended)):
        xla_s = sharding_impls.logical_sharding(aval.shape, aval.dtype, xla_s)
      try:
        new_out_shardings.append(_gspmd_to_named_sharding(xla_s, aval, orig))  # pyrefly: ignore[bad-argument-type]
      except:
        new_out_shardings.append(xla_s)
    else:
      xla_hlo_s = xla_s._to_xla_hlo_sharding(aval.ndim)
      orig_hlo_s = orig._to_xla_hlo_sharding(aval.ndim)
      # MANUAL HloSharding comes from other partitioning frameworks.
      if (not dtypes.issubdtype(aval.dtype, dtypes.extended) and
          not xla_hlo_s.is_manual() and aval.size != 0 and
          (not op_shardings.are_hlo_shardings_equal(xla_hlo_s, orig_hlo_s) or
           xla_s.memory_kind != orig.memory_kind)):
        raise AssertionError(
            f"Unexpected XLA sharding override: (XLA) {xla_s} != {orig} "
            "(User sharding)")
      new_out_shardings.append(orig)
  return new_out_shardings


def finalize_shardings(shardings, device_assignment):
  if len(device_assignment) == 1:
    return [make_single_device_sharding(device_assignment[0], memory_kind=o.memory_kind)
            if isinstance(o, GSPMDSharding) else o for o in shardings]
  return shardings


def get_prop_to_input_output(in_shardings, out_shardings,
                             num_ordered_effects):
  allow_prop_to_inputs = (False,) * num_ordered_effects + tuple(
      isinstance(i, (UnspecifiedValue, AUTO)) for i in in_shardings)
  allow_prop_to_outputs = (False,) * num_ordered_effects + tuple(
      isinstance(o, (UnspecifiedValue, AUTO)) or mlir.contains_unconstrained(o)
      for o in out_shardings)
  return allow_prop_to_inputs, allow_prop_to_outputs


def maybe_concretize_mesh(sharding, da: xc.DeviceList):
  if (isinstance(sharding, NamedSharding) and
      isinstance(sharding.mesh, AbstractMesh)):
    if sharding.mesh.size != len(da):
      raise ValueError(
          f"The size of abstract mesh {sharding.mesh.size} in {sharding} must"
          f" match the length of device assignment: {len(da)}")
    return sharding.update(mesh=_abstract_to_concrete_mesh(sharding.mesh, da))
  return sharding


@dataclasses.dataclass
class UnloadedMeshExecutable:
  xla_executable: Any
  device_list: xc.DeviceList
  backend: xc.Client
  input_avals: Sequence[ShapedArray]
  input_shardings: Sequence[JSharding]
  output_avals: Sequence[ShapedArray]
  output_shardings: Sequence[JSharding]
  committed: bool
  name: str
  unordered_effects: list[core.Effect]
  ordered_effects: list[core.Effect]
  keepalive: Sequence[Any]
  host_callbacks: Sequence[Any]
  kept_var_idx: set[int]
  mut: MutationData | None
  auto_spmd_lowering: bool
  xla_in_layouts: Sequence[Layout | None]
  dispatch_in_layouts: Sequence[Layout | None]
  xla_out_layouts: Sequence[Layout | None]
  all_args_info: AllArgsInfo | None
  pgle_profiler: profiler.PGLEProfiler | None

  def build_unsafe_call(self):
    handle_args = InputsHandler(self.input_shardings, self.dispatch_in_layouts)
    handle_outs = global_avals_to_results_handler(
        self.output_avals, self.output_shardings, self.committed)

    unsafe_call = ExecuteReplicated(
        self.xla_executable, self.name, self.backend, handle_args,
        handle_outs, self.unordered_effects, self.ordered_effects, self.keepalive,
        bool(self.host_callbacks), self.kept_var_idx, self.mut,
        self.pgle_profiler)
    return unsafe_call

  def load(self) -> MeshExecutable:
    return MeshExecutable(self.xla_executable, self.build_unsafe_call,
                          self.input_avals, self.output_avals,
                          self.input_shardings, self.output_shardings,
                          self.auto_spmd_lowering, self.kept_var_idx,
                          self.xla_in_layouts, self.dispatch_in_layouts,
                          self.xla_out_layouts, self.mut, self.all_args_info,
                          self)

  @staticmethod
  def from_hlo(name: str,
               hlo: ir.Module,
               global_in_avals: Sequence[ShapedArray],
               global_out_avals: Sequence[ShapedArray],
               in_shardings: Sequence[JSharding | AUTO],
               out_shardings: Sequence[(JSharding | AUTO | UnspecifiedValue)],
               tuple_args: bool,
               auto_spmd_lowering: bool,
               unordered_effects: list[core.Effect],
               ordered_effects: list[core.Effect],
               host_callbacks: list[Any],
               keepalive: Any,
               kept_var_idx: set[int],
               backend: xc.Client,
               device_list: xc.DeviceList | None,
               committed: bool,
               in_layouts: MaybeLayout,
               out_layouts: MaybeLayout,
               compiler_options_kvs: tuple[tuple[str, Any], ...],
               num_devices: int,

               mut: MutationData | None = None,
               shape_poly_state: mlir.ShapePolyLoweringState | None = None,
               all_args_info: AllArgsInfo | None = None,
               pgle_profiler: profiler.PGLEProfiler | None = None,
               intermediate_shardings: Sequence[JSharding] | None = None,
               context_mesh: Mesh | None = None,
  ) -> MeshExecutable:
    del num_devices  # For compilation, we have an actual device_assignment
    if device_list is None:
      raise RuntimeError(
          "device_assignment cannot be `None` during compilation. Please pass a"
          " tuple of devices to `.compile(device_assignment=)`")

    assert isinstance(device_list, xc.DeviceList)
    in_shardings = tuple(maybe_concretize_mesh(i, device_list)
                         for i in in_shardings)
    out_shardings = tuple(maybe_concretize_mesh(o, device_list)
                          for o in out_shardings)

    if shape_poly_state is not None and shape_poly_state.uses_dim_vars:
      hlo = mlir.refine_polymorphic_shapes(hlo)

    allow_prop_to_inputs, allow_prop_to_outputs = get_prop_to_input_output(
        in_shardings, out_shardings, len(ordered_effects))

    mesh = None
    if auto_spmd_lowering:
      for i in it.chain(in_shardings, out_shardings):
        if isinstance(i, AUTO):
          mesh = i.mesh
          break

    util.test_event("pxla_cached_compilation")
    xla_executable = _cached_compilation(
        hlo, name, mesh, tuple_args, auto_spmd_lowering,
        allow_prop_to_inputs, allow_prop_to_outputs, tuple(host_callbacks),
        backend, device_list, compiler_options_kvs, pgle_profiler)

    if auto_spmd_lowering:
      assert mesh is not None
      in_shardings_xla, out_shardings_xla = _get_mesh_pspec_shardings_from_executable(
          xla_executable, mesh)
      in_shardings = [x if isinstance(i, AUTO) else i
                      for x, i in safe_zip(in_shardings_xla, in_shardings)]
      out_shardings = [x if isinstance(o, AUTO) else o
                       for x, o in safe_zip(out_shardings_xla, out_shardings)]
    else:
      assert mesh is None
      in_shardings = _maybe_get_and_check_in_shardings(
          xla_executable, in_shardings, device_list, global_in_avals,
          len(ordered_effects))
      out_shardings = _maybe_get_and_check_out_shardings(
          xla_executable, out_shardings, device_list, global_out_avals,
          len(ordered_effects))

    # xla_in_layouts are all either None or Layout. Even default
    # layout are concrete layouts and they are used in `compiled.input_formats`
    # to return concrete layouts to users.
    # `dispatch_in_layouts` replaces default layouts with `None` to simplify
    # dispatch logic downstream.
    xla_in_layouts, xla_out_layouts = _get_layouts_from_executable(
        xla_executable, in_layouts, out_layouts, len(ordered_effects))
    del in_layouts, out_layouts
    dispatch_in_layouts = [
        None if is_default_layout(l, s, a) else l
        for l, s, a, in safe_zip(xla_in_layouts, in_shardings, global_in_avals)
    ]

    out_shardings = maybe_recover_user_shardings(
        in_shardings, out_shardings, global_in_avals, global_out_avals,
        intermediate_shardings, context_mesh)

    in_shardings = finalize_shardings(in_shardings, device_list)
    out_shardings = finalize_shardings(out_shardings, device_list)

    return UnloadedMeshExecutable(
        xla_executable=xla_executable,
        device_list=device_list,
        backend=backend,
        input_avals=global_in_avals,
        input_shardings=in_shardings,
        output_avals=global_out_avals,
        output_shardings=out_shardings,
        committed=committed,
        name=name,
        unordered_effects=unordered_effects,
        ordered_effects=ordered_effects,
        keepalive=keepalive,
        host_callbacks=host_callbacks,
        kept_var_idx=kept_var_idx,
        mut=mut,
        auto_spmd_lowering=auto_spmd_lowering,
        xla_in_layouts=xla_in_layouts,
        dispatch_in_layouts=dispatch_in_layouts,
        xla_out_layouts=xla_out_layouts,
        all_args_info=all_args_info,
        pgle_profiler=pgle_profiler).load()


class MeshExecutableFastpathData(NamedTuple):
  xla_executable: xc.LoadedExecutable
  out_pytree_def: Any
  in_shardings: Sequence[JSharding]
  out_shardings: Sequence[JSharding]
  out_avals: Sequence[ShapedArray]
  out_committed: Sequence[bool]
  kept_var_bitvec: Iterable[bool]
  in_device_local_layouts: Sequence[Layout | None]
  const_args: Sequence[ArrayLike]


def clear_in_memory_compilation_cache() -> None:
  """Clears the in-memory compilation cache.

  This function clears all cached executables that were compiled by
  _cached_compilation function.
  """
  _cached_compilation.cache_clear()


@dataclasses.dataclass(frozen=True, kw_only=True)
class JitGlobalCppCacheKeys:
  donate_argnums: tuple[int, ...] | None = None
  donate_argnames: tuple[str, ...] | None = None
  device: xc.Device | None = None
  backend: str | None = None
  in_shardings_treedef: PyTreeDef | None = None
  in_shardings_leaves: tuple[Any, ...] | None = None
  out_shardings_treedef: PyTreeDef | None = None
  out_shardings_leaves: tuple[Any, ...] | None = None
  in_layouts_treedef: PyTreeDef | None = None
  in_layouts_leaves: tuple[Any, ...] | None = None
  out_layouts_treedef: PyTreeDef | None = None
  out_layouts_leaves: tuple[Any, ...] | None = None
  compiler_options_kvs: tuple[tuple[str, Any], ...] | None = None

  @functools.cached_property
  def contains_explicit_attributes(self):
    return (self.donate_argnums is not None or
            self.donate_argnames is not None or
            self.device is not None or
            self.backend is not None or
            any(not isinstance(i, UnspecifiedValue) for i in (self.in_shardings_leaves or [])) or
            any(not isinstance(o, UnspecifiedValue) for o in (self.out_shardings_leaves or [])) or
            any(i is not None for i in (self.in_layouts_leaves or [])) or
            any(o is not None for o in (self.out_layouts_leaves or [])) or
            self.compiler_options_kvs)


def reflatten_outputs_for_dispatch(out_tree, out_flat):
  # We arrive at dispatch having flattened according to the default
  # pytree registry, but we want to re-flatten according to our
  # dispatch-specific registry.
  out_unflat = tree_util.tree_unflatten(out_tree, out_flat)
  return tree_util.dispatch_registry.flatten(out_unflat, None)


class MeshExecutable(stages.Executable):
  __slots__ = [
      "xla_executable", "_unsafe_call", "build_unsafe_call", "in_avals",
      "out_avals", "_in_shardings", "_out_shardings", "_auto_spmd_lowering",
      "_kept_var_idx", "_xla_in_layouts", "_dispatch_in_layouts",
      "_xla_out_layouts", "_mut", "_all_args_info", "_unloaded_executable",
  ]

  def __init__(self, xla_executable, build_unsafe_call, in_avals, out_avals,
               in_shardings, out_shardings, auto_spmd_lowering, kept_var_idx,
               xla_in_layouts, dispatch_in_layouts, xla_out_layouts, mut,
               all_args_info: AllArgsInfo | None = None,
               unloaded_executable=None):
    self.xla_executable = xla_executable
    self.build_unsafe_call = build_unsafe_call
    # in_avals is a list of global and local avals. Aval is global if input
    # is a GDA or jax.Array else local.
    self.in_avals = in_avals  # has const_args, but not dead args
    self.out_avals = out_avals
    self._unsafe_call = None
    self._in_shardings = in_shardings  # has const_args, but not dead args
    self._out_shardings = out_shardings
    self._auto_spmd_lowering = auto_spmd_lowering
    # indices include const_args and dead args
    self._kept_var_idx = kept_var_idx
    self._xla_in_layouts = xla_in_layouts  # has const_args, but not dead args
    self._dispatch_in_layouts = dispatch_in_layouts
    self._xla_out_layouts = xla_out_layouts
    self._mut = mut
    self._all_args_info = all_args_info  # has const_args and also dead args
    self._unloaded_executable = unloaded_executable

  @property
  def unsafe_call(self) -> Callable[..., Any]:
    if self._unsafe_call is None:
      self._unsafe_call = self.build_unsafe_call()
    return self._unsafe_call

  # -- stages.Executable overrides

  def xla_extension_executable(self):
    return self.xla_executable

  def call(self, *args):
    args_after_dce = tuple(a for i, a in enumerate(args) if i in self._kept_var_idx)
    if (self._all_args_info is not None and
        self._all_args_info.debug_info.arg_names is not None):
      arg_names_after_dce = tuple(
          n for i, n in enumerate(self._all_args_info.debug_info.arg_names)
          if i in self._kept_var_idx)
    else:
      arg_names_after_dce = ("",) * len(args_after_dce)

    if self._all_args_info is not None:
      # We check all args before DCE
      check_arg_avals_for_call(self._all_args_info.in_avals,
                               map(core.shaped_abstractify, args),
                               self._all_args_info.debug_info)
    else:
      # We can only check the args after DCE
      check_arg_avals_for_call(self.in_avals,
                               map(core.shaped_abstractify, args_after_dce),
                               core.DebugInfo("MeshExecutable", "<unknown>",
                                              arg_names_after_dce, None))
    if not self._mut:
      check_array_xla_sharding_layout_match(
          args_after_dce, self._in_shardings, self._xla_in_layouts,
          arg_names_after_dce)
    else:
      args_after_dce = [*args_after_dce, *self._mut.in_mut]
      arg_names_after_dce += (("",) * len(self._mut.in_mut))
      check_array_xla_sharding_layout_match(
          args_after_dce, self._in_shardings, self._xla_in_layouts,
          arg_names_after_dce)
    return self.unsafe_call(*args)

  def create_cpp_call(self, params: stages.CompiledCallParams):
    if not (isinstance(self.unsafe_call, ExecuteReplicated) and
            not self.unsafe_call.has_unordered_effects and
            not self.unsafe_call.has_host_callbacks):
      return None

    def aot_cache_miss(*args, **kwargs):
      # args do not include the const args.
      # See https://docs.jax.dev/en/latest/internals/constants.html.
      outs, out_flat, args_flat = stages.Compiled.call(params, *args, **kwargs)

      if not params.is_high:
        out_flat, out_tree_dispatch = reflatten_outputs_for_dispatch(
            params.out_tree, out_flat)
        use_fastpath = (all(isinstance(x, xc.ArrayImpl) for x in out_flat)
                        and not self._mut)
      else:
        out_tree_dispatch = None
        use_fastpath = False

      if use_fastpath:
        out_avals = [o.aval for o in out_flat]
        out_committed = [o._committed for o in out_flat]
        kept_var_bitvec = [i in self._kept_var_idx
                           for i in range(len(params.const_args) + len(args_flat))]
        in_shardings = [
            sharding_impls.physical_sharding(a, s)
            if a is not core.abstract_token and dtypes.issubdtype(a.dtype, dtypes.extended)
            else s
            for s, a in zip(self._in_shardings, self.in_avals)
        ]
        fastpath_data = MeshExecutableFastpathData(
            self.xla_executable, out_tree_dispatch, in_shardings,
            self._out_shardings, out_avals, out_committed, kept_var_bitvec,
            self._dispatch_in_layouts, params.const_args)
      else:
        fastpath_data = None
      return outs, fastpath_data, False  # Do not remove cache entry

    return xc._xla.pjit(
        self.unsafe_call.name, None, aot_cache_miss, [], [],
        JitGlobalCppCacheKeys(), tree_util.dispatch_registry, cc_shard_arg)

def cc_shard_arg(x, sharding, layout):
  return shard_args([sharding], [layout], [xc.ArrayCopySemantics.REUSE_INPUT],
                    [x])[0]


def check_arg_avals_for_call(ref_avals, arg_avals,
                             jaxpr_debug_info: core.DebugInfo):
  if len(ref_avals) != len(arg_avals):
    raise TypeError(
        f"Computation compiled for {len(ref_avals)} inputs "
        f"but called with {len(arg_avals)}")

  arg_names = [f"'{name}'" for name in jaxpr_debug_info.safe_arg_names(len(ref_avals))]

  errors = []
  for ref_aval, arg_aval, name in safe_zip(ref_avals, arg_avals, arg_names):
    # Don't compare shardings of avals because you can lower with
    # numpy arrays + in_shardings and call compiled executable with
    # sharded arrays. We also have sharding checks downstream.
    if (ref_aval.shape, ref_aval.dtype) != (arg_aval.shape, arg_aval.dtype):
      errors.append(
          f"Argument {name} compiled with {ref_aval.str_short()} and called "
          f"with {arg_aval.str_short()}")
  if errors:
    max_num_errors = 5
    str_errors = "\n".join(errors[:max_num_errors])
    if len(errors) >= max_num_errors:
      num_mismatch_str = f"The first {max_num_errors} of {len(errors)}"
    else:
      num_mismatch_str = "The"
    raise TypeError(
        "Argument types differ from the types for which this computation was "
        "compiled. Perhaps you are calling the compiled executable with a "
        "different enable_x64 mode than when it was AOT compiled? "
        f"{num_mismatch_str} mismatches are:\n{str_errors}")


def check_device_backend_on_shardings(shardings) -> bool:
  for i in shardings:
    if isinstance(i, (UnspecifiedValue, AUTO)):
      continue
    if getattr(i, '_device_backend', False):
      return True
  return False


def check_array_xla_sharding_layout_match(
    args,
    in_shardings: Sequence[JSharding],
    in_layouts: Sequence[Layout],
    arg_names: Sequence[str]
) -> None:
  errors = []
  num_errors = 5
  for arg, xs, xl, name in zip(args, in_shardings, in_layouts, arg_names):
    if not isinstance(arg, array.ArrayImpl):
      continue
    if isinstance(xs, (UnspecifiedValue, AUTO)):
      continue

    db_xs = check_device_backend_on_shardings([xs])

    if (not db_xs and arg._committed and
        not arg.sharding.is_equivalent_to(xs, arg.ndim)):
      errors.append((
          f"Argument {name} with shape {arg.aval.str_short()}:\n"
          f"  Passed sharding: {arg.sharding}\n"
          f"  Required sharding: {xs}",
          "sharding"))

    if (not db_xs and arg._committed and
        arg.format.layout is not None and xl is not None and
        arg.format.layout != xl):
      errors.append((
          f"Argument {name} with shape {arg.aval.str_short()}:\n"
          f"  Passed layout: {arg.format.layout}\n"
          f"  Required layout: {xl}",
          "layout"))

  if errors:
    first_errors, error_kinds = unzip2(errors[:num_errors])
    str_errors = '\n'.join(first_errors)
    if all(k == 'sharding' for k in error_kinds):
      kind_str = r'shardings'
    elif all(k == 'layout' for k in error_kinds):
      kind_str = 'layouts'
    else:
      kind_str = 'shardings and layouts'
    num_mismatch_str = (
        f"the {len(errors)} mismatches" if len(errors) < num_errors else
        f"{num_errors} mismatches out of {len(errors)}")
    raise ValueError(
        f"Computation was compiled for input {kind_str} that disagree with the "
        f"{kind_str} of arguments passed to it. "
        f"Here are {num_mismatch_str}:\n{str_errors}")

def batch_spec(spec, dim, val):
  too_short = dim - len(spec)
  if too_short > 0:
    spec += (None,) * too_short
  new_partitions = tuple_insert(spec, dim, val)
  return PartitionSpec(*new_partitions)
