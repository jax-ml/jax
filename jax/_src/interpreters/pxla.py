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
from collections import namedtuple
from collections.abc import Callable, Sequence, Iterable
import dataclasses
from functools import partial, cached_property
import functools
import itertools as it
import logging
import math
from typing import Any, NamedTuple, Union, cast
import warnings

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
from jax._src import linear_util as lu
from jax._src import op_shardings
from jax._src import sharding_specs
from jax._src import pjit
from jax._src import profiler
from jax._src import sharding_impls
from jax._src import stages
from jax._src import tree_util
from jax._src import typing
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.abstract_arrays import array_types
from jax._src.core import DShapedArray
from jax._src.core import ShapedArray
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import mlir
from jax._src.layout import Layout, AutoLayout, Format
from jax._src.lib import xla_client as xc
from jax._src.lib import jaxlib_extension_version
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.partition_spec import PartitionSpec
from jax._src.sharding import Sharding as JSharding
from jax._src.mesh import (AbstractMesh, Mesh, get_abstract_mesh,
                           get_concrete_mesh)
from jax._src.sharding_impls import (
    ArrayMapping, ArrayMappingOrAutoOrUnspecified, AUTO, UnspecifiedValue,
    get_array_mapping as _get_array_mapping, array_mapping_to_axis_resources,
    SingleDeviceSharding, GSPMDSharding, NamedSharding,
    PartitionSpec as P)
from jax._src.util import (safe_map, safe_zip, partition_list, wrap_name,
                           tuple_update, tuple_delete, distributed_debug_log,
                           unzip2, HashableFunction, weakref_lru_cache,
                           tuple_insert)
from jax._src.state.types import AbstractRef, RefEffect
from jax._src.typing import ArrayLike


# Built in Python lists don't support weak refs but subclasses of lists do.
class WeakRefList(list):
  pass


xe = xc._xla

unsafe_map, map = map, safe_map  # type: ignore
zip, unsafe_zip = safe_zip, zip  # type: ignore

logger = logging.getLogger(__name__)

Index = Union[int, slice, tuple[Union[int, slice], ...]]
PyTreeDef = tree_util.PyTreeDef

NoSharding = sharding_specs.NoSharding
Chunked = sharding_specs.Chunked
Unstacked = sharding_specs.Unstacked

ShardedAxis = sharding_specs.ShardedAxis
Replicated = sharding_specs.Replicated

AvalDimSharding = Union[Unstacked, Chunked, NoSharding]
MeshAxisName = sharding_impls.MeshAxisName
MeshDimAssignment = Union[ShardedAxis, Replicated]
ShardingSpec = sharding_specs.ShardingSpec

### util


def to_xc_copy_semantics(copy_semantics):
  out = []
  for cs in copy_semantics:
    if cs is None or cs == dispatch.CopySemantics.ALIAS:
      out.append(xc.ArrayCopySemantics.REUSE_INPUT)
    elif cs == dispatch.CopySemantics.COPY:
      out.append(xc.ArrayCopySemantics.ALWAYS_COPY)
    elif cs == dispatch.CopySemantics.DONATE:
      out.append(xc.ArrayCopySemantics.DONATE_INPUT)
    else:
      assert isinstance(cs, xc.ArrayCopySemantics)
      out.append(cs)
  return out


def identity(x): return x

@profiler.annotate_function
def shard_args(shardings: Sequence[JSharding], layouts, copy_semantics,
               args, canonicalize=True) -> Sequence[xc.ArrayImpl]:
  xc_copy_semantics = to_xc_copy_semantics(copy_semantics)
  del copy_semantics
  # Fast path for one argument.
  if len(args) == 1:
    arg = args[0]
    if canonicalize:
      arg = dtypes.canonicalize_value(arg)
    return shard_arg_handlers[type(arg)]([arg], shardings, layouts,
                                         xc_copy_semantics)

  # type(arg) -> (list[indices], list[args], list[shardings], list[layouts],
  #               list[copy_semantics])
  batches = collections.defaultdict(lambda: ([], [], [], [], []))  # type: ignore
  for i, (arg, sharding, layout, cs) in enumerate(
      safe_zip(args, shardings, layouts, xc_copy_semantics)):
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
  for t, (indices, a, s, l, cs) in batches.items():
    outs = shard_arg_handlers[t](a, s, l, cs)
    for i, out in safe_zip(indices, outs):
      results[i] = out
  assert all(result is not None for result in results)
  return results


shard_arg_handlers: dict[
    Any, Callable[[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]],
                  Sequence[Any]]
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
  except xe.XlaRuntimeError as e:
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

def _shard_darray(xs, shardings, layouts, copy_semantics):
  return shard_args(shardings, layouts, copy_semantics, [x._data for x in xs])
shard_arg_handlers[core.DArray] = _shard_darray

def _shard_mutable_array(xs, shardings, layouts, copy_semantics):
  return shard_args(shardings, layouts, copy_semantics, [x._buf for x in xs])
shard_arg_handlers[core.MutableArray] = _shard_mutable_array

def batched_device_put(aval: core.ShapedArray,
                       sharding: JSharding, xs: Sequence[Any],
                       devices: Sequence[xc.Device], committed: bool = True):
  util.test_event("batched_device_put_start")
  try:
    bufs = [x for x, d in safe_zip(xs, devices)
            if (isinstance(x, array.ArrayImpl) and
                dispatch.is_single_device_sharding(x.sharding) and
                x.devices() == {d})]
    if len(bufs) == len(xs) > 0:
      return array.ArrayImpl(
          aval, sharding, bufs, committed=committed, _skip_checks=True)
    return xc.batched_device_put(aval, sharding, xs, list(devices), committed)
  finally:
    util.test_event("batched_device_put_end")

def _shard_aval(size, axis: int, aval):
  try:
    return _shard_aval_handlers[type(aval)](size, axis, aval)
  except KeyError as err:
    raise TypeError(f"No _shard_aval handler for type: {type(aval)}") from err
_shard_aval_handlers: dict[type[core.AbstractValue], Callable[[int, int, Any], Any]] = {}

def _shard_abstract_array(size, axis: int, x):
  try:
    if x.shape[axis] != size:
      raise ValueError(f"Axis size {size} does not match dimension {axis} of "
                       f"shape {x.shape}")
  except IndexError:
    raise ValueError(f"Cannot split a {x.dim}D value along axis {axis}") from None
  if config.pmap_no_rank_reduction.value:
    return x.update(shape=tuple_update(x.shape, axis, 1))
  else:
    return x.update(shape=tuple_delete(x.shape, axis))
_shard_aval_handlers[ShapedArray] = _shard_abstract_array


def local_aval_to_result_handler(
    aval: core.AbstractValue,
    sharding: JSharding,
    indices: tuple[Index, ...] | None,
) -> Callable[[list[xc.ArrayImpl]], Any]:
  """Returns a function for handling the raw buffers of a single output aval.

  Args:
    aval: The local output AbstractValue.
    sharding_spec: Indicates how the output is sharded across devices, or None
      for non-array avals.
    indices: The pre-computed result of spec_to_indices, or None for non-array
      avals.

  Returns:
    A function for handling the Buffers that will eventually be produced
    for this output. The function will return an object suitable for returning
    to the user, e.g. an Array.
  """
  try:
    return local_result_handlers[(type(aval))](aval, sharding, indices)
  except KeyError as err:
    raise TypeError(
        f"No pxla_result_handler for type: {type(aval)}") from err

PxlaResultHandler = Callable[..., Callable[[Any], Any]]
local_result_handlers: dict[type[core.AbstractValue], PxlaResultHandler] = {}


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

global_result_handlers: dict[type[core.AbstractValue], PxlaResultHandler] = {}

### lazy device-memory persistence and result handling

### the xla_pmap primitive and its rules are comparable to xla_call in xla.py


def xla_pmap_impl_lazy(
    fun: lu.WrappedFun,
    *args,
    backend: str | None,
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: int,
    devices: Sequence[Any] | None,
    name: str,
    in_axes: Sequence[int | None],
    out_axes_thunk: Callable[[], Sequence[int | None]],
    donated_invars: Sequence[bool],
    is_explicit_global_axis_size: bool,
) -> tuple[Callable, list[ArrayLike]]:
  if (config.disable_jit.value and
      not is_explicit_global_axis_size and not any(donated_invars)):
    def _emap_apply_fn(*args):
      return _emap_impl(fun, *args, backend=backend, axis_name=axis_name,
                        axis_size=axis_size, global_axis_size=global_axis_size,
                        devices=devices, name=name, in_axes=in_axes,
                        out_axes_thunk=out_axes_thunk,
                        donated_invars=donated_invars,
                        is_explicit_global_axis_size=is_explicit_global_axis_size)
    return _emap_apply_fn, []
  abstract_args = unsafe_map(core.abstractify, args)
  compiled_fun, fingerprint, const_args = parallel_callable(
      fun, backend, axis_name, axis_size, global_axis_size, devices, name,
      in_axes, out_axes_thunk, donated_invars,
      is_explicit_global_axis_size, *abstract_args)

  # Don't re-abstractify args unless logging is enabled for performance.
  if config.distributed_debug.value:
    distributed_debug_log(("Running pmapped function", name),
                          ("python function", fun.f),
                          ("devices", devices),
                          ("abstract args", map(core.abstractify, args)),
                          ("fingerprint", fingerprint))
  return compiled_fun, const_args

def xla_pmap_impl(fun: lu.WrappedFun, *args, **params):
  compiled_fun, const_args = xla_pmap_impl_lazy(fun, *args, **params)
  return compiled_fun(*const_args, *args)

class EmapInfo(NamedTuple):
  backend: str | None
  devices: Sequence[Any] | None

def _emap_impl(fun: lu.WrappedFun, *args,
               backend: str | None,
               axis_name: core.AxisName,
               axis_size: int,
               global_axis_size: int,
               devices: Sequence[Any] | None,
               name: str,
               in_axes: Sequence[int | None],
               out_axes_thunk: Callable[[], Sequence[int | None]],
               donated_invars: Sequence[bool],
               is_explicit_global_axis_size: bool,
               ):
  # TODO(sharadmv,mattjj): implement these cases
  if any(d for d in donated_invars):
    raise NotImplementedError("Buffer donation not supported in eager pmap.")
  if is_explicit_global_axis_size:
    raise NotImplementedError("Non-default global_axis_size not supported in "
                              "eager pmap.")

  emap_info = EmapInfo(backend, devices)
  shard_axes = [{} if in_axis is None else {axis_name: in_axis} for in_axis in in_axes]
  trace = MapTrace(axis_name, emap_info)
  with core.extend_axis_env_nd([(axis_name, axis_size)]):
    tracers = [MapTracer(trace, arg, s) for arg, s in zip(args, shard_axes)]
    with core.set_current_trace(trace):
      ans = fun.call_wrapped(*tracers)

    out_tracers = map(trace.to_map_tracer, ans)
    outvals, out_axes_src = unzip2((t.val, t.shard_axes) for t in out_tracers)

  out_axes = out_axes_thunk()

  platform = xb.get_backend(backend).platform
  donate_argnums = (1,) if platform in {"cuda", "rocm", "tpu"} else ()
  new_outvals = []
  for out_axis_src, out_axis, outval in zip(out_axes_src, out_axes, outvals):
    with api.disable_jit(False):
      donate_argnums_ = donate_argnums
      if isinstance(outval, array.ArrayImpl):
        # We don't want to donate if it's already sharded.
        donate_argnums_ = ()
      out = api.pmap(
          lambda _, x: x,
          in_axes=(0, out_axis_src.get(axis_name)),
          out_axes=out_axis,
          devices=(None if devices is None else list(devices)),
          backend=backend,
          donate_argnums=donate_argnums_)(np.arange(axis_size), outval)
      new_outvals.append(out)
  return new_outvals

def _map_schedule(idx: tuple[int | None, ...]) -> tuple[int | None, ...]:
  # In order to do a multi-map (a simultaneous map over several axes), we will
  # nest several maps. Each time we do a map, we "remove" an input axis so we
  # need to update the remaining map axes. For example, if we are to map over
  # the axes 0, 3, and 4, we make three calls to pmap with in_axes as 0, 2, 2.
  return tuple(None if i is None else
               i - sum(j is not None and j < i for j in idx[:l])
               for l, i in enumerate(idx))


# We're often creating `f`s on the fly and we try to carefully make them have
# the right __hash__ and __eq__. However, despite our attempts pmap's caching
# still ends up not working, because it has a separate cache per
# _function object_. Adding this annotation here lets us reuse the same pmap
# callable for all equivalent primitive pmaps.
@util.cache(max_size=None, trace_context_in_key=False)
def _multi_pmap(f: Callable, info: EmapInfo, names: list[core.AxisName],
                all_axes: list[tuple[int | None, ...]]
                ) -> tuple[Callable, dict[core.AxisName, int]]:
  used_names = []
  for i, name in reversed(list(enumerate(names))):
    in_axes = tuple(arg_axis[i] for arg_axis in all_axes)
    if any(in_axis is not None for in_axis in in_axes):
      f = api.pmap(
          f,
          in_axes=in_axes,
          axis_name=name,
          out_axes=0,
          backend=info.backend,
          devices=(None if info.devices is None else list(info.devices)))
      used_names.append(name)
  out_shard_axes = {name: i for i, name in enumerate(reversed(used_names))}
  return f, out_shard_axes

FakePrimitive = namedtuple("FakePrimitive", ["multiple_results", "bind"])

class MapTrace(core.Trace):
  __slots__ = ("axis_name", "emap_info")

  def __init__(self, axis_name, emap_info):
    super().__init__()
    self.emap_info = emap_info
    self.axis_name = axis_name

  def to_map_tracer(self, val):
    if isinstance(val, MapTracer):
      return val
    else:
      return MapTracer(self, val, {})

  def process_primitive(self, primitive, tracers, params):
    from jax._src.lax import parallel  # pytype: disable=import-error
    if primitive is parallel.axis_index_p:
      return self.process_axis_index(**params)  # pytype: disable=missing-parameter
    if primitive is parallel.psum_p:
      f = HashableFunction(
          lambda *xs: parallel.psum(
            xs, axis_name=params['axes'], axis_index_groups=params['axis_index_groups']),
          (primitive, tuple(params.items())))
    else:
      f = HashableFunction(lambda *args: primitive.bind(*args, **params),
                           (primitive, tuple(params.items())))
    tracers = map(self.to_map_tracer, tracers)
    vals, shard_axes = unzip2([(t.val, t.shard_axes) for t in tracers])
    info = self.emap_info
    names = core.get_axis_env().axis_names()
    all_axes = tuple(_map_schedule(map(s.get, names)) for s in shard_axes)  # pytype: disable=wrong-arg-types  # always-use-return-annotations
    f_mapped, out_shard_axes = _multi_pmap(f, self.emap_info, names, all_axes)
    with core.eval_context(), api.disable_jit(False):
      outvals = f_mapped(*vals)
    if primitive.multiple_results:
      return [MapTracer(self, val, out_shard_axes) for val in outvals]
    return MapTracer(self, outvals, out_shard_axes)

  def process_call(self, call_primitive, fun, tracers, params):
    raise NotImplementedError

  def process_map(self, map_primitive, fun, tracers, params):
    if params['devices'] is not None:
      raise ValueError("Nested pmap with explicit devices argument.")
    if not config.disable_jit.value:
      bind = HashableFunction(
          lambda *args, **kwargs: map_primitive.bind(fun, *args, **kwargs),
          (map_primitive, fun))
      fake_primitive = FakePrimitive(multiple_results=True, bind=bind)
      return self.process_primitive(fake_primitive, tracers, params)
    axis_name, in_axes, out_axes_thunk, axis_size = (params["axis_name"],
        params["in_axes"], params["out_axes_thunk"], params["axis_size"])
    vals, shard_axes = unzip2((t.val, t.shard_axes) for t in tracers)
    shard_axes = [{axis_name: _annot_to_flat(np.ndim(v), s.values(), ax), **s}
                  if ax is not None else s
                  for v, ax, s in zip(vals, in_axes, shard_axes)]
    in_tracers = map(partial(MapTracer, self), vals, shard_axes)
    with core.extend_axis_env_nd([(axis_name, axis_size)]):
      with core.set_current_trace(self):
        ans = fun.call_wrapped(*in_tracers)
      out_tracers = map(self.to_map_tracer, ans)
      out, outaxes = unzip2((t.val, t.shard_axes) for t in out_tracers)
    out, outaxes = unzip2(_match_annot(axis_name, axis_size, v, s, dst)
                           for v, s, dst in zip(out, outaxes, out_axes_thunk()))
    return map(partial(MapTracer, self), out, outaxes)

  def process_custom_jvp_call(self, prim, fun, jvp, tracers, *, symbolic_zeros):
    if symbolic_zeros:
      msg = ("custom_jvp with symbolic_zeros=True not supported with eager pmap. "
             "Please open an issue at https://github.com/jax-ml/jax/issues !")
      raise NotImplementedError(msg)
    del prim, jvp, symbolic_zeros  # always base main, can drop jvp
    with core.set_current_trace(self):
      return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers,
                              out_trees, symbolic_zeros):
    if symbolic_zeros:
      msg = ("custom_vjp with symbolic_zeros=True not supported with eager pmap. "
             "Please open an issue at https://github.com/jax-ml/jax/issues !")
      raise NotImplementedError(msg)
    del primitive, fwd, bwd, out_trees, symbolic_zeros  # always base main, drop vjp
    with core.set_current_trace(self):
      return fun.call_wrapped(*tracers)

  def process_axis_index(self, axis_name):
    from jax._src.lax import lax, parallel  # pytype: disable=import-error
    bind = HashableFunction(
        lambda _: parallel.axis_index(axis_name),
        (parallel.axis_index, axis_name))
    fake_primitive = FakePrimitive(multiple_results=False, bind=bind)
    range = lax.iota(np.int32, core.get_axis_env().axis_size(axis_name))
    dummy_tracer = MapTracer(self, range, {axis_name: 0})
    return self.process_primitive(fake_primitive, (dummy_tracer,), {})

def _annot_to_flat(ndim: int, mapped_axes: Iterable[int],
                 annotation: int | None) -> int | None:
  if annotation is None: return None
  mapped_axes_ = set(mapped_axes)
  return [i for i in range(ndim) if i not in mapped_axes_][annotation]

def _match_annot(axis_name: core.AxisName, axis_size: int, val: Any,
                 shard_axis_src: dict[core.AxisName, int],
                 dst_annotation: int | None
                 ) -> tuple[Any, dict[core.AxisName, int]]:
  shard_axis_out = dict(shard_axis_src)
  src = shard_axis_out.pop(axis_name, None)
  dst = _annot_to_flat(np.ndim(val) + (src is None), shard_axis_out.values(),
                       dst_annotation)
  with core.eval_context():
    if src == dst:
      outval = val
    elif type(src) == type(dst) == int:
      outval = batching.moveaxis(val, src, dst)
      shard_axis_out = _moveaxis(np.ndim(val), shard_axis_src, src, dst)
    elif src is None and dst is not None:
      outval = batching.broadcast(val, axis_size, dst)
      shard_axis_out = {n: d + (dst <= d) for n, d in shard_axis_out.items()}
    else:
      raise NotImplementedError
  return outval, shard_axis_out

def _moveaxis(ndim: int, shard_axes: dict[core.AxisName, int],
              src: int, dst: int) -> dict[core.AxisName, int]:
  lst: list[core.AxisName | None] = [None] * ndim
  for k, v in shard_axes.items():
    lst[v] = k
  name = lst.pop(src)
  lst.insert(dst - (src < dst), name)
  return {name: i for i, name in enumerate(lst) if name is not None}

class MapTracer(core.Tracer):
  __slots__ = ["val", "shard_axes"]

  def __init__(self, trace: MapTrace, val, shard_axes: dict[core.AxisName, int]):
    self._trace = trace
    self.val = val
    self.shard_axes = shard_axes
    assert all(val < self.val.ndim for val in self.shard_axes.values())

  @property
  def aval(self):
    aval = core.abstractify(self.val)
    shard_axes = dict(self.shard_axes)
    for axis_idx in sorted(shard_axes.values())[::-1]:
      aval = core.mapped_aval(aval.shape[axis_idx], axis_idx, aval)
    return aval

  def full_lower(self):
    return self

  def __str__(self):
    named_axes = [f"{k}={v}" for k, v in self.shard_axes.items()]
    return f"{self.val}{{{','.join(named_axes)}}}"

@lu.cache
def parallel_callable(fun: lu.WrappedFun,
                      backend_name: str | None,
                      axis_name: core.AxisName,
                      axis_size: int,
                      global_axis_size: int,
                      devices: Sequence[Any] | None,
                      name: str,
                      in_axes: Sequence[int | None],
                      out_axes_thunk: Callable[[], Sequence[int | None]],
                      donated_invars: Sequence[bool],
                      is_explicit_global_axis_size: bool,
                      *avals):
  closed_jaxpr, xc_backend, replicas, shards, pci = get_pmap_jaxpr(
      fun, backend_name, axis_name,
      axis_size=axis_size, global_axis_size=global_axis_size,
      devices=devices, name=fun.__name__, in_axes=in_axes,
      out_axes_thunk=out_axes_thunk, avals=avals)
  pmap_computation = lower_parallel_callable(
      fun, axis_name, axis_size, global_axis_size, devices, name,
      in_axes, donated_invars,
      is_explicit_global_axis_size, avals,
      lowering_platforms=None, lowering_parameters=mlir.LoweringParameters(),
      closed_jaxpr=closed_jaxpr, backend=xc_backend, replicas=replicas,
      shards=shards, pci=pci)
  pmap_executable = pmap_computation.compile()
  return WeakRefList([pmap_executable.unsafe_call, pmap_executable.fingerprint,
                      pmap_computation.const_args])


@dataclasses.dataclass(frozen=True)
class ParallelCallableInfo:
  name: str
  backend: xc.Client
  axis_name: core.AxisName
  axis_size: int
  global_axis_size: int
  devices: Sequence[xc.Device] | None
  in_axes: Iterable[int | None]
  out_axes_thunk: Callable[[], Sequence[int | None]]
  avals: Sequence[core.AbstractValue]

  @cached_property
  def local_devices(self):
    if self.devices:
      out = [d for d in self.devices
             if d.process_index == xb.process_index(self.backend)]
      assert len(out) > 0
    else:
      out = None
    return out

  @cached_property
  def out_axes(self):
    return self.out_axes_thunk()


class ShardInfo(NamedTuple):
  sharded_avals: Sequence[core.AbstractValue]
  out_sharded_avals: Sequence[core.ShapedArray]
  global_sharded_avals: Sequence[core.AbstractValue]
  num_local_shards: int
  num_global_shards: int


class ReplicaInfo(NamedTuple):
  jaxpr_replicas: int
  num_local_replicas: int
  num_global_replicas: int



_initial_style_primitives: set[core.Primitive] = set()


def register_initial_style_primitive(prim: core.Primitive):
  _initial_style_primitives.add(prim)

def _jaxpr_replicas(jaxpr: core.Jaxpr) -> int:
  """The number of replicas needed for a jaxpr.

  For a eqn, multiply the `axis_size` with the `jaxpr_replicas` of the
  subjaxprs. For a list of eqns, take the maximum number of replicas.
  """
  return max(unsafe_map(_eqn_replicas, jaxpr.eqns), default=1)

# TODO(mattjj): this function assumes that only pmap has a parameter named
# axis_size, and that it corresponds to cross-replica mapping
def _eqn_replicas(eqn: core.JaxprEqn) -> int:
  call_jaxpr = eqn.params.get("call_jaxpr")
  if call_jaxpr:
    return eqn.params.get('axis_size', 1) * _jaxpr_replicas(call_jaxpr)
  elif eqn.primitive in _initial_style_primitives:
    return _initial_style_primitive_replicas(eqn.params)
  else:
    return 1

def _initial_style_primitive_replicas(params: dict[str, Any]) -> int:
  return max(core.traverse_jaxpr_params(_jaxpr_replicas, params).values(),
             default=1)


def find_replicas(
    jaxpr: core.Jaxpr, axis_size: int, global_axis_size: int
) -> ReplicaInfo:
  # TODO(skyewm): replace this with a chain of pmaps and/or sharded_jits
  jaxpr_replicas = _jaxpr_replicas(jaxpr)
  num_local_replicas = axis_size * jaxpr_replicas
  num_global_replicas = global_axis_size * jaxpr_replicas
  return ReplicaInfo(jaxpr_replicas, num_local_replicas, num_global_replicas)

@lu.transformation2
def _change_argument_ranks(f, in_axes, out_axes_thunk, *args):
  from jax._src.lax import lax  # pytype: disable=import-error
  args = tuple(
      arg if in_axis is None else lax.squeeze(arg, dimensions=(in_axis,))
      for in_axis, arg in zip(in_axes, args)
  )
  results = f(*args)
  out_axes = out_axes_thunk()
  return tuple(
      x if axis is None else lax.expand_dims(x, dimensions=(axis,))
      for x, axis in zip(results, out_axes)
  )


def stage_parallel_callable(
    pci: ParallelCallableInfo, fun: lu.WrappedFun
) -> tuple[core.Jaxpr, list[Any], ReplicaInfo, ShardInfo]:
  sharded_avals = tuple(
      _shard_aval(pci.axis_size, axis, aval) if axis is not None else aval
      for axis, aval in safe_zip(pci.in_axes, pci.avals))

  orig_fun = fun
  if config.pmap_no_rank_reduction.value:
    fun = _change_argument_ranks(fun, pci.in_axes, pci.out_axes_thunk)
  else:
    fun = orig_fun
  with core.extend_axis_env_nd([(pci.axis_name, pci.global_axis_size)]):
    with dispatch.log_elapsed_time(
        "Finished tracing + transforming {fun_name} for pmap in {elapsed_time} sec",
        fun_name=fun.__name__, event=dispatch.JAXPR_TRACE_EVENT):
      jaxpr, out_sharded_avals, consts = pe.trace_to_jaxpr_dynamic(fun, sharded_avals)

  assert len(out_sharded_avals) == len(pci.out_axes), (
      len(out_sharded_avals), len(pci.out_axes))

  replicas = find_replicas(jaxpr, pci.axis_size, pci.global_axis_size)
  num_local_shards = replicas.num_local_replicas
  num_global_shards = replicas.num_global_replicas

  shards = ShardInfo(
      sharded_avals, out_sharded_avals, sharded_avals,
      num_local_shards, num_global_shards)

  return jaxpr, consts, replicas, shards


def get_pmap_jaxpr(
    fun: lu.WrappedFun,
    backend_name: str | None,
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: int,
    devices: Sequence[xc.Device] | None,
    name: str,
    in_axes: Iterable[int | None],
    out_axes_thunk: Callable[[], Sequence[int | None]],
    avals: Sequence[core.AbstractValue]):
  if devices is not None and backend_name is None:
    backend = xb.get_device_backend(devices[0])
  else:
    backend = xb.get_backend(backend_name)

  pci = ParallelCallableInfo(
      name, backend, axis_name, axis_size, global_axis_size, devices,
      in_axes, out_axes_thunk, avals)
  with core.extend_axis_env_nd([(axis_name, axis_size)]):
    jaxpr, consts, replicas, shards = stage_parallel_callable(pci, fun)
  jaxpr = core.remove_named_axis_effects(jaxpr, {axis_name})
  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  return closed_jaxpr, backend, replicas, shards, pci


@profiler.annotate_function
def lower_parallel_callable(
    fun: lu.WrappedFun,
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: int,
    devices: Sequence[xc.Device] | None,
    name: str,
    in_axes: Iterable[int | None],
    donated_invars: Sequence[bool],
    is_explicit_global_axis_size: bool,
    avals: Sequence[core.AbstractValue],
    *,
    lowering_platforms: tuple[str, ...] | None,
    lowering_parameters: mlir.LoweringParameters,
    closed_jaxpr: core.ClosedJaxpr,
    backend: xc.Client,
    replicas: ReplicaInfo,
    shards: ShardInfo,
    pci: ParallelCallableInfo) -> PmapComputation:
  # Determine global_axis_size for use in AxisEnv.
  # TODO(mattjj,skyewm): revive this check (inner_pmap always False now)
  # if xb.process_count() > 1 and global_axis_size is None and inner_pmap:
  #   raise ValueError("'axis_size' must be specified for nested multi-host pmaps")
  if (xb.process_count() == 1 and is_explicit_global_axis_size
      and global_axis_size != axis_size):
    raise ValueError(
        f"Specified axis_size {global_axis_size} doesn't match received "
        f"axis_size {axis_size}.")

  jaxpr = closed_jaxpr.jaxpr
  arg_names = jaxpr._debug_info.safe_arg_names(len(closed_jaxpr.in_avals))
  if lowering_parameters.hoist_constants_as_args:
    const_args = core.jaxpr_const_args(jaxpr)
    num_const_args = len(const_args)
    in_axes = (None,) * num_const_args + in_axes  # type: ignore
    donated_invars = (False,) * num_const_args + donated_invars  # type: ignore
    const_arg_avals = [core.shaped_abstractify(c) for c in const_args]
    jaxpr_avals = const_arg_avals + closed_jaxpr.in_avals  # type: ignore
    shards = ShardInfo(
        tuple(const_arg_avals) + shards.sharded_avals,  # type: ignore
        shards.out_sharded_avals,
        tuple(const_arg_avals) + shards.global_sharded_avals,  # type: ignore
        shards.num_local_shards, shards.num_global_shards)
    pci = dataclasses.replace(pci, in_axes=in_axes,
                              avals=tuple(const_arg_avals) + tuple(pci.avals))
    arg_names = ("",) * num_const_args + arg_names
  else:
    jaxpr_avals = closed_jaxpr.in_avals
    const_args = []
    num_const_args = 0

  no_nested_sharding = False
  must_run_on_all_devices = False
  if not is_explicit_global_axis_size:
    if xb.process_count(backend) > 1:
      if devices:
        # This allows each host in a multi-host pmap to run on a different number
        # of devices, but precludes nested sharding (i.e. inner pmaps).
        no_nested_sharding = True
      else:
        # This assumes all hosts run on the same number of devices. We make sure
        # this assumption is true by requiring that the pmap is run on all devices
        # (and making the further assumption that each host has the same number of
        # devices). Nested sharding is ok in this case.
        must_run_on_all_devices = True

  if logger.isEnabledFor(logging.DEBUG):
    logger.debug("sharded_avals: %s", shards.sharded_avals)
    logger.debug("global_sharded_avals: %s", shards.global_sharded_avals)
    logger.debug("num_replicas: %d  num_local_replicas: %d",
                 replicas.num_global_replicas, replicas.num_local_replicas)
    logger.debug("devices: %s", devices)
    logger.debug("local_devices: %s", pci.local_devices)

  if (xb.process_count(backend) > 1 and must_run_on_all_devices and
      shards.num_local_shards != xb.local_device_count(backend)):
    if shards.num_local_shards == axis_size:
      raise ValueError(
         f"On multi-host platforms, the input to pmapped functions must have "
         f"leading axis size equal to the number of local devices if no "
         f"`devices` argument is specified. Got {axis_size=}, "
         f"num_local_devices={xb.local_device_count(backend)}")
    else:
      raise ValueError(
        f"On multi-host platforms, pmapped functions must run across all "
        f"devices, i.e. num_replicas * num_partitions should equal the "
        f"number of local devices. Got "
        f"num_replicas={replicas.num_local_replicas}, and "
        f"num_local_devices={xb.local_device_count(backend)}")

  if no_nested_sharding and replicas.jaxpr_replicas > 1:
    raise ValueError(
      f"On multi-host platforms, pmapped functions that both have `devices` "
      f"specified and contain an inner_pmap must specify an "
      f"`axis_size` (or remove the `devices` argument). Got nested_replicas="
      f"{replicas.jaxpr_replicas}")

  log_priority = logging.WARNING if config.log_compiles.value else logging.DEBUG
  if logger.isEnabledFor(log_priority):
    logger.log(log_priority,
               "Compiling %s (%d) for %d devices with args %s. (num_replicas=%d)",
               fun.__name__, id(fun),
               shards.num_global_shards, avals, replicas.num_global_replicas)

  axis_env = sharding_impls.AxisEnv(
      replicas.num_global_replicas, (axis_name,), (global_axis_size,))
  replicated_args = [axis is None for axis in in_axes]
  tuple_args = dispatch.should_tuple_args(len(shards.global_sharded_avals),
                                          backend.platform)
  module_name = wrap_name('pmap', name)
  platforms = lowering_platforms or (backend.platform,)
  with core.extend_axis_env_nd([(axis_name, global_axis_size)]):
    ordered_effects = list(
        effects.ordered_effects.filter_in(closed_jaxpr.effects))
    if ordered_effects:
      raise ValueError("Ordered effects not supported in `pmap`.")
    unordered_effects = list(
        effects.ordered_effects.filter_not_in(closed_jaxpr.effects))
    with dispatch.log_elapsed_time(
        "Finished jaxpr to MLIR module conversion {fun_name} in {elapsed_time:.9f} sec",
        fun_name=module_name, event=dispatch.JAXPR_TO_MLIR_MODULE_EVENT):
      lowering_result = mlir.lower_jaxpr_to_module(
          module_name,
          closed_jaxpr,
          num_const_args=num_const_args,
          in_avals=jaxpr_avals,
          ordered_effects=ordered_effects,
          backend=backend,
          platforms=platforms,
          axis_context=sharding_impls.ReplicaAxisContext(axis_env),
          donated_args=donated_invars,
          replicated_args=replicated_args,
          arg_shardings=None,
          result_shardings=None,
          arg_names=arg_names,
          result_names=jaxpr._debug_info.safe_result_paths(len(jaxpr.outvars)),
          num_replicas=replicas.num_global_replicas,
          lowering_parameters=lowering_parameters)
  return PmapComputation(lowering_result.module,
                         const_args,
                         platforms=platforms,
                         pci=pci, replicas=replicas,
                         shards=shards, tuple_args=tuple_args,
                         unordered_effects=unordered_effects,
                         ordered_effects=ordered_effects,
                         keepalive=lowering_result.keepalive,
                         host_callbacks=lowering_result.host_callbacks,
                         jaxpr_debug_info=closed_jaxpr.jaxpr._debug_info,
                         shape_poly_state=lowering_result.shape_poly_state)


def _pmap_unmap_shaped_array(size: int, axis: int | None, aval: ShapedArray
                             ) -> ShapedArray:
  if axis is None: return aval
  elif type(axis) is int:
    return ShapedArray(tuple_update(aval.shape, axis, size), aval.dtype,
                       weak_type=aval.weak_type)
  else: raise TypeError(axis)


AvalMapHandlerPair = tuple[Any, Callable]
_pmap_aval_mapping_handlers: dict[type, AvalMapHandlerPair] = {
    ShapedArray:   (Any, _pmap_unmap_shaped_array),
}

def _pmap_unmapped_aval(size: core.AxisSize, axis: int | None,
                       aval: core.AbstractValue) -> core.AbstractValue:
  if not config.pmap_no_rank_reduction.value:
    return core.unmapped_aval(size, axis, aval)

  _, handler = _pmap_aval_mapping_handlers.get(type(aval), (None, None))
  if handler is not None:
    return handler(size, axis, aval)
  else:
    raise TypeError(f"no unmapping handler for {aval} of type {type(aval)}")


class PmapComputation(stages.Lowering):
  _hlo: ir.Module
  _executable: PmapExecutable | None

  def __init__(self, hlo: ir.Module, const_args: list[ArrayLike],
               **compile_args):
    self._executable = None
    self._hlo = hlo
    self.const_args = const_args
    self.compile_args = compile_args

  # -- stages.Lowering overrides

  def stablehlo(self) -> ir.Module:
    return self._hlo

  @profiler.annotate_function
  def compile(self, compiler_options=None, *, device_assignment=None
              ) -> PmapExecutable:
    if self._executable is None or compiler_options is not None:
      executable = UnloadedPmapExecutable.from_hlo(
          self._hlo, **self.compile_args,
          compiler_options=compiler_options)
      if compiler_options is None:
        self._executable = executable
      return executable
    return self._executable

def _cast_to_shaped_array(aval: core.AbstractValue) -> ShapedArray:
  assert isinstance(aval, ShapedArray), aval
  return aval

@dataclasses.dataclass
class UnloadedPmapExecutable:
  compiled: Any
  backend: xb.XlaBackend
  local_input_avals: Sequence[core.AbstractValue]
  input_shardings: Sequence[JSharding]
  local_output_avals: Sequence[ShapedArray]
  output_shardings: Sequence[JSharding]
  unordered_effects: list[core.Effect]
  ordered_effects: list[core.Effect]
  keepalive: Sequence[Any]
  host_callbacks: Sequence[Any]
  jaxpr_debug_info: core.DebugInfo

  def build_execute_fun(self):
    input_indices = []
    for aval, spec in safe_zip(self.local_input_avals, self.input_shardings):
      assert isinstance(spec, sharding_impls.PmapSharding), spec
      assert isinstance(aval, core.ShapedArray), aval
      input_indices.append(
          sharding_specs.spec_to_indices(aval.shape, spec.sharding_spec)
          if spec.sharding_spec is not None else None)
    handle_outs = local_avals_to_results_handler(self.local_output_avals,
                                                 self.output_shardings)
    handle_args = InputsHandler(self.input_shardings,
                                [None] * len(self.input_shardings),
                                self.compiled.local_devices(), input_indices)
    execute_fun = ExecuteReplicated(self.compiled, "parallel computation",
                                    self.backend, handle_args, handle_outs,
                                    self.unordered_effects,
                                    self.ordered_effects, self.keepalive,
                                    bool(self.host_callbacks),
                                    set(range(len(input_indices))), None)
    return execute_fun

  def load(self) -> PmapExecutable:
    fingerprint = getattr(self.compiled, "fingerprint", None)

    return PmapExecutable(
        self.compiled, self.build_execute_fun, fingerprint,
        self.local_input_avals, self)

  @staticmethod
  def from_hlo(hlo: ir.Module,
               pci: ParallelCallableInfo,
               replicas: ReplicaInfo,
               shards: ShardInfo,
               tuple_args: bool,
               unordered_effects: list[core.Effect],
               ordered_effects: list[core.Effect],
               host_callbacks: list[Any],
               keepalive: Any,
               jaxpr_debug_info: core.DebugInfo,
               platforms: Sequence[str],
               shape_poly_state: mlir.ShapePolyLoweringState | None = None,
               compiler_options=None):
    del platforms
    if shape_poly_state is not None and shape_poly_state.uses_dim_vars:
      hlo = mlir.refine_polymorphic_shapes(hlo)

    devices = pci.devices
    if devices is None:
      if shards.num_global_shards > xb.device_count(pci.backend):
        msg = ("compiling computation that requires {} logical devices, but only {} XLA "
               "devices are available (num_replicas={})")
        raise ValueError(msg.format(shards.num_global_shards,
                                    xb.device_count(pci.backend),
                                    replicas.num_global_replicas))
      # On a single host, we simply grab the first N devices from jax.devices().
      # In the single host case, we want the default device order of pmap to
      # match jax.devices().
      # On multiple hosts, we create a default device assignment that ensures
      # each host is responsible for a contiguous set of replicas.
      if shards.num_global_shards > shards.num_local_shards:
        # TODO(skye): use a locality-aware assignment that satisfies the above
        # constraint.
        devices = [d for process_index in range(xb.process_count(pci.backend))
                  for d in xb.local_devices(process_index, pci.backend)]
      else:
        devices = xb.local_devices(backend=pci.backend)[:shards.num_local_shards]
    else:
      if shards.num_local_shards != len(pci.local_devices):
        local_devices_str = ", ".join(map(str, pci.local_devices))
        if shards.num_local_shards == pci.axis_size:
          raise ValueError(
              f"Leading axis size of input to pmapped function must equal the "
              f"number of local devices passed to pmap. Got axis_size="
              f"{pci.axis_size}, num_local_devices={len(pci.local_devices)}.\n"
              f"(Local devices available to pmap: {local_devices_str})")
        else:
          raise ValueError(
              f"pmapped function requires {shards.num_local_shards} local "
              f"devices to run due to nested pmapped or other parallel "
              f"functions, but only {len(pci.local_devices)} are available.\n"
              f"(outer axis size: {pci.axis_size}, local devices available to "
              f"pmap: {local_devices_str})")
      if shards.num_global_shards != len(devices):
        raise ValueError("compiling computation that creates %s shards, "
                        "but %s devices were specified" %
                        (shards.num_global_shards, len(devices)))

    # 'devices' may be 1D or 2D at this point (e.g.
    # get_default_device_assignment() returns 2D assignment, caller may have
    # provided 1D list of devices).
    # Convert to 2D in case it's 1D and we have > 1 partitions.
    num_partitions = 1
    device_assignment: np.ndarray = np.array(devices).reshape(
        (replicas.num_global_replicas, num_partitions))
    compile_options = compiler.get_compile_options(
        num_replicas=replicas.num_global_replicas,
        num_partitions=num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=False,
        env_options_overrides=compiler_options,
        detailed_logging=compiler.use_detailed_logging(hlo),
        backend=pci.backend,
    )
    compile_options.parameter_is_tupled_arguments = tuple_args

    process_index = xb.process_index(pci.backend)
    local_device_assignment = np.array([
        d for d in device_assignment.flat if d.process_index == process_index
    ])

    input_sharding_specs = [
        sharding_specs.pmap_sharding_spec(
            replicas.num_local_replicas, pci.axis_size,
            cast(ShapedArray, aval).shape, in_axis)
        for aval, in_axis in safe_zip(shards.sharded_avals, pci.in_axes)]
    in_shardings = _get_pmap_sharding(local_device_assignment,
                                      input_sharding_specs)

    local_unmapped_avals = [
        _cast_to_shaped_array(
            _pmap_unmapped_aval(pci.axis_size, out_axis, aval))
        if out_axis is not None else aval
        for aval, out_axis in safe_zip(shards.out_sharded_avals, pci.out_axes)]
    out_specs = [
        sharding_specs.pmap_sharding_spec(
            replicas.num_local_replicas, pci.axis_size, aval.shape, out_axis)
        for aval, out_axis in safe_zip(
            shards.out_sharded_avals, pci.out_axes)]
    out_shardings = _get_pmap_sharding(local_device_assignment, out_specs)

    with dispatch.log_elapsed_time(
        "Finished XLA compilation of {fun_name} in {elapsed_time:.9f} sec",
        fun_name=pci.name, event=dispatch.BACKEND_COMPILE_EVENT):
      # `executable_devices` contains devices for output shardings of a pmapped
      # function. It contains only local devices for correspondence with
      # `PmapSharding`s, which also contain only local devices.
      executable_devices = _create_device_list(
          tuple(local_device_assignment.flat))
      assert executable_devices is not None
      compiled = compiler.compile_or_get_cached(
          pci.backend, hlo, device_assignment, compile_options,
          host_callbacks, executable_devices)

    return UnloadedPmapExecutable(
        compiled=compiled,
        backend=pci.backend,
        local_input_avals=pci.avals,
        input_shardings=in_shardings,
        local_output_avals=local_unmapped_avals,
        output_shardings=out_shardings,
        unordered_effects=unordered_effects,
        ordered_effects=ordered_effects,
        keepalive=keepalive,
        host_callbacks=host_callbacks,
        jaxpr_debug_info=jaxpr_debug_info).load()


class PmapExecutable(stages.Executable):
  __slots__ = ["xla_executable", "_unsafe_call", "build_unsafe_call",
               "fingerprint", "in_avals", "_unloaded_executable"]

  def __init__(self, xla_executable, build_unsafe_call, fingerprint,
               in_avals,
               unloaded_executable: UnloadedPmapExecutable):
    self.xla_executable = xla_executable
    self._unsafe_call = None
    self.build_unsafe_call = build_unsafe_call
    self.fingerprint = fingerprint
    self.in_avals = in_avals
    self._unloaded_executable = unloaded_executable

  @property
  def unsafe_call(self) -> Callable[..., Any]:
    if self._unsafe_call is None:
      self._unsafe_call = self.build_unsafe_call()
    return self._unsafe_call  # type: ignore

  # -- stages.Executable overrides

  def xla_extension_executable(self):
    return self.xla_executable

  @profiler.annotate_function
  def call(self, *args):
    # TODO(frostig): do we need to check sharding and sharded avals?
    arg_avals = map(core.abstractify, args)
    check_arg_avals_for_call(self.in_avals, arg_avals,
                             self._unloaded_executable.jaxpr_debug_info)
    return self.unsafe_call(*args)  # pylint: disable=not-callable


def _get_pmap_sharding(devices, specs):
  return [sharding_impls.PmapSharding(devices, spec) for spec in specs]


class InputsHandler:
  __slots__ = ("handler", "in_shardings", "in_layouts", "local_devices",
               "input_indices")

  def __init__(self, in_shardings, in_layouts, local_devices=None,
               input_indices=None):
    self.handler = partial(shard_args, in_shardings, in_layouts,
                           [None] * len(in_shardings))
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
  # `out_avals` is the `Array` global avals when using pjit. It is the
  # local one when using `pmap`.
  __slots__ = ("handlers", "out_shardings", "out_avals")

  def __init__(self, handlers, out_shardings, out_avals):
    self.handlers = handlers
    self.out_shardings = out_shardings
    self.out_avals = out_avals

  def __call__(self, out_bufs):
    return [h(bufs) for h, bufs in safe_zip(self.handlers, out_bufs)]


def local_avals_to_results_handler(
    unmapped_local_out_avals: Sequence[ShapedArray],
    local_shardings: Sequence[JSharding]) -> ResultsHandler:
  out_indices = [tuple(s.devices_indices_map(aval.shape).values())
                 for s, aval in safe_zip(local_shardings, unmapped_local_out_avals)]
  handlers = [
      local_aval_to_result_handler(aval, s, idcs)
      for aval, s, idcs in safe_zip(unmapped_local_out_avals, local_shardings, out_indices)
  ]
  return ResultsHandler(handlers, local_shardings, unmapped_local_out_avals)


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
          assert isinstance(token.sharding, sharding_impls.SingleDeviceSharding)
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
    args = [x for i, x in enumerate(args) if i in self.kept_var_idx]
    if self.mut:
      args = [*args, *self.mut.in_mut]
    input_bufs = self.in_handler(args)
    with profiler.PGLEProfiler.trace(self.pgle_profiler):
      if (self.ordered_effects or self.has_unordered_effects
          or self.has_host_callbacks):
        input_bufs = self._add_tokens_to_inputs(input_bufs)
        results = self.xla_executable.execute_sharded(
            input_bufs, with_tokens=True
        )

        result_token_bufs = results.disassemble_prefix_into_single_device_arrays(
            len(self.ordered_effects))
        sharded_runtime_token = results.consume_token()
        self._handle_token_bufs(result_token_bufs, sharded_runtime_token)
      else:
        results = self.xla_executable.execute_sharded(input_bufs)

      if dispatch.needs_check_special():
        out_arrays = results.disassemble_into_single_device_arrays()
        for arrays in out_arrays:
          dispatch.check_special(self.name, arrays)
        out = self.out_handler(out_arrays)
      else:
        out = results.consume_with_handlers(self.out_handler.handlers)

      if (self.pgle_profiler is not None and self.pgle_profiler.is_running()
          and len(out) > 0):
        out[0].block_until_ready()

    if self.mut is None:
      return out
    else:
      out_ = []
      for i, o in zip(self.mut.out_mut, out):
        if i is not None:
          args[i]._buf._replace_with(o)  # type: ignore
        else:
          out_.append(o)
      return out_


xla_pmap_p = core.MapPrimitive('xla_pmap')
xla_pmap = xla_pmap_p.bind
xla_pmap_p.def_impl(xla_pmap_impl)

def _pmap_partial_eval_custom_params_updater(
    unks_in, inst_in, kept_outs_known, kept_outs_staged, num_res, params_known,
    params_staged):
  # prune inputs to jaxpr_known according to unks_in
  donated_invars_known, _ = partition_list(unks_in, params_known['donated_invars'])
  in_axes_known, _ = partition_list(unks_in, params_known['in_axes'])
  _, out_axes_known = partition_list(kept_outs_known, params_known['out_axes'])
  out_axes_known = out_axes_known + [0] * num_res
  new_params_known = dict(params_known, in_axes=tuple(in_axes_known),
                          out_axes=tuple(out_axes_known),
                          donated_invars=tuple(donated_invars_known))

  # added num_res new inputs to jaxpr_staged, pruning according to inst_in
  _, donated_invars_staged = partition_list(inst_in, params_staged['donated_invars'])
  donated_invars_staged = [False] * num_res + donated_invars_staged
  _, in_axes_staged = partition_list(inst_in, params_staged['in_axes'])
  in_axes_staged = [0] * num_res + in_axes_staged
  _, out_axes_staged = partition_list(kept_outs_staged, params_staged['out_axes'])
  new_params_staged = dict(params_staged, in_axes=tuple(in_axes_staged),
                           out_axes=tuple(out_axes_staged),
                           donated_invars=tuple(donated_invars_staged))
  return new_params_known, new_params_staged

def _pmap_partial_eval_custom_res_maker(params_known, aval):
  return core.unmapped_aval(params_known['axis_size'], 0, aval)

def _pmap_dce_rule(used_outputs, eqn):
  # just like pe.dce_jaxpr_call_rule, except handles in_axes / out_axes
  if not any(used_outputs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None
  axis_name = eqn.params["axis_name"]
  with core.extend_axis_env_nd([(axis_name, eqn.params["global_axis_size"])]):
    new_jaxpr, used_inputs = pe.dce_jaxpr(eqn.params['call_jaxpr'], used_outputs)
  _, donated_invars = partition_list(used_inputs, eqn.params['donated_invars'])
  _, in_axes = partition_list(used_inputs, eqn.params['in_axes'])
  _, out_axes = partition_list(used_outputs, eqn.params['out_axes'])
  new_params = dict(eqn.params, call_jaxpr=new_jaxpr,
                    donated_invars=tuple(donated_invars),
                    in_axes=tuple(in_axes), out_axes=tuple(out_axes))
  if not any(used_inputs) and not any(used_outputs) and not new_jaxpr.effects:
    return used_inputs, None
  else:
    effs = core.filter_named_axis_effects(new_jaxpr.effects, {axis_name})
    new_eqn = pe.new_jaxpr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs) if used],
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, effs, eqn.source_info)
    return used_inputs, new_eqn


def _xla_call_partial_eval_update_params(
    params: core.ParamDict, kept_inputs: Sequence[bool], num_new_inputs: int
  ) -> core.ParamDict:
  donated_invars = params['donated_invars']
  if not kept_inputs and donated_invars:
    # JaxprTrace.post_process_call creates a call with no input tracers
    donated_invars = (False,) * num_new_inputs
  else:
    assert len(kept_inputs) == len(donated_invars)
    # JaxprTrace.process_call drops known input tracers
    donated_invars = [d for d, kept in zip(donated_invars, kept_inputs) if kept]
    # Any new inputs are prepended to the left, so mark those as not donated.
    donated_invars = [False] * num_new_inputs + donated_invars
  return dict(params, donated_invars=tuple(donated_invars))

def xla_call_jvp_update_params(params, nz_tangents):
  donated_invars = params['donated_invars']
  donated_tangents = [d for d, nz in zip(donated_invars, nz_tangents) if nz]
  new_donated_invars = (*donated_invars, *donated_tangents)
  return dict(params, donated_invars=new_donated_invars)

def _xla_call_linearize_update_params(params, num_new_inputs, nz_tangents):
  donated_invars_prev = params['donated_invars']
  donated_invars = (*(False for _ in range(num_new_inputs)),
                    *(d for d, nz in zip(donated_invars_prev, nz_tangents) if nz))
  return dict(params, donated_invars=donated_invars)

def _xla_call_transpose_update_params(params, undef_primals, nonzero_cts):
  donated_invars = params['donated_invars']
  donated_primals = [d for d, u in zip(donated_invars, undef_primals) if not u]
  donated_cotangents = [False for nz in nonzero_cts if nz]
  return dict(params, donated_invars=(*donated_primals, *donated_cotangents))


# Set param update handlers to update `donated_invars` just like xla_call_p
pe.call_param_updaters[xla_pmap_p] = _xla_call_partial_eval_update_params
pe.partial_eval_jaxpr_custom_rules[xla_pmap_p] = \
    partial(pe.call_partial_eval_custom_rule,
            'call_jaxpr', _pmap_partial_eval_custom_params_updater,
            res_aval=_pmap_partial_eval_custom_res_maker)
pe.dce_rules[xla_pmap_p] = _pmap_dce_rule
ad.call_param_updaters[xla_pmap_p] = xla_call_jvp_update_params
ad.call_linearize_param_updaters[xla_pmap_p] = _xla_call_linearize_update_params
ad.call_transpose_param_updaters[xla_pmap_p] = _xla_call_transpose_update_params

ad.primitive_transposes[xla_pmap_p] = partial(ad.map_transpose, xla_pmap_p)

def _unravel_index_hlo(axis_env):
  div = mlir.ir_constant(
      np.array(axis_env.nreps // math.prod(axis_env.sizes), np.uint32))
  mod = mlir.ir_constant(np.array(axis_env.sizes[-1], np.uint32))
  return hlo.remainder(hlo.divide(hlo.replica_id(), div), mod)

def _hlo_shard(aval, axis_env, x, in_axis):
  if aval is core.abstract_token:
    return x
  elif isinstance(aval, core.ShapedArray):
    if dtypes.issubdtype(aval.dtype, dtypes.extended):
      aval = core.physical_element_aval(aval.dtype)
    dims = list(aval.shape)
    zero = mlir.ir_constant(np.zeros((), dtype=np.uint32))
    idxs = [zero] * len(dims)
    idxs.insert(in_axis, _unravel_index_hlo(axis_env))
    dims_unsqueezed = dims.copy()
    dims_unsqueezed.insert(in_axis, 1)
    dynamic_slice_result = hlo.dynamic_slice(
        x, idxs, mlir.dense_int_array(dims_unsqueezed))
    return hlo.reshape(mlir.aval_to_ir_type(aval), dynamic_slice_result)
  else:
    raise TypeError(aval)


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


# TODO(b/110096942): more efficient gather
def _hlo_unshard(ctx: mlir.LoweringRuleContext, aval, axis_env, out_axis, x):
  if aval is core.abstract_token:
    return x
  elif isinstance(aval, core.ShapedArray):
    dims = list(aval.shape)
    padded_aval = aval.update(shape=[axis_env.sizes[-1]] + dims)
    padded = mlir.full_like_aval(ctx, 0, padded_aval)
    zero = mlir.ir_constant(np.zeros((), dtype=np.uint32))
    idxs = [_unravel_index_hlo(axis_env)] + [zero] * len(dims)
    broadcast_result = hlo.broadcast(x, mlir.dense_int_array([1]))
    padded = hlo.dynamic_update_slice(padded, broadcast_result, idxs)
    replica_groups = mlir.dense_int_elements(
      axis_groups(axis_env, axis_env.names[-1]))
    out = hlo.cross_replica_sum(padded, replica_groups)
    if out_axis != 0:
      # TODO(apaszke,mattjj): Change the indices to DynamicUpdateSlice instead
      perm = list(range(1, len(dims)))
      perm.insert(out_axis, 0)
      transposed_dims = list(dims)
      transposed_dims.insert(out_axis, axis_env.sizes[-1])
      out = hlo.transpose(out, mlir.dense_int_array(perm))

    return out
  else:
    raise TypeError(aval)

def _extend_axis_env(env: sharding_impls.AxisEnv, name, size: int):
  return sharding_impls.AxisEnv(env.nreps, env.names + (name,),
                                env.sizes + (size,))


def _pmap_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, axis_name,
                   axis_size, global_axis_size, devices, name,
                   call_jaxpr: core.Jaxpr, backend=None, in_axes, out_axes,
                   donated_invars, is_explicit_global_axis_size):
  del donated_invars  # Unused.
  mlir.check_backend_matches(backend, ctx.module_context.platforms)
  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  if ctx.module_context.axis_env.names and devices is not None:
    raise ValueError("Nested pmap with explicit devices argument.")
  new_env = _extend_axis_env(ctx.module_context.axis_env, axis_name,
                             global_axis_size)
  # Shard the in_nodes that are mapped
  in_avals = [v.aval for v in call_jaxpr.invars]
  in_nodes_sharded = (
    _hlo_shard(aval, new_env, in_node, in_axis)
    if in_axis is not None else in_node
    for aval, in_node, in_axis in zip(in_avals, in_nodes, in_axes))

  with core.extend_axis_env_nd([(axis_name, global_axis_size)]):
    sub_ctx = ctx.module_context.replace(
        axis_context=sharding_impls.ReplicaAxisContext(new_env))
    sharded_outs, _ = mlir.jaxpr_subcomp(
        sub_ctx, call_jaxpr,
        ctx.name_stack.extend(util.wrap_name('pmap', name)),
        mlir.TokenSet(), (), *in_nodes_sharded,
        dim_var_values=ctx.dim_var_values, const_lowering=ctx.const_lowering)
  out_avals = [v.aval for v in call_jaxpr.outvars]
  outs = [_hlo_unshard(ctx, aval, new_env, out_axis, shard)
          for aval, out_axis, shard in zip(out_avals, out_axes, sharded_outs)]
  return outs

mlir.register_lowering(xla_pmap_p, _pmap_lowering)


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


full_to_shard_p = core.Primitive('full_to_shard')

@full_to_shard_p.def_abstract_eval
def _full_to_shard_abstract_eval(x, axes, mesh, **_):
  # TODO: Assert x is a global aval! Or ideally check that it's global in dims from axes!
  return tile_aval_nd(mesh.shape, axes, x)

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

@partial(mlir.register_lowering, full_to_shard_p)
def _full_to_shard_lowering(ctx, x, *, axes: ArrayMapping, mesh: Mesh,
                            manual_axes: frozenset[sharding_impls.MeshAxisName]):
  # TODO: Can we short-circuit for replicated values? Probably not.
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  sharding_proto = (
      NamedSharding(mesh, array_mapping_to_axis_resources(axes))
      ._to_xla_hlo_sharding(aval_in.ndim).to_proto())
  unspecified_dims = set(range(aval_in.ndim)) - set(axes.values())
  sx = mlir.wrap_with_sharding_op(ctx, x, aval_in, sharding_proto,
                                  unspecified_dims=unspecified_dims)
  proto = manual_proto(aval_in, manual_axes, mesh)
  return (mlir.wrap_with_full_to_shard_op(ctx, sx, aval_out, proto,
                                          unspecified_dims=unspecified_dims),)

shard_to_full_p = core.Primitive('shard_to_full')

@shard_to_full_p.def_abstract_eval
def _shard_to_full_abstract_eval(x, axes, mesh, **_):
  # TODO: Assert x is a global aval! Or ideally check that it's global in dims from axes!
  return untile_aval_nd(mesh.shape, axes, x)

@partial(mlir.register_lowering, shard_to_full_p)
def _shard_to_full_lowering(ctx: mlir.LoweringRuleContext, x, *, axes: ArrayMapping, mesh: Mesh,
                            manual_axes: frozenset[sharding_impls.MeshAxisName]):
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  proto = manual_proto(aval_in, manual_axes, mesh)  # type: ignore
  unspecified_dims = set(range(aval_in.ndim)) - set(axes.values())  # type: ignore
  sx = mlir.wrap_with_sharding_op(ctx, x, aval_in, proto,
                                  unspecified_dims=unspecified_dims)
  sharding_proto = (
      NamedSharding(mesh, array_mapping_to_axis_resources(axes))
      ._to_xla_hlo_sharding(aval_out.ndim).to_proto())
  return (mlir.wrap_with_shard_to_full_op(ctx, sx, aval_out, sharding_proto,
                                          unspecified_dims),)


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
    devices: Sequence[xc.Device] | None,
) -> tuple[xc.Client, tuple[xc.Device, ...]]:
  first_sharding_info = None
  devices = () if devices is None else tuple(devices)

  for sh, s_type, source_info in shardings:
    if isinstance(sh, UnspecifiedValue):
      continue
    if isinstance(sh, NamedSharding) and isinstance(sh.mesh, AbstractMesh):
      continue
    if first_sharding_info is None:
      first_sharding_info = (
          (sh.mesh._flat_devices_tuple, s_type, source_info) if isinstance(sh, AUTO)
           else (sh._device_assignment, s_type, source_info))
    arr_device_assignment = (sh.mesh._flat_devices_tuple if isinstance(sh, AUTO)
                             else sh._device_assignment)
    if not devices:
      if first_sharding_info[0] != arr_device_assignment:
        raise stages.DeviceAssignmentMismatchError([
            stages.DeviceAssignmentMismatch(*first_sharding_info),
            stages.DeviceAssignmentMismatch(arr_device_assignment, s_type, source_info)])
    else:
      if devices != arr_device_assignment:
        raise stages.DeviceAssignmentMismatchError([
            stages.DeviceAssignmentMismatch(devices, stages.MismatchType.CONTEXT_DEVICES, None),
            stages.DeviceAssignmentMismatch(arr_device_assignment, s_type, source_info)])
  if first_sharding_info is None and devices:
    final_device_assignment = devices
  elif first_sharding_info is None:
    final_device_assignment = (get_default_device(),)
  else:
    final_device_assignment = first_sharding_info[0]  # type: ignore
  return xb.get_device_backend(final_device_assignment[0]), final_device_assignment

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
  in_mut: list[core.MutableArray]
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
) -> tuple[core.ClosedJaxpr, list[core.MutableArray]]:
  jaxpr = closed_jaxpr.jaxpr
  hoist = [isinstance(c, core.MutableArray) for c in closed_jaxpr.consts]
  consts, in_mut = partition_list(hoist, closed_jaxpr.consts)
  constvars, mutvars = partition_list(hoist, jaxpr.constvars)
  invars = (*jaxpr.invars, *mutvars)
  effects = pe.make_jaxpr_effects(constvars, invars, jaxpr.outvars, jaxpr.eqns)
  # TODO(mattjj): debug_info must be updated...
  jaxpr = core.Jaxpr(constvars, invars, jaxpr.outvars, jaxpr.eqns,
                     effects, closed_jaxpr.jaxpr.debug_info)
  return core.ClosedJaxpr(jaxpr, consts), in_mut

@weakref_lru_cache
def _discharge_internal_refs(jaxpr: core.ClosedJaxpr) -> core.ClosedJaxpr:
  from jax._src.state.discharge import discharge_state  # pytype: disable=import-error
  jaxpr_, consts = discharge_state(jaxpr.jaxpr, jaxpr.consts)
  jaxpr_._debug_info = jaxpr.jaxpr._debug_info
  return core.ClosedJaxpr(jaxpr_, consts)


class SemanticallyEqualShardings:

  def __init__(self, shardings: tuple[GSPMDSharding | UnspecifiedValue, ...],
               avals: tuple[core.AbstractValue]):
    gspmd_shardings = [
        s if (isinstance(s, (UnspecifiedValue, AUTO)) or
              (isinstance(s, NamedSharding) and isinstance(s.mesh, AbstractMesh)))
        else to_gspmd_sharding(s, a.ndim)  # pytype: disable=attribute-error
        for s, a in zip(shardings, avals)]
    self._gspmd_shardings = gspmd_shardings
    self.shardings = shardings
    self.avals = avals

  def __hash__(self):
    return hash(tuple(
        (s._hlo_sharding_hash, s.memory_kind)
        if isinstance(s, GSPMDSharding) else s for s in self._gspmd_shardings))

  def __eq__(self, other):
    if not isinstance(other, SemanticallyEqualShardings):
      return False
    return all(
        (op_shardings.are_op_shardings_equal(s._hlo_sharding, o._hlo_sharding)
         and s.memory_kind == o.memory_kind)
        if (isinstance(s, GSPMDSharding) and isinstance(o, GSPMDSharding))
        else s == o
        for s, o in zip(self._gspmd_shardings, other._gspmd_shardings)
    )


def _raise_warnings_or_errors_for_jit_of_pmap(
    nreps: int, backend: xc.Client, name: str, jaxpr: core.Jaxpr) -> None:
  if nreps > 1:
    warnings.warn(
        f"The function {name} includes a pmap. Using "
         "jit-of-pmap can lead to inefficient data movement, as the outer jit "
         "does not preserve sharded data representations and instead collects "
         "input and output arrays onto a single device. "
         "Consider removing the outer jit unless you know what you're doing. "
         "See https://github.com/jax-ml/jax/issues/2926. Or "
         "use jax.shard_map instead of pmap under jit compilation.")

  if nreps > xb.device_count(backend):
    raise ValueError(
        f"compiling computation `{name}` that requires {nreps} replicas, but "
        f"only {xb.device_count(backend)} XLA devices are available.")

  if xb.process_count(backend) > 1 and (
      nreps > 1 or dispatch.jaxpr_has_primitive(jaxpr, "xla_pmap")
  ):
    raise NotImplementedError(
        "jit of multi-host pmap not implemented (and jit-of-pmap can cause "
        "extra data movement anyway, so maybe you don't want it after all).")


@weakref_lru_cache
def _cached_lowering_to_hlo(closed_jaxpr: core.ClosedJaxpr, module_name, backend,
                            num_const_args: int,
                            in_avals,
                            semantic_in_shardings, semantic_out_shardings,
                            in_layouts, out_layouts, num_devices, device_assignment,
                            donated_invars, all_default_mem_kind,
                            inout_aliases: None | tuple[None | int, ...],
                            propagated_out_mem_kinds: tuple[None | str, ...],
                            platforms: tuple[str, ...],
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

  # Look at the number of replcas present in the jaxpr. In
  # lower_sharding_computation, nreps > 1 during `jit(pmap)` cases. This is
  # handled here so as to deprecate the lower_xla_callable codepath when
  # `jax.Array` is turned on by default.
  # TODO(yashkatariya): Remove this when `jit(pmap)` is removed.
  nreps = _jaxpr_replicas(jaxpr)
  _raise_warnings_or_errors_for_jit_of_pmap(nreps, backend, module_name, jaxpr)

  in_mlir_shardings: list[JSharding | AUTO | None] | None
  out_mlir_shardings: list[JSharding | AUTO | None] | None
  axis_ctx: mlir.AxisContext

  if nreps == 1:
    in_mlir_shardings = map(_to_logical_sharding, in_avals, in_shardings)
    out_mlir_shardings = map(_to_logical_sharding, out_avals, out_shardings)
    replicated_args = [False] * len(in_avals)
    axis_ctx = sharding_impls.ShardingContext(num_devices, device_assignment,
                                              abstract_mesh)
    num_partitions = num_devices
  else:
    # This path is triggered for `jit(pmap)` cases.
    replicated_args = None
    in_mlir_shardings = None
    out_mlir_shardings = None
    axis_env = sharding_impls.AxisEnv(nreps, (), ())
    axis_ctx = sharding_impls.ReplicaAxisContext(axis_env)
    num_partitions = 1


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
        num_replicas=nreps,
        num_partitions=num_partitions,
        all_default_mem_kind=all_default_mem_kind,
        input_output_aliases=inout_aliases,
        propagated_out_mem_kinds=propagated_out_mem_kinds,
        lowering_parameters=lowering_parameters)
  tuple_args = dispatch.should_tuple_args(len(in_avals), backend.platform)
  unordered_effects = list(
      effects.ordered_effects.filter_not_in(closed_jaxpr.effects))
  return (lowering_result.module, lowering_result.keepalive,
          lowering_result.host_callbacks, unordered_effects, ordered_effects,
          nreps, tuple_args, lowering_result.shape_poly_state)


@util.cache(max_size=2048, trace_context_in_key=False)
def _create_device_list_cached(device_assignment: tuple[xc.Device, ...]
                             ) -> xc.DeviceList:
  return xc.DeviceList(device_assignment)

def _create_device_list(
    device_assignment: tuple[xc.Device, ...] | xc.DeviceList | None
    ) -> xc.DeviceList | None:
  if device_assignment is None or isinstance(device_assignment, xc.DeviceList):
    return device_assignment  # type: ignore
  return _create_device_list_cached(device_assignment)


@weakref_lru_cache
def jaxpr_transfer_mem_kinds(jaxpr: core.Jaxpr):
  out = []  # type: ignore
  for eqn in jaxpr.eqns:
    if eqn.primitive is dispatch.device_put_p:
      out.extend(d for d in eqn.params['devices']
                 if isinstance(d, core.MemorySpace))
  for subjaxpr in core.subjaxprs(jaxpr):
    out.extend(jaxpr_transfer_mem_kinds(subjaxpr))
  return out


def are_all_shardings_default_mem_kind(
    device_list: xc.DeviceList | None, shardings
):
  if device_list is None:
    return True

  try:
    default_mem_kind = device_list.default_memory_kind
  except:
    return True

  for i in shardings:
    if isinstance(i, (UnspecifiedValue, AUTO)):
      continue
    mem_kind = (core.mem_space_to_kind(i) if isinstance(i, core.MemorySpace)
                else i.memory_kind)
    if mem_kind is None:
      continue
    if mem_kind != default_mem_kind:
      return False
  return True


@weakref_lru_cache
def get_out_layouts_via_propagation(closed_jaxpr: core.ClosedJaxpr
                                    ) -> tuple[None | Layout]:
  env = {}  # type: ignore
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
    # TODO(yashkatariya): Replace this with a registration system when there are
    # more primitives for layout propagation.
    if eqn.primitive is pjit.sharding_constraint_p:
      out_eqn_layouts = [eqn.params['layout']]
    else:
      out_eqn_layouts = [None] * len(eqn.outvars)
    safe_map(write, eqn.outvars, out_eqn_layouts)
  return tuple(safe_map(read, jaxpr.outvars))


def _get_num_devices(
    shardings, device_assignment
  ) -> tuple[int, tuple[xc.Device, ...] | None]:
  """Number of lowering devices, and the device_assignment to use.

  If all the specified shardings have an abstract mesh, then we are compiling
  with abstract devices, and the returned device_assignment is None.
  """
  abstract_mesh, any_concrete_sharding = None, False
  for s in shardings:
    if isinstance(s, UnspecifiedValue):
      continue
    elif (isinstance(s, NamedSharding) and isinstance(s.mesh, AbstractMesh) and
          not s.mesh.empty):
      if abstract_mesh is not None and abstract_mesh != s.mesh:
        raise ValueError("AbstractMesh should be the same across all "
                         f"shardings. Got {abstract_mesh} and {s.mesh}")
      abstract_mesh = s.mesh
    else:
      any_concrete_sharding = True
  if (any_concrete_sharding and abstract_mesh is not None and
      len(device_assignment) != abstract_mesh.size):
    raise ValueError(
        f"AbstractMesh size: {abstract_mesh.size} does not match the"
        f" device assignment size: {len(device_assignment)}")
  if any_concrete_sharding or abstract_mesh is None:
    return len(device_assignment), device_assignment
  return abstract_mesh.size, None


MaybeLayout = Sequence[Union[Layout, AutoLayout, None]]


class AllArgsInfo(NamedTuple):
  """Avals and debug_info for all arguments prior to DCE."""
  in_avals: Sequence[core.ShapedArray]
  debug_info: core.DebugInfo


@util.cache(max_size=2048, trace_context_in_key=False)
def to_gspmd_sharding(s: JSharding, ndim: int) -> GSPMDSharding:
  if isinstance(s, GSPMDSharding):
    return s
  return GSPMDSharding(s._internal_device_list, s._to_xla_hlo_sharding(ndim),
                       memory_kind=s.memory_kind)


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
  const_args = core.jaxpr_const_args(closed_jaxpr.jaxpr)
  num_const_args = len(const_args)
  if num_const_args:
    const_arg_avals = [core.shaped_abstractify(c) for c in const_args]
    global_in_avals = const_arg_avals + global_in_avals  # type: ignore
    ca_shardings = pjit.const_args_shardings(const_args)
    in_shardings = ca_shardings + in_shardings  # type: ignore
    ca_layouts = pjit.const_args_layouts(const_args, const_arg_avals,
                                          ca_shardings)
    in_layouts = ca_layouts + in_layouts  # type: ignore

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
    all_args_info = AllArgsInfo(
        const_arg_avals + all_args_info.in_avals,  # type: ignore
        all_args_info.debug_info._replace(
            arg_names=(("",) * num_const_args + all_args_info.debug_info.arg_names)))
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

  out = []
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


def _get_context_mesh(context_mesh: Mesh) -> Mesh:
  # Don't update the mesh because the old `with mesh` ctx mgr is set.
  if get_concrete_mesh().empty:
    return context_mesh
  cur_mesh = get_abstract_mesh()
  if cur_mesh.empty or context_mesh.empty:
    return context_mesh
  if cur_mesh == context_mesh.abstract_mesh:
    return context_mesh
  return Mesh(context_mesh.devices, context_mesh.axis_names,
              axis_types=cur_mesh.axis_types)


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
    context_mesh: Mesh,
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
  auto_spmd_lowering = check_if_any_auto(
      it.chain.from_iterable([in_shardings, out_shardings]))

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

  devices_from_context = (None if context_mesh.empty else
                          context_mesh._flat_devices_tuple)
  # Device assignment across all inputs, outputs and shardings inside jaxpr
  # should be the same.
  unique_intermediate_shardings = util.stable_unique(
      dispatch.get_intermediate_shardings(jaxpr))
  unique_const_shardings = util.stable_unique(in_shardings[:len(const_args)])
  unique_in_shardings = util.stable_unique(in_shardings[len(const_args):])
  unique_out_shardings = util.stable_unique(out_shardings)
  # TODO(necula): Replace `None` with `source_info` for unique_const_shardings
  backend, device_assignment = _get_and_check_device_assignment(
      it.chain(
          ((i, stages.MismatchType.ARG_SHARDING, None) for i in unique_in_shardings),
          ((c, stages.MismatchType.CONST_SHARDING, None) for c in unique_const_shardings),
          ((o, stages.MismatchType.OUT_SHARDING, None) for o in unique_out_shardings),
          ((js, stages.MismatchType.SHARDING_INSIDE_COMPUTATION, source_info)
           for js, source_info in unique_intermediate_shardings)),
      devices_from_context)
  unique_intermediate_shardings = [js for js, _ in unique_intermediate_shardings]
  unique_in_shardings = unique_in_shardings | unique_const_shardings  # type: ignore
  del unique_const_shardings

  for a in global_out_avals:
    if (a is not core.abstract_token and not a.sharding.mesh.empty and
        a.sharding.mesh.are_all_axes_explicit and
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

  prim_requires_devices = dispatch.jaxpr_has_prim_requiring_devices(jaxpr)

  # TODO(yashkatariya): All device specific logic should go in compilation
  # but this requires a big refactor. The current `_get_num_devices` logic
  # is good enough to lower with AbstractMesh but cannot be compiled. Once
  # I refactor, this will also work well with mesh being provided at
  # compile time.
  # Sets device_assignment to None if only abstractMesh and unspecified exists.
  num_devices, device_assignment = _get_num_devices(
      it.chain(unique_in_shardings, unique_out_shardings,
               unique_intermediate_shardings),
      device_assignment)
  if device_assignment is None:
    if lowering_platforms is None:
      raise ValueError(
          "Passing lowering_platforms via jax.export or "
          " jit(f).trace(*args).lower(lowering_platforms=...) is required when"
          " only AbstractMesh exists in a jitted computation.")
    if prim_requires_devices:
      raise ValueError(
          "AbstractMesh cannot be used when jaxpr contains primitives that"
          " require devices to be present during lowering.")

  committed = bool(
      devices_from_context
      or num_devices > 1
      or any(not isinstance(s, UnspecifiedValue) for s in it.chain(
          unique_in_shardings, unique_out_shardings,
          unique_intermediate_shardings)))

  device_list = _create_device_list(device_assignment)

  transfer_mem_kind_in_jaxpr = jaxpr_transfer_mem_kinds(jaxpr)
  all_default_mem_kind = are_all_shardings_default_mem_kind(
      device_list,
      it.chain(unique_in_shardings, unique_out_shardings,
               unique_intermediate_shardings, transfer_mem_kind_in_jaxpr))  # pytype: disable=wrong-arg-types

  if all_default_mem_kind:
    propagated_out_mem_kinds = (None,) * len(global_out_avals)
  else:
    propagated_out_mem_kinds = tuple(
        core.mem_space_to_kind(o.memory_space) for o in closed_jaxpr.out_avals)  # type: ignore

  out_shardings = _concretize_abstract_out_shardings(
      out_shardings, global_out_avals, device_assignment,
      propagated_out_mem_kinds)

  # 2. Build up the HLO

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
   nreps, tuple_args, shape_poly_state) = _cached_lowering_to_hlo(
       closed_jaxpr, module_name, backend,
       len(const_args), tuple(global_in_avals),
       semantic_in_shardings, semantic_out_shardings,
       in_layouts, out_layouts, num_devices,
       tuple(device_list) if prim_requires_devices else None,  # type: ignore[arg-type]
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
      spmd_lowering=True,
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
      pmap_nreps=nreps,
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
  elif isinstance(aval, (ShapedArray, DShapedArray, AbstractRef)):
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

    if self._executable is None or compiler_options_kvs or device_assignment:
      executable = UnloadedMeshExecutable.from_hlo(
          self._name, self._hlo, **self.compile_args,
          compiler_options_kvs=compiler_options_kvs,
          device_list=compilation_device_list)
      if not compiler_options_kvs:
        self._executable = executable
      return executable
    return self._executable

  def cost_analysis(self) -> dict[str, float]:
    backend = self.compile_args["backend"]
    if xb.using_pjrt_c_api(backend):
      raise NotImplementedError(
          "Lowered.cost_analysis not implemented on platform "
          f"'{backend.platform}'. Use compile().cost_analysis() for "
          "post-compilation cost estimates.")
    return xe.hlo_module_cost_analysis(backend, self.hlo().as_hlo_module())


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
    out_op_shardings = out_op_shardings * num_out_avals  # type: ignore

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


_orig_out_sharding_handlers = {}

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
_orig_out_sharding_handlers[SingleDeviceSharding] = _gspmd_to_single_device_sharding  # type: ignore


def _get_out_sharding_from_orig_sharding(
    out_shardings, out_avals, orig_in_s, orig_aval):
  out = []
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
    computation, mesh, spmd_lowering, tuple_args, auto_spmd_lowering,
    allow_prop_to_inputs, allow_prop_to_outputs, backend,
    np_dev, pmap_nreps, compiler_options):
  if pmap_nreps > 1:
    num_replicas, num_partitions = pmap_nreps, 1
  elif spmd_lowering:
    num_replicas, num_partitions = 1, np_dev.size
  else:
    num_replicas, num_partitions = np_dev.size, 1

  if pmap_nreps > 1:
    # In `jit` device_assignment is set to None when num_replicas > 1. Do
    # the same thing here too.
    xla_device_assignment = None
  else:
    xla_device_assignment = np_dev.reshape((num_replicas, num_partitions))

  fdo_profile = compiler_options.pop("fdo_profile", None)

  compile_options = compiler.get_compile_options(
      num_replicas=num_replicas,
      num_partitions=num_partitions,
      device_assignment=xla_device_assignment,
      use_spmd_partitioning=spmd_lowering,
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
def _cached_compilation(computation, name, mesh, spmd_lowering,
                        tuple_args, auto_spmd_lowering, allow_prop_to_inputs,
                        allow_prop_to_outputs, host_callbacks, backend,
                        da, pmap_nreps, compiler_options_kvs, pgle_profiler):
  # One would normally just write: dev = np.array(device_assignment)
  # The formulation below is substantially faster if there are many devices.
  dev = np.vectorize(lambda i: da[i], otypes=[object])(np.arange(len(da)))
  compiler_options = dict(compiler_options_kvs)

  compile_options = create_compile_options(
      computation, mesh, spmd_lowering, tuple_args, auto_spmd_lowering,
      allow_prop_to_inputs, allow_prop_to_outputs, backend,
      dev, pmap_nreps, compiler_options)

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
      orig_hlo_s = orig._to_xla_hlo_sharding(aval.ndim)  # pytype: disable=attribute-error
      # MANUAL HloSharding comes from other partitioning frameworks.
      if (not dtypes.issubdtype(aval.dtype, dtypes.extended) and
          not xla_hlo_s.is_manual() and
          (not op_shardings.are_op_shardings_equal(xla_hlo_s, orig_hlo_s))):
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
        new_out_shardings.append(_gspmd_to_named_sharding(xla_s, aval, orig))  # pytype: disable=wrong-arg-types
      except:
        new_out_shardings.append(xla_s)
    else:
      xla_hlo_s = xla_s._to_xla_hlo_sharding(aval.ndim)
      orig_hlo_s = orig._to_xla_hlo_sharding(aval.ndim)  # pytype: disable=attribute-error
      # MANUAL HloSharding comes from other partitioning frameworks.
      if (not dtypes.issubdtype(aval.dtype, dtypes.extended) and
          not xla_hlo_s.is_manual() and
          (not op_shardings.are_op_shardings_equal(xla_hlo_s, orig_hlo_s) or
           xla_s.memory_kind != orig.memory_kind)):  # pytype: disable=attribute-error
        raise AssertionError(
            f"Unexpected XLA sharding override: (XLA) {xla_s} != {orig} "
            "(User sharding)")
      new_out_shardings.append(orig)
  return new_out_shardings


def finalize_shardings(shardings, device_assignment):
  if len(device_assignment) == 1:
    return [SingleDeviceSharding(device_assignment[0], memory_kind=o.memory_kind)
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
  backend: xb.XlaBackend
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
               spmd_lowering: bool,
               tuple_args: bool,
               auto_spmd_lowering: bool,
               unordered_effects: list[core.Effect],
               ordered_effects: list[core.Effect],
               host_callbacks: list[Any],
               keepalive: Any,
               kept_var_idx: set[int],
               backend: xb.XlaBackend,
               device_list: xc.DeviceList | None,
               committed: bool,
               in_layouts: MaybeLayout,
               out_layouts: MaybeLayout,
               compiler_options_kvs: tuple[tuple[str, Any], ...],
               num_devices: int,
               pmap_nreps: int = 1,
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
      for i in it.chain.from_iterable([in_shardings, out_shardings]):
        if isinstance(i, AUTO):
          mesh = i.mesh
          break

    util.test_event("pxla_cached_compilation")
    xla_executable = _cached_compilation(
        hlo, name, mesh, spmd_lowering,
        tuple_args, auto_spmd_lowering, allow_prop_to_inputs,
        allow_prop_to_outputs, tuple(host_callbacks), backend, device_list,
        pmap_nreps, compiler_options_kvs, pgle_profiler)

    if auto_spmd_lowering:
      assert mesh is not None
      in_shardings_xla, out_shardings_xla = _get_mesh_pspec_shardings_from_executable(
          xla_executable, mesh)
      in_shardings = [x if isinstance(i, AUTO) else i  # type: ignore
                      for x, i in safe_zip(in_shardings_xla, in_shardings)]
      out_shardings = [x if isinstance(o, AUTO) else o  # type: ignore
                       for x, o in safe_zip(out_shardings_xla, out_shardings)]
    else:
      if pmap_nreps == 1:
        assert mesh is None
        in_shardings = _maybe_get_and_check_in_shardings(
            xla_executable, in_shardings, device_list, global_in_avals,
            len(ordered_effects))
        out_shardings = _maybe_get_and_check_out_shardings(
            xla_executable, out_shardings, device_list, global_out_avals,
            len(ordered_effects))
      else:
        in_shardings, out_shardings, committed, device_list = _get_metadata_jit_pmap(
            xla_executable.local_devices(), len(in_shardings), len(out_shardings))

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
            any(not isinstance(i, UnspecifiedValue) for i in self.in_shardings_leaves) or
            any(not isinstance(o, UnspecifiedValue) for o in self.out_shardings_leaves) or
            any(i is not None for i in self.in_layouts_leaves) or
            any(o is not None for o in self.out_layouts_leaves) or
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
    self.in_avals = in_avals  # includes the const_args
    self.out_avals = out_avals
    self._unsafe_call = None
    self._in_shardings = in_shardings
    self._out_shardings = out_shardings
    self._auto_spmd_lowering = auto_spmd_lowering
    self._kept_var_idx = kept_var_idx
    self._xla_in_layouts = xla_in_layouts
    self._dispatch_in_layouts = dispatch_in_layouts
    self._xla_out_layouts = xla_out_layouts
    self._mut = mut
    self._all_args_info = all_args_info
    self._unloaded_executable = unloaded_executable

  @property
  def unsafe_call(self) -> Callable[..., Any]:
    if self._unsafe_call is None:
      self._unsafe_call = self.build_unsafe_call()
    return self._unsafe_call  # type: ignore

  # -- stages.Executable overrides

  def xla_extension_executable(self):
    return self.xla_executable

  def call(self, *args):
    args_after_dce = [a for i, a in enumerate(args) if i in self._kept_var_idx]
    if self._all_args_info is None:
      kept_args = args_after_dce
      ref_avals = self.in_avals
      # TODO(necula): ensure we have actual debug info; need it before DCE.
      # See https://github.com/jax-ml/jax/issues/26480.
      debug_info = core.DebugInfo(
          "MeshExecutable", "<unknown>",
          tuple(f"args[{i}]" for i in range(len(args))), ())
    else:
      kept_args = args
      ref_avals = self._all_args_info.in_avals
      debug_info = self._all_args_info.debug_info

    check_arg_avals_for_call(ref_avals, map(core.shaped_abstractify, kept_args),
                             debug_info)

    if not self._mut:
      arg_names = [n for i, n in enumerate(debug_info.arg_names)
                   if i in self._kept_var_idx]
      check_array_xla_sharding_layout_match(
          args_after_dce, self._in_shardings, self._xla_in_layouts, arg_names)
    else:
      args_after_dce = [*args_after_dce, *self._mut.in_mut]
      arg_names = debug_info.arg_names + ('',) * len(self._mut.in_mut)
      check_array_xla_sharding_layout_match(
          args_after_dce, self._in_shardings, self._xla_in_layouts, arg_names)
    return self.unsafe_call(*args)  # pylint: disable=not-callable

  def create_cpp_call(self, params: stages.CompiledCallParams):
    if not (isinstance(self.unsafe_call, ExecuteReplicated) and
            not self.unsafe_call.has_unordered_effects and
            not self.unsafe_call.has_host_callbacks):
      return None

    def aot_cache_miss(*args, **kwargs):
      # args do not include the const args
      # See https://docs.jax.dev/en/latest/internals/constants.html.
      outs, out_flat, args_flat = stages.Compiled.call(params, *args, **kwargs)
      out_flat, out_tree_dispatch = reflatten_outputs_for_dispatch(
          params.out_tree, out_flat)
      use_fastpath = (all(isinstance(x, xc.ArrayImpl) for x in out_flat)
                      and not self._mut)
      if jaxlib_extension_version < 366:
        use_fastpath = use_fastpath and not params.const_args

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
  return shard_args([sharding], [layout], [None], [x])[0]


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
        f"compiled. {num_mismatch_str} mismatches are:\n{str_errors}")


def _get_metadata_jit_pmap(local_devices, num_in_shardings, num_out_shardings):
  # Create replicated shardings for jit(pmap) path with local devices
  # because multihost jit(pmap) is not allowed.
  gs = sharding_impls.GSPMDSharding.get_replicated(local_devices)
  in_shardings = [gs] * num_in_shardings
  out_shardings = [gs] * num_out_shardings
  # jit(pmap) will generate Arrays with multi-device sharding.
  # It is unsupported for these shardings to be uncommitted, so force
  # the outputs to be committed.
  committed = True
  return (in_shardings, out_shardings, committed,
          _create_device_list(tuple(local_devices)))


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
  new_partitions = tuple_insert(spec, dim, val)  # type: ignore
  return PartitionSpec(*new_partitions)

def get_array_mapping(pspec: PartitionSpec) -> ArrayMappingOrAutoOrUnspecified:
  pspec = sharding_impls.prepare_axis_resources(pspec, "pspec to array_mapping")
  return _get_array_mapping(pspec)
