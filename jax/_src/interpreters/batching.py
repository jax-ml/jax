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
from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
from functools import partial
from typing import Any, Union, TypeAlias

import numpy as np

from jax._src import config
from jax._src import core
from jax._src.core import typeof
from jax._src import source_info_util
from jax._src import linear_util as lu
from jax._src.partition_spec import PartitionSpec as P
from jax._src.sharding_impls import NamedSharding
from jax._src import mesh as mesh_lib
from jax._src.ad_util import Zero, SymbolicZero, add_jaxvals, add_jaxvals_p
from jax._src.core import Trace, Tracer, TraceTag, AxisName
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (tree_unflatten, tree_flatten, PyTreeDef)
from jax._src.typing import Array
from jax._src.util import (unzip2, safe_map, safe_zip, split_list,
                           canonicalize_axis, moveaxis, as_hashable_function,
                           curry, memoize, weakref_lru_cache, tuple_insert)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

### vmappable typeclass

Vmappable = Any
Elt = Any
MapSpec = Any
AxisSize = Any
MeshAxis = Any
GetIdx = Callable[[], Tracer]  # TODO(mattjj): revise this laziness
ToEltHandler = Callable[[Callable, GetIdx, Vmappable, MapSpec], Elt]
FromEltHandler = Callable[[Callable, AxisSize, Elt, MapSpec], Vmappable]
MakeIotaHandler = Callable[[AxisSize], Array]

def to_elt(trace: Trace, get_idx: GetIdx, x: Vmappable, spec: MapSpec) -> Elt:
  handler = to_elt_handlers.get(type(x))
  if handler:
    return handler(partial(to_elt, trace, get_idx), get_idx, x, spec)
  elif isinstance(spec, int) or spec is None:
    spec = spec and canonicalize_axis(spec, len(np.shape(x)))
    return (BatchTracer(trace, x, spec, source_info_util.current())
            if spec is not None else x)
  else:
    # TODO(mvoz): This is a terrible place to fall into if you pass
    # a non jumble type in, make it clearer what went wrong.
    assert False, f'Unexpected type in ELT? {type(x)}'


to_elt_handlers: dict[type, ToEltHandler] = {}

def from_elt(trace: BatchTrace, axis_size: AxisSize, mesh_axis: MeshAxis,
             sum_match: bool, i: int, x: Elt, spec: MapSpec) -> tuple[Vmappable, MapSpec]:
  handler = from_elt_handlers.get(type(x))
  if handler:
    def _cont(axis_size, elt, axis):
      return from_elt(trace, axis_size, mesh_axis, sum_match, i, elt, axis)[0]
    return handler(_cont, axis_size, x, spec), spec
  val, bdim = trace.to_batch_info(x)
  bdim_inferred = bdim if spec is infer else spec
  try:
    return matchaxis(trace.axis_data.name, axis_size, mesh_axis,
                     bdim, spec, val, sum_match=sum_match), bdim_inferred
  except SpecMatchError:
    raise SpecMatchError(i, x.batch_dim, spec) from None
from_elt_handlers: dict[type, FromEltHandler] = {}

def make_iota(axis_size: AxisSize) -> Array:
  # Callers of this utility, via batch() or vtile(), must be in a context
  # where lax is importable.
  from jax import lax  # pytype: disable=import-error
  handler = make_iota_handlers.get(type(axis_size))
  if handler:
    return handler(axis_size)
  else:
    return lax.iota('int32', int(axis_size))
make_iota_handlers: dict[type, MakeIotaHandler] = {}

def register_vmappable(data_type: type, spec_type: type, axis_size_type: type,
                       to_elt: Callable, from_elt: Callable,
                       make_iota: Callable | None):
  vmappables[data_type] = (spec_type, axis_size_type)
  spec_types.add(spec_type)
  to_elt_handlers[data_type] = to_elt
  from_elt_handlers[data_type] = from_elt
  if make_iota: make_iota_handlers[axis_size_type] = make_iota
vmappables: dict[type, tuple[type, type]] = {}
spec_types: set[type] = set()

def unregister_vmappable(data_type: type) -> None:
  _, axis_size_type = vmappables.pop(data_type)
  del to_elt_handlers[data_type]
  del from_elt_handlers[data_type]
  if axis_size_type in make_iota_handlers:
    del make_iota_handlers[axis_size_type]
  global spec_types
  spec_types = (
      set() | {spec_type for spec_type, _ in vmappables.values()}
  )

def is_vmappable(x: Any) -> bool:
  return type(x) in vmappables

@lu.transformation_with_aux2
def flatten_fun_for_vmap(f: Callable,
                         store: lu.Store, in_tree: PyTreeDef, *args_flat):
  py_args, py_kwargs = tree_unflatten(in_tree, args_flat)
  ans = f(*py_args, **py_kwargs)
  ans, out_tree = tree_flatten(ans, is_leaf=is_vmappable)
  store.store(out_tree)
  return ans


### tracer

# TODO(mattjj): use a special sentinel type rather than None
NotMapped: TypeAlias = type(None)
not_mapped = None


class BatchTracer(Tracer):
  __slots__ = ['val', 'batch_dim', 'source_info']

  def __init__(self, trace, val, batch_dim: NotMapped | int,
               source_info: source_info_util.SourceInfo | None = None):
    if config.enable_checks.value:
      assert type(batch_dim) in (NotMapped, int)
      if type(batch_dim) is int:
        aval = core.get_aval(val)
        assert 0 <= batch_dim < len(aval.shape)
    self._trace = trace
    self.val = val
    self.batch_dim = batch_dim
    self.source_info = source_info

  def _short_repr(self):
    return f"VmapTracer(aval={self.aval}, batched={typeof(self.val)})"

  @property
  def aval(self):
    aval = core.get_aval(self.val)
    if self._trace.axis_data.spmd_name is not None:
      if config._check_vma.value:
        aval = aval.update(
            vma=aval.vma - frozenset(self._trace.axis_data.spmd_name))
    if self.batch_dim is not_mapped:
      return aval
    elif type(self.batch_dim) is int:
      return core.mapped_aval(aval.shape[self.batch_dim], self.batch_dim, aval)
    else:
      raise Exception("batch dim should be int or `not_mapped`")

  def full_lower(self):
    if self.batch_dim is not_mapped:
      return core.full_lower(self.val)
    else:
      return self

  def _origin_msg(self):
    if self.source_info is None:
      return ""
    return (f"\nThis BatchTracer with object id {id(self)} was created on line:"
            f"\n  {source_info_util.summarize(self.source_info)}")

  def _contents(self):
    return [('val', self.val), ('batch_dim', self.batch_dim)]

  def get_referent(self):
    if self.batch_dim is None or type(self.batch_dim) is int:
      return core.get_referent(self.val)
    else:
      return self

@dataclasses.dataclass(frozen=True)
class AxisData:
  name : Any
  size : Any
  # Only one of spmd_axis_name and explicit_mesh_axis is set.
  spmd_name : Any
  # short for private `_explicit_mesh_axis`. The public property is called
  # `.explicit_mesh_axis`
  _ema: tuple[Any, ...] | None

  @property
  def explicit_mesh_axis(self):
    assert self._ema is None or isinstance(self._ema, tuple)
    if self._ema is None:
      return None
    cur_mesh = mesh_lib.get_abstract_mesh()
    if cur_mesh.empty:
      return self._ema
    ema0_type = cur_mesh._name_to_type[self._ema[0]]
    assert all(cur_mesh._name_to_type[e] == ema0_type for e in self._ema)
    if ema0_type != mesh_lib.AxisType.Explicit:
      return None
    return self._ema

  def __repr__(self):
    return (f'AxisData(name={self.name}, size={self.size},'
            f' spmd_name={self.spmd_name},'
            f' explicit_mesh_axis={self.explicit_mesh_axis})')

  __str__ = __repr__


def get_sharding_for_vmap(axis_data, orig_sharding, axis):
  val = axis_data.explicit_mesh_axis
  # TODO(yashkatariya): Preserve unreduced here using
  # `orig_sharding.spec.update`
  new_spec = P(*tuple_insert(orig_sharding.spec, axis, val))
  return NamedSharding(orig_sharding.mesh, new_spec)


class BatchTrace(Trace):

  def __init__(self, parent_trace, tag, axis_data):
    super().__init__()
    self.parent_trace = parent_trace
    assert isinstance(axis_data, AxisData)
    self.axis_data = axis_data
    self.tag = tag
    self.requires_low = False

  def to_batch_info(self, val):
    if isinstance(val, BatchTracer) and val._trace.tag is self.tag:
      return val.val, val.batch_dim
    else:
      return val, not_mapped

  def process_primitive(self, p, tracers, params):
    vals_in, dims_in = unzip2(map(self.to_batch_info, tracers))
    args_not_mapped = all(bdim is not_mapped for bdim in dims_in)
    if p in fancy_primitive_batchers:
      if (args_not_mapped
          and p in skippable_batchers
          and not any(self.axis_data.name == axis_name
                      for axis_name in skippable_batchers[p](params))):
        return p.bind_with_trace(self.parent_trace, vals_in, params)
      else:
        with core.set_current_trace(self.parent_trace):
          val_out, dim_out = fancy_primitive_batchers[p](
              self.axis_data, vals_in, dims_in, **params)
    elif args_not_mapped:
      return p.bind_with_trace(self.parent_trace, vals_in, params)
    else:
      raise NotImplementedError(f"Batching rule for '{p}' not implemented")
    src = source_info_util.current()
    if p.multiple_results:
      with core.set_current_trace(self.parent_trace):  # val_out may be lazy map
        return [BatchTracer(self, x, d, src) if d is not not_mapped else x
                for x, d in zip(val_out, dim_out)]
    else:
      return (BatchTracer(self, val_out, dim_out, src)
              if dim_out is not not_mapped else val_out)

  def process_call(self, call_primitive, f, tracers, params):
    assert call_primitive.multiple_results
    params = dict(params, name=params.get('name', f.__name__))
    vals, dims = unzip2(map(self.to_batch_info, tracers))
    f_, dims_out = batch_subtrace(f, self.tag, self.axis_data, tuple(dims))

    with core.set_current_trace(self.parent_trace):
      vals_out = call_primitive.bind(f_, *vals, **params)
    src = source_info_util.current()
    return [BatchTracer(self, v, d, src) for v, d in zip(vals_out, dims_out())]

  def process_map(self, map_primitive, f: lu.WrappedFun, tracers, params):
    vals, dims = unzip2(map(self.to_batch_info, tracers))
    # The logic for the dimension math below is as follows:
    # ╔═════════════╦════════════════════════════════════════╦═══════════╗
    # ║ d / in_axis ║ None                                   ║ int       ║
    # ╠═════════════╬════════════════════════════════════════╩═══════════╣
    # ║ None        ║ No extra axis, so in_axis unaffected               ║
    # ╠═════════════╬════════════════════════════════════════╦═══════════╣
    # ║ int         ║ Not mapped, so batching dim unaffected ║ See below ║
    # ╚═════════════╩════════════════════════════════════════╩═══════════╝
    # When both d and in_axis are defined then:
    # - If `d <= in_axis`, we have to move the `in_axis` one dimension further;
    # - If `d >  in_axis`, we have to decrement `d` (as `in_axis` will get removed).
    def both_mapped(in_out_axis, d):
      return in_out_axis is not None and d is not not_mapped
    new_in_axes = tuple(
      in_axis + 1 if both_mapped(in_axis, d) and d <= in_axis else in_axis
      for d, in_axis in zip(dims, params['in_axes']))
    new_dims = tuple(
      d - 1 if both_mapped(in_axis, d) and in_axis < d else d
      for d, in_axis in zip(dims, params['in_axes']))
    f, dims_out = batch_subtrace(f, self.tag, self.axis_data, new_dims)
    out_axes_thunk = params['out_axes_thunk']
    # NOTE: This assumes that the choice of the dimensions over which outputs
    #       are batched is entirely dependent on the function and not e.g. on the
    #       data or its shapes.
    @as_hashable_function(closure=out_axes_thunk)
    def new_out_axes_thunk():
      return tuple(out_axis + 1 if both_mapped(out_axis, d) and d < out_axis else out_axis
                    for out_axis, d in zip(out_axes_thunk(), dims_out()))
    new_params = dict(params, in_axes=new_in_axes, out_axes_thunk=new_out_axes_thunk)
    with core.set_current_trace(self.parent_trace):
      vals_out = map_primitive.bind(f, *vals, **new_params)
    dims_out_ = [d + 1 if both_mapped(out_axis, d) and out_axis <= d else d
                  for d, out_axis in zip(dims_out(), out_axes_thunk())]
    src = source_info_util.current()
    return [BatchTracer(self, v, d, src) for v, d in zip(vals_out, dims_out_)]

  def process_custom_jvp_call(self, prim, fun, jvp, tracers, *, symbolic_zeros):
    in_vals, in_dims = unzip2(map(self.to_batch_info, tracers))
    fun, out_dims1 = batch_subtrace(fun, self.tag, self.axis_data, in_dims)
    jvp, out_dims2 = batch_custom_jvp_subtrace(jvp, self.tag, self.axis_data, in_dims)
    out_vals = prim.bind_with_trace(self.parent_trace, (fun, jvp, *in_vals),
                                    dict(symbolic_zeros=symbolic_zeros))
    fst, out_dims = lu.merge_linear_aux(out_dims1, out_dims2)
    src = source_info_util.current()
    return [BatchTracer(self, v, d, src) for v, d in zip(out_vals, out_dims)]

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, *, out_trees,
                              symbolic_zeros):  # pytype: disable=signature-mismatch
    in_vals, in_dims = unzip2(map(self.to_batch_info, tracers))
    fwd_in_dims = [d for in_dim in in_dims for d in [in_dim, not_mapped]]

    fun, out_dims1 = batch_subtrace(fun, self.tag, self.axis_data, in_dims)
    fwd, out_dims2 = batch_subtrace(fwd, self.tag, self.axis_data, fwd_in_dims)

    def bwd_in_dims():
      _, _, input_fwds = out_trees()
      pruned_dims = iter(out_dims2())
      full_dims = [next(pruned_dims) if f is None else in_dims[f] for f in input_fwds]
      return [*full_dims, *pruned_dims]

    bwd = batch_custom_vjp_bwd(bwd, self.tag, self.axis_data, bwd_in_dims, in_dims)
    out_vals = prim.bind_with_trace(self.parent_trace,
                                    (fun, fwd, bwd) + tuple(in_vals),
                                    dict(out_trees=out_trees, symbolic_zeros=symbolic_zeros))
    fst, out_dims = lu.merge_linear_aux(out_dims1, out_dims2)
    if not fst:
      _, res_tree, input_fwds = out_trees()
      num_res = res_tree.num_leaves - sum(f is not None for f in input_fwds)
      _, out_dims = split_list(out_dims, [num_res])
    src = source_info_util.current()
    return [BatchTracer(self, v, d, src) for v, d in zip(out_vals, out_dims)]

### API for batching callables with vmappable inputs and outputs

def batch(fun: lu.WrappedFun, axis_data,
          in_dims, out_dim_dests, sum_match=False) -> lu.WrappedFun:
  # we split up _batch_inner and _batch_outer for the leak checker
  f = _batch_inner(fun, axis_data, out_dim_dests, sum_match)
  return _batch_outer(f, axis_data, in_dims)

@lu.transformation2
def _batch_outer(f, axis_data, in_dims, *in_vals):
  tag = TraceTag()
  with source_info_util.transform_name_stack('vmap'):
    outs, out_dim_srcs, trace = f(tag, in_dims, *in_vals)
  with core.ensure_no_leaks(trace): del trace
  return outs, out_dim_srcs

@lu.transformation2
def _batch_inner(f: Callable, axis_data, out_dim_dests, sum_match, tag, in_dims, *in_vals):
  in_dims = in_dims() if callable(in_dims) else in_dims
  with core.take_current_trace() as parent_trace:
    trace = BatchTrace(parent_trace, tag, axis_data)
    idx = memoize(lambda: BatchTracer(trace, make_iota(axis_data.size), 0,
                                      source_info_util.current()))
    with core.set_current_trace(parent_trace):
      in_tracers = map(partial(to_elt, trace, idx), in_vals, in_dims)
    # TODO(yashkatariya): Instead of `add_explicit_mesh_axis_names`, we should
    # create a new mesh by removing the axis_data.explicit_mesh_axis from it.
    with (core.set_current_trace(trace),
          core.extend_axis_env_nd([(axis_data.name, axis_data.size)]),
          core.add_spmd_axis_names(axis_data.spmd_name),
          core.add_explicit_mesh_axis_names(axis_data.explicit_mesh_axis)):
      outs = f(*in_tracers)
      out_dim_dests = out_dim_dests() if callable(out_dim_dests) else out_dim_dests
      out_vals, out_dim_srcs = unzip2(
          map(partial(from_elt, trace, axis_data.size, axis_data.explicit_mesh_axis, sum_match),
              range(len(outs)), outs, out_dim_dests))
  return out_vals, out_dim_srcs, trace

# NOTE: This divides the in_axes by the tile_size and multiplies the out_axes by it.
def vtile(f_flat: lu.WrappedFun,
          in_axes_flat: tuple[int | None, ...],
          out_axes_flat: tuple[int | None, ...],
          tile_size: int | None,
          axis_name: AxisName):
  @curry
  def tile_axis(arg, axis: int | None, tile_size):
    if axis is None:
      return arg
    shape = list(arg.shape)
    shape[axis:axis+1] = [tile_size, shape[axis] // tile_size]
    return arg.reshape(shape)

  def untile_axis(out, axis: int | None):
    if axis is None:
      return out
    shape = list(out.shape)
    shape[axis:axis+2] = [shape[axis] * shape[axis+1]]
    return out.reshape(shape)

  @lu.transformation2
  def _map_to_tile(f, *args_flat):
    sizes = (x.shape[i] for x, i in safe_zip(args_flat, in_axes_flat) if i is not None)
    tile_size_ = tile_size or next(sizes, None)
    assert tile_size_ is not None, "No mapped arguments?"
    outputs_flat = f(*map(tile_axis(tile_size=tile_size_), args_flat, in_axes_flat))
    return map(untile_axis, outputs_flat, out_axes_flat)

  axis_data = AxisData(axis_name, tile_size, None, None)
  return _map_to_tile(batch(f_flat, axis_data, in_axes_flat, out_axes_flat))

### API for batching functions with jaxpr type inputs and outputs

@lu.transformation_with_aux2
def batch_subtrace(f, store, tag, axis_data, in_dims, *in_vals):
  with core.take_current_trace() as parent_trace:
    trace = BatchTrace(parent_trace, tag, axis_data)
    with core.set_current_trace(trace):
      in_dims = in_dims() if callable(in_dims) else in_dims
      in_tracers = [BatchTracer(trace, x, dim, source_info_util.current())
                    if dim is not None else x for x, dim in zip(in_vals, in_dims)]
      outs = f(*in_tracers)
    out_vals, out_dims = unzip2(map(trace.to_batch_info, outs))
  store.store(out_dims)
  return out_vals

### API for batching jaxprs

def batch_jaxpr2(
    closed_jaxpr: core.ClosedJaxpr,
    axis_data,
    in_axes: tuple[int | NotMapped, ...],
  ) -> tuple[core.ClosedJaxpr, tuple[int | NotMapped ]]:
  return _batch_jaxpr2(closed_jaxpr, axis_data, tuple(in_axes))

@weakref_lru_cache
def _batch_jaxpr2(
    closed_jaxpr: core.ClosedJaxpr,
    axis_data,
    in_axes: tuple[int | NotMapped ],
  ) -> tuple[core.ClosedJaxpr, tuple[int | NotMapped, ...]]:
  f = lu.wrap_init(core.jaxpr_as_fun(closed_jaxpr),
                   debug_info=closed_jaxpr.jaxpr.debug_info)
  f, out_axes = _batch_jaxpr_inner(f, axis_data)
  f = _batch_jaxpr_outer(f, axis_data, in_axes)
  avals_in2 = []
  for aval, b in unsafe_zip(closed_jaxpr.in_avals, in_axes):
    if b is not_mapped:
      avals_in2.append(aval)
    else:
      aval = core.unmapped_aval(
          axis_data.size, b, aval, axis_data.explicit_mesh_axis)
      if axis_data.spmd_name is not None:
        if config._check_vma.value:
          aval = aval.update(vma=aval.vma | frozenset(axis_data.spmd_name))  # type: ignore
      avals_in2.append(aval)
  jaxpr_out, _, consts = pe.trace_to_jaxpr_dynamic(f, avals_in2)
  return core.ClosedJaxpr(jaxpr_out, consts), out_axes()

def batch_jaxpr(closed_jaxpr, axis_data, in_batched, instantiate):
  inst = tuple(instantiate) if isinstance(instantiate, list) else instantiate
  return _batch_jaxpr(closed_jaxpr, axis_data, tuple(in_batched), inst)

def _batch_jaxpr(closed_jaxpr, axis_data, in_batched, instantiate):
  assert (isinstance(instantiate, bool) or
          isinstance(instantiate, (list, tuple)) and
          all(isinstance(b, bool) for b in instantiate))
  if isinstance(instantiate, bool):
    instantiate = [instantiate] * len(closed_jaxpr.out_avals)
  in_axes = [0 if b else not_mapped for b in in_batched]
  out_axes_dest = [0 if inst else zero_if_mapped for inst in instantiate]
  return batch_jaxpr_axes(closed_jaxpr, axis_data, in_axes, out_axes_dest)

def batch_jaxpr_axes(closed_jaxpr, axis_data, in_axes, out_axes_dest):
  return _batch_jaxpr_axes(closed_jaxpr, axis_data, tuple(in_axes), tuple(out_axes_dest))

@weakref_lru_cache
def _batch_jaxpr_axes(closed_jaxpr: core.ClosedJaxpr,
                      axis_data: AxisData,
                      in_axes: Sequence[int], out_axes_dest: Sequence[int]):
  f = lu.wrap_init(core.jaxpr_as_fun(closed_jaxpr),
                   debug_info=closed_jaxpr.jaxpr.debug_info)
  f, out_axes = _batch_jaxpr_inner(f, axis_data)
  f, out_batched = _match_axes_jaxpr(f, axis_data, out_axes_dest, out_axes)
  f = _batch_jaxpr_outer(f, axis_data, in_axes)
  avals_in = [core.unmapped_aval(axis_data.size, b, aval,
                                 axis_data.explicit_mesh_axis)
              if b is not not_mapped
              else aval for aval, b in unsafe_zip(closed_jaxpr.in_avals, in_axes)]
  jaxpr_out, _, consts = pe.trace_to_jaxpr_dynamic(f, avals_in)
  return core.ClosedJaxpr(jaxpr_out, consts), out_batched()

@lu.transformation_with_aux2
def _batch_jaxpr_inner(f, store, axis_data, tag, in_axes, *in_vals):
  with core.take_current_trace() as parent_trace:
    trace = BatchTrace(parent_trace, tag, axis_data)
    in_tracers = [BatchTracer(trace, val, dim) if dim is not None else val
                  for val, dim in zip(in_vals, in_axes)]
    # TODO(yashkatariya): Instead of `add_explicit_mesh_axis_names`, we should
    # create a new mesh by removing the axis_data.explicit_mesh_axis from it.
    with (core.set_current_trace(trace),
          core.extend_axis_env_nd([(axis_data.name, axis_data.size)]),
          core.add_spmd_axis_names(axis_data.spmd_name),
          core.add_explicit_mesh_axis_names(axis_data.explicit_mesh_axis)):
      outs = f(*in_tracers)
    out_vals, out_axes = unzip2(map(trace.to_batch_info, outs))
  store.store(out_axes)
  return out_vals

@lu.transformation_with_aux2
def _match_axes_jaxpr(f, store, axis_data, out_axes_dest, out_axes, trace, in_axes,
                      *in_vals):
  out_vals = f(trace, in_axes, *in_vals)
  out_axes = out_axes()
  out_axes_dest = [(None if src is not_mapped else 0)
                   if dst is zero_if_mapped else dst
                   for src, dst in unsafe_zip(out_axes, out_axes_dest)]
  if len(out_axes_dest) != len(out_axes):
    out_axis_dest, = out_axes_dest
    out_axes_dest = [out_axis_dest] * len(out_axes)
  out_vals = map(partial(matchaxis, axis_data.name, axis_data.size,
                         axis_data.explicit_mesh_axis),
                 out_axes, out_axes_dest, out_vals)
  out_batched = [dst is not None for dst in out_axes_dest]
  store.store(out_batched)
  return out_vals

@lu.transformation2
def _batch_jaxpr_outer(f, axis_data, in_dims, *in_vals):
  in_dims = in_dims() if callable(in_dims) else in_dims
  in_dims = [canonicalize_axis(ax, np.ndim(x)) if isinstance(ax, int)
             else ax for x, ax in unsafe_zip(in_vals, in_dims)]
  tag = TraceTag()
  return f(tag, in_dims, *in_vals)

def _merge_bdims(x, y):
  if x == y:
    return x
  elif x is not_mapped:
    return y
  elif y is not_mapped:
    return x
  else:
    return x  # arbitrary

class ZeroIfMapped: pass
zero_if_mapped = ZeroIfMapped()

### functions for handling custom_vjp

@lu.transformation_with_aux2
def batch_custom_jvp_subtrace(f, store, tag, axis_data, in_dims, *in_vals):
  size = axis_data.size
  mesh_axis = axis_data.explicit_mesh_axis
  with core.take_current_trace() as parent_trace:
    trace = BatchTrace(parent_trace, tag, axis_data)
    in_tracers = [val if dim is None else
                  SymbolicZero(core.mapped_aval(size, dim, val.aval))
                  if type(val) is SymbolicZero else BatchTracer(trace, val, dim)
                  for val, dim in zip(in_vals, in_dims * 2)]
    with core.set_current_trace(trace):
      out_tracers: list[BatchTracer | SymbolicZero] = f(*in_tracers)
  out_vals, out_dims = unzip2(map(trace.to_batch_info, out_tracers))
  out_primals, out_tangents = split_list(out_vals, [len(out_vals) // 2])
  out_primal_bds, out_tangent_bds = split_list(out_dims, [len(out_vals) // 2])
  out_dims = map(_merge_bdims, out_primal_bds, out_tangent_bds)
  out_primals  = map(partial(matchaxis, trace.axis_data.name, size, mesh_axis),
                     out_primal_bds, out_dims,  out_primals)
  out_tangents = map(partial(_matchaxis_symzeros, trace.axis_data.name, size, mesh_axis),
                     out_tangent_bds, out_dims, out_tangents)
  store.store(out_dims)
  return out_primals + out_tangents

def batch_custom_vjp_bwd(bwd: lu.WrappedFun, tag: core.TraceTag,
                         axis_data: AxisData,
                         in_dims: Callable[[], Sequence[int | None]],
                         out_dim_dests: Sequence[int | None]) -> lu.WrappedFun:
  axis_size = axis_data.size
  axis_name = axis_data.name
  mesh_axis = axis_data.explicit_mesh_axis
  def new_bwd(*args):
    in_dims_ = in_dims() if callable(in_dims) else in_dims
    args = [SymbolicZero(core.mapped_aval(axis_size, dim, x.aval))
            if type(x) is SymbolicZero else x
            for x, dim in zip(args, in_dims_)]
    in_dims_ = [None if type(x) is SymbolicZero else d
                for x, d in zip(args, in_dims_)]
    bwd_, out_dims_thunk = batch_subtrace(bwd, tag, axis_data, in_dims_)
    bwd_ = _match_axes_and_sum(bwd_, axis_size, axis_name, mesh_axis,
                               out_dims_thunk, out_dim_dests)
    return bwd_.call_wrapped(*args)
  return lu.wrap_init(new_bwd, debug_info=bwd.debug_info)

@lu.transformation2
def _match_axes_and_sum(f, axis_size, axis_name, mesh_axis, out_dims_thunk,
                        out_dim_dests, *in_vals):
  # this is like _match_axes, but we do reduce-sums as needed
  out_vals = f(*in_vals)
  return map(partial(_matchaxis_symzeros, axis_name, axis_size, mesh_axis,
                     sum_match=True),
             out_dims_thunk(), out_dim_dests, out_vals)

def _matchaxis_symzeros(axis_name, sz, mesh_axis, src, dst, x, sum_match=False):
  # Just like `matchaxis`, but handles symbolic zeros using ad_util.py
  # TODO(mattjj): dedup with matchaxis
  if isinstance(x, (Zero, SymbolicZero)):
    if src == dst:
      return x
    elif type(src) == type(dst) == int:
      aval = core.mapped_aval(sz, src, x.aval)
      return type(x)(core.unmapped_aval(sz, dst, aval, mesh_axis))
    elif src is not_mapped and dst is not not_mapped:
      return type(x)(core.unmapped_aval(sz, dst, x.aval, mesh_axis))
    elif dst is not_mapped and sum_match:
      return type(x)(core.mapped_aval(sz, src, x.aval))
    else:
      raise ValueError((axis_name, x, src, dst))
  else:
    return matchaxis(axis_name, sz, mesh_axis, src, dst, x, sum_match=sum_match)


### utilities for defining primitives' batching rules

BatchingRule = Callable[
    ...,
    tuple[Any, Union[int, None, tuple[Union[int, None], ...]]]
]
fancy_primitive_batchers: dict[core.Primitive, Callable] = {}

# backwards compat shim. TODO: delete
class AxisPrimitiveBatchersProxy:
  def __setitem__(self, prim, batcher):
    def wrapped(axis_data, vals, dims, **params):
      return batcher(axis_data.size, axis_data.name, None, vals, dims, **params)
    fancy_primitive_batchers[prim] = wrapped
axis_primitive_batchers = AxisPrimitiveBatchersProxy()

# backwards compat shim. TODO: delete
class PrimitiveBatchersProxy:
  def __setitem__(self, prim, batcher):
    def wrapped(axis_data, vals, dims, **params):
      del axis_data
      if all(d is None for d in dims):
        o = prim.bind(*vals, **params)
        return (o, [None] * len(o)) if prim.multiple_results else (o, None)
      return batcher(vals, dims, **params)
    fancy_primitive_batchers[prim] = wrapped

  def __delitem__(self, prim):
    del fancy_primitive_batchers[prim]
primitive_batchers = PrimitiveBatchersProxy()


# Presence in this table allows fancy batchers to be skipped by batch traces for
# irrelevant axes. The Callable takes params and returns a list of relevant axes
# TODO(yashkatariya): remove this
skippable_batchers : dict[core.Primitive, Callable] = {}

def defvectorized(prim):
  fancy_primitive_batchers[prim] = partial(vectorized_batcher, prim)

def vectorized_batcher(prim, axis_data, batched_args, batch_dims, **params):
  assert not prim.multiple_results
  if all(d is None for d in batch_dims):
    return prim.bind(*batched_args, **params), None
  assert all(batch_dims[0] == bd for bd in batch_dims[1:]), batch_dims
  return prim.bind(*batched_args, **params), batch_dims[0]

def defbroadcasting(prim):
  fancy_primitive_batchers[prim] = partial(broadcast_batcher, prim)

def broadcast_batcher(prim, axis_data, args, dims, **params):
  assert len(args) > 1
  if all(d is None for d in dims):
    o = prim.bind(*args, **params)
    return (o, [None] * len(o)) if prim.multiple_results else (o, None)
  shape, dim = next((x.shape, d) for x, d in zip(args, dims)
                    if d is not not_mapped)
  if all(core.definitely_equal_shape(shape, x.shape) and d == dim
         for x, d in zip(args, dims) if np.ndim(x)):
    # if there's only agreeing batch dims and scalars, just call the primitive
    out = prim.bind(*args, **params)
    return (out, (dim,) * len(out)) if prim.multiple_results else (out, dim)
  else:
    # We pass size of 1 here because (1) at least one argument has a real batch
    # dimension and (2) all unmapped axes can have a singleton axis inserted and
    # then rely on the primitive's built-in broadcasting.
    args = [bdim_at_front(x, d, 1) if np.ndim(x) else x
            for x, d in zip(args, dims)]
    ndim = max(np.ndim(x) for x in args)  # special-case scalar broadcasting
    args = [_handle_scalar_broadcasting(ndim, x, d) for x, d in zip(args, dims)]
    out = prim.bind(*args, **params)
    return (out, (0,) * len(out)) if prim.multiple_results else (out, 0)

def _handle_scalar_broadcasting(nd, x, d):
  # Callers of this utility, via broadcast_batcher() or defbroadcasting(),
  # must be in a context where lax is importable.
  from jax import lax  # pytype: disable=import-error
  if d is not_mapped or nd == np.ndim(x):
    return x
  else:
    return lax.expand_dims(x, tuple(range(np.ndim(x), nd)))

def defreducer(prim, ident):
  fancy_primitive_batchers[prim] = partial(reducer_batcher, prim, ident)

def reducer_batcher(prim, ident, axis_data, batched_args, batch_dims, axes,
                    **params):
  if all(d is None for d in batch_dims):
    return prim.bind(*batched_args, axes=axes, **params), None
  def out_axis(axes, axis):
    return int(list(np.delete(np.arange(operand.ndim), axes)).index(axis))
  operand, = batched_args
  bdim, = batch_dims
  if isinstance(bdim, int):
    axes = tuple(np.where(np.less(axes, bdim), axes, np.add(axes, 1)))
    bdim_out = out_axis(axes, bdim)
    if 'input_shape' in params:
      params = dict(params, input_shape=operand.shape)
    return prim.bind(operand, axes=axes, **params), bdim_out
  else:
    assert False

def expand_dims_batcher(prim, args, dims, **params):
  """A batching rule for primitives that support matching leading batch
  dimensions in all arguments.
  """
  size, = {x.shape[bd] for x, bd in zip(args, dims) if bd is not not_mapped}
  args = [bdim_at_front(x, bd, size) for x, bd in zip(args, dims)]
  out = prim.bind(*args, **params)
  return (out, (0,) * len(out)) if prim.multiple_results else (out, 0)

### general utilities for manipulating axes on jaxpr types (not vmappables)

def broadcast(x, sz, axis, mesh_axis):
  # Callers of this utility must be in a context where lax is importable.
  from jax import lax  # pytype: disable=import-error
  shape = list(np.shape(x))
  shape.insert(axis, sz)
  broadcast_dims = tuple(np.delete(np.arange(len(shape)), axis))
  x_aval = core.get_aval(x)
  if x_aval.sharding.mesh.empty:
    mesh_axis = None
  new_spec = P(*tuple_insert(x_aval.sharding.spec, axis, mesh_axis))
  sharding = x_aval.sharding.update(spec=new_spec)
  # TODO(dougalm, yashkatariya): Delete this context manager once we figure
  # out how to ensure jaxpr arguments always have the context mesh.
  with mesh_lib.use_abstract_mesh(sharding.mesh):
    x = lax.broadcast_in_dim(x, shape, broadcast_dims, out_sharding=sharding)
    if config._check_vma.value:
      # TODO(yashkatariya,parkers): don't do this, fix during fixit week 2026
      spmd_names = core.get_axis_env().spmd_axis_names
      if len(spmd_names) > 1:
        raise NotImplementedError
      if spmd_names:
        x = core.pvary(x, tuple(spmd_names))
    return x

def matchaxis2(axis_data, src, dst, x, sum_match=False):
  return matchaxis(axis_data.name, axis_data.size, axis_data.explicit_mesh_axis,
                   src, dst, x, sum_match)

def matchaxis(axis_name, sz, mesh_axis, src, dst, x, sum_match=False):
  try:
    _ = core.get_aval(x)
  except TypeError as e:
    raise TypeError(f"Output from batched function {x!r} with type "
                    f"{type(x)} is not a valid JAX type") from e
  if src == dst or dst is infer:
    return x
  elif type(src) == type(dst) == int:
    return moveaxis(x, src, dst)
  elif src is not_mapped and type(dst) is int:
    return broadcast(x, sz, canonicalize_axis(dst, np.ndim(x) + 1), mesh_axis)
  elif src is not_mapped and dst is sum_axis:
    return x
  elif dst is not_mapped and sum_match or dst is sum_axis:
    return x.sum(src)
  else:
    if (not isinstance(axis_name, core._TempAxisName) and
        axis_name is not core.no_axis_name):
      raise ValueError(f'vmap has mapped output ({axis_name=}) but out_axes is {dst}')
    else:
      raise SpecMatchError(None, None, None)

class SpecMatchError(Exception):
  def __init__(self, leaf_idx, src, dst):
    self.leaf_idx = leaf_idx
    self.src = src
    self.dst = dst

def bdim_at_front(x, bdim, size, mesh_axis=None):
  if bdim is not_mapped:
    return broadcast(x, size, 0, mesh_axis=mesh_axis)
  else:
    return moveaxis(x, bdim, 0)


def add_batched(axis_data, batched_args, batch_dims):
  bdx, bdy = batch_dims
  x, y = batched_args
  mesh_axis = axis_data.explicit_mesh_axis
  if bdx == bdy:
    return add_jaxvals(x, y), bdx
  elif bdx is not_mapped:
    x = broadcast(x, y.shape[bdy], bdy, mesh_axis=mesh_axis)
    return add_jaxvals(x, y), bdy
  elif bdy is not_mapped:
    y = broadcast(y, x.shape[bdx], bdx, mesh_axis=mesh_axis)
    return add_jaxvals(x, y), bdx
  else:
    x = moveaxis(x, bdx, bdy)
    return add_jaxvals(x, y), bdy

fancy_primitive_batchers[add_jaxvals_p] = add_batched
skippable_batchers[add_jaxvals_p] = lambda _: ()

### mutable arrays

defvectorized(core.ref_p)

### hijax

class Sum: pass
sum_axis = Sum()
spec_types.add(Sum)

class Infer: pass
infer = Infer()
spec_types.add(Infer)
