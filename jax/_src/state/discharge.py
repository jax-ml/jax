# Copyright 2022 The JAX Authors.
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
"""Module for discharging state primitives."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
from functools import partial
import math
import operator
from typing import Any, Protocol, TypeVar

from jax._src import ad_util
from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import pjit
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import tree_util
from jax._src import custom_derivatives
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax
from jax._src.lax import slicing as lax_slicing
from jax._src.state import indexing
from jax._src.state.primitives import addupdate_p, get_p, swap_p
from jax._src.state.types import (
    AbstractRef,
    RefBitcaster,
    RefEffect,
    RefReshaper,
    get_ref_aval_from_value,
    uninitialized,
)
from jax._src.state.utils import bitcast, hoist_consts_to_refs
from jax._src.typing import Array
from jax._src.util import (foreach, safe_map, safe_zip, split_list, unzip2,
                           weakref_lru_cache)
import numpy as np

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip
PyTreeDef = tree_util.PyTreeDef

## Discharging state

# Let's say we have a jaxpr that takes in `Ref`s and outputs regular JAX values
# (`Ref`s should never be outputs from jaxprs). We'd like to convert that jaxpr
# into a "pure" jaxpr that takes in and outputs values and no longer has the
# `Read/Write/Accum` effects.

def discharge_state(jaxpr: core.Jaxpr, consts: Sequence[Any], * ,
                    should_discharge: bool | Sequence[bool] = True,
                    ) -> tuple[core.Jaxpr, list[Any]]:
  """Converts a jaxpr that takes in `Ref`s into one that doesn't."""
  if isinstance(should_discharge, bool):
    should_discharge = [should_discharge] * len(jaxpr.invars)
  in_avals = [v.aval.inner_aval
              if isinstance(v.aval, AbstractRef) and d
              else v.aval for v, d in zip(jaxpr.invars, should_discharge)]
  eval_jaxpr = lu.wrap_init(partial(_eval_jaxpr_discharge_state, jaxpr,
                                    should_discharge, consts),
                            debug_info=jaxpr.debug_info)
  new_jaxpr, _ , new_consts = pe.trace_to_jaxpr_dynamic(eval_jaxpr, in_avals)
  return new_jaxpr, new_consts

# TODO(mattjj): migrate callers to discharge_state2 for caching
def discharge_state2(jaxpr: core.ClosedJaxpr,
                     should_discharge: bool | Sequence[bool] = True,
                     ) -> core.ClosedJaxpr:
  if isinstance(should_discharge, bool):
    should_discharge = (should_discharge,) * len(jaxpr.in_avals)
  return _discharge_state2(jaxpr, tuple(should_discharge))

@weakref_lru_cache
def _discharge_state2(jaxpr: core.ClosedJaxpr,
                      should_discharge: tuple[bool, ...],
                      ) -> core.ClosedJaxpr:
  jaxpr_, consts = discharge_state(jaxpr.jaxpr, jaxpr.consts,
                                   should_discharge=should_discharge)
  return core.ClosedJaxpr(jaxpr_, consts)

@dataclasses.dataclass
class Environment:
  env: dict[core.Var, Any]

  def read(self, v: core.Atom) -> Any:
    if type(v) is core.Literal:
      return v.val
    assert isinstance(v, core.Var)
    return self.env[v]

  def write(self, v: core.Var, val: Any) -> None:
    self.env[v] = val

class DischargeRule(Protocol):

  def __call__(self, in_avals: Sequence[core.AbstractValue],
      out_avals: Sequence[core.AbstractValue], *args: Any,
      **params: Any) -> tuple[Sequence[Any | None], Sequence[Any]]:
    ...

_discharge_rules: dict[core.Primitive, DischargeRule] = {}

class PartialDischargeRule(Protocol):
  """A partial discharge rule.

  Exactly like a discharge rule only it accepts a `should_discharge`
  argument that indicates which inputs should be discharged and the
  return value returns a tuple of which the first element is the new
  inputs or none but only the ones that correspond to `True` entries
  in `should_charge`.
  """

  def __call__(self, should_discharge: Sequence[bool],
      in_avals: Sequence[core.AbstractValue],
      out_avals: Sequence[core.AbstractValue], *args: Any,
      **params: Any) -> tuple[Sequence[Any | None], Sequence[Any]]:
    ...

_partial_discharge_rules: dict[core.Primitive, PartialDischargeRule] = {}

def register_discharge_rule(prim: core.Primitive):
  def register(f: DischargeRule):
    _discharge_rules[prim] = f
  return register

def register_partial_discharge_rule(prim: core.Primitive):
  def register(f: PartialDischargeRule):
    _partial_discharge_rules[prim] = f
  return register


def _eval_jaxpr_discharge_state(
    jaxpr: core.Jaxpr, should_discharge: Sequence[bool], consts: Sequence[Any],
    *args: Any):
  env = Environment({})

  foreach(env.write, jaxpr.constvars, consts)
  # Here some args may correspond to `Ref` avals but they'll be treated like
  # regular values in this interpreter.
  foreach(env.write, jaxpr.invars, args)

  refs_to_discharge = {id(v.aval) for v, d in zip(jaxpr.invars, should_discharge)
                       if d and isinstance(v.aval, AbstractRef)}

  for eqn in jaxpr.eqns:
    name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
    traceback = eqn.source_info.traceback
    with source_info_util.user_context(
        traceback, name_stack=name_stack), eqn.ctx.manager:
      should_discharge = [id(v.aval) in refs_to_discharge for v in eqn.invars]
      if eqn.primitive is core.mutable_array_p:
        [invar], [outvar] = eqn.invars, eqn.outvars
        ans = env.read(invar)
        refs_to_discharge.add(id(outvar.aval))
      elif eqn.primitive is core.freeze_p:
        [invar], [outvar] = eqn.invars, eqn.outvars
        ans = env.read(invar)
        refs_to_discharge.remove(id(invar.aval))
      elif any(should_discharge) or core.internal_mutable_array_effect in eqn.effects:
        if eqn.primitive in _partial_discharge_rules:
          rule: DischargeRule = partial(_partial_discharge_rules[eqn.primitive], should_discharge)
        elif eqn.primitive in _discharge_rules:
          rule = _discharge_rules[eqn.primitive]
        else:
          raise NotImplementedError(
              f"No state discharge rule implemented for primitive: {eqn.primitive}")
        invals = map(env.read, eqn.invars)
        in_avals = [v.aval for v in eqn.invars]
        out_avals = [v.aval for v in eqn.outvars]
        new_invals, ans = rule(
            in_avals, out_avals, *invals, **eqn.params)
        for invar, should, new_inval in zip(eqn.invars, should_discharge, new_invals):
          if new_inval is not None:
            if not should:
              raise ValueError(
                  f"Did not ask for inval to be discharged but it was. ({invar=},"
                  f" {new_inval=})"
              )
            env.write(invar, new_inval)  # type: ignore[arg-type]
      else:
        # Default primitive rule, similar to `core.eval_jaxpr`. Note that here
        # we assume any higher-order primitives inside of the jaxpr are *not*
        # stateful.
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        ans = eqn.primitive.bind(*subfuns, *map(env.read, eqn.invars),
                                **bind_params)
    if eqn.primitive.multiple_results:
      foreach(env.write, eqn.outvars, ans)
    else:
      env.write(eqn.outvars[0], ans)
  # By convention, we return the outputs of the jaxpr first and then the final
  # values of the `Ref`s. Callers to this function should be able to split
  # them up by looking at `len(jaxpr.outvars)`.
  out_vals = map(env.read, jaxpr.outvars)
  ref_vals = map(
      env.read, [v for v in jaxpr.invars if id(v.aval) in refs_to_discharge])
  return out_vals + ref_vals

def _is_trivial_indexer(indexer: indexing.NDIndexer):
  """Returns whether the indexer selects the entire shape."""
  for s, idx in zip(indexer.shape, indexer.indices):
    if not isinstance(idx, indexing.Slice):
      return False
    if idx.is_dynamic_start or idx.is_dynamic_size:
      return False
    if idx.start != 0 or idx.size != s:
      return False
  return True


def _maybe_convert_to_slice(
    indexer: indexing.NDIndexer
) -> list[tuple[int, int, int]] | None:
  args = []

  for i in indexer.indices:
    if not isinstance(i, indexing.Slice):
      return None

    start = i.start
    end = i.start + (i.size - 1) * i.stride + 1
    stride = i.stride

    # cannot convert to static `slice` if `start` or `end` is dynamic
    if not isinstance(start, int) or not isinstance(end, int):
      return None

    args.append((start, end, stride))

  return args


def _maybe_convert_to_dynamic_slice(
    indexer: indexing.NDIndexer,
) -> (
    tuple[tuple[Array | int, ...], tuple[Array | int, ...], tuple[int, ...]]
    | None
):
  # An NDIndexer only corresponds to a `dynamic_slice` or `dynamic_update_slice`
  # if each of the indexers is a `Slice` or a ()-shaped value.
  if not all(isinstance(i, indexing.Slice) or not np.shape(i)
             for i in indexer.indices):
    return None

  # `lax.dynamic_slice` does not handle striding
  for i in indexer.indices:
    if isinstance(i, indexing.Slice) and i.stride > 1:
      return None

  _convert_i32 = lambda x: lax.convert_element_type(x, np.dtype("int32"))
  starts = tuple(
      _convert_i32(i.start) if isinstance(i, indexing.Slice)
      else _convert_i32(i) for i in indexer.indices
  )
  sizes = tuple(
      i.size if isinstance(i, indexing.Slice) else 1 for i in indexer.indices
  )
  squeeze_dims = tuple(
      i
      for i, idx in enumerate(indexer.indices)
      if not isinstance(idx, indexing.Slice)
  )
  return starts, sizes, squeeze_dims


# In this code, indexing is handled in three ways: `slice`, `dynamic_slice`, and
# gather. For the gather case, the goal is to create a gather array, which means
# that we need to convert all other types of indexers into integer array
# indexers. This is done by looping over all indexers and checking if they are
# not integer array indexers, and if not, performing the conversion. However,
# during this process, the indexing semantics may change. Specifically,
# according to the indexing rules of NumPy, when there are integer array
# indexers separated by other indexers, the axes corresponding to the integer
# array indexers need to be moved to the front. After we convert all other
# indexers to integer array indexers, the distinction between integer array
# indexers and other types of indexers is lost. As a result, it becomes
# impossible to determine which axes should be moved to the front. In this case,
# we need to transpose the target array before the gather operation. We also
# need to transpose the target array back after the gather operation, if it is
# used in subsequent computations.
def _maybe_transpose_before_gather(
    indexer: indexing.NDIndexer
) -> tuple[int, ...] | None:
  is_int_indexing, _, _ = indexing.unpack_ndindexer(indexer)

  int_indexers_contiguous = bool(
      np.all(np.diff(np.where(is_int_indexing)[0]) == 1)
  )
  if int_indexers_contiguous:
    return None  # no transpose needed

  int_indexer_idxs: list[int] = []
  non_int_indexer_idxs: list[int] = []
  for i, is_int_index in enumerate(is_int_indexing):
    (int_indexer_idxs if is_int_index else non_int_indexer_idxs).append(i)
  transpose_order = (*int_indexer_idxs, *non_int_indexer_idxs)
  return transpose_order


def _perform_transpose_before_gather(
    target_arr: Array,
    indexer: indexing.NDIndexer,
    transpose_order: tuple[int, ...],
) -> tuple[Array, indexing.NDIndexer]:
  new_target_arr = target_arr.transpose(transpose_order)
  reordered_indices = tuple(indexer.indices[i] for i in transpose_order)
  new_indexer = indexing.NDIndexer(
      indices=reordered_indices,
      shape=indexer.shape,
      int_indexer_shape=indexer.int_indexer_shape,
  )
  return new_target_arr, new_indexer


def _convert_to_gather_arrays(indexer: indexing.NDIndexer) -> tuple[Array, ...]:
  # This is the general gather case. We need to create the gather arrays.
  total_shape = indexer.get_indexer_shape()
  is_int_indexing, _, _ = indexing.unpack_ndindexer(indexer)

  if any(is_int_indexing):
    n_idxers = len(indexer.indices)
    int_indexer_shape = indexer.int_indexer_shape
    n_int_indexers = sum(1 for p in is_int_indexing if p)
    last_int_index_idx = n_idxers - 1 - is_int_indexing[::-1].index(True)
    n_slice_index_dims_after_int = n_idxers - last_int_index_idx - 1

  def get_idx_in_shape_after_indexing(i):
    if not any(is_int_indexing):
      return i

    if i < n_idxers - n_slice_index_dims_after_int - n_int_indexers:
      return i
    if i < n_idxers - n_slice_index_dims_after_int:
      raise ValueError
    return i - n_int_indexers + len(int_indexer_shape)

  arrs = []
  for i, idxer in enumerate(indexer.indices):
    if isinstance(idxer, indexing.Slice):
      idx_in_shape_after_indexing = get_idx_in_shape_after_indexing(i)
      arr = (
          lax.iota(np.int32, total_shape[idx_in_shape_after_indexing])
          * idxer.stride
          + idxer.start
      )
      diff = len(total_shape) - idx_in_shape_after_indexing - 1
      arr = arr.reshape(arr.shape + (1,) * diff)
      arrs.append(arr)
    elif isinstance(idxer, (np.ndarray, Array)):
      diff = n_idxers - 1 - last_int_index_idx
      arr = idxer.reshape(idxer.shape + (1,) * diff)
      arrs.append(arr)
    else:
      raise ValueError(f"Invalid type of idxer: {type(idxer).__name__}")

  return tuple(arrs)


@register_discharge_rule(get_p)
def _get_discharge_rule(
    in_avals: Sequence[core.AbstractValue],
    out_avals: Sequence[core.AbstractValue], x, *idx,
    tree):
  del in_avals, out_avals
  y = _get_discharge(x, idx, tree)
  return (None,) * (len(idx) + 1), y


def _index_array(x, indexer: indexing.NDIndexer):
  if _is_trivial_indexer(indexer):
    return x
  # Try the three APIs in the following order: `lax.slice`,
  # `lax.dynamic_slice` and gather
  if maybe_slice := _maybe_convert_to_slice(indexer):
    x = lax_slicing.slice(x, *zip(*maybe_slice))
  # If everything in the indexer is a slice or ()-shaped, we can also
  # use `lax.dynamic_slice` with 1-sized slices for ()-shaped indices.
  # We need to squeeze out the 1-sized slices at the end.
  elif maybe_dynamic_slice := _maybe_convert_to_dynamic_slice(indexer):
    starts, sizes, squeeze_dims = maybe_dynamic_slice
    y = lax_slicing.dynamic_slice(x, starts, sizes)
    x = lax.squeeze(y, squeeze_dims)
  else:
    transpose_order = _maybe_transpose_before_gather(indexer)
    if transpose_order is not None:
      x, indexer = _perform_transpose_before_gather(x, indexer, transpose_order)
    arrays = _convert_to_gather_arrays(indexer)
    x = x[arrays]
  return x


def transform_array(x, transforms):
  if transforms is None:
    transforms = []
  result = x
  for transform in transforms:
    if transform is None:
      continue
    match transform:
      case indexing.NDIndexer():
        result = _index_array(result, transform)
      case RefBitcaster():
        result = bitcast(result, transform.dtype)
      case RefReshaper():
        result = result.reshape(transform.shape)
      case _:
        raise NotImplementedError(f"Unsupported transform: {transform}")
  return result

def transform_swap_array(x, transforms, val):
  if transforms is None:
    transforms = []

  # Will hold the value read from `x` before the swap, and will have the same
  # shape as `val`.
  new_val = x
  # List of intermediate results by transforming `x`.
  intermediates = [x]

  # Read phase (forward loop)
  for transform in transforms:
    match transform:
      case indexing.NDIndexer():
        indexer = transform
        if _is_trivial_indexer(indexer):
          intermediates.append(intermediates[-1])
          continue
        # If everything in the indexer is a slice or ()-shaped, we can also
        # use `lax.dynamic_slice` with 1-sized slices for ()-shaped indices.
        # We need to squeeze out the 1-sized slices at the end.
        if maybe_slice := _maybe_convert_to_dynamic_slice(indexer):
          starts, sizes, squeeze_dims = maybe_slice
          new_val = lax.squeeze(
              lax_slicing.dynamic_slice(new_val, starts, sizes), squeeze_dims
          )
        else:
          transpose_order = _maybe_transpose_before_gather(indexer)
          if transpose_order is not None:
            new_val, indexer = _perform_transpose_before_gather(
                new_val, indexer, transpose_order
            )
          arrays = _convert_to_gather_arrays(indexer)
          new_val = new_val[arrays]
          # Here, we don't need to transpose `new_val` back because it now holds
          # the result of the indexing, and is no longer the original array that
          # was indexed into.
        intermediates.append(new_val)
      case RefBitcaster():
        intermediates.append(bitcast(new_val, transform.dtype))
      case RefReshaper():
        intermediates.append(new_val.reshape(transform.shape))
      case _:
        raise NotImplementedError(f"Unsupported transform: {transform}")

  # Will hold the final state of the `x` after `val` has been written to the
  # transformed location, and will have the same shape as `x`.
  new_x = val

  # Write phase (reversed loop)
  for intermediate, transform in reversed(zip(intermediates[:-1], transforms)):
    if isinstance(transform, indexing.NDIndexer):
      indexer = transform
      if _is_trivial_indexer(indexer):
        continue
      if maybe_slice := _maybe_convert_to_dynamic_slice(indexer):
        starts, _, squeeze_dims = maybe_slice
        new_x = lax_slicing.dynamic_update_slice(
            intermediate, lax.expand_dims(new_x, squeeze_dims), starts
        )
      else:
        transpose_order = _maybe_transpose_before_gather(indexer)
        if transpose_order is not None:
          intermediate, indexer = _perform_transpose_before_gather(
              intermediate, indexer, transpose_order
          )
        arrays = _convert_to_gather_arrays(indexer)
        new_x = intermediate.at[arrays].set(new_x)  # pytype: disable=attribute-error
        if transpose_order is not None:
          transpose_order_inversed = np.argsort(transpose_order)
          new_x = new_x.transpose(transpose_order_inversed)
    else:
      raise NotImplementedError(f"Unsupported transform: {transform}")

  return new_val, new_x


def _get_discharge(x, idx, tree):
  transforms = tree_util.tree_unflatten(tree, idx)
  return transform_array(x, transforms)

@register_discharge_rule(swap_p)
def _swap_discharge_rule(
    in_avals: Sequence[core.AbstractValue],
    out_avals: Sequence[core.AbstractValue], x, val, *idx,
    tree):
  del in_avals, out_avals
  z, x_new = _swap_discharge(x, val, idx, tree)
  return (x_new, None) + (None,) * len(idx), z

def _swap_discharge(x, val, idx, tree):
  transforms = tree_util.tree_unflatten(tree, idx)
  return transform_swap_array(x, transforms, val)

@register_discharge_rule(addupdate_p)
def _addupdate_discharge_rule(
    in_avals: Sequence[core.AbstractValue],
    out_avals: Sequence[core.AbstractValue], x, val, *idx,
    tree):
  del in_avals, out_avals
  ans = _addupdate_discharge(x, val, idx, tree)
  return (ans, None) + (None,) * len(idx), []

def _addupdate_discharge(x, val, idx, tree):
  transforms = tree_util.tree_unflatten(tree, idx)
  if len(transforms) > 1:
    raise NotImplementedError("Only single indexer is supported.")
  indexer = transforms[0]

  if _is_trivial_indexer(indexer):
    return x + val

  # If everything in the indexer is a slice or ()-shaped, we can also
  # use `lax.dynamic_slice` with 1-sized slices for ()-shaped indices.
  # We need to squeeze out the 1-sized slices at the end.
  if maybe_slice := _maybe_convert_to_dynamic_slice(indexer):
    starts, sizes, squeeze_dims = maybe_slice
    x_old = lax_slicing.dynamic_slice(x, starts, sizes)
    val = lax.expand_dims(val, squeeze_dims)
    y = lax_slicing.dynamic_update_slice(x, x_old + val, starts)
    return y

  transpose_order = _maybe_transpose_before_gather(indexer)
  if transpose_order is not None:
    x, indexer = _perform_transpose_before_gather(x, indexer, transpose_order)
  arrays = _convert_to_gather_arrays(indexer)
  x = x.at[arrays].add(val)
  if transpose_order is not None:
    transpose_order_inversed = np.argsort(transpose_order)
    x = x.transpose(transpose_order_inversed)
  return x


@weakref_lru_cache
def _cached_closed_jaxpr_discharge(closed_jaxpr: core.ClosedJaxpr):
  jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.consts
  num_outs = len(jaxpr.outvars)
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, consts)
  discharged_closed_jaxpr = core.ClosedJaxpr(discharged_jaxpr, discharged_consts)
  fun = lu.wrap_init(core.jaxpr_as_fun(discharged_closed_jaxpr),
                     debug_info=discharged_jaxpr.debug_info)
  return discharged_closed_jaxpr, num_outs, fun

@register_discharge_rule(core.closed_call_p)
def _closed_call_discharge_rule(
    in_avals: Sequence[core.AbstractValue], _,*args,
    call_jaxpr: core.ClosedJaxpr):
  discharged_closed_jaxpr, num_outs, fun = _cached_closed_jaxpr_discharge(call_jaxpr)
  out_and_ref_vals = core.closed_call_p.bind(fun, *args,
                                             call_jaxpr=discharged_closed_jaxpr)
  out_vals, ref_vals = split_list(out_and_ref_vals, [num_outs])
  ref_vals_iter = iter(ref_vals)
  new_invals = tuple(next(ref_vals_iter) if isinstance(aval, AbstractRef)
                     else None for aval in in_avals)
  sentinel = object()
  assert next(ref_vals_iter, sentinel) is sentinel
  return new_invals, out_vals

# # `run_state`

run_state_p = core.Primitive("run_state")
run_state_p.multiple_results = True

def _default_initialization(x):
  assert hasattr(x, 'shape')
  assert hasattr(x, 'dtype')
  dtype = np.dtype(x)
  if np.issubdtype(dtype, np.integer):
    value = np.iinfo(dtype).min
  else:
    value = math.nan
  return lax.full(x.shape, value, dtype)

def _run_state_impl(*args: Any, jaxpr: core.Jaxpr,
                    which_linear: tuple[bool, ...],
                    is_initialized: tuple[bool, ...]):
  del which_linear
  discharged_jaxpr, consts = discharge_state(jaxpr, ())
  # Initialize the args that are not initialized.
  args_it = iter(args)
  args = tuple(
      next(args_it) if is_init else _default_initialization(var.aval)
      for is_init, var in zip(is_initialized, discharged_jaxpr.invars)
  )
  return core.eval_jaxpr(discharged_jaxpr, consts, *args)
run_state_p.def_impl(_run_state_impl)
mlir.register_lowering(run_state_p, mlir.lower_fun(_run_state_impl))

def _run_state_abstract_eval(*avals: core.AbstractValue, jaxpr: core.Jaxpr,
                             which_linear: tuple[bool, ...],
                             is_initialized: tuple[bool, ...]):
  del which_linear
  assert sum(is_initialized) == len(avals)
  # When we abstractly evaluate `run_state`, we want to keep track of which
  # input avals are `Ref`s and which are not. If an aval is a `Ref`, we want to
  # "propagate" out its inner effects. Otherwise, the effects are local to this
  # `run_state`.
  inner_to_outer_aval_mapping = {}
  outer_ref_index = 0
  for i, is_init in enumerate(is_initialized):
    if not is_init:
      pass
    inner_to_outer_aval_mapping[i] = outer_ref_index
    outer_ref_index += 1
  nonlocal_effects = set()
  is_ref = {i for i, aval in enumerate(avals) if isinstance(aval, AbstractRef)}
  for eff in jaxpr.effects:
    if not isinstance(eff, RefEffect):
      nonlocal_effects.add(eff)
      continue
    if eff.input_index not in inner_to_outer_aval_mapping:
      # This means that this effect corresponds to an uninitialized Ref and
      # should not propagate out of the primitive.
      continue
    # If we do propagate the effect, we need to update the input index to
    # correspond to the outer index.
    outer_index = inner_to_outer_aval_mapping[eff.input_index]
    if outer_index in is_ref:
      # This means that the effect corresponds to a Ref from an outside scope.
      nonlocal_effects.add(
          eff.replace(input_index=inner_to_outer_aval_mapping[eff.input_index])
      )
  assert len(jaxpr.invars) == len(is_initialized)
  if not all(is_initialized):
    raise NotImplementedError  # Uninitialized refs are not in avals.
  return avals, nonlocal_effects
run_state_p.def_effectful_abstract_eval(_run_state_abstract_eval)

def _run_state_jvp(primals: Sequence[Any], tangents: Sequence[Any], *,
                   jaxpr: core.Jaxpr, which_linear: tuple[bool, ...],
                   is_initialized: tuple[bool, ...]):
  if not all(is_initialized):
    raise NotImplementedError("Uninitialized Refs are not supported in jvp.")
  nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  discharged_jaxpr, body_consts = discharge_state(jaxpr, ())
  for _ in range(len(nonzero_tangents)):
    _, out_nonzero_tangents = ad.jvp_jaxpr(
        core.ClosedJaxpr(discharged_jaxpr, body_consts),
        nonzero_tangents, instantiate=nonzero_tangents)
    if out_nonzero_tangents == nonzero_tangents:
      break
    nonzero_tangents = map(operator.or_, nonzero_tangents, out_nonzero_tangents)
  else:
    raise Exception("Invalid fixpoint")
  del discharged_jaxpr, body_consts, out_nonzero_tangents
  tangents = [ad.instantiate_zeros(t) if inst else t
              for t, inst in zip(tangents, nonzero_tangents)]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  closed_jvp_jaxpr, _ = ad.jvp_jaxpr(pe.close_jaxpr(jaxpr),
                                     nonzero_tangents, [])
  jvp_jaxpr_, jvp_consts = closed_jvp_jaxpr.jaxpr, closed_jvp_jaxpr.consts
  jvp_jaxpr = hoist_consts_to_refs(jvp_jaxpr_)
  jvp_which_linear = (*(False,) * len(jvp_consts), *which_linear, *(True,) * len(tangents))
  out = run_state_p.bind(*jvp_consts, *primals, *tangents, jaxpr=jvp_jaxpr,
                         which_linear=jvp_which_linear,
                         # TODO(sharadmv): compute this in the general case
                         is_initialized=(True,) * len(jvp_jaxpr.invars))
  out_consts, out_primals, out_tangents = split_list(out, [len(jvp_consts),
                                                           len(primals)])
  del out_consts
  out_tangents_iter = iter(out_tangents)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_primal_value(p)
                  for p, nz in zip(out_primals, nonzero_tangents)]
  return out_primals, out_tangents
ad.primitive_jvps[run_state_p] = _run_state_jvp

@register_discharge_rule(run_state_p)
def _run_state_discharge_rule(in_avals: Sequence[core.AbstractValue],
                              out_avals: Sequence[core.AbstractValue],
                              *args: Any, jaxpr: core.Jaxpr,
                              which_linear: Sequence[bool],
                              is_initialized: tuple[bool, ...]):
  if not all(is_initialized):
    raise NotImplementedError(
        "Uninitialized Refs are not supported in discharge."
    )
  del out_avals
  out_vals = run_state_p.bind(*args, jaxpr=jaxpr, which_linear=which_linear,
                              is_initialized=is_initialized)
  new_invals = []
  for aval, out_val in zip(in_avals, out_vals):
    new_invals.append(out_val if isinstance(aval, AbstractRef) else None)
  return new_invals, out_vals

def initial_style_jaxpr(
    fun: Callable, in_tree: PyTreeDef, in_avals: Sequence[core.AbstractValue],
    dbg: core.DebugInfo,
  ) -> tuple[core.Jaxpr, list[Any], PyTreeDef]:
  return _initial_style_jaxpr(fun, in_tree, tuple(in_avals), dbg)

@weakref_lru_cache
def _initial_style_jaxpr(fun: Callable,
                         in_tree: api_util.PyTreeDef,
                         in_avals: Sequence[core.AbstractValue],
                         debug: core.DebugInfo):
  fun_, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(fun, debug_info=debug),
      tree_util.treedef_tuple((in_tree,)))
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(fun_, in_avals)
  return jaxpr, consts, out_tree_thunk()


T = TypeVar('T')
def run_state(f: Callable[..., None]) -> Callable[[T], T]:
  def wrapped(args):
    dbg = api_util.debug_info("run_state", f, (args,), {})
    flat_args, in_tree = tree_util.tree_flatten(args)
    ref_avals, ref_args = unzip2(map(get_ref_aval_from_value, flat_args))
    # There may be some uninitialized values here in ref_args.
    jaxpr_, consts, _ = initial_style_jaxpr(f, in_tree, ref_avals, dbg)
    jaxpr = hoist_consts_to_refs(jaxpr_)
    which_linear = (False,) * (len(consts) + len(ref_args))
    refs_is_initialized = tuple(r is not uninitialized for r in ref_args)
    init_args = tuple(r for r in ref_args if r is not uninitialized)
    # Consts are always initialized.
    is_initialized = (True,) * len(consts) + refs_is_initialized
    out_const_flat = run_state_p.bind(*consts, *init_args, jaxpr=jaxpr,
                                      which_linear=which_linear,
                                      is_initialized=is_initialized)
    _, out_flat = split_list(out_const_flat, [len(consts)])
    return in_tree.unflatten(out_flat)
  return wrapped

def run_state_reference(f: Callable[..., None]):
  def wrapped(args):
    dbg = api_util.debug_info("run_state", f, (args,), {})
    flat_args, in_tree = tree_util.tree_flatten(args)
    ref_avals, ref_args = unzip2(map(get_ref_aval_from_value, flat_args))
    jaxpr_, consts, _ = initial_style_jaxpr(f, in_tree, ref_avals, dbg)
    jaxpr = hoist_consts_to_refs(jaxpr_)
    discharged_jaxpr, discharged_consts = discharge_state(jaxpr, ())

    # Initialize any uninitialized values here in ref_args in the reference.
    ref_args = [
        _default_initialization(aval) if r is uninitialized else r
        for r, aval in zip(ref_args, ref_avals)
    ]

    out_const_flat = core.eval_jaxpr(discharged_jaxpr, discharged_consts,
                                     *consts, *ref_args)
    _, out_flat = split_list(out_const_flat, [len(consts)])
    return in_tree.unflatten(out_flat)
  return wrapped

@register_discharge_rule(pjit.jit_p)
def _pjit_state_discharge_rule(
    in_avals, out_avals, *args, jaxpr, in_shardings, out_shardings,
    in_layouts, out_layouts, **params):
  if not all(isinstance(s, sharding_impls.UnspecifiedValue) for s in (*in_shardings, *out_shardings)):
    raise NotImplementedError

  if not (all(l is None for l in in_layouts) and
          all(l is None for l in out_layouts)):
    raise NotImplementedError

  discharged_jaxpr = discharge_state2(jaxpr)
  new_in_shardings = (sharding_impls.UNSPECIFIED,) * len(discharged_jaxpr.in_avals)
  new_out_shardings = (sharding_impls.UNSPECIFIED,) * len(discharged_jaxpr.out_avals)
  new_in_layouts = (None,) * len(discharged_jaxpr.in_avals)
  new_out_layouts = (None,) * len(discharged_jaxpr.out_avals)
  out_and_ref_vals = pjit.jit_p.bind(
      *args, jaxpr=discharged_jaxpr, in_shardings=new_in_shardings,
      out_shardings=new_out_shardings, in_layouts=new_in_layouts,
      out_layouts=new_out_layouts, **params)
  out_vals, ref_vals = split_list(out_and_ref_vals, [len(jaxpr.out_avals)])
  ref_vals_iter = iter(ref_vals)
  new_invals = tuple(next(ref_vals_iter) if isinstance(aval, AbstractRef)
                     else None for aval in in_avals)
  sentinel = object()
  assert next(ref_vals_iter, sentinel) is sentinel
  return new_invals, out_vals


@register_discharge_rule(custom_derivatives.custom_vjp_call_p)
def custom_vjp_call_discharge(in_avals, out_avals, *args, call_jaxpr,
                              fwd_jaxpr_thunk, bwd, out_trees, symbolic_zeros,
                              num_consts):
  # Discharge happens after all AD is done, so we can discard the AD rules.
  del fwd_jaxpr_thunk, bwd, out_trees, symbolic_zeros, num_consts
  dis_jaxpr, dis_consts = discharge_state(call_jaxpr.jaxpr, call_jaxpr.consts)
  outs = _eval_jaxpr_ad_error(dis_jaxpr, dis_consts, args)
  out_vals, ref_vals = split_list(outs, [len(call_jaxpr.out_avals)])
  ref_vals_ = iter(ref_vals)
  new_invals = [next(ref_vals_) if isinstance(aval, AbstractRef) else None
                for aval in in_avals]
  assert next(ref_vals_, None) is None
  return new_invals, out_vals

@partial(custom_derivatives.custom_jvp, nondiff_argnums=(0,))
def _eval_jaxpr_ad_error(dis_jaxpr, consts, args):
  return core.eval_jaxpr(dis_jaxpr, consts, *args)
@_eval_jaxpr_ad_error.defjvp
def _eval_jaxpr_ad_error_jvp(*_):
  raise Exception("should be unreachable, AD after discharge")
