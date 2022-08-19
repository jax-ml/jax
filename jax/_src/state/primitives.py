# Copyright 2022 Google LLC
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
"""Module for state primitives."""
from functools import partial

from typing import Any, Generic, List, Tuple, TypeVar

from jax import core
from jax._src import ad_util
from jax._src import pretty_printer as pp
from jax._src.util import safe_map, safe_zip
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp

from jax._src.state.types import ShapedArrayRef, StateEffect

## General utilities

Array = Any
T = TypeVar('T')
class Ref(Generic[T]): pass

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

## get/swap/addupdate implementations

# `get` reads a value from a `Ref` type, a.k.a.:
# a = get_p.bind(x)
# or we can read using indices:
# a = get_p.bind(x, 0, 1)
# Staging out `a = get_p.bind(x)` where the aval of `x` is
# `ShapedArrayRef((3,), np.dtype('float32'))` leads to a jaxpr eqn printed like
#   a:f32[3] <- x[]
get_p = core.Primitive("get")

def _get_impl(ref: Ref, *idx: int):
  del ref, idx
  raise ValueError("Cannot run stateful primitive.")
get_p.def_impl(_get_impl)

def ref_get(ref: Ref, idx: Tuple[int, ...]) -> Array:
  """Reads a value from a `Ref`, a.k.a. value <- ref[idx]."""
  idx = map(jnp.int32, idx)
  return get_p.bind(ref, *idx)

# `swap` mutates a `Ref`, setting its value and returns its previous value.
# b = swap_p.bind(x, a)
# It generalizes the setting operation for a `Ref` as we can ignore the return
# value:
# _ = swap_p.bind(x, a)
# `swap_p` also takes in index arguments following the value, i.e.:
# _ = swap_p.bind(x, a, 0, 1)
# Staging out `b = swap_p.bind(x, a)` where the aval of `x` is
# `ShapedArrayRef((3,), np.dtype('float32'))` and the aval of `a` is
# `ShapedArray((3,), np.dtype('float32'))` leads to a jaxpr eqn printed like
#   b:f32[3], x:Ref{f32[3]} <- x, a
# Staging out `_ = swap_p.bind(x, a, i, j)` where the aval of `x` is
# `ShapedArrayRef((3,), np.dtype('float32'))` , the aval of `a` is
# `ShapedArray((3,), np.dtype('float32'))`, and the avals of both `i` and `j`
# are `ShapedArray((), np.dtype('int32'))` leads to a jaxpr eqn printed like
#   x:Ref{f32[3]}[i, j] <- a
swap_p = core.Primitive("swap")

def _swap_impl(ref: Ref, value: Array, *idx: int):
  del ref, value, idx
  raise ValueError("Cannot run stateful primitive.")
swap_p.def_impl(_swap_impl)

def ref_swap(ref: Ref, idx: Tuple[int, ...], value: Array) -> Array:
  """Sets a `Ref`'s value and returns the original value."""
  idx = map(jnp.int32, idx)
  return swap_p.bind(ref, value, *idx)

def ref_set(ref: Ref, idx: Tuple[int, ...], value: Array) -> None:
  """Sets a `Ref`'s value, a.k.a. ref[idx] <- value."""
  ref_swap(ref, idx, value)

# `addupdate_p` mutates a `Ref`, adding a value to its existing value.
# Semantically,
# ```
# addupdate ref a *idx
# ```
# is equivalent to
# ```
# b = get ref *idx
# c = add b x
# _ = swap ref c *idx
# ```
addupdate_p = core.Primitive('addupdate')
addupdate_p.multiple_results = True

def _addupdate_impl(ref: Ref, value: Array, *idx: int):
  del ref, idx, value
  raise ValueError("Can't evaluate `addupdate` outside a stateful context.")
addupdate_p.def_impl(_addupdate_impl)

def ref_addupdate(ref: Ref, idx: Tuple[int, ...], x: Array) -> None:
  """Mutates a ref with an additive update i.e. `ref[idx] += x`."""
  return addupdate_p.bind(ref, x, *idx)

## get/set/addupdate abstract evaluation rules

def _get_abstract_eval(ref_aval: ShapedArrayRef, *idx: int):
  if not isinstance(ref_aval, ShapedArrayRef):
    raise ValueError(f"`get` must be called on `Ref` types: {ref_aval}.")
  return (core.ShapedArray(ref_aval.shape[len(idx):], ref_aval.dtype),
          {StateEffect})
get_p.def_effectful_abstract_eval(_get_abstract_eval)


def _swap_abstract_eval(ref_aval: ShapedArrayRef, val_aval: core.AbstractValue,
                        *idx: int):
  if not isinstance(ref_aval, ShapedArrayRef):
    raise ValueError(f"`swap` must be called on `Ref` types: {ref_aval}.")
  val_aval = core.raise_to_shaped(val_aval)
  assert isinstance(val_aval, core.ShapedArray)
  expected_output_shape = ref_aval.shape[len(idx):]
  if expected_output_shape != val_aval.shape:
    raise ValueError("Invalid shape for `swap`. "
                     f"Ref shape: {ref_aval.shape}. "
                     f"Value shape: {val_aval.shape}. "
                     f"Indices: {idx}. ")
  if ref_aval.dtype != val_aval.dtype:
    raise ValueError("Invalid dtype for `swap`. "
                     f"Ref dtype: {ref_aval.dtype}. "
                     f"Value shape: {val_aval.dtype}. ")
  return (core.ShapedArray(ref_aval.shape[len(idx):], ref_aval.dtype),
          {StateEffect})
swap_p.def_effectful_abstract_eval(_swap_abstract_eval)


def _addupdate_abstract_eval(ref_aval: ShapedArrayRef,
                             val_aval: core.AbstractValue,
                             *idx: int):
  if not isinstance(ref_aval, ShapedArrayRef):
    raise ValueError(f"`addupdate` must be called on `Ref` types: {ref_aval}.")
  val_aval = core.raise_to_shaped(val_aval)
  assert isinstance(val_aval, core.ShapedArray)
  expected_output_shape = ref_aval.shape[len(idx):]
  if expected_output_shape != val_aval.shape:
    raise ValueError("Invalid shape for `swap`. "
                     f"Ref shape: {ref_aval.shape}. "
                     f"Value shape: {val_aval.shape}. "
                     f"Indices: {idx}. ")
  return [], {StateEffect}
addupdate_p.def_effectful_abstract_eval(_addupdate_abstract_eval)

## Pretty printing for `get` and `swap` in jaxprs

pp_ref = partial(pp.color, intensity=pp.Intensity.NORMAL,
                 foreground=pp.Color.GREEN)

def _get_pp_rule(eqn, context, settings):
  # Pretty prints `a = get x i` as `x[i] <- a`
  y, = eqn.outvars
  x, *idx = eqn.invars
  idx = ','.join(core.pp_var(i, context) for i in idx)
  lhs = core.pp_vars([y], context, print_shapes=settings.print_shapes)
  return [lhs, pp.text(' <- '), pp_ref(pp.concat([
    pp.text(core.pp_var(x, context)), pp.text('['), pp.text(idx), pp.text(']')
    ]))]
core.pp_eqn_rules[get_p] = _get_pp_rule

def _swap_pp_rule(eqn, context, settings):
  y, = eqn.outvars
  x, v, *idx = eqn.invars
  idx = ','.join(core.pp_var(i, context) for i in idx)
  if type(y) is core.DropVar:
    # In the case of a set (ignored return value),
    # pretty print `_ = swap x v i` as `x[i] <- v`
    del y
    return [
      pp_ref(pp.concat([
        pp.text(core.pp_var(x, context)),
        pp.text('['), pp.text(idx), pp.text(']')
      ])), pp.text(' <- '), pp.text(core.pp_var(v, context))]
  else:
    # pretty-print `y:T = swap x v i` as `y:T, x[i] <- x[i], v`
    x_i = pp.concat([pp.text(core.pp_var(x, context)),
                     pp.text('['), pp.text(idx), pp.text(']')])
    y = core.pp_vars([y], context, print_shapes=settings.print_shapes)
    return [y, pp.text(', '), x_i, pp.text(' <- '),
            x_i, pp.text(', '), pp.text(core.pp_var(v, context))]
core.pp_eqn_rules[swap_p] = _swap_pp_rule

def _addupdate_pp_rule(eqn, context, settings):
  # pretty-print ` = addupdate x i v` as `x[i] += v`
  () = eqn.outvars
  x, v, *idx = eqn.invars
  idx = ','.join(core.pp_var(i, context) for i in idx)
  return [
    pp_ref(pp.concat([
        pp.text(core.pp_var(x, context)),
        pp.text('['), pp.text(idx), pp.text(']')
      ])), pp.text(' += '), pp.text(core.pp_var(v, context))]
core.pp_eqn_rules[addupdate_p] = _addupdate_pp_rule

## get/swap/addupdate JVP rules

def _get_jvp(primals: List[Any], tangents: List[Any]):
  ref_primal, *idx = primals
  assert isinstance(ref_primal.aval, ShapedArrayRef)
  ref_tangent, *_ = tangents
  assert isinstance(ref_tangent.aval, ShapedArrayRef)
  return ref_get(ref_primal, idx), ref_get(ref_tangent, idx)  # type: ignore[arg-type]
ad.primitive_jvps[get_p] = _get_jvp

def _swap_jvp(primals: List[Any], tangents: List[Any]):
  ref_primal, x_primal, *idx = primals
  assert isinstance(ref_primal.aval, ShapedArrayRef)
  ref_tangent, x_tangent, *_ = tangents
  assert isinstance(ref_tangent.aval, ShapedArrayRef)
  x_tangent = ad_util.instantiate(x_tangent)
  return (ref_swap(ref_primal, idx, x_primal),  # type: ignore[arg-type]
          ref_swap(ref_tangent, idx, x_tangent))  # type: ignore[arg-type]
ad.primitive_jvps[swap_p] = _swap_jvp

def addupdate_jvp_rule(primals: List[Any], tangents: List[Any]):
  ref_primal, x_primal, *idx = primals
  ref_tangent, x_tangent, *_ = tangents
  x_tangent = ad_util.instantiate(x_tangent)
  addupdate_p.bind(ref_primal, x_primal, *idx)
  addupdate_p.bind(ref_tangent, x_tangent, *idx)
  return [], []
ad.primitive_jvps[addupdate_p] = addupdate_jvp_rule

##  get/swap/addupdate transpose rules

def _get_transpose(g, ref, *idx):
  # get transpose is addupdate
  if type(g) is not ad_util.Zero:
    ref_addupdate(ref, idx, g)
  return [None] + [None] * len(idx)
ad.primitive_transposes[get_p] = _get_transpose

def _swap_transpose(g, ref, x, *idx):
  # swap transpose is swap
  x_bar = ref_swap(ref, idx, ad_util.instantiate(g))
  return [None, x_bar] + [None] * len(idx)
ad.primitive_transposes[swap_p] = _swap_transpose

def addupdate_transpose(cts_in, ref, x, *idx):
  # addupdate transpose is get
  del cts_in, x
  g = ref_get(ref, idx)
  return [None, g] + [None] * len(idx)
ad.primitive_transposes[addupdate_p] = addupdate_transpose

## get/swap/addupdate partial_eval_custom rules

def _state_partial_eval_custom(prim, saveable, unks_in, inst_in, eqn):
  if any(unks_in):
    res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
    return None, eqn, [True] * len(eqn.outvars), [True] * len(eqn.outvars), res
  elif saveable(get_p, *[var.aval for var in eqn.invars], **eqn.params):
    return eqn, None, [False] * len(eqn.outvars), [False] * len(eqn.outvars), []
  res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
  return eqn, eqn, [False] * len(eqn.outvars), [True] * len(eqn.outvars), []

pe.partial_eval_jaxpr_custom_rules[get_p] = partial(_state_partial_eval_custom,
                                                    get_p)
pe.partial_eval_jaxpr_custom_rules[swap_p] = partial(_state_partial_eval_custom,
                                                     swap_p)
pe.partial_eval_jaxpr_custom_rules[addupdate_p] = partial(
    _state_partial_eval_custom, addupdate_p)
