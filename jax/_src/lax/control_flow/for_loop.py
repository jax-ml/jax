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
"""Module for the `for_loop` primitive."""
from functools import partial

from typing import Any, Dict, List, Sequence, Tuple

from jax import core
from jax import lax
from jax import linear_util as lu
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax._src import ad_util
from jax._src import pretty_printer as pp
from jax._src import util

## JAX utilities

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

## Helpful type aliases
Ref = Any
Array = Any

## State effect

class StateEffect: pass
State = StateEffect

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
  raise ValueError("Can't evaluate `get` outside a stateful context.")
get_p.def_impl(_get_impl)

def ref_get(ref: Ref, idx: Tuple[int]) -> Array:
  """Reads a value from a `Ref`, a.k.a. value <- ref[idx]."""
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
  del ref, idx, value
  raise ValueError("Can't evaluate `swap` outside a stateful context.")
swap_p.def_impl(_swap_impl)

def ref_swap(ref: Ref, idx: Tuple[int], value: Array) -> Array:
  """Sets a `Ref`'s value and returns the original value."""
  return swap_p.bind(ref, value, *idx)

def ref_set(ref: Ref, idx: Tuple[int], value: Array) -> None:
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

def ref_addupdate(ref: Ref, idx: Tuple[int], x: Array) -> None:
  """Mutates a ref with an additive update i.e. `ref[idx] += x`."""
  return addupdate_p.bind(ref, x, *idx)

## get/set/addupdate abstract evaluation rules

# We need an aval for `Ref`s so we can represent `get` and `swap` in Jaxprs.
# A `ShapedArrayRef` is a abstract value for mutable containers of array types
class ShapedArrayRef(core.AbstractValue):
  __slots__ = ["shape", "dtype"]

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  def join(self, other):
    assert core.symbolic_equal_shape(self.shape, other.shape)
    assert self.dtype == other.dtype
    return self

  def _getitem(self, tracer, idx) -> Array:
    if not isinstance(idx, tuple):
      idx = idx,
    return ref_get(tracer, idx)

  def _setitem(self, tracer, idx, val) -> None:
    if not isinstance(idx, tuple):
      idx = idx,
    return ref_set(tracer, idx, val)

  def __repr__(self) -> str:
    a = core.ShapedArray(self.shape, self.dtype)
    return f'Ref{{{a.str_short()}}}'

  def at_least_vspace(self):
    return self

core.raise_to_shaped_mappings[ShapedArrayRef] = lambda aval, _: aval

def _get_abstract_eval(ref_aval: ShapedArrayRef, *idx: int):
  if not isinstance(ref_aval, ShapedArrayRef):
    raise ValueError(f"`get` must be called on `Ref` types: {ref_aval}.")
  return core.ShapedArray(ref_aval.shape[len(idx):], ref_aval.dtype), {State}
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
  return core.ShapedArray(ref_aval.shape[len(idx):], ref_aval.dtype), {State}
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
  return [], {State}
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
  return (ref_swap(ref_tangent, idx, x_primal),  # type: ignore[arg-type]
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


## Discharging state

# Let's say we have a jaxpr that takes in `Ref`s and outputs regular JAX values
# (`Ref`s should never be outputs from jaxprs). We'd like to convert that jaxpr
# into a "pure" jaxpr that takes in and outputs values and no longer has the
# `State` effect.

def discharge_state(jaxpr: core.Jaxpr, consts: Sequence[Any]) -> Tuple[core.Jaxpr, List[Any]]:
  """Converts a jaxpr that takes in `Ref`s into one that doesn't."""
  in_avals = [core.ShapedArray(v.aval.shape, v.aval.dtype)
              if type(v.aval) is ShapedArrayRef
              else v.aval for v in jaxpr.invars]
  eval_jaxpr = lu.wrap_init(partial(_eval_jaxpr_discharge_state, jaxpr, consts))
  new_jaxpr, _ , new_consts = pe.trace_to_jaxpr_dynamic(eval_jaxpr, in_avals)
  return new_jaxpr, new_consts

def _dynamic_index(x, idx):
  if not idx: return x
  ndim = len(x.shape)
  starts = [*idx] + [lax.full_like(idx[0], 0, shape=())] * (ndim - len(idx))
  sizes = (1,) * len(idx) + x.shape[len(idx):]
  out = lax.dynamic_slice(x, starts, sizes)
  return out.reshape(x.shape[len(idx):])

def _dynamic_update_index(x, idx, val):
  if not idx: return val
  ndim = len(x.shape)
  starts = [*idx] + [lax.full_like(idx[0], 0, shape=())] * (ndim - len(idx))
  update = val.reshape((1,) * len(idx) + x.shape[len(idx):])
  return lax.dynamic_update_slice(x, update, starts)

def _eval_jaxpr_discharge_state(jaxpr: core.Jaxpr, consts: Sequence[Any],
                                *args: Any):
  env: Dict[core.Var, Any] = {}

  def read(v: core.Atom) -> Any:
    if type(v) is core.Literal:
      return v.val
    assert isinstance(v, core.Var)
    return env[v]

  def write(v: core.Var, val: Any) -> None:
    env[v] = val

  map(write, jaxpr.constvars, consts)
  # Here some args may correspond to `Ref` avals but they'll be treated like
  # regular values in this interpreter.
  map(write, jaxpr.invars, args)

  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    if eqn.primitive is get_p:
       # `y <- x[i]` becomes `y = ds x i`
      x, *idx = in_vals
      write(eqn.outvars[0], _dynamic_index(x, idx))
    elif eqn.primitive is swap_p:
      # `z, x[i] <- x[i], val` becomes:
      #    z = ds x i
      #    x = dus x i val
      x, val, *idx = in_vals
      write(eqn.outvars[0], _dynamic_index(x, idx))
      assert isinstance(eqn.invars[0], core.Var)
      write(eqn.invars[0], _dynamic_update_index(x, idx, val))
    elif eqn.primitive is addupdate_p:
      # `x[i] += val` becomes:
      #    y = ds x i
      #    z = y + val
      #    x = dus x i z
      x, val, *idx = in_vals
      ans = _dynamic_update_index(x, idx, val + _dynamic_index(x, idx))
      assert isinstance(eqn.invars[0], core.Var)
      write(eqn.invars[0], ans)
    else:
      # Default primitive rule, similar to `core.eval_jaxpr`. Note that here
      # we assume any higher-order primitives inside of the jaxpr are *not*
      # stateful.
      subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
      ans = eqn.primitive.bind(*subfuns, *map(read, eqn.invars), **bind_params)
      if eqn.primitive.multiple_results:
        map(write, eqn.outvars, ans)
      else:
        write(eqn.outvars[0], ans)
  # By convention, we return the outputs of the jaxpr first and then the final
  # values of the `Ref`s. Callers to this function should be able to split
  # them up by looking at `len(jaxpr.outvars)`.
  out_vals = map(read, jaxpr.outvars)
  ref_vals = map(
      read, [v for v in jaxpr.invars if type(v.aval) is ShapedArrayRef])
  return out_vals + ref_vals
