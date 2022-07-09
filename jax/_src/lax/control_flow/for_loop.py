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
import operator

from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

from jax import core
from jax import lax
from jax import linear_util as lu
from jax.api_util import flatten_fun_nokwargs
from jax.interpreters import ad
from jax.interpreters import masking
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.tree_util import (tree_flatten, tree_structure, tree_unflatten,
                           treedef_tuple, tree_map, tree_leaves, PyTreeDef)
from jax._src import ad_util
from jax._src import dtypes
from jax._src import pretty_printer as pp
from jax._src import source_info_util
from jax._src.util import (partition_list, merge_lists, safe_map, safe_zip,
                           split_list)
import jax.numpy as jnp

from jax._src.lax.control_flow import loops
from jax._src.lax.control_flow.common import _abstractify, _initial_style_jaxpr

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

## Helpful type aliases
S = TypeVar('S')
T = TypeVar('T')
class Ref(Generic[T]): pass
Array = Any

## State effect

class StateEffect: pass
State = StateEffect()

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
  del ref, idx, value
  raise ValueError("Can't evaluate `swap` outside a stateful context.")
swap_p.def_impl(_swap_impl)

def ref_swap(ref: Ref, idx: Tuple[int], value: Array) -> Array:
  """Sets a `Ref`'s value and returns the original value."""
  idx = map(jnp.int32, idx)
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
  if ref_aval.dtype != val_aval.dtype:
    raise ValueError("Invalid dtype for `swap`. "
                     f"Ref dtype: {ref_aval.dtype}. "
                     f"Value shape: {val_aval.dtype}. ")
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
  # Pretty prints `a = get x i` as `a <- x[i]`
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

## `for_loop` implementation

for_p = core.Primitive('for')
for_p.multiple_results = True

### Tracing utilities

def _hoist_consts_to_refs(jaxpr: core.Jaxpr) -> core.Jaxpr:
  num_consts = len(jaxpr.constvars)

  # Note that this function is meant for use w/ `for_loop` since it assumes
  # that the index is the first argument and preserves this after hoisting
  # consts.
  def _hoist(i, *consts_args):
    const_refs, args = split_list(consts_args, [num_consts])
    # We immediately read the const values out of the `Ref`s.
    consts = [r[()] for r in const_refs]
    return core.eval_jaxpr(jaxpr, consts, i, *args)
  assert all(isinstance(var.aval, core.ShapedArray) for var in jaxpr.constvars)
  const_avals = [ShapedArrayRef(var.aval.shape, var.aval.dtype) for var in  # pytype: disable=attribute-error
                 jaxpr.constvars]
  i_aval, *arg_avals = [var.aval for var in jaxpr.invars]
  in_avals = [i_aval, *const_avals, *arg_avals]
  hoisted_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(_hoist), in_avals)
  assert not consts, "All consts should have been converted to refs"
  return hoisted_jaxpr

def _trace_to_jaxpr_with_refs(f, state_tree: PyTreeDef,
                              state_avals: Sequence[core.AbstractValue]
                              ) -> Tuple[core.Jaxpr, List[Any], PyTreeDef]:
  f, out_tree_thunk = flatten_fun_nokwargs(
      lu.wrap_init(f), treedef_tuple((tree_structure(0), state_tree)))
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      f, state_avals)
  return jaxpr, consts, out_tree_thunk()

def val_to_ref_aval(x) -> ShapedArrayRef:
  aval = core.raise_to_shaped(core.get_aval(x))
  if type(aval) is not core.ShapedArray:
    raise Exception(f"can't make ref from {x}")
  return ShapedArrayRef(aval.shape, aval.dtype)

def for_loop(nsteps: int, body: Callable[[Array, Ref[S]], None], init_state: S,
             *, reverse: bool = False) -> S:
  """A for-loop combinator that allows read/write semantics in the loop body.

  `for_loop` is a higher-order function that enables writing loops that can be
  staged out in JIT-ted JAX computations. Unlike `jax.lax.fori_loop`, it allows
  mutation in its body using `Ref`s.

  `for_loop` will initialize `Ref`s with the values in `init_state`. Each
  iteration, `body` will be called with the current `Ref`s, which can be read
  from and written to using `ref_get` and `ref_set`.

  `for_loop` is semantically equivalent to the following Python code:

  ```python
  def for_loop(nsteps, body, init_state):
    refs = tree_map(make_ref, init_state)
    for i in range(nsteps):
      body(i, refs)
    return tree_map(ref_get, refs)
  ```

  Args:
    nsteps: Number of iterations
    body: A callable that takes in the iteration number as its first argument
      and `Ref`s corresponding to `init_state` as its second argument.
      `body` is free to read from and write to its `Ref`s. `body` should
       not return anything.
    init_state: A Pytree of JAX-compatible values used to initialize the `Ref`s
      that will be passed into the for loop body.
  Returns:
    A Pytree of values representing the output of the for loop.
  """
  flat_state, state_tree = tree_flatten(init_state)
  state_avals = map(val_to_ref_aval, flat_state)
  idx_aval = core.ShapedArray((), jnp.dtype("int32"))
  jaxpr, consts, out_tree = _trace_to_jaxpr_with_refs(
      body, state_tree, [idx_aval, *state_avals])
  if out_tree != tree_structure(None):
    raise Exception("`body` should not return anything.")
  # Remove constvars from jaxpr and turn them into `Ref`s
  jaxpr = _hoist_consts_to_refs(jaxpr)
  which_linear = (False,) * (len(consts) + len(flat_state))
  out_flat = for_p.bind(*consts, *flat_state, jaxpr=jaxpr, nsteps=int(nsteps),
                        reverse=reverse, which_linear=which_linear)
  # Consts are `Ref`s so they are both inputs and outputs. We remove them from
  # the outputs.
  out_flat = out_flat[len(consts):]
  return tree_unflatten(state_tree, out_flat)

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

def scan(f: Callable[[Carry, X], Tuple[Carry, Y]],
         init: Carry,
         xs: X,
         length: Optional[int] = None,
         reverse: bool = False,
         unroll: int = 1) -> Tuple[Carry, Y]:
  if unroll != 1:
    raise NotImplementedError("Unroll not implemented")
  if not callable(f):
    raise TypeError("scan: f argument should be a callable.")
  xs_flat, xs_tree = tree_flatten(xs)

  try:
    lengths = [x.shape[0] for x in xs_flat]
  except AttributeError as err:
    msg = "scan got value with no leading axis to scan over: {}."
    raise ValueError(
      msg.format(', '.join(str(x) for x in xs_flat
                           if not hasattr(x, 'shape')))) from err

  if length is not None:
    length = int(length)
    if not all(length == l for l in lengths):
      msg = ("scan got `length` argument of {} which disagrees with "
             "leading axis sizes {}.")
      raise ValueError(msg.format(length, [x.shape[0] for x in xs_flat]))
  else:
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
      msg = "scan got values with different leading axis sizes: {}."
      raise ValueError(msg.format(', '.join(str(x.shape[0]) for x in xs_flat)))
    elif len(unique_lengths) == 0:
      msg = "scan got no values to scan over and `length` not provided."
      raise ValueError(msg)
    else:
      length, = unique_lengths

  x_shapes = [masking.padded_shape_as_value(x.shape[1:]) for x in xs_flat]
  x_dtypes = [dtypes.canonicalize_dtype(x.dtype) for x in xs_flat]
  x_avals = tuple(map(core.ShapedArray, x_shapes, x_dtypes))

  def _create_jaxpr(init):
    init_flat = tree_leaves(init)
    _, in_tree = tree_flatten((init, xs))

    carry_avals = tuple(map(_abstractify, init_flat))
    jaxpr, _, out_tree = _initial_style_jaxpr(
        f, in_tree, carry_avals + x_avals, "scan")
    return jaxpr, out_tree
  jaxpr, out_tree = _create_jaxpr(init)
  _, ys_avals = tree_unflatten(out_tree, jaxpr.out_avals)
  ys = tree_map(lambda aval: jnp.zeros([length, *aval.shape], aval.dtype),
                ys_avals)
  def for_body(i, refs):
    carry_refs, xs_refs, ys_refs = refs
    carry = tree_map(lambda x: x[()], carry_refs)
    x = tree_map(lambda x: x[i], xs_refs)
    carry, y = f(carry, x)
    tree_map(lambda c_ref, c: ref_set(c_ref, (), c), carry_refs, carry)
    tree_map(lambda y_ref, y: ref_set(y_ref, (i,), y), ys_refs, y)
  assert isinstance(length, int)
  init, _, ys = for_loop(length, for_body, (init, xs, ys), reverse=reverse)
  return init, ys


@for_p.def_abstract_eval
def _for_abstract_eval(*avals, jaxpr, **__):
  return list(avals)

def _for_impl(*args, jaxpr, nsteps, reverse, which_linear):
  del which_linear
  discharged_jaxpr, consts = discharge_state(jaxpr, ())
  def cond(carry):
    i, _ = carry
    return i < nsteps
  def body(carry):
    i, state = carry
    i_ = nsteps - i - 1 if reverse else i
    next_state = core.eval_jaxpr(discharged_jaxpr, consts, i_, *state)
    return i + 1, next_state
  _, state = lax.while_loop(cond, body, (jnp.int32(0), list(args)))
  return state
mlir.register_lowering(for_p, mlir.lower_fun(_for_impl, multiple_results=True))
for_p.def_impl(partial(xla.apply_primitive, for_p))

def _for_jvp(primals, tangents, *, jaxpr, nsteps, reverse, which_linear):
  nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  # We need to find out which `Ref`s have nonzero tangents after running the
  # for loop. Ordinarily we do this with a fixed point on the body jaxpr but
  # a `for` body jaxpr is stateful and has no outputs. We therefore discharge
  # the state effect from the jaxpr and we will now have a "symmetric" jaxpr
  # where the inputs line up with the outputs. We use this discharged jaxpr
  # for the fixed point.
  discharged_jaxpr, body_consts = discharge_state(jaxpr, ())
  for _ in range(len(nonzero_tangents)):
    _, out_nonzero_tangents = ad.jvp_jaxpr(
        core.ClosedJaxpr(discharged_jaxpr, body_consts),
        [False] + nonzero_tangents, instantiate=nonzero_tangents)
    if out_nonzero_tangents == nonzero_tangents:
      break
    nonzero_tangents = map(operator.or_, nonzero_tangents, out_nonzero_tangents)
  else:
    raise Exception("Invalid fixpoint")
  tangents = [ad.instantiate_zeros(t) if inst else t for t, inst in
      zip(tangents, nonzero_tangents)]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  closed_jaxpr = core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, [False] + nonzero_tangents, [])
  jvp_jaxpr, jvp_consts = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts
  jvp_which_linear = ((False,) * len(jvp_consts) + which_linear
                      + (True,) * len(tangents))
  out_flat = for_p.bind(*jvp_consts, *primals, *tangents, jaxpr=jvp_jaxpr,
                        nsteps=nsteps, reverse=reverse,
                        which_linear=jvp_which_linear)
  # `out_flat` includes constant inputs into the `for_loop` which are
  # converted into outputs as well. We don't care about these in AD so we
  # throw them out.
  _, out_primals, out_tangents = split_list(out_flat,
                                            [len(jvp_consts), len(primals)])
  out_tangents_iter = iter(out_tangents)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(out_primals, nonzero_tangents)]
  return out_primals, out_tangents
ad.primitive_jvps[for_p] = _for_jvp


def _partial_eval_jaxpr_custom(jaxpr, in_unknowns, policy):
  # A simple wrapper around `pe.partial_eval_jaxpr_custom` that assumes all
  # inputs are instantiated and doesn't ensure any outputs are unknown or
  # instantiated.
  return pe.partial_eval_jaxpr_custom(
      jaxpr, in_unknowns, [True] * len(in_unknowns), False, False, policy)

_save_everything = lambda *_, **__: True

def _for_partial_eval(trace: pe.JaxprTrace, *tracers: pe.JaxprTracer,
                      jaxpr: core.Jaxpr, nsteps: int, reverse: bool,
                      which_linear: Tuple[bool]) -> List[pe.JaxprTracer]:
  num_inputs = len(tracers)
  in_unknowns = [not t.pval.is_known() for t in tracers]
  # We first need to run a fixpoint to determine which of the `Ref`s are unknown
  # after running the for loop. We want to use the jaxpr to determine which
  # `Ref`s are unknown after executing the for loop body given which `Ref`s are
  # unknown before. However, the jaxpr has no outputs. Instead, we discharge
  # the body and run the fixpoint with the discharged jaxpr. We can do this
  # because the outputs of the jaxpr are one-to-one with the inputs.
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, ())
  discharged_jaxpr = discharged_jaxpr.replace(
      invars=discharged_jaxpr.constvars + discharged_jaxpr.invars,
      constvars=[])
  for _ in range(num_inputs):
    jaxpr_in_unknowns = [False] * len(discharged_consts) + [False, *in_unknowns]
    _, _, out_unknowns, _, _ = _partial_eval_jaxpr_custom(
        discharged_jaxpr, jaxpr_in_unknowns, _save_everything)
    out_unknowns = list(out_unknowns)
    if out_unknowns == in_unknowns:
      break
    in_unknowns = map(operator.or_, in_unknowns, out_unknowns)
  else:
    raise Exception("Invalid fixpoint")
  del out_unknowns  # redundant since it's the same as `in_unknowns`
  tracers = tuple(trace.instantiate_const(t) if uk else t
                  for t, uk in zip(tracers, in_unknowns))

  # We use `partial_eval_jaxpr_custom` here because it won't remove effectful
  # primitives like `get`/`set`.
  jaxpr_known_resout, jaxpr_unknown_resin_, _, _, num_res = \
        _partial_eval_jaxpr_custom(jaxpr, [False, *in_unknowns],
                                   _save_everything)
  # `partial_eval_jaxpr_custom` will give us jaxprs that have hybrid `Ref` and
  # regular valued input/outputs. However, we'd like to bind these jaxprs to a
  # `for`, which expects only `Ref` inputs and no output. We need to convert
  # both of these jaxprs into ones that are compatible with `for`.
  # TODO(sharadmv,mattjj): implement "passthrough" optimization.
  # TODO(sharadmv,mattjj): rematerialize loop-dependent values instead of
  # passing the loop index as a residual

  # `jaxpr_known_resout` is a jaxpr that maps from all the input `Refs`
  # to output residual values (none of them should be `Ref`s). We'll need to
  # convert the output residual values into `Ref`s that are initially empty
  # `Ref`s that are written to at the end of the jaxpr.

  # # Loop-invariant residual optimization
  # Here we are interested in finding out which of the residuals are *not*
  # dependent on the loop index. If a residual is not dependent on the loop
  # index, we don't need add an extra loop dimension we're reading from when we
  # convert it from an output into a write.

  # In order to detect which residuals are loop-invariant, we need to run a
  # fixpoint. This is because the residual could be dependent on a `Ref` that
  # changes each iteration of the loop so we need to first detect which `Ref`s
  # are loop-varying. We can do this by discharging the state from the jaxpr and
  # running partial_eval with initially only the loop-index being loop-varying.
  # The fixpoint will eventually propagate the loop-varying-ness over the
  # inputs/outputs and we will converge.
  loop_var_res = [False] * len(jaxpr_known_resout.outvars)
  loop_var_refs = [False] * (len(jaxpr_known_resout.invars) - 1)
  discharged_jaxpr_known_resout = core.ClosedJaxpr(
      *discharge_state(jaxpr_known_resout, ()))
  for _ in range(len(discharged_jaxpr_known_resout.jaxpr.invars)):
    (_, _, loop_var_outputs, _) = pe.partial_eval_jaxpr_nounits(
          discharged_jaxpr_known_resout, [True] + loop_var_refs, False)
    loop_var_res, loop_var_refs_ = split_list(
        loop_var_outputs, [len(loop_var_res)])
    if loop_var_refs == loop_var_refs_:
      break
    loop_var_refs = map(operator.or_, loop_var_refs, loop_var_refs_)
  # Now that the fixpoint is complete, we know which residuals are
  # loop-invariant.
  loop_invar_res = map(operator.not_, loop_var_res)

  jaxpr_known, res_avals = _convert_outputs_to_writes(nsteps,
                                                      jaxpr_known_resout,
                                                      loop_invar_res)
  # We now run the known jaxpr to obtain our residual values.
  known_tracers, _ = partition_list(in_unknowns, tracers)
  known_vals = [t.pval.get_known() for t in known_tracers]
  empty_res = map(ad_util.zeros_like_aval, res_avals)
  jaxpr_known_args = [*known_vals, *empty_res]
  jaxpr_known_which_linear = (False,) * len(jaxpr_known_args)
  out_flat = for_p.bind(*jaxpr_known_args, jaxpr=jaxpr_known, nsteps=nsteps,
                        reverse=reverse, which_linear=jaxpr_known_which_linear)
  known_outputs, residuals = split_list(out_flat, [len(known_tracers)])
  residuals = map(trace.new_instantiated_const, residuals)

  # Now we handle the `jaxpr_unknown` that expects residual values as inputs.
  # This jaxpr is the output of `partial_eval_jaxpr_custom` that marks which
  # inputs are actually used.
  # `partial_eval_jaxpr_custom` doesn't remove extra inputs/outputs for you
  # so we use `dce_jaxpr` here to do that.
  jaxpr_unknown_resin, used_inputs = pe.dce_jaxpr(
        jaxpr_unknown_resin_, [], [True] * num_res + [True, *in_unknowns])
  used_res, (used_i,), used_refs = split_list(used_inputs, [num_res, 1])
  assert all(used_res), "All residuals should be used"
  # To make it compatible with `for`, we need to convert those residual values
  # into `Ref`s.
  jaxpr_unknown = _convert_inputs_to_reads(nsteps, len(res_avals),
                                           jaxpr_unknown_resin,
                                           loop_invar_res)
  # Since not all inputs are used in jaxpr_unknown, we filter the input tracers
  # down using the output of `dce_jaxpr`.
  _, used_tracers = partition_list(used_refs, tracers)
  _, used_which_linear = partition_list(used_refs, which_linear)
  which_linear_unknown = (False,) * num_res + tuple(used_which_linear)
  unknown_inputs = [*residuals, *used_tracers]
  # Outputs match inputs so we construct output tracers that look like the input
  # tracers.
  res_ref_unknown_outputs = [
      pe.JaxprTracer(trace, pe.PartialVal.unknown(t.aval), None)
      for t in unknown_inputs]
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack):]
  source = source_info_util.current().replace(name_stack=name_stack)

  eqn = pe.new_eqn_recipe(unknown_inputs, res_ref_unknown_outputs,
                          for_p, dict(jaxpr=jaxpr_unknown, nsteps=nsteps,
                                      reverse=reverse,
                                      which_linear=which_linear_unknown),
                          core.no_effects, source)
  _, unknown_outputs = split_list(res_ref_unknown_outputs, [num_res])
  for t in unknown_outputs: t.recipe = eqn
  return merge_lists(in_unknowns, known_outputs, unknown_outputs)
pe.custom_partial_eval_rules[for_p] = _for_partial_eval

def _convert_outputs_to_writes(
    nsteps: int, jaxpr: core.Jaxpr, loop_invar_res: Sequence[bool]
    ) -> Tuple[core.Jaxpr, List[core.ShapedArray]]:
  assert not jaxpr.constvars, "Jaxpr shouldn't have constvars."

  in_avals = [v.aval for v in jaxpr.invars]  # [i, *orig_ref_avals]
  @lu.wrap_init
  def eval_jaxpr(i, *refs):
    # We split the refs into the original input refs and the dummy residual
    # refs.
    orig_refs, residual_refs = split_list(refs, [len(in_avals) - 1])
    residual_vals = core.eval_jaxpr(jaxpr, (), i, *orig_refs)
    for res_ref, res_val, loop_invar in zip(residual_refs, residual_vals,
                                            loop_invar_res):
      if loop_invar:
        res_ref[()] = res_val
      else:
        res_ref[i] = res_val
    return []
  res_ref_avals = [ShapedArrayRef(v.aval.shape, v.aval.dtype)  # pytype: disable=attribute-error
                   if loop_invar else
                   ShapedArrayRef((nsteps, *v.aval.shape), v.aval.dtype)
                   for v, loop_invar in zip(jaxpr.outvars, loop_invar_res)]
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [*in_avals, *res_ref_avals])
  assert not consts
  return jaxpr, [core.ShapedArray(a.shape, a.dtype) for a in res_ref_avals]

def _convert_inputs_to_reads(
    nsteps: int, num_res: int, jaxpr: core.Jaxpr,
    loop_invar_res: Sequence[bool]) -> core.Jaxpr:
  assert not jaxpr.constvars, "Jaxpr should not have constvars"

  @lu.wrap_init
  def eval_jaxpr(i, *refs):
    residual_refs, orig_refs = split_list(refs, [num_res])
    residual_vals = [r[()] if loop_invar else r[i] for r, loop_invar
                     in zip(residual_refs, loop_invar_res)]
    () = core.eval_jaxpr(jaxpr, (), *residual_vals, i, *orig_refs)
    return []

  res_val_avals, (i_aval,), orig_ref_avals = \
      split_list([v.aval for v in jaxpr.invars], [num_res, 1])
  res_ref_avals = [ShapedArrayRef(aval.shape, aval.dtype) if loop_invar else
                   ShapedArrayRef((nsteps, *aval.shape), aval.dtype)  # pytype: disable=attribute-error
                   for aval, loop_invar in zip(res_val_avals, loop_invar_res)]

  jaxpr, _, () = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [i_aval, *res_ref_avals, *orig_ref_avals])
  return jaxpr

### Testing utility

def discharged_for_loop(nsteps, body, init_state, *, reverse: bool = False):
  """A `for_loop` implementation that discharges its body right away.

  Potentially useful for testing and benchmarking.
  """
  flat_state, state_tree = tree_flatten(init_state)
  state_avals = map(val_to_ref_aval, flat_state)
  idx_aval = core.ShapedArray((), jnp.dtype("int32"))
  jaxpr, consts, out_tree = _trace_to_jaxpr_with_refs(
      body, state_tree, [idx_aval, *state_avals])
  if out_tree != tree_structure(None):
    raise Exception("`body` should not return anything.")
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, consts)

  def fori_body(i, carry):
    i = jnp.int32(i)
    if reverse:
      i = nsteps - i - 1
    out_flat = core.eval_jaxpr(discharged_jaxpr, discharged_consts,
                               i, *carry)
    return out_flat
  out_flat = loops.fori_loop(0, nsteps, fori_body, flat_state)
  return tree_unflatten(state_tree, out_flat)
