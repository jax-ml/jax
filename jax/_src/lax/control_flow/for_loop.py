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
"""Module for the `for_loop` primitive."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
import operator
from typing import Any, Generic, TypeVar

from jax._src import api_util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (tree_flatten, tree_structure, tree_unflatten,
                                treedef_tuple, tree_map, tree_leaves, PyTreeDef)

from jax._src import ad_util
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src.lax import lax
from jax._src.state.types import (AbstractRef)
from jax._src.state import discharge as state_discharge
from jax._src.state import primitives as state_primitives
from jax._src.state import utils as state_utils
from jax._src.state import types as state_types
from jax._src.typing import Array
from jax._src.util import (safe_map, safe_zip,
                           split_list, weakref_lru_cache)
from jax._src.lax.control_flow import loops
from jax._src.lax.control_flow.common import _initial_style_jaxpr

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

## Helpful type aliases
S = TypeVar('S')
T = TypeVar('T')
class Ref(Generic[T]): pass

ref_set = state_primitives.ref_set
ref_get = state_primitives.ref_get
ref_addupdate = state_primitives.ref_addupdate
discharge_state = state_discharge.discharge_state


## `for_loop` implementation

for_p = core.Primitive('for')
for_p.multiple_results = True
for_p.skip_canonicalization = True

### Tracing utilities

def _trace_to_jaxpr_with_refs(f: Callable, state_tree: PyTreeDef,
                              state_avals: Sequence[core.AbstractValue],
                              debug_info: core.DebugInfo,
                              ) -> tuple[core.Jaxpr, list[Any], PyTreeDef]:
  f, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(f, debug_info=debug_info),
      treedef_tuple((tree_structure(0), state_tree)))
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      f, state_avals)
  return jaxpr, consts, out_tree_thunk()

def for_loop(nsteps: int | Sequence[int],
             body: Callable[[Array, Ref[S]], None], init_state: S,
             *, reverse: bool = False, unroll: int = 1) -> S:
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
    unroll: A positive int specifying, in the underlying operation of the
      `for` primitive, how many iterations to unroll within a single iteration
      of a loop. Higher values may speed up execution time at the cost of longer
      compilation time.
  Returns:
    A Pytree of values representing the output of the for loop.
  """
  if unroll < 1:
    raise ValueError("`unroll` must be a positive integer.")
  if isinstance(nsteps, int):
    nsteps = [nsteps]
  if len(nsteps) > 1:
    outer_step, *rest_steps = nsteps
    def wrapped_body(i, refs):
      vals = tree_map(lambda ref: ref_get(ref, ()), refs)
      vals = for_loop(
          rest_steps, functools.partial(body, i), vals, unroll=unroll)
      tree_map(lambda ref, val: ref_set(ref, (), val), refs, vals)
    return for_loop(outer_step, wrapped_body, init_state, unroll=unroll)
  dbg = api_util.debug_info("for_loop", body, (0, init_state), {})
  nsteps, = nsteps
  flat_state, state_tree = tree_flatten(init_state)
  state_avals = map(state_utils.val_to_ref_aval, flat_state)
  idx_aval = core.ShapedArray((), dtypes.default_int_dtype())
  jaxpr, consts, out_tree = _trace_to_jaxpr_with_refs(
      body, state_tree, [idx_aval, *state_avals], dbg)
  if out_tree != tree_structure(None):
    raise Exception("`body` should not return anything.")
  jaxpr = state_utils.hoist_consts_to_refs(jaxpr, index=1)
  which_linear = (False,) * (len(consts) + len(flat_state))
  out_flat = for_p.bind(*consts, *flat_state, jaxpr=jaxpr, nsteps=int(nsteps),
                        reverse=reverse, which_linear=which_linear,
                        unroll=unroll)
  # Consts are `Ref`s so they are both inputs and outputs. We remove them from
  # the outputs.
  out_flat = out_flat[len(consts):]
  return tree_unflatten(state_tree, out_flat)

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

def scan(f: Callable[[Carry, X], tuple[Carry, Y]],
         init: Carry,
         xs: X | None = None,
         length: int | None = None,
         reverse: bool = False,
         unroll: int = 1) -> tuple[Carry, Y]:
  if not callable(f):
    raise TypeError("scan: f argument should be a callable.")
  if unroll < 1:
    raise ValueError("`unroll` must be a positive integer.")
  xs_flat, _ = tree_flatten(xs)

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

  x_shapes = [x.shape[1:] for x in xs_flat]
  x_dtypes = [dtypes.canonicalize_dtype(x.dtype) for x in xs_flat]
  x_avals = tuple(map(core.ShapedArray, x_shapes, x_dtypes))

  init_flat = tree_leaves(init)
  _, in_tree = tree_flatten((init, xs))
  dbg = api_util.debug_info("scan", f, (init, xs), {})
  carry_avals = tuple(map(core.get_aval, init_flat))
  jaxpr, consts, out_tree = _initial_style_jaxpr(
      f, in_tree, carry_avals + x_avals, dbg)
  del carry_avals

  _, ys_avals = tree_unflatten(out_tree, jaxpr.out_avals)
  ys = tree_map(lambda aval: lax.full([length, *aval.shape], 0, aval.dtype),
                ys_avals)
  def for_body(i, refs):
    carry_refs, xs_refs, ys_refs = refs
    flat_carry_refs = tree_leaves(carry_refs)
    flat_xs_refs = tree_leaves(xs_refs)
    flat_ys_refs = tree_leaves(ys_refs)
    flat_outs = core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts,
                                * consts,
                                * map(lambda x: x[()], flat_carry_refs),
                                * map(lambda x: x[i], flat_xs_refs))
    flat_carry_out, flat_ys_out = split_list(flat_outs, [len(flat_carry_refs)])
    map(lambda c_ref, c: ref_set(c_ref, (), c), flat_carry_refs, flat_carry_out)
    map(lambda y_ref, y: ref_set(y_ref, (i,), y), flat_ys_refs, flat_ys_out)

  assert isinstance(length, int)
  api_util.save_wrapped_fun_debug_info(for_body, dbg)
  init, _, ys = for_loop(length, for_body, (init, xs, ys), reverse=reverse,
                         unroll=unroll)
  return init, ys

@for_p.def_effectful_abstract_eval
def _for_abstract_eval(*avals, jaxpr, **__):
  # Find out for each of the `Ref`s in our jaxpr what effects they have.
  jaxpr_aval_effects = state_types.get_ref_state_effects(
      [v.aval for v in jaxpr.invars], jaxpr.effects)[1:]
  aval_effects = [{eff.replace(input_index=eff.input_index - 1)
                      for eff in effs} for aval, effs
                  in zip(avals, jaxpr_aval_effects)
                  if isinstance(aval, AbstractRef)]
  nonlocal_state_effects = core.join_effects(*aval_effects)
  return list(avals), nonlocal_state_effects

@state_discharge.register_discharge_rule(for_p)
def _for_discharge_rule(in_avals, _, *args: Any, jaxpr: core.Jaxpr,
                        reverse: bool, which_linear: Sequence[bool],
                        nsteps: int, unroll: int
                        ) -> tuple[Sequence[Any | None], Sequence[Any]]:
  out_vals = for_p.bind(*args, jaxpr=jaxpr, reverse=reverse,
                        which_linear=which_linear, nsteps=nsteps,
                        unroll=unroll)
  new_invals = []
  for aval, out_val in zip(in_avals, out_vals):
    new_invals.append(out_val if isinstance(aval, AbstractRef) else None)
  return new_invals, out_vals

def _for_impl(*args, jaxpr, nsteps, reverse, which_linear, unroll):
  del which_linear
  discharged_jaxpr, consts = discharge_state(jaxpr, ())
  def body(i, state):
    i_ = nsteps - i - 1 if reverse else i
    return core.eval_jaxpr(discharged_jaxpr, consts, i_, *state)
  return _for_impl_unrolled(body, nsteps, unroll, *args)

def _for_impl_unrolled(body, nsteps, unroll, *args):
  remainder = nsteps % unroll
  i = lax.full((), 0, dtypes.default_int_dtype())
  state = list(args)

  for _ in range(remainder):
    state = body(i, state)
    i = i + 1

  def cond(carry):
    i, _ = carry
    return i < nsteps
  def while_body(carry):
    i, state = carry
    for _ in range(unroll):
      state = body(i, state)
      i = i + 1
    return i, state
  _, state = loops.while_loop(cond, while_body, (i, state))
  return state

mlir.register_lowering(for_p, mlir.lower_fun(_for_impl, multiple_results=True))
for_p.def_impl(functools.partial(dispatch.apply_primitive, for_p))

@weakref_lru_cache
def _cached_for_jaxpr(jaxpr):
  discharged_jaxpr, body_consts = discharge_state(jaxpr, ())
  return core.ClosedJaxpr(discharged_jaxpr, body_consts)

def _for_vmap(axis_data, args, dims, *,
              jaxpr, nsteps, reverse, which_linear, unroll):
  init_batched = [d is not batching.not_mapped for d in dims]
  closed_jaxpr = _cached_for_jaxpr(jaxpr)
  batched = init_batched
  for _ in range(len(batched)):
    _, out_batched = batching.batch_jaxpr(
        closed_jaxpr, axis_data, [False] + batched, instantiate=batched)
    if out_batched == batched:
      break
    batched = map(operator.or_, batched, out_batched)
  else:
    raise Exception("Invalid fixpoint")
  args = [batching.broadcast(x, axis_data.size, 0, axis_data.explicit_mesh_axis)
          if now_bat and not was_bat
          else batching.moveaxis(x, d, 0) if now_bat else x
          for x, d, was_bat, now_bat in zip(args, dims, init_batched, batched)]
  batched_jaxpr_, _ = batching.batch_jaxpr(
      pe.close_jaxpr(jaxpr), axis_data, [False] + batched, [])
  batched_jaxpr, () = batched_jaxpr_.jaxpr, batched_jaxpr_.consts  # TODO consts
  out_flat = for_p.bind(*args, jaxpr=batched_jaxpr, nsteps=nsteps,
                        reverse=reverse, which_linear=which_linear,
                        unroll=unroll)
  return out_flat, [0 if b else batching.not_mapped for b in batched]
batching.fancy_primitive_batchers[for_p] = _for_vmap

def _for_jvp(primals, tangents, *, jaxpr, nsteps, reverse, which_linear,
             unroll):
  nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  # We need to find out which `Ref`s have nonzero tangents after running the
  # for loop. Ordinarily we do this with a fixed point on the body jaxpr but
  # a `for` body jaxpr is stateful and has no outputs. We therefore discharge
  # the state effect from the jaxpr and we will now have a "symmetric" jaxpr
  # where the inputs line up with the outputs. We use this discharged jaxpr
  # for the fixed point.
  closed_jaxpr = _cached_for_jaxpr(jaxpr)
  for _ in range(len(nonzero_tangents)):
    _, out_nonzero_tangents = ad.jvp_jaxpr(
        closed_jaxpr,
        [False] + nonzero_tangents, instantiate=nonzero_tangents)
    if out_nonzero_tangents == nonzero_tangents:
      break
    nonzero_tangents = map(operator.or_, nonzero_tangents, out_nonzero_tangents)
  else:
    raise Exception("Invalid fixpoint")
  tangents = [ad.instantiate_zeros(t) if inst else t
              for t, inst in zip(tangents, nonzero_tangents)]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  closed_jaxpr = pe.close_jaxpr(jaxpr)
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, [False] + nonzero_tangents, [])
  jvp_jaxpr, () = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts  # TODO consts
  jvp_which_linear = which_linear + (True,) * len(tangents)
  out_flat = for_p.bind(*primals, *tangents, jaxpr=jvp_jaxpr,
                        nsteps=nsteps, reverse=reverse,
                        which_linear=jvp_which_linear, unroll=unroll)
  # `out_flat` includes constant inputs into the `for_loop` which are converted
  # into outputs as well. We don't care about these in AD so we throw them out.
  out_primals, out_tangents = split_list(out_flat, [len(primals)])
  out_tangents_iter = iter(out_tangents)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_primal_value(p)
                  for p, nz in zip(out_primals, nonzero_tangents)]
  return out_primals, out_tangents
ad.primitive_jvps[for_p] = _for_jvp

### Testing utility

def discharged_for_loop(nsteps, body, init_state, *, reverse: bool = False):
  """A `for_loop` implementation that discharges its body right away.

  Potentially useful for testing and benchmarking.
  """
  flat_state, state_tree = tree_flatten(init_state)
  state_avals = map(state_utils.val_to_ref_aval, flat_state)
  idx_aval = core.ShapedArray((), dtypes.default_int_dtype())
  debug = api_util.debug_info("discharged_for_loop", body, (0, init_state), {})
  jaxpr, consts, out_tree = _trace_to_jaxpr_with_refs(
      body, state_tree, [idx_aval, *state_avals], debug)
  if out_tree != tree_structure(None):
    raise Exception("`body` should not return anything.")
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, consts)

  def fori_body(i, carry):
    i = lax.convert_element_type(i, dtypes.default_int_dtype())
    if reverse:
      i = nsteps - i - 1
    out_flat = core.eval_jaxpr(discharged_jaxpr, discharged_consts,
                               i, *carry)
    return out_flat
  out_flat = loops.fori_loop(0, nsteps, fori_body, flat_state)
  return tree_unflatten(state_tree, out_flat)

def run_state(f, init_state):
  @functools.wraps(f)
  def wrapped_body(_, *args):
    return f(*args)
  return for_loop(1, wrapped_body, init_state)
