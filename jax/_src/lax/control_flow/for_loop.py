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

from typing import Any, Callable, Generic, List, Optional, Sequence, Tuple, TypeVar

from jax import core
from jax import lax
from jax import linear_util as lu
from jax.api_util import flatten_fun_nokwargs
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.tree_util import (tree_flatten, tree_structure, tree_unflatten,
                           treedef_tuple, tree_map, tree_leaves, PyTreeDef)
from jax._src import ad_util
from jax._src import dtypes
from jax._src import source_info_util
from jax._src import state
from jax._src.util import (partition_list, merge_lists, safe_map, safe_zip,
                           split_list)
import jax.numpy as jnp
import numpy as np

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

StateEffect = state.StateEffect
ShapedArrayRef = state.ShapedArrayRef
ref_set = state.ref_set
ref_get = state.ref_get
ref_addupdate = state.ref_addupdate
discharge_state = state.discharge_state


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

  x_shapes = [x.shape[1:] for x in xs_flat]
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

def _for_vmap(axis_size, axis_name, main_type, args, dims, *,
              jaxpr, nsteps, reverse, which_linear):
  init_batched = [d is not batching.not_mapped for d in dims]
  discharged_jaxpr, body_consts = discharge_state(jaxpr, ())
  batched = init_batched
  for _ in range(len(batched)):
    _, out_batched = batching.batch_jaxpr(
        core.ClosedJaxpr(discharged_jaxpr, body_consts),
        axis_size, [False] + batched, instantiate=batched,
        axis_name=axis_name, main_type=main_type)
    if out_batched == batched:
      break
    batched = map(operator.or_, batched, out_batched)
  else:
    raise Exception("Invalid fixpoint")
  args = [batching.broadcast(x, axis_size, 0) if now_bat and not was_bat
          else batching.moveaxis(x, d, 0) if now_bat else x
          for x, d, was_bat, now_bat in zip(args, dims, init_batched, batched)]
  batched_jaxpr_, _ = batching.batch_jaxpr(
      core.ClosedJaxpr(jaxpr, []), axis_size, [False] + batched, [],
      axis_name=axis_name, main_type=main_type)
  batched_jaxpr, () = batched_jaxpr_.jaxpr, batched_jaxpr_.consts  # TODO consts
  out_flat = for_p.bind(*args, jaxpr=batched_jaxpr, nsteps=nsteps,
                        reverse=reverse, which_linear=which_linear)
  return out_flat, [0 if b else batching.not_mapped for b in batched]
batching.axis_primitive_batchers[for_p] = _for_vmap

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
  tangents = [ad.instantiate_zeros(t) if inst else t
              for t, inst in zip(tangents, nonzero_tangents)]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  closed_jaxpr = core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, [False] + nonzero_tangents, [])
  jvp_jaxpr, () = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts  # TODO consts
  jvp_which_linear = which_linear + (True,) * len(tangents)
  out_flat = for_p.bind(*primals, *tangents, jaxpr=jvp_jaxpr,
                        nsteps=nsteps, reverse=reverse,
                        which_linear=jvp_which_linear)
  # `out_flat` includes constant inputs into the `for_loop` which are converted
  # into outputs as well. We don't care about these in AD so we throw them out.
  out_primals, out_tangents = split_list(out_flat, [len(primals)])
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
                      which_linear: Tuple[bool, ...]) -> List[pe.JaxprTracer]:
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
    _, _, out_unknowns, _, _, = pe.partial_eval_jaxpr_custom(
        discharged_jaxpr, jaxpr_in_unknowns, [True] * len(jaxpr_in_unknowns),
          in_unknowns, False, _save_everything)
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
  # We assume the known inputs are nonlinear which is okay to do for AD but not
  # necessarily okay for general partial eval.
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

def transpose_jaxpr(jaxpr: core.Jaxpr, which_linear: List[bool]) -> core.Jaxpr:
  def trans(i, *args):
    # First we want to run the computation to read all the residual refs. We can
    # do that by using partial evaluation with all linear inputs unknown.
    res_jaxpr, tangent_jaxpr_, *_ = \
        _partial_eval_jaxpr_custom(jaxpr, [False, *which_linear],
                                   _save_everything)
    res_args = [x for x, lin in zip(args, which_linear) if not lin]
    res = core.eval_jaxpr(res_jaxpr, (), i, *res_args)

    # Now that we have residual values, we run the tangent jaxpr. It takes as
    # input the residuals, the loop index, and all the refs (at least, the ones
    # that are used in the body). Luckily, `tangent_jaxpr_` has all known and
    # unknown inputs!
    tangent_jaxpr, used = pe.dce_jaxpr(tangent_jaxpr_, [])
    used_res, (used_i,), used_ct = split_list(used, [len(res), 1])
    primals_args = [*(r for u, r in zip(used_res, res) if u)]
    if used_i:
      primals_args = [*primals_args, i]
    ct_args = [x for x, u in zip(args, used_ct) if u]
    ad.backward_pass(
        tangent_jaxpr, (), False, (), (*primals_args, *ct_args), ())
    return []
  jaxpr_trans, _, _ = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(trans), [v.aval for v in jaxpr.invars])
  return jaxpr_trans

def _for_transpose(in_cts, *args, jaxpr, nsteps, reverse, which_linear):
  # if any in_ct is nonzero, we definitely want it in args_ (and the
  # corresponding x in args could be an undefined primal, but doesnt have to be)
  # for non-res stuff:
  #   getting and setting => (nonzero ct, UndefinedPrimal arg)
  #   just setting =>        (nonzero ct, not UndefinedPrimal, dummy value)
  #   just getting =>        (zero ct   , UndefinedPrimal arg)
  # for res stuff:
  #                          (zero ct   , not UndefinedPrimal)
  args_ = []
  which_linear_transpose = []
  for x, ct in zip(args, in_cts):
    if   type(ct) is     ad_util.Zero and not ad.is_undefined_primal(x):
      # this is a residual, take x!
      args_.append(x)
      which_linear_transpose.append(False)
    elif type(ct) is     ad_util.Zero and     ad.is_undefined_primal(x):
      # the loop was 'just getting', plug in a zero
      args_.append(ad_util.zeros_like_aval(x.aval))
      which_linear_transpose.append(False)
    elif type(ct) is not ad_util.Zero and not ad.is_undefined_primal(x):
      # the loop was 'just setting', grab that cotangent! x is dummy
      args_.append(ct)
      which_linear_transpose.append(False)
    elif type(ct) is not ad_util.Zero and     ad.is_undefined_primal(x):
      # the loop was 'getting and setting', grab that cotangent!
      args_.append(ct)
      which_linear_transpose.append(True)

  jaxpr_transpose = transpose_jaxpr(jaxpr, which_linear)
  assert len(args_) == len(jaxpr_transpose.invars) - 1
  all_outs = for_p.bind(*args_, jaxpr=jaxpr_transpose, nsteps=nsteps,
                        reverse=not reverse,
                        which_linear=tuple(which_linear_transpose))
  ct_outs = [ct if ad.is_undefined_primal(x) else None
             for x, ct in zip(args, all_outs)]
  return ct_outs
ad.primitive_transposes[for_p] = _for_transpose

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
