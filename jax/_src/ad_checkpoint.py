# Copyright 2021 Google LLC
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

from functools import partial
import operator as op
from typing import Callable, Optional, List, Tuple
import types

import jax
from jax import core
from jax import linear_util as lu
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
from jax.interpreters import mlir
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src import ad_util
from jax._src import source_info_util
from jax._src.api_util import flatten_fun
from jax._src.traceback_util import api_boundary
from jax._src.util import (unzip2, wraps, split_list, partition_list, safe_map,
                           safe_zip, merge_lists)

source_info_util.register_exclusion(__file__)

# TODO(mattjj): before this can be the standard remat implementation, we must:
#   [ ] fix up callers who use the 'concrete' option (now removed)
#   [ ] implement remat-of-control-flow-primitives (passing through the policy)

map = safe_map
zip = safe_zip


### Policies

def everything_saveable(*_, **__) -> bool:
  # This is the effective policy without any use of jax.remat.
  return True

def nothing_saveable(*_, **__) -> bool:
  # This is the effective policy when using jax.remat without explicit policy.
  return False

def checkpoint_dots(prim, *_, **__) -> bool:
  # Matrix multiplies are expensive, so let's save them (and nothing else).
  return prim in {jax._src.lax.lax.dot_general_p,
                  jax._src.lax.convolution.conv_general_dilated_p}

def dot_with_no_batch_dims(prim, *_, **params) -> bool:
  # This is a useful heuristic for transformers.
  if prim is jax._src.lax.lax.dot_general_p:
    (_, _), (lhs_b, rhs_b) = params['dimension_numbers']
    if not lhs_b and not rhs_b:
      return True
  return False

name_p = core.Primitive('name')

def save_any_names_but_these(*names_not_to_save):
  # Save named values, excluding the names given.
  names_not_to_save = frozenset(names_not_to_save)
  def policy(prim, *_, **params):
    if prim is name_p:
      return params['name'] not in names_not_to_save
    return False  # only allow saving named values
  return policy

def save_only_these_names(*names_which_can_be_saved):
  # Save named values, only among the names given.
  names_which_can_be_saved = set(names_which_can_be_saved)
  def policy(prim, *_, **params):
    if prim is name_p:
      return params['name'] in names_which_can_be_saved
    return False  # not saveable unless it's in the allow-list
  return policy

checkpoint_policies = types.SimpleNamespace(
    everything_saveable=everything_saveable,
    nothing_saveable=nothing_saveable,
    checkpoint_dots=checkpoint_dots,
    checkpoint_dots_with_no_batch_dims=dot_with_no_batch_dims,
    save_any_names_but_these=save_any_names_but_these,
    save_only_these_names=save_only_these_names,
)


### Main API

def checkpoint(fun: Callable, prevent_cse: bool = True,
               policy: Optional[Callable[..., bool]] = None
               ) -> Callable:
  """Make ``fun`` recompute internal linearization points when differentiated.

  The :func:`jax.checkpoint` decorator, aliased to ``jax.remat``, provides a
  way to trade off computation time and memory cost in the context of automatic
  differentiation, especially with reverse-mode autodiff like :func:`jax.grad`
  and :func:`jax.vjp` but also with :func:`jax.linearize`.

  When differentiating a function in reverse-mode, by default all the
  linearization points (e.g. inputs to elementwise nonlinear primitive
  operations) are stored when evaluating the forward pass so that they can be
  reused on the backward pass. This evaluation strategy can lead to a high
  memory cost, or even to poor performance on hardware accelerators where memory
  access is much more expensive than FLOPs.

  An alternative evaluation strategy is for some of the linearization points to
  be recomputed (i.e. rematerialized) rather than stored. This approach can
  reduce memory usage at the cost of increased computation.

  This function decorator produces a new version of ``fun`` which follows
  the rematerialization strategy rather than the default store-everything
  strategy. That is, it returns a new version of ``fun`` which, when
  differentiated, doesn't store any of its intermediate linearization points.
  Instead, these linearization points are recomputed from the function's saved
  inputs.

  See the examples below.

  Args:
    fun: Function for which the autodiff evaluation strategy is to be changed
      from the default of storing all intermediate linearization points to
      recomputing them. Its arguments and return value should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
    concrete: Optional, boolean indicating whether ``fun`` may involve
      value-dependent Python control flow (default False). Support for such
      control flow is optional, and disabled by default, because in some
      edge-case compositions with :func:`jax.jit` it can lead to some extra
      computation.
    prevent_cse: Optional, boolean indicating whether to prevent common
      subexpression elimination (CSE) optimizations in the HLO generated from
      differentiation. This CSE prevention has costs because it can foil other
      optimizations, and because it can incur high overheads on some backends,
      especially GPU. The default is True because otherwise, under a ``jit`` or
      ``pmap``, CSE can defeat the purpose of this decorator. But in some
      settings, like when used inside a ``scan``, this CSE prevention mechanism
      is unnecessary, in which case ``prevent_cse`` can be set to False.
    policy: This is an experimental feature and the API is likely to change.
      Optional callable, one of the attributes of ``jax.checkpoint_policies``,
      which takes as input a type-level specification of a first-order primitive
      application and returns a boolean indicating whether the corresponding
      output value(s) can be saved as a residual (or, if not, instead must be
      recomputed in the (co)tangent computation).

  Returns:
    A function (callable) with the same input/output behavior as ``fun`` but
    which, when differentiated using e.g. :func:`jax.grad`, :func:`jax.vjp`, or
    :func:`jax.linearize`, recomputes rather than stores intermediate
    linearization points, thus potentially saving memory at the cost of extra
    computation.

  Here is a simple example:

  >>> import jax
  >>> import jax.numpy as jnp

  >>> @jax.checkpoint
  ... def g(x):
  ...   y = jnp.sin(x)
  ...   z = jnp.sin(y)
  ...   return z
  ...
  >>> jax.value_and_grad(g)(2.0)
  (DeviceArray(0.78907233, dtype=float32, weak_type=True), DeviceArray(-0.2556391, dtype=float32, weak_type=True))

  Here, the same value is produced whether or not the :func:`jax.checkpoint`
  decorator is present. When the decorator is not present, the values
  ``jnp.cos(2.0)`` and ``jnp.cos(jnp.sin(2.0))`` are computed on the forward
  pass and are stored for use in the backward pass, because they are needed
  on the backward pass and depend only on the primal inputs. When using
  :func:`jax.checkpoint`, the forward pass will compute only the primal outputs
  and only the primal inputs (``2.0``) will be stored for the backward pass.
  At that time, the value ``jnp.sin(2.0)`` is recomputed, along with the values
  ``jnp.cos(2.0)`` and ``jnp.cos(jnp.sin(2.0))``.

  While ``jax.checkpoint`` controls what values are stored from the forward-pass
  to be used on the backward pass, the total amount of memory required to
  evaluate a function or its VJP depends on many additional internal details of
  that function. Those details include which numerical primitives are used,
  how they're composed, where jit and control flow primitives like scan
  are used, and other factors.

  The :func:`jax.checkpoint` decorator can be applied recursively to express
  sophisticated autodiff rematerialization strategies. For example:

  >>> def recursive_checkpoint(funs):
  ...   if len(funs) == 1:
  ...     return funs[0]
  ...   elif len(funs) == 2:
  ...     f1, f2 = funs
  ...     return lambda x: f1(f2(x))
  ...   else:
  ...     f1 = recursive_checkpoint(funs[:len(funs)//2])
  ...     f2 = recursive_checkpoint(funs[len(funs)//2:])
  ...     return lambda x: f1(jax.checkpoint(f2)(x))
  ...
  """
  @wraps(fun)
  @api_boundary
  def fun_remat(*args, **kwargs):
    args_flat, in_tree = tree_flatten((args, kwargs))
    flat_fun, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
    in_avals = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
    debug = pe.debug_info(fun, in_tree, False, "checkpoint")
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals, debug)
    out_flat = remat_p.bind(
        *consts, *args_flat, jaxpr=pe.convert_constvars_jaxpr(jaxpr),
        prevent_cse=prevent_cse, differentiated=False, policy=policy)
    return tree_unflatten(out_tree(), out_flat)
  return fun_remat

remat = checkpoint  # alias


### Utilities

def saved_residuals(f, *args, **kwargs) -> List[Tuple[core.AbstractValue, str]]:
  args, in_tree = tree_flatten((args, kwargs))

  def f_(*args):
    args, kwargs = tree_unflatten(in_tree, args)
    return f(*args, **kwargs)

  jaxpr = jax.make_jaxpr(lambda *args: jax.linearize(f_, *args)[1])(*args).jaxpr
  res_lits = [x for x in jaxpr.outvars if     isinstance(x, core.Literal)]
  res_vars = {x for x in jaxpr.outvars if not isinstance(x, core.Literal)}

  results = []

  for x in res_lits:
    results.append((x.aval, 'from a literal'))

  for v in jaxpr.constvars:
    if v in res_vars:
      results.append((v.aval, 'from a constant'))

  assert len(jaxpr.invars) == len(args)
  for i, v in enumerate(jaxpr.invars):
    if v in res_vars:
      src = f'from {pe.arg_info_pytree(f, in_tree, True, [i])}'
      results.append((v.aval, src))

  for eqn in jaxpr.eqns:
    src = source_info_util.summarize(eqn.source_info)
    for v in eqn.outvars:
      if v in res_vars:
        if eqn.primitive is name_p:
          results.append((v.aval, f"named '{eqn.params['name']}' from {src}"))
        else:
          results.append((v.aval, f'from {src}'))

  assert len(results) == len(jaxpr.outvars)
  return results

def print_saved_residuals(f, *args, **kwargs):
  for aval, src in saved_residuals(f, *args, **kwargs):
    print(f'{aval.str_short(short_dtypes=True)} {src}')


### Implementation

remat_p = core.Primitive('remat2')
remat_p.multiple_results = True

@remat_p.def_impl
def remat_impl(*args, jaxpr, prevent_cse, differentiated, policy):
  del prevent_cse, differentiated, policy  # Unused.
  return core.eval_jaxpr(jaxpr, (), *args)

@remat_p.def_effectful_abstract_eval
def remat_abstract_eval(*args, jaxpr, prevent_cse, differentiated, policy):
  del args, prevent_cse, differentiated, policy  # Unused.
  return [v.aval for v in jaxpr.outvars], jaxpr.effects

def remat_jvp(primals, tangents, jaxpr, prevent_cse, differentiated, policy):
  assert not jaxpr.constvars
  in_nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  jaxpr_ = core.ClosedJaxpr(jaxpr, ())
  jaxpr_jvp_, out_nonzeros = ad.jvp_jaxpr(jaxpr_, in_nonzeros, False)
  nonzero_tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  jaxpr_jvp = pe.convert_constvars_jaxpr(jaxpr_jvp_.jaxpr)
  outs = remat_p.bind(
      *jaxpr_jvp_.consts, *primals, *nonzero_tangents, jaxpr=jaxpr_jvp,
      prevent_cse=prevent_cse, differentiated=differentiated, policy=policy)
  out_primals, out_tangents_ = split_list(outs, [len(jaxpr.outvars)])
  out_tangents_ = iter(out_tangents_)
  out_tangents = [next(out_tangents_) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(out_primals, out_nonzeros)]
  return out_primals, out_tangents
ad.primitive_jvps[remat_p] = remat_jvp

def remat_partial_eval(trace, *tracers, jaxpr, **params):
  assert not jaxpr.constvars
  if jaxpr.effects:
    raise NotImplementedError(
        'Effects not supported in partial-eval of `checkpoint`/`remat`.')
  policy = params['policy'] or nothing_saveable
  in_unknowns = [not t.is_known() for t in tracers]
  jaxpr_known, jaxpr_staged, out_unknowns, out_inst, num_res = \
      pe.partial_eval_jaxpr_custom(
          jaxpr, in_unknowns, [True] * len(in_unknowns), False, False, policy)

  # DCE jaxpr_staged, keeping only instantiated outputs which are unknown
  _, out_inst_unknown = partition_list(out_inst, out_unknowns)
  jaxpr_unknown, in_used_staged = pe.dce_jaxpr(jaxpr_staged, out_inst_unknown)
  used_res, in_used_staged = split_list(in_used_staged, [num_res])

  # DCE jaxpr_known, keeping all known outputs but discarding dce'd res
  out_used_known = [True] * (len(out_unknowns) - sum(out_unknowns)) + used_res
  jaxpr_known, in_used_known = pe.dce_jaxpr(jaxpr_known, out_used_known)
  num_res = sum(used_res)

  # compute known outputs and residuals (hoisted out of remat primitive)
  _, in_consts_ = unzip2(t.pval for t in tracers if t.pval.is_known())
  _, in_consts = partition_list(in_used_known, in_consts_)
  out_consts = core.eval_jaxpr(jaxpr_known, (), *in_consts)
  out_knowns, residuals = split_list(out_consts, [len(out_consts)-num_res])

  # set up unknown outputs with a recipe to call remat
  res_tracers = map(trace.new_instantiated_const, residuals)
  _, tracers_staged = partition_list(in_used_staged, tracers)
  in_jaxpr_tracers = res_tracers + map(trace.instantiate_const, tracers_staged)
  out_jaxpr_tracers = [pe.JaxprTracer(trace, pe.PartialVal.unknown(x.aval), None)
                       for x in jaxpr_unknown.outvars]
  new_params = dict(params, jaxpr=jaxpr_unknown, differentiated=True)
  recipe = pe.new_eqn_recipe(in_jaxpr_tracers, out_jaxpr_tracers, remat_p,
                             new_params, jaxpr_unknown.effects,
                             source_info_util.current())
  for t in out_jaxpr_tracers: t.recipe = recipe

  # zip together known and unknown outputs
  return merge_lists(out_unknowns, out_knowns, out_jaxpr_tracers)
pe.custom_partial_eval_rules[remat_p] = remat_partial_eval

def remat_partial_eval_custom_params_updater(*args):
  *_, params_known, params_staged = args
  return params_known, dict(params_staged, differentiated=True)
pe.partial_eval_jaxpr_custom_rules[remat_p] = \
    partial(pe.call_partial_eval_custom_rule, 'jaxpr',
            remat_partial_eval_custom_params_updater)

def remat_transpose(reduce_axes, out_cts, *in_primals, jaxpr, **params):
  assert not jaxpr.constvars
  cell = lambda: None

  @lu.wrap_init
  def transposed(*args):
    in_primals, out_cts = tree_unflatten(treedef, args)
    in_pvals = [pe.PartialVal.unknown(x.aval) if ad.is_undefined_primal(x) else
                pe.PartialVal.known(x) for x in in_primals]
    primal_fun = lu.wrap_init(partial(core.eval_jaxpr, jaxpr, ()))
    t_jaxpr, _, consts = pe.trace_to_jaxpr_nounits(primal_fun, in_pvals, False)
    dummy_args = [ad.UndefinedPrimal(v.aval) for v in t_jaxpr.invars]
    in_cts = ad.backward_pass(t_jaxpr, reduce_axes, False, consts, dummy_args,
                              out_cts)
    in_cts_ = iter(in_cts)
    in_cts = [next(in_cts_) if ad.is_undefined_primal(x)
              else ad_util.Zero(x.aval) for x in in_primals]
    assert next(in_cts_, None) is None
    in_cts, cell.treedef = tree_flatten(in_cts)
    return in_cts

  args, treedef = tree_flatten((in_primals, out_cts))
  in_avals = [core.raise_to_shaped(core.get_aval(x)) for x in args]
  transposed_jaxpr_, _, consts = pe.trace_to_jaxpr_dynamic(transposed, in_avals)
  transposed_jaxpr = pe.convert_constvars_jaxpr(transposed_jaxpr_)
  in_cts = remat_p.bind(*consts, *args, jaxpr=transposed_jaxpr, **params)
  return tree_unflatten(cell.treedef, in_cts)  # type: ignore
ad.reducing_transposes[remat_p] = remat_transpose

def remat_vmap(axis_size, axis_name, main_type, args, dims, *, jaxpr, **params):
  assert not jaxpr.constvars
  jaxpr_ = core.ClosedJaxpr(jaxpr, ())
  jaxpr_batched_, out_batched = batching.batch_jaxpr_axes(
      jaxpr_, axis_size, dims, [batching.zero_if_mapped] * len(jaxpr.outvars),
      axis_name=axis_name, main_type=main_type)
  jaxpr_batched, consts = jaxpr_batched_.jaxpr, jaxpr_batched_.consts
  out_dims = [0 if b else None for b in out_batched]
  return remat_p.bind(*consts, *args, jaxpr=jaxpr_batched, **params), out_dims
batching.axis_primitive_batchers[remat_p] = remat_vmap

# TODO(mattjj,sharadmv): test this more
# TODO(mattjj,sharadmv): de-duplicate with pe.dce_jaxpr_call_rule
def remat_dce(used_outputs: List[bool], eqn: core.JaxprEqn
              ) -> Tuple[List[bool], Optional[core.JaxprEqn]]:
  new_jaxpr, used_inputs = pe.dce_jaxpr(eqn.params['jaxpr'], used_outputs)
  new_params = dict(eqn.params, jaxpr=new_jaxpr)
  if not any(used_inputs) and not any(used_outputs) and not new_jaxpr.effects:
    return used_inputs, None
  else:
    new_eqn = pe.new_jaxpr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs) if used],
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, new_jaxpr.effects, eqn.source_info)
    return used_inputs, new_eqn
pe.dce_rules[remat_p] = remat_dce


def checkpoint_name(x, name):
  return name_p.bind(x, name=name)

name_p.def_impl(lambda x, *, name: x)
name_p.def_abstract_eval(lambda x, *, name: x)

def name_jvp(primals, tangents, *, name):
  (x,), (xdot,) = primals, tangents
  return name_p.bind(x, name=name), xdot  # don't name the tangent value
ad.primitive_jvps[name_p] = name_jvp

mlir.register_lowering(name_p, lambda ctx, x, *, name: [x])

def name_batcher(args, dims, *, name):
  (x,), (d,) = args, dims
  return name_p.bind(x, name=name), d
batching.primitive_batchers[name_p] = name_batcher
