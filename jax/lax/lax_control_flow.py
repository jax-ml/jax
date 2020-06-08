# coding=utf-8
# Copyright 2019 Google LLC
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
"""
Control flow primitives.
"""


import collections
import functools
import inspect
import itertools
import operator
from typing import Callable, Sequence

import numpy as onp

import jax
from jax import core
from jax import dtypes
from jax import util
from jax.lax import lax
from jax import linear_util as lu
from jax.abstract_arrays import ConcreteArray, ShapedArray, raise_to_shaped
from jax.api_util import flatten_fun_nokwargs
from jax.core import get_aval, typecheck, typematch
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import batching
from jax.interpreters import masking
from jax.lib import xla_bridge as xb
from jax.lib import xla_client
from jax.util import (partial, unzip2, unzip4, safe_map, safe_zip, split_list,
                      split_dict, cache, extend_name_stack)
from jax.tree_util import (tree_flatten, tree_unflatten, treedef_is_leaf,
                           treedef_children, treedef_tuple, tree_multimap)
from jax import ad_util

xops = xla_client.ops

_map = safe_map
zip = safe_zip
_reduce = functools.reduce

@cache()
def _initial_style_untyped_jaxpr(fun: Callable, in_tree, in_avals):
  in_pvals = [pe.PartialVal.unknown(aval) for aval in in_avals]
  wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  with core.initial_style_staging():
    jaxpr, out_pvals, consts = pe.trace_to_jaxpr(
      wrapped_fun, in_pvals, instantiate=True, stage_out=False)
  return jaxpr, out_pvals, consts, out_tree

@cache()
def _initial_style_jaxpr(fun: Callable, in_tree, in_avals):
  jaxpr, out_pvals, consts, out_tree = _initial_style_untyped_jaxpr(
      fun, in_tree, in_avals)
  out_avals = _map(raise_to_shaped, unzip2(out_pvals)[0])
  const_avals = tuple(raise_to_shaped(core.get_aval(c)) for c in consts)
  typed_jaxpr = core.TypedJaxpr(pe.convert_constvars_jaxpr(jaxpr),
                                (), const_avals + in_avals, out_avals)
  return typed_jaxpr, consts, out_tree()

def _initial_style_jaxprs_with_common_consts(funs: Sequence[Callable],
                                             in_tree, in_avals):
  # When staging the branches of a conditional into jaxprs, constants are
  # extracted from each branch and converted to jaxpr arguments. To use the
  # staged jaxprs as the branches to a conditional *primitive*, we need for
  # their (input) signatures to match. This function "joins" the staged jaxprs:
  # for each one, it makes another that accepts *all* constants, but only uses
  # those that it needs (dropping the rest).

  jaxprs, all_out_pvals, all_consts, all_out_trees = unzip4([
      _initial_style_untyped_jaxpr(fun, in_tree, in_avals) for fun in funs])

  newvar = core.gensym(jaxprs, suffix='_')
  all_const_avals = tuple(
      tuple(raise_to_shaped(core.get_aval(c)) for c in consts)
      for consts in all_consts)
  unused_const_vars = tuple(
      tuple(newvar(aval) for aval in const_avals)
      for const_avals in all_const_avals)

  def pad_jaxpr_constvars(i, jaxpr):
    prefix = util.concatenate(unused_const_vars[:i])
    suffix = util.concatenate(unused_const_vars[i+1:])
    constvars = prefix + jaxpr.constvars + suffix
    return core.Jaxpr(constvars=constvars, invars=jaxpr.invars,
                      outvars=jaxpr.outvars, eqns=jaxpr.eqns)

  const_avals = tuple(util.concatenate(all_const_avals))

  def type_and_const_convert_jaxpr(jaxpr, out_pvals):
    out_avals = _map(raise_to_shaped, unzip2(out_pvals)[0])
    return core.TypedJaxpr(pe.convert_constvars_jaxpr(jaxpr),
                           (), const_avals + in_avals, out_avals)

  jaxprs = [pad_jaxpr_constvars(i, jaxpr) for i, jaxpr in enumerate(jaxprs)]
  typed_jaxprs = _map(type_and_const_convert_jaxpr, jaxprs, all_out_pvals)

  return (tuple(typed_jaxprs),
          tuple(util.concatenate(all_consts)),
          tuple(out_tree() for out_tree in all_out_trees))

def _abstractify(x):
  return raise_to_shaped(core.get_aval(x))

def _disable_jit_impl(prim, interp, *args, **kwargs):
  if jax.api._jit_is_disabled():
    return interp(*args, **kwargs)
  else:
    return xla.apply_primitive(prim, *args, **kwargs)


### fori_loop and while_loop

def _fori_cond_fun(loop_carry):
  i, upper, _ = loop_carry
  return lax.lt(i, upper)

@cache()
def _fori_body_fun(body_fun):
  def while_body_fun(loop_carry):
    i, upper, x = loop_carry
    return lax.add(i, lax._const(i, 1)), upper, body_fun(i, x)
  return while_body_fun

@cache()
def _fori_scan_body_fun(body_fun):
  def scanned_fun(loop_carry, _):
    i, upper, x = loop_carry
    return (lax.add(i, lax._const(i, 1)), upper, body_fun(i, x)), None
  return scanned_fun

def fori_loop(lower, upper, body_fun, init_val):
  """Loop from ``lower`` to ``upper`` by reduction to :func:`jax.lax.while_loop`.

  The type signature in brief is

  .. code-block:: haskell

    fori_loop :: Int -> Int -> ((int, a) -> a) -> a -> a

  The semantics of ``fori_loop`` are given by this Python implementation::

    def fori_loop(lower, upper, body_fun, init_val):
      val = init_val
      for i in range(lower, upper):
        val = body_fun(i, val)
      return val

  Unlike that Python version, ``fori_loop`` is implemented in terms of a call to
  :func:`jax.lax.while_loop`. See the :func:`jax.lax.while_loop` documentation
  for more information.

  Also unlike the Python analogue, the loop-carried value ``val`` must hold a
  fixed shape and dtype across all iterations (and not just be consistent up to
  NumPy rank/shape broadcasting and dtype promotion rules, for example). In
  other words, the type ``a`` in the type signature above represents an array
  with a fixed shape and dtype (or a nested tuple/list/dict container data
  structure with a fixed structure and arrays with fixed shape and dtype at the
  leaves).

  Args:
    lower: an integer representing the loop index lower bound (inclusive)
    upper: an integer representing the loop index upper bound (exclusive)
    body_fun: function of type ``(int, a) -> a``.
    init_val: initial loop carry value of type ``a``.

  Returns:
    Loop value from the final iteration, of type ``a``.
  """
  # TODO(phawkins): perhaps do more type checking here, better error messages.
  lower_dtype = dtypes.canonicalize_dtype(lax.dtype(lower))
  upper_dtype = dtypes.canonicalize_dtype(lax.dtype(upper))
  if lower_dtype != upper_dtype:
    msg = ("lower and upper arguments to fori_loop must have equal types, "
           "got {} and {}")
    raise TypeError(msg.format(lower_dtype.name, upper_dtype.name))

  # If we can specialize on the trip count, call scan instead of a while_loop
  # to enable efficient reverse-mode differentiation.
  try:
    lower_ = int(lower)
    upper_ = int(upper)
  except TypeError:
    use_scan = False
  else:
    use_scan = False  # TODO(mattjj): re-enable this

  if use_scan:
    (_, _, result), _ = scan(_fori_scan_body_fun(body_fun),
                             (lower, upper, init_val), None,
                             length=upper_ - lower_)
  else:
    _, _, result = while_loop(_fori_cond_fun, _fori_body_fun(body_fun),
                              (lower, upper, init_val))
  return result


def while_loop(cond_fun, body_fun, init_val):
  """Call ``body_fun`` repeatedly in a loop while ``cond_fun`` is True.

  The type signature in brief is

  .. code-block:: haskell

    while_loop :: (a -> Bool) -> (a -> a) -> a -> a

  The semantics of ``while_loop`` are given by this Python implementation::

    def while_loop(cond_fun, body_fun, init_val):
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val

  Unlike that Python version, ``while_loop`` is a JAX primitive and is lowered
  to a single XLA While HLO. That makes it useful for reducing compilation times
  for jit-compiled functions, since native Python loop constructs in an ``@jit``
  function are unrolled, leading to large XLA computations.

  Also unlike the Python analogue, the loop-carried value ``val`` must hold a
  fixed shape and dtype across all iterations (and not just be consistent up to
  NumPy rank/shape broadcasting and dtype promotion rules, for example). In
  other words, the type ``a`` in the type signature above represents an array
  with a fixed shape and dtype (or a nested tuple/list/dict container data
  structure with a fixed structure and arrays with fixed shape and dtype at the
  leaves).

  Another difference from using Python-native loop constructs is that
  ``while_loop`` is not reverse-mode differentiable because XLA computations
  require static bounds on memory requirements.

  Args:
    cond_fun: function of type ``a -> Bool``.
    body_fun: function of type ``a -> a``.
    init_val: value of type ``a``, a type that can be a scalar, array, or any
      pytree (nested Python tuple/list/dict) thereof, representing the initial
      loop carry value.

  Returns:
    The output from the final iteration of body_fun, of type ``a``.
  """
  if jax.api._jit_is_disabled():
    try:
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val
    except core.ConcretizationTypeError:
      # Can't run this while_loop in Python (e.g. because there's a vmap
      # transformation on it), so we fall back to the primitive version.
      pass

  init_vals, in_tree = tree_flatten((init_val,))
  init_avals = tuple(_map(_abstractify, init_vals))
  cond_jaxpr, cond_consts, cond_tree = _initial_style_jaxpr(cond_fun, in_tree, init_avals)
  body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(body_fun, in_tree, init_avals)
  if not treedef_is_leaf(cond_tree) or len(cond_jaxpr.out_avals) != 1:
    msg = "cond_fun must return a boolean scalar, but got pytree {}."
    raise TypeError(msg.format(cond_tree))
  if cond_jaxpr.out_avals[0].strip_weak_type() != ShapedArray((), onp.bool_):
    msg = "cond_fun must return a boolean scalar, but got output type(s) {}."
    raise TypeError(msg.format(cond_jaxpr.out_avals))

  in_tree_children = in_tree.children()
  assert len(in_tree_children) == 1
  _check_tree_and_avals("body_fun output and input",
                        # Extract the subtree and avals for the first element of the return tuple
                        body_tree, body_jaxpr.out_avals,
                        in_tree_children[0], init_avals)
  outs = while_p.bind(*itertools.chain(cond_consts, body_consts, init_vals),
                      cond_nconsts=len(cond_consts), cond_jaxpr=cond_jaxpr,
                      body_nconsts=len(body_consts), body_jaxpr=body_jaxpr)
  return tree_unflatten(body_tree, outs)

def _while_loop_abstract_eval(*args, **kwargs):
  return _map(raise_to_shaped, kwargs["body_jaxpr"].out_avals)

def _while_loop_translation_rule(c, axis_env, name_stack, avals, backend, *args,
                                 cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts):
  cond_consts, body_consts, init_vals = split_list(args, [cond_nconsts, body_nconsts])
  batched = bool(cond_jaxpr.out_avals[0].shape)

  # Since jaxprs don't have tuples and have multiple return values, but we need
  # the HLO While loop to take a single tuple input and output a single boolean
  # (for the cond computation) or a single tuple output (for the body
  # computation), we build XLA computations that handle the tuple munging before
  # generating a Call into the computations formed from the jaxprs.

  init_carry = xops.Tuple(c, cond_consts + body_consts + init_vals)

  cond_c = xb.make_computation_builder("cond_computation")
  cond_carry = xb.parameter(cond_c, 0, c.get_shape(init_carry))
  cond_carry_elts = [xops.GetTupleElement(cond_carry, i) for i in range(len(args))]
  x, _, z = split_list(cond_carry_elts, [cond_nconsts, body_nconsts])
  pred, = xla.jaxpr_subcomp(cond_c, cond_jaxpr.jaxpr, backend, axis_env,
                            _map(partial(xb.constant, cond_c),
                                 cond_jaxpr.literals),
                            extend_name_stack(name_stack, 'cond'), *(x + z))
  if batched:
    scalar = ShapedArray((), onp.bool_)
    or_ = xla.primitive_subcomputation(lax.or_p, scalar, scalar)
    pred = xops.Reduce(cond_c, [pred], [xb.constant(cond_c, onp.array(False))], or_,
                         list(range(cond_jaxpr.out_avals[0].ndim)))

  body_c = xb.make_computation_builder("body_computation")
  body_carry = xb.parameter(body_c, 0, c.get_shape(init_carry))
  body_carry_elts = [xops.GetTupleElement(body_carry, i) for i in range(len(args))]
  x, y, z = split_list(body_carry_elts, [cond_nconsts, body_nconsts])
  new_z = xla.jaxpr_subcomp(body_c, body_jaxpr.jaxpr, backend, axis_env,
                            _map(partial(xb.constant, body_c), body_jaxpr.literals),
                            extend_name_stack(name_stack, 'body'), *(y + z))
  if batched:
    body_pred, = xla.jaxpr_subcomp(body_c, cond_jaxpr.jaxpr, backend, axis_env,
                                   _map(partial(xb.constant, body_c), cond_jaxpr.literals),
                                   extend_name_stack(name_stack, 'body_pred'), *(x + z))
    new_z = _map(partial(_pred_bcast_select, body_c, body_pred), new_z, z, body_jaxpr.out_avals)
    assert _map(body_c.get_shape, new_z) == _map(body_c.get_shape, z) # no broadcast
  new_carry = xops.Tuple(body_c, list(itertools.chain(x, y, new_z)))

  ans = xops.While(cond_c.build(pred), body_c.build(new_carry), init_carry)
  ans_elts = [xops.GetTupleElement(ans, i) for i in range(len(args))]
  _,  _, z = split_list(ans_elts, [cond_nconsts, body_nconsts])
  return xops.Tuple(c, z)

def _pred_bcast_select(c, pred, x, y, x_y_aval: core.AbstractValue):
  pred_shape = c.get_shape(pred).dimensions()
  x_shape = c.get_shape(x).dimensions()
  y_shape = c.get_shape(y).dimensions()
  assert x_shape == y_shape
  if x_y_aval is core.abstract_unit:
    return x
  elif x_y_aval is core.abstract_token:
    return xops.AfterAll(c, [x, y])
  else:
    assert pred_shape == x_shape[:len(pred_shape)] == y_shape[:len(pred_shape)]
    bcast_pred = xops.BroadcastInDim(pred, x_shape, list(range(len(pred_shape))))
    return xops.Select(bcast_pred, x, y)

def _while_loop_batching_rule(args, dims, cond_nconsts, cond_jaxpr,
                              body_nconsts, body_jaxpr):
  size, = {x.shape[d] for x, d in zip(args, dims) if d is not batching.not_mapped}
  orig_batched = [d is not batching.not_mapped for d in dims]
  cconst_bat, bconst_bat, init_bat = split_list(orig_batched, [cond_nconsts, body_nconsts])

  # Fixpoint computation of which carry are batched: either
  # batched from init, or the carry out is batched. Each iteration promotes
  # at least one carry to batched. We need at most len(carry) iterations,
  # but we need one last iteration to prepare the jaxpr based on the final
  # carry_bat.
  carry_bat = init_bat
  for _ in range(1 + len(carry_bat)):
    batched = bconst_bat + carry_bat
    body_jaxpr_batched, carry_bat_out = batching.batch_jaxpr(
        body_jaxpr, size, batched, instantiate=carry_bat)
    cond_jaxpr_batched, (pred_bat,) = batching.batch_jaxpr(
        cond_jaxpr, size, cconst_bat + carry_bat,
        instantiate=bool(cond_jaxpr.out_avals[0].shape))
    carry_bat_out = _map(partial(operator.or_, pred_bat), carry_bat_out)
    if carry_bat_out == carry_bat:
      break
    else:
      carry_bat = _map(operator.or_, carry_bat, carry_bat_out)
  else:
    assert False, "Fixpoint not reached"

  consts, init = split_list(args, [cond_nconsts + body_nconsts])
  const_dims, init_dims = split_list(dims, [cond_nconsts + body_nconsts])
  new_consts = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
                else x for x, d in zip(consts, const_dims)]
  new_init = [batching.broadcast(x, size, 0) if now_bat and not was_bat
              else batching.moveaxis(x, d, 0) if now_bat and d != 0 else x
              for x, d, was_bat, now_bat in zip(init, init_dims, init_bat, carry_bat)]

  outs = while_p.bind(*(new_consts + new_init),
                      cond_nconsts=cond_nconsts, cond_jaxpr=cond_jaxpr_batched,
                      body_nconsts=body_nconsts, body_jaxpr=body_jaxpr_batched)
  out_bdims = [0 if b else batching.not_mapped for b in carry_bat]
  return outs, out_bdims

def _while_loop_jvp(primals, tangents, cond_nconsts, cond_jaxpr, body_nconsts,
                    body_jaxpr):
  nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  cconst_nz, bconst_nz, init_nz = split_list(nonzeros, [cond_nconsts, body_nconsts])

  carry_nz = init_nz
  for _ in range(1 + len(carry_nz)):
    body_nonzeros = bconst_nz + carry_nz
    body_jvp, nonzeros_out = ad.jvp_jaxpr(
        body_jaxpr, body_nonzeros, instantiate=carry_nz)
    if nonzeros_out == carry_nz:
      break
    carry_nz = _map(operator.or_, carry_nz, nonzeros_out)
  else:
    assert False, "Fixpoint not reached"

  nonzeros = cconst_nz + body_nonzeros
  tangents = [ad.instantiate_zeros(t) if nz else t
              for t, nz in zip(tangents, nonzeros)]

  cconst, bconst, init = split_list(primals, [cond_nconsts, body_nconsts])
  _, bconst_dot, init_dot = split_list(tangents, [cond_nconsts, body_nconsts])
  bconst_dot = _prune_zeros(bconst_dot)
  init_dot = _prune_zeros(init_dot)

  num_carry = len(primals) - cond_nconsts - body_nconsts

  body_jvp_rearranged = ad.rearrange_binders(
      body_jvp,
      [body_nconsts, num_carry], [len(bconst_dot), len(init_dot)],
      [num_carry], [len(init_dot)])

  newvar = core.gensym([cond_jaxpr.jaxpr])
  invars_aug = (
      cond_jaxpr.jaxpr.invars + [newvar(get_aval(x)) for x in init_dot])
  cond_jaxpr_augmented = core.Jaxpr(cond_jaxpr.jaxpr.constvars,
                                    invars_aug,
                                    cond_jaxpr.jaxpr.outvars,
                                    cond_jaxpr.jaxpr.eqns)
  in_avals_aug = (cond_jaxpr.in_avals[:cond_nconsts] +
                  body_jvp_rearranged.in_avals[body_nconsts + len(bconst_dot):])
  cond_jaxpr_augmented = core.TypedJaxpr(cond_jaxpr_augmented,
                                         cond_jaxpr.literals,
                                         in_avals_aug,
                                         cond_jaxpr.out_avals)

  out = while_p.bind(
      *(cconst + bconst + bconst_dot + init + init_dot),
      cond_nconsts=cond_nconsts,
      cond_jaxpr=cond_jaxpr_augmented,
      body_nconsts=len(bconst) + len(bconst_dot),
      body_jaxpr=body_jvp_rearranged)

  out_carry, out_carry_dot = split_list(out, [num_carry])
  out_tangents_iter = iter(out_carry_dot)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(out_carry, nonzeros_out)]
  return out_carry, out_tangents

def _while_partial_eval(trace: pe.JaxprTrace, *tracers: pe.Tracer, cond_nconsts: int,
                        cond_jaxpr: pe.TypedJaxpr, body_nconsts: int,
                        body_jaxpr: pe.TypedJaxpr) -> Sequence[pe.Tracer]:
  """An implementation of partial evaluation for while.
  As long as some carry (and hence output) are known and the output
  of `cond_jaxpr` is known, we use a portion of the loop body to compute the known
  outputs of the `while_loop`. For the unknown outputs we generate Jaxpr to run
  the whole while, including recomputing the known parts.

  This means that we don't actually save any computation by partial
  evaluation if there are unknown outputs.

  What this achieves is that we can give a proper error for reverse
  differentiation of `while`, because in that use of partial evaluation the
  primal inputs are considered "known", and only the tangent computation is
  unknown (see issue #2129).
  """
  unknowns = [not t.pval.is_known() for t in tracers]
  params = dict(cond_nconsts=cond_nconsts, cond_jaxpr=cond_jaxpr,
                body_nconsts=body_nconsts, body_jaxpr=body_jaxpr)

  cond_consts_uk, body_consts_uk, carry_init_uk = split_list(unknowns, [cond_nconsts, body_nconsts])
  # Fixpoint computation of unknown carry. Each iteration promotes
  # at least one carry to unknown. We need one last iteration to prepare the jaxpr.
  carry_uk = carry_init_uk
  for _ in range(1 + len(carry_uk)):
    body_jaxpr_known, _, carry_out_uk = pe.partial_eval_jaxpr(
        body_jaxpr, body_consts_uk + carry_uk, instantiate=carry_uk,
        trace_type=trace.master.trace_type)
    if carry_out_uk == carry_uk:
      break
    else:
      carry_uk = _map(operator.or_, carry_uk, carry_out_uk)
  else:
    assert False, "Fixpoint not reached"

  cond_jaxpr_known, _, cond_uk = pe.partial_eval_jaxpr(
    cond_jaxpr, cond_consts_uk + carry_uk, instantiate=False,
    trace_type=trace.master.trace_type)

  if cond_uk[0] or  all([not uk for uk in unknowns]) or all(unknowns):
    # If conditional is unknown, or all inputs are known, or all are unknown,
    # just do the default processing.
    return trace.default_process_primitive(while_p, tracers, params)

  # Run the known part of the while. Prepare the inputs, as constants (if known), or
  # as core.unit.
  in_consts = [ core.unit if uk else t.pval.get_known()
                for uk, t in zip(cond_consts_uk + body_consts_uk + carry_uk,
                                 tracers)]
  # There should be no residuals for the cond_jaxpr_known
  assert 1 == len(cond_jaxpr_known.out_avals)
  # We ignore the residuals from the body_jaxpr_known, so the type of inputs matches
  # the type of outputs; residuals are at the end
  if len(body_jaxpr_known.out_avals) > len(body_jaxpr.out_avals):
    # TODO(necula): this is not quite enough; we should drop the residual computations also
    body_jaxpr_known.out_avals = body_jaxpr_known.out_avals[:len(body_jaxpr.out_avals)]
    body_jaxpr_known.jaxpr.outvars = body_jaxpr_known.jaxpr.outvars[:len(body_jaxpr.out_avals)]
  out_known = while_p.bind(
    *in_consts,
    cond_nconsts=cond_nconsts,
    cond_jaxpr=cond_jaxpr_known,
    body_nconsts=body_nconsts,
    body_jaxpr=body_jaxpr_known)

  # Run the whole while_loop to get all the outputs, then merge with known ones
  out_all: Sequence[pe.Tracer] = trace.default_process_primitive(while_p, tracers, params)
  out_tracers: Sequence[pe.Tracer] = [
    out_unknown if uk
    else pe.JaxprTracer(trace, pe.PartialVal.known(known), out_unknown.recipe)
    for uk, out_unknown, known in zip(carry_uk, out_all, out_known)]

  return out_tracers

def _while_transpose_error(*_, **kwargs):
  raise ValueError("Reverse-mode differentiation does not work for "
                   "lax.while_loop or lax.fori_loop. "
                   "Try using lax.scan instead.")

while_p = lax.Primitive('while')
while_p.multiple_results = True
while_p.def_impl(partial(xla.apply_primitive, while_p))
while_p.def_abstract_eval(_while_loop_abstract_eval)
ad.primitive_jvps[while_p] = _while_loop_jvp
pe.custom_partial_eval_rules[while_p] = _while_partial_eval
xla.initial_style_translations[while_p] = _while_loop_translation_rule
ad.primitive_transposes[while_p] = _while_transpose_error
batching.primitive_batchers[while_p] = _while_loop_batching_rule


### cond and switch

def switch(index, branches: Sequence[Callable], operand):
  """Apply exactly one of ``branches`` given by ``index``.

  If ``index`` is out of bounds, it is clamped to within bounds.

  Has the semantics of the following Python::

    def switch(index, branches, operand):
      index = clamp(0, index, len(branches) - 1)
      return branches[index](operand)

  Arguments:
    index: Integer scalar type, indicating which branch function to apply.
    branches: Sequence of functions (A -> B) to be applied based on `index`.
    operand: Operand (A) input to whichever branch is applied.
  """
  if len(onp.shape(index)) != 0:
    raise TypeError(
        f"Branch index must be scalar, "
        f"got {index} of shape {onp.shape(index)}.")

  try:
    index_dtype = dtypes.result_type(index)
  except TypeError as err:
    msg = f"Index type must be an integer, got {index}."
    raise TypeError(msg) from err

  if index_dtype.kind not in 'iu':
    raise TypeError(
        f"Index type must be an integer, got {index} as {index_dtype}")

  branches = tuple(branches)

  if len(branches) == 0:
    raise ValueError("Empty branch sequence")
  elif len(branches) == 1:
    return branches[0](operand)

  index = lax.convert_element_type(index, onp.int32)
  lo = onp.array(0, onp.int32)
  hi = onp.array(len(branches) - 1, onp.int32)
  index = lax.clamp(lo, index, hi)

  if (jax.api._jit_is_disabled() and
      isinstance(core.get_aval(index), ConcreteArray)):
    return branches[int(index)](operand)

  ops, ops_tree = tree_flatten((operand,))
  ops_avals = tuple(_map(_abstractify, ops))

  jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
      branches, ops_tree, ops_avals)

  for i, (out_tree, jaxpr) in enumerate(zip(out_trees[1:], jaxprs[1:])):
    _check_tree_and_avals(f"branch 0 and {i + 1} outputs",
                          out_trees[0], jaxprs[0].out_avals,
                          out_tree, jaxpr.out_avals)

  linear = (False,) * (len(consts) + len(ops))
  out = cond_p.bind(
      index, *consts, *ops, branches=jaxprs, linear=linear)
  return tree_unflatten(out_trees[0], out)


def cond(*args, **kwargs):
  """Conditionally apply ``true_fun`` or ``false_fun``.

  Has equivalent semantics to this Python implementation::

    def cond(pred, true_fun, false_fun, operand):
      if pred:
        return true_fun(operand)
      else:
        return false_fun(operand)

  Pred must be a scalar type.

  Arguments:
    pred: Boolean scalar type, indicating which branch function to
      apply. Collections (list, tuple) are not supported.
    true_fun: Function (A -> B), to be applied if `pred` is True.
    false_fun: Function (A -> B), to be applied if `pred` is False.
    operand: Operand (A) input to either branch depending on `pred`.
  """

  # detect an attempt to call the former, deprecated cond
  try:
    ba = inspect.signature(_cond_with_per_branch_args).bind(*args, **kwargs)
  except TypeError:
    pass
  else:
    return _cond_with_per_branch_args(*ba.args)

  return _cond(*args, **kwargs)

def _cond(pred, true_fun: Callable, false_fun: Callable, operand):
  if len(onp.shape(pred)) != 0:
    raise TypeError(
        f"Pred must be a scalar, got {pred} of shape {onp.shape(pred)}.")

  try:
    pred_dtype = dtypes.result_type(pred)
  except TypeError as err:
    msg = ("Pred type must be either boolean or number, got {}.")
    raise TypeError(msg.format(pred)) from err

  if pred_dtype.kind != 'b':
    if pred_dtype.kind in 'iuf':
      pred = pred != 0
    else:
      msg = ("Pred type must be either boolean or number, got {}.")
      raise TypeError(msg.format(pred_dtype))

  if jax.api._jit_is_disabled() and isinstance(core.get_aval(pred), ConcreteArray):
    if pred:
      return true_fun(operand)
    else:
      return false_fun(operand)

  ops, ops_tree = tree_flatten((operand,))
  ops_avals = tuple(_map(_abstractify, ops))

  jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
      (true_fun, false_fun), ops_tree, ops_avals)
  true_jaxpr, false_jaxpr = jaxprs
  out_tree, false_out_tree = out_trees

  _check_tree_and_avals("true_fun and false_fun output",
                        out_tree, true_jaxpr.out_avals,
                        false_out_tree, false_jaxpr.out_avals)

  index = lax.convert_element_type(pred, onp.int32)

  linear = (False,) * (len(consts) + len(ops))
  out = cond_p.bind(
      index, *consts, *ops,
      branches=(false_jaxpr, true_jaxpr), linear=linear)
  return tree_unflatten(out_tree, out)

def _cond_with_per_branch_args(pred,
                               true_operand, true_fun: Callable,
                               false_operand, false_fun: Callable):
  """Conditionally apply ``true_fun`` or ``false_fun``.

  Has equivalent semantics to this Python implementation::

    def cond(pred, true_operand, true_fun, false_operand, false_fun):
      if pred:
        return true_fun(true_operand)
      else:
        return false_fun(false_operand)

  Pred has to be a scalar type, collection types (list, tuple) are not supported
  """
  return _cond(pred,
               lambda op: true_fun(op[0]),
               lambda op: false_fun(op[1]),
               (true_operand, false_operand))

def _cond_abstract_eval(*args, **kwargs):
  return _map(raise_to_shaped, kwargs["branches"][0].out_avals)

def _cond_translation_rule(c, axis_env, name_stack, avals, backend,
                           index, *args, branches, linear):
  del linear  # Unused.

  def make_computation(name, jaxpr, op_shape):
    c = xb.make_computation_builder(name + '_comp')
    op = xb.parameter(c, 0, op_shape)
    ops = [xops.GetTupleElement(op, i) for i in range(len(jaxpr.in_avals))]
    outs = xla.jaxpr_subcomp(c, jaxpr.jaxpr, backend, axis_env,
                             _map(partial(xb.constant, c), jaxpr.literals),
                             extend_name_stack(name_stack, name + '_fun'), *ops)
    return c.build(xops.Tuple(c, outs))

  op = xops.Tuple(c, args)
  op_shape = c.get_shape(op)
  branch_computations = [
      make_computation(f'branch_{i}', jaxpr, op_shape)
      for i, jaxpr in enumerate(branches)]
  return xops.Conditional(index, branch_computations, [op] * len(branches))

def _select_tree(indices, branch_vals):
  assert len(branch_vals) > 0
  if len(branch_vals) == 1:
    return branch_vals[0]
  mid = len(branch_vals) // 2
  mid = onp.array(mid, dtypes.canonicalize_dtype(lax.dtype(indices)))
  return lax.select(lax.lt(indices, mid),
                    _select_tree(indices, branch_vals[:mid]),
                    _select_tree(indices - mid, branch_vals[mid:]))

def _cond_index_bcast_and_select_tree(indices, branch_vals):
  if all(core.get_aval(x) is core.abstract_unit for x in branch_vals):
    return branch_vals[0]
  else:
    bcast_indices = lax.broadcast_in_dim(
        indices, onp.shape(branch_vals[0]), list(range(onp.ndim(indices))))
    return _select_tree(bcast_indices, branch_vals)

def _cond_batching_rule(args, dims, branches, linear):
  # TODO: maybe avoid moving arg axes to front if we're promoting to select?
  size, = {x.shape[d] for x, d in zip(args, dims) if d is not batching.not_mapped}
  args = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
          else x for x, d in zip(args, dims)]
  orig_bat = [d is not batching.not_mapped for d in dims]
  del dims
  index, *ops = args
  index_bat, *bat = orig_bat

  branches_out_bat = [batching.batch_jaxpr(jaxpr, size, bat, False)[1]
                      for jaxpr in branches]
  out_bat = [any(bat) for bat in zip(*branches_out_bat)]

  branches_batched = tuple(batching.batch_jaxpr(jaxpr, size, bat, out_bat)[0]
                           for jaxpr in branches)

  if index_bat:
    branch_outs = []
    for jaxpr in branches_batched:
      out = core.jaxpr_as_fun(jaxpr)(*ops)
      out = [batching.broadcast(x, size, 0) if not b else x
             for x, b in zip(out, out_bat)]
      branch_outs.append(out)
    return [_cond_index_bcast_and_select_tree(index, outs)
            for outs in zip(*branch_outs)], [0] * len(branch_outs[0])
  else:
    out_dims = [0 if b else batching.not_mapped for b in out_bat]
    out = cond_p.bind(
        index, *ops, branches=branches_batched, linear=linear)
    return out, out_dims

def _cond_jvp(primals, tangents, branches, linear):
  nonzeros = [type(t) is not ad_util.Zero for t in tangents]

  index_nz, *ops_nz = nonzeros
  assert index_nz is False

  branches_out_nz = [ad.jvp_jaxpr(jaxpr, ops_nz, instantiate=False)[1]
                     for jaxpr in branches]
  out_nz = [any(nz) for nz in zip(*branches_out_nz)]

  branches_jvp = tuple(ad.jvp_jaxpr(jaxpr, ops_nz, instantiate=out_nz)[0]
                       for jaxpr in branches)

  index, *ops = primals
  _, *ops_dot = tangents
  ops_dot = _prune_zeros(ops_dot)

  ops_lin = tuple(linear)
  linear_jvp = ops_lin + (True,) * len(ops_dot)
  out = cond_p.bind(
      index, *ops, *ops_dot, branches=branches_jvp, linear=linear_jvp)
  out_primals, out_tangents = split_list(out, [len(out_nz)])
  out_tangents_iter = iter(out_tangents)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(out_primals, out_nz)]
  return out_primals, out_tangents

def _cond_partial_eval(trace, *tracers, branches, linear):
  unknowns = [t.pval[0] is not None for t in tracers]

  index_uk, *ops_uk = unknowns

  if index_uk:
    # When the branch index is unknown, we stage out the whole cond.
    params = dict(branches=branches, linear=linear)
    return trace.default_process_primitive(cond_p, tracers, params)

  branches_out_uks = []
  for branch_jaxpr in branches:
    _, _, out_uks = pe.partial_eval_jaxpr(branch_jaxpr, ops_uk,
                                          instantiate=False,
                                          trace_type=trace.master.trace_type)
    branches_out_uks.append(out_uks)
  out_uks = [any(uks) for uks in zip(*branches_out_uks)]

  branches_1, branches_2, branch_res_avals = [], [], []
  for branch_jaxpr in branches:
    branch_jaxpr_1, branch_jaxpr_2, _ = pe.partial_eval_jaxpr(
        branch_jaxpr, ops_uk, instantiate=out_uks,
        trace_type=trace.master.trace_type)
    branch_num_res = len(branch_jaxpr_1.out_avals) - len(out_uks)

    # move residuals to the front
    move = [False] * len(ops_uk) + [True] * branch_num_res
    branch_jaxpr_2 = pe.move_binders_to_front(branch_jaxpr_2, move)

    # TODO(frostig,mattjj): pe.partial_eval_jaxpr should raise to shaped avals
    res_avals = _map(
        raise_to_shaped, branch_jaxpr_2.in_avals[:branch_num_res])

    branches_1.append(branch_jaxpr_1)
    branches_2.append(branch_jaxpr_2)
    branch_res_avals.append(res_avals)

  branches_1 = tuple(branches_1)
  branches_2 = tuple(branches_2)

  for jaxpr in branches_2[1:]:
    assert len(jaxpr.out_avals) == len(branches_2[0].out_avals)

  num_outs = len(branches_2[0].out_avals)

  branches_1 = _join_cond_outputs(branches_1, branch_res_avals, num_outs)
  branches_2 = _join_cond_pe_staged_jaxpr_inputs(branches_2, branch_res_avals)

  # TODO(frostig,mattjj): reinstate this assertion once pe.partial_eval_jaxpr
  # raises to shaped avals
  # for j in branches_1[1:]:
  #   assert j.out_avals == branches_1[0].out_avals
  num_res = sum(_map(len, branch_res_avals))

  _, in_consts = unzip2([t.pval for t in tracers])
  out_consts_res = cond_p.bind(*in_consts, branches=branches_1, linear=linear)
  out_consts, res = split_list(out_consts_res, [len(out_consts_res) - num_res])

  # TODO(frostig,mattjj): remove raised_to_shaped of avals once
  # pe.partial_eval_jaxpr handles it
  out_avals = _map(raise_to_shaped, branches_2[0].out_avals)
  out_pvs = [aval if uk else None for aval, uk in zip(out_avals, out_uks)]

  index_tracer = trace.instantiate_const(tracers[0])

  ops_tracers = [trace.instantiate_const(t) if uk
                 else trace.new_instantiated_literal(core.unit)
                 for uk, t in zip(unknowns[1:], tracers[1:])]

  res_tracers = _map(trace.new_instantiated_const, res)

  out_tracers = [pe.JaxprTracer(trace, pe.PartialVal((pv, const)), None)
                 for pv, const in zip(out_pvs, out_consts)]

  linear_2 = (False,) * num_res + linear
  params = dict(branches=branches_2, linear=linear_2)
  eqn = pe.new_eqn_recipe(
      [index_tracer] + res_tracers + ops_tracers, out_tracers, cond_p, params)
  for t in out_tracers: t.recipe = eqn
  return out_tracers

# When partially evaluating conditionals, each branch produces residuals
# depending on the computation carried out by the branch, and a corresponding
# staged jaxpr that accepts those residuals as its first few inputs. The
# residual-producing branches are staged as jaxprs and bound right away in a
# conditional. The residual-consuming jaxprs are assembled together in a jaxpr
# conditional. The following two helper functions are used to ensure that both
# collections of jaxprs (those evaluated and those staged) are valid for joint
# use under their respective conditionals.

# Because every branch might produce different residuals, the branches' output
# signatures might not match. But we need branch signatures to match in order to
# bind them in a conditional. This function "joins" the residual outputs of the
# branches by concatenation. Each augmented branch returns zero-filled values in
# the place of all other branches' residuals.
def _join_cond_outputs(jaxprs, res_avals_per_jaxpr, num_non_res_outputs):
  def augment_jaxpr(i, jaxpr):
    res_avals_prefix = util.concatenate(res_avals_per_jaxpr[:i])
    res_avals_suffix = util.concatenate(res_avals_per_jaxpr[i+1:])

    @lu.wrap_init
    def f_aug(*args):
      outs_and_residuals = core.jaxpr_as_fun(jaxpr)(*args)
      outs, residuals = split_list(outs_and_residuals, [num_non_res_outputs])
      zeros_prefix = _map(ad_util.zeros_like_aval, res_avals_prefix)
      zeros_suffix = _map(ad_util.zeros_like_aval, res_avals_suffix)
      return outs + zeros_prefix + residuals + zeros_suffix

    return _make_typed_jaxpr(f_aug, jaxpr.in_avals)

  return tuple(augment_jaxpr(i, jaxpr) for i, jaxpr in enumerate(jaxprs))

# To use these staged jaxprs as the branches of another conditional, we need for
# their (input) signatures to match. This function "joins" the staged jaxprs:
# for each one, it makes another that accepts *all* residuals, but still only
# uses those that it needs (dropping the rest).
def _join_cond_pe_staged_jaxpr_inputs(jaxprs, res_avals_per_jaxpr):
  newvar = core.gensym([j.jaxpr for j in jaxprs], suffix='_')
  unused_res_vars = tuple(
      tuple(newvar(aval) for aval in res_avals)
      for res_avals in res_avals_per_jaxpr)

  def pad_jaxpr_res_avals(i, jaxpr):
    res_vars_prefix = util.concatenate(unused_res_vars[:i])
    res_vars_suffix = util.concatenate(unused_res_vars[i+1:])
    res_avals_prefix = util.concatenate(res_avals_per_jaxpr[:i])
    res_avals_suffix = util.concatenate(res_avals_per_jaxpr[i+1:])

    res_avals = res_avals_per_jaxpr[i]
    num_res = len(res_avals)
    res_vars = jaxpr.jaxpr.invars[:num_res]

    non_res_vars = jaxpr.jaxpr.invars[num_res:]
    non_res_avals = jaxpr.in_avals[num_res:]

    aug_invars = res_vars_prefix + res_vars + res_vars_suffix + non_res_vars
    aug_avals = res_avals_prefix + res_avals + res_avals_suffix + non_res_avals

    jaxpr_aug = core.Jaxpr(jaxpr.jaxpr.constvars, aug_invars,
                           jaxpr.jaxpr.outvars, jaxpr.jaxpr.eqns)
    jaxpr_aug = core.TypedJaxpr(jaxpr_aug, jaxpr.literals, aug_avals,
                                jaxpr.out_avals)
    return jaxpr_aug

  return tuple(pad_jaxpr_res_avals(i, jaxpr) for i, jaxpr in enumerate(jaxprs))

def _transpose_cond_jaxpr(jaxpr, num_res):
  res_avals, primal_avals = split_list(jaxpr.in_avals, [num_res])
  primal_avals = _map(raise_to_shaped, primal_avals)

  @lu.wrap_init
  def transposed(*args):
    res, cts_out = split_list(args, [num_res])
    primals = res + [ad.UndefinedPrimal(aval) for aval in primal_avals]
    cts_in = ad.backward_pass(
        jaxpr.jaxpr, jaxpr.literals, primals, cts_out)
    _, cts_in = split_list(cts_in, [num_res])
    return _map(ad.instantiate_zeros_aval, primal_avals, cts_in)

  return _make_typed_jaxpr(transposed, res_avals + jaxpr.out_avals)

def _cond_transpose(cts, *args, branches, linear):
  index, *ops = args
  in_avals = _map(raise_to_shaped, branches[0].in_avals)
  num_res = len(ops) - sum(linear)

  branches_trans = tuple(
      _transpose_cond_jaxpr(jaxpr, num_res) for jaxpr in branches)
  lin_in_avals = _map(
      raise_to_shaped, [a for a, l in zip(in_avals, linear) if l])
  assert all(jaxpr.out_avals == lin_in_avals for jaxpr in branches_trans)

  res = ops[:num_res]
  cts = _map(ad.instantiate_zeros_aval, branches[0].out_avals, cts)
  linear_trans = (False,) * num_res + (True,) * len(cts)

  out = cond_p.bind(
      index, *res, *cts, branches=branches_trans, linear=linear_trans)
  assert all(_map(typecheck, lin_in_avals, out))

  out_iter = iter(out)
  out = [next(out_iter) if l else None for l in linear]
  assert next(out_iter, None) is None
  return [None] + out

def cond_bind(*args, branches, linear):
  if not core.skip_checks:
    assert len(branches) > 0
    assert len(linear) + 1 == len(args)
    assert len(args) == 1 + len(branches[0].in_avals)
    jaxpr0 = branches[0]
    for jaxpr in branches[1:]:
      assert len(jaxpr0.in_avals) == len(jaxpr.in_avals)
      assert len(jaxpr0.out_avals) == len(jaxpr.out_avals)
      assert all(_map(typematch, jaxpr0.in_avals, jaxpr.in_avals))
      assert all(_map(typematch, jaxpr0.out_avals, jaxpr.out_avals))
    index, *ops = args
    assert dtypes.result_type(index) == onp.int32
    for jaxpr in branches:
      assert all(_map(typecheck, jaxpr.in_avals, ops))
      core.check_jaxpr(jaxpr.jaxpr)
  return core.Primitive.bind(cond_p, *args, branches=branches, linear=linear)

cond_p = lax.Primitive('cond')
cond_p.multiple_results = True
cond_p.def_impl(partial(xla.apply_primitive, cond_p))
cond_p.def_abstract_eval(_cond_abstract_eval)
cond_p.def_custom_bind(cond_bind)
ad.primitive_jvps[cond_p] = _cond_jvp
ad.primitive_transposes[cond_p] = _cond_transpose
pe.custom_partial_eval_rules[cond_p] = _cond_partial_eval
batching.primitive_batchers[cond_p] = _cond_batching_rule
xla.initial_style_translations[cond_p] = _cond_translation_rule


### scan

def scan(f, init, xs, length=None, reverse=False):
  """Scan a function over leading array axes while carrying along state.

  The type signature in brief is

  .. code-block:: haskell

    scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])

  where we use [t] here to denote the type t with an additional leading axis.
  That is, if t is an array type then [t] represents the type with an additional
  leading axis, and if t is a pytree (container) type with array leaves then [t]
  represents the type with the same pytree structure and corresponding leaves
  each with an additional leading axis.

  When ``a`` is an array type or None, and ``b`` is an array type, the semantics
  of ``scan`` are given roughly by this Python implementation::

    def scan(f, init, xs, length=None):
      if xs is None:
        xs = [None] * length
      carry = init
      ys = []
      for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
      return carry, np.stack(ys)

  Unlike that Python version, both ``a`` and ``b`` may be arbitrary pytree
  types, and so multiple arrays can be scanned over at once and produce multiple
  output arrays. (None is actually an empty pytree.)

  Also unlike that Python version, ``scan`` is a JAX primitive and is lowered to
  a single XLA While HLO. That makes it useful for reducing compilation times
  for jit-compiled functions, since native Python loop constructs in an ``@jit``
  function are unrolled, leading to large XLA computations.

  Finally, the loop-carried value ``carry`` must hold a fixed shape and dtype
  across all iterations (and not just be consistent up to NumPy rank/shape
  broadcasting and dtype promotion rules, for example). In other words, the type
  ``c`` in the type signature above represents an array with a fixed shape and
  dtype (or a nested tuple/list/dict container data structure with a fixed
  structure and arrays with fixed shape and dtype at the leaves).

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output.
    init: an initial loop carry value of type ``c``, which can be a scalar,
      array, or any pytree (nested Python tuple/list/dict) thereof, representing
      the initial loop carry value. This value must have the same structure as
      the first element of the pair returned by ``f``.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
    length: optional integer specifying the number of loop iterations, which
      must agree with the sizes of leading axes of the arrays in ``xs`` (but can
      be used to perform scans where no input ``xs`` are needed).
    reverse: optional boolean specifying whether to run the scan iteration
      forward (the default) or in reverse, equivalent to reversing the leading
      axes of the arrays in both ``xs`` and in ``ys``.

  Returns:
    A pair of type ``(c, [b])`` where the first element represents the final
    loop carry value and the second element represents the stacked outputs of
    the second output of ``f`` when scanned over the leading axis of the inputs.
  """
  init_flat, init_tree = tree_flatten(init)
  xs_flat, xs_tree = tree_flatten(xs)
  in_flat, in_tree = tree_flatten((init, xs))

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

  if jax.api._jit_is_disabled():
    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(length)):
      xs_slice = [_index_array(i, core.get_aval(x), x) for x in xs_flat]
      carry, y = f(carry, tree_unflatten(xs_tree, xs_slice))
      ys.append(y)
    stack = lambda y, *ys: (y if core.get_aval(y) is core.abstract_unit
                            else jax.numpy.stack((y, *ys)))
    ys = tree_multimap(stack, *maybe_reversed(ys))
    return carry, ys

  carry_avals = tuple(_map(_abstractify, init_flat))
  x_shapes = [masking.padded_shape_as_value(x.shape[1:]) for x in xs_flat]
  x_dtypes = [x.dtype for x in xs_flat]
  x_avals = tuple(_map(ShapedArray, x_shapes, x_dtypes))
  jaxpr, consts, out_tree = _initial_style_jaxpr(f, in_tree, carry_avals + x_avals)
  out_tree_children = out_tree.children()
  if len(out_tree_children) != 2:
    msg = "scan body output must be a pair, got {}."
    raise TypeError(msg.format(tree_unflatten(out_tree, jaxpr.out_avals)))
  _check_tree_and_avals("scan carry output and input",
                        # Extract the subtree and avals for the first element of the return tuple
                        out_tree_children[0], jaxpr.out_avals[:out_tree_children[0].num_leaves],
                        init_tree, carry_avals)

  out = scan_p.bind(*itertools.chain(consts, in_flat),
                    reverse=reverse, length=length, jaxpr=jaxpr,
                    num_consts=len(consts), num_carry=len(init_flat),
                    linear=(False,) * (len(consts) + len(in_flat)))
  return tree_unflatten(out_tree, out)

def _scan_impl(*args, reverse, length, num_consts, num_carry, jaxpr, linear):
  consts, init, xs = split_list(args, [num_consts, num_carry])
  _, _, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
  _, y_avals = split_list(jaxpr.out_avals, [num_carry])

  def cond_fun(vals):
    i, *_ = vals
    return i < length

  def body_fun(vals):
    [i], carry, ys = split_list(vals, [1, num_carry])
    i_ = length - i - 1 if reverse else i
    x = _map(partial(_index_array, i_), x_avals, xs)
    out_flat = core.jaxpr_as_fun(jaxpr)(*(consts + carry + x))
    carry_out, y_updates = split_list(out_flat, [num_carry])
    ys_out = _map(partial(_update_array, i_), y_avals, ys, y_updates)
    return [i + 1] + carry_out + ys_out

  ys_init = _map(partial(_empty_array, length), y_avals)
  if length == 0:
    return init + ys_init
  else:
    init_val = [lax._const(length, 0)] + init + ys_init
    _, *outs = while_loop(cond_fun, body_fun, init_val)
    return outs

def _index_array(i, aval, x):
  if aval is core.abstract_unit:
    return core.unit
  else:
    return lax.dynamic_index_in_dim(x, i, keepdims=False)

def _empty_array(sz, aval):
  if aval is core.abstract_unit:
    return core.unit
  else:
    return lax.full((sz,) + aval.shape, 0, aval.dtype)

def _update_array(i, aval, xs, x):
  if aval is core.abstract_unit:
    return core.unit
  else:
    return lax.dynamic_update_index_in_dim(xs, x, i, 0)

def _scan_abstract_eval(*args, reverse, length, num_consts, num_carry, jaxpr, linear):
  carry_avals, y_avals = split_list(jaxpr.out_avals, [num_carry])
  ys_avals = [ShapedArray((length,) + aval.shape, aval.dtype)
              if aval is not core.abstract_unit else aval for aval in y_avals]
  return carry_avals + ys_avals

def _scan_jvp(primals, tangents, reverse, length, jaxpr, num_consts, num_carry,
              linear):
  num_xs = len(jaxpr.in_avals) - num_carry - num_consts
  num_ys = len(jaxpr.out_avals) - num_carry
  nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  const_nz, init_nz, xs_nz = split_list(nonzeros, [num_consts, num_carry])

  # Fixpoint computation of which carry are not ad.zero: either
  # non-zero from init, or the carry out is non-zero. Each iteration promotes
  # at least one carry to non-zero. We need at most len(carry) iterations,
  # but we need one last iteration to prepare the jaxpr based on the final
  # carry_nz.
  carry_nz = init_nz
  for _ in range(1 + len(carry_nz)):
    nonzeros = const_nz + carry_nz + xs_nz
    jaxpr_jvp, nonzeros_out = ad.jvp_jaxpr(
        jaxpr, nonzeros, instantiate=carry_nz + [False] * num_ys)
    carry_nz_out, _ = nonzeros_out[:num_carry], nonzeros_out[num_carry:]
    if carry_nz_out == carry_nz:
      break
    else:
      carry_nz = _map(operator.or_, carry_nz, carry_nz_out)
  else:
    assert False, "Fixpoint not reached"

  tangents = [ad.instantiate_zeros(t) if nz else t
              for t, nz in zip(tangents, nonzeros)]

  consts, init, xs = split_list(primals, [num_consts, num_carry])
  all_tangents = split_list(tangents, [num_consts, num_carry])
  consts_dot, init_dot, xs_dot = _map(_prune_zeros, all_tangents)

  jaxpr_jvp_rearranged = ad.rearrange_binders(
      jaxpr_jvp,
      [num_consts, num_carry, num_xs], [len(consts_dot), len(init_dot), len(xs_dot)],
      [num_carry, num_ys], [len(init_dot), sum(nonzeros_out) - len(init_dot)])

  consts_linear, init_linear, xs_linear = split_list(linear, [num_consts, num_carry])
  jaxpr_jvp_linear = tuple(consts_linear + [True] * len(consts_dot)
                           + init_linear + [True] * len(init_dot)
                           + xs_linear + [True] * len(xs_dot))

  out_flat = scan_p.bind(
      *(consts + consts_dot + init + init_dot + xs + xs_dot),
      reverse=reverse, length=length, jaxpr=jaxpr_jvp_rearranged,
      num_consts=num_consts+len(consts_dot), num_carry=num_carry+len(init_dot),
      linear=jaxpr_jvp_linear)

  carry, carry_dot, ys, ys_dot = split_list(out_flat, [num_carry, len(init_dot), num_ys])
  primals_out = carry + ys
  tangents_out_iter = iter(carry_dot + ys_dot)
  tangents_out = [next(tangents_out_iter) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(primals_out, nonzeros_out)]
  return primals_out, tangents_out

def _prune_zeros(ts):
  return [t for t in ts if type(t) is not ad_util.Zero]

def _scan_partial_eval(trace, *tracers, reverse, length, num_consts, num_carry,
                       jaxpr, linear):
  if trace.master.trace_type is pe.StagingJaxprTrace:
    params = {"reverse": reverse, "length": length, "num_consts": num_consts,
              "num_carry": num_carry, "jaxpr": jaxpr, "linear": linear}
    return trace.default_process_primitive(scan_p, tracers, params)

  num_ys = len(jaxpr.out_avals) - num_carry

  unknowns = [t.pval[0] is not None for t in tracers]
  const_uk, init_uk, xs_uk = split_list(unknowns, [num_consts, num_carry])

  # Fixpoint computation of which carry are unknown (not a constant): either
  # unknown from init, or the carry out is unknown. Each iteration promotes
  # at least one carry to unknown. We need at most len(carry) iterations,
  # but we need one last iteration to prepare the jaxpr based on the final
  # carry_uk.
  carry_uk = init_uk
  for _ in range(1 + len(carry_uk)):
    unknowns = const_uk + carry_uk + xs_uk
    jaxpr_1, jaxpr_2, out_uk = pe.partial_eval_jaxpr(
        jaxpr, unknowns, instantiate=carry_uk + [False] * num_ys,
        trace_type=trace.master.trace_type)
    carry_uk_out = out_uk[:num_carry]
    if carry_uk_out == carry_uk:
      break
    else:
      carry_uk = _map(operator.or_, carry_uk, carry_uk_out)
  else:
    assert False, "Fixpoint not reached"
  num_res = len(jaxpr_1.out_avals) - len(jaxpr_2.out_avals)

  # The residuals are treated as extensive outputs of jaxpr_1 (and extensive
  # inputs to jaxpr_2), but residuals that are loop-invariant can be hoisted.
  # TODO(mattjj): hoist other loop-invariant values here too (instantiate=False)
  invariant_pvals = [pe.PartialVal.known(core.unit if uk else t.pval[1])
                     for uk, t in zip(unknowns[:num_consts], tracers[:num_consts])]
  other_pvals = [pe.PartialVal.unknown(a) for a in jaxpr_1.in_avals[num_consts:]]
  in_pvals_1 = invariant_pvals + other_pvals
  untyped_jaxpr_1, out_pvals_1, consts_1 = pe.trace_to_jaxpr(
      lu.wrap_init(core.jaxpr_as_fun(jaxpr_1)), in_pvals_1,
      instantiate=[True] * (num_carry + num_ys) + [False] * num_res)
  const_avals_1 = [raise_to_shaped(core.get_aval(c)) for c in consts_1]
  in_avals_1 = [core.abstract_unit] * num_consts + jaxpr_1.in_avals[num_consts:]
  out_avals_1 = [core.abstract_unit if pv is None else pv for pv, c in out_pvals_1]

  # TODO(cjfj): Explain the need for the code below.
  for var in untyped_jaxpr_1.invars[:num_consts]:
    var.aval = core.abstract_unit

  jaxpr_1_opt = pe.TypedJaxpr(pe.convert_constvars_jaxpr(untyped_jaxpr_1),
                              (), const_avals_1 + in_avals_1, out_avals_1)
  num_consts_1 = num_consts + len(consts_1)
  # any now-known residuals are intensive, so we want to revise jaxpr_2 to take
  # those inputs as constants rather than as extensive inputs
  _, _, res_pvals = split_list(out_pvals_1, [num_carry, num_ys])
  intensive_residuals = [const for pv, const in res_pvals if pv is None]
  move = [False] * len(jaxpr_1.in_avals) + [pv is None for pv, _ in res_pvals]
  jaxpr_2_opt = pe.move_binders_to_front(jaxpr_2, move)
  num_consts_2 = num_consts + len(intensive_residuals)

  in_consts = (list(consts_1) + [core.unit] * num_consts +
               [core.unit if uk else t.pval[1]
                for uk, t in zip(unknowns[num_consts:], tracers[num_consts:])])
  linear_1 = ([False] * len(consts_1) + [True] * num_consts +
              [lin or uk for uk, lin
               in zip(unknowns[num_consts:], linear[num_consts:])])
  out_flat = scan_p.bind(
      *in_consts, reverse=reverse, length=length, jaxpr=jaxpr_1_opt,
      num_consts=num_consts_1, num_carry=num_carry, linear=tuple(linear_1))
  out_carry, ys, res_and_units = split_list(out_flat, [num_carry, num_ys])
  extensive_residuals = [r for r, (pv, _) in zip(res_and_units, res_pvals) if pv is not None]

  new_tracers = [trace.instantiate_const(t) if uk else trace.new_instantiated_literal(core.unit)
                 for uk, t in zip(unknowns, tracers)]
  carry_avals, y_avals = split_list(jaxpr.out_avals, [num_carry])
  ys_avals = _map(partial(_promote_aval_rank, length), y_avals)
  out_avals = carry_avals + ys_avals
  out_pvs = [aval if uk else None for aval, uk in zip(out_avals, out_uk)]

  out_consts = out_carry + ys
  int_res_tracers = _map(trace.new_instantiated_const, intensive_residuals)
  ext_res_tracers = _map(trace.new_instantiated_const, extensive_residuals)
  out_tracers = [pe.JaxprTracer(trace, pe.PartialVal((pv, const)), None)
                 for pv, const in zip(out_pvs, out_consts)]
  linear_2 = ([False] * len(int_res_tracers) +
              [lin or not uk for uk, lin in zip(unknowns, linear)] +
              [False] * len(ext_res_tracers))
  eqn = pe.new_eqn_recipe(int_res_tracers + new_tracers + ext_res_tracers,
                          out_tracers, scan_p,
                          dict(reverse=reverse, length=length, jaxpr=jaxpr_2_opt,
                               num_consts=num_consts_2,
                               num_carry=num_carry, linear=tuple(linear_2)))
  for t in out_tracers: t.recipe = eqn
  return out_tracers

def _promote_aval_rank(sz, aval):
  if aval is core.abstract_unit:
    return core.abstract_unit
  else:
    return ShapedArray((sz,) + aval.shape, aval.dtype)

def _scan_transpose(cts, *args, reverse, length, num_consts, num_carry, jaxpr, linear):
  # we've only implemented transposing scans with specific lin/nonlin patterns
  consts_lin, init_lin, xs_lin = split_list(linear, [num_consts, num_carry])
  num_ires = len(consts_lin) - sum(consts_lin)
  num_eres = len(xs_lin) - sum(xs_lin)
  if consts_lin != [False] * num_ires + [True] * (len(consts_lin) - num_ires):
    raise NotImplementedError
  if xs_lin != [True] * (len(xs_lin) - num_eres) + [False] * num_eres:
    raise NotImplementedError
  if not all(init_lin):
    pass  # TODO(mattjj): error check https://github.com/google/jax/issues/1963

  consts, _, xs = split_list(args, [num_consts, num_carry])
  ires, _ = split_list(consts, [num_ires])
  _, eres = split_list(xs, [sum(xs_lin)])
  assert not any(ad.is_undefined_primal(r) for r in ires)
  assert not any(ad.is_undefined_primal(r) for r in eres)

  carry_avals, y_avals = split_list(jaxpr.out_avals, [num_carry])
  ys_avals = _map(partial(_promote_aval_rank, length), y_avals)
  ct_carry, ct_ys = split_list(cts, [num_carry])
  ct_carry = _map(ad.instantiate_zeros_aval, carry_avals, ct_carry)
  ct_ys = _map(ad.instantiate_zeros_aval, ys_avals, ct_ys)
  ct_consts = _map(ad_util.zeros_like_aval, jaxpr.in_avals[num_ires:num_consts])

  #       jaxpr :: [ires, T d] -> [T c] -> [T a, eres] -> ([T c], [T b])
  # jaxpr_trans :: [ires] -> [CT d, CT c] -> [CT b, eres] -> ([CT d, CT c], [CT a])
  jaxpr_trans = _transpose_scan_jaxpr(
      num_ires, num_consts - num_ires, num_eres, jaxpr)
  linear_trans = ([False] * num_ires +
                  [True] * (len(ct_consts) + len(ct_carry) + len(ct_ys)) +
                  [False] * num_eres)

  outs = scan_p.bind(
      *(ires + ct_consts + ct_carry + ct_ys + eres), reverse=not reverse,
      length=length, jaxpr=jaxpr_trans, num_consts=num_ires,
      num_carry=num_consts-num_ires+num_carry, linear=tuple(linear_trans))
  ct_consts, ct_init, ct_xs = split_list(outs, [num_consts - num_ires, num_carry])
  return [None] * num_ires + ct_consts + ct_init + ct_xs + [None] * num_eres

# transpose_scan_jaxpr :: ([res1, c, a, res2] -> b)
#                         -> ([res1, CT c, CT b, res2] -> [CT c, CT a])
def _transpose_scan_jaxpr(num_res1, num_c, num_res2, jaxpr):
  num_a = len(jaxpr.in_avals) - num_res1 - num_c - num_res2
  res1_avals, c_avals, a_avals, res2_avals = split_list(
      jaxpr.in_avals, [num_res1, num_c, num_a])
  num_b = len(jaxpr.out_avals)
  b_avals = list(jaxpr.out_avals)

  @lu.wrap_init
  def transposed(*res1_cbar_bbar_res2):
    res1, c_bar, b_bar, res2 = split_list(
        res1_cbar_bbar_res2, [num_res1, num_c, num_b])
    primals = (res1 + [ad.UndefinedPrimal(aval) for aval in c_avals] +
               [ad.UndefinedPrimal(aval) for aval in a_avals] + res2)
    cbar_abar = ad.backward_pass(jaxpr.jaxpr, jaxpr.literals, primals,
                                    b_bar)
    _, new_c_bar, a_bar, _ = split_list(cbar_abar, [num_res1, num_c, num_a])
    a_bar = _map(ad.instantiate_zeros_aval, a_avals, a_bar)
    c_bar = _map(ad.instantiate_zeros_aval, c_avals,
                _map(ad.add_tangents, c_bar, new_c_bar))
    return c_bar + a_bar
  return _make_typed_jaxpr(transposed, res1_avals + c_avals + b_avals + res2_avals)

def _make_typed_jaxpr(traceable: lu.WrappedFun, in_avals: Sequence[core.AbstractValue]):
  pvals = [pe.PartialVal.unknown(aval) for aval in in_avals]
  jaxpr, pvals_out, consts = pe.trace_to_jaxpr(traceable, pvals, instantiate=True)
  out_avals, _ = unzip2(pvals_out)
  return core.TypedJaxpr(jaxpr, consts, in_avals, _map(raise_to_shaped, out_avals))


def _scan_batching_rule(args, dims, reverse, length, jaxpr, num_consts,
                        num_carry, linear):
  num_ys = len(jaxpr.out_avals) - num_carry
  size, = {x.shape[d] for x, d in zip(args, dims) if d is not batching.not_mapped}
  orig_batched = [d is not batching.not_mapped for d in dims]
  const_batched, init_batched, xs_batched = split_list(orig_batched, [num_consts, num_carry])

  # Fixpoint computation of which carry are batched: either
  # batched from init, or the carry out is batched. Each iteration promotes
  # at least one carry to batched. We need at most len(carry) iterations,
  # but we need one last iteration to prepare the jaxpr based on the final
  # carry_batched.
  carry_batched = init_batched
  for _ in range(1 + len(carry_batched)):
    batched = const_batched + carry_batched + xs_batched
    jaxpr_batched, batched_out = batching.batch_jaxpr(
        jaxpr, size, batched, instantiate=carry_batched + [False] * num_ys)
    carry_batched_out, ys_batched = batched_out[:num_carry], batched_out[num_carry:]
    if carry_batched_out == carry_batched:
      break
    else:
      carry_batched = _map(operator.or_, carry_batched, carry_batched_out)
  else:
    assert False, "Fixpoint not reached"

  consts, init, xs = split_list(args, [num_consts, num_carry])
  consts_bdims, init_bdims, xs_bdims = split_list(dims, [num_consts, num_carry])
  new_consts = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
                else x for x, d in zip(consts, consts_bdims)]
  new_init = [batching.broadcast(x, size, 0) if now_batched and not was_batched
              else batching.moveaxis(x, d, 0) if now_batched else x
              for x, d, was_batched, now_batched in
              zip(init, init_bdims, init_batched, carry_batched)]
  new_xs = [batching.moveaxis(x, d, 1) if d is not batching.not_mapped and d != 1
            else x for x, d in zip(xs, xs_bdims)]
  new_args = new_consts + new_init + new_xs

  outs = scan_p.bind(*new_args, reverse=reverse, length=length, jaxpr=jaxpr_batched,
                     num_consts=num_consts, num_carry=num_carry, linear=linear)
  carry_bdims = [0 if b else batching.not_mapped for b in carry_batched]
  ys_bdims = [1 if b else batching.not_mapped for b in ys_batched]
  return outs, carry_bdims + ys_bdims

def _scan_shape_rule(shapes, reverse, length, jaxpr,
                     num_consts, num_carry, linear):
  const_shexprs, init_shexprs, xs_shexprs = split_list(shapes, [num_consts, num_carry])
  _, y_avals = split_list(jaxpr.out_avals, [num_carry])
  ys_shapes = [(length,) + tuple(y_aval.shape) for y_aval in y_avals]
  return init_shexprs + ys_shapes

def _scan_masking_rule(shape_envs, padded_vals, shape_exprs, reverse, length,
                       jaxpr, num_consts, num_carry, linear):
  out_shape = _scan_shape_rule(shape_exprs, reverse, length, jaxpr,
                               num_consts, num_carry, linear)
  dynamic_length = length.evaluate(shape_envs.logical)
  masked_jaxpr = _masked_scan_jaxpr(jaxpr, num_consts, num_carry)
  consts, init, xs = split_list(padded_vals, [num_consts, num_carry])
  max_length, = {x.shape[0] for x in xs}
  const_linear, init_linear, xs_linear = split_list(linear, [num_consts, num_carry])
  out_vals = scan_p.bind(
      *itertools.chain([dynamic_length] + consts, [0], init, xs),
      reverse=reverse, length=max_length, jaxpr=masked_jaxpr,
      num_consts=1 + num_consts, num_carry=1 + num_carry,
      linear=tuple([False] + const_linear + [False] + init_linear + xs_linear))
  return out_vals[1:], out_shape

def _masked_scan_jaxpr(jaxpr, num_consts, num_carry):
  fun = core.jaxpr_as_fun(jaxpr)

  @lu.wrap_init
  def masked(*args):
    [dynamic_length], consts, [i], carry, xs = split_list(
        args, [1, num_consts, 1, num_carry])
    out = fun(*(consts + carry + xs))
    new_carry, ys = split_list(out, [num_carry])
    new_carry = [lax.select(i < dynamic_length, new_c, c)
                 for new_c, c in zip(new_carry, carry)]
    return [i + 1] + new_carry + ys

  aval = ShapedArray((), dtypes.int_)
  const_avals, carry_avals, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
  return _make_typed_jaxpr(masked, [aval] + const_avals + [aval] + carry_avals + x_avals)

def scan_bind(*args, reverse, length, num_consts, num_carry, jaxpr, linear):
  if not core.skip_checks:
    assert len(linear) == len(args)
    consts, init, xs = split_list(args, [num_consts, num_carry])
    consts_avals, init_avals, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
    assert all(_map(typecheck, consts_avals, consts)), (consts, consts_avals)
    assert all(_map(typecheck, init_avals, init))
    carry_avals, _ = split_list(jaxpr.out_avals, [num_carry])
    assert all(_map(typematch, init_avals, carry_avals))
    core.check_jaxpr(jaxpr.jaxpr)
  return core.Primitive.bind(scan_p, *args, reverse=reverse, length=length,
                             jaxpr=jaxpr, num_consts=num_consts,
                             num_carry=num_carry, linear=linear)

scan_p = core.Primitive("scan")
scan_p.multiple_results = True
scan_p.def_custom_bind(scan_bind)
scan_p.def_impl(_scan_impl)
# scan_p.def_impl(partial(xla.apply_primitive, scan_p))  # TODO(mattjj): re-enable
scan_p.def_abstract_eval(_scan_abstract_eval)
ad.primitive_jvps[scan_p] = _scan_jvp
ad.primitive_transposes[scan_p] = _scan_transpose
pe.custom_partial_eval_rules[scan_p] = _scan_partial_eval
xla.initial_style_translations[scan_p] = \
    xla.lower_fun_initial_style(_scan_impl)
batching.primitive_batchers[scan_p] = _scan_batching_rule
masking.shape_parameterized_primitive_rules[scan_p] = _scan_masking_rule


def map(f, xs):
  """Map a function over leading array axes.

  Like Python's builtin map, except inputs and outputs are in the form of
  stacked arrays. Consider using the ``jax.vmap`` transform instead, unless you
  need to apply a function element by element for reduced memory usage or
  heterogeneous computation with other control flow primitives.

  When ``xs`` is an array type, the semantics of ``map`` are given by this
  Python implementation::

    def map(f, xs):
      return np.stack([f(x) for x in xs])

  Like ``scan``, ``map`` is implemented in terms of JAX primitives so many of
  the same advantages over a Python loop apply: ``xs`` may be an arbitrary
  nested pytree type, and the mapped computation is compiled only once.

  Args:
    f: a Python function to apply element-wise over the first axis or axes of
      ``xs``.
    xs: values over which to map along the leading axis.

  Returns:
    Mapped values.
  """
  g = lambda _, x: ((), f(x))
  _, ys = scan(g, (), xs)
  return ys


def _concat_masking_rule(padded_vals, logical_shapes, dimension):
  result = lax.concatenate(padded_vals, dimension)  # fragmented
  offset = 0
  for padded_val, logical_shape in zip(padded_vals, logical_shapes):
    result = _memcpy(dimension, logical_shape[dimension], padded_val,
                     result, offset)
    offset = offset + logical_shape[dimension]
  return result

def _memcpy(axis, num, src, dst, offset):
  def body(i, dst):
    update = lax.dynamic_index_in_dim(src, i, axis)
    return lax.dynamic_update_index_in_dim(dst, update, i + offset, axis)
  return fori_loop(0, num, body, dst)

masking.masking_rules[lax.concatenate_p] = _concat_masking_rule


def _check_tree(func_name, expected_name, actual_tree, expected_tree):
  if actual_tree != expected_tree:
    raise TypeError(
        "{}() output pytree structure must match {}, got {} and {}."
        .format(func_name, expected_name, actual_tree, expected_tree))


def _check_tree_and_avals(what, tree1, avals1, tree2, avals2):
  """Raises TypeError if (tree1, avals1) does not match (tree2, avals2).

  Corresponding `tree` and `avals` must match in the sense that the number of
  leaves in `tree` must be equal to the length of `avals`. `what` will be
  prepended to details of the mismatch in TypeError.
  """
  if tree1 != tree2:
    msg = ("{} must have same type structure, got {} and {}.")
    raise TypeError(msg.format(what, tree1, tree2))
  if not all(safe_map(typematch, avals1, avals2)):
    msg = ("{} must have identical types, "
           "got\n{}\nand\n{}.")
    raise TypeError(msg.format(what, tree_unflatten(tree1, avals1),
                               tree_unflatten(tree2, avals2)))


def _stop_gradient_fun(f):
  """Create a version of f() that stops all gradients."""
  def wrapper(*args, **kwargs):
    args_flat, in_args_tree = tree_flatten((args, kwargs))
    args_avals = tuple(_map(_abstractify, args_flat))
    g = lambda a, b: f(*a, **b)
    jaxpr, consts, out_tree = _initial_style_jaxpr(g, in_args_tree, args_avals)
    out = core.jaxpr_as_fun(jaxpr)(*lax.stop_gradient(consts + tuple(args_flat)))
    return tree_unflatten(out_tree, out)
  return wrapper


_RootTuple = collections.namedtuple('_RootTuple', 'f, solve, l_and_s')


def _split_root_args(args, const_lengths):
  params_list = split_list(args, list(const_lengths))
  return _RootTuple(*params_list[:-1]), params_list[-1]


def custom_root(f, initial_guess, solve, tangent_solve):
  """Differentiably solve for a roots of a function.

  This is a low-level routine, mostly intended for internal use in JAX.
  Gradients of custom_root() are defined with respect to closed-over variables
  from the provided function ``f`` via the implicit function theorem:
  https://en.wikipedia.org/wiki/Implicit_function_theorem

  Args:
    f: function for which to find a root. Should accept a single argument,
      return a tree of arrays with the same structure as its input.
    initial_guess: initial guess for a zero of f.
    solve: function to solve for the roots of f. Should take two positional
      arguments, f and initial_guess, and return a solution with the same
      structure as initial_guess such that func(solution) = 0. In other words,
      the following is assumed to be true (but not checked)::

        solution = solve(f, initial_guess)
        error = f(solution)
        assert all(error == 0)

    tangent_solve: function to solve the tangent system. Should take two
      positional arguments, a linear function ``g`` (the function ``f``
      linearized at its root) and a tree of array(s) ``y`` with the same
      structure as initial_guess, and return a solution ``x`` such that
      ``g(x)=y``:

      - For scalar ``y``, use ``lambda g, y: y / g(1.0)``.
      - For vector ``y``, you could use a linear solve with the Jacobian, if
        dimensionality of ``y`` is not too large:
        ``lambda g, y: np.linalg.solve(jacobian(g)(y), y)``.

  Returns:
    The result of calling solve(f, initial_guess) with gradients defined via
    implicit differentiation assuming ``f(solve(f, initial_guess)) == 0``.
  """
  guess_flat, in_args_tree = tree_flatten((initial_guess,))
  guess_avals = tuple(_map(_abstractify, guess_flat))
  f_jaxpr, f_consts, out_tree = _initial_style_jaxpr(
      f, in_args_tree, guess_avals)

  in_tree, = treedef_children(in_args_tree)
  _check_tree("f", "initial_guess", out_tree, in_tree)

  solve_jaxpr, solve_consts, solution_tree = _initial_style_jaxpr(
      partial(solve, _stop_gradient_fun(f)), in_args_tree, guess_avals)
  _check_tree("solve", "initial_guess", solution_tree, in_tree)

  def linearize_and_solve(x, b):
    unchecked_zeros, f_jvp = jax.linearize(f, x)
    return tangent_solve(f_jvp, b)

  l_and_s_jaxpr, l_and_s_consts, out_tree = _initial_style_jaxpr(
      linearize_and_solve, treedef_tuple((in_tree,) * 2), guess_avals * 2)
  _check_tree("tangent_solve", "x", out_tree, in_tree)

  all_consts = [f_consts, solve_consts, l_and_s_consts]
  const_lengths = _RootTuple(*_map(len, all_consts))
  jaxprs = _RootTuple(f_jaxpr, solve_jaxpr, l_and_s_jaxpr)

  out_flat = _custom_root(
      const_lengths, jaxprs, *(_flatten(all_consts) + guess_flat))
  return tree_unflatten(out_tree, out_flat)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _custom_root(const_lengths, jaxprs, *args):
  params, initial_guess = _split_root_args(args, const_lengths)
  solution = core.jaxpr_as_fun(jaxprs.solve)(*(params.solve + initial_guess))
  return solution


@_custom_root.defjvp
def _root_jvp(const_lengths, jaxprs, primals, tangents):
  params, _ = _split_root_args(primals, const_lengths)
  solution = _custom_root(const_lengths, jaxprs, *primals)

  params_dot, _ = _split_root_args(tangents, const_lengths)

  # F(m, u) = 0      # system of equations in u, parameterized by m
  #                  # solution is u*(m) defined in a neighborhood
  # F(m, u*(m)) = 0  # satisfied in a neighborhood
  #
  # ∂_0 F(m, u*(m)) + ∂_1 F(m, u*(m)) ∂ u*(m) = 0       # implied by line above
  # ∂ u*(m) = - (∂_1 F(m, u*(m)))^{-1} ∂_0 F(m, u*(m))  # rearrange
  #
  # ∂ u*(m)[v] = - (∂_1 F(m, u*(m)))^{-1} [∂_0 F(m, u*(m))[v]]  # jvp

  f = core.jaxpr_as_fun(jaxprs.f)
  linearize_and_solve = partial(
      core.jaxpr_as_fun(jaxprs.l_and_s), *params.l_and_s)
  f_at_solution = lambda *params: f(*itertools.chain(params, solution))
  _, rhs = ad.jvp(lu.wrap_init(f_at_solution)).call_wrapped(
      params.f, params_dot.f)
  solution_dot = _map(
      operator.neg, linearize_and_solve(*itertools.chain(solution, rhs)))

  return solution, solution_dot


class _LinearSolveTuple(collections.namedtuple(
    '_LinearSolveTuple', 'matvec, vecmat, solve, transpose_solve')):

  def transpose(self):
    return type(self)(self.vecmat, self.matvec, self.transpose_solve, self.solve)


def _split_linear_solve_args(args, const_lengths):
  params_list = split_list(args, list(const_lengths))
  return _LinearSolveTuple(*params_list[:-1]), params_list[-1]


def _transpose_function(linear_fun, primals):
  """Transpose a linear function."""
  # TODO(shoyer): can we use something more direct than the vjp machinery?
  # It's particularly awkward that we need the second argument to give
  # particular values of the primals, which are entirely arbitrary.
  _, vjp_fun = jax.vjp(linear_fun, primals)

  def transposed_fun(x):
    (y,) = vjp_fun(x)
    return y

  return transposed_fun


def _flatten(args):
  return [x for arg in args for x in arg]


def _check_shapes(func_name, expected_name, actual, expected, tree):
  actual_shapes = _map(onp.shape, actual)
  expected_shapes = _map(onp.shape, expected)
  if actual_shapes != expected_shapes:
    raise ValueError('{}() output shapes must match {}, got {} and {}'
                     .format(func_name, expected_name,
                             tree_unflatten(tree, actual_shapes),
                             tree_unflatten(tree, expected_shapes)))


def custom_linear_solve(
    matvec, b, solve, transpose_solve=None, symmetric=False):
  """Perform a matrix-free linear solve with implicitly defined gradients.

  This function allows for overriding or defining gradients for a linear
  solve directly via implicit differentiation at the solution, rather than by
  differentiating *through* the solve operation. This can sometimes be much faster
  or more numerically stable, or differentiating through the solve operation
  may not even be implemented (e.g., if ``solve`` uses ``lax.while_loop``).

  Required invariant::

      x = solve(matvec, b)  # solve the linear equation
      assert matvec(x) == b  # not checked

  Args:
    matvec: linear function to invert. Must be differentiable.
    b: constant right handle side of the equation. May be any nested structure
      of arrays.
    solve: higher level function that solves for solution to the linear
      equation, i.e., ``solve(matvec, x)) == x`` for all ``x`` of the same form
      as ``b``. This function need not be differentiable.
    transpose_solve: higher level function for solving the transpose linear
      equation, i.e., ``transpose_solve(vecmat, x) == x``, where ``vecmat`` is
      the transpose of the linear map ``matvec`` (computed automatically with
      autodiff). Required for backwards mode automatic differentiation, unless
      ``symmetric=True``, in which case ``solve`` provides the default value.
    symmetric: bool indicating if it is safe to assume the linear map
      corresponds to a symmetric matrix, i.e., ``matvec == vecmat``.

  Returns:
    Result of ``solve(matvec, b)``, with gradients defined assuming that the
    solution ``x`` satisfies the linear equation ``matvec(x) == b``.
  """
  if transpose_solve is None and symmetric:
    transpose_solve = solve

  b_flat, in_args_tree = tree_flatten((b,))
  b_avals = tuple(_map(_abstractify, b_flat))
  matvec_jaxpr, matvec_consts, out_tree = _initial_style_jaxpr(
      matvec, in_args_tree, b_avals)

  tree, = treedef_children(in_args_tree)
  _check_tree("matvec", "b", out_tree, tree)

  solve_jaxpr, solve_consts, out_tree = _initial_style_jaxpr(
      partial(solve, matvec), in_args_tree, b_avals)
  _check_tree("solve", "b", out_tree, tree)

  if transpose_solve is None:
    vecmat_jaxpr = tr_solve_jaxpr = None
    vecmat_consts = tr_solve_consts = []
  else:
    if symmetric:
      vecmat = matvec
      vecmat_jaxpr = matvec_jaxpr
      vecmat_consts = matvec_consts
    else:
      vecmat = _transpose_function(matvec, b)
      vecmat_jaxpr, vecmat_consts, out_tree = _initial_style_jaxpr(
          vecmat, in_args_tree, b_avals)
      assert out_tree == tree

    tr_solve_jaxpr, tr_solve_consts, out_tree = _initial_style_jaxpr(
        partial(transpose_solve, vecmat), in_args_tree, b_avals)
    _check_tree("transpose_solve", "b", out_tree, tree)

  all_consts = [matvec_consts, vecmat_consts, solve_consts, tr_solve_consts]
  const_lengths = _LinearSolveTuple(*_map(len, all_consts))
  jaxprs = _LinearSolveTuple(
      matvec_jaxpr, vecmat_jaxpr, solve_jaxpr, tr_solve_jaxpr)

  out_flat = linear_solve_p.bind(
      *(_flatten(all_consts) + b_flat),
      const_lengths=const_lengths, jaxprs=jaxprs, tree=tree)
  return tree_unflatten(tree, out_flat)


def _linear_solve_abstract_eval(*args, **kwargs):
  return _map(raise_to_shaped, args[sum(kwargs['const_lengths']):])


def _custom_linear_solve_impl(*args, **kwargs):
  const_lengths, jaxprs, tree = split_dict(
      kwargs, ['const_lengths', 'jaxprs', 'tree'])
  params, b = _split_linear_solve_args(args, const_lengths)
  x = core.jaxpr_as_fun(jaxprs.solve)(*(params.solve + b))
  _check_shapes('solve', 'b', x, b, tree)
  return x


def _tangent_linear_map(func, params, params_dot, *x):
  """Compute the tangent of a linear map.

  Assuming ``func(*params, *x)`` is linear in ``x`` and computes ``A @ x``,
  this function computes ``∂A @ x``.
  """
  assert any(type(p) is not ad_util.Zero for p in params_dot)
  zeros = _map(ad_util.Zero.from_value, x)
  _, out_tangent = ad.jvp(lu.wrap_init(func)).call_wrapped(
      params + list(x), params_dot + zeros)
  return out_tangent


def _custom_linear_solve_jvp(primals, tangents, const_lengths, jaxprs, tree):
  # A x - b = 0
  # ∂A x + A ∂x - ∂b = 0
  # ∂x = A^{-1} (∂b - ∂A x)

  kwargs = dict(const_lengths=const_lengths, jaxprs=jaxprs, tree=tree)
  x = linear_solve_p.bind(*primals, **kwargs)

  params, _ = _split_linear_solve_args(primals, const_lengths)
  params_dot, b_dot = _split_linear_solve_args(tangents, const_lengths)

  if all(type(p) is ad_util.Zero for p in params_dot.matvec):
    # no need to evaluate matvec_tangents
    rhs = b_dot
  else:
    matvec_tangents = _tangent_linear_map(
        core.jaxpr_as_fun(jaxprs.matvec), params.matvec, params_dot.matvec, *x)
    _check_shapes("matvec", "b", matvec_tangents, x, tree)
    rhs = _map(ad.add_tangents, b_dot, _map(operator.neg, matvec_tangents))

  x_dot = linear_solve_p.bind(*(_flatten(params) + rhs), **kwargs)

  return x, x_dot


def _linear_solve_transpose_rule(cotangent, *primals, **kwargs):
  const_lengths, jaxprs, tree = split_dict(
      kwargs, ['const_lengths', 'jaxprs', 'tree'])

  if jaxprs.transpose_solve is None:
    raise TypeError('transpose_solve required for backwards mode automatic '
                    'differentiation of custom_linear_solve')

  params, b = _split_linear_solve_args(primals, const_lengths)
  assert all(ad.is_undefined_primal(x) for x in b)
  cotangent_b = linear_solve_p.bind(
      *(_flatten(params.transpose()) + cotangent),
      const_lengths=const_lengths.transpose(), jaxprs=jaxprs.transpose(),
      tree=tree)
  return [None] * sum(const_lengths) + cotangent_b


def _linear_solve_batching_rule(args, dims, **kwargs):
  const_lengths, jaxprs, tree = split_dict(kwargs,
                                           ["const_lengths", "jaxprs", "tree"])
  orig_bat = [d is not batching.not_mapped for d in dims]
  size, = {
      a.shape[d] for a, d in zip(args, dims) if d is not batching.not_mapped
  }

  params, b = _split_linear_solve_args(args, const_lengths)
  params_dims, b_dims = _split_linear_solve_args(dims, const_lengths)
  params_bat, orig_b_bat = _split_linear_solve_args(orig_bat, const_lengths)

  (matvec, vecmat, solve, solve_t) = jaxprs
  (matvec_bat, vecmat_bat, solve_bat, solve_t_bat) = params_bat

  # Fixpoint computation of which parts of x and b are batched; we need to
  # ensure this is consistent between all four jaxprs
  b_bat = orig_b_bat
  x_bat = [False] * len(solve.out_avals)
  for i in range(1 + len(orig_b_bat) + len(solve.out_avals)):
    # Apply vecmat and solve -> new batched parts of x
    solve_jaxpr_batched, solve_x_bat = batching.batch_jaxpr(
        solve, size, solve_bat + b_bat, instantiate=x_bat)
    if vecmat is None:
      vecmat_jaxpr_batched = None
      x_bat_out = solve_x_bat
    else:
      vecmat_jaxpr_batched, vecmat_x_bat = batching.batch_jaxpr(
          vecmat, size, vecmat_bat + b_bat, instantiate=x_bat)
      x_bat_out = _map(operator.or_, vecmat_x_bat, solve_x_bat)
    # Apply matvec and solve_t -> new batched parts of b
    matvec_jaxpr_batched, matvec_b_bat = batching.batch_jaxpr(
        matvec, size, matvec_bat + x_bat_out, instantiate=b_bat)
    if solve_t is None:
      solve_t_jaxpr_batched = None
      b_bat_out = _map(operator.or_, matvec_b_bat, orig_b_bat)
    else:
      solve_t_jaxpr_batched, solve_t_b_bat = batching.batch_jaxpr(
          solve_t, size, solve_t_bat + x_bat_out, instantiate=b_bat)
      b_bat_out = _map(lambda m, s, o: m or s or o, matvec_b_bat, solve_t_b_bat,
                      orig_b_bat)
    if x_bat_out == x_bat and b_bat_out == b_bat:
      break
    else:
      x_bat = x_bat_out
      b_bat = b_bat_out
  else:
    assert False, "Fixedpoint not reached"

  batched_jaxprs = _LinearSolveTuple(matvec_jaxpr_batched, vecmat_jaxpr_batched,
                                     solve_jaxpr_batched, solve_t_jaxpr_batched)

  # Move batched axes to the front
  new_params = [
      batching.moveaxis(x, d, 0)
      if d is not batching.not_mapped and d != 0 else x
      for x, d in zip(_flatten(params), _flatten(params_dims))
  ]
  # Broadcast out b if necessary
  new_b = [
      batching.broadcast(x, size, 0) if now_bat and not was_bat else
      batching.moveaxis(x, d, 0) if now_bat and d != 0 else x
      for x, d, was_bat, now_bat in zip(b, b_dims, orig_b_bat, b_bat)
  ]

  outs = linear_solve_p.bind(
      *(new_params + new_b),
      const_lengths=const_lengths,
      jaxprs=batched_jaxprs,
      tree=tree)
  out_dims = [0 if batched else batching.not_mapped for batched in b_bat]
  return outs, out_dims


linear_solve_p = core.Primitive('custom_linear_solve')
linear_solve_p.multiple_results = True
linear_solve_p.def_impl(_custom_linear_solve_impl)
linear_solve_p.def_abstract_eval(_linear_solve_abstract_eval)
ad.primitive_jvps[linear_solve_p] = _custom_linear_solve_jvp
xla.initial_style_translations[linear_solve_p] = \
    xla.lower_fun_initial_style(_custom_linear_solve_impl)
ad.primitive_transposes[linear_solve_p] = _linear_solve_transpose_rule
batching.primitive_batchers[linear_solve_p] = _linear_solve_batching_rule


def _interleave(a, b):
  """Given two Tensors of static shape, interleave them along the first axis."""
  # TODO(mattjj)
  import jax.numpy as np
  # [a b c ...] [d e f ...] -> [a d b e c f ...]
  half_num_elems = b.shape[0]

  if a.shape[0] > b.shape[0]:
    return np.concatenate(
        [np.reshape(np.stack([a[: -1], b], axis=1),
                    (2 * half_num_elems,) + a.shape[1:]),
         a[-1:]], axis=0)
  else:
    return np.reshape(np.stack([a, b], axis=1),
                      (2 * half_num_elems,) + a.shape[1:])

def associative_scan(fn, elems):
  """Perform a scan with an associative binary operation, in parallel.

  Args:
    fn: Python callable implementing an associative binary operation with
      signature `r = fn(a, b)`. This must satisfy associativity:
      `fn(a, fn(b, c)) == fn(fn(a, b), c)`. The inputs and result are
      (possibly nested structures of) `Tensor`(s), matching `elems`. Each
      `Tensor` has a leading batch dimension in place of `num_elems`; the `fn`
      is expected to map over this dimension. The result `r` has the same shape
      (and structure) as the two inputs `a` and `b`.
    elems: A (possibly nested structure of) `Tensor`(s), each with leading
      dimension `num_elems`, which must be known statically.
  Returns:
    result: A (possibly nested structure of) `Tensor`(s) of the same shape
      and structure as `elems`, in which the `k`th element is the result of
      recursively applying `fn` to combine the first `k` elements of
      `elems`. For example, given `elems = [a, b, c, ...]`, the result
      would be `[a, fn(a, b), fn(fn(a, b), c), ...]`.

  #### Examples

  ```python
  # Example 1: Partials sums of numbers.

  np.associative_scan(operator.add, np.arange(0, 4))
  # ==> [ 0, 1, 3, 6]

  # Example 2: Partial products of random matrices.

  np.associative_scan(np.matmul, matrices)
  ```
  """
  elems_flat, tree = tree_flatten(elems)

  def lowered_fn(a_flat, b_flat):
    # Lower `fn` to operate on flattened sequences of elems.
    a = tree_unflatten(tree, a_flat)
    b = tree_unflatten(tree, b_flat)
    c = fn(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat

  # Check that all inputs have a consistent leading dimension `num_elems`.
  num_elems = int(elems_flat[0].shape[0])

  if not all(int(elem.shape[0]) == num_elems for elem in elems_flat[1:]):
    raise ValueError('Input `Tensor`s must have the same first dimension.'
                     ' (saw: {})'.format([elems.shape for elem in elems_flat]))

  if num_elems < 2:
    return elems

  # Summary of algorithm:
  #
  # Consider elements of `_scan(elems)` at odd indices. That's the same as first
  # summing successive pairs of elements of `elems` and performing a scan on
  # that half sized tensor. We perform the latter scan by recursion.
  #
  # Now consider the even elements of `_scan(elems)`. These can be computed
  # from the odd elements of `_scan(elems)` by adding each odd element of
  # `_scan(elems)` to the matching even element in the original `elems`.
  #
  # We return the odd and even elements interleaved.
  #
  # For the base case of the recursion we return the first element
  # of `elems` followed by the sum of the first two elements computed as
  # a (small two-down-to-one) reduction step.
  def _scan(elems):
    """Perform scan on `elems`."""

    num_elems = elems[0].shape[0]

    reduced_elems = lowered_fn([elem[0:-1:2] for elem in elems],
                               [elem[1::2] for elem in elems])

    if reduced_elems[0].shape[0] == 1:
      # Base case has either 2 or 3 elements.
      if num_elems == 2:
        return [lax.concatenate([elem[0:1], reduced_elem], dimension=0)
                for (reduced_elem, elem) in zip(reduced_elems, elems)]
      elif num_elems == 3:
        reduced_reduced_elems = lowered_fn(
          reduced_elems,
          [elem[2:3] for elem in elems])
        return [
            lax.concatenate([elem[0:1], reduced_elem, reduced_reduced_elem],
                            dimension=0)
            for (reduced_reduced_elem, reduced_elem, elem)
            in zip(reduced_reduced_elems, reduced_elems, elems)]

    # Recursively compute scan for partially reduced tensors.
    odd_elems = _scan(reduced_elems)

    if num_elems % 2 == 0:
      results = lowered_fn([odd_elem[:-1] for odd_elem in odd_elems],
                           [elem[2::2] for elem in elems])
    else:
      results = lowered_fn([odd_elem for odd_elem in odd_elems],
                           [elem[2::2] for elem in elems])

    # The first element of a scan is the same as the first element
    # of the original `elems`.
    even_elems = [lax.concatenate([elem[0:1], result], dimension=0)
                  for (elem, result) in zip(elems, results)]
    return tuple(_map(_interleave, even_elems, odd_elems))

  scans = _scan(elems_flat)

  return tree_unflatten(tree, scans)
