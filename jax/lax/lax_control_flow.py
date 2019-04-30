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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from jax import api
from jax import core
from jax.lax import lax
from jax import linear_util as lu
from jax.abstract_arrays import ConcreteArray, ShapedArray, UnshapedArray
from jax.api_util import (
    pytree_to_flatjaxtuple, pytree_fun_to_flatjaxtuple_fun,
    pytree_to_jaxtupletree, pytree_fun_to_jaxtupletree_fun)
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.util import partial, unzip2
from jax.tree_util import build_tree, tree_unflatten

def while_loop(cond_fun, body_fun, init_val):
  """Call `body_fun` repeatedly in a loop while `cond_fun` is True.

  Arguments:
    cond_fun: pure function of type `T -> Bool`.
    body_fun: pure function of type `T -> T`.
    init_val: value of type `T`, a type that can be a scalar, array, or any
      (nested) Python tuple/list/dict thereof.

  Returns:
    The output from the final iteration of body_fun, of type `T`.

  The semantics of `while_loop` are given by this Python implementation::

    def while_loop(cond_fun, body_fun, init_val):
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val

  Unlike that pure Python version, `while_loop` is a JAX primitive and is
  lowered to a single XLA While HLO. That makes it useful for reducing
  compilation times for jit-compiled functions, since native Python loop
  constructs in an `@jit` function are unrolled, leading to large XLA
  computations.

  Another difference from using Python-native loop constructs is that
  `while_loop` is not (yet) reverse-mode differentiable because XLA computations
  require static bounds on memory requirements.
  """
  init_val_flat, in_tree = pytree_to_jaxtupletree(init_val)
  flat_body_fun, out_tree = pytree_fun_to_jaxtupletree_fun(lu.wrap_init(body_fun), (in_tree,))
  flat_cond_fun, _ = pytree_fun_to_jaxtupletree_fun(lu.wrap_init(cond_fun), (in_tree,))

  carry_pval_flat = carry_aval, _ = lax._abstractify(init_val_flat)
  cond_jaxpr, cond_pval_out, cond_consts = pe.trace_to_jaxpr(flat_cond_fun, (carry_pval_flat,))
  body_jaxpr, body_pval_out, body_consts = pe.trace_to_jaxpr(flat_body_fun, (carry_pval_flat,), instantiate=True)
  carry_aval_out, _ = body_pval_out
  assert isinstance(carry_aval_out, core.AbstractValue)
  assert carry_aval == core.lattice_join(carry_aval, carry_aval_out)

  cond_pv, cond_const = cond_pval_out
  if cond_pv is None:
    # cond_fun evaluates to a constant, so don't need to generate a while_loop
    if cond_const:
      raise ValueError("infinite loop with no effects")
    else:
      return init_val
  else:
    assert isinstance(cond_pv, core.AbstractValue)
    if (not isinstance(cond_pv, ShapedArray) or cond_pv.shape
        or cond_pv.dtype != onp.bool_):
      msg = "while_loop cond_fun must return a scalar boolean, got {}."
      raise TypeError(msg.format(cond_pv))

  # We don't want to promote literal constants as loop arguments; there are
  # sometimes many of them. We pass tracers as loop arguments, but leave
  # nontracers as constants. We also sort the constants so the nontracers are
  # first.
  def split_tracers_and_nontracers(jaxpr, consts):
    tracer = []
    nontracer = []
    for x in zip(jaxpr.constvars, consts):
      # TODO(phawkins): We avoid treating DeviceArrays as constant literals so
      # we don't copy large arrays back to the host. We probably should relax
      # this and either always copy small constants, or opportunistically use
      # DeviceArray values for which we already know npy_value.
      not_literal_const = isinstance(x[1], (core.Tracer, xla.DeviceArray))
      (tracer if not_literal_const else nontracer).append(x)
    tracer_vars, tracer_consts = unzip2(tracer)
    nontracer_vars, nontracer_consts = unzip2(nontracer)
    return nontracer_vars + tracer_vars, nontracer_consts, tracer_consts

  cond_split = split_tracers_and_nontracers(cond_jaxpr, cond_consts)
  cond_jaxpr.constvars, cond_nontracer_consts, cond_tracer_consts = cond_split
  body_split = split_tracers_and_nontracers(body_jaxpr, body_consts)
  body_jaxpr.constvars, body_nontracer_consts, body_tracer_consts = body_split

  if out_tree() != in_tree:
    raise TypeError("body_fun input and output must have identical structure")
  out_flat = while_p.bind(
      init_val_flat,
      core.pack(cond_tracer_consts), core.pack(body_tracer_consts),
      cond_consts=lax._OpaqueParam(cond_nontracer_consts),
      body_consts=lax._OpaqueParam(body_nontracer_consts),
      aval_out=carry_aval_out, cond_jaxpr=cond_jaxpr, body_jaxpr=body_jaxpr)
  return build_tree(out_tree(), out_flat)


def _while_loop_abstract_eval(init_val, cond_tracer_consts, body_tracer_consts,
                              cond_consts, body_consts, aval_out,
                              cond_jaxpr, body_jaxpr):
  return maybe_tracer_tuple_to_abstract_tuple(aval_out)

def _while_loop_translation_rule(c, init_val, cond_tracer_consts,
                                 body_tracer_consts, cond_consts, body_consts,
                                 aval_out, cond_jaxpr, body_jaxpr):
  loop_carry = c.Tuple(init_val, cond_tracer_consts, body_tracer_consts)
  shape = c.GetShape(loop_carry)

  loop_carry_var = pe.Var(0, "loop_carry")
  outvar = pe.Var(0, "loop_carry_out")
  cond_var = pe.Var(0, "cond_consts")
  body_var = pe.Var(0, "body_consts")

  num_cond_consts = len(cond_consts.val)
  assert len(cond_jaxpr.invars) == 1
  cond_jaxpr_converted = cond_jaxpr.copy()
  cond_jaxpr_converted.constvars = cond_jaxpr.constvars[:num_cond_consts]
  cond_jaxpr_converted.invars = [loop_carry_var]
  cond_jaxpr_converted.eqns = (
      [_unpack_eqn(loop_carry_var, [cond_jaxpr.invars[0], cond_var, body_var]),
       _unpack_eqn(cond_var, cond_jaxpr.constvars[num_cond_consts:])]
      + list(cond_jaxpr.eqns))

  num_body_consts = len(body_consts.val)
  assert len(body_jaxpr.invars) == 1
  body_jaxpr_converted = body_jaxpr.copy()
  body_jaxpr_converted.constvars = body_jaxpr.constvars[:num_body_consts]
  body_jaxpr_converted.invars = [loop_carry_var]
  body_jaxpr_converted.outvar = outvar
  body_jaxpr_converted.eqns = (
      [_unpack_eqn(loop_carry_var, [body_jaxpr.invars[0], cond_var, body_var]),
       _unpack_eqn(body_var, body_jaxpr.constvars[num_body_consts:])]
      + list(body_jaxpr.eqns) +
      [_pack_eqn([body_jaxpr.outvar, cond_var, body_var], outvar)])

  cond_computation = xla.jaxpr_computation(
      cond_jaxpr_converted, cond_consts.val, (), shape)
  body_computation = xla.jaxpr_computation(
      body_jaxpr_converted, body_consts.val, (), shape)
  full_ans = c.While(cond_computation, body_computation, loop_carry)
  return c.GetTupleElement(full_ans, 0)

def _while_loop_batching_rule(batched_args, batch_dims, cond_consts,
                              body_consts, aval_out, cond_jaxpr, body_jaxpr):
  # See https://github.com/google/jax/issues/441 for a discussion.
  # To batch a while_loop, we need to do some masking, since the elements of the
  # batch may run for different numbers of iterations. We perform that masking
  # using lax.select, and keep the loop running so long as any of the batch
  # elements need by effectively using an np.any(...) in the cond_fun.
  # The basic strategy here is to lift `cond_jaxpr` and `body_jaxpr` back into
  # traceable Python functions using `core.eval_jaxpr`. Then we can batch them
  # using `batching.batch_transform` (the transform underlying `api.vmap`). This
  # code also avoids broadcasting `cond_tracer_consts` and `body_tracer_consts`.
  init_val, cond_tracer_consts, body_tracer_consts = batched_args
  init_val_bd, cond_tracer_consts_bd, body_tracer_consts_bd = batch_dims

  sizes = lax._reduce(set.union, map(batching.dimsize, batch_dims, batched_args))
  size = sizes.pop()
  assert not sizes

  # TODO(mattjj): if cond_tracer_consts_bd is also None, we could keep cond_fun
  # unbatched and avoid the masking logic, but we ignore that optimization
  init_val = batching.bdim_at_front(init_val, init_val_bd, size,
                                    force_broadcast=True)
  init_val_bd = 0

  def batched_cond_fun(batched_loop_carry):
    @lu.wrap_init
    def lifted(loop_carry, cond_tracer_consts):
      cond_tracer_consts = tuple(x for x in cond_tracer_consts)
      return core.eval_jaxpr(
          cond_jaxpr, cond_consts.val + cond_tracer_consts, (), loop_carry)
    f = batching.batch_transform(lifted, size, (init_val_bd, cond_tracer_consts_bd), 0)
    preds = f.call_wrapped((batched_loop_carry, cond_tracer_consts))
    return lax.reduce(preds, onp.array(False), lax.bitwise_or, [0])

  def batched_body_fun(batched_loop_carry):
    @lu.wrap_init
    def lifted(loop_carry, cond_tracer_consts, body_tracer_consts):
      cond_tracer_consts = tuple(x for x in cond_tracer_consts)
      body_tracer_consts = tuple(x for x in body_tracer_consts)
      pred = core.eval_jaxpr(
          cond_jaxpr, cond_consts.val + cond_tracer_consts, (), loop_carry)
      new_loop_carry = core.eval_jaxpr(
          body_jaxpr, body_consts.val + body_tracer_consts, (), loop_carry)
      return _jaxtupletree_select(pred, new_loop_carry, loop_carry)
    f = batching.batch_transform(
        lifted, size, (init_val_bd, cond_tracer_consts_bd, body_tracer_consts_bd),
        init_val_bd)
    return f.call_wrapped((batched_loop_carry, cond_tracer_consts, body_tracer_consts))

  return while_loop(batched_cond_fun, batched_body_fun, init_val), init_val_bd


def _jaxtupletree_select(pred, on_true, on_false):
  aval = core.get_aval(on_true)
  if type(aval) is core.AbstractTuple:
    return core.pack(map(partial(_jaxtupletree_select, pred), on_true, on_false))
  elif isinstance(aval, UnshapedArray):
    return lax.select(pred, on_true, on_false)
  else:
    raise TypeError(aval)


while_p = lax.Primitive('while')
while_p.def_impl(partial(xla.apply_primitive, while_p))
while_p.def_abstract_eval(_while_loop_abstract_eval)
xla.translations[while_p] = _while_loop_translation_rule
batching.primitive_batchers[while_p] = _while_loop_batching_rule


def cond(pred, true_operand, true_fun, false_operand, false_fun):
  def trace_jaxpr(fun, operand):
    op_flat, in_tree = pytree_to_flatjaxtuple(operand)
    fun_flat, out_tree = pytree_fun_to_flatjaxtuple_fun(lu.wrap_init(fun), (in_tree,))
    jaxpr, pvout, consts = pe.trace_to_jaxpr(fun_flat, (lax._abstractify(op_flat),))
    return op_flat, jaxpr, consts, pvout, out_tree

  true_data = trace_jaxpr(true_fun, true_operand)
  true_op, true_jaxpr, true_consts, true_pval, true_tree = true_data
  false_data = trace_jaxpr(false_fun, false_operand)
  false_op, false_jaxpr, false_consts, false_pval, false_tree = false_data

  if true_tree() != false_tree():
    msg = "true_fun and false_fun outputs must have identical structure"
    raise TypeError(msg)

  try:
    joined_pval = pe.join_pvals(true_pval, false_pval)
  except TypeError:
    msg = "could not merge true_fun and false_fun output pvals: {} and {}."
    raise TypeError(msg.format(true_pval, false_pval))
  revis = _revise_cond_jaxpr(joined_pval, true_pval, true_jaxpr, true_consts)
  true_jaxpr, true_consts = revis
  revis = _revise_cond_jaxpr(joined_pval, false_pval, false_jaxpr, false_consts)
  false_jaxpr, false_consts = revis
  aval_out, _ = joined_pval

  out = cond_p.bind(pred, true_op, core.pack(true_consts), false_op,
                    core.pack(false_consts), aval_out=aval_out,
                    true_jaxpr=true_jaxpr, false_jaxpr=false_jaxpr)
  out = pe.merge_pvals(out, joined_pval)
  return tree_unflatten(true_tree(), out)

def _revise_cond_jaxpr(new_pval, old_pval, jaxpr, consts):
  new_pv, new_const = new_pval
  old_pv, old_const = old_pval
  if new_pv == old_pv:
    # we didn't move up the lattice by joining with the other side
    return jaxpr, consts
  elif old_pv is None:
    # we moved up the lattice from totally-known, so make a new jaxpr that
    # returns a single constant JaxTuple with elements that are constants
    # drawn from consts where new_pv is unknown
    assert not jaxpr.eqns and not consts
    outvar = pe.Var(0, "_cond")
    new_jaxpr = jaxpr.copy()
    new_jaxpr.constvars = [outvar]
    new_jaxpr.outvar = outvar
    new_consts = (core.pack([core.unit if pv is None else old_c
                             for pv, old_c in zip(new_pv, old_const)]),)
    return new_jaxpr, new_consts
  else:
    # we moved up the lattice, but not from totally-constant, so adapt the
    # japxr to return some new constants in places that are now unknown but
    # weren't before
    eqn = jaxpr.eqns[-1]
    assert eqn.primitive == core.pack_p
    assert len(eqn.outvars) == 1 and eqn.outvars[0] == jaxpr.outvar
    newvar = pe.gensym("_cond")
    new_constvars, new_constvals = unzip2(
        [(newvar(), c) for new, old, c in zip(new_pv, old_pv, old_const)
         if old is None and new is not None])
    new_consts = consts + tuple(new_constvals)
    new_jaxpr = jaxpr.copy()
    new_jaxpr.constvars = tuple(jaxpr.constvars) + tuple(new_constvars)
    newvars = iter(new_constvars)
    new_invars = [next(newvars) if old is None and new is not None else v
                  for new, old, v in zip(new_pv, old_pv, eqn.invars)]
    new_jaxpr.eqns = (list(jaxpr.eqns[:-1]) +
                      [_pack_eqn(new_invars, jaxpr.outvar)])
    return new_jaxpr, new_consts

def _unpack_eqn(invar, outvars):
  return core.JaxprEqn([invar], outvars, core.identity_p, (), True, {})

def _pack_eqn(invars, outvar):
  return core.JaxprEqn(invars, [outvar], core.pack_p, (), False, {})


def _cond_abstract_eval(pred, true_op, true_consts, false_op, false_consts,
                        aval_out, true_jaxpr, false_jaxpr):
  if not isinstance(pred, ShapedArray) or pred.shape or pred.dtype != onp.bool_:
    msg = "cond pred must be a scalar boolean type, got {}."
    raise TypeError(msg.format(pred))
  if isinstance(pred, ConcreteArray):
    return true_op if pred else false_op
  else:
    return maybe_tracer_tuple_to_abstract_tuple(aval_out)


def _cond_translation_rule(c, pred, true_op, true_consts, false_op,
                           false_consts, aval_out, true_jaxpr, false_jaxpr):
  def make_computation(jaxpr, operand):
    assert len(jaxpr.invars) == 1
    arg_var = pe.Var(0, "arg")
    consts_var = pe.Var(0, "consts")
    jaxpr_converted = jaxpr.copy()
    jaxpr_converted.constvars = []
    jaxpr_converted.invars = [arg_var]
    jaxpr_converted.eqns = (
        [_unpack_eqn(arg_var, [jaxpr.invars[0], consts_var]),
        _unpack_eqn(consts_var, jaxpr.constvars)]
        + list(jaxpr.eqns))
    return xla.jaxpr_computation(jaxpr_converted, (), (), c.GetShape(operand))

  true_arg = c.Tuple(true_op, true_consts)
  true_comp = make_computation(true_jaxpr, true_arg)

  false_arg = c.Tuple(false_op, false_consts)
  false_comp = make_computation(false_jaxpr, false_arg)

  return c.Conditional(pred, true_arg, true_comp, false_arg, false_comp)

cond_p = lax.Primitive('cond')
cond_p.def_impl(partial(xla.apply_primitive, cond_p))
cond_p.def_abstract_eval(_cond_abstract_eval)
xla.translations[cond_p] = _cond_translation_rule


def fori_loop(lower, upper, body_fun, init_val):
  """Loop from `lower` to `upper` by reduction to `while_loop`.

  Arguments:
    lower: loop index lower bound (inclusive)
    upper: loop index upper bound (exclusive)
    body_fun: function of type (int, T) -> T, where T is the type of `init_val`
    init_val: initial loop value, of type T

  Returns:
    Loop value from the final iteration, of type T.

  The semantics of `fori_loop` are given by this Python implementation::

    def fori_loop(lower, upper, body_fun, init_val):
      val = init_val
      for i in range(lower, upper):
        val = body_fun(i, val)
      return val

  Unlike that pure Python version, `fori_loop` is implemented in terms of a call
  to `while_loop`. See the docstring for `while_loop` for more information.
  """
  def while_cond_fun(loop_carry):
    i, _ = loop_carry
    return lax.lt(i, upper)

  def while_body_fun(loop_carry):
    i, x = loop_carry
    return lax.add(i, lax._const(i, 1)), body_fun(i, x)

  _, result = while_loop(while_cond_fun, while_body_fun, (lower, init_val))
  return result


def maybe_tracer_tuple_to_abstract_tuple(tup):
  if isinstance(tup, pe.JaxprTracerTuple):
    return core.AbstractTuple(list(map(maybe_tracer_tuple_to_abstract_tuple, tup)))
  elif isinstance(tup, core.AbstractValue):
    return tup
  elif tup is None:
    return core.AbstractTuple(())  # TODO(dougalm): check this
  else:
    raise TypeError(tup)
