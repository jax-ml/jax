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
from jax.lax import _abstractify
from jax import linear_util as lu
from jax.abstract_arrays import ConcreteArray, ShapedArray, UnshapedArray
from jax.api_util import (
    pytree_to_flatjaxtuple, pytree_fun_to_flatjaxtuple_fun,
    pytree_to_jaxtupletree, pytree_fun_to_jaxtupletree_fun)
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import ad
from jax.util import partial, unzip2, safe_map, safe_zip
from jax.tree_util import build_tree, tree_unflatten
from jax import ad_util

map = safe_map
zip = safe_zip


### fori_loop and while_loop

def fori_loop(lower, upper, body_fun, init_val):
  """Loop from ``lower`` to ``upper`` by reduction to ``while_loop``.

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
  ``while_loop``. See the docstring for ``while_loop`` for more information.

  Args:
    lower: an integer representing the loop index lower bound (inclusive)
    upper: an integer representing the loop index upper bound (exclusive)
    body_fun: function of type ``(int, a) -> a``.
    init_val: initial loop carry value of type ``a``.

  Returns:
    Loop value from the final iteration, of type ``a``.
  """
  def while_cond_fun(loop_carry):
    i, _ = loop_carry
    return lax.lt(i, upper)

  def while_body_fun(loop_carry):
    i, x = loop_carry
    return lax.add(i, lax._const(i, 1)), body_fun(i, x)

  _, result = while_loop(while_cond_fun, while_body_fun, (lower, init_val))
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
  init_val_flat, in_tree = pytree_to_jaxtupletree(init_val)
  flat_body_fun, out_tree = pytree_fun_to_jaxtupletree_fun(lu.wrap_init(body_fun), (in_tree,))
  flat_cond_fun, _ = pytree_fun_to_jaxtupletree_fun(lu.wrap_init(cond_fun), (in_tree,))

  carry_pval_flat = carry_aval, _ = _abstractify(init_val_flat)
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
  return _maybe_tracer_tuple_to_abstract_tuple(aval_out)

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


### cond

def cond(pred, true_operand, true_fun, false_operand, false_fun):
  def trace_jaxpr(fun, operand):
    op_flat, in_tree = pytree_to_flatjaxtuple(operand)
    fun_flat, out_tree = pytree_fun_to_flatjaxtuple_fun(lu.wrap_init(fun), (in_tree,))
    jaxpr, pvout, consts = pe.trace_to_jaxpr(fun_flat, (_abstractify(op_flat),))
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
    new_invars = [next(newvars) if old is None and new is not None else
                  (core.unitvar if new is None and old is None else v)
                  for new, old, v in zip(new_pv, old_pv, eqn.invars)]
    new_jaxpr.eqns = (list(jaxpr.eqns[:-1]) +
                      [_pack_eqn(new_invars, jaxpr.outvar)])
    return new_jaxpr, new_consts

def _unpack_eqn(invar, outvars):
  return core.JaxprEqn([invar], outvars, core.identity_p, (), False, True, {})

def _pack_eqn(invars, outvar):
  return core.JaxprEqn(invars, [outvar], core.pack_p, (), False, False, {})


def _cond_abstract_eval(pred, true_op, true_consts, false_op, false_consts,
                        aval_out, true_jaxpr, false_jaxpr):
  if not isinstance(pred, ShapedArray) or pred.shape or pred.dtype != onp.bool_:
    msg = "cond pred must be a scalar boolean type, got {}."
    raise TypeError(msg.format(pred))
  if isinstance(pred, ConcreteArray):
    return true_op if pred else false_op
  else:
    return _maybe_tracer_tuple_to_abstract_tuple(aval_out)


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


def _maybe_tracer_tuple_to_abstract_tuple(tup):
  if isinstance(tup, pe.JaxprTracerTuple):
    return core.AbstractTuple(list(map(_maybe_tracer_tuple_to_abstract_tuple, tup)))
  elif isinstance(tup, core.AbstractValue):
    return tup
  elif tup is None:
    return core.AbstractTuple(())
  else:
    raise TypeError(tup)


### scan

def _convert_zeros(convert_symbolic, example, tangent):
  if tangent is ad.zero:
    if not convert_symbolic:
      return core.unit
    else:
      return ad.zeros_like_jaxval(example)
  elif type(tangent) is ad.TangentTuple:
    return core.pack(map(_convert_zeros, convert_symbolic, example, tangent))
  else:
    return tangent

def _demote_aval_rank(xs):
  assert isinstance(xs, core.AbstractValue)
  if isinstance(xs, core.AbstractTuple):
    return core.AbstractTuple(map(_demote_aval_rank, xs))
  else:
    return ShapedArray(xs.shape[1:], xs.dtype)

def _promote_aval_rank(n, xs):
  assert isinstance(xs, core.AbstractValue)
  if isinstance(xs, core.AbstractTuple):
    return core.AbstractTuple(map(partial(_promote_aval_rank, n), xs))
  else:
    return ShapedArray((n,) + xs.shape, xs.dtype)

def _leading_dim_size(xs):
  if isinstance(xs, core.JaxTuple):
    return _leading_dim_size(xs[0])
  else:
    return xs.shape[0]

def _empty_arrays(aval):
  assert isinstance(aval, core.AbstractValue)
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(_empty_arrays, aval))
  else:
    return lax.full(aval.shape, 0, aval.dtype)

def _index_arrays(i, aval, xs):
  assert isinstance(aval, core.AbstractValue)
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(partial(_index_arrays, i), aval, xs))
  else:
    return lax.dynamic_index_in_dim(xs, i, keepdims=False)

def _update_arrays(i, aval, xs, x):
  assert isinstance(aval, core.AbstractValue)
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(partial(_update_arrays, i), aval, xs, x))
  else:
    return lax.dynamic_update_index_in_dim(xs, x[None, ...], i, axis=0)


def scan(f, init, xs):
  """Scan a function over leading array axes while carrying along state.

  The type signature in brief is

  .. code-block:: haskell

    scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])

  where we use [t] here to denote the type t with an additional leading axis.
  That is, if t is an array type then [t] represents the type with an additional
  leading axis, and if t is a pytree (container) type with array leaves then [t]
  represents the type with the same pytree structure and corresponding leaves
  each with an additional leading axis.

  When both ``a`` and ``b`` are array types, the semantics of ``scan`` are given
  by this Python implementation::

    def scan(f, init, xs):
      carry = init
      ys = []
      for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
      return carry, np.stack(ys)

  Unlike that Python version, both ``a`` and ``b`` may be arbitrary pytree
  types, and so multiple arrays can be scanned over at once and produce multiple
  output arrays.

  Also unlike that Python version, ``scan`` is a JAX primitive and is lowered to
  a single XLA While HLO. That makes it useful for reducing compilation times
  for jit-compiled functions, since native Python loop constructs in an ``@jit``
  function are unrolled, leading to large XLA computations.

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output.
    init: an initial loop carry value of type ``c``, which can be a scalar,
      array, or any pytree (nested Python tuple/list/dict) thereof, representing
      the initial loop carry value.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.

  Returns:
    A pair of type ``(c, [b])`` where the first element represents the final
    loop carry value and the second element represents the stacked outputs of
    the second output of ``f`` when scanned over the leading axis of the inputs.
  """
  (init, xs), in_trees = unzip2(map(pytree_to_jaxtupletree, (init, xs)))
  f, out_tree = pytree_fun_to_jaxtupletree_fun(lu.wrap_init(f), in_trees)
  carry_pval = carry_aval, _ = _abstractify(init)
  xs_aval, _ = _abstractify(xs)
  x_aval = _demote_aval_rank(xs_aval)
  x_pval = pe.PartialVal((x_aval, core.unit))
  jaxpr, pval_out, consts = pe.trace_to_jaxpr(
      f, (carry_pval, x_pval), instantiate=True)
  pv_out, const_out = pval_out
  assert isinstance(pv_out, core.AbstractTuple) and const_out == core.unit
  carry_aval_out, y_aval = pv_out
  if carry_aval != carry_aval_out:
    msg = ("scanned function carry output does not match carry input: "
           "input carry is {} and output carry is {}")
    raise TypeError(msg.format(carry_aval, carry_aval_out))
  lifted_jaxpr = pe._closure_convert_jaxpr(jaxpr)
  consts_aval, _ = _abstractify(core.pack(consts))
  in_avals = (consts_aval, carry_aval, x_aval)
  out_aval = core.AbstractTuple((carry_aval, y_aval))
  jaxpr = core.TypedJaxpr(lifted_jaxpr, (), in_avals, out_aval)
  length = _leading_dim_size(xs)
  out = scan_p.bind(core.pack(consts), init, xs,
                    forward=True, length=length, jaxpr=jaxpr)
  return build_tree(out_tree(), out)


def _scan_impl(consts, init, xs, forward, length, jaxpr):
  _, _, x_aval = jaxpr.in_avals
  _, y_aval = jaxpr.out_aval
  ys_aval = _promote_aval_rank(length, y_aval)

  def body_fun(i, vals):
    idx = i if forward else length - i - 1
    carry, ys = vals
    x = _index_arrays(idx, x_aval, xs)
    carry_out, y = core.jaxpr_as_fun(jaxpr)(consts, carry, x)
    ys_out = _update_arrays(idx, y_aval, ys, y)
    return (carry_out, ys_out)

  ys_init = _empty_arrays(ys_aval)
  carry, ys = fori_loop(0, length, body_fun, (init, ys_init))
  return core.pack((carry, ys))


def _scan_jvp(primals, tangents, forward, length, jaxpr):
  consts, init, xs = primals
  consts_dot, init_dot, xs_dot = tangents
  consts_aval, carry_aval, x_aval = jaxpr.in_avals
  _, y_aval = jaxpr.out_aval

  consts_nonzeros = ad.get_nonzeros(consts_dot)
  init_nonzeros = ad.get_nonzeros(init_dot)
  xs_nonzeros = ad.get_nonzeros(xs_dot)  # same as x_nonzeros b/c arrays

  carry_nonzeros = init_nonzeros
  for _ in range(1000):
    nonzeros = (consts_nonzeros, carry_nonzeros, xs_nonzeros)
    jaxpr_jvp, nonzeros_out = ad.jvp_jaxpr(jaxpr, nonzeros,
                                           instantiate=(carry_nonzeros, False))
    carry_nonzeros_out, ys_nonzeros = nonzeros_out
    if carry_nonzeros_out == carry_nonzeros:
      break
    else:
      carry_nonzeros = _binary_lattice_join(carry_nonzeros_out, carry_nonzeros)
  else:
    raise FixedPointError

  # convert_zeros is like strip_zeros but uses explicit lattice information to
  # instantiate zeros in some cases, namely in init_dot based on the fixed point
  nonzero_init_dot = _convert_zeros(carry_nonzeros, init, init_dot)
  nonzero_consts_dot = _convert_zeros(consts_nonzeros, consts, consts_dot)
  nonzero_xs_dot = _convert_zeros(xs_nonzeros, xs, xs_dot)

  consts_dual = core.pack((consts, nonzero_consts_dot))
  init_dual = core.pack((init, nonzero_init_dot))
  xs_dual = core.pack((xs, nonzero_xs_dot))

  carry_out_dual, ys_dual = scan_p.bind(
      consts_dual, init_dual, xs_dual,
      forward=forward, length=length, jaxpr=jaxpr_jvp)

  ys, ys_dot = ys_dual
  ys_dot = ad.put_zeros(ad.TangentTuple, ys_nonzeros, ys_dot)

  carry_out, carry_out_dot = carry_out_dual
  carry_out_dot = ad.put_zeros(ad.TangentTuple, carry_nonzeros_out, carry_out_dot)
  return core.pack((carry_out, ys)), ad.TangentTuple((carry_out_dot, ys_dot))

def _binary_lattice_join(a, b):
  t = (type(a), type(b))
  if t == (tuple, tuple):
    return tuple(map(_binary_lattice_join, a, b))
  elif t == (tuple, bool):
    return tuple(map(_binary_lattice_join, a, (b,) * len(a)))
  elif t == (bool, tuple):
    return tuple(map(_binary_lattice_join, (a,) * len(b), b))
  elif t == (bool, bool):
    return a or b
  else:
    raise TypeError((type(a), type(b)))


def _scan_partial_eval(trace, *tracers, **kwargs):
  jaxpr = kwargs.pop('jaxpr')
  length = kwargs.pop('length')
  forward = kwargs.pop('forward')
  assert not kwargs
  in_pvs, in_consts = unzip2([t.pval for t in tracers])
  sc_consts, sc_init, sc_xs = map(pe.unknown, in_pvs)

  sc_carry = sc_init
  for i in range(1000):
    second_components = (sc_consts, sc_carry, sc_xs)
    jaxpr_1, jaxpr_2, sc_out = pe.partial_eval_jaxpr(jaxpr, second_components,
                                                     instantiate=(sc_carry, False))
    sc_carry_out, sc_ys = sc_out
    if sc_carry_out == sc_carry:
      break
    else:
      sc_carry = _binary_lattice_join(sc_carry, sc_carry_out)
  else:
    raise FixedPointError

  consts_tracer, init_tracer, xs_tracer = tracers
  lifted_init_tracer = _lift_tracer(trace, init_tracer, sc_carry)
  lifted_tracers = consts_tracer, lifted_init_tracer, xs_tracer
  in_pvs, in_consts = unzip2([t.pval for t in lifted_tracers])

  carry_aval, y_aval = jaxpr.out_aval
  ys_aval = _promote_aval_rank(length, y_aval)
  out_aval = core.AbstractTuple((carry_aval, ys_aval))
  out_pv = _put_known_pvs(sc_out, out_aval)

  out_carry, (ys, residuals) = scan_p.bind(
      *in_consts, forward=forward, length=length, jaxpr=jaxpr_1)
  out_const = core.pack((out_carry, ys))
  residuals_tracer = trace.new_instantiated_const(core.pack(residuals))
  d, c, a = lifted_tracers
  new_tracers = (d, c, (a, residuals_tracer))
  eqn = core.JaxprEqn(new_tracers, None, scan_p, (), True, False,
                      dict(forward=forward, length=length, jaxpr=jaxpr_2))
  return pe.JaxprTracer(trace, pe.PartialVal((out_pv, out_const)), eqn)

def _lift_tracer(trace, tracer, is_unknown):
  t = type(is_unknown)
  if t is bool:
    if is_unknown:
      return trace.instantiate_const(tracer)
    else:
      return tracer
  elif t is tuple:
    tracers = map(trace.full_raise, tracer)
    return core.pack(map(partial(_lift_tracer, trace), tracers, is_unknown))
  else:
    raise TypeError(t)

def _put_known_pvs(is_unknown, aval):
  if is_unknown is False:
    return None
  elif is_unknown is True:
    return aval
  else:
    return pe.JaxprTracerTuple(map(_put_known_pvs, is_unknown, aval))


def _scan_transpose(ct, consts, init, xs, forward, length, jaxpr):
  assert consts is None and init is None
  assert type(xs) is tuple
  a, res = xs
  assert a is None and res is not None

  # jaxpr :: d -> c -> (a, res) ->  (c, b)
  # jaxpr_lifted :: res -> (d, c, a) -> (c, b)
  # jaxpr_lifted_trans :: res -> (CT c, CT b) -> (CT d, CT c, CT a)
  # jaxpr_trans :: * -> (CT c, CT d) -> (CT b, res) -> ((CT c, CT d), CT a)
  assert type(jaxpr.jaxpr.invars[2]) is tuple  # assume restructuring
  jaxpr_lifted = rearrange_binders(
      lambda d, c, a_res: (a_res[1], (d, c, a_res[0])), jaxpr)
  jaxpr_lifted_trans = _transpose_jaxpr(jaxpr_lifted)
  jaxpr_trans = _move_stuff_and_add_add(jaxpr_lifted_trans)

  c_aval, b_aval = jaxpr.out_aval
  d_aval, c_aval2, _ = jaxpr.in_avals
  assert c_aval == c_aval2
  bs_aval = _promote_aval_rank(length, b_aval)
  ct_d = ad_util.zeros_like_aval(d_aval)
  ct_c, ct_bs = ad.instantiate_zeros_aval(core.AbstractTuple((c_aval, bs_aval)), ct)
  carry_ct = core.pack((ct_c, ct_d))

  # jaxpr_trans :: * -> (CT c, CT d) -> (CT b, res) -> ((CT c, CT d), CT a)
  core.check_jaxpr(jaxpr_trans.jaxpr)
  unit_aval, (ct_c_aval, ct_d_aval), (ct_b_aval, _) = jaxpr_trans.in_avals
  assert core.lattice_join(ct_c_aval, core.get_aval(ct_c)) == ct_c_aval
  assert core.lattice_join(ct_d_aval, core.get_aval(ct_d)) == ct_d_aval

  out = scan_p.bind(
      core.unit, carry_ct, core.pack((ct_bs, res)),
      forward=not forward, length=length, jaxpr=jaxpr_trans)
  (ct_init, ct_consts), ct_as = out
  return ct_consts, ct_init, (ct_as, None)

def rearrange_binders(f, typed_jaxpr):
  jaxpr = typed_jaxpr.jaxpr.copy()
  jaxpr.invars = f(*jaxpr.invars)
  in_avals = f(*typed_jaxpr.in_avals)
  core.skip_checks or core.check_jaxpr(jaxpr)
  return core.TypedJaxpr(jaxpr, typed_jaxpr.literals, in_avals,
                         typed_jaxpr.out_aval)

_scan_newvar = pe.gensym('_scan')

def _move_stuff_and_add_add(typed_jaxpr):
  # jaxpr_lifted_trans :: res -> (CT c, CT b) -> (CT d, CT c, CT a)
  # jaxpr_trans :: * -> (CT c, CT d) -> (CT b, res) -> ((CT c, CT d), CT a)

  res_aval, (CTc_aval, CTb_aval) = typed_jaxpr.in_avals
  CTd_aval, CTc_aval2, CTa_aval = typed_jaxpr.out_aval
  assert CTc_aval == CTc_aval2
  in_avals = (core.AbstractTuple(()), core.AbstractTuple((CTc_aval, CTd_aval)),
              core.AbstractTuple((CTb_aval, res_aval)))
  out_aval = core.AbstractTuple((core.AbstractTuple((CTc_aval, CTd_aval)),
                                 CTa_aval))

  jaxpr = typed_jaxpr.jaxpr.copy()
  # assume the jaxpr isn't restructuring any inputs
  assert not any(type(invar) is tuple for invar in jaxpr.invars)

  # munge input side
  CTc_in = _scan_newvar()
  CTb_in = _scan_newvar()
  CTd_in = _scan_newvar()
  res_in, CTc_CTb_in = jaxpr.invars
  jaxpr.invars = ((), (CTc_in, CTd_in), (CTb_in, res_in))
  jaxpr.eqns = (
      [pe._pack_eqn([CTc_in, CTb_in], CTc_CTb_in)] +
      jaxpr.eqns)

  # munge output side
  CTd_new = _scan_newvar()
  CTd_sum = _scan_newvar()
  CTc = _scan_newvar()
  CTa = _scan_newvar()
  partial_out = _scan_newvar()
  outvar = _scan_newvar()
  jaxpr.eqns = (
      jaxpr.eqns +
      [pe._unpack_eqn(jaxpr.outvar, [CTd_new, CTc, CTa]),
       _add_any_eqn(CTd_sum, CTd_new, CTd_in),
       pe._pack_eqn([CTc, CTd_sum], partial_out),
       pe._pack_eqn([partial_out, CTa], outvar)])
  jaxpr.outvar = outvar

  # TODO(mattjj): add a check_typed_jaxpr and use it here
  core.skip_checks or core.check_jaxpr(jaxpr)
  return core.TypedJaxpr(jaxpr, typed_jaxpr.literals, in_avals, out_aval)

def _add_any_eqn(tot, a, b):
  return core.JaxprEqn([a, b], [tot], ad_util.add_jaxvals_p, (), False, False, {})


# transpose_jaxpr :: (res -> a -> b) -> (res -> CT b -> CT a)
def _transpose_jaxpr(jaxpr):
  assert len(jaxpr.in_avals) == 2

  @lu.wrap_init
  def transposed(res, b_bar):
    _, (_, a_bar) = ad.backward_pass(jaxpr.jaxpr, jaxpr.literals, (),
                                     (res, None), b_bar)
    a_bar = ad.instantiate_zeros_aval(jaxpr.in_avals[1], a_bar)
    return a_bar

  transposed_jaxpr = _make_typed_jaxpr(transposed, (jaxpr.in_avals[0], jaxpr.out_aval))
  return transposed_jaxpr

def _make_typed_jaxpr(traceable, in_avals):
  pvals = [pe.PartialVal((aval, core.unit)) for aval in in_avals]
  jaxpr, pval_out, consts = pe.trace_to_jaxpr(traceable, pvals, instantiate=True)
  out_aval, _ = pval_out
  assert isinstance(out_aval, core.AbstractValue)
  return core.TypedJaxpr(jaxpr, consts, in_avals, out_aval)


class FixedPointError(Exception): pass


scan_p = core.Primitive("scan")
scan_p.def_impl(_scan_impl)
ad.primitive_jvps[scan_p] = _scan_jvp
ad.primitive_transposes[scan_p] = _scan_transpose
pe.custom_partial_eval_rules[scan_p] = _scan_partial_eval
xla.translations[scan_p] = partial(xla.lower_fun, _scan_impl)
