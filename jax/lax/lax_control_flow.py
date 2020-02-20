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
import itertools
import operator
import threading

import numpy as onp

from jax import api
from jax import core
from jax import dtypes
from jax.lax import lax
from jax import linear_util as lu
from jax.abstract_arrays import ShapedArray, raise_to_shaped
from jax.api_util import flatten_fun_nokwargs, apply_flat_fun_nokwargs
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import batching
from jax.interpreters import masking
from jax.lib import xla_bridge as xb
from jax.lib import xla_client
from jax.util import (partial, unzip2, safe_map, safe_zip, split_list,
                      split_dict, cache, extend_name_stack)
from jax.tree_util import (tree_flatten, tree_unflatten, treedef_is_leaf,
                           treedef_children, treedef_tuple)
from jax import ad_util

_map = safe_map
zip = safe_zip
_reduce = functools.reduce


@cache()
def _initial_style_jaxpr(fun, in_tree, in_avals):
  in_pvals = [pe.PartialVal((aval, core.unit)) for aval in in_avals]
  fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr(fun, in_pvals, instantiate=True,
                                               stage_out_calls=True)
  out_avals = _map(raise_to_shaped, unzip2(out_pvals)[0])
  const_avals = tuple(raise_to_shaped(core.get_aval(c)) for c in consts)
  typed_jaxpr = core.TypedJaxpr(pe.convert_constvars_jaxpr(jaxpr),
                                (), const_avals + in_avals, out_avals)
  return typed_jaxpr, consts, out_tree()

def _abstractify(x):
  return raise_to_shaped(core.get_aval(x))

def typecheck(aval, x):
  aval = raise_to_shaped(aval).strip_weak_type()
  try:
    return aval == core.lattice_join(aval, core.get_aval(x)).strip_weak_type()
  except TypeError:
    return False

def typematch(aval1, aval2):
  return (raise_to_shaped(aval1).strip_weak_type() ==
          raise_to_shaped(aval2).strip_weak_type())

class FixedPointError(Exception): pass


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
  # TODO: perhaps do more type checking here, for better error messages.
  lower_dtype = dtypes.canonicalize_dtype(lax.dtype(lower))
  upper_dtype = dtypes.canonicalize_dtype(lax.dtype(upper))
  if lower_dtype != upper_dtype:
    msg = ("lower and upper arguments to fori_loop must have equal types, "
           "got {} and {}")
    raise TypeError(msg.format(lower_dtype.name, upper_dtype.name))
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

def _while_loop_translation_rule(c, axis_env, name_stack, *args, **kwargs):
  backend = kwargs.pop('backend')
  cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts = split_dict(
      kwargs, ["cond_jaxpr", "body_jaxpr", "cond_nconsts", "body_nconsts"])
  cond_consts, body_consts, init_vals = split_list(args, [cond_nconsts, body_nconsts])
  batched = bool(cond_jaxpr.out_avals[0].shape)

  # Since jaxprs don't have tuples and have multiple return values, but we need
  # the HLO While loop to take a single tuple input and output a single boolean
  # (for the cond computation) or a single tuple output (for the body
  # computation), we build XLA computations that handle the tuple munging before
  # generating a Call into the computations formed from the jaxprs.

  init_carry = c.Tuple(*(cond_consts + body_consts + init_vals))

  cond_c = xb.make_computation_builder("cond_computation")
  cond_carry = cond_c.ParameterWithShape(c.GetShape(init_carry))
  cond_carry_elts = [cond_c.GetTupleElement(cond_carry, i) for i in range(len(args))]
  x, _, z = split_list(cond_carry_elts, [cond_nconsts, body_nconsts])
  pred, = xla.jaxpr_subcomp(cond_c, cond_jaxpr.jaxpr, backend, axis_env,
                            _map(cond_c.Constant, cond_jaxpr.literals),
                            extend_name_stack(name_stack, 'cond'), *(x + z))
  if batched:
    scalar = ShapedArray((), onp.bool_)
    or_ = xla.primitive_subcomputation(lax.or_p, scalar, scalar)
    pred = cond_c.Reduce(pred, cond_c.Constant(onp.array(False)), or_,
                         list(range(cond_jaxpr.out_avals[0].ndim)))

  body_c = xb.make_computation_builder("body_computation")
  body_carry = body_c.ParameterWithShape(c.GetShape(init_carry))
  body_carry_elts = [body_c.GetTupleElement(body_carry, i) for i in range(len(args))]
  x, y, z = split_list(body_carry_elts, [cond_nconsts, body_nconsts])
  new_z = xla.jaxpr_subcomp(body_c, body_jaxpr.jaxpr, backend, axis_env,
                            _map(body_c.Constant, body_jaxpr.literals),
                            extend_name_stack(name_stack, 'body'), *(y + z))
  if batched:
    body_pred, = xla.jaxpr_subcomp(body_c, cond_jaxpr.jaxpr, backend, axis_env,
                                   _map(body_c.Constant, cond_jaxpr.literals),
                                   extend_name_stack(name_stack, 'body_pred'), *(x + z))
    new_z = _map(partial(_pred_bcast_select, body_c, body_pred), new_z, z)
    assert _map(body_c.GetShape, new_z) == _map(body_c.GetShape, z) # no broadcast
  new_carry = body_c.Tuple(*itertools.chain(x, y, new_z))

  ans = c.While(cond_c.Build(pred), body_c.Build(new_carry), init_carry)
  ans_elts = [c.GetTupleElement(ans, i) for i in range(len(args))]
  _,  _, z = split_list(ans_elts, [cond_nconsts, body_nconsts])
  return c.Tuple(*z)

def _pred_bcast_select(c, pred, x, y):
  pred_shape = c.GetShape(pred).dimensions()
  x_shape = c.GetShape(x).dimensions()
  y_shape = c.GetShape(y).dimensions()
  assert x_shape == y_shape
  assert pred_shape == x_shape[:len(pred_shape)] == y_shape[:len(pred_shape)]
  bcast_pred = c.BroadcastInDim(pred, x_shape, list(range(len(pred_shape))))
  return c.Select(bcast_pred, x, y)

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
        cond_jaxpr, size, cconst_bat + carry_bat, instantiate=False)
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
              else batching.moveaxis(x, d, 0) if now_bat else x
              for x, d, was_bat, now_bat in zip(init, init_dims, init_bat, carry_bat)]

  outs = while_p.bind(*(new_consts + new_init),
                      cond_nconsts=cond_nconsts, cond_jaxpr=cond_jaxpr_batched,
                      body_nconsts=body_nconsts, body_jaxpr=body_jaxpr_batched)
  out_bdims = [0 if b else batching.not_mapped for b in carry_bat]
  return outs, out_bdims

def _while_loop_jvp(primals, tangents, cond_nconsts, cond_jaxpr, body_nconsts,
                    body_jaxpr):
  nonzeros = [t is not ad_util.zero for t in tangents]
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
  tangents = [ad.instantiate_zeros(x, t) if t is ad_util.zero and nz else t
              for x, t, nz in zip(primals, tangents, nonzeros)]

  cconst, bconst, init = split_list(primals, [cond_nconsts, body_nconsts])
  _, bconst_dot, init_dot = split_list(tangents, [cond_nconsts, body_nconsts])
  bconst_dot = _prune_zeros(bconst_dot)
  init_dot = _prune_zeros(init_dot)

  num_carry = len(primals) - cond_nconsts - body_nconsts

  body_jvp_rearranged = ad.rearrange_binders(
      body_jvp,
      [body_nconsts, num_carry], [len(bconst_dot), len(init_dot)],
      [num_carry], [len(init_dot)])

  newvar = core.gensym('')
  invars_aug = (
      cond_jaxpr.jaxpr.invars + [newvar() for _ in range(len(init_dot))])
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
  out_tangents = [next(out_tangents_iter) if nz else ad_util.zero
                  for nz in nonzeros_out]
  return out_carry, out_tangents

while_p = lax.Primitive('while')
while_p.multiple_results = True
while_p.def_impl(partial(xla.apply_primitive, while_p))
while_p.def_abstract_eval(_while_loop_abstract_eval)
ad.primitive_jvps[while_p] = _while_loop_jvp
xla.initial_style_translations[while_p] = _while_loop_translation_rule
batching.primitive_batchers[while_p] = _while_loop_batching_rule


### cond

def cond(pred, true_operand, true_fun, false_operand, false_fun):
  """Conditionally apply ``true_fun`` or ``false_fun``.

  Has equivalent semantics to this Python implementation::

    def cond(pred, true_operand, true_fun, false_operand, false_fun):
      if pred:
        return true_fun(true_operand)
      else:
        return false_fun(false_operand)

  Pred has to be a scalar type, collection types (list, tuple) are not supported

  """

  if len(onp.shape(pred)) != 0:
    raise TypeError("Pred must be a scalar, got {} of shape {}.".format(pred, onp.shape(pred)))

  try:
    pred_dtype = dtypes.result_type(pred)
  except TypeError:
    msg = ("Pred type must be either boolean or number, got {}.")
    raise TypeError(msg.format(pred))

  if pred_dtype.kind != 'b':
    if pred_dtype.kind in 'iuf':
      pred = pred != 0
    else:
      msg = ("Pred type must be either boolean or number, got {}.")
      raise TypeError(msg.format(pred_dtype))
  true_ops, true_tree = tree_flatten((true_operand,))
  true_avals = tuple(_map(_abstractify, true_ops))
  true_jaxpr, true_consts, true_out_tree = _initial_style_jaxpr(true_fun, true_tree, true_avals)
  false_ops, false_tree = tree_flatten((false_operand,))
  false_avals = tuple(_map(_abstractify, false_ops))
  false_jaxpr, false_consts, false_out_tree = _initial_style_jaxpr(false_fun, false_tree, false_avals)
  _check_tree_and_avals("true_fun and false_fun output",
                        true_out_tree, true_jaxpr.out_avals,
                        false_out_tree, false_jaxpr.out_avals)
  linear = (False,) * (len(true_consts) + len(true_ops) + len(false_consts) +
                       len(false_ops))
  out = cond_p.bind(
      *itertools.chain([pred], true_consts, true_ops, false_consts, false_ops),
      true_jaxpr=true_jaxpr, false_jaxpr=false_jaxpr, linear=linear)
  return tree_unflatten(true_out_tree, out)

def _cond_abstract_eval(*args, **kwargs):
  return _map(raise_to_shaped, kwargs["true_jaxpr"].out_avals)

def _cond_translation_rule(c, axis_env, name_stack, pred, *args,
                           true_jaxpr, false_jaxpr, linear, backend=None):
  del linear  # Unused.
  true_ops, false_ops = split_list(args, [len(true_jaxpr.in_avals)])

  def make_computation(name, jaxpr, op_shape):
    c = xb.make_computation_builder(name + '_comp')
    op = c.ParameterWithShape(op_shape)
    ops = [c.GetTupleElement(op, i) for i in range(len(jaxpr.in_avals))]
    outs = xla.jaxpr_subcomp(c, jaxpr.jaxpr, backend, axis_env,
                             _map(c.Constant, jaxpr.literals),
                             extend_name_stack(name_stack, name + '_fun'), *ops)
    return c.Build(c.Tuple(*outs))

  true_op = c.Tuple(*true_ops)
  true_c = make_computation('true', true_jaxpr, c.GetShape(true_op))

  false_op = c.Tuple(*false_ops)
  false_c = make_computation('false', false_jaxpr, c.GetShape(false_op))

  return c.Conditional(pred, true_op, true_c, false_op, false_c)

def _cond_pred_bcast_select(pred, x, y):
  if core.get_aval(x) is core.get_aval(y) is core.abstract_unit:
    return x
  else:
    bcast_pred = lax.broadcast_in_dim(pred, onp.shape(x), list(range(onp.ndim(pred))))
    return lax.select(bcast_pred, x, y)

def _cond_batching_rule(args, dims, true_jaxpr, false_jaxpr, linear):
  # TODO: maybe avoid moving arg axes to front if we're promoting to select?
  size, = {x.shape[d] for x, d in zip(args, dims) if d is not batching.not_mapped}
  args = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
          else x for x, d in zip(args, dims)]
  orig_bat = [d is not batching.not_mapped for d in dims]
  del dims
  (pred,), true_ops, false_ops = split_list(args, [1, len(true_jaxpr.in_avals)])
  (pred_bat,), t_bat, f_bat = split_list(orig_bat, [1, len(true_jaxpr.in_avals)])

  _, true_out_bat = batching.batch_jaxpr(true_jaxpr, size, t_bat, False)
  _, false_out_bat = batching.batch_jaxpr(false_jaxpr, size, f_bat, False)
  out_bat = [a or b for a, b in zip(true_out_bat, false_out_bat)]

  true_jaxpr_batched, _ = batching.batch_jaxpr(true_jaxpr, size, t_bat, out_bat)
  false_jaxpr_batched, _ = batching.batch_jaxpr(false_jaxpr, size, f_bat, out_bat)

  if pred_bat:
    true_out = core.jaxpr_as_fun(true_jaxpr_batched)(*true_ops)
    false_out = core.jaxpr_as_fun(false_jaxpr_batched)(*false_ops)
    true_out = [batching.broadcast(x, size, 0) if not b else x
                for x, b in zip(true_out, out_bat)]
    false_out = [batching.broadcast(x, size, 0) if not b else x
                 for x, b in zip(false_out, out_bat)]
    return [_cond_pred_bcast_select(pred, t, f)
            for t, f in zip(true_out, false_out)], [0] * len(true_out)
  else:
    out_dims = [0 if b else batching.not_mapped for b in out_bat]
    out = cond_p.bind(
      *itertools.chain([pred], true_ops, false_ops),
      true_jaxpr=true_jaxpr_batched, false_jaxpr=false_jaxpr_batched, linear=linear)
    return out, out_dims

def _cond_jvp(primals, tangents, true_jaxpr, false_jaxpr, linear):
  nonzeros = [t is not ad_util.zero for t in tangents]

  (pred_nz,), t_nz, f_nz = split_list(nonzeros, [1, len(true_jaxpr.in_avals)])
  assert pred_nz is False

  _, true_out_nz = ad.jvp_jaxpr(true_jaxpr, t_nz, instantiate=False)
  _, false_out_nz = ad.jvp_jaxpr(false_jaxpr, f_nz, instantiate=False)
  out_nz = [a or b for a, b in zip(true_out_nz, false_out_nz)]

  true_jvp, _ = ad.jvp_jaxpr(true_jaxpr, t_nz, instantiate=out_nz)
  false_jvp, _ = ad.jvp_jaxpr(false_jaxpr, f_nz, instantiate=out_nz)

  (pred,), tops, fops = split_list(primals, [1, len(true_jaxpr.in_avals)])
  _, tops_dot, fops_dot = split_list(tangents, [1, len(true_jaxpr.in_avals)])

  tops_dot = _prune_zeros(tops_dot)
  fops_dot = _prune_zeros(fops_dot)

  tops_lin, fops_lin = _map(tuple, split_list(linear, [len(tops)]))
  linear_jvp = (tops_lin + (True,) * len(tops_dot) +
                fops_lin + (True,) * len(fops_dot))
  out = cond_p.bind(
      *itertools.chain([pred], tops, tops_dot, fops, fops_dot),
      true_jaxpr=true_jvp, false_jaxpr=false_jvp, linear=linear_jvp)
  out_primals, out_tangents = split_list(out, [len(out_nz)])
  out_tangents_iter = iter(out_tangents)
  out_tangents = [
      next(out_tangents_iter) if nz else ad_util.zero for nz in out_nz]
  return out_primals, out_tangents

def _cond_partial_eval(trace, *tracers, true_jaxpr, false_jaxpr, linear):
  unknowns = [t.pval[0] is not None for t in tracers]

  (pred_uk,), t_uk, f_uk = split_list(unknowns, [1, len(true_jaxpr.in_avals)])

  if pred_uk:
    # When the predicate is unknown, we stage out the whole cond.
    params = dict(true_jaxpr=true_jaxpr, false_jaxpr=false_jaxpr, linear=linear)
    return trace.default_process_primitive(cond_p, tracers, params)

  _, _, t_out_uks = pe.partial_eval_jaxpr(true_jaxpr, t_uk, instantiate=False)
  _, _, f_out_uks = pe.partial_eval_jaxpr(false_jaxpr, f_uk, instantiate=False)
  out_uks = [a or b for a, b in zip(t_out_uks, f_out_uks)]

  true_jaxpr_1, true_jaxpr_2, _ = pe.partial_eval_jaxpr(true_jaxpr, t_uk,
                                                        instantiate=out_uks)
  false_jaxpr_1, false_jaxpr_2, _ = pe.partial_eval_jaxpr(false_jaxpr, f_uk,
                                                          instantiate=out_uks)

  num_t_res = len(true_jaxpr_1.out_avals) - len(out_uks)
  num_f_res = len(false_jaxpr_1.out_avals) - len(out_uks)

  move = [False] * len(true_jaxpr.in_avals) + [True] * num_t_res
  true_jaxpr_2 = pe.move_binders_to_front(true_jaxpr_2, move)
  move = [False] * len(false_jaxpr.in_avals) + [True] * num_f_res
  false_jaxpr_2 = pe.move_binders_to_front(false_jaxpr_2, move)

  # TODO(frostig,mattjj): pe.partial_eval_jaxpr should raise to shaped avals
  t_res_avals = _map(raise_to_shaped, true_jaxpr_2.in_avals[:num_t_res])
  f_res_avals = _map(raise_to_shaped, false_jaxpr_2.in_avals[:num_f_res])

  assert len(true_jaxpr_2.out_avals) == len(false_jaxpr_2.out_avals)
  num_outs = len(true_jaxpr_2.out_avals)

  true_jaxpr_1 = _join_cond_outputs(
      true_jaxpr_1, num_outs, f_res_avals, zeros_on_left=False)
  false_jaxpr_1 = _join_cond_outputs(
      false_jaxpr_1, num_outs, t_res_avals, zeros_on_left=True)

  # TODO(frostig,mattjj): reinstate this assertion once pe.partial_eval_jaxpr
  # raises to shaped avals
  # assert true_jaxpr_1.out_avals == false_jaxpr_1.out_avals
  num_res = num_t_res + num_f_res

  _, in_consts = unzip2([t.pval for t in tracers])
  out_consts_res = cond_p.bind(
      *in_consts, true_jaxpr=true_jaxpr_1, false_jaxpr=false_jaxpr_1,
      linear=linear)
  out_consts, res = split_list(out_consts_res, [len(out_consts_res) - num_res])

  # TODO(frostig,mattjj): remove raised_to_shaped of avals once
  # pe.partial_eval_jaxpr handles it
  out_avals = _map(raise_to_shaped, true_jaxpr_2.out_avals)
  out_pvs = [aval if uk else None for aval, uk in zip(out_avals, out_uks)]

  pred_tracer = trace.instantiate_const(tracers[0])

  ops_tracers = [trace.instantiate_const(t) if uk
                 else trace.new_instantiated_literal(core.unit)
                 for uk, t in zip(unknowns[1:], tracers[1:])]
  true_ops_tracers, false_ops_tracers = split_list(
      ops_tracers, [len(true_jaxpr.in_avals)])

  res_tracers = _map(trace.new_instantiated_const, res)
  true_res_tracers, false_res_tracers = split_list(res_tracers, [num_t_res])

  out_tracers = [pe.JaxprTracer(trace, pe.PartialVal((pv, const)), None)
                 for pv, const in zip(out_pvs, out_consts)]

  tops_lin, fops_lin = _map(tuple, split_list(linear, [len(true_jaxpr.in_avals)]))
  linear_2 = ((False,) * num_t_res + tops_lin + (False,) * num_f_res + fops_lin)
  params = dict(true_jaxpr=true_jaxpr_2, false_jaxpr=false_jaxpr_2,
                linear=linear_2)
  eqn = pe.new_eqn_recipe([pred_tracer] +
                          true_res_tracers + true_ops_tracers +
                          false_res_tracers + false_ops_tracers,
                          out_tracers,
                          cond_p, params)
  for t in out_tracers: t.recipe = eqn
  return out_tracers

def _join_cond_outputs(jaxpr, num_prefix, zeros_avals, zeros_on_left):
  @lu.wrap_init
  def f_aug(*args):
    prefix_and_rest = core.jaxpr_as_fun(jaxpr)(*args)
    prefix, rest = split_list(prefix_and_rest, [num_prefix])
    zeros = [ad_util.zeros_like_aval(a) for a in zeros_avals]
    if zeros_on_left:
      return prefix + zeros + rest
    else:
      return prefix + rest + zeros

  return _make_typed_jaxpr(f_aug, jaxpr.in_avals)

def _transpose_cond_jaxpr(jaxpr, num_res):
  num_non_res = len(jaxpr.in_avals) - num_res
  res_avals, primal_avals = split_list(jaxpr.in_avals, [num_res])
  primal_avals = _map(raise_to_shaped, primal_avals)

  @lu.wrap_init
  def transposed(*args):
    res, cts_out = split_list(args, [num_res])
    primals = res + [ad.undefined_primal] * num_non_res
    cts_in = ad.backward_pass(
        jaxpr.jaxpr, jaxpr.literals, primals, cts_out)
    _, cts_in = split_list(cts_in, [num_res])
    return _map(ad.instantiate_zeros_aval, primal_avals, cts_in)

  return _make_typed_jaxpr(transposed, res_avals + jaxpr.out_avals)

def _cond_transpose(cts, *args, true_jaxpr, false_jaxpr, linear):
  (pred,), tops, fops = split_list(args, [1, len(true_jaxpr.in_avals)])
  tops_lin, fops_lin = split_list(linear, [len(true_jaxpr.in_avals)])
  in_avals = _map(raise_to_shaped, true_jaxpr.in_avals + false_jaxpr.in_avals)

  num_t_res = len(tops) - sum(tops_lin)
  num_f_res = len(fops) - sum(fops_lin)

  t_jaxpr_trans = _transpose_cond_jaxpr(true_jaxpr, num_t_res)
  f_jaxpr_trans = _transpose_cond_jaxpr(false_jaxpr, num_f_res)
  lin_in_avals = _map(raise_to_shaped, [a for a, l in zip(in_avals, linear) if l])
  assert t_jaxpr_trans.out_avals + f_jaxpr_trans.out_avals == lin_in_avals

  t_jaxpr_trans_ = _join_cond_outputs(
      t_jaxpr_trans, 0, f_jaxpr_trans.out_avals, zeros_on_left=False)
  f_jaxpr_trans_ = _join_cond_outputs(
      f_jaxpr_trans, 0, t_jaxpr_trans.out_avals, zeros_on_left=True)
  assert t_jaxpr_trans_.out_avals == f_jaxpr_trans_.out_avals == lin_in_avals

  t_res, _ = split_list(tops, [num_t_res])
  f_res, _ = split_list(fops, [num_f_res])

  linear_trans = ((False,) * num_t_res + (True,) * len(cts) +
                  (False,) * num_f_res + (True,) * len(cts))

  cts = _map(ad.instantiate_zeros_aval, true_jaxpr.out_avals, cts)

  out = cond_p.bind(
      pred, *itertools.chain(t_res, cts, f_res, cts),
      true_jaxpr=t_jaxpr_trans_, false_jaxpr=f_jaxpr_trans_,
      linear=linear_trans)
  assert all(_map(typecheck, lin_in_avals, out))

  out_iter = iter(out)
  out = [next(out_iter) if l else None for l in linear]
  assert next(out_iter, None) is None
  return [None] + out

def cond_bind(*args, true_jaxpr, false_jaxpr, linear):
  if not core.skip_checks:
    assert len(linear) + 1 == len(args)
    assert len(args) == 1 + len(true_jaxpr.in_avals) + len(false_jaxpr.in_avals)
    (pred,), tops, fops = split_list(args, [1, len(true_jaxpr.in_avals)])
    assert all(_map(typecheck, true_jaxpr.in_avals, tops))
    assert all(_map(typecheck, false_jaxpr.in_avals, fops))
    core.check_jaxpr(true_jaxpr.jaxpr)
    core.check_jaxpr(false_jaxpr.jaxpr)
  return core.Primitive.bind(cond_p, *args, true_jaxpr=true_jaxpr,
                             false_jaxpr=false_jaxpr, linear=linear)

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

def scan(f, init, xs, length=None):
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

  Returns:
    A pair of type ``(c, [b])`` where the first element represents the final
    loop carry value and the second element represents the stacked outputs of
    the second output of ``f`` when scanned over the leading axis of the inputs.
  """
  init_flat, init_tree = tree_flatten(init)
  xs_flat, _ = tree_flatten(xs)
  in_flat, in_tree = tree_flatten((init, xs))

  try:
    lengths = [x.shape[0] for x in xs_flat]
  except AttributeError:
    msg = "scan got value with no leading axis to scan over: {}."
    raise ValueError(msg.format(', '.join(str(x) for x in xs_flat
                                          if not hasattr(x, 'shape'))))

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
                    forward=True, length=length, jaxpr=jaxpr,
                    num_consts=len(consts), num_carry=len(init_flat),
                    linear=(False,) * (len(consts) + len(in_flat)))
  return tree_unflatten(out_tree, out)

def _scan_impl(*args, forward, length, num_consts, num_carry, jaxpr, linear):
  consts, init, xs = split_list(args, [num_consts, num_carry])
  _, _, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
  _, y_avals = split_list(jaxpr.out_avals, [num_carry])

  def body_fun(i, vals):
    i = i if forward else length - i - 1
    carry, ys = split_list(vals, [num_carry])
    x = _map(partial(_index_array, i), x_avals, xs)
    out_flat = core.jaxpr_as_fun(jaxpr)(*(consts + carry + x))
    carry_out, y_updates = split_list(out_flat, [num_carry])
    ys_out = _map(partial(_update_array, i), y_avals, ys, y_updates)
    return carry_out + ys_out

  ys_init = _map(partial(_empty_array, length), y_avals)
  return fori_loop(lax._const(length, 0), length, body_fun, init + ys_init)

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

# TODO(mattjj): make scan a primitive
# def _scan_abstract_eval(*args, forward, length, num_consts, num_carry, jaxpr, linear):
#   carry_avals, y_avals = split_list(jaxpr.out_avals, [num_carry])
#   ys_avals = [ShapedArray((length,) + aval.shape, aval.dtype)
#               if aval is not core.abstract_unit else aval for aval in y_avals]
#   return carry_avals + y_avals

def _scan_jvp(primals, tangents, forward, length, jaxpr, num_consts, num_carry,
              linear):
  num_xs = len(jaxpr.in_avals) - num_carry - num_consts
  num_ys = len(jaxpr.out_avals) - num_carry
  nonzeros = [t is not ad_util.zero for t in tangents]
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
    carry_nz_out, ys_nz = nonzeros_out[:num_carry], nonzeros_out[num_carry:]
    if carry_nz_out == carry_nz:
      break
    else:
      carry_nz = _map(operator.or_, carry_nz, carry_nz_out)
  else:
    assert False, "Fixpoint not reached"

  tangents = [ad.instantiate_zeros(x, t) if t is ad_util.zero and nz else t
              for x, t, nz in zip(primals, tangents, nonzeros)]

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
      forward=forward, length=length, jaxpr=jaxpr_jvp_rearranged,
      num_consts=num_consts+len(consts_dot), num_carry=num_carry+len(init_dot),
      linear=jaxpr_jvp_linear)

  carry, carry_dot, ys, ys_dot = split_list(out_flat, [num_carry, len(init_dot), num_ys])
  primals_out = carry + ys
  tangents_out_iter = iter(carry_dot + ys_dot)
  tangents_out = [next(tangents_out_iter) if nz else ad_util.zero
                  for nz in nonzeros_out]
  return primals_out, tangents_out

def _prune_zeros(ts):
  return [t for t in ts if t is not ad_util.zero]

def _scan_partial_eval(trace, *tracers, forward, length, num_consts, num_carry,
                       jaxpr, linear):
  num_xs = len(jaxpr.in_avals) - num_carry - num_consts
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
        jaxpr, unknowns, instantiate=carry_uk + [False] * num_ys)
    carry_uk_out, ys_uk = out_uk[:num_carry], out_uk[num_carry:]
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
  invariant_pvals = [pe.PartialVal((None, core.unit if uk else t.pval[1]))
                     for uk, t in zip(unknowns[:num_consts], tracers[:num_consts])]
  other_pvals = [pe.PartialVal((a, core.unit)) for a in jaxpr_1.in_avals[num_consts:]]
  in_pvals_1 = invariant_pvals + other_pvals
  untyped_jaxpr_1, out_pvals_1, consts_1 = pe.trace_to_jaxpr(
      lu.wrap_init(core.jaxpr_as_fun(jaxpr_1)), in_pvals_1,
      instantiate=[True] * (num_carry + num_ys) + [False] * num_res)
  const_avals_1 = [raise_to_shaped(core.get_aval(c)) for c in consts_1]
  in_avals_1 = [core.abstract_unit] * num_consts + jaxpr_1.in_avals[num_consts:]
  out_avals_1 = [core.abstract_unit if pv is None else pv for pv, c in out_pvals_1]
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
      *in_consts, forward=forward, length=length, jaxpr=jaxpr_1_opt,
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
                          dict(forward=forward, length=length, jaxpr=jaxpr_2_opt,
                               num_consts=num_consts_2,
                               num_carry=num_carry, linear=tuple(linear_2)))
  for t in out_tracers: t.recipe = eqn
  return out_tracers

def _promote_aval_rank(sz, aval):
  if aval is core.abstract_unit:
    return core.abstract_unit
  else:
    return ShapedArray((sz,) + aval.shape, aval.dtype)

def _scan_transpose(cts, *args, forward, length, num_consts, num_carry, jaxpr, linear):
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
  assert not any(r is ad.undefined_primal for r in ires)
  assert not any(r is ad.undefined_primal for r in eres)

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
      *(ires + ct_consts + ct_carry + ct_ys + eres), forward=not forward,
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
    primals = res1 + [ad.undefined_primal] * (num_c + num_a) + res2
    cbar_abar = ad.backward_pass(jaxpr.jaxpr, jaxpr.literals, primals,
                                    b_bar)
    _, new_c_bar, a_bar, _ = split_list(cbar_abar, [num_res1, num_c, num_a])
    a_bar = _map(ad.instantiate_zeros_aval, a_avals, a_bar)
    c_bar = _map(ad.instantiate_zeros_aval, c_avals,
                _map(ad.add_tangents, c_bar, new_c_bar))
    return c_bar + a_bar
  return _make_typed_jaxpr(transposed, res1_avals + c_avals + b_avals + res2_avals)

def _make_typed_jaxpr(traceable, in_avals):
  pvals = [pe.PartialVal((aval, core.unit)) for aval in in_avals]
  jaxpr, pvals_out, consts = pe.trace_to_jaxpr(traceable, pvals, instantiate=True)
  out_avals, _ = unzip2(pvals_out)
  return core.TypedJaxpr(jaxpr, consts, in_avals, _map(raise_to_shaped, out_avals))


def _scan_batching_rule(args, dims, forward, length, jaxpr, num_consts,
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

  outs = scan_p.bind(*new_args, forward=forward, length=length, jaxpr=jaxpr_batched,
                     num_consts=num_consts, num_carry=num_carry, linear=linear)
  carry_bdims = [0 if b else batching.not_mapped for b in carry_batched]
  ys_bdims = [1 if b else batching.not_mapped for b in ys_batched]
  return outs, carry_bdims + ys_bdims

def _scan_shape_rule(shapes, forward, length, jaxpr,
                     num_consts, num_carry, linear):
  const_shexprs, init_shexprs, xs_shexprs = split_list(shapes, [num_consts, num_carry])
  _, y_avals = split_list(jaxpr.out_avals, [num_carry])
  ys_shapes = [(length,) + tuple(y_aval.shape) for y_aval in y_avals]
  return init_shexprs + ys_shapes

def _scan_masking_rule(shape_envs, padded_vals, shape_exprs, forward, length,
                       jaxpr, num_consts, num_carry, linear):
  out_shape = _scan_shape_rule(shape_exprs, forward, length, jaxpr,
                               num_consts, num_carry, linear)
  dynamic_length = masking.eval_dim_expr(shape_envs.logical, length)
  masked_jaxpr = _masked_scan_jaxpr(jaxpr, num_consts, num_carry)
  consts, init, xs = split_list(padded_vals, [num_consts, num_carry])
  max_length, = {x.shape[0] for x in xs}
  const_linear, init_linear, xs_linear = split_list(linear, [num_consts, num_carry])
  out_vals = scan_p.bind(
      *itertools.chain([dynamic_length] + consts, [0], init, xs),
      forward=forward, length=max_length, jaxpr=masked_jaxpr,
      num_consts=1 + num_consts, num_carry=1 + num_carry,
      linear=[False] + const_linear + [False] + init_linear + xs_linear)
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

def scan_bind(*args, forward, length, num_consts, num_carry, jaxpr, linear):
  if not core.skip_checks:
    assert len(linear) == len(args)
    consts, init, xs = split_list(args, [num_consts, num_carry])
    consts_avals, init_avals, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
    xs_avals = _map(partial(_promote_aval_rank, length), x_avals)
    assert all(_map(typecheck, consts_avals, consts)), (consts, consts_avals)
    assert all(_map(typecheck, init_avals, init))
    # assert all(_map(typecheck, xs_avals, xs))
    carry_avals, _ = split_list(jaxpr.out_avals, [num_carry])
    assert all(_map(typematch, init_avals, carry_avals))
    core.check_jaxpr(jaxpr.jaxpr)
  return core.Primitive.bind(scan_p, *args, forward=forward, length=length,
                             jaxpr=jaxpr, num_consts=num_consts,
                             num_carry=num_carry, linear=linear)

scan_p = core.Primitive("scan")
scan_p.multiple_results = True
scan_p.def_custom_bind(scan_bind)
scan_p.def_impl(_scan_impl)
ad.primitive_jvps[scan_p] = _scan_jvp
ad.primitive_transposes[scan_p] = _scan_transpose
pe.custom_partial_eval_rules[scan_p] = _scan_partial_eval
xla.initial_style_translations[scan_p] = xla.lower_fun(_scan_impl, initial_style=True)
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


def _concat_masking_rule(padded_vals, logical_shapes, dimension, operand_shapes):
  del operand_shapes  # Unused.
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

  Corresponding `tree` and `avals` must match in the sense that the number of leaves in
  `tree` must be equal to the length of `avals`.
  `what` will be prepended to details of the mismatch in TypeError.
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
    unchecked_zeros, f_jvp = api.linearize(f, x)
    return tangent_solve(f_jvp, b)

  l_and_s_jaxpr, l_and_s_consts, out_tree = _initial_style_jaxpr(
      linearize_and_solve, treedef_tuple((in_tree,) * 2), guess_avals * 2)
  _check_tree("tangent_solve", "x", out_tree, in_tree)

  all_consts = [f_consts, solve_consts, l_and_s_consts]
  const_lengths = _RootTuple(*_map(len, all_consts))
  jaxprs = _RootTuple(f_jaxpr, solve_jaxpr, l_and_s_jaxpr)

  out_flat = root_p.bind(
      *(_flatten(all_consts) + guess_flat),
      const_lengths=const_lengths, jaxprs=jaxprs)
  return tree_unflatten(out_tree, out_flat)


def _root_abstract_eval(*args, **kwargs):
  return _map(raise_to_shaped, args[sum(kwargs['const_lengths']):])


def _root_impl(*args, **kwargs):
  const_lengths, jaxprs = split_dict(kwargs, ['const_lengths', 'jaxprs'])
  params, initial_guess = _split_root_args(args, const_lengths)
  solution = core.jaxpr_as_fun(jaxprs.solve)(*(params.solve + initial_guess))
  return solution


def _root_jvp(primals, tangents, const_lengths, jaxprs):
  params, _ = _split_root_args(primals, const_lengths)
  solution = tuple(root_p.bind(
      *primals, const_lengths=const_lengths, jaxprs=jaxprs))

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


root_p = core.Primitive('root')
root_p.multiple_results = True
root_p.def_impl(_root_impl)
root_p.def_abstract_eval(_root_abstract_eval)
ad.primitive_jvps[root_p] = _root_jvp
xla.initial_style_translations[root_p] = xla.lower_fun(
    _root_impl, initial_style=True)
# TODO(shoyer): write batching rule


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
  _, vjp_fun = api.vjp(linear_fun, primals)

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
    actual_shape_tree = tree_unflatten(tree, actual_shapes)
    act_shape_tree = tree_unflatten(tree, actual_shapes)
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
  assert any(p is not ad_util.zero for p in params_dot)
  zeros = [ad_util.zero] * len(x)
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

  if all(p is ad_util.zero for p in params_dot.matvec):
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
  assert b == [ad.undefined_primal] * len(b)
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
xla.initial_style_translations[linear_solve_p] = xla.lower_fun(
    _custom_linear_solve_impl, initial_style=True)
ad.primitive_transposes[linear_solve_p] = _linear_solve_transpose_rule
batching.primitive_batchers[linear_solve_p] = _linear_solve_batching_rule
