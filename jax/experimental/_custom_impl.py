# Copyright 2023 The JAX Authors.
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
custom_impl

Experimental transform to replace a primitive with a custom implementaion.
"""
from functools import partial, wraps

import jax
from jax import core
from jax import lax
from jax.api_util import flatten_fun, shaped_abstractify
from jax.experimental import pjit
from jax.interpreters import partial_eval as pe
from jax.linear_util import wrap_init
from jax.tree_util import tree_flatten, tree_map, tree_unflatten


def safe_map(f, *args):
  args = list(map(list, args))
  if len(set(map(len, args))) != 1:
    raise ValueError(f"length mismatch: {list(map(len, args))}")
  return list(map(f, *args))


def custom_impl(prim, impl):
  """Experimental transformation to inject custom primitive implementations

  Example:

    from jax.experimental import custom_impl
    from jax import lax

    def f32_dot_general(x, y, **kwargs):
      x_f32 = x.astype('float32')
      y_f32 = y.astype('float32')
      return lax.dot_general(x_f32, y_f32, **kwargs).astype(x.dtype)

    @custom_impl(lax.dot_general_p, f32_dot_general)
    def func(x):
      return x.T @ x

    out = func(x)  # uses f32_dot_general in place of all dot_general calls
  """
  if not isinstance(prim, core.Primitive):
    raise ValueError(
        f"First argument to custom_impl should be a primitive. Got {prim}")
  new_impls = {prim: impl}
  def custom_impl_transformation(fun):
    @wraps(fun)
    def wrapped(*args, **kwargs):
      args_flat, in_tree = tree_flatten((args, kwargs))
      in_avals_flat = [core.get_aval(arg) for arg in args_flat]
      wrapped_fun, out_tree = flatten_fun(wrap_init(fun), in_tree)
      jaxpr, out_avals_flat, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals_flat)
      result = _custom_impl_jaxpr(new_impls, jaxpr,consts, *args)
      assert len(out_avals_flat) == len(result)
      return tree_unflatten(out_tree(), result)
    return wrapped
  return custom_impl_transformation


def _custom_impl_jaxpr(custom_impls, jaxpr, consts, *args):
  env = {}

  def read(var):
    if type(var) is core.Literal:
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val

  safe_map(write, jaxpr.invars, args)
  safe_map(write, jaxpr.constvars, consts)

  for eqn in jaxpr.eqns:
    invals = safe_map(read, eqn.invars)
    in_avals = [core.get_aval(inval) for inval in invals]
    # TODO(jakevdp): support other higher-order primitives.
    if eqn.primitive in (pjit.pjit_p, lax.scan_p):
      new_jaxpr = jax.make_jaxpr(partial(_custom_impl_jaxpr, custom_impls,
                                         eqn.params['jaxpr'].jaxpr,
                                         eqn.params['jaxpr'].literals))(*in_avals)
      outvals = eqn.primitive.bind(*invals, **{**eqn.params, 'jaxpr': new_jaxpr})
    elif eqn.primitive == lax.while_p:
      new_cond_jaxpr = jax.make_jaxpr(partial(_custom_impl_jaxpr, custom_impls,
                                              eqn.params['cond_jaxpr'].jaxpr,
                                              eqn.params['cond_jaxpr'].literals))(*in_avals)
      new_body_jaxpr = jax.make_jaxpr(partial(_custom_impl_jaxpr, custom_impls,
                                              eqn.params['body_jaxpr'].jaxpr,
                                              eqn.params['body_jaxpr'].literals))(*in_avals)
      outvals = eqn.primitive.bind(*invals, **{**eqn.params,
                                               'cond_jaxpr': new_cond_jaxpr,
                                               'body_jaxpr': new_body_jaxpr})
    elif eqn.primitive in custom_impls:
      outvals = custom_impls[eqn.primitive](*invals, **eqn.params)
      out_avals = tree_map(lambda val: shaped_abstractify(core.get_aval(val)), outvals)
      expected_out_avals = [var.aval for var in eqn.outvars]
      if not eqn.primitive.multiple_results:
        expected_out_avals = expected_out_avals[0]
      if out_avals != expected_out_avals:
        raise ValueError(
            f"custom impl for {eqn.primitive} returned the wrong output types.\n"
            f"  expected: {expected_out_avals}\n"
            f"  actual:   {out_avals}")
    else:
      outvals = eqn.primitive.bind(*invals, **eqn.params)
    if not eqn.primitive.multiple_results:
      outvals = [outvals]
    safe_map(write, eqn.outvars, outvals)
  return safe_map(read, jaxpr.outvars)
