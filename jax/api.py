# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as onp

from . import core
from . import linear_util as lu
from .core import pack, eval_jaxpr
from .api_util import flatten_fun, unflatten_fun, tree_to_jaxtuples
from .tree_util import (process_pytree, node_types, build_tree, PyTreeDef, leaf,
                        tree_map)
from .util import unzip2, unzip3, curry, partial, safe_map, WrapHashably
from .abstract_arrays import ShapedArray
from .interpreters import partial_eval as pe
from .interpreters import xla
from .interpreters import ad
from .interpreters import batching

map = safe_map

def jit(fun, static_argnums=()):
  def f_jitted(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
    f, dyn_args = argnums_partial(f, dyn_argnums, args)
    args_flat, in_trees = unzip2(map(tree_to_jaxtuples, dyn_args))
    check_args(args_flat)
    flat_fun, out_tree = flatten_fun(f, in_trees)
    out_flat = xla.xla_call(flat_fun, *args_flat)
    return build_tree(out_tree(), out_flat)

  return f_jitted

def grad(fun, argnums=0):
  def grad_f(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    ans, vjp_py = vjp(f_partial, *dyn_args)
    check_scalar(ans)
    g = vjp_py(onp.ones((), onp.result_type(ans)))
    return g[0] if isinstance(argnums, int) else g


  return grad_f

@curry
def jacfwd(fun, x):
  fun = lu.wrap_init(fun)
  pushfwd = partial(jvp, fun, (x,))
  std_basis = onp.eye(onp.size(x)).reshape((-1,) + onp.shape(x)),
  y, jac_flat = vmap(pushfwd, std_basis, out_axes=(None, 0))
  return jac_flat.reshape(onp.shape(y) + onp.shape(x))

@curry
def jacrev(fun, x):
  fun = lu.wrap_init(fun)
  y, pullback = vjp(fun, x)
  std_basis = onp.eye(onp.size(y)).reshape((-1,) + onp.shape(y))
  jac_flat, = vmap(pullback, std_basis, out_axes=onp.ndim(y))
  return jac_flat.reshape(onp.shape(y) + onp.shape(x))

def hessian(fun):
  return jacfwd(jacrev(fun))

def vmap(fun, *args, **kwargs):
  in_axes = kwargs.pop("in_axes", 0)
  out_axes = kwargs.pop("out_axes", 0)
  if kwargs:
    msg = "vmap keyword args must be 'in_axes' and/or 'out_axes', got {}."
    raise TypeError(msg.format(', '.join(kwargs)))

  if type(in_axes) is int:
    in_axes = (in_axes,) * len(args)
  if not isinstance(fun, lu.WrappedFun):
    fun = lu.wrap_init(fun)
  in_flat, in_trees = unzip2(map(tree_to_jaxtuples, args))
  flat_fun, out_tree = flatten_fun(fun, in_trees)
  out_flat = batching.batch(flat_fun, in_flat, in_axes, out_axes)
  return build_tree(out_tree(), out_flat)

def jvp(fun, primals, tangents):
  def flatten_arg(primal, tangent):
    primal_jtuple, tree_def = tree_to_jaxtuples(primal)
    tangent_jtuple, tree_def_2 = tree_to_jaxtuples(tangent)
    assert tree_def == tree_def_2, (tree_def, tree_def_2)
    return primal_jtuple, tangent_jtuple, tree_def

  if not isinstance(fun, lu.WrappedFun):
    fun = lu.wrap_init(fun)
  ps_flat, ts_flat, in_trees = unzip3(map(flatten_arg, primals, tangents))
  flat_fun, out_tree = flatten_fun(fun, in_trees)
  out_primal, out_tangent = ad.jvp(flat_fun).call_wrapped(ps_flat, ts_flat)
  return (build_tree(out_tree(), out_primal), build_tree(out_tree(), out_tangent))

def linearize(traceable, *primals):
  fun = lu.wrap_init(traceable)
  primals_flat, in_trees = unzip2(map(tree_to_jaxtuples, primals))
  flat_fun, out_tree = flatten_fun(fun, in_trees)
  out_primal, out_pval, jaxpr, consts = ad.linearize(flat_fun, *primals_flat)
  out_tree = out_tree()
  out_primal_py = build_tree(out_tree, out_primal)
  lifted_jvp = partial(lift_linearized, jaxpr, consts, (in_trees, out_tree), out_pval)
  return out_primal_py, lifted_jvp

def lift_linearized(jaxpr, consts, io_tree, out_pval, py_args):
  def fun(*args):
    primals = pack(args) # doesn't matter what these are-they'll be ignored
    tangents = pack(args)
    _, ans = eval_jaxpr(jaxpr, consts, (), primals, tangents)
    return pe.merge_pvals(ans, out_pval)

  return unflatten_fun(fun, io_tree, *py_args)

def vjp(fun, *primals):
  if not isinstance(fun, lu.WrappedFun):
    fun = lu.wrap_init(fun)
  primals_flat, in_trees = unzip2(map(tree_to_jaxtuples, primals))
  check_args(primals_flat)
  flat_fun, out_tree = flatten_fun(fun, in_trees)
  out_primal, out_vjp = ad.vjp(flat_fun, primals_flat)
  out_tree = out_tree()
  out_primal_py = build_tree(out_tree, out_primal)
  ct_in_trees = [out_tree]
  ct_out_tree = PyTreeDef(node_types[tuple], None, in_trees)
  def out_vjp_packed(cotangent_in):
    return out_vjp(cotangent_in)
  vjp_py = partial(unflatten_fun, out_vjp_packed, (ct_in_trees, ct_out_tree))
  return out_primal_py, vjp_py


def trace_to_jaxpr(traceable, py_pvals, **kwargs):
  fun = lu.wrap_init(traceable)
  pvals, in_trees = unzip2(map(tree_to_pval_tuples, py_pvals))
  flat_fun, out_tree = flatten_fun(fun, in_trees)
  jaxpr, out_pval, consts = pe.trace_to_jaxpr(flat_fun, pvals, **kwargs)
  return jaxpr, consts, out_pval, (in_trees, out_tree())

def lift_jaxpr(jaxpr, consts, io_tree, pvals, py_args):
  def fun(*args):
    ans = eval_jaxpr(jaxpr, consts, (), *args)
    return pe.merge_pvals(ans, pvals)
  return unflatten_fun(fun, io_tree, *py_args)


device_put = jit(lambda x: x)
device_get_array = lambda x: x.copy() if type(x) is xla.DeviceArray else x
device_get = partial(tree_map, device_get_array)


@lu.transformation_with_aux
def flatten_fun(in_trees, *args, **kwargs):
  py_args = map(build_tree, in_trees, args)
  ans = yield py_args
  yield process_pytree(pack, ans)


def unflatten_fun(fun, io_tree, *py_args):
  in_trees_expected, out_tree = io_tree
  args, in_trees = unzip2(map(tree_to_jaxtuples, py_args))
  for i, (in_tree, expected) in enumerate(zip(in_trees, in_trees_expected)):
    if in_tree != expected:
      raise TypeError("Expected {}, got {}".format(expected, in_tree))

  ans = fun(*args)
  return build_tree(out_tree, ans)


tree_to_pval_tuples = partial(process_pytree, pe.pack_pvals)
tree_to_jaxtuples = partial(process_pytree, pack)


def argnums_partial(f, dyn_argnums, args):
  if isinstance(dyn_argnums, int):
    dyn_argnums = (dyn_argnums,)
  else:
    dyn_argnums = tuple(dyn_argnums)
  fixed_args = tuple([None if i in dyn_argnums else WrapHashably(arg)
                      for i, arg in enumerate(args)])
  dyn_args = [args[i] for i in dyn_argnums]
  return argnums_partial_(f, dyn_argnums, fixed_args), dyn_args

@lu.transformation
def argnums_partial_(dyn_argnums, fixed_args, *dyn_args):
  args = [None if arg is None else arg.val for arg in fixed_args]
  for i, arg in zip(dyn_argnums, dyn_args):
    args[i] = arg
  ans = yield args
  yield ans

def check_args(args):
  for arg in args:
    if not (isinstance(arg, core.Tracer) or core.valid_jaxtype(arg)):
      raise TypeError("Argument '{}' of type {} is not a valid JAX type"
                      .format(arg, type(arg)))

def check_scalar(x):
  msg = "Gradient only defined for scalar-output functions. Output was: {}".format
  try:
    aval = core.get_aval(x)
    if not (isinstance(aval, ShapedArray) and aval.shape == ()):
      raise TypeError(msg(x))
  except TypeError:
    raise TypeError(msg(x))
