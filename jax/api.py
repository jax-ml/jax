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

"""
User-facing transformations.

These mostly wrap internal transformations, providing convenience flags to
control behavior and handling Python containers (tuples/lists/dicts) of
arguments and outputs.
"""

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

def _wraps(wrapped):
  def decorator(wrapper):
    wrapper.__name__ = getattr(wrapped, "__name__", "<unnamed function>")
    wrapper.__module__ = getattr(wrapped, "__module__", "<unknown module>")
    if hasattr(wrapped, "__doc__"):
      wrapper.__doc__ = getattr(wrapped, "__doc__")
    return wrapper
  return decorator


def jit(fun, static_argnums=()):
  """Sets up `fun` for just-in-time compilation with XLA.

  Args:
    fun: Function to be jitted. Should be a pure function, as side-effects may
        only be executed once. Its positional arguments and return value should
        be arrays, scalars, or standard Python containers (tuple/list/dict)
        thereof. Keyword arguments and positional arguments specified by
        `static_argnums` can be anything at all. These are treated as static
        (see below).
    static_argnums: A tuple of ints. Specifies which arguments to treat as
        static (compile-time constant). Operations that only depend on static
        arguments will be constant-folded. Calling the jitted function with
        different values for these constants will trigger recompilation.
   Returns: A wrapped version of `fun`, set up for just-in-time compilation.
  """
  @_wraps(fun)
  def f_jitted(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
    f, dyn_args = argnums_partial(f, dyn_argnums, args)
    args_flat, in_trees = unzip2(map(tree_to_jaxtuples, dyn_args))
    check_args(args_flat)
    flat_fun, out_tree = flatten_fun(f, in_trees)
    out_flat = xla.xla_call(flat_fun, *args_flat)
    return build_tree(out_tree(), out_flat)

  f_jitted.__name__ = "jit({})".format(f_jitted.__name__)
  return f_jitted

def grad(fun, argnums=0):
  """Creates a function which evaluates the gradient of `fun`.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
        `argnums` should be arrays, scalars, or standard Python containers. It
        should return a scalar (which includes arrays with shape `()` but not
        arrays with shape `(1,)` etc.)
    argnums: Integer or tuple of integers. Specifies which positional
        argument(s) to differentiate with respect to.
  Returns: A function with the same arguments as `fun`, that evaluates the
      gradient of `fun`. If `argnums` is an integer then the gradient has the
      same shape and type as the positional argument indicated by that integer.
      If argnums is a tuple of integers, the gradient is a tuple of values with
      the same shapes and types as the corresponding arguments.
  """
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
  """Jacobian of `fun`, evaluated column-by-column using forward-mode AD"""
  fun = lu.wrap_init(fun)
  pushfwd = partial(jvp, fun, (x,))
  std_basis = onp.eye(onp.size(x)).reshape((-1,) + onp.shape(x)),
  y, jac_flat = vmap(pushfwd, out_axes=(None, -1))(std_basis)
  return jac_flat.reshape(onp.shape(y) + onp.shape(x))

@curry
def jacrev(fun, x):
  """Jacobian of `fun`, evaluated row-by-row using reverse-mode AD"""
  fun = lu.wrap_init(fun)
  y, pullback = vjp(fun, x)
  std_basis = onp.eye(onp.size(y)).reshape((-1,) + onp.shape(y))
  jac_flat, = vmap(pullback, out_axes=0)(std_basis)
  return jac_flat.reshape(onp.shape(y) + onp.shape(x))

def hessian(fun):
  return jacfwd(jacrev(fun))

def vmap(fun, in_axes=0, out_axes=0):
  """Vectorizing map. Creates a function which maps `fun` over additional axes.

  Args:
    fun: Function to be mapped over additional axes.
    in_axes, out_axes: Specifies which axes to map over. These may be integers,
        None, or (possibly nested) tuples of integers or None.
  Returns: Batched/vectorized version of `fun` with arguments that correspond to
      those of `fun`, but with extra array axes at positions indicated by
      `in_axes`, and a return value that corresponds to that of `fun`, but with
      extra array axes at positions indicated by `out_axes`.

  For example, we can implement a matrix-matrix product using a vector dot
  product:

    vv = lambda x, y: np.vdot(x, y)  #  ([a], [a]) -> []
    mv = vmap(vv, (0, None), 0)      #  ([a,b], [b]) -> [a]
    mm = vmap(mv, (None, 1), 1)      #  ([a,b], [b,c]) -> [a,c]

  (`[a,b]` indicates an array with shape (a,b))

  """
  def batched_fun(*args, **kwargs):
    if not isinstance(fun, lu.WrappedFun):
      f = lu.wrap_init(fun)
    in_axes_ = (in_axes,) * len(args) if type(in_axes) is int else in_axes
    in_flat, in_trees = unzip2(map(tree_to_jaxtuples, args))
    flat_fun, out_tree = flatten_fun(f, in_trees)
    out_flat = batching.batch(flat_fun, in_flat, in_axes_, out_axes)
    return build_tree(out_tree(), out_flat)

  return batched_fun

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

def make_jaxpr(f):
  # TODO(frostig): handle container trees etc.
  def pv_like(x):
    aval = ShapedArray(onp.shape(x), onp.result_type(x))
    return pe.PartialVal((aval, core.unit))

  @_wraps(f)
  def jaxpr_maker(*args):
    jaxpr, _, _, _ = trace_to_jaxpr(f, map(pv_like, args))
    return jaxpr

  jaxpr_maker.__name__ = "make_jaxpr({})".format(jaxpr_maker.__name__)
  return jaxpr_maker


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
