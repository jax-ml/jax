# coding=utf-8
# Copyright 2020 Google LLC
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

from functools import partial, update_wrapper
import inspect
import itertools as it

from . import core
from . import linear_util as lu
from .tree_util import tree_flatten, tree_unflatten
from .util import safe_zip, safe_map, unzip2, split_list, curry
from .api_util import flatten_fun_nokwargs, argnums_partial, wrap_hashably
from .abstract_arrays import raise_to_shaped
from .ad_util import zero
from .interpreters import partial_eval as pe
from .interpreters import ad
from .interpreters import batching
from .interpreters import xla

map = safe_map
zip = safe_zip


### util

def _resolve_kwargs(fun, args, kwargs):
  ba = inspect.signature(fun).bind(*args, **kwargs)
  ba.apply_defaults()
  if ba.kwargs:
    raise TypeError("keyword arguments could not be resolved to positions")
  else:
    return ba.args

def _initial_style_jaxpr(fun, in_avals):
  in_pvals = [pe.PartialVal((aval, core.unit)) for aval in in_avals]
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr(fun, in_pvals, instantiate=True,
                                               stage_out_calls=True)
  out_avals = map(raise_to_shaped, unzip2(out_pvals)[0])
  const_avals = [raise_to_shaped(core.get_aval(c)) for c in consts]
  typed_jaxpr = core.TypedJaxpr(pe.convert_constvars_jaxpr(jaxpr),
                                (), const_avals + in_avals, out_avals)
  return typed_jaxpr, consts

def _add_args(f, extra_args, left):
  return _add_args_(f, tuple(map(wrap_hashably, extra_args)), left)

@lu.transformation
def _add_args_(extra_args, left, *args, **kwargs):
  extra_args = tuple([arg.val for arg in extra_args])
  args = (extra_args + args) if left else (args + extra_args)
  yield (yield args, kwargs)

@curry
def transformation_with_equal_aux(gen, fun: lu.WrappedFun, *gen_static_args):
  out_store = StoreEqualValues()
  out_thunk = lambda: out_store.val
  return fun.wrap(gen, gen_static_args, out_store), out_thunk

class StoreEqualValues(lu.Store):
  """A Store that allows storing equal values multiple times."""
  def store(self, val):
    if self._val is not lu._EMPTY_STORE_VALUE:
      try:
        same = self._val == val
      except:
        same = False
      if not same:
        raise lu.StoreException("Store occupied")
    self._val = val


### JVPs

class custom_jvp:
  """Set up a JAX-transformable function for a custom JVP rule definition.

  This class is meant to be used as a function decorator. Instances are
  callables that behave similarly to the underlying function to which the
  decorator was applied, except when a differentiation transformation (like
  ``jax.jvp`` or ``jax.grad``) is applied, in which case a custom user-supplied
  JVP rule function is used instead of tracing into and performing automatic
  differentiation of the underlying function's implementation. There is a single
  instance method, ``defjvp``, which defines the custom JVP rule.

  For example:

    import jax.numpy as np

    @jax.custom_jvp
    def f(x, y):
      return np.sin(x) * y

    @f.defjvp
    def f_jvp(primals, tangents):
      x, y = primals
      x_dot, y_dot = tangents
      primal_out = f(x, y)
      tangent_out = np.cos(x) * x_dot * y - np.sin(x) * y_dot
      return primal_out, tangent_out

  For a more detailed introduction, see the tutorial_.

  .. _tutorial: https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
  """

  def __init__(self, fun, nondiff_argnums=()):
    self.fun = fun
    self.nondiff_argnums = nondiff_argnums
    self.jvp = None
    update_wrapper(self, fun)

  def defjvp(self, jvp):
    """Define a custom JVP rule for the function represented by this instance.

    Args:
      jvp: a Python callable representing the custom JVP rule. When there are no
        ``nondiff_argnums``, the ``jvp`` function should accept two arguments,
        where the first is a tuple of primal inputs and the second is a tuple of
        tangent inputs. The lengths of both tuples is equal to the number of
        parameters of the ``custom_jvp`` function. The ``jvp`` function should
        produce as output a pair where the first element is the primal output
        and the second element is the tangent output. Elements of the input and
        output tuples may be arrays or any nested tuples/lists/dicts thereof.

    Returns:
      None.

    Example:

      import jax.numpy as np

      @jax.custom_jvp
      def f(x, y):
        return np.sin(x) * y

      @f.defjvp
      def f_jvp(primals, tangents):
        x, y = primals
        x_dot, y_dot = tangents
        primal_out = f(x, y)
        tangent_out = np.cos(x) * x_dot * y - np.sin(x) * y_dot
        return primal_out, tangent_out
    """
    self.jvp = jvp

  def __call__(self, *args, **kwargs):
    if not self.jvp:
      msg = "No JVP defined for custom_jvp function {} using defjvp."
      raise AttributeError(msg.format(self.__name__)) from None
    args = _resolve_kwargs(self.fun, args, kwargs)
    if self.nondiff_argnums:
      dyn_argnums = [i for i in range(len(args)) if i not in self.nondiff_argnums]
      f_, dyn_args = argnums_partial(lu.wrap_init(self.fun), dyn_argnums, args)
      static_args = [args[i] for i in self.nondiff_argnums]
      jvp = _add_args(lu.wrap_init(self.jvp), static_args, left=True)
    else:
      f_, dyn_args = lu.wrap_init(self.fun), args
      jvp = lu.wrap_init(self.jvp)
    args_flat, in_tree = tree_flatten(dyn_args)
    flat_fun, out_tree1 = flatten_fun_nokwargs(f_, in_tree)
    flat_jvp, out_tree2 = _flatten_jvp(jvp, in_tree)
    out_flat = custom_jvp_call(flat_fun, *args_flat, jvp=flat_jvp)
    try: out_tree = out_tree1()
    except lu.StoreException: out_tree = out_tree2()
    return tree_unflatten(out_tree, out_flat)

@transformation_with_equal_aux
def _flatten_jvp(in_tree, *args):
  primals_in, tangents_in = split_list(args, [len(args) // 2])
  py_primals = tree_unflatten(in_tree, primals_in)
  py_tangents = tree_unflatten(in_tree, tangents_in)
  py_primals_out, py_tangents_out = yield (py_primals, py_tangents), {}
  primals_out, out_tree = tree_flatten(py_primals_out)
  tangents_out, out_tree2 = tree_flatten(py_tangents_out)
  if out_tree != out_tree2:
    msg = ("Custom JVP rule must produce primal and tangent outputs with equal "
           "container (pytree) structures, but got {} and {} respectively.")
    raise TypeError(msg.format(out_tree, out_tree2)) from None
  primal_avals_out = [raise_to_shaped(core.get_aval(x)) for x in primals_out]
  tangent_avals_out = [raise_to_shaped(core.get_aval(t)) for t in tangents_out]
  if primal_avals_out != tangent_avals_out:
    if len(primal_avals_out) == 1:
      (av1,), (av2,) = primal_avals_out, tangent_avals_out
      msg = ("Custom JVP rule must produce primal and tangent outputs with "
             "equal shapes and dtypes, but got {} and {} respectively.")
      raise TypeError(msg.format(av1.str_short(), av2.str_short()))
    else:
      msg = ("Custom JVP rule must produce primal and tangent outputs with "
             "equal shapes and dtypes, but got:\n{}")
      disagreements = (
          "  primal {} for tangent {}".format(av1.str_short(), av2.str_short())
          for av1, av2 in zip(primal_avals_out, tangent_avals_out) if av1 != av2)
      raise TypeError(msg.format('\n'.join(disagreements)))
  yield primals_out + tangents_out, out_tree

def _custom_deriv_call_bind(primitive, f, *args, **params):
  top_trace = core.find_top_trace(args)
  level = (core.trace_state.trace_stack.next_level(True)
           if top_trace is None else top_trace.level)
  if top_trace is None:
    with core.new_sublevel():
      return primitive.impl(f, *args, **params)
  else:
    tracers = map(top_trace.full_raise, args)
    outs = top_trace.process_call(primitive, f, tracers, params)
    outs = map(core.full_lower, outs)
  return outs

def _custom_call_impl(f, *args, **params):
  return f.call_wrapped(*args)

custom_jvp_call_p = core.Primitive('custom_jvp_call')
custom_jvp_call_p.multiple_results = True
custom_jvp_call = partial(_custom_deriv_call_bind, custom_jvp_call_p)
custom_jvp_call_p.def_custom_bind(custom_jvp_call)
custom_jvp_call_p.def_impl(_custom_call_impl)

def _custom_jvp_call_jvp(trace, call_primitive, fun, tracers, params):
  primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
  primals_in = map(core.full_lower, primals_in)
  tangents_in = map(ad.instantiate_zeros, primals_in, tangents_in)
  outs = params['jvp'].call_wrapped(*it.chain(primals_in, tangents_in))
  primals_out, tangents_out = split_list(outs, [len(outs) // 2])
  return map(partial(ad.JVPTracer, trace), primals_out, tangents_out)
ad.call_jvp_rules[custom_jvp_call_p] = _custom_jvp_call_jvp

def _custom_jvp_call_vmap(trace, call_primitive, fun, tracers, params):
  in_vals, in_dims = unzip2((t.val, t.batch_dim) for t in tracers)
  jvp = params['jvp']
  fun, out_dims = batching.batch_subtrace(fun, trace.master, in_dims)
  jvp, out_dims2 = batching.batch_subtrace(jvp, trace.master, in_dims * 2)
  out_vals = custom_jvp_call(fun, *in_vals, jvp=jvp)
  try: out_dims = out_dims()
  except lu.StoreException: out_dims = out_dims2()[:len(out_vals)]
  return [batching.BatchTracer(trace, v, d) for v, d in zip(out_vals, out_dims)]
batching.call_batching_rules[custom_jvp_call_p] = _custom_jvp_call_vmap

def _custom_jvp_call_partial_eval(trace, call_primitive, fun, tracers, params):
  return custom_jvp_call_jaxpr(fun, params['jvp'], *tracers)
pe.call_partial_eval_rules[custom_jvp_call_p] = _custom_jvp_call_partial_eval


def custom_jvp_call_jaxpr(fun, jvp, *args):
  in_avals = [raise_to_shaped(core.get_aval(x)) for x in args]
  jaxpr, consts = _initial_style_jaxpr(fun, in_avals)
  return custom_jvp_call_jaxpr_p.bind(*it.chain(consts, args), jaxpr=jaxpr,
                                      jvp=jvp, num_consts=len(consts))

def _custom_call_jaxpr_impl(*args, jaxpr, **kwargs):
  del kwargs
  return core.jaxpr_as_fun(jaxpr)(*args)

def _custom_call_jaxpr_abstract_eval(*args, jaxpr, **kwargs):
  del kwargs
  return jaxpr.out_avals

def _custom_jvp_call_jaxpr_jvp(primals, tangents, jaxpr, jvp, num_consts):
  _, primals = split_list(primals, [num_consts])
  zero_tangents, tangents = split_list(tangents, [num_consts])
  assert all(t is zero for t in zero_tangents)
  outs = jvp.call_wrapped(*(primals + tangents))
  primals_out, tangents_out = split_list(outs, [len(outs) // 2])
  return primals_out, tangents_out

def _custom_jvp_call_jaxpr_vmap(args, in_dims, jaxpr, jvp, num_consts):
  size, = {x.shape[d] for x, d in zip(args, in_dims)
           if d is not batching.not_mapped}
  args = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
          else x for x, d in zip(args, in_dims)]
  in_batched = [d is not batching.not_mapped for d in in_dims]
  del in_dims
  batched_jaxpr, out_batched = batching.batch_jaxpr(jaxpr, size, in_batched, False)
  out_dims = [0 if b else batching.not_mapped for b in out_batched]

  jvp_in_dims = [0 if b else batching.not_mapped for b in in_batched] * 2
  batched_jvp = batching.batch_fun(jvp, jvp_in_dims, lambda: out_dims * 2)

  batched_outs = custom_jvp_call_jaxpr_p.bind(
      *args, jaxpr=batched_jaxpr, jvp=batched_jvp, num_consts=num_consts)
  return batched_outs, out_dims

# If a (multi)linear function is defined with a custom jvp, then
# custom_jvp_call_jaxpr can appear in jaxprs to be transposed. We transpose it
# like a core.call.
def _custom_jvp_call_jaxpr_transpose(cts, *args, jaxpr, **kwargs):
  name = 'custom_jvp_call_jaxpr_linear'
  return ad.call_transpose(core.call_p, dict(name=name), jaxpr.jaxpr,
                           tuple(jaxpr.literals) + args, cts)

custom_jvp_call_jaxpr_p = core.Primitive('custom_jvp_call_jaxpr')
custom_jvp_call_jaxpr_p.multiple_results = True
custom_jvp_call_jaxpr_p.def_impl(_custom_call_jaxpr_impl)
custom_jvp_call_jaxpr_p.def_abstract_eval(_custom_call_jaxpr_abstract_eval)
ad.primitive_jvps[custom_jvp_call_jaxpr_p] = _custom_jvp_call_jaxpr_jvp
ad.primitive_transposes[custom_jvp_call_jaxpr_p] = _custom_jvp_call_jaxpr_transpose
batching.primitive_batchers[custom_jvp_call_jaxpr_p] = _custom_jvp_call_jaxpr_vmap
xla.initial_style_translations[custom_jvp_call_jaxpr_p] = \
    xla.lower_fun_initial_style(_custom_call_jaxpr_impl)


### VJPs

class custom_vjp:
  """Set up a JAX-transformable function for a custom VJP rule definition.

  This class is meant to be used as a function decorator. Instances are
  callables that behave similarly to the underlying function to which the
  decorator was applied, except when a reverse-mode differentiation
  transformation (like ``jax.grad``) is applied, in which case a custom
  user-supplied VJP rule function is used instead of tracing into and performing
  automatic differentiation of the underlying function's implementation. There
  is a single instance method, ``defvjp``, which defines the custom VJP rule.

  This decorator precludes the use of forward-mode automatic differentiation.

  For example:

    import jax.numpy as np

    @jax.custom_vjp
    def f(x, y):
      return np.sin(x) * y

    def f_fwd(x, y):
      return f(x, y), (np.cos(x), np.sin(x), y)

    def f_bwd(res, g):
      cos_x, sin_x, y = res
      return (cos_x * g * y, -sin_x * g)

    f.defvjp(f_fwd, f_bwd)

  For a more detailed introduction, see the tutorial_.

  .. _tutorial: https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
  """

  def __init__(self, fun, nondiff_argnums=()):
    self.fun = fun
    self.nondiff_argnums = nondiff_argnums
    self.fwd = None
    self.bwd = None
    update_wrapper(self, fun)

  def defvjp(self, fwd, bwd):
    """Define a custom VJP rule for the function represented by this instance.

    Args:
      fwd: a Python callable representing the forward pass of the custom VJP
        rule. When there are no ``nondiff_argnums``, the ``fwd`` function has
        the same input signature as the underlying primal function. It should
        return as output a pair, where the first element represents the primal
        output and the second element represents any "residual" values to store
        from the forward pass for use on the backward pass by the function
        ``bwd``. Input arguments and elements of the output pair may be arrays
        or nested tuples/lists/dicts thereof.
      bwd: a Python callable representing the backward pass of the custom VJP
        rule. When there are no ``nondiff_argnums``, the ``bwd`` function takes
        two arguments, where the first is the "residual" values produced on the
        forward pass by ``fwd``, and the second is the output cotangent with the
        same structure as the primal function output. The output of ``bwd`` must
        be a tuple of length equal to the number of arguments of the primal
        function, and the tuple elements may be arrays or nested
        tuples/lists/dicts thereof so as to match the structure of the primal
        input arguments.

    Returns:
      None.

    Example:

      import jax.numpy as np

      @jax.custom_vjp
      def f(x, y):
        return np.sin(x) * y

      def f_fwd(x, y):
        return f(x, y), (np.cos(x), np.sin(x), y)

      def f_bwd(res, g):
        cos_x, sin_x, y = res
        return (cos_x * g * y, -sin_x * g)

      f.defvjp(f_fwd, f_bwd)
    """
    self.fwd = fwd
    self.bwd = bwd

  def __call__(self, *args, **kwargs):
    if not self.fwd or not self.bwd:
      msg = "No VJP defined for custom_vjp function {} using defvjp."
      raise AttributeError(msg.format(self.__name__))
    args = _resolve_kwargs(self.fun, args, kwargs)
    if self.nondiff_argnums:
      dyn_argnums = [i for i in range(len(args)) if i not in self.nondiff_argnums]
      f_, dyn_args = argnums_partial(lu.wrap_init(self.fun), dyn_argnums, args)
      static_args = [args[i] for i in self.nondiff_argnums]
      fwd, _ = argnums_partial(lu.wrap_init(self.fwd), dyn_argnums, args)
      bwd = _add_args(lu.wrap_init(self.bwd), static_args, left=True)
    else:
      f_, dyn_args = lu.wrap_init(self.fun), args
      fwd, bwd = lu.wrap_init(self.fwd), lu.wrap_init(self.bwd)
    args_flat, in_tree = tree_flatten(dyn_args)
    flat_fun, out_tree = flatten_fun_nokwargs(f_, in_tree)
    flat_fwd, out_trees = _flatten_fwd(fwd, in_tree)
    flat_bwd = _flatten_bwd(bwd, in_tree, out_trees)
    out_flat = custom_vjp_call(flat_fun, *args_flat, fwd=flat_fwd, bwd=flat_bwd,
                               out_trees=out_trees)
    try: out_tree = out_tree()
    except lu.StoreException: out_tree, _ = out_trees()
    return tree_unflatten(out_tree, out_flat)

custom_vjp_call_p = core.Primitive('custom_vjp_call')
custom_vjp_call_p.multiple_results = True
custom_vjp_call = partial(_custom_deriv_call_bind, custom_vjp_call_p)
custom_vjp_call_p.def_custom_bind(custom_vjp_call)
custom_vjp_call_p.def_impl(_custom_call_impl)

@transformation_with_equal_aux
def _flatten_fwd(in_tree, *args):
  py_args = tree_unflatten(in_tree, args)
  py_outs, res = yield py_args, {}
  out, out_tree = tree_flatten(py_outs)
  res, res_tree = tree_flatten(res)
  yield res + out, (out_tree, res_tree)

@lu.transformation
def _flatten_bwd(in_tree, out_trees, *args):
  out_tree, res_tree = out_trees()
  res, cts_out = split_list(args, [res_tree.num_leaves])
  py_res = tree_unflatten(res_tree, res)
  py_cts_out = tree_unflatten(out_tree, cts_out)
  py_cts_in = yield (py_res, py_cts_out), {}
  cts_in, in_tree2 = tree_flatten(py_cts_in)
  if in_tree != in_tree2:
    msg = ("Custom VJP rule must produce an output with the same container "
           "(pytree) structure as the args tuple of the primal function, "
           "and in particular must produce a tuple of length equal to the "
           "number of arguments to the primal function, but got VJP output "
           "structure {} for primal input structure {}.")
    raise TypeError(msg.format(in_tree2, in_tree)) from None
  yield cts_in

def _custom_vjp_call_jvp(trace, call_primitive, fun, tracers, params):
  primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
  tangents_in = map(ad.instantiate_zeros, primals_in, tangents_in)
  fwd, bwd, out_trees = params['fwd'], params['bwd'], params['out_trees']
  res_and_primals_out = fwd.call_wrapped(*map(core.full_lower, primals_in))
  out_tree, res_tree = out_trees()
  res, primals_out = split_list(res_and_primals_out, [res_tree.num_leaves])
  avals_out = [raise_to_shaped(core.get_aval(x)) for x in primals_out]
  tangents_out = custom_lin_p.bind(
      *it.chain(res, tangents_in), num_res=res_tree.num_leaves, bwd=bwd,
      avals_out=avals_out)
  return map(partial(ad.JVPTracer, trace), primals_out, tangents_out)
ad.call_jvp_rules[custom_vjp_call_p] = _custom_vjp_call_jvp

def _custom_vjp_call_vmap(trace, call_primitive, fun, tracers, params):
  in_vals, in_dims = unzip2((t.val, t.batch_dim) for t in tracers)
  fwd, bwd, out_trees = params['fwd'], params['bwd'], params['out_trees']
  fun, out_dims = batching.batch_subtrace(fun, trace.master, in_dims)
  fwd, out_dims2 = batching.batch_subtrace(fwd, trace.master, in_dims)
  bwd = batching.batch_fun(bwd, out_dims2, in_dims)
  out_vals = custom_vjp_call(fun, *in_vals, fwd=fwd, bwd=bwd,
                              out_trees=out_trees)
  try: out_dims = out_dims()
  except lu.StoreException: out_dims = out_dims2()
  out_dims = out_dims[-len(out_vals) % len(out_dims):]
  return [batching.BatchTracer(trace, v, d) for v, d in zip(out_vals, out_dims)]
batching.call_batching_rules[custom_vjp_call_p] = _custom_vjp_call_vmap

def _custom_vjp_call_partial_eval(trace, call_primitive, fun, tracers, params):
  return custom_vjp_call_jaxpr(fun, params['fwd'], params['bwd'],
                                params['out_trees'], *tracers)
pe.call_partial_eval_rules[custom_vjp_call_p] = _custom_vjp_call_partial_eval


custom_lin_p = core.Primitive('custom_lin')
custom_lin_p.def_abstract_eval(lambda *_, avals_out, **__: avals_out)
custom_lin_p.multiple_results = True

def _raise_custom_vjp_error_on_jvp(*args, **kwargs):
  raise TypeError("can't apply forward-mode autodiff (jvp) to a custom_vjp "
                  "function.")
custom_lin_p.def_impl(_raise_custom_vjp_error_on_jvp)

def _custom_lin_transpose(cts_out, *invals, num_res, bwd, avals_out):
  res, _ = split_list(invals, [num_res])
  cts_out = map(ad.instantiate_zeros_aval, avals_out, cts_out)
  cts_in = bwd.call_wrapped(*(res + cts_out))
  cts_in_flat, in_tree = tree_flatten(cts_in)
  return [None] * num_res + cts_in_flat
ad.primitive_transposes[custom_lin_p] = _custom_lin_transpose


def custom_vjp_call_jaxpr(fun, fwd, bwd, out_trees, *args):
  in_avals = [raise_to_shaped(core.get_aval(x)) for x in args]
  jaxpr, consts = _initial_style_jaxpr(fun, in_avals)
  return custom_vjp_call_jaxpr_p.bind(
      *it.chain(consts, args), jaxpr=jaxpr, fwd=fwd, bwd=bwd,
      out_trees=out_trees, num_consts=len(consts))

def _custom_vjp_call_jaxpr_jvp(primals, tangents, jaxpr, fwd, bwd, out_trees,
                               num_consts):
  _, primals = split_list(primals, [num_consts])
  zero_tangents, tangents = split_list(tangents, [num_consts])
  assert all(t is zero for t in zero_tangents)
  tangents = map(ad.instantiate_zeros, primals, tangents)
  res_and_primals_out = fwd.call_wrapped(*primals)
  out_tree, res_tree = out_trees()
  res, primals_out = split_list(res_and_primals_out, [res_tree.num_leaves])
  avals_out = [raise_to_shaped(core.get_aval(x)) for x in primals_out]
  tangents_out = custom_lin_p.bind(
      *it.chain(res, tangents), num_res=res_tree.num_leaves, bwd=bwd,
      avals_out=avals_out)
  return primals_out, tangents_out

def _custom_vjp_call_jaxpr_vmap(args, in_dims, jaxpr, fwd, bwd, out_trees,
                                num_consts):
  size, = {x.shape[d] for x, d in zip(args, in_dims)
           if d is not batching.not_mapped}
  args = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
          else x for x, d in zip(args, in_dims)]
  in_batched = [d is not batching.not_mapped for d in in_dims]
  del in_dims
  batched_jaxpr, out_batched = batching.batch_jaxpr(jaxpr, size, in_batched, False)
  out_dims = [0 if b else batching.not_mapped for b in out_batched]

  fwd_in_dims = [0 if b else batching.not_mapped for b in in_batched]
  batched_fwd, fwd_out_dims = batching.batch_fun2(fwd, fwd_in_dims)
  batched_bwd = batching.batch_fun(bwd, fwd_out_dims, fwd_in_dims)

  batched_outs = custom_vjp_call_jaxpr_p.bind(
      *args, jaxpr=batched_jaxpr, fwd=batched_fwd, bwd=batched_bwd,
      out_trees=out_trees, num_consts=num_consts)
  return batched_outs, out_dims

custom_vjp_call_jaxpr_p = core.Primitive('custom_vjp_call_jaxpr')
custom_vjp_call_jaxpr_p.multiple_results = True
custom_vjp_call_jaxpr_p.def_impl(_custom_call_jaxpr_impl)
custom_vjp_call_jaxpr_p.def_abstract_eval(_custom_call_jaxpr_abstract_eval)
ad.primitive_jvps[custom_vjp_call_jaxpr_p] = _custom_vjp_call_jaxpr_jvp
batching.primitive_batchers[custom_vjp_call_jaxpr_p] = _custom_vjp_call_jaxpr_vmap
xla.initial_style_translations[custom_vjp_call_jaxpr_p] = \
    xla.lower_fun_initial_style(_custom_call_jaxpr_impl)
