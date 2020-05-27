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

from functools import partial, update_wrapper, reduce
import inspect
import itertools as it
import operator as op

from . import core
from . import linear_util as lu
from .tree_util import tree_flatten, tree_unflatten, tree_map, tree_multimap
from .util import safe_zip, safe_map, unzip2, split_list, curry
from .api_util import flatten_fun_nokwargs, argnums_partial, wrap_hashably
from .abstract_arrays import raise_to_shaped
from .ad_util import zero, stop_gradient_p
from .interpreters import partial_eval as pe
from .interpreters import ad
from .interpreters import batching
from .interpreters import xla
from .interpreters.batching import not_mapped, batch_jaxpr

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

def _add_args(f, extra_args, left):
  return _add_args_(f, tuple(map(wrap_hashably, extra_args)), left)

@lu.transformation
def _add_args_(extra_args, left, *args, **kwargs):
  extra_args = tuple([arg.val for arg in extra_args])
  args = (extra_args + args) if left else (args + extra_args)
  yield (yield args, kwargs)

def _memoize(thunk):
  cell = []
  saved_state = core.trace_state.copy()
  def memoized():
    if not cell:
      prev_state, core.trace_state = core.trace_state, saved_state
      try:
        cell.append(thunk())
      finally:
        core.trace_state = prev_state
    return cell[0]
  return memoized

def _initial_style_jaxpr(fun, in_avals):
  in_pvals = [pe.PartialVal.unknown(aval) for aval in in_avals]
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr(fun, in_pvals, instantiate=True,
                                               bottom=True, stage_out=False)
  assert not any(isinstance(c, core.Tracer) for c in consts)
  out_avals = map(raise_to_shaped, unzip2(out_pvals)[0])
  typed_jaxpr = core.TypedJaxpr(jaxpr, consts, in_avals, out_avals)
  return typed_jaxpr

def sum_tangents(x, *xs):
  return reduce(ad.add_tangents, xs, x)

def zeros_like_pytree(x):
  return tree_map(lambda _: zero, x)

def stop_gradient(x):
  return tree_map(_stop_gradient, x)

def _stop_gradient(x):
  if isinstance(x, core.Tracer) or core.valid_jaxtype(x):
    return stop_gradient_p.bind(x)
  else:
    return x


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

  For example::

    import jax.numpy as jnp

    @jax.custom_jvp
    def f(x, y):
      return jnp.sin(x) * y

    @f.defjvp
    def f_jvp(primals, tangents):
      x, y = primals
      x_dot, y_dot = tangents
      primal_out = f(x, y)
      tangent_out = jnp.cos(x) * x_dot * y - jnp.sin(x) * y_dot
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

    Example::

      import jax.numpy as jnp

      @jax.custom_jvp
      def f(x, y):
        return jnp.sin(x) * y

      @f.defjvp
      def f_jvp(primals, tangents):
        x, y = primals
        x_dot, y_dot = tangents
        primal_out = f(x, y)
        tangent_out = jnp.cos(x) * x_dot * y - jnp.sin(x) * y_dot
        return primal_out, tangent_out
    """
    self.jvp = jvp

  def defjvps(self, *jvps):
    """Convenience wrapper for defining JVPs for each argument separately.

    This convenience wrapper cannot be used together with ``nondiff_argnums``.

    Args:
      *jvps: a sequence of functions, one for each positional argument of the
        ``custom_jvp`` function. Each function takes as arguments the tangent
        value for the corresponding primal input, the primal output, and the
        primal inputs. See the example below.

    Returns:
      None.

    Example::

      @jax.custom_jvp
      def f(x, y):
        return jnp.sin(x) * y

      f.defjvps(lambda x_dot, primal_out, x, y: jnp.cos(x) * x_dot * y,
                lambda y_dot, primal_out, x, y: -jnp.sin(x) * y_dot)
    """
    if self.nondiff_argnums:
      raise TypeError("Can't use ``defjvps`` with ``nondiff_argnums``.")

    def jvp(primals, tangents):
      primal_out = self(*primals)
      zeros = zeros_like_pytree(primal_out)
      all_tangents_out = [jvp(t, primal_out, *primals) if jvp else zeros
                          for t, jvp in zip(tangents, jvps)]
      tangent_out = tree_multimap(sum_tangents, *all_tangents_out)
      return primal_out, tangent_out

    self.defjvp(jvp)

  def __call__(self, *args, **kwargs):
    if not self.jvp:
      msg = "No JVP defined for custom_jvp function {} using defjvp."
      raise AttributeError(msg.format(self.__name__))
    args = _resolve_kwargs(self.fun, args, kwargs)
    if self.nondiff_argnums:
      is_nondiff = [False] * len(args)
      for i in self.nondiff_argnums: is_nondiff[i] = True
      args = [stop_gradient(x) if b else x for b, x in zip(is_nondiff, args)]
      dyn_argnums = [i for i, b in enumerate(is_nondiff) if not b]
      f_, dyn_args = argnums_partial(lu.wrap_init(self.fun), dyn_argnums, args)
      static_args = [args[i] for i in self.nondiff_argnums]
      jvp = _add_args(lu.wrap_init(self.jvp), static_args, left=True)
    else:
      f_, dyn_args = lu.wrap_init(self.fun), args
      jvp = lu.wrap_init(self.jvp)
    args_flat, in_tree = tree_flatten(dyn_args)
    flat_fun, out_tree1 = flatten_fun_nokwargs(f_, in_tree)
    flat_jvp, out_tree2 = _flatten_jvp(jvp, in_tree)
    if core.trace_state.initial_style:
      out_flat = custom_jvp_call_jaxpr(flat_fun, flat_jvp, *args_flat)
      out_tree = out_tree1()
    else:
      out_flat = custom_jvp_call(flat_fun, flat_jvp, *args_flat)
      _, out_tree = lu.merge_linear_aux(out_tree1, out_tree2)
    return tree_unflatten(out_tree, out_flat)

@lu.transformation_with_aux
def _flatten_jvp(in_tree, *args):
  primals_in, tangents_in = split_list(args, [len(args) // 2])
  py_primals = tree_unflatten(in_tree, primals_in)
  py_tangents = tree_unflatten(in_tree, tangents_in)
  pair_out = yield (py_primals, py_tangents), {}
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    msg = ("Custom JVP rule must produce a pair (list or tuple of length two) "
           "representing primal and tangent outputs, got {}.")
    raise TypeError(msg.format(pair_out))
  py_primals_out, py_tangents_out = pair_out
  primals_out, out_tree = tree_flatten(py_primals_out)
  tangents_out, out_tree2 = tree_flatten(py_tangents_out)
  if out_tree != out_tree2:
    msg = ("Custom JVP rule must produce primal and tangent outputs with equal "
           "container (pytree) structures, but got {} and {} respectively.")
    raise TypeError(msg.format(out_tree, out_tree2))
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

def _custom_jvp_call_bind(prim, fun, jvp, *args):
  args = map(core.full_lower, args)
  top_trace = core.find_top_trace(args)
  level = (core.trace_state.trace_stack.next_level(True)
           if top_trace is None else top_trace.level)
  if top_trace is None:
    with core.new_sublevel():
      outs = prim.impl(fun, jvp, *args)
  else:
    tracers = map(top_trace.full_raise, args)
    outs = top_trace.process_custom_jvp_call(prim, fun, jvp, tracers)
  return map(core.full_lower, outs)

def _custom_jvp_call_impl(fun, _, *args):
  return fun.call_wrapped(*args)

custom_jvp_call_p = core.Primitive('custom_jvp_call')
custom_jvp_call_p.multiple_results = True
custom_jvp_call = partial(_custom_jvp_call_bind, custom_jvp_call_p)
custom_jvp_call_p.def_custom_bind(custom_jvp_call)
custom_jvp_call_p.def_impl(_custom_jvp_call_impl)


def custom_jvp_call_jaxpr(fun, jvp, *args):
  in_avals = [raise_to_shaped(core.get_aval(x)) for x in args]
  fun_jaxpr = _initial_style_jaxpr(fun, in_avals)
  jvp_jaxpr_thunk = _memoize(lambda: _initial_style_jaxpr(jvp, in_avals * 2))
  return custom_jvp_call_jaxpr_p.bind(*args, fun_jaxpr=fun_jaxpr,
                                      jvp_jaxpr_thunk=jvp_jaxpr_thunk)

def _custom_jvp_call_jaxpr_impl(*args, fun_jaxpr, **_):
  return core.jaxpr_as_fun(fun_jaxpr)(*args)

def _custom_jvp_call_jaxpr_abstract_eval(*_, fun_jaxpr, **__):
  return fun_jaxpr.out_avals

custom_jvp_call_jaxpr_p = core.Primitive('custom_jvp_call_jaxpr')
custom_jvp_call_jaxpr_p.multiple_results = True
custom_jvp_call_jaxpr_p.def_impl(_custom_jvp_call_jaxpr_impl)
custom_jvp_call_jaxpr_p.def_abstract_eval(_custom_jvp_call_jaxpr_abstract_eval)

def _custom_jvp_call_jaxpr_jvp(primals, tangents, *, fun_jaxpr, jvp_jaxpr_thunk):
  jvp_jaxpr = jvp_jaxpr_thunk()
  tangents = map(ad.instantiate_zeros, primals, tangents)
  outs = core.jaxpr_as_fun(jvp_jaxpr)(*primals, *tangents)
  return split_list(outs, [len(outs) // 2])
ad.primitive_jvps[custom_jvp_call_jaxpr_p] = _custom_jvp_call_jaxpr_jvp

def _custom_jvp_call_jaxpr_vmap(args, in_dims, *, fun_jaxpr, jvp_jaxpr_thunk):
  size, = {x.shape[d] for x, d in zip(args, in_dims) if d is not not_mapped}
  args = [batching.moveaxis(x, d, 0) if d is not not_mapped and d != 0
          else x for x, d in zip(args, in_dims)]
  num_out = len(fun_jaxpr.out_avals)

  in_batched = [d is not not_mapped for d in in_dims]
  batched_fun_jaxpr, out_batched = batch_jaxpr(fun_jaxpr, size, in_batched, False)
  out_dims1 = [0 if b else not_mapped for b in out_batched]
  out_dims2 = []

  @_memoize
  def batched_jvp_jaxpr_thunk():
    jvp_jaxpr = jvp_jaxpr_thunk()
    _, all_batched = batch_jaxpr(jvp_jaxpr, size, in_batched * 2, False)
    primals_batched, tangents_batched = split_list(all_batched, [num_out])
    out_batched = map(op.or_, primals_batched, tangents_batched)
    out_dims2.append([0 if b else not_mapped for b in out_batched])
    batched_jvp_jaxpr, _ = batch_jaxpr(jvp_jaxpr, size, in_batched * 2,
                                       out_batched * 2)
    return batched_jvp_jaxpr

  batched_outs = custom_jvp_call_jaxpr_p.bind(
      *args, fun_jaxpr=batched_fun_jaxpr, jvp_jaxpr_thunk=batched_jvp_jaxpr_thunk)
  out_dims = out_dims2[0] if out_dims2 else out_dims1
  return batched_outs, out_dims
batching.primitive_batchers[custom_jvp_call_jaxpr_p] = _custom_jvp_call_jaxpr_vmap

xla.initial_style_translations[custom_jvp_call_jaxpr_p] = \
    xla.lower_fun_initial_style(_custom_jvp_call_jaxpr_impl)

# If a (multi)linear function is defined with a custom jvp, then
# custom_jvp_call_jaxpr can appear in jaxprs to be transposed. Since it's
# already been linearized, we can drop the jvp rule.
def _custom_jvp_call_jaxpr_transpose(cts, *args, fun_jaxpr, jvp_jaxpr_thunk):
  del jvp_jaxpr_thunk
  return ad.backward_pass(fun_jaxpr.jaxpr, fun_jaxpr.literals, args, cts)
ad.primitive_transposes[custom_jvp_call_jaxpr_p] = _custom_jvp_call_jaxpr_transpose


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

  For example::

    import jax.numpy as jnp

    @jax.custom_vjp
    def f(x, y):
      return jnp.sin(x) * y

    def f_fwd(x, y):
      return f(x, y), (jnp.cos(x), jnp.sin(x), y)

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

    Example::

      import jax.numpy as jnp

      @jax.custom_vjp
      def f(x, y):
        return jnp.sin(x) * y

      def f_fwd(x, y):
        return f(x, y), (jnp.cos(x), jnp.sin(x), y)

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
      is_nondiff = [False] * len(args)
      for i in self.nondiff_argnums: is_nondiff[i] = True
      args = [stop_gradient(x) if b else x for b, x in zip(is_nondiff, args)]
      dyn_argnums = [i for i, b in enumerate(is_nondiff) if not b]
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
    if core.trace_state.initial_style:
      out_flat = custom_vjp_call_jaxpr(flat_fun, flat_fwd, flat_bwd,
                                       *args_flat, out_trees=out_trees)
      out_tree = out_tree()
    else:
      out_flat = custom_vjp_call(flat_fun, flat_fwd, flat_bwd,
                                 *args_flat, out_trees=out_trees)
      fst, aux = lu.merge_linear_aux(out_tree, out_trees)
      out_tree = aux if fst else aux[0]
    return tree_unflatten(out_tree, out_flat)

@lu.transformation_with_aux
def _flatten_fwd(in_tree, *args):
  py_args = tree_unflatten(in_tree, args)
  pair_out = yield py_args, {}
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    msg = ("Custom VJP fwd function must produce a pair (list or tuple of "
           "length two) representing primal outputs and residuals (values "
           "stored from the forward pass for use on the backward pass), "
           "got {}.")
    raise TypeError(msg.format(pair_out))
  py_outs, res = pair_out
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

def _custom_vjp_call_bind(prim, fun, fwd, bwd, *args, out_trees):
  args = map(core.full_lower, args)
  top_trace = core.find_top_trace(args)
  level = (core.trace_state.trace_stack.next_level(True)
           if top_trace is None else top_trace.level)
  if top_trace is None:
    with core.new_sublevel():
      outs = prim.impl(fun, fwd, bwd, *args, out_trees=out_trees)
  else:
    tracers = map(top_trace.full_raise, args)
    outs = top_trace.process_custom_vjp_call(prim, fun, fwd, bwd, tracers,
                                             out_trees=out_trees)
    outs = map(core.full_lower, outs)
  return map(core.full_lower, outs)

def _custom_vjp_call_impl(fun, fwd, bwd, *args, out_trees):
  del fwd, bwd, out_trees  # Unused.
  return fun.call_wrapped(*args)

custom_vjp_call_p = core.Primitive('custom_vjp_call')
custom_vjp_call_p.multiple_results = True
custom_vjp_call = partial(_custom_vjp_call_bind, custom_vjp_call_p)
custom_vjp_call_p.def_custom_bind(custom_vjp_call)
custom_vjp_call_p.def_impl(_custom_vjp_call_impl)

def custom_vjp_call_jaxpr(fun, fwd, bwd, *args, out_trees):
  in_avals = [raise_to_shaped(core.get_aval(x)) for x in args]
  fun_jaxpr = _initial_style_jaxpr(fun, in_avals)
  fwd_jaxpr_thunk = _memoize(lambda: _initial_style_jaxpr(fwd, in_avals))
  return custom_vjp_call_jaxpr_p.bind(*args, fun_jaxpr=fun_jaxpr,
                                      fwd_jaxpr_thunk=fwd_jaxpr_thunk, bwd=bwd,
                                      out_trees=out_trees)

def _custom_vjp_call_jaxpr_impl(*args, fun_jaxpr, **_):
  return core.jaxpr_as_fun(fun_jaxpr)(*args)

def _custom_vjp_call_jaxpr_abstract_eval(*_, fun_jaxpr, **__):
  return fun_jaxpr.out_avals

custom_vjp_call_jaxpr_p = core.Primitive('custom_vjp_call_jaxpr')
custom_vjp_call_jaxpr_p.multiple_results = True
custom_vjp_call_jaxpr_p.def_impl(_custom_vjp_call_jaxpr_impl)
custom_vjp_call_jaxpr_p.def_abstract_eval(_custom_vjp_call_jaxpr_abstract_eval)

def _custom_vjp_call_jaxpr_jvp(primals, tangents, *, fun_jaxpr, fwd_jaxpr_thunk,
                               bwd, out_trees):
  tangents = map(ad.instantiate_zeros, primals, tangents)
  fwd_jaxpr = fwd_jaxpr_thunk()
  out_tree, res_tree = out_trees()
  res_and_primals_out = core.jaxpr_as_fun(fwd_jaxpr)(*primals)
  res, primals_out = split_list(res_and_primals_out, [res_tree.num_leaves])
  avals_out = [raise_to_shaped(core.get_aval(x)) for x in primals_out]
  tangents_out = ad.custom_lin_p.bind(
      *res, *tangents, num_res=res_tree.num_leaves, bwd=bwd, avals_out=avals_out)
  return primals_out, tangents_out
ad.primitive_jvps[custom_vjp_call_jaxpr_p] = _custom_vjp_call_jaxpr_jvp

def _custom_vjp_call_jaxpr_vmap(args, in_dims, *, fun_jaxpr, fwd_jaxpr_thunk,
                                bwd, out_trees):
  size, = {x.shape[d] for x, d in zip(args, in_dims) if d is not not_mapped}
  args = [batching.moveaxis(x, d, 0) if d is not not_mapped and d != 0
          else x for x, d in zip(args, in_dims)]

  in_batched = [d is not not_mapped for d in in_dims]
  batched_fun_jaxpr, out_batched = batch_jaxpr(fun_jaxpr, size, in_batched, False)
  out_dims1 = [0 if b else not_mapped for b in out_batched]
  out_dims2 = []

  @_memoize
  def batched_fwd_jaxpr_thunk():
    fwd_jaxpr = fwd_jaxpr_thunk()
    batched_fwd_jaxpr, out_batched = batch_jaxpr(fwd_jaxpr, size, in_batched, False)
    out_dims2.append([0 if b else not_mapped for b in out_batched])
    return batched_fwd_jaxpr

  fwd_in_dims = [0 if b else not_mapped for b in in_batched]
  fwd_out_dims = lambda: out_dims2[0]
  batched_bwd = batching.batch_fun(bwd, fwd_out_dims, fwd_in_dims, sum_match=True)

  batched_outs = custom_vjp_call_jaxpr_p.bind(
      *args, fun_jaxpr=batched_fun_jaxpr,
      fwd_jaxpr_thunk=batched_fwd_jaxpr_thunk, bwd=batched_bwd,
      out_trees=out_trees)
  out_dims = out_dims2[0] if out_dims2 else out_dims1
  return batched_outs, out_dims
batching.primitive_batchers[custom_vjp_call_jaxpr_p] = _custom_vjp_call_jaxpr_vmap

xla.initial_style_translations[custom_vjp_call_jaxpr_p] = \
    xla.lower_fun_initial_style(_custom_vjp_call_jaxpr_impl)
