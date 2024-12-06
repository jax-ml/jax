# Copyright 2024 The JAX Authors.
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

from __future__ import annotations

from collections.abc import Sequence, Callable
from functools import partial, update_wrapper
from typing import Any, Generic, TypeVar

from jax._src import ad_util
from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util
from jax._src.errors import UnexpectedTracerError
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import tree_flatten, tree_unflatten, tree_leaves
from jax._src.util import (merge_lists, safe_map, safe_zip, split_list, unzip2,
                           Unhashable)

map = safe_map
zip = safe_zip

ReturnT = TypeVar("ReturnT")
T = TypeVar("T")


class custom_primitive(core.Primitive, Generic[ReturnT]):
  fun: Callable[..., ReturnT]
  static_argnums: Sequence[int]
  instantiate_zeros: bool
  multiple_results = True

  vmap: Callable[..., ReturnT] | None = None
  jvp: Callable[..., tuple[ReturnT, Any]] | None = None
  transpose: Callable[..., Any] | None = None

  def __init__(self, fun: Callable[..., ReturnT], *,
               name: str | None = None,
               static_argnums: Sequence[int] = (),
               instantiate_zeros: bool = False) -> None:
    update_wrapper(self, fun)
    self.fun = fun
    self.static_argnums = static_argnums
    self.instantiate_zeros = instantiate_zeros
    if name is None:
      name = getattr(fun, "__name__", str(fun))
    super().__init__(name)

    self.def_effectful_abstract_eval(_custom_primitive_abstract_eval)
    mlir.register_lowering(self, _custom_primitive_lowering)
    ad.primitive_transposes[self] = partial(_custom_primitive_transpose, self)

  def def_vmap(self, fun: Callable[..., ReturnT]) -> Callable[..., ReturnT]:
    self.vmap = fun
    return fun

  def def_jvp(
      self,
      jvp: Callable[..., tuple[ReturnT, T]],
  ) -> Callable[..., tuple[ReturnT, T]]:
    self.jvp = jvp
    return jvp

  def def_transpose(self, transpose: Callable[..., T]) -> Callable[..., T]:
    self.transpose = transpose
    return transpose

  @traceback_util.api_boundary
  def __call__(self, *args: Any, **kwargs: Any) -> ReturnT:
    args = api_util.resolve_kwargs(self.fun, args, kwargs)

    if self.static_argnums:
      for i in self.static_argnums:
        _check_for_tracers(args[i])
      static_argnums = set(self.static_argnums)
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      fun, dyn_args = api_util.argnums_partial(
          lu.wrap_init(self.fun), dyn_argnums, args,
          require_static_args_hashable=False)
      static_args = [args[i] for i in sorted(self.static_argnums)]
    else:
      fun, dyn_args = lu.wrap_init(self.fun), args
      static_args = []

    args_flat, tree_in = tree_flatten(dyn_args)
    flat_fun, fun_out_type = _flatten_fun_nokwargs(fun, tree_in)

    if self.vmap is not None:
      rule_name = getattr(self.vmap, "__name__", str(self.vmap))
      vmap = _add_args(lu.wrap_init(self.vmap), static_args)
      flat_vmap, vmap_out_type = _flatten_vmap(vmap, self.name, rule_name,
                                               tree_in, fun_out_type)
    else:
      flat_vmap = None
      vmap_out_type = None

    if self.jvp is not None:
      rule_name = getattr(self.jvp, "__name__", str(self.jvp))
      jvp = _add_args(lu.wrap_init(self.jvp), static_args)
      flat_jvp, jvp_out_type = _flatten_jvp(jvp, self.name, rule_name, tree_in,
                                            fun_out_type)
    else:
      flat_jvp = None
      jvp_out_type = None

    if self.transpose is not None:
      rule_name = getattr(self.transpose, "__name__", str(self.transpose))
      transpose = _add_args(lu.wrap_init(self.transpose), static_args)
      flat_transpose = _flatten_transpose(transpose, self.name, rule_name,
                                          tree_in, fun_out_type)
    else:
      flat_transpose = None

    out_flat = self.bind(
        *args_flat, fun=flat_fun, vmap=flat_vmap, jvp=flat_jvp,
        transpose=flat_transpose, instantiate_zeros=self.instantiate_zeros)

    _, (out_tree, _) = merge_linear_aux_multi(fun_out_type, vmap_out_type,
                                              jvp_out_type)

    return tree_unflatten(out_tree, out_flat)

  def impl(self, *_, **__):
    raise NotImplementedError

  def bind_with_trace(self, trace, args, params):
    return trace.process_custom_primitive(self, args, **params)

  def get_bind_params(self, params):
    new_params = dict(params)
    num_consts = new_params.pop("num_consts")
    fun_jaxpr = new_params.pop("fun_jaxpr")
    new_params["fun"] = lu.wrap_init(core.jaxpr_as_fun(fun_jaxpr))
    new_params["jvp"] = _lift_jvp_jaxpr_thunk(
        num_consts, new_params.pop("jvp_jaxpr_thunk"))

    return [], new_params


def _check_for_tracers(x):
  for leaf in tree_leaves(x):
    if isinstance(leaf, core.Tracer):
      raise UnexpectedTracerError(
          "Found a JAX Tracer object passed as an argument to a custom "
          "primitive in a position indicated by static_argnums as static. "
          "Tracers cannot be passed as static arguments to custom primitives; "
          "instead, static_argnums should only be used for arguments that "
          "can't be or contain JAX tracers, e.g. function-valued arguments. "
          "In particular, array-valued arguments should typically not be "
          "indicated as static_argnums."
      )


def merge_linear_aux_multi(*aux):
  for i, a in enumerate(aux):
    if a is None:
      continue
    try:
      out = a()
    except lu.StoreException:
      pass
    else:
      for b in aux[i + 1:]:
        if b is None:
          continue
        try:
          b()
        except lu.StoreException:
          pass
        else:
          raise lu.StoreException("multiple stores occupied")
      return i, out
  raise lu.StoreException("none of the stores are occupied")


def _lift_jvp_jaxpr_thunk(num_consts, jvp_jaxpr_thunk):
  if jvp_jaxpr_thunk is None:
    return None
  @lu.wrap_init
  def jvp(*args):
    primals, tangents = split_list(args, [len(args) // 2])
    _, primals = split_list(primals, [num_consts])
    _, tangents = split_list(tangents, [num_consts])
    jaxpr, consts = jvp_jaxpr_thunk()
    return core.eval_jaxpr(jaxpr, consts, *primals, *tangents)
  return jvp


def _add_args(f, extra_args):
  if extra_args:
    return _add_args_(f, tuple(Unhashable(arg) for arg in extra_args))
  else:
    return f


@lu.transformation
def _add_args_(extra_args, *args, **kwargs):
  extra_args = tuple(arg.val for arg in extra_args)
  all_args = (extra_args + args)
  yield (yield all_args, kwargs)


@lu.transformation_with_aux
def _flatten_fun_nokwargs(in_tree, *args_flat):
  py_args = tree_unflatten(in_tree, args_flat)
  ans = yield py_args, {}
  ans_flat, ans_tree = tree_flatten(ans)
  ans_avals = [core.get_aval(x) for x in ans_flat]
  yield ans_flat, (ans_tree, ans_avals)


@partial(lu.transformation_with_aux, use_eq_store=True)
def _flatten_vmap(name, rule_name, tree_in, maybe_out_type, axis_size, *args):
  batched_in, args = split_list(args, [len(args) // 2])
  py_batched_in = tree_unflatten(tree_in, batched_in)
  py_args = tree_unflatten(tree_in, args)
  pair_out = yield (axis_size, py_batched_in, *py_args), {}
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    raise TypeError(
        f"The vmap rule {rule_name} for custom primitive {name} must produce a "
        "pair (list or tuple of length two) representing the batched outputs "
        f"and the output batching pattern, but got {pair_out}."
    )
  py_out, py_batched_out = pair_out
  out, tree_out = tree_flatten(py_out)
  batched_out, tree_batched_out = tree_flatten(py_batched_out)
  if tree_out != tree_batched_out:
    raise TypeError(
        f"The vmap rule {rule_name} for custom primitive {name} must produce "
        "value and batching outputs with equal pytree structures, but got "
        f"{tree_out} and {tree_batched_out} respectively."
    )

  avals_out = [core.mapped_aval(axis_size, 0, core.get_aval(x))
               if b else core.get_aval(x)
               for x, b in zip(out, batched_out)]
  try:
    out_type_ = maybe_out_type()
  except lu.StoreException:
    pass  # This means that the primal function hasn't been executed
  else:
    tree_out_, avals_out_ = out_type_
    ty_tree = tree_unflatten(tree_out, [a.str_short() for a in avals_out])
    orig_ty_tree = tree_unflatten(tree_out_, [a.str_short() for a in avals_out_])
    if tree_out != tree_out_:
      raise TypeError(
          f"The vmap rule {rule_name} for custom primitive {name} must produce a "
          "pair (list or tuple of length two) where the first element represents "
          "the primal output. This output must have the same pytree structure as "
          "the primal, but the output's first element had pytree structure:\n"
          f"""    {str(ty_tree).replace("'", "")}\n"""
          "while the primal function had output pytree structure:\n"
          f"""    {str(orig_ty_tree).replace("'", "")}."""
      )
    if not all(map(core.typematch, avals_out, avals_out_)):
      raise TypeError(
          f"The vmap rule {rule_name} for custom primitive {name} must produce a "
          "pair (list or tuple of length two) where the first element represents "
          "the primal output. This output must have leaves of the same shape and "
          "dtype as the primal, but the output's first element had the following "
          "shapes and dtypes:\n"
          f"""    {str(ty_tree).replace("'", "")}\n"""
          "while the primal function had the following output shapes and dtypes:\n"
          f"""    {str(orig_ty_tree).replace("'", "")}"""
      )

  yield (*out, *batched_out), (tree_out, avals_out)


@partial(lu.transformation_with_aux, use_eq_store=True)
def _flatten_jvp(name, rule_name, tree_in, maybe_out_type, *args):
  primals_in, tangents_in = split_list(args, [len(args) // 2])
  py_primals = tree_unflatten(tree_in, primals_in)
  py_tangents = tree_unflatten(tree_in, tangents_in)
  pair_out = yield (py_primals, py_tangents), {}
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    raise TypeError(
        f"The jvp rule {rule_name} for custom primitive {name} must produce a "
        "pair (list or tuple of length two) representing primal and tangent "
        f"outputs, but got {pair_out}."
    )
  py_primals_out, py_tangents_out = pair_out

  primals_out, tree_prim = tree_flatten(py_primals_out)
  tangents_out, tree_tan = tree_flatten(py_tangents_out)
  if tree_prim != tree_tan:
    raise TypeError(
        f"The jvp rule {rule_name} for custom primitive {name} must produce "
        "primal and tangent outputs with equal pytree structures, but got "
        f"{tree_prim} and {tree_tan} respectively."
    )

  primal_avals_out = map(core.get_aval, primals_out)
  expected_tangent_avals_out = [
    core.get_aval(x).strip_weak_type().to_tangent_aval()
    for x in primals_out]
  tangent_avals_out = [core.get_aval(t).strip_weak_type()
                       if type(t) is not ad_util.SymbolicZero
                       else t.aval.strip_weak_type()
                       for t in tangents_out]
  if expected_tangent_avals_out != tangent_avals_out:
    disagreements = "\n".join(
        f"  primal {av1.str_short()} with tangent {av2.str_short()}, expecting "
        f"tangent {av3.str_short()}"
        for av1, av2, av3 in zip(primal_avals_out, tangent_avals_out,
                                 expected_tangent_avals_out) if av2 != av3)
    raise TypeError(
        f"The jvp rule {rule_name} for custom primitive {name} must produce "
        "primal and tangent outputs with corresponding shapes and dtypes, but "
        f"{rule_name} returned:\n{disagreements}"
    )

  try:
    out_type_ = maybe_out_type()
  except lu.StoreException:
    pass  # This means that the primal function hasn't been executed
  else:
    tree_out, avals_out = out_type_
    prim_ty_tree = tree_unflatten(tree_prim, [a.str_short()
                                              for a in primal_avals_out])
    orig_ty_tree = tree_unflatten(tree_out, [a.str_short() for a in avals_out])
    if tree_prim != tree_out:
      raise TypeError(
          f"The jvp rule {rule_name} for custom primitive {name} must produce a "
          "pair (list or tuple of length two) where the first element represents "
          "the primal output. This output must have the same pytree structure as "
          "the primal, but the output's first element had pytree structure:\n"
          f"""    {str(prim_ty_tree).replace("'", "")}\n"""
          "while the primal function had output pytree structure:\n"
          f"""    {str(orig_ty_tree).replace("'", "")}."""
      )
    if not all(map(core.typematch, primal_avals_out, avals_out)):
      raise TypeError(
          f"The jvp rule {rule_name} for custom primitive {name} must produce a "
          "pair (list or tuple of length two) where the first element represents "
          "the primal output. This output must have leaves of the same shape and "
          "dtype as the primal, but the output's first element had the following "
          "shapes and dtypes:\n"
          f"""    {str(prim_ty_tree).replace("'", "")}\n"""
          "while the primal function had the following output shapes and dtypes:\n"
          f"""    {str(orig_ty_tree).replace("'", "")}"""
      )

  yield (*primals_out, *tangents_out), (tree_prim, primal_avals_out)


@lu.transformation
def _flatten_transpose(name, rule_name, tree_in, maybe_out_type, *args):
  pass


def _custom_primitive_abstract_eval(*args, fun_jaxpr, **_):
  del args  # unused
  return fun_jaxpr.out_avals, fun_jaxpr.effects


def _custom_primitive_lowering(ctx, *args, fun_jaxpr, **_):
  consts = mlir._ir_consts(fun_jaxpr.consts)
  out, tokens = mlir.jaxpr_subcomp(ctx.module_context, fun_jaxpr.jaxpr,
                                   ctx.name_stack, ctx.tokens_in, consts,
                                   *args, dim_var_values=ctx.dim_var_values)
  ctx.set_tokens_out(tokens)
  return out


def _custom_primitive_transpose(primitive, cts, *args, transpose, **params):
  if transpose is None:
    raise NotImplementedError(
        f"Transpose rule for custom primitive '{primitive}' not implemented")
  assert 0


def _eval_process_custom_primitive(trace, primitive, tracers, *, fun, **_):
  del trace, primitive  # unused
  return fun.call_wrapped(*tracers)

core.EvalTrace.process_custom_primitive = _eval_process_custom_primitive


def _pe_process_custom_primitive(trace, primitive, tracers, *, fun, jvp,
                                 **params):
  tracers = map(trace.to_jaxpr_tracer, tracers)
  if all(t.is_known() for t in tracers):
    with core.set_current_trace(trace.parent_trace):
      vals = [t.pval[1] for t in tracers]
      return primitive.bind(*vals, fun=fun, jvp=jvp, **params)

  tracers = map(trace.instantiate_const_abstracted, tracers)
  in_knowns, in_avals, () = pe.partition_pvals([t.pval for t in tracers])
  fun = pe.trace_to_subjaxpr_nounits(fun, trace, True)
  fun, aux = pe.partial_eval_wrapper_nounits(fun, (*in_knowns,), (*in_avals,))
  with core.set_current_trace(trace.parent_trace):
    out_flat = primitive.bind(fun=fun, jvp=jvp, **params)
  out_knowns, out_avals, jaxpr, env = aux()
  out_consts, res = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
  res_tracers = map(trace.new_instantiated_const, res)
  env_tracers = map(trace.to_jaxpr_tracer, env)
  out_tracers = [pe.JaxprTracer(trace, pe.PartialVal.unknown(a), None)
                for a in out_avals]
  closed_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))

  if jvp is not None:
    tangent_avals = [t.to_tangent_aval() for t in in_avals]
    @pe._memoize
    def jvp_jaxpr_thunk():
      jvp_ = pe.trace_to_subjaxpr_nounits(jvp, trace, True)
      jvp_, aux = pe.partial_eval_wrapper_nounits(
          jvp_, (*in_knowns, *in_knowns), (*in_avals, *tangent_avals))
      out_flat = jvp_.call_wrapped()
      *_, jaxpr, env = aux()
      _, res = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
      converted_jaxpr = pe.convert_envvars_to_constvars(jaxpr, len(env))
      return converted_jaxpr, (*res, *env)
  else:
    jvp_jaxpr_thunk = None

  new_params = dict(fun_jaxpr=closed_jaxpr, jvp_jaxpr_thunk=jvp_jaxpr_thunk,
                    num_consts=len(res) + len(env), **params)
  name_stack = trace._current_truncated_name_stack()
  source = source_info_util.current().replace(name_stack=name_stack)
  eqn = pe.new_eqn_recipe((*res_tracers, *env_tracers, *tracers),
                          out_tracers, primitive, new_params, jaxpr.effects,
                          source)
  for t in out_tracers:
    t.recipe = eqn
  return merge_lists(out_knowns, out_tracers, out_consts)

pe.JaxprTrace.process_custom_primitive = _pe_process_custom_primitive


def _staging_process_custom_primitive(trace, primitive, tracers, fun, jvp,
                                      **params):
  tracers = map(trace.to_jaxpr_tracer, tracers)
  in_avals = [t.aval for t in tracers]
  fun_jaxpr, out_avals, consts, () = pe.trace_to_jaxpr_dynamic(fun, in_avals)
  closed_fun_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(fun_jaxpr))

  if jvp is not None:
    tangent_avals = [t.to_tangent_aval() for t in in_avals]
    @pe._memoize
    def jvp_jaxpr_thunk():
      in_avals_ = (*in_avals, *tangent_avals)
      jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(jvp, in_avals_)
      return jaxpr, consts
  else:
    jvp_jaxpr_thunk = None

  out_tracers = [pe.DynamicJaxprTracer(trace, a) for a in out_avals]
  invars = map(trace.getvar, tracers)
  constvars = map(trace.getvar, map(trace.to_jaxpr_tracer, consts))
  outvars = map(trace.makevar, out_tracers)
  eqn = pe.new_jaxpr_eqn([*constvars, *invars], outvars, primitive,
                         dict(num_consts=len(consts),
                              fun_jaxpr=closed_fun_jaxpr,
                              jvp_jaxpr_thunk=jvp_jaxpr_thunk,
                              **params),
                      fun_jaxpr.effects,
                      source_info_util.current())
  trace.frame.add_eqn(eqn)
  return out_tracers

pe.DynamicJaxprTrace.process_custom_primitive = _staging_process_custom_primitive


def _batch_process_custom_primitive(trace, primitive, tracers, *, vmap, **_):
  if vmap is None:
    raise NotImplementedError(
        f"Batching rule for custom primitive '{primitive}' not implemented")
  in_vals, in_dims = unzip2(map(trace.to_batch_info, tracers))
  axis_size, = {x.shape[d] for x, d in zip(in_vals, in_dims)
                if d is not batching.not_mapped}
  in_vals = [x if d is batching.not_mapped else util.moveaxis(x, d, 0)
             for x, d in zip(in_vals, in_dims)]
  in_batched = [d is not batching.not_mapped for d in in_dims]

  # Clear the linear_util caches because sometimes we need to call the vmap
  # rule multiple times, for example when inside of control flow.
  for store in vmap.stores:
    store and store.reset()

  with core.set_current_trace(trace.parent_trace):
    out = vmap.call_wrapped(axis_size, *in_batched, *in_vals)

  out_vals, out_batched = split_list(out, [len(out) // 2])
  out_dims = [0 if b else batching.not_mapped for b in out_batched]
  src = source_info_util.current()
  return [batching.BatchTracer(trace, v, d, src)
          for v, d in zip(out_vals, out_dims)]

batching.BatchTrace.process_custom_primitive = _batch_process_custom_primitive


def _jvp_process_custom_primitive(trace, primitive, tracers, *, jvp,
                                  instantiate_zeros, **params):
  if jvp is None:
    raise NotImplementedError(
        f"JVP rule for custom primitive '{primitive}' not implemented")
  primals_in, tangents_in = unzip2(map(trace.to_primal_tangent_pair, tracers))
  if all(type(t) is ad_util.Zero for t in tangents_in):
    return primitive.bind_with_trace(trace.parent_trace, primals_in, jvp=jvp,
                                     **params)
  with core.set_current_trace(trace.parent_trace):
    if instantiate_zeros:
      tangents_in = map(ad.instantiate_zeros, tangents_in)
    else:
      tangents_in = map(ad.replace_internal_symbolic_zeros, tangents_in)
    outs = jvp.call_wrapped(*primals_in, *tangents_in)
  primals_out, tangents_out = split_list(outs, [len(outs) // 2])
  tangents_out = map(ad.replace_rule_output_symbolic_zeros, tangents_out)
  return map(partial(ad.maybe_jvp_tracer, trace), primals_out, tangents_out)

ad.JVPTrace.process_custom_primitive = _jvp_process_custom_primitive
