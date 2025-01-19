# Copyright 2025 The JAX Authors.
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

from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax._src import api_util
from jax._src import core
from jax._src import custom_api_util
from jax._src import errors
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe

source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


@custom_api_util.register_custom_decorator_type
class custom_dce:
  """Customize the DCE behavior of a JAX-transformable function.

  JAX uses dead code elimination (DCE) to remove unused computations from a
  JAX program. This typically works transparently when the program is
  completely specified by known JAX operations, but opaque kernels like calls
  to :py:func:`~jax.experimental.pallas.pallas_call` or
  :py:func:`~jax.ffi.ffi_call`, for example, may cause problems.

  This decorator allows users to customize the DCE behavior of a function by
  defining a custom DCE rule. For a ``custom_dce`` wrapped function
  ``f(*args)``, the signature of the DCE rule is ``dce_rule(used_outs, *args)``
  where ``used_outs`` is a Pytree with the same structure as the output of
  ``f``, and each leaf is is a ``bool`` indicating which outputs should
  be computed. The remaining arguments ``*args`` are the original arguments to
  ``f``. The rule ``dce_rule`` should return a Pytree with only the outputs
  that were flagged as used in ``used_outs``.

  For example::

    >>> @jax.experimental.custom_dce.custom_dce
    ... def f(x, y):
    ...   return jnp.sin(x) * y, x * jnp.sin(y)
    ...
    >>> @f.def_dce
    ... def f_dce_rule(used_outs, x, y):
    ...   outs = []
    ...   if used_outs[0]:
    ...     outs.append(jnp.sin(x) * y)
    ...   if used_outs[1]:
    ...     outs.append(x * jnp.sin(y))
    ...   return outs

  In this example, ``used_outs`` is a ``tuple`` with two ``bool``s indicating
  which outputs are required. The DCE rule returns only the required
  outputs.

  If the ``static_argnums`` argument is provided to ``custom_dce``, the
  indicated arguments are treated as static when the function is traced, and
  they will be moved to the front when calling the DCE rule. For example, if
  ``fun`` takes 2 arguments ``fun(x, y)``, and ``static_argnums`` is ``(1,)``,
  then the DCE rule will be called as ``dce_rule(y, used_outs, x)``.
  """

  fun: Callable[..., Any]
  static_argnums: Sequence[int]
  dce_rule: Callable[..., Any] | None

  def __init__(
      self, fun: Callable[..., Any], *, static_argnums: Sequence[int] = ()
  ):
    functools.update_wrapper(self, fun)
    self.fun = fun
    self.static_argnums = static_argnums
    self.dce_rule = None

  __getattr__ = custom_api_util.forward_attr

  def def_dce(
      self,
      dce_rule: Callable[..., Any],
  ) -> Callable[..., Any]:
    """Define a custom DCE rule for this function.

    Args:
      dce_rule: A function that takes (a) any arguments indicated as static
        using ``static_argnums``, (b) a Pytree of ``bool``s (``used_outs``)
        indicating which outputs should be computed, and (c) the rest of the
        (non-static) arguments to the original function. The rule should return
        a Pytree with only the outputs that were flagged as used in
        ``used_outs``.
    """
    self.dce_rule = dce_rule
    return dce_rule

  @traceback_util.api_boundary
  def __call__(self, *args, **kwargs):
    fun_name = util.fun_name(self.fun)
    if self.dce_rule is None:
      raise AttributeError(
          f"No DCE rule defined for custom_dce function {fun_name} using "
          "def_dce."
      )
    rule_name = util.fun_name(self.dce_rule)
    args = api_util.resolve_kwargs(self.fun, args, kwargs)
    if self.static_argnums:
      static_argnums = set(self.static_argnums)
      for i in static_argnums:
        check_for_tracers(args[i])
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      fun, dyn_args = api_util.argnums_partial(
          lu.wrap_init(self.fun),
          dyn_argnums,
          args,
          require_static_args_hashable=False,
      )
      static_args = [args[i] for i in self.static_argnums]
      dce_rule = api_util.prepend_static_args(
          lu.wrap_init(self.dce_rule), static_args
      )
    else:
      fun = lu.wrap_init(self.fun)
      dce_rule = lu.wrap_init(self.dce_rule)
      dyn_args = args

    args_flat, in_tree = tree_util.tree_flatten(dyn_args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    in_avals = [core.get_aval(x) for x in args_flat]

    @pe._memoize
    def dce_jaxpr_thunk(*used_outs: bool):
      for store in dce_rule.stores:
        if store:
          store.reset()
      flat_rule, rule_out_tree = flatten_dce_rule(
          dce_rule, fun_name, rule_name, used_outs, in_tree, out_tree()
      )
      assert self.dce_rule is not None
      debug = pe.tracing_debug_info(
          self.dce_rule, in_tree, rule_out_tree, False, "custom_dce_rule"
      )
      dce_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
          flat_rule, in_avals, debug
      )
      # TODO(danfm): add support for consts.
      assert not consts

      # This second round of DCE is used to work out which inputs are actually
      # referenced by the DCEed Jaxpr. To avoid infinite recursion when the DCE
      # rule calls back into the primal, we replace all custom_dce primitives
      # with a sentinel primitive with a no-op DCE rule.
      dce_jaxpr = swap_primitives(dce_jaxpr, custom_dce_p, dce_sential_p)
      dce_jaxpr, used_ins = pe.dce_jaxpr(
          dce_jaxpr, [True] * len(dce_jaxpr.outvars)
      )
      dce_jaxpr = swap_primitives(dce_jaxpr, dce_sential_p, custom_dce_p)

      return pe.close_jaxpr(dce_jaxpr), used_ins

    debug = pe.tracing_debug_info(
        self.fun, in_tree, out_tree, False, "custom_dce"
    )
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals, debug)
    # TODO(danfm): add support for consts.
    assert not consts
    closed_call = pe.close_jaxpr(jaxpr)
    out_flat = custom_dce_p.bind(
        *args_flat, fun_jaxpr=closed_call, dce_jaxpr_thunk=dce_jaxpr_thunk
    )
    return tree_util.tree_unflatten(out_tree(), out_flat)


def check_for_tracers(x):
  # TODO(danfm): de-duplicate this with the version in custom_derivatives
  for leaf in tree_util.tree_leaves(x):
    if isinstance(leaf, core.Tracer):
      msg = (
          "Found a JAX Tracer object passed as an argument to a custom_dce "
          "function in a position indicated by static_argnums as static. "
          "Tracers cannot be passed as static arguments to custom_dce "
          "functions; instead, static_argnums should only be used for "
          "arguments that can't be or contain JAX tracers, e.g. "
          "function-valued arguments. In particular, array-valued arguments "
          "should typically not be indicated as static_argnums."
      )
      raise errors.UnexpectedTracerError(msg)


@lu.transformation_with_aux2
def flatten_dce_rule(
    f, store, fun_name, rule_name, used_outs, in_tree, out_tree, *args_flat
):
  py_used_outs = tree_util.tree_unflatten(out_tree, used_outs)
  py_args = tree_util.tree_unflatten(in_tree, args_flat)
  py_out = f(py_used_outs, *py_args)
  out_flat, rule_out_tree = tree_util.tree_flatten(py_out)
  # TODO(danfm): this check could be stricter. We could check that the Pytree
  # structure is the same as out_tree filtered to the used_outs.
  if len(out_flat) != sum(used_outs):
    raise TypeError(
        f"The custom DCE rule {rule_name} for function {fun_name} must return "
        f"a pytree that only includes the requested outputs. {rule_name} "
        f"returned {py_out} with {len(out_flat)} leaves, but {sum(used_outs)} "
        "leaves were expected."
    )
  store.store(rule_out_tree)
  return out_flat


def custom_dce_impl(*args, fun_jaxpr, **_):
  return core.jaxpr_as_fun(fun_jaxpr)(*args)


def custom_dce_abstract_eval(*args, fun_jaxpr, **_):
  del args  # unused
  return fun_jaxpr.out_avals, fun_jaxpr.effects


def custom_dce_batching(axis_data, args, dims, *, fun_jaxpr, dce_jaxpr_thunk):
  in_batched = [d is not batching.not_mapped for d in dims]
  args = [
      batching.moveaxis(x, d, 0) if b else x
      for b, x, d in zip(in_batched, args, dims)
  ]
  batched_fun_jaxpr, out_batched = batching.batch_jaxpr(
      fun_jaxpr, axis_data, in_batched, True
  )

  @pe._memoize
  def batched_dce_jaxpr_thunk(*used_outs: bool):
    dce_jaxpr, used_ins = dce_jaxpr_thunk(*used_outs)
    dce_jaxpr_batched, _ = batching.batch_jaxpr(
        dce_jaxpr,
        axis_data,
        [b for used, b in zip(used_ins, in_batched) if used],
        True,
    )
    return dce_jaxpr_batched, used_ins

  out_flat = custom_dce_p.bind(
      *args,
      fun_jaxpr=batched_fun_jaxpr,
      dce_jaxpr_thunk=batched_dce_jaxpr_thunk,
  )
  out_dims = [0 if b else batching.not_mapped for b in out_batched]
  return out_flat, out_dims


def custom_dce_jvp(primals, tangents, *, fun_jaxpr, **_):
  in_nz = [not isinstance(t, ad.Zero) for t in tangents]
  tangents = [t for nz, t in zip(in_nz, tangents) if nz]
  jvp_jaxpr, out_nz = ad.jvp_jaxpr(fun_jaxpr, in_nz, False)

  # TODO(danfm): We should avoid losing the DCE rule here, but it is more
  # straightforward to implement it like this to start. Instead, we should
  # bind a custom_dce primitive. To support that, we would need to add a
  # partial eval rule, and maybe a transpose rule.
  out = core.call_p.bind(
      lu.wrap_init(core.jaxpr_as_fun(jvp_jaxpr)), *primals, *tangents
  )

  out_primals, out_tangents = util.split_list(out, [len(out_nz)])
  out_tangents_iter = iter(out_tangents)
  out_tangents = [
      next(out_tangents_iter) if nz else ad.Zero.from_primal_value(p)
      for p, nz in zip(out_primals, out_nz)
  ]
  return out_primals, out_tangents


def custom_dce_rule(used_outs: Sequence[bool], eqn: core.JaxprEqn):
  if not any(used_outs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None
  if all(used_outs):
    return [True] * len(eqn.invars), eqn

  dce_jaxpr_thunk = eqn.params["dce_jaxpr_thunk"]
  jaxpr, used_ins = dce_jaxpr_thunk(*used_outs)
  invars = [v for used, v in zip(used_ins, eqn.invars) if used]
  outvars = [v for used, v in zip(used_outs, eqn.outvars) if used]

  @pe._memoize
  def new_dce_jaxpr_thunk(*new_used_outs: bool):
    if all(new_used_outs):
      return jaxpr, used_ins
    all_used_outs = util.merge_lists(
        used_outs,
        [False] * (len(used_outs) - len(new_used_outs)),
        new_used_outs,
    )
    new_jaxpr, all_used_ins = dce_jaxpr_thunk(*all_used_outs)
    not_used, new_used_ins = util.partition_list(used_ins, all_used_ins)
    assert not any(not_used)
    return new_jaxpr, new_used_ins

  new_params = dict(eqn.params)
  new_params["dce_jaxpr_thunk"] = new_dce_jaxpr_thunk
  new_params["fun_jaxpr"] = jaxpr
  new_eqn = pe.new_jaxpr_eqn(
      invars,
      outvars,
      custom_dce_p,
      new_params,
      jaxpr.effects,
      eqn.source_info,
      eqn.ctx,
  )
  return used_ins, new_eqn


custom_dce_p = core.Primitive("custom_dce_call")
custom_dce_p.multiple_results = True
custom_dce_p.def_impl(custom_dce_impl)
custom_dce_p.def_effectful_abstract_eval(custom_dce_abstract_eval)
mlir.register_lowering(
    custom_dce_p, mlir.lower_fun(custom_dce_impl, multiple_results=True)
)
batching.fancy_primitive_batchers[custom_dce_p] = custom_dce_batching
ad.primitive_jvps[custom_dce_p] = custom_dce_jvp
pe.dce_rules[custom_dce_p] = custom_dce_rule


def swap_primitives(
    jaxpr: core.Jaxpr, old: core.Primitive, new: core.Primitive
) -> core.Jaxpr:
  new_eqns = []
  for eqn in jaxpr.eqns:
    if eqn.primitive is old:
      new_eqns.append(eqn.replace(primitive=new))
    else:
      new_eqns.append(eqn)
  return jaxpr.replace(eqns=new_eqns)


dce_sential_p = core.Primitive("dce_sential")
dce_sential_p.multiple_results = True
dce_sential_p.def_impl(custom_dce_impl)
dce_sential_p.def_effectful_abstract_eval(custom_dce_abstract_eval)
