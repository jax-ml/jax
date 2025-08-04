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

  In JAX, DCE is performed when a function is staged out using
  :py:func:`jax.jit`, so it won't be applied when running JAX in eager mode.
  Similarly, the ``custom_dce`` decorator requires that both the decorated
  function and the custom DCE rule be compatible with :py:func:`~jax.jit`.

  This decorator allows users to customize the DCE behavior of a function by
  defining a custom DCE rule. For a ``custom_dce`` wrapped function
  ``f(*args)``, the signature of the DCE rule is ``dce_rule(used_outs, *args)``
  where ``used_outs`` is a Pytree with the same structure as the output of
  ``f``, and each leaf is is a ``bool`` indicating which outputs should
  be computed. The remaining arguments ``*args`` are the original arguments to
  ``f``. The rule ``dce_rule`` should return a Pytree with the same structure
  as the original output of ``f``, but any unused outputs can be replaced with
  ``None``.

  For example::

    >>> @jax.experimental.custom_dce.custom_dce
    ... def f(x, y):
    ...   return jnp.sin(x) * y, x * jnp.sin(y)
    ...
    >>> @f.def_dce
    ... def f_dce_rule(used_outs, x, y):
    ...   return (
    ...       jnp.sin(x) * y if used_outs[0] else None,
    ...       x * jnp.sin(y) if used_outs[1] else None,
    ...   )

  In this example, ``used_outs`` is a ``tuple`` with two ``bool`` values,
  indicating which outputs are required. The DCE rule only computes the
  required outputs, replacing the unused outputs with ``None``.

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
        using ``static_argnums``, (b) a Pytree of ``bool`` values
        (``used_outs``) indicating which outputs should be computed, and (c)
        the rest of the (non-static) arguments to the original function. The
        rule should return a Pytree with with the same structure as the output
        of the original function, but any unused outputs (as indicated by
        ``used_outs``) can be replaced with ``None``.
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
    debug = api_util.debug_info("custom_dce", self.fun,
                                args, {},
                                static_argnums=self.static_argnums)
    debug_rule = api_util.debug_info("custom_dce_rule", self.dce_rule,
                                     args, {},
                                     static_argnums=self.static_argnums)
    try:
      args = api_util.resolve_kwargs(self.fun, args, kwargs)
    except TypeError as e:
      raise TypeError(
          "The input arguments to the custom_dce-decorated function "
          f"{debug.func_name} could not be resolved to positional-only "
          f"arguments. Binding failed with the error:\n{e}"
      ) from e

    if self.static_argnums:
      static_argnums = set(self.static_argnums)
      for i in static_argnums:
        check_for_tracers(args[i])
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      fun, dyn_args = api_util.argnums_partial(
          lu.wrap_init(self.fun, debug_info=debug),
          dyn_argnums,
          args,
          require_static_args_hashable=False,
      )
      static_args = [args[i] for i in self.static_argnums]
      dce_rule = api_util.prepend_static_args(
          lu.wrap_init(self.dce_rule, debug_info=debug_rule), static_args
      )
    else:
      fun = lu.wrap_init(self.fun, debug_info=debug)
      dce_rule = lu.wrap_init(self.dce_rule, debug_info=debug_rule)
      dyn_args = args

    args_flat, in_tree = tree_util.tree_flatten(dyn_args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    in_avals = [core.get_aval(x) for x in args_flat]

    @pe._memoize
    def dce_jaxpr_thunk(
        *used_outs: bool,
    ) -> tuple[core.ClosedJaxpr, Sequence[bool]]:
      for store in dce_rule.stores:
        if store:
          store.reset()
      flat_rule, rule_out_tree = flatten_dce_rule(  # pylint: disable=unbalanced-tuple-unpacking
          dce_rule,
          fun_name,
          rule_name,
          used_outs,
          in_tree,
          out_tree(),
          out_avals,
      )
      assert self.dce_rule is not None
      dce_jaxpr, _, dce_consts = pe.trace_to_jaxpr_dynamic(
          flat_rule, in_avals
      )

      # This second round of DCE is used to work out which inputs are actually
      # referenced by the DCEed Jaxpr. To avoid infinite recursion when the DCE
      # rule calls back into the primal, we replace all custom_dce primitives
      # with a sentinel primitive with a no-op DCE rule.
      dce_jaxpr = swap_primitives(dce_jaxpr, custom_dce_p, dce_sential_p)
      dce_jaxpr, used_ins = pe.dce_jaxpr(
          dce_jaxpr, [True] * len(dce_jaxpr.outvars)
      )
      dce_jaxpr = swap_primitives(dce_jaxpr, dce_sential_p, custom_dce_p)

      return core.ClosedJaxpr(dce_jaxpr, dce_consts), used_ins

    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    closed_call = pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))
    out_avals = closed_call.out_avals
    out_flat = custom_dce_p.bind(
        *consts,
        *args_flat,
        num_consts=len(consts),
        fun_jaxpr=closed_call,
        dce_jaxpr_thunk=dce_jaxpr_thunk,
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
    f: Callable[..., Any],
    store: lu.Store,
    fun_name: str,
    rule_name: str,
    used_outs: Sequence[bool],
    in_tree,
    out_tree,
    out_avals: Sequence[core.AbstractValue],
    *args_flat,
):
  py_used_outs = tree_util.tree_unflatten(out_tree, used_outs)
  py_args = tree_util.tree_unflatten(in_tree, args_flat)
  py_out = f(py_used_outs, *py_args)
  if isinstance(py_out, list) and len(py_out) == len(
      tree_util.treedef_children(out_tree)
  ):
    py_out = tuple(py_out)

  # For error checking purposes, we need to reformat the pytree structure
  # of the output of the DCE rule to match the original output. The catch is
  # that the DCE rule can return a None to indicated an unused subtree, so we
  # need to rebuild those subtrees with a sentinel value at the leaves. This
  # logic is very similar to what is used in custom_dervatives._flatten_bwd.
  sentinel = object()
  dummy = tree_util.tree_unflatten(out_tree, [object()] * out_tree.num_leaves)
  keypaths, _ = util.unzip2(tree_util.tree_flatten_with_path(dummy)[0])
  out_flat = []

  def append(x, d):
    num_leaves = len(tree_util.tree_flatten(d)[0])
    if x is None and d is not None:
      out_flat.extend([sentinel] * num_leaves)
    elif x is not None:
      out_flat.extend([x] * num_leaves)
    return x

  _, rule_tree = tree_util.tree_flatten(py_out)
  try:
    tree_util.tree_map(append, py_out, dummy, is_leaf=lambda x: x is None)
  except ValueError:
    raise TypeError(
        f"Custom DCE rule {rule_name} for function {fun_name} must produce an "
        "output with the same container (pytree) structure as the output of "
        f"the function {fun_name}. But, the DCE rule returned structure "
        f"{rule_tree}, while {fun_name} returned {out_tree}."
    ) from None

  results = []
  for kp, used, aval, val in zip(keypaths, used_outs, out_avals, out_flat):
    if not used:
      continue
    if val is sentinel:
      raise ValueError(
          f"Custom DCE rule {rule_name} for function {fun_name} must produce "
          "values for all of the required outputs (as specified by the "
          "used_outs argument to the rule). But, at "
          f"output{tree_util.keystr(kp)}, the DCE rule returned None instead "
          f"of an output with shape/dtype {aval.str_short()}."
      )
    if not core.typematch(aval, aval_ := core.get_aval(val)):
      raise ValueError(
          f"Custom DCE rule {rule_name} for function {fun_name} must produce "
          "an output with the same shapes/dtypes as the output of the function "
          f"{fun_name}. But, at output{tree_util.keystr(kp)}, the DCE rule "
          f"returned an output with shape/dtype {aval_.str_short()}, while "
          f"{aval.str_short()} was expected."
      )
    results.append(val)

  store.store(rule_tree)
  return results


def custom_dce_impl(*args, fun_jaxpr: core.ClosedJaxpr, **_):
  return core.jaxpr_as_fun(fun_jaxpr)(*args)


def custom_dce_abstract_eval(*args, fun_jaxpr: core.ClosedJaxpr, **_):
  del args  # unused
  return fun_jaxpr.out_avals, fun_jaxpr.effects


def custom_dce_batching(
    axis_data: batching.AxisData,
    args,
    dims,
    *,
    num_consts: int,
    fun_jaxpr: core.ClosedJaxpr,
    dce_jaxpr_thunk: Callable[..., tuple[core.ClosedJaxpr, Sequence[bool]]],
):
  in_batched = [d is not batching.not_mapped for d in dims]
  args = [
      batching.moveaxis(x, d, 0) if b else x
      for b, x, d in zip(in_batched, args, dims)
  ]
  batched_fun_jaxpr, out_batched = batching.batch_jaxpr(
      fun_jaxpr, axis_data, in_batched, False
  )

  @pe._memoize
  def batched_dce_jaxpr_thunk(
      *used_outs: bool,
  ) -> tuple[core.ClosedJaxpr, Sequence[bool]]:
    dce_jaxpr, used_ins = dce_jaxpr_thunk(*used_outs)
    used_out_batched = [b for used, b in zip(used_outs, out_batched) if used]
    dce_jaxpr_batched, dce_out_batched = batching.batch_jaxpr(
        dce_jaxpr,
        axis_data,
        [b for used, b in zip(used_ins, in_batched[num_consts:]) if used],
        used_out_batched,
    )
    # TODO(danfm): For now we require that the DCE rule produce the same
    # batching pattern for the used outputs as the original function's batching.
    # While it is OK for the DCE rule to produce _less batched_ outputs (using
    # instantiate), we don't support the opposite case. It seems like we could
    # solve this by using instantiate=True when batching fun_jaxpr, but this
    # ends up changing the batching behavior sufficiently that it breaks some
    # real world use cases. Revisit if needed.
    assert all(a == b for a, b in zip(dce_out_batched, used_out_batched))
    return dce_jaxpr_batched, used_ins

  out_flat = custom_dce_p.bind(
      *args,
      num_consts=num_consts,
      fun_jaxpr=batched_fun_jaxpr,
      dce_jaxpr_thunk=batched_dce_jaxpr_thunk,
  )
  out_dims = [0 if b else batching.not_mapped for b in out_batched]
  return out_flat, out_dims


def custom_dce_jvp(primals, tangents, *, fun_jaxpr: core.ClosedJaxpr, **_):
  in_nz = [not isinstance(t, ad.Zero) for t in tangents]
  tangents = [t for nz, t in zip(in_nz, tangents) if nz]
  jvp_jaxpr, out_nz = ad.jvp_jaxpr(fun_jaxpr, in_nz, False)

  # TODO(danfm): We should avoid losing the DCE rule here, but it is more
  # straightforward to implement it like this to start. Instead, we should
  # bind a custom_dce primitive. To support that, we would need to add a
  # partial eval rule, and maybe a transpose rule. That being said, I expect
  # that most users of this API would compose this with a custom_jvp or
  # custom_vjp, which makes this less urgent.
  out = core.call_p.bind(
      lu.wrap_init(core.jaxpr_as_fun(jvp_jaxpr),
                   debug_info=jvp_jaxpr.jaxpr.debug_info), *primals, *tangents
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

  num_consts = eqn.params["num_consts"]
  dce_jaxpr_thunk = eqn.params["dce_jaxpr_thunk"]
  jaxpr, used_ins = dce_jaxpr_thunk(*used_outs)

  @pe._memoize
  def new_dce_jaxpr_thunk(
      *new_used_outs: bool,
  ) -> tuple[core.ClosedJaxpr, Sequence[bool]]:
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

  _, invars = util.split_list(eqn.invars, [num_consts])
  invars = [v for used, v in zip(used_ins, invars) if used]
  outvars = [v for used, v in zip(used_outs, eqn.outvars) if used]
  new_params = dict(
      eqn.params,
      num_consts=0,
      fun_jaxpr=jaxpr,
      dce_jaxpr_thunk=new_dce_jaxpr_thunk,
  )
  new_eqn = pe.new_jaxpr_eqn(
      invars,
      outvars,
      custom_dce_p,
      new_params,
      jaxpr.effects,
      eqn.source_info,
      eqn.ctx,
  )
  return [False] * eqn.params["num_consts"] + used_ins, new_eqn


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
