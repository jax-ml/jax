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

from collections.abc import Callable
from functools import partial
from typing import Any

from jax._src import api
from jax._src import core
from jax._src import tree_util
from jax._src import util
from jax._src import xla_metadata
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir


set_xla_metadata = xla_metadata.set_xla_metadata

ATTACH_TO_VALUE_FLAG_KEY = "attach_to_value"
ATTACH_TO_GRAD_FLAG_KEY = "attach_to_grad"
METADATA_TAGGING_FN_KEY = "_metadata_tagging_fn"

# -----------------------------------------------------------------------------

# `xla_metadata_func_p` is a primitive for attaching frontend metadata to ops

xla_metadata_func_p = core.Primitive("xla_metadata_func")
xla_metadata_func_p.multiple_results = True


@xla_metadata_func_p.def_impl
def _xla_metadata_func_impl(*args, fn, xla_metadata, **kwargs):
  # The metadata is only used during lowering, ignored at runtime here.
  results = fn(*args, **kwargs)
  if not isinstance(results, tuple):
    return (results,)
  return results


@xla_metadata_func_p.def_abstract_eval
def _xla_metadata_func_abstract_eval(*avals, fn, xla_metadata, **kwargs):
  # Trace `fn` with the input abstract values (`avals`) and kwargs
  wrapped_fun = lambda *a: fn(*a, **kwargs)
  # Use make_jaxpr to get the actual output abstract values (including sharding)
  closed_jaxpr = api.make_jaxpr(wrapped_fun)(*avals)
  # Return the flat list of output avals
  return closed_jaxpr.out_avals


def _xla_metadata_func_lowering_rule(
    ctx: mlir.LoweringRuleContext, *args: ir.Value, fn, xla_metadata, **kwargs
):
  xla_metadata = dict(xla_metadata)
  fn_to_lower = partial(fn, **kwargs) if kwargs else fn

  # 1. Get the standard lowering rule for fn_to_lower
  fn_lowering = mlir.lower_fun(
      fn_to_lower, multiple_results=len(ctx.avals_out) > 1
  )

  # 2. Run the lowering rule
  results = fn_lowering(ctx, *args)

  # 3. Find the root / leaf-node operation(s) and attach metadata.
  flat_results = tree_util.tree_leaves(results)
  if not flat_results:
    return results

  if xla_metadata.get(ATTACH_TO_VALUE_FLAG_KEY, True):
    for res_val in flat_results:
      if isinstance(res_val, ir.Value) and res_val.owner is not None:
        _attach_xla_metadata_to_op(xla_metadata, res_val.owner)

  return results


mlir.register_lowering(xla_metadata_func_p, _xla_metadata_func_lowering_rule)


def _xla_metadata_func_jvp_rule(
    primals, tangents, *, fn, xla_metadata, **kwargs
):
  fn_with_kwargs = partial(fn, **kwargs) if kwargs else fn
  primals_out, tangents_out = api.jvp(fn_with_kwargs, primals, tangents)

  primals_out = xla_metadata_value_p.bind(
      primals_out, xla_metadata=xla_metadata
  )
  tangents_out = xla_metadata_value_p.bind(
      tangents_out, xla_metadata=xla_metadata
  )

  if not isinstance(primals_out, tuple):
    primals_out = (primals_out,)
  if not isinstance(tangents_out, tuple):
    tangents_out = (tangents_out,)

  return primals_out, tangents_out


ad.primitive_jvps[xla_metadata_func_p] = _xla_metadata_func_jvp_rule


def _xla_metadata_func_batching_rule(
    batched_args, bdims, *, fn, xla_metadata, **f_kwargs
):
  results = xla_metadata_func_p.bind(
      *batched_args, fn=fn, xla_metadata=xla_metadata, **f_kwargs
  )
  if all(bdim is batching.not_mapped for bdim in bdims):
    out_bdims = tree_util.tree_map(lambda x: batching.not_mapped, results)
  else:
    # This assumes that if any input is batched, all outputs are batched
    # along axis 0. This is a common convention for vmap.
    out_bdims = [0] * len(results)
  return results, out_bdims


batching.primitive_batchers[xla_metadata_func_p] = (
    _xla_metadata_func_batching_rule
)


# -----------------------------------------------------------------------------

# `xla_metadata_value_p` is a primitive for attaching frontend metadata to
# a value's producing (parent) op

xla_metadata_value_p = core.Primitive("xla_metadata_value")
xla_metadata_value_p.def_impl(lambda value, *, xla_metadata: value)
xla_metadata_value_p.def_abstract_eval(lambda aval, *, xla_metadata: aval)
ad.deflinear2(xla_metadata_value_p, lambda ct, _, **params: (ct,))


def _xla_metadata_value_lowering_rule(
    ctx: mlir.LoweringRuleContext, value_mlir: ir.Value, *, xla_metadata
):
  xla_metadata = dict(xla_metadata)
  # If value is produced by a multiply * 1.0, this will be optimized away.
  # This is a common pattern in JAX for operations like sin(x) and exp(x).
  # We want to attach metadata to the multiply op in this case.
  if value_mlir.owner.name == "stablehlo.multiply":

    def _is_constant_op(op):
      return (
          hasattr(op, "owner")
          and hasattr(op.owner, "name")
          and op.owner.name == "stablehlo.constant"
      )

    operands = value_mlir.owner.operation.operands
    non_constant_operands = [op for op in operands if not _is_constant_op(op)]

    if len(operands) > 1 and len(non_constant_operands) == 1:
      if not isinstance(non_constant_operands[0].owner, ir.Block):
        _attach_xla_metadata_to_op(
            xla_metadata,
            non_constant_operands[0].owner,
        )
        return [value_mlir]

  # Attach metadata to the operation that produced value_mlir.
  if value_mlir.owner is not None:
    if not isinstance(value_mlir.owner, ir.Block):
      if value_mlir.owner.name != "stablehlo.constant":
        _attach_xla_metadata_to_op(xla_metadata, value_mlir.owner)

  return [value_mlir]


mlir.register_lowering(xla_metadata_value_p, _xla_metadata_value_lowering_rule)


def _xla_metadata_value_jvp_rule(primals, tangents, *, xla_metadata):
  (out_primal,) = primals
  (out_tangent,) = tangents
  out_primal = xla_metadata_value_p.bind(
      out_primal, xla_metadata=xla_metadata
  )
  out_tangent = xla_metadata_value_p.bind(
      out_tangent, xla_metadata=xla_metadata
  )
  return out_primal, out_tangent


ad.primitive_jvps[xla_metadata_value_p] = _xla_metadata_value_jvp_rule


def _xla_metadata_value_batching_rule(batched_args, bdims, *, xla_metadata):
  (value_batched,) = batched_args
  (value_bdim,) = bdims
  out_batched = xla_metadata_value_p.bind(
      value_batched, xla_metadata=xla_metadata
  )
  return out_batched, value_bdim


batching.primitive_batchers[xla_metadata_value_p] = (
    _xla_metadata_value_batching_rule
)


# -----------------------------------------------------------------------------


def metadata_tagging_hook(cts_out, metadata):
  hashable_metadata = tuple(sorted(metadata.items()))
  cts_out = tree_util.tree_map(
      lambda t: xla_metadata_value_p.bind(t, xla_metadata=hashable_metadata)
      if not isinstance(t, ad.Zero)
      else t,
      cts_out,
  )
  return cts_out


def _attach_xla_metadata_to_op(
    xla_metadata: dict[str, Any], op: ir.Operation
) -> None:
  xla_metadata = xla_metadata.copy()
  xla_metadata.pop(ATTACH_TO_VALUE_FLAG_KEY, None)
  xla_metadata.pop(ATTACH_TO_GRAD_FLAG_KEY, None)
  xla_metadata.pop(METADATA_TAGGING_FN_KEY, None)

  ctx_attributes = {}
  existing_attributes = {}
  if xla_metadata:
    for k, v in xla_metadata.items():
      ctx_attributes[k] = ir.StringAttr.get(str(v).lower())
    # Combine with existing mhlo.frontend_attributes
    op_attributes_dict = {attr.name: attr.attr for attr in op.attributes}
    for k, attributes in op_attributes_dict.items():
      if k == "mhlo.frontend_attributes":
        v_dict = {attr.name: attr.attr for attr in attributes}
        for fa_key, fa_val in v_dict.items():
          existing_attributes[fa_key] = fa_val
    op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
        ctx_attributes | existing_attributes
    )


def attach_metadata(*args, **kwargs):
  """Attaches XLA metadata to JAX functions or values, or acts as a decorator factory.

  This function supports three main modes of operation:

  1. Direct Function Wrapping:
     The first positional argument is a callable (function).
     All keyword arguments are treated as metadata.
     Example:
       def my_func(x): return x * 2.0
       wrapped_func = attach_metadata(my_func, key="value")
       result = wrapped_func(data)

  2. Value Tagging:
     The first positional argument is a JAX value (e.g., Tracer, ndarray).
     All keyword arguments are treated as metadata. The metadata is attached
     to the operation that produced the value.
     Example:
       y = some_fn(x)
       y_tagged = attach_metadata(y, key="value")

  3. Decorator Factory:
     Used without positional arguments. Metadata is passed as keyword arguments.
     When used as a decorator, it applies metadata via a context manager
     around the decorated function's execution.
     Example:
       @attach_metadata(key="value")
       def my_function(x):
         return x + 1.0

     Note: While it can be used as a context manager (`with
     attach_metadata(...):`),
     for that direct purpose, `set_xla_metadata` might be more idiomatic.

  Args:
    *args: - If one argument: The function (Callable) or JAX value (Any) to be
      tagged. - If empty: Decorator factory / context manager mode.
    **kwargs: Metadata key-value pairs (e.g., op_name="custom_op", key=value).
      These are used as the `xla_metadata` for the underlying primitives or for
      the `set_xla_metadata` context manager. Two boolean flags are also
      supported to control how metadata is attached:
      - `attach_to_value`: (Default: True) In value-tagging mode, controls
        whether metadata is attached to the forward-pass value.
      - `attach_to_grad`: (Default: True) In value-tagging or
        function-wrapping modes, controls whether metadata is also attached to
        the gradient during automatic differentiation.

  Returns:
    - A wrapped function if a function is passed as the first argument.
    - A tagged value (or tuple of tagged values, if a tuple is passed)
      if a JAX value is passed as the first argument.
    - A decorator if called without positional arguments.

  Raises:
    TypeError: If more than one positional argument is provided.
  """
  target = None
  if len(args) == 1:
    target = args[0]
  elif len(args) > 1:
    raise TypeError(
        f"attach_metadata expects 0 or 1 positional arguments, got {len(args)}"
    )
  xla_metadata = kwargs
  if xla_metadata.get(ATTACH_TO_GRAD_FLAG_KEY, True):
    xla_metadata[METADATA_TAGGING_FN_KEY] = metadata_tagging_hook
  hashable_metadata = tuple(sorted(xla_metadata.items()))

  # Case 1: Attach metadata to the final operation of a function
  #   e.g: attach_metadata(my_func, key=value)
  if target is not None and callable(target):
    fn = target
    @util.wraps(fn)
    def wrapped_fn(*fn_args, **fn_kwargs):
      results_tuple = xla_metadata_func_p.bind(
          *fn_args, fn=fn, xla_metadata=hashable_metadata, **fn_kwargs
      )
      return results_tuple[0] if len(results_tuple) == 1 else results_tuple
    return wrapped_fn

  # Case 2: Attach metadata to the operation directly producing 'value'.
  #   e.g: attach_metadata(my_value, key=value)
  elif target is not None:
    value = target
    if xla_metadata.get(ATTACH_TO_VALUE_FLAG_KEY, True):
      value = tree_util.tree_map(
          lambda v: xla_metadata_value_p.bind(v, xla_metadata=hashable_metadata),
          value,
      )
    return value

  # Case 3: Attach metadata to all operations within the decorated function.
  #   e.g: @attach_metadata(key=value)
  elif target is None:
    def _context_manager_decorator(
        fn_to_decorate: Callable[..., Any],
    ) -> Callable[..., Any]:
      @util.wraps(fn_to_decorate)
      def _wrapped_fn_with_context(*fn_args, **fn_kwargs):
        with set_xla_metadata(**xla_metadata):
          return fn_to_decorate(*fn_args, **fn_kwargs)
      return _wrapped_fn_with_context
    return _context_manager_decorator
