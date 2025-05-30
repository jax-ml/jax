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

import jax
from jax._src import core
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.xla_metadata import set_xla_metadata

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
  closed_jaxpr = jax.make_jaxpr(wrapped_fun)(*avals)
  # Return the flat list of output avals
  return closed_jaxpr.out_avals


def _xla_metadata_func_lowering_rule(
    ctx: mlir.LoweringRuleContext, *args: ir.Value, fn, xla_metadata, **kwargs
):
  fn_to_lower = partial(fn, **kwargs) if kwargs else fn

  # 1. Get the standard lowering rule for fn_to_lower
  fn_lowering = mlir.lower_fun(
      fn_to_lower, multiple_results=len(ctx.avals_out) > 1
  )

  # 2. Run the lowering rule
  results = fn_lowering(ctx, *args)

  # 3. Find the root / leaf-node operation(s) and attach metadata.
  flat_results = tree_util.tree_leaves(results)
  if not flat_results:  # Nothing to attach metadata to
    return results

  for res_val in flat_results:
    if isinstance(res_val, ir.Value) and res_val.owner is not None:
      _attach_xla_metadata_to_op(xla_metadata, res_val.owner)

  return results

mlir.register_lowering(xla_metadata_func_p, _xla_metadata_func_lowering_rule)


def _xla_metadata_func_jvp_rule(
    primals, tangents, *, fn, xla_metadata, **kwargs
):
  fn_with_kwargs = partial(fn, **kwargs) if kwargs else fn
  # with context:
  primals_out, tangents_out = jax.jvp(fn_with_kwargs, primals, tangents)

  # jvp
  # partial eval (unzipping)
  # transpose

  primals_out = xla_metadata_value_p.bind(primals_out,
                                          xla_metadata=xla_metadata,
                                          **kwargs)

  # TODO(nbasile) - Add support for tangents.
  # tangents_out = xla_metadata_value_p.bind(tangents_out,
  #                                          xla_metadata=xla_metadata,
  #                                          **kwargs)
  # tangents_out = xla_metadata_func_p.bind(tangents_out,
  #                                         fn=fn,
  #                                         xla_metadata=xla_metadata,
  #                                         **kwargs)[0]

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
    # along axis 0.
    # This is a common convention for vmap.
    out_bdims = [0] * len(results)
  return results, out_bdims

batching.primitive_batchers[xla_metadata_func_p] = (
    _xla_metadata_func_batching_rule
)


# -----------------------------------------------------------------------------

# `xla_metadata_value_p` is a primitive for attaching frontend metadata to
# a value's producing op

xla_metadata_value_p = core.Primitive("xla_metadata_value")


@xla_metadata_value_p.def_impl
def _xla_metadata_value_impl(value, *, xla_metadata):
  # Metadata is used at lowering time.
  return value


@xla_metadata_value_p.def_abstract_eval
def _xla_metadata_value_abstract_eval(aval, *, xla_metadata):
  # The abstract value of the output is the same as the input.
  return aval


def _xla_metadata_value_lowering_rule(
    ctx: mlir.LoweringRuleContext, value_mlir: ir.Value, *, xla_metadata
):
  # Attach metadata to the operation that produced value_mlir.
  if value_mlir.owner is not None:
    _attach_xla_metadata_to_op(xla_metadata, value_mlir.owner)
  # Pass the value through.
  return [value_mlir]

mlir.register_lowering(xla_metadata_value_p, _xla_metadata_value_lowering_rule)


def _xla_metadata_value_jvp_rule(primals, tangents, *, xla_metadata):
  (value_primal,) = primals
  (value_tangent,) = tangents
  out_primal = xla_metadata_value_p.bind(
      value_primal, xla_metadata=xla_metadata
  )
  out_tangent = xla_metadata_value_p.bind(
      value_tangent, xla_metadata=xla_metadata
  )
  return out_primal, out_tangent

ad.primitive_jvps[xla_metadata_value_p] = _xla_metadata_value_jvp_rule


def _xla_metadata_value_transpose_rule(
    cotangents, primals, *, xla_metadata, **kwargs
):
  return (cotangents,)

ad.primitive_transposes[xla_metadata_value_p] = (
    _xla_metadata_value_transpose_rule
)


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


def _attach_xla_metadata_to_op(
    xla_metadata: dict[str, Any], op: ir.Operation
) -> None:
  ctx_attributes = {}
  existing_attributes = {}
  if xla_metadata:
    for k, v in xla_metadata.items():
      ctx_attributes[k] = ir.StringAttr.get(str(v).lower())
    if isinstance(op, ir.Operation):
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
       wrapped_func = attach_metadata(my_func, op_name="wrapped_op")
       result = wrapped_func(data)

  2. Value Tagging:
     The first positional argument is a JAX value (e.g., Tracer, ndarray).
     All keyword arguments are treated as metadata. The metadata is attached
     to the operation that produced the value.
     Example:
       y = some_fn(x)
       y_tagged = attach_metadata(y, op_name="value_tag_op")

  3. Decorator Factory:
     Used without positional arguments. Metadata is passed as keyword arguments.
     When used as a decorator, it applies metadata via a context manager
     around the decorated function's execution.
     Example:
       @attach_metadata(op_name="my_special_op")
       def my_function(x):
         return x + 1.0

     Note: While it can be used as a context manager (`with attach_metadata(...):`),
     for that direct purpose, `set_xla_metadata` might be more idiomatic.

  Args:
    *args:
      - If one argument: The function (Callable) or JAX value (Any) to be tagged.
      - If empty: Decorator factory / context manager mode.
    **kwargs: Metadata key-value pairs (e.g., op_name="custom_op", key=value).
              These are used as the `xla_metadata` for the underlying
              primitives or for the `set_xla_metadata` context manager.

  Returns:
    - A decorator if called without positional arguments.
    - A wrapped function if a function is passed as the first argument.
    - A tagged value (or tuple of tagged values, if a tuple is passed)
      if a JAX value is passed as the first argument.

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

  # All **kwargs are considered the metadata payload.
  xla_metadata = kwargs

  # Case 1: Wrap a function directly
  # attach_metadata(my_func, key=value)
  if target is not None and callable(target):
    fn = target
    @util.wraps(fn)
    def wrapped_fn(*fn_args, **fn_kwargs):
      # xla_metadata_func_p.bind returns a tuple because multiple_results=True
      results_tuple = xla_metadata_func_p.bind(
          *fn_args, fn=fn, xla_metadata=xla_metadata, **fn_kwargs
      )
      # If the original fn was intended to return a single value,
      # _xla_metadata_func_impl would have wrapped it in a tuple.
      # We unpack it here for better ergonomics if it's a single-element tuple.
      if len(results_tuple) == 1:
        return results_tuple[0]
      return results_tuple
    return wrapped_fn

  # Case 2: Tag a value
  # attach_metadata(my_value, key=value)
  elif target is not None:
    value = target
    # `xla_metadata_value_p` will attach metadata to the op producing `value`.
    if isinstance(value, tuple):
      return tuple(xla_metadata_value_p.bind(v, xla_metadata=xla_metadata)
                   for v in value)
    else:
      return xla_metadata_value_p.bind(value, xla_metadata=xla_metadata)

  # Case 3: Decorator factory usage
  # @attach_metadata(key=value)
  elif target is None:
    def _context_manager_decorator(fn_to_decorate: Callable[..., Any]
                                   ) -> Callable[..., Any]:
      @util.wraps(fn_to_decorate)
      def _wrapped_fn_with_context(*fn_args, **fn_kwargs):
        with set_xla_metadata(**xla_metadata):
          return fn_to_decorate(*fn_args, **fn_kwargs)
      return _wrapped_fn_with_context
    return _context_manager_decorator
