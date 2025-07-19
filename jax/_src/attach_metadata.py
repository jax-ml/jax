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
from typing import Any

from jax._src import api
from jax._src import core
from jax._src.custom_derivatives import custom_vjp
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

# -----------------------------------------------------------------------------

# `xla_metadata_value_p` is an identity primitive for attaching frontend_attributes
# to the primitive's producing (parent/owner) op.

xla_metadata_value_p = core.Primitive("xla_metadata_value")
xla_metadata_value_p.def_impl(lambda value, *, xla_metadata: value)
xla_metadata_value_p.def_abstract_eval(lambda aval, *, xla_metadata: aval)


def _xla_metadata_value_lowering_rule(
    ctx: mlir.LoweringRuleContext, value_mlir: ir.Value, *, xla_metadata
):
  xla_metadata = dict(xla_metadata)
  op_to_attach_metadata = _target_op_to_attach_metadata(value_mlir)
  if op_to_attach_metadata and xla_metadata.get(ATTACH_TO_VALUE_FLAG_KEY, True):
    _attach_xla_metadata_to_op(xla_metadata, op_to_attach_metadata)
  return [value_mlir]


mlir.register_lowering(
    xla_metadata_value_p, _xla_metadata_value_lowering_rule, cacheable=False
)


def _xla_metadata_value_transpose_rule(cts_out, *primals, xla_metadata):
  xla_metadata = dict(xla_metadata)
  if xla_metadata.get(ATTACH_TO_GRAD_FLAG_KEY, True):
    xla_metadata[ATTACH_TO_VALUE_FLAG_KEY] = True
    hashable_metadata = tuple(sorted(xla_metadata.items()))
    cts_out = xla_metadata_value_p.bind(cts_out, xla_metadata=hashable_metadata)
  return (cts_out,)


ad.deflinear2(xla_metadata_value_p, _xla_metadata_value_transpose_rule)


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


def custom_vjp_metadata_wrapper(
    fn: Callable[..., Any],
    attach_to_value: bool = True,
    attach_to_grad: bool = True,
    **xla_metadata
):
  """Wraps a function to tag its last operation with metadata."""
  hashable_metadata = tuple(sorted(xla_metadata.items()))

  @custom_vjp
  def wrapped_fn(*args):
    value = fn(*args)
    if attach_to_value:
      value = tree_util.tree_map(
          lambda v: xla_metadata_value_p.bind(
              v, xla_metadata=hashable_metadata
          ),
          value,
      )
    return value

  def fwd(*args):
    primal_out, vjp_fn = api.vjp(fn, *args)
    return primal_out, vjp_fn

  def bwd(vjp_fn, cts_in):
    cts_out = vjp_fn(cts_in)
    if attach_to_grad:
      cts_out = tree_util.tree_map(
          lambda ct: xla_metadata_value_p.bind(
              ct, xla_metadata=hashable_metadata
          ),
          cts_out,
      )
    return cts_out

  wrapped_fn.defvjp(fwd, bwd)
  return wrapped_fn


def _target_op_to_attach_metadata(value_mlir: ir.Value) -> ir.Operation | None:
  # Attach to the value's owner, except in certain cases.
  # E.g., if the owner is a Block, or if the owner op will be optimized away.
  op = value_mlir.owner
  if op is None or isinstance(op, ir.Block):
    return None

  # If value_mlir is produced by a multiply * 1.0, it'll be optimized away.
  # This is a common pattern in jax.grad() for operations like sin(x) and exp(x).
  # We want to attach metadata to the non-constant operand of the multiply op.
  if op.name == "stablehlo.multiply":

    def _is_constant_op_with_value_1(operand: ir.Value) -> bool:
      return (
          hasattr(operand, "owner")
          and hasattr(operand.owner, "name")
          and operand.owner.name == "stablehlo.constant"
          and list(operand.owner.attributes["value"]) == [1.0]
      )

    non_constant_operands = [
        o for o in op.operands if not _is_constant_op_with_value_1(o)
    ]

    # If there's one non-constant operand, its owner is our real target.
    if len(non_constant_operands) == 1:
      target_op = non_constant_operands[0].owner
      # The target's owner could also be a Block, so re-check.
      if not target_op or isinstance(target_op, ir.Block):
        return None
      return target_op

  return op


def _attach_xla_metadata_to_op(
    xla_metadata: dict[str, Any], op: ir.Operation
) -> None:
  xla_metadata = xla_metadata.copy()
  xla_metadata.pop(ATTACH_TO_VALUE_FLAG_KEY, None)
  xla_metadata.pop(ATTACH_TO_GRAD_FLAG_KEY, None)

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


def attach_metadata(
    fn: Callable[..., Any] | None = None,
    attach_to_value: bool = True,
    attach_to_grad: bool = True,
    **xla_metadata
):
  """Attaches XLA metadata to the final JAX operation of a function.

  This function supports two modes of operation:

  1. Direct Function Wrapping:
     The first positional argument is a callable (function). Example:
     ```
        def fn(x):
          x = jnp.sin(x) + jnp.cos(x)
          return x * 2.0
        wrapped_fn = attach_metadata(fn, **metadata)  # Tags the `multiply` op
        value = wrapped_fn(x)
     ```

  2. Decorator Factory:
     Used without positional arguments. When used as a decorator, it applies
     metadata via a context manager around the decorated function's execution.
     Example:
     ```
        @attach_metadata(**metadata)  # Tags all operations in the decorated fn
        def my_function(x):
          return x * 2.0
     ```

    Note: While it can be used as a context manager (`with
    attach_metadata(...):`),
    for that direct purpose, `jax.xla_metadata.set_xla_metadata` may be more
    idiomatic.

  Args:
    fn: - If not None: The function (Callable) to be tagged.
        - If None: Decorator factory / context manager mode.
    attach_to_value: Controls whether metadata is attached to the forward-pass
      value.
    attach_to_grad: Controls whether metadata is attached to the gradient during
      automatic differentiation.
    **metadata: Metadata key-value pairs (e.g. key="value"). These will be
      attached as `frontend_attributes` in the resulting HLO graph.

  Returns:
    - A wrapped function if a function is passed as the first argument.
    - A decorator if called without positional arguments.

  Raises:
    TypeError: If `fn` is provided but is not a callable.
  """
  # Case 1: Attach metadata to the final JAX operation of a function
  if fn is not None and callable(fn):
    return custom_vjp_metadata_wrapper(
        fn, attach_to_value, attach_to_grad, **xla_metadata
    )

  # Case 2: Attach metadata to all operations within the decorated function.
  elif fn is None:
    def context_manager_decorator_metadata_wrapper(
        fn: Callable[..., Any],
    ) -> Callable[..., Any]:
      @util.wraps(fn)
      def _wrapped_fn_with_context(*fn_args, **fn_kwargs):
        with set_xla_metadata(**xla_metadata):
          return fn(*fn_args, **fn_kwargs)

      return _wrapped_fn_with_context

    return context_manager_decorator_metadata_wrapper

  elif fn is not None and not callable(fn):
    raise TypeError(
        "`attach_metadata()` expects a `Callable`. To attach to a value, "
        "use the `attach_metadata_jvp()` or `attach_metadata_vjp()` functions "
        "instead."
    )


def attach_metadata_jvp(value: Any, **xla_metadata):
  """Attaches XLA metadata to the JAX operation which directly produced the value
  (Forward-pass only). Example Usage:
  ```
    def fn(x):
      value = jnp.sin(x) * jnp.cos(x)
      return attach_metadata_jvp(value, **metadata)  # Tags the `multiply` op
  ```

  Args:
    value: The value to be tagged.
    **metadata: Metadata key-value pairs (e.g. key="value"). These will be
      attached as `frontend_attributes` in the resulting HLO graph.

  Returns:
    - The value with metadata attached.

  Raises:
    TypeError: If `value` is None or a callable.
  """
  if value is None or callable(value):
    raise TypeError(
        "`attach_metadata_jvp()` expects a value. To attach to a function,"
        " use `attach_metadata()` instead."
    )

  if xla_metadata.get(ATTACH_TO_VALUE_FLAG_KEY, True):
    xla_metadata[ATTACH_TO_GRAD_FLAG_KEY] = False
    hashable_metadata = tuple(sorted(xla_metadata.items()))
    value = tree_util.tree_map(
        lambda v: xla_metadata_value_p.bind(v, xla_metadata=hashable_metadata),
        value,
    )
  return value


def attach_metadata_vjp(value: Any, **xla_metadata):
  """Attaches XLA metadata to the JAX operation which directly consumes the value
  (Backward-pass only). Example Usage:
  ```
    @jax.grad
    def fn(x):
      x = attach_metadata_vjp(x, **metadata)  # Tags the `cosine` op
      return jnp.sin(x)
  ```

  Args:
    value: The value to be tagged.
    **metadata: Metadata key-value pairs (e.g. key="value"). These will be attached
      as `frontend_attributes` in the resulting HLO graph.

  Returns:
    - The value with metadata attached.

  Raises:
    TypeError: If `value` is None or a callable.
  """
  if value is None or callable(value):
    raise TypeError(
        "`attach_metadata_vjp()` expects a value -- to attach to a function,"
        " use `attach_metadata()` instead."
    )

  if xla_metadata.get(ATTACH_TO_GRAD_FLAG_KEY, True):
    xla_metadata[ATTACH_TO_VALUE_FLAG_KEY] = False
    hashable_metadata = tuple(sorted(xla_metadata.items()))
    value = tree_util.tree_map(
        lambda v: xla_metadata_value_p.bind(v, xla_metadata=hashable_metadata),
        value,
    )
  return value
