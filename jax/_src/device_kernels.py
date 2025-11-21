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


from jax._src import core
from jax._src.typing import (Array, ArrayLike, DeprecatedArg, DuckTypedArray)

from jax._src.interpreters import mlir

from jax._src.lib.mlir import ir

from typing import Any, Callable

from collections.abc import Sequence

from jax._src import dispatch
from jax._src import effects
from jax._src import util

# Import existing implementations from ffi.py
from jax._src.ffi import (
    _result_avals, _convert_layout_for_lowering
)
from jax._src.hashable_array import HashableArray

# Import interpreters for vmap support
from jax._src.interpreters import ad
from jax._src.interpreters import batching
import functools

# Import for batch partitioning support
from jax._src.lib import xla_client
from jax._src import xla_bridge

map, unsafe_map = util.safe_map, map

# Create wrapper functions to maintain dict interface
def _wrap_kwargs_hashable(kwargs: dict[str, Any]) -> dict[str, Any]:
  """Wrapper to maintain dict interface while using ffi implementation."""
  from jax._src.ffi import _wrap_kwargs_hashable as ffi_wrap
  wrapped = ffi_wrap(kwargs)
  return dict(wrapped)

def _unwrap_kwargs_hashable(kwargs: dict[str, Any]) -> dict[str, Any]:
  """Wrapper to maintain dict interface while using ffi implementation."""
  from jax._src.ffi import _unwrap_kwargs_hashable as ffi_unwrap
  return ffi_unwrap(tuple(kwargs.items()))

ResultMetadata = DuckTypedArray | core.AbstractToken

KERNEL_TYPE_TO_CALL_TARGET: dict[str, str] = {
    "ptx": "__gpu$xla.gpu.ptx",
}
SUPPORTED_KERNEL_TYPES: list[str] = list(KERNEL_TYPE_TO_CALL_TARGET.keys())

def register_device_kernel_as_batch_partitionable(kernel_type: str) -> None:
  """Registers a device kernel type as batch partitionable.

  This allows the kernel to be automatically partitioned across leading dimensions
  without requiring custom partitioning logic. The kernel will be executed
  independently on each shard of data.

  Args:
    kernel_type: The kernel type to register (e.g., "ptx")
  """
  if kernel_type not in SUPPORTED_KERNEL_TYPES:
    raise ValueError(f"Unsupported kernel type: {kernel_type}. Supported types are: {SUPPORTED_KERNEL_TYPES}")

  call_target = KERNEL_TYPE_TO_CALL_TARGET[kernel_type]
  xla_client.register_custom_call_as_batch_partitionable(call_target)
  xla_bridge.register_plugin_callbacks(
      functools.partial(xla_client.register_custom_call_as_batch_partitionable,
                        call_target))

# Register PTX kernels as batch partitionable by default
register_device_kernel_as_batch_partitionable("ptx")

def build_device_kernel_lowering_function(
    kernel_data: str,
    kernel_name: str,
    call_target: str,
    *,
    grid_dims: tuple[int, ...] = (1, 1, 1),
    block_dims: tuple[int, ...] = (256, 1, 1),
    shared_mem_bytes: int = 0,
    output_indices: Sequence[int] | None = None,
    has_side_effect: bool = False,
    operand_layouts: Sequence[Sequence[int]] | None = None,
    result_layouts: Sequence[Sequence[int]] | None = None,
    **lowering_args: Any,
) -> Callable[..., ir.Operation]:
  """Build a lowering op for a device kernel.

  By default, this lowering rule can use the input and output abstract values to
  compute the input and output types and shapes for the custom call, assuming
  row-major layouts.

  Note that layouts passed to this function should be in minor-to-major order
  (as expected by XLA).

  Args:
    kernel_data: Source code for the kernel
    kernel_name: Name of the kernel function to call
    call_target: The name of the custom call target
    grid_dims: Grid dimensions for kernel launch
    block_dims: Block dimensions for kernel launch
    shared_mem_bytes: Bytes of shared memory to allocate
    output_indices: Indices of outputs in argument list
    has_side_effect: Whether the kernel has side effects
    operand_layouts: A sequence of layouts (dimension orders) for each operand.
      By default, the operands are assumed to be row-major.
    result_layouts: A sequence of layouts (dimension orders) for each result.
      By default, the results are assumed to be row-major.
    lowering_args: Any other arguments to :func:`mlir.custom_call` will also be
      passed through if provided as extra arguments to this function.
  """

  def _lowering_op(
    ctx: mlir.LoweringRuleContext, *operands: ir.Value, **params: Any
  ) -> ir.Operation:
    if isinstance(output_indices, HashableArray):
        output_indices_val = list(output_indices.val)
    elif output_indices is not None:
        output_indices_val = list(output_indices)
    else:
        output_indices_val = []

    backend_config = {
        "name": kernel_name,
        "kernel_data": kernel_data,
        "kernel_type": "ptx",
        "grid_x": grid_dims[0],
        "grid_y": grid_dims[1],
        "grid_z": grid_dims[2],
        "block_x": block_dims[0],
        "block_y": block_dims[1],
        "block_z": block_dims[2],
        "shared_mem_bytes": shared_mem_bytes,
        "output_indices": output_indices_val,
    }

    backend_config = {k: mlir.ir_attribute(v) for k, v in backend_config.items()}

    kwargs = dict(lowering_args)
    kwargs.setdefault("api_version", 4)
    kwargs["backend_config"] = backend_config
    kwargs["has_side_effect"] = has_side_effect

    if "result_types" not in kwargs:
      kwargs["result_types"] = [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]

    if operand_layouts is None:
      kwargs["operand_layouts"] = map(_convert_layout_for_lowering, ctx.avals_in)
    else:
      kwargs["operand_layouts"] = [
          _convert_layout_for_lowering(*args)
          for args in zip(ctx.avals_in, operand_layouts)]

    if result_layouts is None:
      kwargs["result_layouts"] = map(_convert_layout_for_lowering, ctx.avals_out)
    else:
      kwargs["result_layouts"] = [
          _convert_layout_for_lowering(*args)
          for args in zip(ctx.avals_out, result_layouts)]

    return mlir.custom_call(call_target, operands=operands, **kwargs)

  return _lowering_op

def kernel_lowering(
    kernel_data: str,
    kernel_name: str,
    call_target: str,
    *,
    grid_dims: tuple[int, ...] = (1, 1, 1),
    block_dims: tuple[int, ...] = (256, 1, 1),
    shared_mem_bytes: int = 0,
    output_indices: Sequence[int] | None = None,
    has_side_effect: bool = False,
    operand_layouts: Sequence[Sequence[int]] | None = None,
    result_layouts: Sequence[Sequence[int]] | None = None,
    **lowering_args: Any
) -> mlir.LoweringRule:
  """Build a lowering rule for a device kernel.

  By default, this lowering rule can use the input and output abstract values to
  compute the input and output types and shapes for the custom call, assuming
  row-major layouts.

  Note that layouts passed to this function should be in minor-to-major order
  (as expected by XLA).

  Args:
    kernel_data: Source code for the kernel
    kernel_name: Name of the kernel function to call
    call_target: The name of the custom call target
    grid_dims: Grid dimensions for kernel launch
    block_dims: Block dimensions for kernel launch
    shared_mem_bytes: Bytes of shared memory to allocate
    output_indices: Indices of outputs in argument list
    has_side_effect: Whether the kernel has side effects
    operand_layouts: A sequence of layouts (dimension orders) for each operand.
      By default, the operands are assumed to be row-major.
    result_layouts: A sequence of layouts (dimension orders) for each result.
      By default, the results are assumed to be row-major.
    lowering_args: Any other arguments to :func:`mlir.custom_call` will also be
      passed through if provided as extra arguments to this function.
  """

  def _lowering(
    ctx: mlir.LoweringRuleContext, *operands: ir.Value, **params: Any
  ) -> Sequence[ir.Value | Sequence[ir.Value]]:
    result = build_device_kernel_lowering_function(
        kernel_data,
        kernel_name,
        call_target,
        grid_dims=grid_dims,
        block_dims=block_dims,
        shared_mem_bytes=shared_mem_bytes,
        output_indices=output_indices,
        has_side_effect=has_side_effect,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
        **lowering_args,
    )(ctx, *operands, **params)

    return result.results  # type: ignore

  return _lowering

def _normalize_grid_block_dims(grid_dims: int | tuple[int, ...] | list[int], block_dims: int | tuple[int, ...] | list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
  if isinstance(grid_dims, int):
    grid_dims = (grid_dims, 1, 1)
  elif len(grid_dims) == 1:
    grid_dims = (grid_dims[0], 1, 1)
  elif len(grid_dims) == 2:
    grid_dims = (*grid_dims, 1)
  elif len(grid_dims) > 3:
    raise ValueError(f"Expected at most 3 grid dimensions, got {len(grid_dims)}")

  if isinstance(block_dims, int):
    block_dims = (block_dims, 1, 1)
  elif len(block_dims) == 1:
    block_dims = (block_dims[0], 1, 1)
  elif len(block_dims) == 2:
    block_dims = (*block_dims, 1)
  elif len(block_dims) > 3:
    raise ValueError(f"Expected at most 3 block dimensions, got {len(block_dims)}")

  return tuple(grid_dims), tuple(block_dims)

def kernel_call(
    kernel_data: str,
    kernel_name: str,
    result_shape_dtypes: ResultMetadata | Sequence[ResultMetadata],
    *args: ArrayLike,
    kernel_type: str = "ptx",
    grid_dims: int | tuple[int, ...] | list[int] = (1, 1, 1),
    block_dims: int | tuple[int, ...] | list[int] = (256, 1, 1),
    shared_mem_bytes: int = 0,
    has_side_effect: bool = False,
    output_indices: Sequence[int] | None = None,
    vmap_method: str | None = None,
    vectorized: bool | DeprecatedArg = DeprecatedArg(),
    **kwargs: Any,
) -> Array | list[Array]:  # type: ignore
    """Call a device kernel with the specified kernel type.

    Currently supported kernel types:
    - ptx: NVIDIA PTX kernel code for CUDA GPUs

    Args:
        kernel_data: Source code for the kernel
        kernel_name: Name of the kernel function to call
        result_shape_dtypes: Shape and dtype of the result(s)
        *args: Input arrays
        kernel_type: Type of kernel code (e.g., "ptx")
        grid_dims: Grid dimensions for kernel launch
        block_dims: Block dimensions for kernel launch
        shared_mem_bytes: Bytes of shared memory to allocate
        has_side_effect: Whether the kernel has side effects
        output_indices: Indices of outputs in argument list
        vmap_method: Method for vmapping the kernel
        vectorized: Whether the kernel is vectorized
        **kwargs: Additional arguments for specific kernel types

    Returns:
        Result array(s) from kernel execution
    """
    # Check for deprecated vectorized argument
    if not isinstance(vectorized, DeprecatedArg):
        raise ValueError(
            "The 'vectorized' argument of jax.experimental.device_kernels.kernel_call is deprecated. "
            "Use 'vmap_method' instead.")

    # Validate kernel_type
    if kernel_type not in SUPPORTED_KERNEL_TYPES:
        raise ValueError(f"Unsupported kernel type: {kernel_type}. Supported types are: {SUPPORTED_KERNEL_TYPES}")

    # Kernel-specific validation
    if kernel_type == "ptx" and ".entry" not in kernel_data:
        raise ValueError("PTX code must contain an .entry point")

    if isinstance(result_shape_dtypes, Sequence):
        multiple_results = True
        result_avals = _result_avals(result_shape_dtypes)
    else:
        multiple_results = False
        result_avals = _result_avals((result_shape_dtypes,))

    if output_indices is not None:
        expected_num_outputs = len(result_avals)
        if not isinstance(output_indices, (list, tuple)):
            raise ValueError("output_indices must be a sequence")
        if len(output_indices) != expected_num_outputs:
            raise ValueError(
                f"Expected {expected_num_outputs} output indices but got {len(output_indices)}"
            )
        if not all(isinstance(idx, int) and 0 <= idx < len(args) + expected_num_outputs for idx in output_indices):
            raise ValueError(
                f"Output indices must be integers in range [0, {len(args) + expected_num_outputs}), got {output_indices}"
            )

    output_indices = () if output_indices is None else tuple(output_indices)

    grid_dims, block_dims = _normalize_grid_block_dims(grid_dims, block_dims)
    call_target = KERNEL_TYPE_TO_CALL_TARGET[kernel_type]

    kernel_kwargs = {
        "grid_x": grid_dims[0],
        "grid_y": grid_dims[1],
        "grid_z": grid_dims[2],
        "block_x": block_dims[0],
        "block_y": block_dims[1],
        "block_z": block_dims[2],
        "shared_mem_bytes": shared_mem_bytes,
        "output_indices": output_indices,
        "call_target": call_target,
        **kwargs,
    }

    results = kernel_call_p.bind(
        *args,
        result_avals=result_avals,
        vectorized=vectorized,
        vmap_method=vmap_method,
        kernel_name=kernel_name,
        kernel_data=kernel_data,
        has_side_effect=has_side_effect,
        **_wrap_kwargs_hashable(kernel_kwargs),
    )
    return results if multiple_results else results[0]

class KernelEffect(effects.Effect):
  def __str__(self):
    return "Kernel"

_KernelEffect = KernelEffect()
effects.lowerable_effects.add_type(KernelEffect)
effects.control_flow_allowed_effects.add_type(KernelEffect)

def kernel_call_abstract_eval(
    *avals_in,
    result_avals: tuple[core.AbstractValue, ...],
    has_side_effect: bool,
    **_,
):
    out_vma = core.standard_vma_rule('kernel_call', *avals_in)
    effects = {_KernelEffect} if has_side_effect else core.no_effects
    return tuple(r if r is core.abstract_token else r.update(vma=out_vma)
               for r in result_avals), effects


def kernel_call_jvp(*args, kernel_name, **_):
    """JVP rule for kernel_call - reuses FFI's implementation."""
    from jax._src.ffi import ffi_call_jvp
    return ffi_call_jvp(*args, target_name=kernel_name, **_)


def kernel_call_transpose(*args, kernel_name, **_):
    """Transpose rule for kernel_call - reuses FFI's implementation."""
    from jax._src.ffi import ffi_call_transpose
    return ffi_call_transpose(*args, target_name=kernel_name, **_)


def kernel_call_batching_rule(
    prim,
    args,
    dims,
    *,
    vmap_method: str | None,
    result_avals: Sequence[core.ShapedArray],
    **kwargs: Any,
):
    """Batching rule for kernel_call - reuses FFI's implementation.

    The batching logic is identical to FFI's implementation since both handle
    custom calls with layout updates for vmapping.
    """
    from jax._src.ffi import ffi_batching_rule
    return ffi_batching_rule(prim, args, dims, vmap_method=vmap_method,
                             result_avals=result_avals, **kwargs)


def kernel_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *operands: ir.Value,
    kernel_name: str,
    kernel_data: str,
    has_side_effect: bool,
    **kwargs: Any,
) -> Sequence[ir.Value]:

    call_target = kwargs.get("call_target")
    if call_target is None:
        raise ValueError("call_target must be provided")

    rule = kernel_lowering(
        kernel_data,
        kernel_name,
        call_target,
        grid_dims=(kwargs["grid_x"], kwargs["grid_y"], kwargs["grid_z"]),
        block_dims=(kwargs["block_x"], kwargs["block_y"], kwargs["block_z"]),
        shared_mem_bytes=kwargs["shared_mem_bytes"],
        output_indices=kwargs["output_indices"],
        has_side_effect=has_side_effect,
        operand_layouts=kwargs.get("operand_layouts"),
        result_layouts=kwargs.get("result_layouts"),
    )

    return rule(ctx, *operands, **_unwrap_kwargs_hashable(kwargs))

kernel_call_p = core.Primitive("kernel_call")
kernel_call_p.multiple_results = True
dispatch.simple_impl(kernel_call_p)
kernel_call_p.def_effectful_abstract_eval(kernel_call_abstract_eval)
ad.primitive_jvps[kernel_call_p] = kernel_call_jvp
ad.primitive_transposes[kernel_call_p] = kernel_call_transpose
batching.primitive_batchers[kernel_call_p] = functools.partial(
    kernel_call_batching_rule, kernel_call_p)
mlir.register_lowering(kernel_call_p, kernel_call_lowering)
