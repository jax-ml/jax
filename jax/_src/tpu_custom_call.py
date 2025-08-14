# Copyright 2023 The JAX Authors.
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

"""JAX bindings for Mosaic."""

# mypy: ignore-errors
from __future__ import annotations

import base64
import collections.abc
from collections.abc import Callable, Sequence
import dataclasses
import enum
import io
import json
from typing import Any, TypedDict

from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import sharding_impls
from jax._src.cloud_tpu_init import is_cloud_tpu_older_than
from jax._src.frozen_dict import FrozenDict
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lib import tpu
from jaxlib.mlir import ir
from jaxlib.mlir.passmanager import PassManager

try:
  from absl import flags
  FLAGS = flags.FLAGS
except ImportError:
  FLAGS = {}

_MOSAIC_ALLOW_HLO = config.bool_state(
    name="jax_mosaic_allow_hlo",
    default=False,
    help="Allow hlo dialects in Mosaic",
)


# Controls the IR serialization version. Upon incrementing the
# default version in jaxlib/mosaic/dialect/tpu/transforms/serde.cc we must
# continue to use the old serialization version when in forward compatibility
# mode: for 1 month when exporting, or when using old cloud TPU.
#
# This can be achieved by adding:
#    if ctx.is_forward_compat() or is_cloud_tpu_older_than(<today>):
#       return <previous_serialization_version>
#    return None
#
# We should also add a TODO to remove the conditional one month later.
def get_ir_version(ctx: mlir.LoweringRuleContext) -> int | None:
  if is_cloud_tpu_older_than(2025, 6, 19):
    return 4
  if is_cloud_tpu_older_than(2025, 7, 25):
    return 5
  if is_cloud_tpu_older_than(2025, 7, 27):
    return 6
  # TODO(naumsmogers): remove the forward compatibility check after 2025-09-14.
  if ctx.is_forward_compat() or is_cloud_tpu_older_than(2025, 8, 14):
    return 7
  return None


tpu_custom_call_p = core.Primitive("tpu_custom_call")
tpu_custom_call_p.multiple_results = True
dispatch.simple_impl(tpu_custom_call_p)


def tpu_custom_call_batcher(axis_data, args, dims, **kwargs):
  if axis_data.size != 1:
    raise NotImplementedError(
        "tpu_custom_call does not support non-trivial batching."
    )
  unbatched_args = tuple(
      a if (d is batching.not_mapped or d is None) else a[d]
      for a, d in zip(args, dims, strict=True)
  )
  out_unbatched = tpu_custom_call_p.bind(*unbatched_args, **kwargs)
  out = tuple(o[None] for o in out_unbatched)
  return out, (0,) * len(out)
batching.fancy_primitive_batchers[tpu_custom_call_p] = tpu_custom_call_batcher


class MemorySpace(enum.Enum):
  HBM = enum.auto()
  VMEM = enum.auto()
  SEMAPHORE_MEM = enum.auto()
  SMEM = enum.auto()
  HOST = enum.auto()

  @property
  def color(self) -> int:
    if self == MemorySpace.HBM:
      return 0
    elif self == MemorySpace.VMEM:
      return 1
    elif self == MemorySpace.SEMAPHORE_MEM:
      return 2
    elif self == MemorySpace.SMEM:
      return 4
    elif self == MemorySpace.HOST:
      return 5
    else:
      raise ValueError("invalid memory space: " + str(self))


class CostEstimate(TypedDict):
  flops: int
  transcendentals: int
  bytes_accessed: int


@dataclasses.dataclass(frozen=True)
class CustomCallBackendConfig:
  """Represents an unserialized backend config for custom calls."""
  lowered_module_asm: bytes
  has_communication: bool
  collective_id: int | None
  device_type: str | None
  cost_estimate: CostEstimate | None
  needs_hlo_passes: bool
  needs_layout_passes: bool
  vmem_limit_bytes: int | None
  flags: dict[str, bool | int | float] | None
  allow_input_fusion: Sequence[bool] | None
  serialization_format: int | None
  internal_scratch_in_bytes: int | None
  output_memory_spaces: tuple[MemorySpace | None, ...] | None
  disable_bounds_checks: bool
  active_core_count: int | None
  input_memory_spaces: tuple[MemorySpace | None, ...] | None
  skip_device_barrier: bool

  def __post_init__(self):
    if self.allow_input_fusion is not None:
      object.__setattr__(self, "allow_input_fusion",
                         tuple(self.allow_input_fusion))
    if self.cost_estimate is not None:
      object.__setattr__(self, "cost_estimate",
                         FrozenDict(self.cost_estimate))

  # We omit the body while printing, because primitive params get embedded
  # in HLO metadata, and the body blows up its size.
  def __repr__(self):
    return "CustomCallBackendConfig(<omitted>)"

  def to_json(self) -> bytes:
    """Serializes the backend config into JSON."""
    # We format the JSON ourselves, because json.dumps seems to be overly slow.
    config = io.BytesIO()
    config.write(b'{"custom_call_config": {"body": "')
    config.write(base64.b64encode(self.lowered_module_asm))
    config.write(b'"')
    if self.has_communication:
      config.write(b', "has_communication": ')
      config.write(str(self.has_communication).lower().encode("ascii"))
    if self.collective_id is not None:
      config.write(b', "collective_id": ')
      config.write(str(self.collective_id).encode("ascii"))
    if self.cost_estimate is not None:
      config.write(b', "cost_estimate": ')
      config.write(
          json.dumps(dict(self.cost_estimate), sort_keys=True).encode("ascii")
      )
    if self.needs_hlo_passes:
      config.write(b', "needs_hlo_passes": ')
      config.write(str(self.needs_hlo_passes).lower().encode("ascii"))
    if self.serialization_format is not None:
      config.write(b', "serialization_format": ')
      config.write(str(self.serialization_format).lower().encode("ascii"))
    if self.needs_layout_passes:
      config.write(b', "needs_layout_passes": ')
      config.write(str(self.needs_layout_passes).lower().encode("ascii"))
    if self.allow_input_fusion is not None:
      config.write(b', "allow_input_fusion": [')
      for i, value in enumerate(self.allow_input_fusion):
        config.write(b"true" if value else b"false")
        # config.write(str(value).lower().encode("ascii"))
        if i + 1 != len(self.allow_input_fusion):
          config.write(b",")
      config.write(b"]")
    if self.internal_scratch_in_bytes is not None:
      config.write(b', "internal_scratch_in_bytes": ')
      config.write(str(self.internal_scratch_in_bytes).encode("ascii"))
    if self.output_memory_spaces is not None:
      config.write(b', "output_memory_colors": [')
      for i, memory_space in enumerate(self.output_memory_spaces):
        if i:
          config.write(b",")
        color = memory_space.color if memory_space is not None else -1
        config.write(str(color).encode("ascii"))
      config.write(b"]")
    if self.input_memory_spaces is not None:
      comma = False
      for i, input_memory_space in enumerate(self.input_memory_spaces):
        if input_memory_space is None:
          continue
        if input_memory_space is MemorySpace.SMEM:
          # TODO(sharadmv): Add support for SMEM (though atm, XLA will not
          # page out SMEM arrays).
          continue
        if input_memory_space not in (
            MemorySpace.HBM,
            MemorySpace.VMEM,
            MemorySpace.SMEM,
        ):
          raise NotImplementedError(
              "input_memory_space_colors only supports HBM, VMEM and SMEM"
          )
        if comma:
          config.write(b",")
        else:
          config.write(b', "input_memory_space_colors": [')
        config.write(
            f'{{"operand_index":{i},"color":{input_memory_space.color}}}'
            .encode("ascii")
        )
        comma = True
      if comma:
        config.write(b"]")
    if self.disable_bounds_checks:
      config.write(b', "disable_bounds_checks": ')
      config.write(str(self.disable_bounds_checks).lower().encode("ascii"))
    if self.skip_device_barrier:
      config.write(b', "skip_device_barrier": ')
      config.write(str(self.skip_device_barrier).lower().encode("ascii"))
    config.write(b"}")  # End of custom_call_config.
    if self.device_type is not None:
      config.write(b', "device_type": ')
      config.write(
          ('"DEVICE_TYPE_' + self.device_type.upper() + '"').encode("ascii")
      )
    if self.vmem_limit_bytes is not None:
      config.write(
          b', "scoped_memory_configs": [{"memory_space":1, "offset": 0,'
          b' "size": '
      )
      config.write(str(self.vmem_limit_bytes).encode("ascii"))
      config.write(b'}]')
    if self.flags is not None:
      config.write(b', "flag_configs": [')
      for i, (flag, value) in enumerate(self.flags.items()):
        config.write(b'{"flag_type": "')
        config.write(flag.encode("ascii"))
        config.write(b'", value: {')
        if isinstance(value, bool):
          config.write(b'"boolean_value": ')
          config.write(b"true" if value else b"false")
        elif isinstance(value, int):
          config.write(b'"integer_value": ')
          config.write(str(value).encode("ascii"))
        elif isinstance(value, float):
          config.write(b'"double_value": ')
          config.write(str(value).encode("ascii"))
        else:
          raise ValueError("invalid flag value: " + str(value))
        config.write(b"}}")
        if i + 1 != len(self.flags):
          config.write(b",")
      config.write(b"]")
    if self.device_type == "sparsecore" and self.active_core_count == 1:
      config.write(b', "megachip_parallelism_config": {"cores": ["0"]}')
    config.write(b"}")
    return config.getvalue()


@tpu_custom_call_p.def_abstract_eval
def _tpu_custom_call_abstract_eval(*_, out_avals, **__):
  return out_avals


def _avals_to_layouts(avals) -> Sequence[Sequence[int]]:
  return [tuple(range(a.ndim - 1, -1, -1)) for a in avals]  # pytype: disable=attribute-error


def _tpu_custom_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,  # pylint: disable=missing-function-docstring
    config: CustomCallBackendConfig,
    has_side_effects: bool,
    kernel_name: str | None,
    out_avals: Any,
    input_output_aliases: tuple[tuple[int, int], ...],
    metadata: Any | None,
) -> ir.OpResultList:
  result_types = [mlir.aval_to_ir_type(aval) for aval in out_avals]
  axis_context = ctx.module_context.axis_context
  if isinstance(axis_context, sharding_impls.SPMDAxisContext):
    if axis_context.manual_axes != frozenset(axis_context.mesh.axis_names):
      raise NotImplementedError(
          "Mosaic kernels cannot be automatically partitioned. Please wrap the"
          " call in a shard_map."
      )
  elif isinstance(axis_context, sharding_impls.ShardingContext):
    if axis_context.num_devices != 1:
      raise NotImplementedError(
          "Mosaic kernels cannot be automatically partitioned. Please wrap the"
          " call in a shard_map."
      )
  elif config.has_communication:
    raise NotImplementedError(
        "Replica lowering for Mosaic kernels not implemented."
    )
  if all(core.is_constant_shape(aval_out.shape) for aval_out in ctx.avals_out):
    result_shapes = None
  else:
    result_shapes = [
        mlir.shape_tensor(mlir.eval_dynamic_shape(ctx, aval_out.shape))
        for aval_out in ctx.avals_out]
  extra_attributes = None
  # Add kernel_name and kernel_metadata as attributes to the custom call op.
  # This is because we do not want to pollute the backend_config with this
  # information.
  if kernel_name is not None:
    extra_attributes = dict(kernel_name=ir.StringAttr.get(kernel_name))
  has_side_effects = has_side_effects if has_side_effects is not None else False
  call = mlir.custom_call(
      "tpu_custom_call",
      result_types=result_types,
      operands=in_nodes,
      backend_config=config.to_json(),
      api_version=1,
      has_side_effect=has_side_effects,
      operand_output_aliases=dict(input_output_aliases),
      operand_layouts=_avals_to_layouts(ctx.avals_in),
      result_layouts=_avals_to_layouts(ctx.avals_out),
      result_shapes=result_shapes,
      extra_attributes=extra_attributes,
  )
  if metadata is not None:
    call.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
        dict(kernel_metadata=ir.StringAttr.get(json.dumps(metadata)))
    )
  return call.results


mlir.register_lowering(tpu_custom_call_p, _tpu_custom_call_lowering,
                       platform="tpu")


def _lower_mosaic_module_to_asm(
    module: ir.Module,
    *,
    device_type: str | None,
    ir_version: int | None = None,
) -> tuple[ir.Module, tuple[bool, bool, bool, bool]]:
  has_communication, has_custom_barrier = tpu.private_has_communication(
      module.operation
  )
  needs_hlo_passes = _MOSAIC_ALLOW_HLO.value
  needs_layout_passes = not device_type
  # We'll mutate the module, so clone it
  with module.context as ctx, module.operation.location as _:
    module_op = module.operation.clone()
    prev_allow_unregistered_dialects = ctx.allow_unregistered_dialects
    ctx.allow_unregistered_dialects = True
    target_version = (
        f"target-version={ir_version}" if ir_version is not None else ""
    )
    try:
      pipeline = PassManager.parse(
          "builtin.module(mosaic-serde{serialize=true " + target_version + "})"
      )
      pipeline.run(module_op)
    finally:
      ctx.allow_unregistered_dialects = prev_allow_unregistered_dialects
    bytecode_buffer = io.BytesIO()
    module_op.write_bytecode(bytecode_buffer, desired_version=0)
    asm = bytecode_buffer.getvalue()
    return asm, (
        has_communication,
        has_custom_barrier,
        needs_hlo_passes,
        needs_layout_passes,
    )


def _get_device_type(module: ir.Module) -> str | None:
  """Determines the device type based on the core_type annotations."""
  sparsecore_func_found = False
  tensorcore_func_found = False

  def assign_device_type_based_on_core_type(op: ir.Operation) -> ir.WalkResult:
    nonlocal sparsecore_func_found
    nonlocal tensorcore_func_found
    if op.name == "func.func":
      if "tpu.core_type" in op.attributes:
        core_type = op.attributes["tpu.core_type"]
        if str(core_type) in [
            f"#tpu.core_type<{c}>"
            for c in ["sc_scalar_subcore", "sc_vector_subcore"]
        ]:
          sparsecore_func_found = True
          if tensorcore_func_found:
            return ir.WalkResult.INTERRUPT
          return ir.WalkResult.SKIP
        if str(core_type) == "#tpu.core_type<tc>":
          tensorcore_func_found = True
          return ir.WalkResult.SKIP
        raise ValueError(f"Unknown core type: {core_type}")
    return ir.WalkResult.ADVANCE

  module.operation.walk(
      assign_device_type_based_on_core_type, walk_order=ir.WalkOrder.PRE_ORDER
  )
  if tensorcore_func_found and sparsecore_func_found:
    raise ValueError(
        "A single Mosaic kernel cannot contain both TensorCore and SparseCore"
        " functions."
    )
  if sparsecore_func_found:
    return "sparsecore"
  return None


def _get_active_core_count(module: ir.Module) -> int | None:

  def get_core_parallel_dim_size(
      dim_semantics: ir.ArrayAttr,
      iter_bounds: ir.DenseI64ArrayAttr,
      other_subkernel_core_dim_size: int | None = None) -> int | None:

    if len(iter_bounds) != len(dim_semantics):
      raise ValueError(
          "The iteration bounds and dimension semantics attributes must have"
          " the same number of elements."
      )

    subkernel_core_dim_size = None

    for dim_idx, (dim_size, dim_sem) in enumerate(
        zip(iter_bounds, dim_semantics)
    ):
      if str(dim_sem) != "#tpu.dimension_semantics<core_parallel>":
        continue

      if ir.ShapedType.is_dynamic_size(dim_size):
        raise ValueError(
            "The iteration bound corresponding to the core-parallel dimension "
            f"{dim_idx} must be statically known."
        )
      if subkernel_core_dim_size is not None:
        raise ValueError(
            "A single Mosaic subkernel cannot contain multiple core sharding "
            "dimensions."
        )
      if (
          other_subkernel_core_dim_size is not None
          and other_subkernel_core_dim_size != dim_size
      ):
        raise ValueError(
            "The iteration bound corresponding to the core-parallel dimension "
            "be the same across all subkernels."
        )
      subkernel_core_dim_size = dim_size

    return subkernel_core_dim_size

  core_parallel_dim_size = None

  for op in module.body.operations:
    if op.operation.name != "func.func":
      continue

    if (
        "iteration_bounds" not in op.attributes
        or "dimension_semantics" not in op.attributes
    ):
      continue

    try:
      iter_bounds = ir.DenseI64ArrayAttr(op.attributes["iteration_bounds"])
    except ValueError as e:
      e.add_note("The iteration bounds attribute must be an array.")
      raise
    try:
      dim_semantics = ir.ArrayAttr(op.attributes["dimension_semantics"])
    except ValueError as e:
      e.add_note("The dimension semantics attribute must be an array.")
      raise

    core_parallel_dim_size = get_core_parallel_dim_size(
        dim_semantics=dim_semantics,
        iter_bounds=iter_bounds,
        other_subkernel_core_dim_size=core_parallel_dim_size,
    )

  return core_parallel_dim_size


def _lower_to_custom_call_config(
    module: ir.Module,
    *,
    vmem_limit_bytes: int | None,
    cost_estimate: CostEstimate | None,
    flags: dict[str, bool | int | float] | None,
    allow_input_fusion: Sequence[bool] | None,
    internal_scratch_in_bytes: int | None,
    collective_id: int | None,
    serialization_format: int | None,
    output_memory_spaces: tuple[MemorySpace | None, ...] | None = None,
    ir_version: int | None = None,
    disable_bounds_checks: bool = False,
    input_memory_spaces: tuple[MemorySpace | None, ...] | None = None,
    skip_device_barrier: bool = False,
) -> CustomCallBackendConfig:
  device_type = _get_device_type(module)
  lowered_module_asm, (
      has_communication,
      has_custom_barrier,
      needs_hlo_passes,
      needs_layout_passes,
  ) = _lower_mosaic_module_to_asm(
      module,
      device_type=device_type,
      ir_version=ir_version,
  )
  active_core_count = _get_active_core_count(module)
  return _lowered_to_custom_call_config(
      lowered_module_asm,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      device_type=device_type,
      serialization_format=serialization_format,
      has_custom_barrier=has_custom_barrier,
      has_communication=has_communication,
      needs_hlo_passes=needs_hlo_passes,
      needs_layout_passes=needs_layout_passes,
      output_memory_spaces=output_memory_spaces,
      disable_bounds_checks=disable_bounds_checks,
      active_core_count=active_core_count,
      input_memory_spaces=input_memory_spaces,
      skip_device_barrier=skip_device_barrier,
  )


def _lowered_to_custom_call_config(
    lowered_module_asm: bytes,
    *,
    vmem_limit_bytes: int | None,
    cost_estimate: CostEstimate | None,
    flags: dict[str, bool | int | float] | None,
    allow_input_fusion: Sequence[bool] | None,
    internal_scratch_in_bytes: int | None,
    collective_id: int | None,
    serialization_format: int | None,
    has_custom_barrier: bool,
    has_communication: bool,
    needs_hlo_passes: bool,
    needs_layout_passes: bool,
    device_type: str | None,
    output_memory_spaces: tuple[MemorySpace | None, ...] | None = None,
    disable_bounds_checks: bool = False,
    active_core_count: int | None = None,
    input_memory_spaces: tuple[MemorySpace | None, ...] | None = None,
    skip_device_barrier: bool = False,
):
  if has_custom_barrier:
    if collective_id is None:
      raise ValueError(
          "collective_id has to be specified when using a custom barrier"
      )
  elif collective_id is not None:
    raise ValueError(
        "collective_id has to be unspecified or None when not using a custom"
        " barrier"
    )
  if vmem_limit_bytes is not None and not isinstance(vmem_limit_bytes, int):
    raise ValueError(
        "vmem_limit_bytes must be an int: provided with a"
        f" {type(vmem_limit_bytes)}."
    )
  config = CustomCallBackendConfig(
      lowered_module_asm,
      has_communication,
      collective_id,
      device_type,
      cost_estimate,
      needs_hlo_passes,
      needs_layout_passes,
      vmem_limit_bytes,
      flags,
      allow_input_fusion,
      serialization_format,
      internal_scratch_in_bytes,
      output_memory_spaces,
      disable_bounds_checks,
      active_core_count=active_core_count,
      input_memory_spaces=input_memory_spaces,
      skip_device_barrier=skip_device_barrier,
  )
  return config


def lower_module_to_custom_call(
    ctx: mlir.LoweringRuleContext,
    *in_nodes: ir.Value,
    module: ir.Module,
    out_type: Any,
    kernel_name: str,
    cost_estimate: CostEstimate | None,
    vmem_limit_bytes: int | None,
    flags: dict[str, bool | int | float] | None,
    allow_input_fusion: Sequence[bool] | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    internal_scratch_in_bytes: int | None,
    collective_id: int | None,
    has_side_effects: bool,
    serialization_format: int | None,
    output_memory_spaces: tuple[MemorySpace | None, ...] | None,
    disable_bounds_checks: bool = False,
    input_memory_spaces: tuple[MemorySpace | None, ...] | None,
    metadata: Any | None = None,
    skip_device_barrier: bool = False,
) -> Sequence[ir.Value]:
  config = _lower_to_custom_call_config(
      module,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      serialization_format=serialization_format,
      output_memory_spaces=output_memory_spaces,
      ir_version=get_ir_version(ctx),
      disable_bounds_checks=disable_bounds_checks,
      input_memory_spaces=input_memory_spaces,
      skip_device_barrier=skip_device_barrier,
  )
  return _tpu_custom_call_lowering(
      ctx,
      *in_nodes,
      config=config,
      has_side_effects=has_side_effects,
      kernel_name=kernel_name,
      out_avals=out_type,
      input_output_aliases=input_output_aliases,
      metadata=metadata,
  )


def as_tpu_kernel(
    module: ir.Module,
    out_type: Any,
    *,
    cost_estimate: CostEstimate | None = None,
    kernel_name: str | None = None,
    vmem_limit_bytes: int | None = None,
    flags: dict[str, bool | int | float] | None = None,
    allow_input_fusion: Sequence[bool] | None = None,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
    internal_scratch_in_bytes: int | None = None,
    collective_id: int | None = None,
    has_side_effects: bool = False,
    serialization_format: int | None = 1,
    output_memory_spaces: tuple[MemorySpace | None, ...] | None = None,
    disable_bounds_checks: bool = False,
    input_memory_spaces: tuple[MemorySpace | None, ...] | None = None,
    metadata: Any | None = None,
) -> Callable[..., Any]:
  """Turns an MLIR Mosaic kernel into a JAX-compatible function."""
  config = _lower_to_custom_call_config(
      module,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      serialization_format=serialization_format,
      output_memory_spaces=output_memory_spaces,
      disable_bounds_checks=disable_bounds_checks,
      input_memory_spaces=input_memory_spaces,
  )
  return _as_jax_callable(
      config,
      has_side_effects,
      out_type,
      kernel_name=kernel_name,
      input_output_aliases=input_output_aliases,
      metadata=metadata,
  )


def lowered_as_tpu_kernel(
    lowered_module: ir.Module,
    out_type: Any,
    *,
    collective_id: int | None = None,
    cost_estimate: CostEstimate | None = None,
    needs_hlo_passes: bool = False,
    needs_layout_passes: bool = False,
    has_communication: bool = False,
    has_side_effects: bool = False,
    has_custom_barrier: bool = False,
    kernel_name: str | None = None,
    vmem_limit_bytes: int | None = None,
    flags: dict[str, bool | int | float] | None = None,
    allow_input_fusion: Sequence[bool] | None = None,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
    serialization_format: int | None = None,
    internal_scratch_in_bytes: int | None = None,
    disable_bounds_checks: bool = False,
    metadata: Any | None = None,
) -> Callable[..., Any]:
  device_type = _get_device_type(lowered_module)
  lowered_module_asm = lowered_module.operation.get_asm(
      binary=True, enable_debug_info=True
  )
  config = _lowered_to_custom_call_config(
      lowered_module_asm,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      device_type=device_type,
      serialization_format=serialization_format,
      has_custom_barrier=has_custom_barrier,
      has_communication=has_communication,
      needs_hlo_passes=needs_hlo_passes,
      needs_layout_passes=needs_layout_passes,
      disable_bounds_checks=disable_bounds_checks,
  )
  return _as_jax_callable(
      config,
      has_side_effects,
      out_type,
      kernel_name=kernel_name,
      input_output_aliases=input_output_aliases,
      metadata=metadata,
  )


def _as_jax_callable(
    config: CustomCallBackendConfig,
    has_side_effects: bool,
    out_type: Any,
    *,
    kernel_name: str | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    metadata: Any | None,
) -> Callable[..., Any]:
  unpack = False
  if not isinstance(out_type, collections.abc.Iterable):
    out_type = (out_type,)
    unpack = True
  out_avals = tuple(core.ShapedArray(ty.shape, ty.dtype) for ty in out_type)

  # We use jax.jit to make sure we hit the fast compilation cache.
  def apply_kernel(*args):
    result = tpu_custom_call_p.bind(
        *args,
        config=config,
        has_side_effects=has_side_effects,
        kernel_name=kernel_name,
        out_avals=out_avals,
        input_output_aliases=input_output_aliases,
        metadata=metadata,
    )
    return result[0] if unpack else result

  return api.jit(apply_kernel)
