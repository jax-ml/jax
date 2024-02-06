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
import dataclasses
import functools
import io
import os
import time
from typing import Any, Callable

from absl import flags
import jax
from jax import core
from jax._src import config
from jax._src.lib import tpu
from jax._src.lib import xla_client
from jax._src.lib.mlir.dialects import hlo
from jax._src.interpreters import mlir
from jax._src import sharding_impls
from jax.interpreters import xla
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo
import numpy as np

FLAGS = flags.FLAGS
_MOSAIC_ON_DEVICE_CHECKS = config.define_string_state(
    name="mosaic_on_device_checks",
    default="",
    help=(
        "If True, additional on-device asserts are inserted into the program,"
        " to verify operation invariants (accesses in-bounds, etc.)"
    ),
)

_MOSAIC_ALLOW_HLO = config.define_bool_state(
    name="jax_mosaic_allow_hlo",
    default=False,
    help="Allow hlo dialects in Mosaic",
)

tpu_custom_call_p = core.Primitive("tpu_custom_call")
tpu_custom_call_p.def_impl(
    functools.partial(xla.apply_primitive, tpu_custom_call_p))
tpu_custom_call_p.multiple_results = True


@dataclasses.dataclass(frozen=True)
class CostEstimate:
  flops: int
  transcendentals: int
  bytes_accessed: int

  def to_json(self) -> bytes:
    return (
        f'{{"flops": {self.flops}, "transcendentals": {self.transcendentals},'
        f' "bytes_accessed": {self.bytes_accessed}}}'
    ).encode('ascii')


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
      config.write(self.cost_estimate.to_json())
    if self.needs_hlo_passes:
      config.write(b', "needs_hlo_passes": ')
      config.write(str(self.needs_hlo_passes).lower().encode("ascii"))
    if self.needs_layout_passes:
      config.write(b', "needs_layout_passes": ')
      config.write(str(self.needs_layout_passes).lower().encode("ascii"))
    config.write(b"}")
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
    config.write(b"}")
    return config.getvalue()


@tpu_custom_call_p.def_abstract_eval
def _tpu_custom_call_abstract_eval(*_, out_avals, **__):
  return out_avals


def _aval_to_layout(aval):
  arange = np.arange(aval.ndim, dtype=np.dtype(np.int64))[::-1].copy()
  return ir.DenseIntElementsAttr.get(arange, type=ir.IndexType.get())


def _avals_to_layouts(avals):
  return ir.ArrayAttr.get([_aval_to_layout(a) for a in avals])


def _tpu_custom_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,  # pylint: disable=missing-function-docstring
    config: CustomCallBackendConfig,
    kernel_name: str | None,
    kernel_regeneration_metadata: bytes | None,
    out_avals: Any,
    input_output_aliases: tuple[tuple[int, int], ...],
) -> ...:
  i32_type = ir.IntegerType.get_signless(32)
  multiple_results = len(out_avals) > 1
  if multiple_results:
    result_type = ir.TupleType.get_tuple(
        [mlir.aval_to_ir_type(aval) for aval in out_avals]
    )
  else:
    result_type = mlir.aval_to_ir_type(out_avals[0])
  axis_context = ctx.module_context.axis_context
  if isinstance(axis_context, sharding_impls.SPMDAxisContext):
    if axis_context.manual_axes != frozenset(axis_context.mesh.axis_names):
      raise NotImplementedError(
          "Mosaic kernels cannot be automatically partitioned. Please wrap the"
          " call in a shard_map or xmap."
      )
  elif isinstance(axis_context, sharding_impls.ShardingContext):
    if axis_context.num_devices != 1:
      raise NotImplementedError(
          "Mosaic kernels cannot be automatically partitioned. Please wrap the"
          " call in a shard_map or xmap."
      )
  elif config.has_communication:
    raise NotImplementedError(
        "Replica lowering for Mosaic kernels not implemented."
    )
  call = stablehlo.CustomCallOp(
      [result_type],
      in_nodes,
      call_target_name=ir.StringAttr.get(b"tpu_custom_call"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(config.to_json()),
      api_version=ir.IntegerAttr.get(i32_type, 1),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=_avals_to_layouts(ctx.avals_in),
      result_layouts=_avals_to_layouts(ctx.avals_out),
      output_operand_aliases=ir.ArrayAttr.get([
          hlo.OutputOperandAlias.get(
              # if len(result_types) == 1 then the aliasing refers implicitly to
              # the only output.
              output_tuple_indices=[output_idx]
              if len(out_avals) > 1
              else [],
              operand_index=input_idx,
              operand_tuple_indices=[],
          )
          for input_idx, output_idx in input_output_aliases
      ]),
  )

  # Add kernel_name and kernel_regeneration_metadata as attributes to the
  # custom call op. This is because we do not want to pollute the backend_config
  # with this information.
  if kernel_name is not None:
    call.attributes["kernel_name"] = ir.StringAttr.get(kernel_name)
  if kernel_regeneration_metadata is not None:
    call.attributes["kernel_regeneration_metadata"] = ir.StringAttr.get(
        base64.b64encode(kernel_regeneration_metadata)
    )
  if multiple_results:
    results = [stablehlo.get_tuple_element(call, mlir.i32_attr(i))
               for i in range(len(out_avals))]
  else:
    results = call.results
  return results


mlir.register_lowering(tpu_custom_call_p, _tpu_custom_call_lowering,
                       platform="tpu")


def as_tpu_kernel(
    module: ir.Module,
    out_type: Any,
    *,
    cost_estimate: CostEstimate | None = None,
    backend: str | xla_client.Client = "tpu",
    device_type: str | None = None,
    kernel_name: str | None = None,
    kernel_regeneration_metadata: bytes | None = None,
    vmem_limit_bytes: int | None = None,
    flags: dict[str, bool | int | float] | None = None,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
) -> Callable[..., Any]:
  """Turns an MLIR Mosaic kernel into a JAX-compatible function."""
  # We use jax.jit to make sure we hit the fast compilation cache.
  some_tpu = jax.devices(backend)[0]
  device_kind = some_tpu.device_kind
  if not device_kind.startswith("TPU v"):
    raise ValueError(f"Unrecognized TPU device kind: {device_kind}.")
  if vmem_limit_bytes is not None and not isinstance(vmem_limit_bytes, int):
    raise ValueError(
        "vmem_limit_bytes must be an int: provided with a"
        f" {type(vmem_limit_bytes)}."
    )
  hardware_generation = int(device_kind[len("TPU v")])
  has_communication, has_custom_barrier = tpu.private_has_communication(
      module.operation
  )
  bytecode_buffer = io.BytesIO()
  module.operation.write_bytecode(bytecode_buffer, desired_version=0)
  asm = bytecode_buffer.getvalue()

  # TODO(amagni): Kernel name and regeneration metadata could alternatively be
  # added as a custom attribute to the MLIR call op rather than including them
  # in the backend_config.
  return _lowered_as_tpu_kernel(
      asm,
      out_type,
      needs_hlo_passes=_MOSAIC_ALLOW_HLO.value,
      needs_layout_passes=not device_type,
      device_type=device_type,
      has_communication=has_communication,
      has_custom_barrier=has_custom_barrier,
      kernel_name=kernel_name,
      kernel_regeneration_metadata=kernel_regeneration_metadata,
      cost_estimate=cost_estimate,
      vmem_limit_bytes=vmem_limit_bytes,
      flags=flags,
      input_output_aliases=input_output_aliases,
  )


def _lowered_as_tpu_kernel(
    lowered_module_asm: bytes,
    out_type: Any,
    *,
    cost_estimate: CostEstimate | None = None,
    needs_hlo_passes: bool = False,
    needs_layout_passes: bool = False,
    device_type: str | None = None,
    has_communication: bool = False,
    has_custom_barrier: bool = False,
    kernel_name: str | None = None,
    kernel_regeneration_metadata: bytes | None = None,
    vmem_limit_bytes: int | None = None,
    flags: dict[str, bool | int | float] | None = None,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
):
  """Turns a low-level MLIR Mosaic kernel into a JAX-compatible function."""
  unpack = False
  if not isinstance(out_type, collections.abc.Iterable):
    out_type = (out_type,)
    unpack = True
  out_avals = tuple(core.ShapedArray(ty.shape, ty.dtype) for ty in out_type)
  def apply_kernel(*args, collective_id: int | None = None):
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
    )
    result = tpu_custom_call_p.bind(
        *args,
        config=config,
        kernel_name=kernel_name,
        kernel_regeneration_metadata=kernel_regeneration_metadata,
        out_avals=out_avals,
        input_output_aliases=input_output_aliases,
    )
    return result[0] if unpack else result
  return jax.jit(apply_kernel, static_argnames=["collective_id"])


def dump_mlir(module: ir.Module, name: str):
  """A helper function to dump mosaic mlir module"""
  try:
    should_dump = FLAGS["xla_mosaic_dump_to"].value
  except KeyError:
    return
  if should_dump == "sponge":
    outdir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", None)
    if outdir:
      path = os.path.join(outdir, f"{time.time_ns()}-mosaic-dump-{name}.txt")
      with open(path, "w") as f:
        f.write(str(module))
