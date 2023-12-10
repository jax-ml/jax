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
from collections.abc import Sequence
import dataclasses
import functools
import io
import os
import re
import time
from typing import Any, Callable

from absl import flags
import jax
from jax import core
from jax._src import config
from jax._src.lib import tpu_mosaic
from jax._src.lib import xla_client
from jax._src.interpreters import mlir
from jax.interpreters import xla
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import mhlo
from jaxlib.mlir.dialects import stablehlo
from jaxlib.mlir.passmanager import PassManager
import numpy as np

FLAGS = flags.FLAGS
_MOSAIC_USE_CPP_PASSES = config.define_bool_state(
    name="mosaic_use_cpp_passes",
    default=True,
    help=(
        "Use C++ implementation for apply-vector-layout and infer-memref-layout"
        " passes (still a WIP)"
    ),
)

tpu = tpu_mosaic.tpu
apply_vector_layout = tpu_mosaic.apply_vector_layout
infer_memref_layout = tpu_mosaic.infer_memref_layout

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
    config.write(b"}")
    if self.device_type is not None:
      config.write(b', "device_type": ')
      config.write(
          ('"DEVICE_TYPE_' + self.device_type.upper() + '"').encode("ascii")
      )
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
  sharding_impls = jax._src.sharding_impls  # pylint: disable=protected-access
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
      output_operand_aliases=None,
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


_LOCATION_REGEX = re.compile(r'loc\("([a-zA-Z/]+)"\("(.*)":([0-9]+):[0-9]+\)\)')
_BUG_PROMPT = """
Please report a bug at: https://github.com/google/jax/issues/new?assignees=apaszke
"""
_OP_ERROR_PATTERN = re.compile(r"'.*' op (.+)")


def _run_pass_pipeline(passes: PassManager, module: ir.Module, pass_name: str):
  try:
    passes.run(module.operation)
    module.operation.verify()
  except ir.MLIRError as e:
    if e.error_diagnostics:
      d = e.error_diagnostics[0]
      diag_msg = d.message
      if match := re.match(_OP_ERROR_PATTERN, diag_msg):
        diag_msg = match.group(1)
      msg = ["Internal TPU kernel compiler error: " + diag_msg, '']
      # TODO(apaszke): Expose MLIR Location APIs instead of parsing
      if match := re.match(_LOCATION_REGEX, str(d.location)):
        name_stack, file, line = match.group(1), match.group(2), match.group(3)
        jax_func_name = name_stack[name_stack.rfind("/") + 1:]
        msg.append("The error was caused by:")
        msg.append(f"  `{jax_func_name}` called at {file}:{line}")
      for note in d.notes:
        note_msg = note.message
        if (op := note_msg.lstrip("see current operation: ")) is not note_msg:
          msg.append("The MLIR operation involved:")
          msg.append("  " + op)
      if len(e.error_diagnostics) > 1:
        msg.append("... additional diagnostics were skipped.")
      msg.append(_BUG_PROMPT)
      raise RuntimeError("\n".join(msg)) from None
    else:
      raise RuntimeError("Unspecified internal compiler error") from e
  dump_mlir(module, pass_name)


def _lower_tpu_kernel(
    module: ir.Module,
    hardware_generation: int,
    device_type: str | None,
) -> ir.Module:
  """Runs MLIR passes lowering the given module to an MLIR module.

  Args:
    module: The MLIR module to lower.
    hardware_generation: The TPU hardware generation to target.

  Returns:
    A pair containing an MLIR module implementing the kernel specified by the
    argument and a tuple of additional constant arguments that should be
    appended to the kernel invocation.

  """
  try:
    module.operation.verify()
  except ir.MLIRError as e:
    raise ValueError("The compiled module fails MLIR verification") from e

  with ir.Context() as ctx, ir.Location.unknown():
    vector_constants = []

    ctx.append_dialect_registry(mlir.upstream_dialects)
    ctx.load_all_available_dialects()
    tpu.register_dialect(ctx)
    mhlo.register_mhlo_dialect(ctx)
    mhlo.register_mhlo_passes()

    if not device_type:
      # We'll mutate the module, so clone it.
      module = ir.Module.parse(
          module.operation.get_asm(binary=True, enable_debug_info=True)
      )
      dump_mlir(module, "original")

      if _MOSAIC_ALLOW_HLO.value:
        # Run hlo dialect conversion: hlo -> linalg -> vector.
        pipeline = [
            "hlo-legalize-to-arithmetic",
            "func.func(hlo-legalize-to-linalg)",
            "func.func(linalg-vectorization)",
        ]
        pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
        _run_pass_pipeline(pipeline, module, "post-hlo-conversion")

      if _MOSAIC_USE_CPP_PASSES.value:
        pipeline = [
            (
                f"func.func(tpu-infer-memref-layout{{hardware-generation={hardware_generation}}})"
            ),
        ]
        pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
        _run_pass_pipeline(pipeline, module, "post-infer-memref-layout")
      else:
        infer_memref_layout.infer_module(module, hardware_generation)
        module.operation.verify()
        dump_mlir(module, "post-infer-memref-layout")

      pipeline = [
          "canonicalize",
          "cse",
          "func.func(tpu-infer-vector-layout{sublane-count=8 lane-count=128})",
      ]
      pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
      _run_pass_pipeline(pipeline, module, "post-infer-vector-layout")

      if _MOSAIC_USE_CPP_PASSES.value:
        pipeline = [
            (
                "func.func(tpu-apply-vector-layout{sublane-count=8"
                f" lane-count=128 hardware-generation={hardware_generation}}})"
            ),
        ]
        pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
        _run_pass_pipeline(pipeline, module, "post-apply-vector-layout")
      else:
        apply_vector_layout.apply(module, hardware_generation)
        module.operation.verify()
        dump_mlir(module, "post-apply-vector-layout")

      pipeline = PassManager.parse("builtin.module(canonicalize)")
      _run_pass_pipeline(pipeline, module, "pre-lower-to-llo")

      for f in module.body:
        if "vector_constants" not in f.attributes:
          continue
        if f.name.value != "main":
          raise NotImplementedError(
              "Only the main function can have non-splat vector constants"
          )
        constant_attrs = ir.ArrayAttr(f.attributes["vector_constants"])
        del f.attributes["vector_constants"]
        for c in constant_attrs:
          c = ir.DenseElementsAttr(c)
          constant_type = ir.VectorType(c.type)
          if constant_type.element_type == ir.IntegerType.get_signless(32):
            dtype = np.int32
          elif ir.F32Type.isinstance(constant_type.element_type):
            dtype = np.float32
          else:
            raise NotImplementedError(constant_type.element_type)
          if np.issubdtype(dtype, np.integer):
            c = ir.DenseIntElementsAttr(c)
          elif np.issubdtype(dtype, np.floating):
            c = ir.DenseFPElementsAttr(c)
          else:
            raise NotImplementedError(dtype)
          vector_constants.append(
              np.asarray(c, dtype=dtype).reshape(constant_type.shape)
          )

    bytecode_buffer = io.BytesIO()
    module.operation.write_bytecode(bytecode_buffer, desired_version=0)
    return bytecode_buffer.getvalue(), tuple(vector_constants)


def as_tpu_kernel(
    module: ir.Module,
    out_type: Any,
    *,
    cost_estimate: CostEstimate | None = None,
    backend: str | xla_client.Client = "tpu",
    device_type: str | None = None,
    kernel_name: str | None = None,
    kernel_regeneration_metadata: bytes | None = None,
    flags: dict[str, bool | int | float] | None = None,
) -> Callable[..., Any]:
  """Turns an MLIR Mosaic kernel into a JAX-compatible function."""
  # We use jax.jit to make sure we hit the fast compilation cache.
  some_tpu = jax.devices(backend)[0]
  device_kind = some_tpu.device_kind
  if not device_kind.startswith("TPU v"):
    raise ValueError(f"Unrecognized TPU device kind: {device_kind}.")
  hardware_generation = int(device_kind[len("TPU v")])
  has_communication, has_custom_barrier = tpu.private_has_communication(
      module.operation
  )
  lowered_module_asm, constants = _lower_tpu_kernel(
      module, hardware_generation, device_type=device_type
  )
  # TODO(amagni): Kernel name and regeneration metadata could alternatively be
  # added as a custom attribute to the MLIR call op rather than including them
  # in the backend_config.
  return _lowered_as_tpu_kernel(
      lowered_module_asm,
      out_type,
      constants,
      device_type=device_type,
      has_communication=has_communication,
      has_custom_barrier=has_custom_barrier,
      kernel_name=kernel_name,
      kernel_regeneration_metadata=kernel_regeneration_metadata,
      cost_estimate=cost_estimate,
      flags=flags,
  )


def _lowered_as_tpu_kernel(
    lowered_module_asm: bytes,
    out_type: Any,
    constants: Sequence[Any] = (),
    *,
    cost_estimate: CostEstimate | None = None,
    device_type: str | None = None,
    has_communication: bool = False,
    has_custom_barrier: bool = False,
    kernel_name: str | None = None,
    kernel_regeneration_metadata: bytes | None = None,
    flags: dict[str, bool | int | float] | None = None,
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
        flags,
    )
    result = tpu_custom_call_p.bind(
        *args,
        *constants,
        config=config,
        kernel_name=kernel_name,
        kernel_regeneration_metadata=kernel_regeneration_metadata,
        out_avals=out_avals,
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
