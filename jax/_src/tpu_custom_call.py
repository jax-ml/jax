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

import base64
import collections.abc
from collections.abc import Sequence
import dataclasses
import functools
import io
from typing import Any, Callable

import jax
from jax import core
from jax.interpreters import mlir
from jax.interpreters import xla
from jax._src.config import config
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import mhlo
from jaxlib.mlir.dialects import stablehlo
from jaxlib.mlir.passmanager import PassManager
from jax._src.lib import tpu_mosaic
import numpy as np

# TODO(sharadmv): remove when minimum jaxlib version is bumped to >= 0.4.14.
if tpu_mosaic is None:
  raise ValueError("Cannot use Mosaic without a jaxlib >= 0.4.14.")
tpu = tpu_mosaic.tpu
apply_vector_layout = tpu_mosaic.apply_vector_layout
infer_memref_layout = tpu_mosaic.infer_memref_layout

config.define_bool_state(
    name="jax_mosaic_allow_hlo",
    default=False,
    help="Allow hlo dialects in Mosaic",
)

tpu_custom_call_p = core.Primitive("tpu_custom_call")
tpu_custom_call_p.def_impl(
    functools.partial(xla.apply_primitive, tpu_custom_call_p))
tpu_custom_call_p.multiple_results = True


@dataclasses.dataclass(frozen=True)
class CustomCallBackendConfig:
  """Represents an unserialized backend config for custom calls."""
  lowered_module_asm: bytes
  has_communication: bool
  collective_id: int | None

  # We omit the body while printing, because primitive params get embedded
  # in HLO metadata, and the body blows up its size.
  def __repr__(self):
    return "CustomCallBackendConfig(<omitted>)"

  def to_json(self):
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
    config.write(b"}}")
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
    if len(axis_context.device_assignment) != 1:
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
  if multiple_results:
    results = [stablehlo.GetTupleElementOp(call, mlir.i32_attr(i)).result
               for i in range(len(out_avals))]
  else:
    results = call.results
  return results


mlir.register_lowering(tpu_custom_call_p, _tpu_custom_call_lowering,
                       platform="tpu")


def _lower_tpu_kernel(module: ir.Module, hardware_generation: int) -> ir.Module:
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
    tpu.register_dialect(ctx)
    mhlo.register_mhlo_dialect(ctx)
    mhlo.register_mhlo_passes()
    # We'll mutate the module, so clone it.
    module = ir.Module.parse(
        module.operation.get_asm(binary=True, enable_debug_info=True)
    )

    if config.jax_mosaic_allow_hlo:
      # Run hlo dialect conversion: hlo -> linalg -> vector.
      pipeline = [
          "hlo-legalize-to-arithmetic",
          "func.func(hlo-legalize-to-linalg)",
          "func.func(linalg-vectorization)",
      ]
      PassManager.parse(f"builtin.module({','.join(pipeline)})").run(
          module.operation
      )

    infer_memref_layout.infer_module(module, hardware_generation)

    pipeline = [
        "canonicalize",
        "cse",
        "func.func(tpu-infer-vector-layout{sublane-count=8 lane-count=128})",
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    module.operation.verify()

    apply_vector_layout.apply(module, hardware_generation)
    module.operation.verify()

    PassManager.parse("builtin.module(canonicalize)").run(module.operation)

    vector_constants = []
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
            np.asarray(c, dtype=dtype).reshape(constant_type.shape))
    bytecode_buffer = io.BytesIO()
    module.operation.write_bytecode(bytecode_buffer, desired_version=0)
    return bytecode_buffer.getvalue(), tuple(vector_constants)


def as_tpu_kernel(
    module: ir.Module,
    out_type: Any,
    *,
    backend: str = "tpu",
) -> Callable[..., Any]:
  """Turns an MLIR Mosaic kernel into a JAX-compatible function."""
  # We use jax.jit to make sure we hit the fast compilation cache.
  some_tpu = jax.devices(backend)[0]
  device_kind = some_tpu.device_kind
  if device_kind.endswith(" pod"):
    device_kind = device_kind[:-len(" pod")]
  if device_kind.endswith(" lite"):
    device_kind = device_kind[:-len(" lite")]
  assert device_kind[:-1] == "TPU v", device_kind
  hardware_generation = int(device_kind[-1])
  has_communication, has_custom_barrier = tpu.private_has_communication(
      module.operation
  )
  lowered_module_asm, constants = _lower_tpu_kernel(module, hardware_generation)
  return _lowered_as_tpu_kernel(
      lowered_module_asm,
      out_type,
      constants,
      has_communication=has_communication,
      has_custom_barrier=has_custom_barrier,
  )


def _lowered_as_tpu_kernel(
    lowered_module_asm: bytes,
    out_type: Any,
    constants: Sequence[Any] = (),
    *,
    has_communication: bool = False,
    has_custom_barrier: bool = False,
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
        lowered_module_asm, has_communication, collective_id
    )
    result = tpu_custom_call_p.bind(
        *args, *constants, config=config, out_avals=out_avals)
    return result[0] if unpack else result
  return jax.jit(apply_kernel, static_argnames=["collective_id"])
