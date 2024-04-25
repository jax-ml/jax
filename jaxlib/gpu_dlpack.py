# Copyright 2024 The JAX Authors.
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

from __future__ import annotations

from functools import partial
import importlib

from jaxlib.mlir import ir

from jaxlib import xla_client

from .gpu_common_utils import GpuLibNotLinkedError
from .hlo_helpers import custom_call
from .hlo_helpers import shape_dtype_to_ir_type


for cuda_module_name in [".cuda", "jax_cuda12_plugin"]:
  try:
    _cuda_dlpack = importlib.import_module(
        f"{cuda_module_name}._dlpack", package="jaxlib"
    )
  except ImportError:
    _cuda_dlpack = None  # type: ignore[assignment]
  else:
    break

if _cuda_dlpack:
  for _name, _value in _cuda_dlpack.registrations().items():
    xla_client.register_custom_call_target(  # type: ignore[call-arg]
        _name, _value, platform="CUDA", api_version=1
    )


def dlpack_callback_lowering(impl, ctx, *args, callback, **kwargs):
  del kwargs  # Unused.

  if not impl:
    raise GpuLibNotLinkedError()

  def _default_layouts(shapes):
    return [list(reversed(range(len(shape)))) for shape in shapes]

  return custom_call(
      "dlpack_callback",
      operands=args,
      result_types=[
          shape_dtype_to_ir_type(aval.shape, aval.dtype)
          for aval in ctx.avals_out
      ],
      backend_config=dict(
          callback=ir.IntegerAttr.get(
              ir.IntegerType.get_signless(64), id(callback)
          ),
      ),
      api_version=4,
      operand_layouts=_default_layouts(aval.shape for aval in ctx.avals_in),  # pytype: disable=attribute-error
      result_layouts=_default_layouts(aval.shape for aval in ctx.avals_out),
  ).results


cuda_dlpack_callback = partial(dlpack_callback_lowering, _cuda_dlpack)
