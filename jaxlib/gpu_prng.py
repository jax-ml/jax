# Copyright 2019 The JAX Authors.
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

import functools
from functools import partial
import importlib
import itertools
import operator

import jaxlib.mlir.ir as ir

from jaxlib import xla_client

from .hlo_helpers import custom_call

for cuda_module_name in [".cuda", "jax_cuda12_plugin"]:
  try:
    _cuda_prng = importlib.import_module(
        f"{cuda_module_name}._prng", package="jaxlib"
    )
  except ImportError:
    _cuda_prng = None
  else:
    break

if _cuda_prng:
  for _name, _value in _cuda_prng.registrations().items():
    # TODO(b/338022728): remove after 6 months, always api_version=1
    api_version = 1 if "_ffi" in _name else 0
    xla_client.register_custom_call_target(_name, _value, platform="CUDA",
                                           api_version=api_version)

try:
  from .rocm import _prng as _hip_prng  # pytype: disable=import-error
  for _name, _value in _hip_prng.registrations().items():
    # TODO(b/338022728): remove after 6 months, always api_version=1
    api_version = 1 if "_ffi" in _name else 0
    xla_client.register_custom_call_target(_name, _value, platform="ROCM",
                                           api_version=api_version)
except ImportError:
  _hip_prng = None

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)

def _threefry2x32_lowering(platform: str, keys, data,
                           length: int | ir.Value | None = None,
                           output_shape: ir.Value | None = None):
  """ThreeFry2x32 kernel for GPU.

  In presence of dynamic shapes, `length` is an `ir.Value` and `output_shape`
  is a 1D tensor describing the shape of the two outputs.
  """
  assert len(keys) == 2, keys
  assert len(data) == 2, data
  assert (ir.RankedTensorType(keys[0].type).element_type ==
          ir.IntegerType.get_unsigned(32)), keys[0].type

  typ = keys[0].type
  dims = ir.RankedTensorType(typ).shape

  for x in itertools.chain(keys, data):
    assert x.type == typ, (x.type, typ)
  ndims = len(dims)
  layout = tuple(range(ndims - 1, -1, -1))
  operand_layouts = [layout] * 4
  operands = [keys[0], keys[1], data[0], data[1]]

  if isinstance(length, int):
    result_shapes = None
  else:
    assert output_shape is not None
    # We pass separately the shapes of the outputs.
    result_shapes = [output_shape, output_shape]

  return custom_call(
      f"{platform}_threefry2x32_ffi",
      api_version=4,
      result_types=[typ, typ],
      operands=operands,
      operand_layouts=operand_layouts,
      result_layouts=[layout] * 2,
      result_shapes=result_shapes).results


cuda_threefry2x32 = partial(_threefry2x32_lowering, "cu")
rocm_threefry2x32 = partial(_threefry2x32_lowering, "hip")
