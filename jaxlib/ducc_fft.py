# Copyright 2020 The JAX Authors.
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

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.stablehlo as hlo


from .hlo_helpers import custom_call
from .cpu import _ducc_fft
import numpy as np

from jaxlib import xla_client

for _name, _value in _ducc_fft.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform="cpu")

FftType = xla_client.FftType


_C2C = 0
_C2R = 1
_R2C = 2


def _dynamic_ducc_fft_descriptor(
    dtype, ndims: int, fft_type: FftType, fft_lengths: list[int]
) -> tuple[bytes]:
  assert len(fft_lengths) >= 1
  assert len(fft_lengths) <= ndims, (fft_lengths, ndims)

  forward = fft_type in (FftType.FFT, FftType.RFFT)
  is_double = np.finfo(dtype).dtype == np.float64
  if fft_type == FftType.RFFT:
    ducc_fft_type = _R2C
  elif fft_type == FftType.IRFFT:
    ducc_fft_type = _C2R
  else:
    ducc_fft_type = _C2C

  # Builds a PocketFftDescriptor flatbuffer. This descriptor is passed to the
  # C++ kernel to describe the FFT to perform.
  axes = [ndims - len(fft_lengths) + d for d in range(len(fft_lengths))]

  descriptor = _ducc_fft.dynamic_ducc_fft_descriptor(
    ndims=ndims,
    is_double=is_double,
    fft_type=ducc_fft_type,
    axes=axes,
    forward=forward)

  return descriptor


def dynamic_ducc_fft_hlo(
    result_type: ir.Type,
    input: ir.Value, *,
    input_dtype: np.dtype, ndims:int, input_shape: ir.Value,
    strides_in: ir.Value, strides_out: ir.Value, scale: ir.Value,
    fft_type: FftType, fft_lengths: list[int], result_shape: ir.Value):
  """DUCC FFT kernel for CPU, with support for dynamic shapes."""
  a_type = ir.RankedTensorType(input.type)

  fft_lengths = list(fft_lengths)
  descriptor_bytes = _dynamic_ducc_fft_descriptor(
      input_dtype, ndims, fft_type, fft_lengths)

  # PocketFft does not allow size 0 dimensions, but we handled this in fft.py
  assert 0 not in a_type.shape

  u8_type = ir.IntegerType.get_unsigned(8)
  descriptor = hlo.ConstantOp(
      ir.DenseElementsAttr.get(
          np.frombuffer(descriptor_bytes, dtype=np.uint8), type=u8_type)).result
  layout = tuple(range(ndims - 1, -1, -1))
  return custom_call(
      "dynamic_ducc_fft",
      [result_type],
      [descriptor, input, input_shape, strides_in, strides_out, scale],
      operand_layouts=[[0], layout, [0], [0], [0], [0]],
      result_layouts=[layout],
      result_shapes=[result_shape])
