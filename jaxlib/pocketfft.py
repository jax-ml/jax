# Copyright 2020 Google LLC
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

from typing import List

from . import _pocketfft
from . import pocketfft_flatbuffers_py_generated as pd
import numpy as np

import flatbuffers
from jaxlib import xla_client

for _name, _value in _pocketfft.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform="cpu")

FftType = xla_client.FftType


def pocketfft(c, a, *, fft_type: FftType, fft_lengths: List[int]):
  """PocketFFT kernel for CPU."""
  shape = c.get_shape(a)
  n = len(shape.dimensions())
  dtype = shape.element_type()

  fft_lengths = list(fft_lengths)
  assert len(fft_lengths) >= 1
  assert len(fft_lengths) <= n, (fft_lengths, n)

  forward = fft_type in (FftType.FFT, FftType.RFFT)
  if fft_type == FftType.RFFT:
    pocketfft_type = pd.PocketFftType.R2C

    assert dtype in (np.float32, np.float64), dtype
    out_dtype = np.dtype(np.complex64 if dtype == np.float32 else np.complex128)
    pocketfft_dtype = (
        pd.PocketFftDtype.COMPLEX64
        if dtype == np.float32 else pd.PocketFftDtype.COMPLEX128)

    assert list(shape.dimensions())[-len(fft_lengths):] == fft_lengths, (
        shape, fft_lengths)
    out_shape = list(shape.dimensions())
    out_shape[-1] = out_shape[-1] // 2 + 1

  elif fft_type == FftType.IRFFT:
    pocketfft_type = pd.PocketFftType.C2R
    assert np.issubdtype(dtype, np.complexfloating), dtype

    out_dtype = np.dtype(np.float32 if dtype == np.complex64 else np.float64)
    pocketfft_dtype = (
        pd.PocketFftDtype.COMPLEX64
        if dtype == np.complex64 else pd.PocketFftDtype.COMPLEX128)

    assert list(shape.dimensions())[-len(fft_lengths):-1] == fft_lengths[:-1]
    out_shape = list(shape.dimensions())
    out_shape[-1] = fft_lengths[-1]
    assert (out_shape[-1] // 2 + 1) == shape.dimensions()[-1]
  else:
    pocketfft_type = pd.PocketFftType.C2C

    assert np.issubdtype(dtype, np.complexfloating), dtype
    out_dtype = dtype
    pocketfft_dtype = (
        pd.PocketFftDtype.COMPLEX64
        if dtype == np.complex64 else pd.PocketFftDtype.COMPLEX128)

    assert list(shape.dimensions())[-len(fft_lengths):] == fft_lengths, (
        shape, fft_lengths)
    out_shape = shape.dimensions()

  # PocketFft does not allow size 0 dimensions.
  if 0 in shape.dimensions() or 0 in out_shape:
    return xla_client.ops.Broadcast(
        xla_client.ops.Constant(c, np.array(0, dtype=out_dtype)), out_shape)

  strides_in = []
  stride = dtype.itemsize
  for d in reversed(shape.dimensions()):
    strides_in.append(stride)
    stride *= d
  strides_in = list(reversed(strides_in))

  strides_out = []
  stride = out_dtype.itemsize
  for d in reversed(out_shape):
    strides_out.append(stride)
    stride *= d
  strides_out = list(reversed(strides_out))

  descriptor = pd.PocketFftDescriptorT()
  descriptor.dtype = pocketfft_dtype
  descriptor.fftType = pocketfft_type
  descriptor.shape = list(shape.dimensions() if fft_type != FftType.IRFFT
                          else out_shape)
  descriptor.stridesIn = strides_in
  descriptor.stridesOut = strides_out
  descriptor.axes = [n - len(fft_lengths) + d for d in range(len(fft_lengths))]
  descriptor.forward = forward
  descriptor.scale = 1. if forward else (1. / np.prod(fft_lengths))
  builder = flatbuffers.Builder(128)
  builder.Finish(descriptor.Pack(builder))
  descriptor_bytes = builder.Output()

  return xla_client.ops.CustomCallWithLayout(
      c,
      b"pocketfft",
      operands=(
          xla_client.ops.Constant(
              c, np.frombuffer(descriptor_bytes, dtype=np.uint8)),
          a,
      ),
      shape_with_layout=xla_client.Shape.array_shape(
          out_dtype, out_shape, tuple(range(n - 1, -1, -1))),
      operand_shapes_with_layout=(
          xla_client.Shape.array_shape(
              np.dtype(np.uint8), (len(descriptor_bytes),), (0,)),
          xla_client.Shape.array_shape(dtype, shape.dimensions(),
                                       tuple(range(n - 1, -1, -1))),
      ))
