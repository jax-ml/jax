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

import jax
# flatbuffers needs importlib.util but fails to import it itself.
import importlib.util  # noqa: F401
from typing import List

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.mhlo as mhlo


from . import _pocketfft
from . import pocketfft_flatbuffers_py_generated as pd
import numpy as np

import flatbuffers
from jaxlib import xla_client

for _name, _value in _pocketfft.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform="cpu")

FftType = xla_client.FftType

flatbuffers_version_2 = hasattr(flatbuffers, "__version__")


def _pocketfft_descriptor(shape: List[int], dtype, fft_type: FftType,
                          fft_lengths: List[int]) -> bytes:
  n = len(shape)
  assert len(fft_lengths) >= 1
  assert len(fft_lengths) <= n, (fft_lengths, n)

  builder = flatbuffers.Builder(128)

  forward = fft_type in (FftType.FFT, FftType.RFFT)
  if fft_type == FftType.RFFT:
    pocketfft_type = pd.PocketFftType.R2C

    assert dtype in (np.float32, np.float64), dtype
    out_dtype = np.dtype(np.complex64 if dtype == np.float32 else np.complex128)
    pocketfft_dtype = (
        pd.PocketFftDtype.COMPLEX64
        if dtype == np.float32 else pd.PocketFftDtype.COMPLEX128)

    assert shape[-len(fft_lengths):] == fft_lengths, (shape, fft_lengths)
    out_shape = list(shape)
    out_shape[-1] = out_shape[-1] // 2 + 1

  elif fft_type == FftType.IRFFT:
    pocketfft_type = pd.PocketFftType.C2R
    assert np.issubdtype(dtype, np.complexfloating), dtype

    out_dtype = np.dtype(np.float32 if dtype == np.complex64 else np.float64)
    pocketfft_dtype = (
        pd.PocketFftDtype.COMPLEX64
        if dtype == np.complex64 else pd.PocketFftDtype.COMPLEX128)

    assert shape[-len(fft_lengths):-1] == fft_lengths[:-1]
    out_shape = list(shape)
    out_shape[-1] = fft_lengths[-1]
    assert (out_shape[-1] // 2 + 1) == shape[-1]
  else:
    pocketfft_type = pd.PocketFftType.C2C

    assert np.issubdtype(dtype, np.complexfloating), dtype
    out_dtype = dtype
    pocketfft_dtype = (
        pd.PocketFftDtype.COMPLEX64
        if dtype == np.complex64 else pd.PocketFftDtype.COMPLEX128)

    assert shape[-len(fft_lengths):] == fft_lengths, (shape, fft_lengths)
    out_shape = shape

  # PocketFft does not allow size 0 dimensions.
  if 0 in shape or 0 in out_shape:
    return b"", out_dtype, out_shape

  # Builds a PocketFftDescriptor flatbuffer. This descriptor is passed to the
  # C++ kernel to describe the FFT to perform.
  pd.PocketFftDescriptorStartShapeVector(builder, n)
  for d in reversed(shape if fft_type != FftType.IRFFT else out_shape):
    builder.PrependUint64(d)
  if flatbuffers_version_2:
    pocketfft_shape = builder.EndVector()
  else:
    pocketfft_shape = builder.EndVector(n)

  pd.PocketFftDescriptorStartStridesInVector(builder, n)
  stride = dtype.itemsize
  for d in reversed(shape):
    builder.PrependUint64(stride)
    stride *= d
  if flatbuffers_version_2:
    strides_in = builder.EndVector()
  else:
    strides_in = builder.EndVector(n)
  pd.PocketFftDescriptorStartStridesOutVector(builder, n)
  stride = out_dtype.itemsize
  for d in reversed(out_shape):
    builder.PrependUint64(stride)
    stride *= d
  if flatbuffers_version_2:
    strides_out = builder.EndVector()
  else:
    strides_out = builder.EndVector(n)

  pd.PocketFftDescriptorStartAxesVector(builder, len(fft_lengths))
  for d in range(len(fft_lengths)):
    builder.PrependUint32(n - d - 1)
  if flatbuffers_version_2:
    axes = builder.EndVector()
  else:
    axes = builder.EndVector(len(fft_lengths))

  scale = 1. if forward else (1. / np.prod(fft_lengths))
  pd.PocketFftDescriptorStart(builder)
  pd.PocketFftDescriptorAddDtype(builder, pocketfft_dtype)
  pd.PocketFftDescriptorAddFftType(builder, pocketfft_type)
  pd.PocketFftDescriptorAddShape(builder, pocketfft_shape)
  pd.PocketFftDescriptorAddStridesIn(builder, strides_in)
  pd.PocketFftDescriptorAddStridesOut(builder, strides_out)
  pd.PocketFftDescriptorAddAxes(builder, axes)
  pd.PocketFftDescriptorAddForward(builder, forward)
  pd.PocketFftDescriptorAddScale(builder, scale)
  descriptor = pd.PocketFftDescriptorEnd(builder)
  builder.Finish(descriptor)
  return builder.Output(), out_dtype, out_shape


def pocketfft_mhlo(a, dtype, *, fft_type: FftType, fft_lengths: List[int]):
  """PocketFFT kernel for CPU."""
  a_type = ir.RankedTensorType(a.type)
  n = len(a_type.shape)

  fft_lengths = list(fft_lengths)
  descriptor_bytes, out_dtype, out_shape = _pocketfft_descriptor(
      list(a_type.shape), dtype, fft_type, fft_lengths)

  if out_dtype == np.float32:
    out_type = ir.F32Type.get()
  elif out_dtype == np.float64:
    out_type = ir.F64Type.get()
  elif out_dtype == np.complex64:
    out_type = ir.ComplexType.get(ir.F32Type.get())
  elif out_dtype == np.complex128:
    out_type = ir.ComplexType.get(ir.F64Type.get())
  else:
    raise ValueError(f"Unknown output type {out_dtype}")

  if 0 in a_type.shape or 0 in out_shape:
    if xla_client._version >= 64:
      zero = mhlo.ConstOp(
          ir.DenseElementsAttr.get(np.array(0, dtype=out_dtype), type=out_type))
    else:
      zero = mhlo.ConstOp(ir.RankedTensorType.get([], out_type),
                          ir.DenseElementsAttr.get(np.array(0, dtype=out_dtype),
                                                   type=out_type))
    if jax._src.lib.mlir_api_version < 9:
      return mhlo.BroadcastOp(
          ir.RankedTensorType.get(out_shape, out_type),
          zero,
          ir.DenseElementsAttr.get(np.asarray(out_shape, np.int64))).result
    else:
      return mhlo.BroadcastOp(
          zero,
          ir.DenseElementsAttr.get(np.asarray(out_shape, np.int64))).result

  u8_type = ir.IntegerType.get_unsigned(8)
  if xla_client._version >= 64:
    descriptor = mhlo.ConstOp(
        ir.DenseElementsAttr.get(np.frombuffer(descriptor_bytes, dtype=np.uint8),
                                 type=u8_type))
  else:
    descriptor = mhlo.ConstOp(
        ir.RankedTensorType.get([len(descriptor_bytes)], u8_type),
        ir.DenseElementsAttr.get(np.frombuffer(descriptor_bytes, dtype=np.uint8),
                                 type=u8_type))
  layout = ir.DenseIntElementsAttr.get(np.arange(n - 1, -1, -1),
                                       type=ir.IndexType.get())
  return mhlo.CustomCallOp(
      [ir.RankedTensorType.get(out_shape, out_type)],
      [descriptor, a],
      call_target_name = ir.StringAttr.get("pocketfft"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(""),
      api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0], np.int64),
                                      type=ir.IndexType.get()),
          layout,
      ]),
      result_layouts=ir.ArrayAttr.get([layout])).result
