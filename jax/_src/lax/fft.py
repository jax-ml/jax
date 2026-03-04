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

from collections.abc import Sequence
import enum
import math

import numpy as np

from jax._src import dispatch
from jax._src import dtypes
from jax._src.api import jit
from jax._src.core import Primitive, is_constant_shape
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lib.mlir.dialects import hlo

__all__ = [
  "fft",
  "fft_p",
]

class FftType(enum.IntEnum):
  "Describes which FFT operation to perform."

  FFT = 0
  "Forward complex-to-complex FFT."

  IFFT = 1
  "Inverse complex-to-complex FFT."

  RFFT = 2
  "Forward real-to-complex FFT."

  IRFFT = 3
  "Inverse real-to-complex FFT."


def _str_to_fft_type(s: str) -> FftType:
  if s in ("fft", "FFT"):
    return FftType.FFT
  elif s in ("ifft", "IFFT"):
    return FftType.IFFT
  elif s in ("rfft", "RFFT"):
    return FftType.RFFT
  elif s in ("irfft", "IRFFT"):
    return FftType.IRFFT
  else:
    raise ValueError(f"Unknown FFT type '{s}'")

@jit(static_argnums=(1, 2))
def fft(x, fft_type: FftType | str, fft_lengths: Sequence[int]):
  if isinstance(fft_type, str):
    typ = _str_to_fft_type(fft_type)
  elif isinstance(fft_type, FftType):
    typ = fft_type
  else:
    raise TypeError(f"Unknown FFT type value '{fft_type}'")

  if typ == FftType.RFFT:
    if np.iscomplexobj(x):
      raise ValueError("only real valued inputs supported for rfft")
    x = lax.convert_element_type(x, dtypes.to_inexact_dtype(dtypes.dtype(x)))
  else:
    x = lax.convert_element_type(x, dtypes.to_complex_dtype(dtypes.dtype(x)))
  if len(fft_lengths) == 0:
    # XLA FFT doesn't support 0-rank.
    return x
  fft_lengths = tuple(fft_lengths)
  return fft_p.bind(x, fft_type=typ, fft_lengths=fft_lengths)

def _fft_impl(x, fft_type, fft_lengths):
  return dispatch.apply_primitive(fft_p, x, fft_type=fft_type, fft_lengths=fft_lengths)

_complex_dtype = lambda dtype: (np.zeros((), dtype) + np.zeros((), np.complex64)).dtype
_real_dtype = lambda dtype: np.finfo(dtype).dtype

def fft_abstract_eval(x, fft_type, fft_lengths):
  if len(fft_lengths) > x.ndim:
    raise ValueError(f"FFT input shape {x.shape} must have at least as many "
                    f"input dimensions as fft_lengths {fft_lengths}.")
  if fft_type == FftType.RFFT:
    if x.dtype not in (np.float32, np.float64):
      raise ValueError(f"RFFT input must be float32 or float64, got {x.dtype}")
    if x.shape[-len(fft_lengths):] != fft_lengths:
      raise ValueError(f"RFFT input shape {x.shape} minor dimensions must "
                      f"be equal to fft_lengths {fft_lengths}")
    shape = (x.shape[:-len(fft_lengths)] + fft_lengths[:-1]
             + (fft_lengths[-1] // 2 + 1,))
    dtype = _complex_dtype(x.dtype)
  elif fft_type == FftType.IRFFT:
    if not np.issubdtype(x.dtype, np.complexfloating):
      raise ValueError("IRFFT input must be complex64 or complex128, got "
                       f"{x.dtype}")
    if x.shape[-len(fft_lengths):-1] != fft_lengths[:-1]:
      raise ValueError(f"IRFFT input shape {x.shape} minor dimensions must "
                      "be equal to all except the last fft_length, got "
                      f"{fft_lengths=}")
    shape = x.shape[:-len(fft_lengths)] + fft_lengths
    dtype = _real_dtype(x.dtype)
  else:
    if not np.issubdtype(x.dtype, np.complexfloating):
      raise ValueError("FFT input must be complex64 or complex128, got "
                       f"{x.dtype}")
    if x.shape[-len(fft_lengths):] != fft_lengths:
      raise ValueError(f"FFT input shape {x.shape} minor dimensions must "
                      f"be equal to fft_lengths {fft_lengths}")
    shape = x.shape
    dtype = x.dtype
  return x.update(shape=shape, dtype=dtype, vma=x.vma)

def _fft_lowering(ctx, x, *, fft_type, fft_lengths):
  if not is_constant_shape(fft_lengths):
    # TODO: https://github.com/openxla/stablehlo/issues/1366
    raise NotImplementedError("Shape polymorphism for FFT with non-constant fft_length is not implemented for TPU and GPU")
  return [
      hlo.FftOp(x, hlo.FftTypeAttr.get(fft_type.name),
                mlir.dense_int_array(fft_lengths)).result
  ]


@jit(static_argnums=1)
def _rfft_transpose(t, fft_lengths):
  # The transpose can be computed directly using irfft with a mask to account
  # for Hermitian redundancy. Mask values are 1 for DC and Nyquist, 2 for others.
  n = fft_lengths[-1]
  is_odd = n % 2
  m = t.shape[-1]
  mask = lax.full_like(t, 2.0, shape=(m,))
  mask = slicing.dynamic_update_slice(
      mask, lax.full_like(t, 1.0, shape=(1,)), (0,)
  )
  if not is_odd:
    mask = slicing.dynamic_update_slice(
        mask, lax.full_like(t, 1.0, shape=(1,)), (m - 1,)
    )

  N = math.prod(fft_lengths)
  # The mask is along the last dimension.
  mask = lax.expand_dims(mask, range(t.ndim - 1))
  out = N * fft(lax.conj(t) / mask, FftType.IRFFT, fft_lengths)
  assert out.dtype == _real_dtype(t.dtype), (out.dtype, t.dtype)
  return out

def _irfft_transpose(t, fft_lengths):
  # The transpose of IRFFT is the RFFT of the cotangent times a scaling
  # factor and a mask. The mask scales the cotangent for the Hermitian
  # symmetric components of the RFFT by a factor of two, since these components
  # are de-duplicated in the RFFT.
  x = fft(t, FftType.RFFT, fft_lengths)
  n = x.shape[-1]
  is_odd = fft_lengths[-1] % 2
  mask = lax.full_like(t, 2.0, shape=(n,), dtype=x.dtype)
  mask = slicing.dynamic_update_slice(
      mask, lax.full_like(t, 1.0, shape=(1,), dtype=x.dtype), (0,)
  )
  if not is_odd:
    mask = slicing.dynamic_update_slice(
        mask, lax.full_like(t, 1.0, shape=(1,), dtype=x.dtype), (n - 1,)
    )

  scale = 1 / math.prod(fft_lengths)
  out = scale * lax.expand_dims(mask, range(x.ndim - 1)) * x
  assert out.dtype == _complex_dtype(t.dtype), (out.dtype, t.dtype)
  # Use JAX's convention for complex gradients
  # https://github.com/jax-ml/jax/issues/6223#issuecomment-807740707
  return lax.conj(out)

def _fft_transpose_rule(t, operand, fft_type, fft_lengths):
  if fft_type == FftType.RFFT:
    result = _rfft_transpose(t, fft_lengths)
  elif fft_type == FftType.IRFFT:
    result = _irfft_transpose(t, fft_lengths)
  else:
    result = fft(t, fft_type, fft_lengths)
  return result,

def _fft_batching_rule(batched_args, batch_dims, fft_type, fft_lengths):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return fft(x, fft_type, fft_lengths), 0

fft_p = Primitive('fft')
fft_p.def_impl(_fft_impl)
fft_p.def_abstract_eval(fft_abstract_eval)
mlir.register_lowering(fft_p, _fft_lowering)
ad.deflinear2(fft_p, _fft_transpose_rule)
batching.primitive_batchers[fft_p] = _fft_batching_rule
