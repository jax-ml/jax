# Copyright 2019 Google LLC
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


from functools import partial
from typing import Union, Sequence

import numpy as np

from jax._src.api import jit, linear_transpose, ShapeDtypeStruct
from jax.core import Primitive
from jax.interpreters import mlir
from jax.interpreters import xla
from jax._src.util import prod
from jax._src import dtypes
from jax import lax
from jax.interpreters import ad
from jax.interpreters import batching
from jax._src.lib.mlir.dialects import mhlo
from jax._src.lib import xla_client
from jax._src.lib import pocketfft
from jax._src.numpy.util import _promote_dtypes_complex, _promote_dtypes_inexact

__all__ = [
  "fft",
  "fft_p",
]

def _str_to_fft_type(s: str) -> xla_client.FftType:
  if s == "FFT":
    return xla_client.FftType.FFT
  elif s == "IFFT":
    return xla_client.FftType.IFFT
  elif s == "RFFT":
    return xla_client.FftType.RFFT
  elif s == "IRFFT":
    return xla_client.FftType.IRFFT
  else:
    raise ValueError(f"Unknown FFT type '{s}'")

@partial(jit, static_argnums=(1, 2))
def fft(x, fft_type: Union[xla_client.FftType, str], fft_lengths: Sequence[int]):
  if isinstance(fft_type, str):
    typ = _str_to_fft_type(fft_type)
  elif isinstance(fft_type, xla_client.FftType):
    typ = fft_type
  else:
    raise TypeError(f"Unknown FFT type value '{fft_type}'")

  if typ == xla_client.FftType.RFFT:
    if np.iscomplexobj(x):
      raise ValueError("only real valued inputs supported for rfft")
    x, = _promote_dtypes_inexact(x)
  else:
    x, = _promote_dtypes_complex(x)
  if len(fft_lengths) == 0:
    # XLA FFT doesn't support 0-rank.
    return x
  fft_lengths = tuple(fft_lengths)
  return fft_p.bind(x, fft_type=typ, fft_lengths=fft_lengths)

def _fft_impl(x, fft_type, fft_lengths):
  return xla.apply_primitive(fft_p, x, fft_type=fft_type, fft_lengths=fft_lengths)

_complex_dtype = lambda dtype: (np.zeros((), dtype) + np.zeros((), np.complex64)).dtype
_real_dtype = lambda dtype: np.finfo(dtype).dtype
_is_even = lambda x: x % 2 == 0

def fft_abstract_eval(x, fft_type, fft_lengths):
  if fft_type == xla_client.FftType.RFFT:
    shape = (x.shape[:-len(fft_lengths)] + fft_lengths[:-1]
             + (fft_lengths[-1] // 2 + 1,))
    dtype = _complex_dtype(x.dtype)
  elif fft_type == xla_client.FftType.IRFFT:
    shape = x.shape[:-len(fft_lengths)] + fft_lengths
    dtype = _real_dtype(x.dtype)
  else:
    shape = x.shape
    dtype = x.dtype
  return x.update(shape=shape, dtype=dtype)

def _fft_lowering(ctx, x, *, fft_type, fft_lengths):
  out_aval, = ctx.avals_out
  return [mhlo.FftOp(mlir.aval_to_ir_type(out_aval), x,
                     mhlo.FftTypeAttr.get(fft_type.name),
                     mlir.dense_int_elements(fft_lengths)).result]


def _fft_lowering_cpu(ctx, x, *, fft_type, fft_lengths):
  x_aval, = ctx.avals_in
  return [pocketfft.pocketfft_mhlo(x, x_aval.dtype, fft_type=fft_type,
                                   fft_lengths=fft_lengths)]


def _naive_rfft(x, fft_lengths):
  y = fft(x, xla_client.FftType.FFT, fft_lengths)
  n = fft_lengths[-1]
  return y[..., : n//2 + 1]

@partial(jit, static_argnums=1)
def _rfft_transpose(t, fft_lengths):
  # The transpose of RFFT can't be expressed only in terms of irfft. Instead of
  # manually building up larger twiddle matrices (which would increase the
  # asymptotic complexity and is also rather complicated), we rely JAX to
  # transpose a naive RFFT implementation.
  dummy_shape = t.shape[:-len(fft_lengths)] + fft_lengths
  dummy_primal = ShapeDtypeStruct(dummy_shape, _real_dtype(t.dtype))
  transpose = linear_transpose(
      partial(_naive_rfft, fft_lengths=fft_lengths), dummy_primal)
  result, = transpose(t)
  assert result.dtype == _real_dtype(t.dtype), (result.dtype, t.dtype)
  return result

def _irfft_transpose(t, fft_lengths):
  # The transpose of IRFFT is the RFFT of the cotangent times a scaling
  # factor and a mask. The mask scales the cotangent for the Hermitian
  # symmetric components of the RFFT by a factor of two, since these components
  # are de-duplicated in the RFFT.
  x = fft(t, xla_client.FftType.RFFT, fft_lengths)
  n = x.shape[-1]
  is_odd = fft_lengths[-1] % 2
  full = partial(lax.full_like, t, dtype=x.dtype)
  mask = lax.concatenate(
      [full(1.0, shape=(1,)),
       full(2.0, shape=(n - 2 + is_odd,)),
       full(1.0, shape=(1 - is_odd,))],
      dimension=0)
  scale = 1 / prod(fft_lengths)
  out = scale * lax.expand_dims(mask, range(x.ndim - 1)) * x
  assert out.dtype == _complex_dtype(t.dtype), (out.dtype, t.dtype)
  # Use JAX's convention for complex gradients
  # https://github.com/google/jax/issues/6223#issuecomment-807740707
  return lax.conj(out)

def _fft_transpose_rule(t, operand, fft_type, fft_lengths):
  if fft_type == xla_client.FftType.RFFT:
    result = _rfft_transpose(t, fft_lengths)
  elif fft_type == xla_client.FftType.IRFFT:
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
if pocketfft:
  mlir.register_lowering(fft_p, _fft_lowering_cpu, platform='cpu')
