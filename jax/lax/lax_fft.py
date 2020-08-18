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

import numpy as np

from jax.abstract_arrays import ShapedArray
from jax.api import jit, vjp
from jax.core import Primitive
from jax.interpreters import xla
from jax.util import prod
from . import dtypes, lax
from .. import lib
from ..lib import xla_client
from ..interpreters import ad
from ..interpreters import batching

xops = xla_client.ops

__all__ = [
  "fft",
  "fft_p",
]

def _promote_to_complex(arg):
  dtype = dtypes.result_type(arg, np.complex64)
  # XLA's FFT op only supports C64 in jaxlib versions 0.1.47 and earlier.
  # TODO(phawkins): remove when minimum jaxlib version is 0.1.48 or newer.
  if lib.version <= (0, 1, 47) and dtype == np.complex128:
    dtype = np.complex64
  return lax.convert_element_type(arg, dtype)

def _promote_to_real(arg):
  dtype = dtypes.result_type(arg, np.float32)
  # XLA's FFT op only supports F32.
  # TODO(phawkins): remove when minimum jaxlib version is 0.1.48 or newer.
  if lib.version <= (0, 1, 47) and dtype == np.float64:
    dtype = np.float32
  return lax.convert_element_type(arg, dtype)

def fft(x, fft_type, fft_lengths):
  if fft_type == xla_client.FftType.RFFT:
    if np.iscomplexobj(x):
      raise ValueError("only real valued inputs supported for rfft")
    x = _promote_to_real(x)
  else:
    x = _promote_to_complex(x)
  if len(fft_lengths) == 0:
    # XLA FFT doesn't support 0-rank.
    return x
  fft_lengths = tuple(fft_lengths)
  return fft_p.bind(x, fft_type=fft_type, fft_lengths=fft_lengths)

def fft_impl(x, fft_type, fft_lengths):
  return xla.apply_primitive(fft_p, x, fft_type=fft_type, fft_lengths=fft_lengths)

_complex_dtype = lambda dtype: (np.zeros((), dtype) + np.zeros((), np.complex64)).dtype
_real_dtype = lambda dtype: np.zeros((), dtype).real.dtype
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
  return ShapedArray(shape, dtype)

def fft_translation_rule(c, x, fft_type, fft_lengths):
  return xops.Fft(x, fft_type, fft_lengths)

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
  dummy_primals = lax.full_like(t, 0.0, _real_dtype(t.dtype), dummy_shape)
  _, jvpfun = vjp(partial(_naive_rfft, fft_lengths=fft_lengths), dummy_primals)
  result, = jvpfun(t)
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
  full = partial(lax.full_like, t, dtype=t.dtype)
  mask = lax.concatenate(
      [full(1.0, shape=(1,)),
       full(2.0, shape=(n - 2 + is_odd,)),
       full(1.0, shape=(1 - is_odd,))],
      dimension=0)
  scale = 1 / prod(fft_lengths)
  out = scale * mask * x
  assert out.dtype == _complex_dtype(t.dtype), (out.dtype, t.dtype)
  return out

def fft_transpose_rule(t, fft_type, fft_lengths):
  if fft_type == xla_client.FftType.RFFT:
    result = _rfft_transpose(t, fft_lengths)
  elif fft_type == xla_client.FftType.IRFFT:
    result = _irfft_transpose(t, fft_lengths)
  else:
    result = fft(t, fft_type, fft_lengths)
  return result,

def fft_batching_rule(batched_args, batch_dims, fft_type, fft_lengths):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return fft(x, fft_type, fft_lengths), 0

fft_p = Primitive('fft')
fft_p.def_impl(fft_impl)
fft_p.def_abstract_eval(fft_abstract_eval)
xla.translations[fft_p] = fft_translation_rule
ad.deflinear(fft_p, fft_transpose_rule)
batching.primitive_batchers[fft_p] = fft_batching_rule
