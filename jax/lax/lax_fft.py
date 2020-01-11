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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from jax.abstract_arrays import ShapedArray
from jax.core import Primitive
from jax.interpreters import xla
from . import dtypes, lax
from ..lib.xla_bridge import xla_client
from ..interpreters import ad
from ..interpreters import batching


def _promote_to_complex(arg):
  dtype = onp.result_type(arg, onp.complex64)
  # XLA's FFT op only supports C64.
  if dtype == onp.complex128:
    dtype = onp.complex64
  return lax.convert_element_type(arg, dtype)

def _promote_to_real(arg):
  dtype = onp.result_type(arg, onp.float64)
  # XLA's FFT op only supports F32.
  if dtype == onp.float64:
    dtype = onp.float32
  return lax.convert_element_type(arg, dtype)

def fft(x, fft_type, fft_lengths):
  if fft_type == xla_client.FftType.RFFT:
    if onp.iscomplexobj(x):
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

_complex_dtype = lambda dtype: (onp.zeros((), dtype) + onp.zeros((), onp.complex64)).dtype
_real_dtype = lambda dtype: onp.zeros((), dtype).real.dtype
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
  return c.Fft(x, fft_type, fft_lengths)

def _prod(xs):
  result = 1
  for x in xs:
    result *= x
  return result

def fft_transpose_rule(t, fft_type, fft_lengths):
  if fft_type == xla_client.FftType.RFFT:
    scale = _prod(fft_lengths)
    result = scale * fft(
        t.conj(), xla_client.FftType.IRFFT, fft_lengths).conj()
  elif fft_type == xla_client.FftType.IRFFT:
    scale = 1 / _prod(fft_lengths)
    result = scale * fft(t, xla_client.FftType.RFFT, fft_lengths)
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
