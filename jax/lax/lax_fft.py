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
from .. import dtypes
from ..interpreters import ad
from ..interpreters import batching


def fft(x, fft_type, fft_lengths=None):
  if fft_lengths is None:
    fft_lengths = x.shape
  elif len(fft_lengths) == 0:
    # XLA FFT doesn't support 0-rank.
    return x
  else:
    fft_lengths = tuple(fft_lengths)
  return fft_p.bind(x, fft_type=fft_type, fft_lengths=fft_lengths)

def fft_impl(x, fft_type, fft_lengths):
  return xla.apply_primitive(fft_p, x, fft_type=fft_type, fft_lengths=fft_lengths)

def fft_abstract_eval(x, fft_type, fft_lengths):
  if not dtypes.issubdtype(x.dtype, onp.complexfloating):
    raise TypeError("FFT requires complex inputs, got {}.".format(x.dtype.name))
  if x.dtype != onp.complex64:
    msg = "FFT is only implemented for complex64 types, got {}."
    raise NotImplementedError(msg.format(x.dtype.name))
  return ShapedArray(x.shape, x.dtype)

def fft_translation_rule(c, x, fft_type, fft_lengths):
  return c.Fft(x, fft_type, fft_lengths)

def fft_transpose_rule(t, fft_type, fft_lengths):
  return fft(t, fft_type, fft_lengths),

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
