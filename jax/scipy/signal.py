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

import scipy.signal as osp_signal

from .. import lax
from ..numpy import lax_numpy as jnp
from ..numpy.lax_numpy import (_wraps, _promote_dtypes_inexact)


# Note: we do not re-use the code from jax.numpy.convolve here, because the handling
# of padding differs slightly between the two implementations (particularly for
# mode='same').
def _convolve_nd(in1, in2, mode):
  if mode not in ["full", "same", "valid"]:
    raise ValueError("mode must be one of ['full', 'same', 'valid']")
  if in1.ndim != in2.ndim:
    raise ValueError("in1 and in2 must have the same number of dimensions")
  in1, in2 = _promote_dtypes_inexact(in1, in2)

  no_swap = all(s1 >= s2 for s1, s2 in zip(in1.shape, in2.shape))
  swap = all(s1 <= s2 for s1, s2 in zip(in1.shape, in2.shape))
  if not (no_swap or swap):
    raise ValueError("One input must be smaller than the other in every dimension.")

  shape_o = in2.shape
  if swap:
    in1, in2 = in2, in1
  shape = in2.shape
  in2 = in2[tuple(slice(None, None, -1) for s in shape)]

  if mode == 'valid':
    padding = [(0, 0) for s in shape]
  elif mode == 'same':
    padding = [(s - 1 - (s_o - 1) // 2, s - s_o + (s_o - 1) // 2)
               for (s, s_o) in zip(shape, shape_o)]
  elif mode == 'full':
    padding = [(s - 1, s - 1) for s in shape]

  strides = tuple(1 for s in shape)
  result = lax.conv_general_dilated(in1[None, None], in2[None, None], strides, padding)
  return result[0, 0]


@_wraps(osp_signal.convolve)
def convolve(in1, in2, mode='full', method='auto'):
  if method != 'auto':
    warnings.warn("convolve() ignores method argument")
  if jnp.issubdtype(in1.dtype, jnp.complexfloating) or jnp.issubdtype(in2.dtype, jnp.complexfloating):
    raise NotImplementedError("convolve() does not support complex inputs")
  if jnp.ndim(in1) != 1 or jnp.ndim(in2) != 1:
    raise ValueError(f"convolve() only supports 1-dimensional inputs.")
  return _convolve_nd(in1, in2, mode)


@_wraps(osp_signal.convolve2d)
def convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
  if boundary != 'fill' or fillvalue != 0:
    raise NotImplementedError("convolve2d() only supports boundary='fill', fillvalue=0")
  if jnp.issubdtype(in1.dtype, jnp.complexfloating) or jnp.issubdtype(in2.dtype, jnp.complexfloating):
    raise NotImplementedError("convolve2d() does not support complex inputs")
  if jnp.ndim(in1) != 2 or jnp.ndim(in2) != 2:
    raise ValueError(f"convolve2d() only supports 2-dimensional inputs.")
  return _convolve_nd(in1, in2, mode)


@_wraps(osp_signal.correlate)
def correlate(in1, in2, mode='full', method='auto'):
  if method != 'auto':
    warnings.warn("correlate() ignores method argument")
  if jnp.issubdtype(in1.dtype, jnp.complexfloating) or jnp.issubdtype(in2.dtype, jnp.complexfloating):
    raise NotImplementedError("correlate() does not support complex inputs")
  if jnp.ndim(in1) != 1 or jnp.ndim(in2) != 1:
    raise ValueError("correlate() only supports {ndim}-dimensional inputs.")
  return _convolve_nd(in1, in2[::-1], mode)


@_wraps(osp_signal.correlate)
def correlate2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
  if boundary != 'fill' or fillvalue != 0:
    raise NotImplementedError("correlate2d() only supports boundary='fill', fillvalue=0")
  if jnp.issubdtype(in1.dtype, jnp.complexfloating) or jnp.issubdtype(in2.dtype, jnp.complexfloating):
    raise NotImplementedError("correlate2d() does not support complex inputs")
  if jnp.ndim(in1) != 2 or jnp.ndim(in2) != 2:
    raise ValueError("correlate2d() only supports {ndim}-dimensional inputs.")
  return _convolve_nd(in1[::-1, ::-1], in2, mode)[::-1, ::-1]
