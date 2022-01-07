# Copyright 2021 Google LLC
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

import scipy.fftpack as osp_fft  # TODO use scipy.fft once scipy>=1.4.0 is used
from jax import lax, numpy as jnp
from jax._src.util import canonicalize_axis
from jax._src.numpy.util import _wraps

def _W4(N, k):
  return jnp.exp(-.5j * jnp.pi * k / N)

def _dct_interleave(x, axis):
  v0 = lax.slice_in_dim(x, None, None, 2, axis)
  v1 = lax.rev(lax.slice_in_dim(x, 1, None, 2, axis), (axis,))
  return lax.concatenate([v0, v1], axis)

def _dct_ortho_norm(out, axis):
  factor = lax.concatenate([lax.full((1,), 4, out.dtype), lax.full((out.shape[axis] - 1,), 2, out.dtype)], 0)
  factor = lax.expand_dims(factor, [a for a in range(out.ndim) if a != axis])
  return out / lax.sqrt(factor * out.shape[axis])

# Implementation based on
# John Makhoul: A Fast Cosine Transform in One and Two Dimensions (1980)

@_wraps(osp_fft.dct)
def dct(x, type=2, n=None, axis=-1, norm=None):
  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')

  axis = canonicalize_axis(axis, x.ndim)
  if n is not None:
    x = lax.pad(x, jnp.array(0, x.dtype),
                [(0, n - x.shape[axis] if a == axis else 0, 0)
                 for a in range(x.ndim)])

  N = x.shape[axis]
  v = _dct_interleave(x, axis)
  V = jnp.fft.fft(v, axis=axis)
  k = lax.expand_dims(jnp.arange(N), [a for a in range(x.ndim) if a != axis])
  out = V * _W4(N, k)
  out = 2 * out.real
  if norm == 'ortho':
    out = _dct_ortho_norm(out, axis)
  return out


def _dct2(x, axes, norm):
  axis1, axis2 = map(partial(canonicalize_axis, num_dims=x.ndim), axes)
  N1, N2 = x.shape[axis1], x.shape[axis2]
  v = _dct_interleave(_dct_interleave(x, axis1), axis2)
  V = jnp.fft.fftn(v, axes=axes)
  k1 = lax.expand_dims(jnp.arange(N1), [a for a in range(x.ndim) if a != axis1])
  k2 = lax.expand_dims(jnp.arange(N2), [a for a in range(x.ndim) if a != axis2])
  out = _W4(N1, k1) * (_W4(N2, k2) * V + _W4(N2, -k2) * jnp.roll(jnp.flip(V, axis=axis2), shift=1, axis=axis2))
  out = 2 * out.real
  if norm == 'ortho':
    return _dct_ortho_norm(_dct_ortho_norm(out, axis1), axis2)
  return out


@_wraps(osp_fft.dctn)
def dctn(x, type=2, s=None, axes=None, norm=None):
  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')

  if axes is None:
    axes = range(x.ndim)

  if len(axes) == 1:
    return dct(x, n=s[0] if s is not None else None, axis=axes[0], norm=norm)

  if s is not None:
    ns = {a: n for a, n in zip(axes, s)}
    pads = [(0, ns[a] - x.shape[a] if a in ns else 0, 0) for a in range(x.ndim)]
    x = lax.pad(x, jnp.array(0, x.dtype), pads)

  if len(axes) == 2:
    return _dct2(x, axes=axes, norm=norm)

  # compose high-D DCTs from 2D and 1D DCTs:
  for axes_block in [axes[i:i+2] for i in range(0, len(axes), 2)]:
    x = dctn(x, axes=axes_block, norm=norm)
  return x
