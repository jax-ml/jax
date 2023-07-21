# Copyright 2021 The JAX Authors.
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

from collections.abc import Sequence
from functools import partial
import math
from typing import Optional

import scipy.fft as osp_fft
from jax import lax
import jax.numpy as jnp
from jax._src.util import canonicalize_axis
from jax._src.numpy.util import _wraps, promote_dtypes_complex
from jax._src.typing import Array

def _W4(N: int, k: Array) -> Array:
  N_arr, k = promote_dtypes_complex(N, k)
  return jnp.exp(-.5j * jnp.pi * k / N_arr)

def _dct_interleave(x: Array, axis: int) -> Array:
  v0 = lax.slice_in_dim(x, None, None, 2, axis)
  v1 = lax.rev(lax.slice_in_dim(x, 1, None, 2, axis), (axis,))
  return lax.concatenate([v0, v1], axis)

def _dct_ortho_norm(out: Array, axis: int) -> Array:
  factor = lax.concatenate([lax.full((1,), 4, out.dtype), lax.full((out.shape[axis] - 1,), 2, out.dtype)], 0)
  factor = lax.expand_dims(factor, [a for a in range(out.ndim) if a != axis])
  return out / lax.sqrt(factor * out.shape[axis])

# Implementation based on
# John Makhoul: A Fast Cosine Transform in One and Two Dimensions (1980)

@_wraps(osp_fft.dct)
def dct(x: Array, type: int = 2, n: Optional[int] = None,
        axis: int = -1, norm: Optional[str] = None) -> Array:
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
  k = lax.expand_dims(jnp.arange(N, dtype=V.real.dtype), [a for a in range(x.ndim) if a != axis])
  out = V * _W4(N, k)
  out = 2 * out.real
  if norm == 'ortho':
    out = _dct_ortho_norm(out, axis)
  return out


def _dct2(x: Array, axes: Sequence[int], norm: Optional[str]) -> Array:
  axis1, axis2 = map(partial(canonicalize_axis, num_dims=x.ndim), axes)
  N1, N2 = x.shape[axis1], x.shape[axis2]
  v = _dct_interleave(_dct_interleave(x, axis1), axis2)
  V = jnp.fft.fftn(v, axes=axes)
  k1 = lax.expand_dims(jnp.arange(N1, dtype=V.dtype),
                       [a for a in range(x.ndim) if a != axis1])
  k2 = lax.expand_dims(jnp.arange(N2, dtype=V.dtype),
                       [a for a in range(x.ndim) if a != axis2])
  out = _W4(N1, k1) * (_W4(N2, k2) * V + _W4(N2, -k2) * jnp.roll(jnp.flip(V, axis=axis2), shift=1, axis=axis2))
  out = 2 * out.real
  if norm == 'ortho':
    return _dct_ortho_norm(_dct_ortho_norm(out, axis1), axis2)
  return out


@_wraps(osp_fft.dctn)
def dctn(x: Array, type: int = 2,
         s: Optional[Sequence[int]]=None,
         axes: Optional[Sequence[int]] = None,
         norm: Optional[str] = None) -> Array:
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


@_wraps(osp_fft.dct)
def idct(x: Array, type: int = 2, n: Optional[int] = None,
        axis: int = -1, norm: Optional[str] = None) -> Array:
  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')

  axis = canonicalize_axis(axis, x.ndim)
  if n is not None:
    x = lax.pad(x, jnp.array(0, x.dtype),
                [(0, n - x.shape[axis] if a == axis else 0, 0)
                 for a in range(x.ndim)])
  N = x.shape[axis]
  x = x.astype(jnp.float32)
  if norm is None:
    x = _dct_ortho_norm(x, axis)
  x = _dct_ortho_norm(x, axis)


  k = lax.expand_dims(jnp.arange(N, dtype=jnp.float32), [a for a in range(x.ndim) if a != axis])
  # everything is complex from here...
  w4 = _W4(N,k)
  x = x.astype(w4.dtype)
  x = x / (_W4(N, k))
  x = x * 2 * N

  x = jnp.fft.ifft(x, axis=axis)
  # convert back to reals..
  out = _dct_deinterleave(x.real, axis)
  return out

@_wraps(osp_fft.idctn)
def idctn(x: Array, type: int = 2,
         s: Optional[Sequence[int]]=None,
         axes: Optional[Sequence[int]] = None,
         norm: Optional[str] = None) -> Array:
  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')

  if axes is None:
    axes = range(x.ndim)

  if len(axes) == 1:
    return idct(x, n=s[0] if s is not None else None, axis=axes[0], norm=norm)

  if s is not None:
    ns = {a: n for a, n in zip(axes, s)}
    pads = [(0, ns[a] - x.shape[a] if a in ns else 0, 0) for a in range(x.ndim)]
    x = lax.pad(x, jnp.array(0, x.dtype), pads)

  # compose high-D DCTs from 1D DCTs:
  for axis in axes:
    x = idct(x, axis=axis, norm=norm)
  return x


def _dct_deinterleave(x: Array, axis: int) -> Array:
  empty_slice = slice(None, None, None)
  ix0 = tuple([slice(None, math.ceil(x.shape[axis]/2), 1) if i == axis else empty_slice for i in range(len(x.shape))])
  ix1  = tuple([slice(math.ceil(x.shape[axis]/2), None, 1) if i == axis else empty_slice for i in range(len(x.shape))])
  v0 = x[ix0]
  v1 = lax.rev(x[ix1], (axis,))
  out = jnp.zeros(x.shape, dtype=x.dtype)
  evens = tuple([slice(None, None, 2) if i == axis else empty_slice for i in range(len(x.shape))])
  odds = tuple([slice(1, None, 2) if i == axis else empty_slice for i in range(len(x.shape))])
  out =  out.at[evens].set(v0)
  out = out.at[odds].set(v1)
  return out
