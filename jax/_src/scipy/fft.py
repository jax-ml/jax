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

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import math
import operator

import numpy as np

from jax._src import dtypes
from jax._src import lax
from jax._src import numpy as jnp
from jax._src.numpy import fft as jnp_fft
from jax._src.numpy.util import (
    promote_dtypes_complex, promote_dtypes_inexact, ensure_arraylike)
from jax._src.util import canonicalize_axis, canonicalize_axis_tuple
from jax._src.typing import Array, ArrayLike

def _W4(N: int, k: Array) -> Array:
  N_arr, k = promote_dtypes_complex(N, k)
  return jnp.exp(-.5j * np.pi * k / N_arr)

def _dct_interleave(x: Array, axis: int) -> Array:
  v0 = lax.slice_in_dim(x, None, None, 2, axis)
  v1 = lax.rev(lax.slice_in_dim(x, 1, None, 2, axis), (axis,))
  return lax.concatenate([v0, v1], axis)

def _dct_ortho_norm(out: Array, axis: int) -> Array:
  factor = lax.concatenate([lax.full((1,), 4, out.dtype), lax.full((out.shape[axis] - 1,), 2, out.dtype)], 0)
  factor = lax.expand_dims(factor, [a for a in range(out.ndim) if a != axis])
  return out / lax.sqrt(factor * out.shape[axis])

def _dst_sign(x: Array, axes: Sequence[int]) -> Array:
  """Multiplies the input elementwise by ``(-1)**sum(indices along axes)``."""
  for axis in axes:
    shape = tuple(s if i == axis else 1 for i, s in enumerate(x.shape))
    parity = lax.broadcasted_iota(np.int32, shape, axis) % 2
    x = jnp.where(parity == 1, -x, x)
  return x

def _pad_axes(x: Array, s: Sequence[int], axes: Sequence[int]) -> Array:
  """Zero-pads or truncates the input to length ``s[i]`` along ``axes[i]``."""
  ns = dict(zip(axes, s))
  pads = [(0, ns[a] - x.shape[a] if a in ns else 0, 0) for a in range(x.ndim)]
  return lax.pad(x, jnp.array(0, x.dtype), pads)

def _dctn_s_axes(ndim: int, s: Sequence[int] | None, axes: Sequence[int] | None
                 ) -> tuple[list[int] | None, tuple[int, ...]]:
  """Normalizes the ``s`` and ``axes`` arguments of the n-dimensional transforms."""
  if s is not None:
    try:
      s = list(s)
    except TypeError:
      assert not isinstance(s, Sequence)
      s = [operator.index(s)]
    if len(s) > ndim:
      raise ValueError(
          f"s must have at most x.ndim ({ndim}) elements, got {len(s)}"
      )
  if axes is None:
    axes = tuple(range(ndim - len(s) if s is not None else 0, ndim))
  else:
    axes = canonicalize_axis_tuple(axes, ndim)
  return s, axes

# Implementation based on
# John Makhoul: A Fast Cosine Transform in One and Two Dimensions (1980)


def dct(x: Array, type: int = 2, n: int | None = None,
        axis: int = -1, norm: str | None = None) -> Array:
  """Computes the discrete cosine transform of the input

  JAX implementation of :func:`scipy.fft.dct`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    n: integer, default = x.shape[axis]. The length of the transform.
      If larger than ``x.shape[axis]``, the input will be zero-padded, if
      smaller, the input will be truncated.
    axis: integer, default=-1. The axis along which the dct will be performed.
    norm: string. The normalization mode: one of ``[None, "backward", "ortho"]``.
      The default is ``None``, which is equivalent to ``"backward"``.

  Returns:
    array containing the discrete cosine transform of x

  See Also:
    - :func:`jax.scipy.fft.dctn`: multidimensional DCT
    - :func:`jax.scipy.fft.idct`: inverse DCT
    - :func:`jax.scipy.fft.idctn`: multidimensional inverse DCT

  Examples:
    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x))
    [[ 6.43  3.56 -2.86]
     [-1.75  1.55 -1.4 ]
     [ 1.33 -2.01 -0.82]]

    When ``n`` smaller than ``x.shape[axis]``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x, n=2))
    [[ 7.3  -0.57]
     [ 0.19 -0.36]
     [-0.   -1.4 ]]

    When ``n`` smaller than ``x.shape[axis]`` and ``axis=0``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x, n=2, axis=0))
    [[ 3.09  4.4  -2.81]
     [ 2.41  2.62  0.76]]

    When ``n`` larger than ``x.shape[axis]`` and ``axis=1``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x, n=4, axis=1))
    [[ 6.43  4.88  0.04 -3.3 ]
     [-1.75  0.73  1.01 -2.18]
     [ 1.33 -1.05 -2.34 -0.07]]
  """
  x = ensure_arraylike("dct", x)

  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.dct: {norm=!r} is not implemented")

  if dtypes.issubdtype(x.dtype, np.complexfloating):
    return lax.complex(
      dct(x.real, type=type, n=n, norm=norm, axis=axis),
      dct(x.imag, type=type, n=n, norm=norm, axis=axis),
    )

  axis = canonicalize_axis(axis, x.ndim)
  if n is not None:
    x = _pad_axes(x, [n], [axis])

  N = x.shape[axis]
  v = _dct_interleave(x, axis)
  # rfft accepts only float32/float64, so narrow and integer inputs are upcast.
  dtype = dtypes.to_inexact_dtype(v.dtype)
  v = v.astype(dtype if dtype in (np.float32, np.float64) else np.float32)
  # The twiddled half spectrum carries the whole transform: the bins above
  # N//2 are its negated imaginary part, read in reverse.
  V = jnp_fft.rfft(v, axis=axis)
  k_shape = tuple(s if i == axis else 1 for i, s in enumerate(V.shape))
  k = lax.broadcasted_iota(V.real.dtype, k_shape, axis)
  V = V * _W4(N, k)
  tail = lax.rev(lax.slice_in_dim(-V.imag, 1, (N + 1) // 2, axis=axis), (axis,))
  out = 2 * lax.concatenate([V.real, tail], axis)
  if norm == 'ortho':
    out = _dct_ortho_norm(out, axis)
  return out


def _dct2(x: Array, axes: Sequence[int], norm: str | None) -> Array:
  axis1, axis2 = map(partial(canonicalize_axis, num_dims=x.ndim), axes)
  N1, N2 = x.shape[axis1], x.shape[axis2]
  v = _dct_interleave(_dct_interleave(x, axis1), axis2)
  V = jnp_fft.fftn(v, axes=axes)
  k1 = lax.expand_dims(jnp.arange(N1, dtype=V.dtype),
                       [a for a in range(x.ndim) if a != axis1])
  k2 = lax.expand_dims(jnp.arange(N2, dtype=V.dtype),
                       [a for a in range(x.ndim) if a != axis2])
  out = _W4(N1, k1) * (_W4(N2, k2) * V + _W4(N2, -k2) * jnp.roll(jnp.flip(V, axis=axis2), shift=1, axis=axis2))
  out = 2 * out.real
  if norm == 'ortho':
    return _dct_ortho_norm(_dct_ortho_norm(out, axis1), axis2)
  return out


def dctn(x: Array, type: int = 2,
         s: Sequence[int] | None=None,
         axes: Sequence[int] | None = None,
         norm: str | None = None) -> Array:
  """Computes the multidimensional discrete cosine transform of the input

  JAX implementation of :func:`scipy.fft.dctn`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    s: integer or sequence of integers. Specifies the shape of the result. If
      not specified, it will default to the shape of ``x`` along the specified
      ``axes``.
    axes: integer or sequence of integers. Specifies the axes along which the
      transform will be computed. If not given, the last ``len(s)`` axes are
      used, or all axes if ``s`` is also not specified.
    norm: string. The normalization mode: one of
      ``[None, "backward", "ortho"]``. The default is ``None``, which is
      equivalent to ``"backward"``.

  Returns:
    array containing the discrete cosine transform of x

  See Also:
    - :func:`jax.scipy.fft.dct`: one-dimensional DCT
    - :func:`jax.scipy.fft.idct`: one-dimensional inverse DCT
    - :func:`jax.scipy.fft.idctn`: multidimensional inverse DCT

  Examples:

    ``jax.scipy.fft.dctn`` computes the transform along both the axes by default
    when ``axes`` argument is ``None`` and ``s`` is also ``None``.

    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x))
    [[ 12.01   6.2  -10.17]
     [  8.84   9.65  -3.54]
     [ 11.25  -1.54  -0.88]]

    When ``s=[2]``, the transform will be computed only along the last axis,
    with its dimension padded or truncated to size ``2``:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x, s=[2]))
    [[ 7.3  -0.57]
     [ 0.19 -0.36]
     [-0.   -1.4 ]]

    When ``s=[2]`` and ``axes=[0]``, the transform will be computed only along
    the specified axis, with its dimension padded or truncated to size ``2``:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x, s=[2], axes=[0]))
    [[ 3.09  4.4  -2.81]
     [ 2.41  2.62  0.76]]

    When ``s=[2, 4]``, shape of the transform will be ``(2, 4)``.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x, s=[2, 4]))
    [[  9.36  11.23   2.12 -10.97]
     [ 11.57   5.86  -1.37  -1.58]]
  """
  x = ensure_arraylike("dctn", x)

  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.dctn: {norm=!r} is not implemented")

  if dtypes.issubdtype(x.dtype, np.complexfloating):
    return lax.complex(
      dctn(x.real, type=type, s=s, norm=norm, axes=axes),
      dctn(x.imag, type=type, s=s, norm=norm, axes=axes),
    )

  s, axes = _dctn_s_axes(x.ndim, s, axes)

  if len(axes) == 1:
    return dct(x, n=s[0] if s is not None else None, axis=axes[0], norm=norm)

  if s is not None:
    x = _pad_axes(x, s, axes)

  if len(axes) == 2:
    return _dct2(x, axes=axes, norm=norm)

  # compose high-D DCTs from 2D and 1D DCTs:
  for axes_block in [axes[i:i+2] for i in range(0, len(axes), 2)]:
    x = dctn(x, axes=axes_block, norm=norm)
  return x


def dst(x: ArrayLike, type: int = 2, n: int | None = None,
        axis: int = -1, norm: str | None = None) -> Array:
  """Computes the discrete sine transform of the input

  JAX implementation of :func:`scipy.fft.dst`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    n: integer, default = x.shape[axis]. The length of the transform.
      If larger than ``x.shape[axis]``, the input will be zero-padded, if
      smaller, the input will be truncated.
    axis: integer, default=-1. The axis along which the dst will be performed.
    norm: string. The normalization mode: one of ``[None, "backward", "ortho"]``.
      The default is ``None``, which is equivalent to ``"backward"``.

  Returns:
    array containing the discrete sine transform of x

  See Also:
    - :func:`jax.scipy.fft.dstn`: multidimensional DST
    - :func:`jax.scipy.fft.idst`: inverse DST
    - :func:`jax.scipy.fft.idstn`: multidimensional inverse DST

  Examples:
    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dst(x))
    [[ 5.24  3.56 -1.67]
     [-0.7   1.55 -2.45]
     [ 1.16 -2.01 -0.65]]

    When ``n`` smaller than ``x.shape[axis]``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dst(x, n=2))
    [[ 5.16 -0.81]
     [ 0.14 -0.51]
     [-0.   -1.98]]
  """
  x = ensure_arraylike("dst", x)

  if type != 2:
    raise NotImplementedError('Only DST type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.dst: {norm=!r} is not implemented")

  x, = promote_dtypes_inexact(x)
  axis = canonicalize_axis(axis, x.ndim)
  # DST-II of (-1)**n * x[n] is the reversed DCT-II of the same sequence:
  # sin(pi (n + 1/2)(k + 1) / N) = (-1)**n cos(pi (n + 1/2)(N - k - 1) / N).
  out = dct(_dst_sign(x, (axis,)), type=type, n=n, axis=axis, norm=norm)
  return lax.rev(out, (axis,))


def dstn(x: ArrayLike, type: int = 2,
         s: Sequence[int] | None = None,
         axes: Sequence[int] | None = None,
         norm: str | None = None) -> Array:
  """Computes the multidimensional discrete sine transform of the input

  JAX implementation of :func:`scipy.fft.dstn`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    s: integer or sequence of integers. Specifies the shape of the result. If
      not specified, it will default to the shape of ``x`` along the specified
      ``axes``.
    axes: integer or sequence of integers. Specifies the axes along which the
      transform will be computed. If not given, the last ``len(s)`` axes are
      used, or all axes if ``s`` is also not specified.
    norm: string. The normalization mode: one of
      ``[None, "backward", "ortho"]``. The default is ``None``, which is
      equivalent to ``"backward"``.

  Returns:
    array containing the discrete sine transform of x

  See Also:
    - :func:`jax.scipy.fft.dst`: one-dimensional DST
    - :func:`jax.scipy.fft.idst`: one-dimensional inverse DST
    - :func:`jax.scipy.fft.idstn`: multidimensional inverse DST

  Examples:
    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dstn(x))
    [[ 5.    4.65 -7.23]
     [ 7.07  9.65 -1.77]
     [14.19  0.01  0.26]]

    When ``s=[2]``, the transform will be computed only along the last axis,
    with its dimension padded or truncated to size ``2``:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dstn(x, s=[2]))
    [[ 5.16 -0.81]
     [ 0.14 -0.51]
     [-0.   -1.98]]
  """
  x = ensure_arraylike("dstn", x)

  if type != 2:
    raise NotImplementedError('Only DST type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.dstn: {norm=!r} is not implemented")

  x, = promote_dtypes_inexact(x)

  s, axes = _dctn_s_axes(x.ndim, s, axes)

  # The multidimensional transform is separable, so the sign flips compose and
  # the result is the reversal of dctn along every transformed axis.
  out = dctn(_dst_sign(x, axes), type=type, s=s, axes=axes, norm=norm)
  return lax.rev(out, axes)


def idct(x: Array, type: int = 2, n: int | None = None,
         axis: int = -1, norm: str | None = None) -> Array:
  """Computes the inverse discrete cosine transform of the input

  JAX implementation of :func:`scipy.fft.idct`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    n: integer, default = x.shape[axis]. The length of the transform.
      If larger than ``x.shape[axis]``, the input will be zero-padded, if
      smaller, the input will be truncated.
    axis: integer, default=-1. The axis along which the dct will be performed.
    norm: string. The normalization mode: one of ``[None, "backward", "ortho"]``.
      The default is ``None``, which is equivalent to ``"backward"``.

  Returns:
    array containing the inverse discrete cosine transform of x

  See Also:
    - :func:`jax.scipy.fft.dct`: DCT
    - :func:`jax.scipy.fft.dctn`: multidimensional DCT
    - :func:`jax.scipy.fft.idctn`: multidimensional inverse DCT

  Examples:

    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x))
    [[ 0.78  0.41 -0.39]
     [-0.12  0.31 -0.23]
     [ 0.17 -0.3  -0.11]]

    When ``n`` smaller than ``x.shape[axis]``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x, n=2))
    [[ 1.12 -0.31]
     [ 0.04 -0.08]
     [ 0.05 -0.3 ]]

    When ``n`` smaller than ``x.shape[axis]`` and ``axis=0``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x, n=2, axis=0))
    [[ 0.38  0.57 -0.45]
     [ 0.43  0.44  0.24]]

    When ``n`` larger than ``x.shape[axis]`` and ``axis=0``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x, n=4, axis=0))
    [[ 0.1   0.38 -0.16]
     [ 0.28  0.18 -0.26]
     [ 0.3   0.15 -0.08]
     [ 0.13  0.3   0.29]]

    ``jax.scipy.fft.idct`` can be used to reconstruct ``x`` from the result
    of ``jax.scipy.fft.dct``

    >>> x_dct = jax.scipy.fft.dct(x)
    >>> jnp.allclose(x, jax.scipy.fft.idct(x_dct))
    Array(True, dtype=bool)
  """
  x = ensure_arraylike("idct", x)

  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.idct: {norm=!r} is not implemented")

  if dtypes.issubdtype(x.dtype, np.complexfloating):
    return lax.complex(
      idct(x.real, type=type, n=n, norm=norm, axis=axis),
      idct(x.imag, type=type, n=n, norm=norm, axis=axis)
    )

  axis = canonicalize_axis(axis, x.ndim)
  if n is not None:
    x = _pad_axes(x, [n], [axis])
  N = x.shape[axis]
  # The spectrum built below must be complex64/complex128, so upcast the
  # real input to float32/float64 first.
  dtype = dtypes.to_inexact_dtype(x.dtype)
  x = x.astype(dtype if dtype in (np.float32, np.float64) else np.float32)
  if norm == 'ortho':
    k_shape = tuple(s if i == axis else 1 for i, s in enumerate(x.shape))
    k = lax.broadcasted_iota(np.int32, k_shape, axis)
    x = x * jnp.where(k == 0, x.dtype.type(math.sqrt(4 * N)),
                      x.dtype.type(math.sqrt(2 * N)))

  # The real part of the inverse FFT is the irfft of the Hermitian part of its
  # input, which for real x is (x[k] - 1j * x[N - k]) / (2 * W4[k]).
  num_bins = N // 2 + 1
  re = lax.slice_in_dim(x, 0, num_bins, axis=axis)
  im = lax.rev(lax.slice_in_dim(x, (N + 1) // 2, N, axis=axis), (axis,))
  im = lax.pad(im, jnp.array(0, x.dtype),
               [(1 if a == axis else 0, 0, 0) for a in range(x.ndim)])
  k_shape = tuple(num_bins if i == axis else 1 for i in range(x.ndim))
  k = lax.broadcasted_iota(x.dtype, k_shape, axis)
  V = lax.complex(re, -im) * (0.5 * jnp.conj(_W4(N, k)))
  v = jnp_fft.irfft(V, n=N, axis=axis)
  return _dct_deinterleave(v, axis)


def idctn(x: Array, type: int = 2,
          s: Sequence[int] | None=None,
          axes: Sequence[int] | None = None,
          norm: str | None = None) -> Array:
  """Computes the multidimensional inverse discrete cosine transform of the input

  JAX implementation of :func:`scipy.fft.idctn`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    s: integer or sequence of integers. Specifies the shape of the result. If
      not specified, it will default to the shape of ``x`` along the specified
      ``axes``.
    axes: integer or sequence of integers. Specifies the axes along which the
      transform will be computed. If not given, the last ``len(s)`` axes are
      used, or all axes if ``s`` is also not specified.
    norm: string. The normalization mode: one of
      ``[None, "backward", "ortho"]``. The default is ``None``, which is
      equivalent to ``"backward"``.

  Returns:
    array containing the inverse discrete cosine transform of x

  See Also:
    - :func:`jax.scipy.fft.dct`: one-dimensional DCT
    - :func:`jax.scipy.fft.dctn`: multidimensional DCT
    - :func:`jax.scipy.fft.idct`: one-dimensional inverse DCT

  Examples:

    ``jax.scipy.fft.idctn`` computes the transform along both the axes by
    default when ``axes`` argument is ``None`` and ``s`` is also ``None``.

    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idctn(x))
    [[ 0.12  0.11 -0.15]
     [ 0.07  0.17 -0.03]
     [ 0.19 -0.07 -0.02]]

    When ``s=[2]``, the transform will be computed only along the last axis,
    with its dimension padded or truncated to size ``2``:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...  print(jax.scipy.fft.idctn(x, s=[2]))
    [[ 1.12 -0.31]
     [ 0.04 -0.08]
     [ 0.05 -0.3 ]]

    When ``s=[2]`` and ``axes=[0]``, the transform will be computed only along
    the specified axis, with its dimension padded or truncated to size ``2``:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...  print(jax.scipy.fft.idctn(x, s=[2], axes=[0]))
    [[ 0.38  0.57 -0.45]
     [ 0.43  0.44  0.24]]

    When ``s=[2, 4]``, shape of the transform will be ``(2, 4)``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...  print(jax.scipy.fft.idctn(x, s=[2, 4]))
    [[ 0.1   0.18  0.07 -0.16]
     [ 0.2   0.06 -0.03 -0.01]]

    ``jax.scipy.fft.idctn`` can be used to reconstruct ``x`` from the result
    of ``jax.scipy.fft.dctn``

    >>> x_dctn = jax.scipy.fft.dctn(x)
    >>> jnp.allclose(x, jax.scipy.fft.idctn(x_dctn))
    Array(True, dtype=bool)
  """
  x = ensure_arraylike("idctn", x)

  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.idctn: {norm=!r} is not implemented")

  if dtypes.issubdtype(x.dtype, np.complexfloating):
    return lax.complex(
      idctn(x.real, type=type, s=s, norm=norm, axes=axes),
      idctn(x.imag, type=type, s=s, norm=norm, axes=axes)
    )

  s, axes = _dctn_s_axes(x.ndim, s, axes)

  if len(axes) == 1:
    return idct(x, n=s[0] if s is not None else None, axis=axes[0], norm=norm)

  if s is not None:
    x = _pad_axes(x, s, axes)

  # compose high-D DCTs from 1D DCTs:
  for axis in axes:
    x = idct(x, axis=axis, norm=norm)
  return x


def idst(x: ArrayLike, type: int = 2, n: int | None = None,
         axis: int = -1, norm: str | None = None) -> Array:
  """Computes the inverse discrete sine transform of the input

  JAX implementation of :func:`scipy.fft.idst`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    n: integer, default = x.shape[axis]. The length of the transform.
      If larger than ``x.shape[axis]``, the input will be zero-padded, if
      smaller, the input will be truncated.
    axis: integer, default=-1. The axis along which the idst will be performed.
    norm: string. The normalization mode: one of ``[None, "backward", "ortho"]``.
      The default is ``None``, which is equivalent to ``"backward"``.

  Returns:
    array containing the inverse discrete sine transform of x

  See Also:
    - :func:`jax.scipy.fft.dst`: DST
    - :func:`jax.scipy.fft.dstn`: multidimensional DST
    - :func:`jax.scipy.fft.idstn`: multidimensional inverse DST

  Examples:
    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.idst(x))
    [[ 0.78  0.61 -0.39]
     [-0.12  0.14 -0.23]
     [ 0.17 -0.28 -0.11]]

    ``jax.scipy.fft.idst`` can be used to reconstruct ``x`` from the result
    of ``jax.scipy.fft.dst``

    >>> x_dst = jax.scipy.fft.dst(x)
    >>> jnp.allclose(x, jax.scipy.fft.idst(x_dst))
    Array(True, dtype=bool)
  """
  x = ensure_arraylike("idst", x)

  if type != 2:
    raise NotImplementedError('Only DST type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.idst: {norm=!r} is not implemented")

  x, = promote_dtypes_inexact(x)
  axis = canonicalize_axis(axis, x.ndim)
  if n is not None:
    # Pad or truncate before reversing: the inverse is the reverse of the idct
    # of the reversed input, and reversing does not commute with truncation.
    x = _pad_axes(x, [n], [axis])
  out = idct(lax.rev(x, (axis,)), type=type, axis=axis, norm=norm)
  return _dst_sign(out, (axis,))


def idstn(x: ArrayLike, type: int = 2,
          s: Sequence[int] | None = None,
          axes: Sequence[int] | None = None,
          norm: str | None = None) -> Array:
  """Computes the multidimensional inverse discrete sine transform of the input

  JAX implementation of :func:`scipy.fft.idstn`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    s: integer or sequence of integers. Specifies the shape of the result. If
      not specified, it will default to the shape of ``x`` along the specified
      ``axes``.
    axes: integer or sequence of integers. Specifies the axes along which the
      transform will be computed. If not given, the last ``len(s)`` axes are
      used, or all axes if ``s`` is also not specified.
    norm: string. The normalization mode: one of
      ``[None, "backward", "ortho"]``. The default is ``None``, which is
      equivalent to ``"backward"``.

  Returns:
    array containing the inverse discrete sine transform of x

  See Also:
    - :func:`jax.scipy.fft.dst`: one-dimensional DST
    - :func:`jax.scipy.fft.dstn`: multidimensional DST
    - :func:`jax.scipy.fft.idst`: one-dimensional inverse DST

  Examples:
    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.idstn(x))
    [[ 0.12  0.1  -0.15]
     [ 0.23  0.25 -0.11]
     [ 0.19  0.02 -0.02]]

    When ``s=[2]``, the transform will be computed only along the last axis,
    with its dimension padded or truncated to size ``2``:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.idstn(x, s=[2]))
    [[ 1.08  0.07]
     [ 0.02 -0.07]
     [-0.05 -0.3 ]]
  """
  x = ensure_arraylike("idstn", x)

  if type != 2:
    raise NotImplementedError('Only DST type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.idstn: {norm=!r} is not implemented")

  x, = promote_dtypes_inexact(x)

  s, axes = _dctn_s_axes(x.ndim, s, axes)

  if s is not None:
    # Pad or truncate before reversing, as in idst.
    x = _pad_axes(x, s, axes)
  out = idctn(lax.rev(x, axes), type=type, axes=axes, norm=norm)
  return _dst_sign(out, axes)


def _dct_deinterleave(x: Array, axis: int) -> Array:
  N = x.shape[axis]
  num_even = (N + 1) // 2
  v0 = lax.slice_in_dim(x, 0, num_even, axis=axis)
  v1 = lax.rev(lax.slice_in_dim(x, num_even, N, axis=axis), (axis,))
  if N % 2:
    # Pad v1 to the length of v0; the padding is sliced off below.
    v1 = lax.pad(v1, jnp.array(0, x.dtype),
                 [(0, 1 if a == axis else 0, 0) for a in range(x.ndim)])
  out = jnp.stack([v0, v1], axis=axis + 1)
  out = lax.reshape(out, x.shape[:axis] + (2 * num_even,) + x.shape[axis + 1:])
  return lax.slice_in_dim(out, 0, N, axis=axis)
