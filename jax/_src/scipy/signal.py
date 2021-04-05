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

from numbers import Number
import operator
import scipy.signal as osp_signal
import warnings

import numpy as np

from jax import lax
from jax._src.numpy import fft
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy import linalg
from jax._src.numpy.lax_numpy import _promote_dtypes_inexact
from jax._src.numpy.util import _wraps


# Note: we do not re-use the code from jax.numpy.convolve here, because the handling
# of padding differs slightly between the two implementations (particularly for
# mode='same').
def _convolve_nd(in1, in2, mode, *, precision):
  if mode not in ["full", "same", "valid"]:
    raise ValueError("mode must be one of ['full', 'same', 'valid']")
  if in1.ndim != in2.ndim:
    raise ValueError("in1 and in2 must have the same number of dimensions")
  if in1.size == 0 or in2.size == 0:
    raise ValueError(f"zero-size arrays not supported in convolutions, got shapes {in1.shape} and {in2.shape}.")
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
  result = lax.conv_general_dilated(in1[None, None], in2[None, None], strides,
                                    padding, precision=precision)
  return result[0, 0]


@_wraps(osp_signal.convolve)
def convolve(in1, in2, mode='full', method='auto',
             precision=None):
  if jnp.issubdtype(in1.dtype, jnp.complexfloating) or jnp.issubdtype(in2.dtype, jnp.complexfloating):
    raise NotImplementedError("convolve() does not support complex inputs")
  if method == 'fft':
    return fftconvolve(in1, in2, mode=mode)
  else:
    if method == 'auto':
      warnings.warn("convolve() 'auto' method falls back to 'direct' method.")
    return _convolve_nd(in1, in2, mode, precision=precision)


@_wraps(osp_signal.convolve2d)
def convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0,
               precision=None):
  if boundary != 'fill' or fillvalue != 0:
    raise NotImplementedError("convolve2d() only supports boundary='fill', fillvalue=0")
  if jnp.issubdtype(in1.dtype, jnp.complexfloating) or jnp.issubdtype(in2.dtype, jnp.complexfloating):
    raise NotImplementedError("convolve2d() does not support complex inputs")
  if jnp.ndim(in1) != 2 or jnp.ndim(in2) != 2:
    raise ValueError("convolve2d() only supports 2-dimensional inputs.")
  return _convolve_nd(in1, in2, mode, precision=precision)


@_wraps(osp_signal.correlate)
def correlate(in1, in2, mode='full', method='auto',
              precision=None):
  return convolve(in1, jnp.flip(in2), mode=mode, method=method, precision=precision)


@_wraps(osp_signal.correlate2d)
def correlate2d(in1, in2, mode='full', boundary='fill', fillvalue=0,
                precision=None):
  if boundary != 'fill' or fillvalue != 0:
    raise NotImplementedError("correlate2d() only supports boundary='fill', fillvalue=0")
  if jnp.issubdtype(in1.dtype, jnp.complexfloating) or jnp.issubdtype(in2.dtype, jnp.complexfloating):
    raise NotImplementedError("correlate2d() does not support complex inputs")
  if jnp.ndim(in1) != 2 or jnp.ndim(in2) != 2:
    raise ValueError("correlate2d() only supports {ndim}-dimensional inputs.")
  return _convolve_nd(in1[::-1, ::-1], in2, mode, precision=precision)[::-1, ::-1]


@_wraps(osp_signal.detrend)
def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=None):
  if overwrite_data is not None:
    raise NotImplementedError("overwrite_data argument not implemented.")
  if type not in ['constant', 'linear']:
    raise ValueError("Trend type must be 'linear' or 'constant'.")
  data, = _promote_dtypes_inexact(jnp.asarray(data))
  if type == 'constant':
    return data - data.mean(axis, keepdims=True)
  else:
    N = data.shape[axis]
    # bp is static, so we use np operations to avoid pushing to device.
    bp = np.sort(np.unique(np.r_[0, bp, N]))
    if bp[0] < 0 or bp[-1] > N:
      raise ValueError("Breakpoints must be non-negative and less than length of data along given axis.")
    data = jnp.moveaxis(data, axis, 0)
    shape = data.shape
    data = data.reshape(N, -1)
    for m in range(len(bp) - 1):
      Npts = bp[m + 1] - bp[m]
      A = jnp.vstack([
        jnp.ones(Npts, dtype=data.dtype),
        jnp.arange(1, Npts + 1, dtype=data.dtype) / Npts
      ]).T
      sl = slice(bp[m], bp[m + 1])
      coef, *_ = linalg.lstsq(A, data[sl])
      data = data.at[sl].add(-jnp.matmul(A, coef, precision=lax.Precision.HIGHEST))
    return jnp.moveaxis(data.reshape(shape), 0, axis)


@_wraps(osp_signal.fftconvolve)
def fftconvolve(in1, in2, mode='full', axes=None):
  if in1.ndim != in2.ndim:
    raise ValueError("in1 and in2 should have the same dimensionality")
  elif in1.ndim == in2.ndim == 0:
    return in1 * in2
  elif in1.size == 0 or in2.size == 0:
    return jnp.array([], dtype=in1.dtype)
  in1, in2, axes = _standarize_freq_domain_conv_axes(in1, in2, mode, axes, sorted_axes=False)
  s1 = in1.shape
  s2 = in2.shape
  shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
           for i in range(in1.ndim)]
  ret = _freq_domain_conv(in1, in2, axes, shape)
  return _apply_conv_mode(ret, s1, s2, mode, axes)


def _freq_domain_conv(in1, in2, axes, shape):
  """Convolve `in1` with `in2` in the frequency domain."""
  if not len(axes):
    return in1 * in2
  in1_freq = fft.rfftn(in1, shape, axes=axes)
  in2_freq = fft.rfftn(in2, shape, axes=axes)
  ret = fft.irfftn(in1_freq * in2_freq, shape, axes=axes)
  return ret


def _standarize_freq_domain_conv_axes(in1, in2, mode, axes, sorted_axes=False):
  """Handle the `axes` argument for `_freq_domain_conv`.
  Returns the inputs and axes in a standard form, eliminating redundant axes,
  swapping the inputs if necessary, and checking for various potential
  errors.
  """
  s1 = in1.shape
  s2 = in2.shape
  _, axes = _init_nd_shape_and_axes(in1, shape=None, axes=axes)
  if not axes:
    raise ValueError("when provided, axes cannot be empty")
  # Axes of length 1 can rely on broadcasting rules for multipy, no fft needed.
  axes = [a for a in axes if s1[a] != 1 and s2[a] != 1]
  if sorted_axes:
    axes.sort()
  if not all(s1[a] == s2[a] or s1[a] == 1 or s2[a] == 1
             for a in range(in1.ndim) if a not in axes):
    raise ValueError("incompatible shapes for in1 and in2:"
                     " {0} and {1}".format(s1, s2))
  if _inputs_swap_needed(mode, s1, s2, axes=axes):
    in1, in2 = in2, in1
  return in1, in2, axes


def _init_nd_shape_and_axes(x, shape, axes):
  """Handle shape and axes arguments for nd transforms"""
  if axes is not None:
    axes = _iterable_of_int(axes, 'axes')
    axes = [a + x.ndim if a < 0 else a for a in axes]
    if any(a >= x.ndim or a < 0 for a in axes):
      raise ValueError("axes exceeds dimensionality of input")
    if len(set(axes)) != len(axes):
      raise ValueError("all axes must be unique")
  if shape is not None:
    shape = _iterable_of_int(shape, 'shape')
    if axes and len(axes) != len(shape):
      raise ValueError("when given, axes and shape arguments have to be of the same length")
    if axes is None:
      if len(shape) > x.ndim:
        raise ValueError("shape requires more axes than are present")
      axes = range(x.ndim - len(shape), x.ndim)
    shape = [x.shape[a] if s == -1 else s for s, a in zip(shape, axes)]
  elif axes is None:
    shape = list(x.shape)
    axes = range(x.ndim)
  else:
    shape = [x.shape[a] for a in axes]
  if any(s < 1 for s in shape):
    raise ValueError(
      "invalid number of data points ({}) specified".format(shape))
  return shape, axes


def _iterable_of_int(x, name=None):
  """Convert `x` to an sequence of ints"""
  if isinstance(x, Number):
    x = (operator.index(x),)
  try:
    x = [int(a) for a in x]
  except TypeError as e:
    name = name or 'value'
    raise ValueError("{} must be a scalar or iterable of integers"
                     .format(name)) from e
  return x


def _apply_conv_mode(ret, s1, s2, mode, axes):
  """Slice result based on the given `mode`."""
  if mode == 'full':
    return ret
  elif mode == 'same':
    return _centered(ret, s1)
  elif mode == 'valid':
    shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
                   for a in range(ret.ndim)]
    return _centered(ret, shape_valid)
  else:
    raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")


def _centered(arr, new_shape):
  """Centered slice of the given array."""
  new_shape = np.asarray(new_shape)
  start_idx = (arr.shape - new_shape) // 2
  end_idx = start_idx + new_shape
  centered_slice = tuple(slice(start_idx[k], end_idx[k]) for k in range(len(end_idx)))
  return arr[centered_slice]


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
  """True iff inputs need to be swapped to be compatible with 'valid' mode."""
  if mode != 'valid':
    return False
  if not shape1:
    return False
  if axes is None:
    axes = range(len(shape1))
  all_shape_1_gte_2 = all(shape1[i] >= shape2[i] for i in axes)
  all_shape_2_gte_1 = all(shape2[i] >= shape1[i] for i in axes)
  if not (all_shape_1_gte_2 or all_shape_2_gte_1):
    raise ValueError("For 'valid' mode, one array must be at least "
                     "as large as the other in every dimension")
  return not all_shape_1_gte_2
