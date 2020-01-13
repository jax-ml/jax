# Copyright 2018 Google LLC
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

from .. import lax
from ..lib import xla_client
from ..util import get_module_functions
from .lax_numpy import _not_implemented
from .lax_numpy import _wraps
from . import lax_numpy as np
from .. import ops as jaxops


def _fft_core(func_name, fft_type, a, s, axes, norm):
  # TODO(skye): implement padding/cropping based on 's'.
  full_name = "jax.np.fft." + func_name
  if s is not None:
    raise NotImplementedError("%s only supports s=None, got %s" % (full_name, s))
  if norm is not None:
    raise NotImplementedError("%s only supports norm=None, got %s" % (full_name, norm))
  if s is not None and axes is not None and len(s) != len(axes):
    # Same error as numpy.
    raise ValueError("Shape and axes have different lengths.")

  orig_axes = axes
  if axes is None:
    if s is None:
      axes = range(a.ndim)
    else:
      axes = range(a.ndim - len(s), a.ndim)

  if len(axes) != len(set(axes)):
    raise ValueError(
        "%s does not support repeated axes. Got axes %s." % (full_name, axes))

  if len(axes) > 3:
    # XLA does not support FFTs over more than 3 dimensions
    raise ValueError(
        "%s only supports 1D, 2D, and 3D FFTs. "
        "Got axes %s with input rank %s." % (full_name, orig_axes, a.ndim))

  # XLA only supports FFTs over the innermost axes, so rearrange if necessary.
  if orig_axes is not None:
    axes = tuple(range(a.ndim - len(axes), a.ndim))
    a = np.moveaxis(a, orig_axes, axes)

  if s is None:
    if fft_type == xla_client.FftType.IRFFT:
      s = [a.shape[axis] for axis in axes[:-1]]
      if axes:
        s += [max(0, 2 * (a.shape[axes[-1]] - 1))]
    else:
      s = [a.shape[axis] for axis in axes]

  transformed = lax.fft(a, fft_type, s)

  if orig_axes is not None:
    transformed = np.moveaxis(transformed, axes, orig_axes)
  return transformed


@_wraps(onp.fft.fftn)
def fftn(a, s=None, axes=None, norm=None):
  return _fft_core('fftn', xla_client.FftType.FFT, a, s, axes, norm)


@_wraps(onp.fft.ifftn)
def ifftn(a, s=None, axes=None, norm=None):
  return _fft_core('ifftn', xla_client.FftType.IFFT, a, s, axes, norm)


@_wraps(onp.fft.rfftn)
def rfftn(a, s=None, axes=None, norm=None):
  return _fft_core('rfftn', xla_client.FftType.RFFT, a, s, axes, norm)


@_wraps(onp.fft.irfftn)
def irfftn(a, s=None, axes=None, norm=None):
  return _fft_core('irfftn', xla_client.FftType.IRFFT, a, s, axes, norm)


def _fft_core_1d(func_name, fft_type, a, s, axis, norm):
  full_name = "jax.np.fft." + func_name
  if isinstance(axis, (list, tuple)):
    raise ValueError(
        "%s does not support multiple axes. Please use %sn. "
        "Got axis = %r." % (full_name, full_name, axis)
    )
  axes = None if axis is None else [axis]
  return _fft_core(func_name, fft_type, a, s, axes, norm)


@_wraps(onp.fft.fft)
def fft(a, n=None, axis=-1, norm=None):
  return _fft_core_1d('fft', xla_client.FftType.FFT, a, s=n, axis=axis,
                      norm=norm)

@_wraps(onp.fft.ifft)
def ifft(a, n=None, axis=-1, norm=None):
  return _fft_core_1d('ifft', xla_client.FftType.IFFT, a, s=n, axis=axis,
                      norm=norm)

@_wraps(onp.fft.rfft)
def rfft(a, n=None, axis=-1, norm=None):
  return _fft_core_1d('rfft', xla_client.FftType.RFFT, a, s=n, axis=axis,
                      norm=norm)

@_wraps(onp.fft.irfft)
def irfft(a, n=None, axis=-1, norm=None):
  return _fft_core_1d('irfft', xla_client.FftType.IRFFT, a, s=n, axis=axis,
                      norm=norm)


def _fft_core_2d(func_name, fft_type, a, s, axes, norm):
  full_name = "jax.np.fft." + func_name
  if len(axes) != 2:
    raise ValueError(
        "%s only supports 2 axes. Got axes = %r."
        % (full_name, axes)
    )
  return _fft_core(func_name, fft_type, a, s, axes, norm)


@_wraps(onp.fft.fft2)
def fft2(a, s=None, axes=(-2,-1), norm=None):
  return _fft_core_2d('fft2', xla_client.FftType.FFT, a, s=s, axes=axes,
                      norm=norm)

@_wraps(onp.fft.ifft2)
def ifft2(a, s=None, axes=(-2,-1), norm=None):
  return _fft_core_2d('ifft2', xla_client.FftType.IFFT, a, s=s, axes=axes,
                      norm=norm)

@_wraps(onp.fft.rfft2)
def rfft2(a, s=None, axes=(-2,-1), norm=None):
  return _fft_core_2d('rfft2', xla_client.FftType.RFFT, a, s=s, axes=axes,
                      norm=norm)

@_wraps(onp.fft.irfft2)
def irfft2(a, s=None, axes=(-2,-1), norm=None):
  return _fft_core_2d('irfft2', xla_client.FftType.IRFFT, a, s=s, axes=axes,
                      norm=norm)


@_wraps(onp.fft.fftfreq)
def fftfreq(n, d=1.0):
  if isinstance(n, list) or isinstance(n, tuple):
    raise ValueError(
          "The n argument of jax.np.fft.fftfreq only takes an int. "
          "Got n = %s." % list(n))

  elif isinstance(d, list) or isinstance(d, tuple):
    raise ValueError(
          "The d argument of jax.np.fft.fftfreq only takes a single value. "
          "Got d = %s." % list(d))

  k = np.zeros(n)
  if n % 2 == 0:
    # k[0: n // 2 - 1] = np.arange(0, n // 2 - 1)
    k = jaxops.index_update(k, jaxops.index[0: n // 2], np.arange(0, n // 2))

    # k[n // 2:] = np.arange(-n // 2, -1)
    k = jaxops.index_update(k, jaxops.index[n // 2:], np.arange(-n // 2, 0))

  else:
    # k[0: (n - 1) // 2] = np.arange(0, (n - 1) // 2)
    k = jaxops.index_update(k, jaxops.index[0: (n - 1) // 2 + 1],
                            np.arange(0, (n - 1) // 2 + 1))

    # k[(n - 1) // 2 + 1:] = np.arange(-(n - 1) // 2, -1)
    k = jaxops.index_update(k, jaxops.index[(n - 1) // 2 + 1:],
                            np.arange(-(n - 1) // 2, 0))

  return k / (d * n)


@_wraps(onp.fft.rfftfreq)
def rfftfreq(n, d=1.0):
  if isinstance(n, list) or isinstance(n, tuple):
    raise ValueError(
          "The n argument of jax.np.fft.rfftfreq only takes an int. "
          "Got n = %s." % list(n))

  elif isinstance(d, list) or isinstance(d, tuple):
    raise ValueError(
          "The d argument of jax.np.fft.rfftfreq only takes a single value. "
          "Got d = %s." % list(d))

  if n % 2 == 0:
    k = np.arange(0, n // 2 + 1)

  else:
    k = np.arange(0, (n - 1) // 2 + 1)

  return k / (d * n)


for func in get_module_functions(onp.fft):
  if func.__name__ not in globals():
    globals()[func.__name__] = _not_implemented(func)
