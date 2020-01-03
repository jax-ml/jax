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


def _promote_to_complex(arg):
  dtype = np.result_type(arg, onp.complex64)
  # XLA's FFT op only supports C64.
  if dtype == onp.complex128:
    dtype = onp.complex64
  return lax.convert_element_type(arg, dtype)


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

  # XLA doesn't support 0-rank axes.
  if len(axes) == 0:
    return a

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
    s = [a.shape[axis] for axis in axes]

  a = _promote_to_complex(a)
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


@_wraps(onp.fft.fft)
def fft(a, n=None, axis=-1, norm=None):
  if isinstance(axis,list) or isinstance(axis,tuple):
    raise ValueError(
      "jax.np.fft.fft does not support multiple axes. "
      "Please use jax.np.fft.fftn. "
      "Got axis %s." % (list(axis))
    )

  if not axis is None:
    axis = [axis]

  return _fft_core('fft', xla_client.FftType.FFT, a, s=n, axes=axis, norm=norm)


@_wraps(onp.fft.ifft)
def ifft(a, n=None, axis=-1, norm=None):
  if isinstance(axis,list) or isinstance(axis,tuple):
    raise ValueError(
      "jax.np.fft.ifft does not support multiple axes. "
      "Please use jax.np.fft.ifftn. "
      "Got axis %s." % (list(axis))
    )

  if not axis is None:
    axis = [axis]

  return _fft_core('ifft', xla_client.FftType.IFFT, a, s=n, axes=axis, norm=norm)


for func in get_module_functions(onp.fft):
  if func.__name__ not in globals():
    globals()[func.__name__] = _not_implemented(func)
