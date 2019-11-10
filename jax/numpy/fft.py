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
from ..lib.xla_bridge import xla_client, canonicalize_dtype
from ..util import get_module_functions
from .lax_numpy import _not_implemented
from .lax_numpy import _wraps
from . import lax_numpy as np


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

  if any(axis in range(a.ndim - 3) for axis in axes):
    raise ValueError(
        "%s only supports 1D, 2D, and 3D FFTs over the innermost axes."
        " Got axes %s with input rank %s." % (full_name, orig_axes, a.ndim))

  if s is None:
    if fft_type == xla_client.FftType.IRFFT:
      s = [a.shape[axis] for axis in axes[:-1]]
      if axes:
        s += [2 * (a.shape[axes[-1]] - 1)]
    else:
      s = [a.shape[axis] for axis in axes]

  return lax.fft(a, fft_type, s)


@_wraps(onp.fft.fftn)
def fftn(a, s=None, axes=None, norm=None):
  return _fft_core('fftn', xla_client.FftType.FFT, a, s, axes, norm)


@_wraps(onp.fft.ifftn)
def ifftn(a, s=None, axes=None, norm=None):
  return _fft_core('ifftn', xla_client.FftType.IFFT, a, s, axes, norm)


@_wraps(onp.fft.fftn)
def rfftn(a, s=None, axes=None, norm=None):
  return _fft_core('rfftn', xla_client.FftType.RFFT, a, s, axes, norm)


@_wraps(onp.fft.ifftn)
def irfftn(a, s=None, axes=None, norm=None):
  return _fft_core('irfftn', xla_client.FftType.IRFFT, a, s, axes, norm)


for func in get_module_functions(onp.fft):
  if func.__name__ not in globals():
    globals()[func.__name__] = _not_implemented(func)
