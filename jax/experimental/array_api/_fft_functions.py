# Copyright 2023 The JAX Authors.
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

import jax.numpy as jnp
from jax._src.numpy.fft import NEEDS_COMPLEX_IN, NEEDS_REAL_IN as _NEEDS_REAL_IN
from jax._src.numpy.util import check_arraylike

NEEDS_REAL_IN = _NEEDS_REAL_IN.union({'rfft', 'rfftn', 'ihfft'})

# TODO(micky774): Remove when jax.numpy.fft deprecation completes. Deprecation
# began 4-18-24.
def _check_input_fft(func_name: str, x):
  check_arraylike('jax.experimental.array_api.' + func_name, x)
  arr = jnp.asarray(x)
  kind = arr.dtype.kind
  suggest_alternative_msg = (
    " or consider using a more appropriate fft function if applicable."
  )
  if func_name in NEEDS_COMPLEX_IN and kind != "c":
    raise ValueError(
      f"{func_name} requires complex-valued input, but received input with type "
      f"{arr.dtype} instead. Please explicitly convert to a complex-valued input "
      "first," + suggest_alternative_msg,
    )
  if func_name in NEEDS_REAL_IN:
    needs_real_msg = (
      f"{func_name} requires real-valued floating-point input, but received "
      f"input with type {arr.dtype} instead. Please convert to a real-valued "
      "floating-point input first"
    )
    if kind == "c":
      raise ValueError(
        needs_real_msg + ", such as by using jnp.real or jnp.imag to take the "
        "real or imaginary components respectively," + suggest_alternative_msg,
      )
    elif kind != "f":
      raise ValueError(needs_real_msg + '.')
  return arr

def fft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional discrete Fourier transform."""
  _check_input_fft('fft', x)
  return jnp.fft.fft(x, n=n, axis=axis, norm=norm)

def ifft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional inverse discrete Fourier transform."""
  _check_input_fft('ifft', x)
  return jnp.fft.ifft(x, n=n, axis=axis, norm=norm)

def fftn(x, /, *, s=None, axes=None, norm='backward'):
  """Computes the n-dimensional discrete Fourier transform."""
  _check_input_fft('fftn', x)
  return jnp.fft.fftn(x, s=s, axes=axes, norm=norm)

def ifftn(x, /, *, s=None, axes=None, norm='backward'):
  """Computes the n-dimensional inverse discrete Fourier transform."""
  _check_input_fft('ifftn', x)
  return jnp.fft.ifftn(x, s=s, axes=axes, norm=norm)

def rfft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional discrete Fourier transform for real-valued input."""
  _check_input_fft('rfft', x)
  return jnp.fft.rfft(x, n=n, axis=axis, norm=norm)

def irfft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional inverse of rfft for complex-valued input."""
  _check_input_fft('irfft', x)
  return jnp.fft.irfft(x, n=n, axis=axis, norm=norm)

def rfftn(x, /, *, s=None, axes=None, norm='backward'):
  """Computes the n-dimensional discrete Fourier transform for real-valued input."""
  _check_input_fft('rfftn', x)
  return jnp.fft.rfftn(x, s=s, axes=axes, norm=norm)

def irfftn(x, /, *, s=None, axes=None, norm='backward'):
  """Computes the n-dimensional inverse of rfftn for complex-valued input."""
  _check_input_fft('irfftn', x)
  return jnp.fft.irfftn(x, s=s, axes=axes, norm=norm)

def hfft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional discrete Fourier transform of a signal with Hermitian symmetry."""
  _check_input_fft('hfft', x)
  return jnp.fft.hfft(x, n=n, axis=axis, norm=norm)

def ihfft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional inverse discrete Fourier transform of a signal with Hermitian symmetry."""
  _check_input_fft('ihfft', x)
  return jnp.fft.ihfft(x, n=n, axis=axis, norm=norm)

def fftfreq(n, /, *, d=1.0, device=None):
  """Returns the discrete Fourier transform sample frequencies."""
  return jnp.fft.fftfreq(n, d=d).to_device(device)

def rfftfreq(n, /, *, d=1.0, device=None):
  """Returns the discrete Fourier transform sample frequencies (for rfft and irfft)."""
  return jnp.fft.rfftfreq(n, d=d).to_device(device)

def fftshift(x, /, *, axes=None):
  """Shift the zero-frequency component to the center of the spectrum."""
  _check_input_fft('fftshift', x)
  return jnp.fft.fftshift(x, axes=axes)

def ifftshift(x, /, *, axes=None):
  """Inverse of fftshift."""
  _check_input_fft('ifftshift', x)
  return jnp.fft.ifftshift(x, axes=axes)
