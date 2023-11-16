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


def fft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional discrete Fourier transform."""
  return jnp.fft.fft(x, n=n, axis=axis, norm=norm)

def ifft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional inverse discrete Fourier transform."""
  return jnp.fft.ifft(x, n=n, axis=axis, norm=norm)

def fftn(x, /, *, s=None, axes=None, norm='backward'):
  """Computes the n-dimensional discrete Fourier transform."""
  return jnp.fft.fftn(x, s=s, axes=axes, norm=norm)

def ifftn(x, /, *, s=None, axes=None, norm='backward'):
  """Computes the n-dimensional inverse discrete Fourier transform."""
  return jnp.fft.ifftn(x, s=s, axes=axes, norm=norm)

def rfft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional discrete Fourier transform for real-valued input."""
  return jnp.fft.rfft(x, n=n, axis=axis, norm=norm)

def irfft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional inverse of rfft for complex-valued input."""
  return jnp.fft.irfft(x, n=n, axis=axis, norm=norm)

def rfftn(x, /, *, s=None, axes=None, norm='backward'):
  """Computes the n-dimensional discrete Fourier transform for real-valued input."""
  return jnp.fft.rfftn(x, s=s, axes=axes, norm=norm)

def irfftn(x, /, *, s=None, axes=None, norm='backward'):
  """Computes the n-dimensional inverse of rfftn for complex-valued input."""
  return jnp.fft.irfftn(x, s=s, axes=axes, norm=norm)

def hfft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional discrete Fourier transform of a signal with Hermitian symmetry."""
  return jnp.fft.hfft(x, n=n, axis=axis, norm=norm)

def ihfft(x, /, *, n=None, axis=-1, norm='backward'):
  """Computes the one-dimensional inverse discrete Fourier transform of a signal with Hermitian symmetry."""
  return jnp.fft.ihfft(x, n=n, axis=axis, norm=norm)

def fftfreq(n, /, *, d=1.0, device=None):
  """Returns the discrete Fourier transform sample frequencies."""
  return jnp.fft.fftfreq(n, d=d).to_device(device)

def rfftfreq(n, /, *, d=1.0, device=None):
  """Returns the discrete Fourier transform sample frequencies (for rfft and irfft)."""
  return jnp.fft.rfftfreq(n, d=d).to_device(device)

def fftshift(x, /, *, axes=None):
  """Shift the zero-frequency component to the center of the spectrum."""
  return jnp.fft.fftshift(x, axes=axes)

def ifftshift(x, /, *, axes=None):
  """Inverse of fftshift."""
  return jnp.fft.ifftshift(x, axes=axes)
