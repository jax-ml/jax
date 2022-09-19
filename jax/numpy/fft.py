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

from jax._src.numpy.fft import (
  ifft as ifft,
  ifft2 as ifft2,
  ifftn as ifftn,
  ifftshift as ifftshift,
  ihfft as ihfft,
  irfft as irfft,
  irfft2 as irfft2,
  irfftn as irfftn,
  fft as fft,
  fft2 as fft2,
  fftfreq as fftfreq,
  fftn as fftn,
  fftshift as fftshift,
  hfft as hfft,
  rfft as rfft,
  rfft2 as rfft2,
  rfftfreq as rfftfreq,
  rfftn as rfftn,
)

# Module initialization is encapsulated in a function to avoid accidental
# namespace pollution.
_NOT_IMPLEMENTED = []
def _init():
  import numpy as np
  from jax._src.numpy import lax_numpy
  from jax._src import util
  # Builds a set of all unimplemented NumPy functions.
  for name, func in util.get_module_functions(np.fft).items():
    if name not in globals():
      _NOT_IMPLEMENTED.append(name)
      globals()[name] = lax_numpy._not_implemented(func)

_init()
del _init
