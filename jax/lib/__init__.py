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

# This module is largely a wrapper around `jaxlib` that performs version
# checking on import.

import os
from typing import Optional

__all__ = [
  'cuda_linalg', 'cuda_prng', 'cusolver', 'rocsolver', 'jaxlib', 'lapack',
  'pocketfft', 'pytree', 'tpu_client', 'version', 'xla_client'
]

try:
  import jaxlib
except ModuleNotFoundError as err:
  raise ModuleNotFoundError(
    'jax requires jaxlib to be installed. See '
    'https://github.com/google/jax#installation for installation instructions.'
    ) from err

from jax.version import _minimum_jaxlib_version as _minimum_jaxlib_version_str
try:
  from jaxlib import version as jaxlib_version
except Exception as err:
  # jaxlib is too old to have version number.
  msg = f'This version of jax requires jaxlib version >= {_minimum_jaxlib_version_str}.'
  raise ImportError(msg) from err

version = tuple(int(x) for x in jaxlib_version.__version__.split('.'))
_minimum_jaxlib_version = tuple(int(x) for x in _minimum_jaxlib_version_str.split('.'))

# Check the jaxlib version before importing anything else from jaxlib.
def _check_jaxlib_version():
  if version < _minimum_jaxlib_version:
    msg = (f'jaxlib is version {jaxlib_version.__version__}, '
           f'but this version of jax requires version {_minimum_jaxlib_version_str}.')

    if version == (0, 1, 23):
      msg += ('\n\nA common cause of this error is that you installed jaxlib '
              'using pip, but your version of pip is too old to support '
              'manylinux2010 wheels. Try running:\n\n'
              'pip install --upgrade pip\n'
              'pip install --upgrade jax jaxlib\n')
    raise ValueError(msg)

_check_jaxlib_version()

from jaxlib import xla_client
from jaxlib import lapack
from jaxlib import pocketfft

xla_extension = xla_client._xla
pytree = xla_client._xla.pytree
jax_jit = xla_client._xla.jax_jit
pmap_lib = xla_client._xla.pmap_lib

try:
  from jaxlib import cusolver  # pytype: disable=import-error
except ImportError:
  cusolver = None

try:
  from jaxlib import cusparse  # pytype: disable=import-error
except ImportError:
  cusparse = None

try:
  from jaxlib import rocsolver  # pytype: disable=import-error
except ImportError:
  rocsolver = None

try:
  from jaxlib import cuda_prng  # pytype: disable=import-error
except ImportError:
  cuda_prng = None

try:
  from jaxlib import cuda_linalg  # pytype: disable=import-error
except ImportError:
  cuda_linalg = None

# Jaxlib code is split between the Jax and the Tensorflow repositories.
# Only for the internal usage of the JAX developers, we expose a version
# number that can be used to perform changes without breaking the master
# branch on the Jax github.
_xla_extension_version = getattr(xla_client, '_version', 0)

try:
  from jaxlib import tpu_client  # pytype: disable=import-error
except:
  tpu_client = None


cuda_path: Optional[str]
cuda_path = os.path.join(os.path.dirname(jaxlib.__file__), "cuda")
if not os.path.isdir(cuda_path):
  cuda_path = None
