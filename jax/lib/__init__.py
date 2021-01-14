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

__all__ = [
  'cuda_prng', 'cusolver', 'rocsolver', 'jaxlib', 'lapack',
  'pytree', 'tpu_client', 'version', 'xla_client'
]

import jaxlib

# Must be kept in sync with the jaxlib version in build/test-requirements.txt
_minimum_jaxlib_version = (0, 1, 55)
try:
  from jaxlib import version as jaxlib_version
except Exception as err:
  # jaxlib is too old to have version number.
  msg = 'This version of jax requires jaxlib version >= {}.'
  raise ImportError(msg.format('.'.join(map(str, _minimum_jaxlib_version)))
                    ) from err

version = tuple(int(x) for x in jaxlib_version.__version__.split('.'))

# Check the jaxlib version before importing anything else from jaxlib.
def _check_jaxlib_version():
  if version < _minimum_jaxlib_version:
    msg = 'jaxlib is version {}, but this version of jax requires version {}.'

    if version == (0, 1, 23):
      msg += ('\n\nA common cause of this error is that you installed jaxlib '
              'using pip, but your version of pip is too old to support '
              'manylinux2010 wheels. Try running:\n\n'
              'pip install --upgrade pip\n'
              'pip install --upgrade jax jaxlib\n')
    raise ValueError(msg.format('.'.join(map(str, version)),
                                '.'.join(map(str, _minimum_jaxlib_version))))

_check_jaxlib_version()

from jaxlib import xla_client
from jaxlib import lapack
if version <  (0, 1, 53):
  from jaxlib import pytree  # pytype: disable=import-error
else:
  pytree = xla_client._xla.pytree
  jax_jit = xla_client._xla.jax_jit

try:
  from jaxlib import cusolver
except ImportError:
  cusolver = None

try:
  from jaxlib import rocsolver  # pytype: disable=import-error
except ImportError:
  rocsolver = None

try:
  from jaxlib import cuda_prng
except ImportError:
  cuda_prng = None

# Jaxlib code is split between the Jax and the Tensorflow repositories.
# Only for the internal usage of the JAX developers, we expose a version
# number that can be used to perform changes without breaking the master
# branch on the Jax github.
_xla_extension_version = getattr(xla_client, '_version', 0)

try:
  from jaxlib import tpu_client  # pytype: disable=import-error
except:
  tpu_client = None

# TODO(phawkins): Make this import unconditional once the minimum jaxlib version
# is 0.1.57 or greater.
try:
  from jaxlib import pocketfft  # pytype: disable=import-error
except:
  pocketfft = None
