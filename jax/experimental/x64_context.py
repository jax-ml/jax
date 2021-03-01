# Copyright 2021 Google LLC
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

"""Context managers for toggling X64 mode.

**Experimental: please give feedback, and expect changes.**
"""

from contextlib import contextmanager
from jax import config

@contextmanager
def enable_x64():
  """Experimental context manager to temporarily enable X64 mode.

  Usage::

    >>> import jax.numpy as jnp
    >>> with enable_x64():
    ...   print(jnp.arange(10.0).dtype)
    ...
    float64

  See Also
  --------
  jax.experimental.disable_x64 :  temporarily disable X64 mode.
  """
  _x64_state = config.x64_enabled
  config._set_x64_enabled(True)
  try:
    yield
  finally:
    config._set_x64_enabled(_x64_state)

@contextmanager
def disable_x64():
  """Experimental context manager to temporarily disable X64 mode.

  Usage::

    >>> import jax.numpy as jnp
    >>> with disable_x64():
    ...   print(jnp.arange(10.0).dtype)
    ...
    float32

  See Also
  --------
  jax.experimental.enable_x64 : temporarily enable X64 mode.
  """
  _x64_state = config.x64_enabled
  config._set_x64_enabled(False)
  try:
    yield
  finally:
    config._set_x64_enabled(_x64_state)
