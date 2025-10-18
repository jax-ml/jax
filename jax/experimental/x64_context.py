# Copyright 2021 The JAX Authors.
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

**Deprecated: use :func:`jax.enable_x64` instead.**
"""

from contextlib import contextmanager
from jax._src import config

@contextmanager
def _enable_x64(new_val: bool = True):
  """Experimental context manager to temporarily enable X64 mode.

  .. warning::

    This context manager is deprecated as of JAX v0.8.0, and will be removed in
    JAX v0.9.0. Use :func:`jax.enable_x64` instead.

  Usage::

    >>> import jax
    >>> x = np.arange(5, dtype='float64')
    >>> with _enable_x64(True):
    ...   print(jnp.asarray(x).dtype)
    ...
    float64

  See Also
  --------
  jax.experimental.disable_x64 : temporarily disable X64 mode.
  """
  with config.enable_x64(new_val):
    yield

@contextmanager
def _disable_x64():
  """Experimental context manager to temporarily disable X64 mode.

  .. warning::

    This context manager is deprecated as of JAX v0.8.0, and will be removed in
    JAX v0.9.0. Use :func:`jax.enable_x64` instead.

  Usage::

    >>> x = np.arange(5, dtype='float64')
    >>> with _disable_x64():
    ...   print(jnp.asarray(x).dtype)
    ...
    float32

  See Also
  --------
  jax.experimental.enable_x64 : temporarily enable X64 mode.
  """
  with config.enable_x64(False):
    yield

_deprecations = {
  # Added for v0.8.0
  "disable_x64": (
    ("jax.experimental.x64_context.disable_x64 is deprecated in JAX v0.8.0 and will be removed"
     " in JAX v0.9.0; use jax.enable_x64(False) instead."),
    _disable_x64
  ),
  "enable_x64": (
    ("jax.experimental.x64_context.enable_x64 is deprecated in JAX v0.8.0 and will be removed"
     " in JAX v0.9.0; use jax.enable_x64(True) instead."),
    _enable_x64
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  enable_x64 = _enable_x64
  disable_x64 = _disable_x64
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
