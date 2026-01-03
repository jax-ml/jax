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

_deprecations = {
  # Remove in v0.10.0
  "disable_x64": (
    ("jax.experimental.x64_context.disable_x64 was removed in JAX v0.9.0;"
     " use jax.enable_x64(False) instead."),
    None
  ),
  "enable_x64": (
    ("jax.experimental.x64_context.enable_x64 was removed in JAX v0.9.0;"
     " use jax.enable_x64(True) instead."),
    None
  ),
}

from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
