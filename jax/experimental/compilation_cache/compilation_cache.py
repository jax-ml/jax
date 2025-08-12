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

from jax._src.compilation_cache import (
  is_initialized as _deprecated_is_initialized,
  initialize_cache as _deprecated_initialize_cache,
  set_cache_dir as set_cache_dir,
  reset_cache as reset_cache,
)

_deprecations = {
    # Added for v0.7.1; deprecation warning has been raised since v0.4.24
    "is_initialized": (
        (
            "compilation_cache.is_initialized was deprecated in JAX v0.4.24 and will"
            " be removed in JAX v0.8.0."
        ),
        _deprecated_is_initialized,
    ),
    "initialize_cache": (
        (
            "compilation_cache.initialize_cache was deprecated in JAX v0.4.24 and will"
            " be removed in JAX v0.8.0. use compilation_cache.set_cache_dir instead."
        ),
        _deprecated_initialize_cache,
    ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  is_initialized = _deprecated_is_initialized
  initialize_cache = _deprecated_initialize_cache
else:
  from jax._src.deprecations import deprecation_getattr
  __getattr__ = deprecation_getattr(__name__, _deprecations)
  del deprecation_getattr
del _typing
