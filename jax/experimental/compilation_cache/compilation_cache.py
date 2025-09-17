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
  set_cache_dir as set_cache_dir,
  reset_cache as reset_cache,
)

_deprecations = {
    # Finalized for v0.8.0; remove in v0.9.0
    "is_initialized": (
        (
            "compilation_cache.is_initialized was deprecated in JAX v0.4.24 and"
            " removed in JAX v0.8.0."
        ),
        None,
    ),
    "initialize_cache": (
        (
            "compilation_cache.initialize_cache was deprecated in JAX v0.4.24 and"
            " removed in JAX v0.8.0. use compilation_cache.set_cache_dir instead."
        ),
        None,
    ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  pass
else:
  from jax._src.deprecations import deprecation_getattr
  __getattr__ = deprecation_getattr(__name__, _deprecations)
  del deprecation_getattr
del _typing
