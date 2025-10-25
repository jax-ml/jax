# Copyright 2018 The JAX Authors.
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

from jax._src import deprecations as _deps

_deps.warn(
    'jax-lib-module',
    (
        'jax.lib.xla_bridge module will be removed in JAX v0.9.0;'
        ' all its APIs were deprecated and removed by JAX v0.8.0.'
    ),
    stacklevel=4
)

_deprecations = {
    # Finalized in JAX v0.8.0; remove these messages in v0.9.0.
    "get_backend": (
        (
            "jax.lib.xla_bridge.get_backend is deprecated and will be removed"
            " in JAX v0.8.0; use jax.extend.backend.get_backend, and please"
            " note that you must `import jax.extend` explicitly."
        ),
        None,
    ),
    "get_compile_options": (
        (
            "jax.lib.xla_bridge.get_compile_options is deprecated in JAX v0.7.0"
            " and will be removed in JAX v0.8.0. Use"
            " jax.extend.backend.get_compile_options, and please note that you"
            " must `import jax.extend` explicitly."
        ),
        None,
    ),
}

__getattr__ = _deps.deprecation_getattr(__name__, _deprecations)
del _deps
