# Copyright 2024 The JAX Authors.
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
        'jax.lib.xla_client module will be removed in JAX v0.9.0;'
        ' all its APIs were deprecated and removed by JAX v0.8.0.'
    ),
    stacklevel=4
)

_deprecations = {
    # Finalized in JAX v0.8.0; remove these messages in v0.9.0.
    "Client": (
        (
            "jax.lib.xla_client.Client was deprecated in JAX v0.6.0 and will be"
            " removed in JAX v0.8.0"
        ),
        None,
    ),
    "CompileOptions": (
        (
            "jax.lib.xla_client.CompileOptions was deprecated in JAX v0.6.0 and"
            " will be removed in JAX v0.8.0"
        ),
        None,
    ),
    "Frame": (
        (
            "jax.lib.xla_client.Frame was deprecated in JAX v0.6.0 and will be"
            " removed in JAX v0.8.0"
        ),
        None,
    ),
    "HloSharding": (
        (
            "jax.lib.xla_client.HloSharding was deprecated in JAX v0.6.0 and"
            " will be removed in JAX v0.8.0"
        ),
        None,
    ),
    "OpSharding": (
        (
            "jax.lib.xla_client.OpSharding was deprecated in JAX v0.6.0 and"
            " will be removed in JAX v0.8.0"
        ),
        None,
    ),
    "Traceback": (
        (
            "jax.lib.xla_client.Traceback was deprecated in JAX v0.6.0 and will"
            " be removed in JAX v0.8.0"
        ),
        None,
    ),
}

__getattr__ = _deps.deprecation_getattr(__name__, _deprecations)
del _deps
