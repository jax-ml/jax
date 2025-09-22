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

_deprecations = {  # pylint: disable=g-statement-before-imports
    # Added April 4 2025.
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

from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
