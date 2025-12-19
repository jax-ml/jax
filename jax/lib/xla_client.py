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
del _deps
