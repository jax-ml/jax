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

from jax._src.config import config as _deprecated_config  # noqa: F401

# Deprecations

_deprecations = {
    # Added October 27, 2023
    "config": (
        "Accessing jax.config via the jax.config submodule is deprecated.",
        _deprecated_config),
}

import typing
if typing.TYPE_CHECKING:
  config = _deprecated_config
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del _deprecated_config
