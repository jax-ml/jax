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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/google/jax/issues/7570

from jax.experimental.x64_context import (
  enable_x64 as enable_x64,
  disable_x64 as disable_x64,
)
from jax._src.callback import (
  io_callback as io_callback
)
from jax._src.earray import (
    EArray as EArray
)

from jax import numpy as _array_api


_deprecations = {
  # Deprecated 01 Aug 2024
  "array_api": (
    "jax.experimental.array_api import is no longer required as of JAX v0.4.32; "
    "jax.numpy supports the array API by default.",
    _array_api
  ),
}

from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
del _array_api
