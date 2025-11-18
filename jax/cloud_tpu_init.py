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

import warnings

warnings.warn(
    "jax.cloud_tpu_init was deprecated in JAX v0.8.1. You should remove imports"
    " of this module.",
    DeprecationWarning, stacklevel=1
)

del warnings

from jax._src.cloud_tpu_init import cloud_tpu_init as _cloud_tpu_init

_deprecations = {
  # Added 2025-10-28, remove in JAX 0.10.
  "cloud_tpu_init": (
    "jax.cloud_tpu_init was deprecated in JAX v0.8.1. You do not need to call "
    "this function explicitly; JAX calls this function automatically.",
    _cloud_tpu_init
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  cloud_tpu_init = _cloud_tpu_init
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
