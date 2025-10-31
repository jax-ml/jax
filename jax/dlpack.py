# Copyright 2020 The JAX Authors.
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


import jax._src.dlpack

from jax._src.dlpack import (
  from_dlpack as from_dlpack,
  is_supported_dtype as is_supported_dtype,
)

_deprecations = {
    # Deprecated in JAX v0.7.0
    "SUPPORTED_DTYPES": (
        (
            "jax.SUPPORTED_DTYPES is deprecated in JAX v0.7.0 and will be removed"
            " in JAX v0.8.0. Use jax.dlpack.is_supported_dtype() instead."
        ),
        None,
    ),
}


from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = jax._src.deprecations.deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
