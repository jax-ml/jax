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
import jax._src.deprecations

from jax._src.dlpack import (
  from_dlpack as from_dlpack,
  SUPPORTED_DTYPES as SUPPORTED_DTYPES,
)

_deprecations = {
    "to_dlpack": (
        (
            "jax.dlpack.to_dlpack was deprecated in JAX v0.6.0 and will be"
            " removed in JAX v0.7.0. Please use the newer DLPack API based on"
            " __dlpack__ and __dlpack_device__ instead. Typically, you can pass"
            " a JAX array directly to the `from_dlpack` function of another"
            " framework without using `to_dlpack`."
        ),
        jax._src.dlpack.to_dlpack,
    ),
}


import typing as _typing

if _typing.TYPE_CHECKING:
  to_dlpack = jax._src.dlpack.to_dlpack
else:
  __getattr__ = jax._src.deprecations.deprecation_getattr(
      __name__, _deprecations
  )
del _typing
