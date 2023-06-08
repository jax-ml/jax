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

# TODO(phawkins): fix users of these aliases and delete this file.

from jax._src.abstract_arrays import array_types as _deprecated_array_types
from jax._src.core import (
  ShapedArray as _deprecated_ShapedArray,
  raise_to_shaped as _deprecated_raise_to_shaped,
)

_deprecations = {
  # Added 06 June 2023
  "array_types": (
    "jax.abstract_arrays.array_types is deprecated.",
    _deprecated_array_types,
  ),
  "ShapedArray": (
    "jax.abstract_arrays.ShapedArray is deprecated. Use jax.core.ShapedArray.",
    _deprecated_ShapedArray,
  ),
  "raise_to_shaped": (
    "jax.abstract_arrays.raise_to_shaped is deprecated. Use jax.core.raise_to_shaped.",
    _deprecated_raise_to_shaped,
  ),
}

import typing
if typing.TYPE_CHECKING:
  from jax._src.abstract_arrays import array_types as array_types
  from jax._src.core import ShapedArray as ShapedArray
  from jax._src.core import raise_to_shaped as raise_to_shaped
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
