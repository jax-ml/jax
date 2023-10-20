# Copyright 2023 The JAX Authors.
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

"""
`jax._src.typing`: JAX type annotations
---------------------------------------

This submodule is a work in progress; when we finalize the contents here, it will be
exported at `jax.typing`. Until then, the contents here should be considered unstable
and may change without notice.

To see the proposal that led to the development of these tools, see
https://github.com/google/jax/pull/11859/.
"""

# Array is a type annotation for standard JAX arrays and tracers produced by
# core functions in jax.lax and jax.numpy; it is not meant to include
# future non-standard array types like KeyArray and BInt. It is imported above.

# ArrayLike is a Union of all objects that can be implicitly converted to a standard
# JAX array (i.e. not including future non-standard array types like KeyArray and BInt).
# It's different than np.typing.ArrayLike in that it doesn't accept arbitrary sequences,
# nor does it accept string data.

from jax._src.basearray import (
    Array as Array,
    ArrayLike as ArrayLike,
)

from jax._src.typing.core import (
    DimSize as DimSize,
    DuckTypedArray as DuckTypedArray,
    Shape as Shape,
)

from jax._src.typing.dtypes import (
    DType as DType,
    DTypeLike as DTypeLike,
    DTypeLikeBool as DTypeLikeBool,
    DTypeLikeComplex as DTypeLikeComplex,
    DTypeLikeFloat as DTypeLikeFloat,
    DTypeLikeInt as DTypeLikeInt,
    DTypeLikeUInt as DTypeLikeUInt,
    ExtendedDType as ExtendedDType,
)
