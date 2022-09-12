# Copyright 2022 Google LLC
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

from __future__ import annotations

from typing import Any, Sequence, Union
from typing_extensions import Protocol
import numpy as np

class HasDtypeAttribute(Protocol):
  dtype: Dtype

Dtype = np.dtype

# Any is here to allow scalar types like np.int32.
# TODO(jakevdp) figure out how to specify these more strictly.
DtypeLike = Union[Any, str, np.dtype, HasDtypeAttribute]

# Shapes are tuples of dimension sizes, which are normally integers. We allow
# modules to extend the set of dimension sizes to contain other types, e.g.,
# symbolic dimensions in jax2tf.shape_poly.DimVar and masking.Poly.
DimSize = Union[int, Any]  # extensible
Shape = Sequence[DimSize]

# Array is a type annotation for standard JAX arrays and tracers produced by
# core functions in jax.lax and jax.numpy; it is not meant to include
# future non-standard array types like KeyArray and BInt.
# For now we set it to Any; in the future this will be more restrictive
# (see https://github.com/google/jax/pull/11859)
# TODO(jakevdp): make this conform to the JEP 12049 plan.
Array = Any

# ArrayLike is a Union of all objects that can be implicitly converted to a standard
# JAX array (i.e. not including future non-standard array types like KeyArray and BInt).
ArrayLike = Union[
  Array,  # JAX array type
  np.ndarray,  # NumPy array type
  np.bool_, np.number,  # NumPy scalar types
  bool, int, float, complex,  # Python scalar types
]
