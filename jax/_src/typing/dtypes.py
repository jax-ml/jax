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

from __future__ import annotations

from typing import Any, Literal, Protocol, TypeVar, Union
import numpy as np
import ml_dtypes

DType = np.dtype
_ScalarType = TypeVar("_ScalarType", covariant=True, bound=np.generic)  # pytype: disable=not-supported-yet

# TODO(jakevdp, froystig): make ExtendedDType a protocol
ExtendedDType = Any

class SupportsDType(Protocol[_ScalarType]):
  @property
  def dtype(self) -> DType[_ScalarType]: ...

# DTypeLike is meant to annotate inputs to np.dtype that return
# a valid JAX dtype. It's different than numpy.typing.DTypeLike
# because JAX doesn't support objects or structured dtypes.
# Unlike np.typing.DTypeLike, we exclude None, and instead require
# explicit annotations when None is acceptable.
# TODO(jakevdp): consider whether to add ExtendedDtype to the union.
DTypeLike = Union[
  str,                 # like 'float32', 'int32'
  type[Any],           # like np.float32, np.int32, float, int
  DType[Any],          # like np.dtype('float32'), np.dtype('int32')
  SupportsDType[Any],  # like jnp.float32, jnp.int32
]

BoolLiterals = Literal['bool', '?']

Int8Literals = Literal['int8', 'i1']
Int16Literals = Literal['int16', 'i2']
Int32Literals = Literal['int32', 'i4']
Int64Literals = Literal['int', 'int64', 'i8']

UInt8Literals = Literal['uint8', 'u1']
UInt16Literals = Literal['uint16', 'u2']
UInt32Literals = Literal['uint32', 'u4']
UInt64Literals = Literal['uint', 'uint64', 'u8']

BFloat16Literals = Literal['bfloat16']
Float16Literals = Literal['float16', 'f2']
Float32Literals = Literal['float32', 'f4']
Float64Literals = Literal['float', 'float64', 'f8']

Complex64Literals = Literal['complex64', 'c8']
Complex128Literals = Literal['complex', 'complex128', 'c16']

# TODO(jakevdp): the use of things like type[float] and type[np.floating]
#                below are not strictly correct: can we do better?

DTypeLikeBool = Union[
  type[bool],
  type[np.bool_],
  DType[np.bool_],
  SupportsDType[np.bool_],
  BoolLiterals,
]

DTypeLikeUInt = Union[
  type[np.unsignedinteger],
  DType[np.unsignedinteger],
  SupportsDType[np.unsignedinteger],
  UInt8Literals,
  UInt16Literals,
  UInt32Literals,
  UInt64Literals
]

DTypeLikeInt = Union[
  type[int],
  type[np.signedinteger],
  DType[np.signedinteger],
  SupportsDType[np.signedinteger],
  Int8Literals,
  Int16Literals,
  Int32Literals,
  Int64Literals,
]

DTypeLikeFloat = Union[
  type[float],
  type[np.floating],
  DType[np.floating],
  SupportsDType[np.floating],
  DType[ml_dtypes.bfloat16],
  SupportsDType[ml_dtypes.bfloat16],
  BFloat16Literals,
  Float16Literals,
  Float32Literals,
  Float64Literals
]

DTypeLikeComplex = Union[
  type[complex],
  type[np.complexfloating],
  DType[np.complexfloating],
  SupportsDType[np.complexfloating],
  Complex64Literals,
  Complex128Literals,
]
