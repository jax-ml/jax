# Copyright 2025 The JAX Authors.
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


# At present JAX doesn't have a reason to distinguish between scalars and arrays
# in its object system. Further, we want JAX scalars to have the same type
# promotion behaviors as JAX arrays. Rather than introducing a new type of JAX
# scalar object with JAX promotion behaviors, instead we make the JAX scalar
# types return JAX arrays when instantiated.

import types
from typing import Any

from jax._src import core
from jax._src import dtypes
from jax._src.numpy.array_constructors import asarray
from jax._src.typing import Array
import numpy as np

# Some objects below rewrite their __module__ attribute to this name.
_PUBLIC_MODULE_NAME = "jax.numpy"


class _ScalarMeta(type):
  dtype: np.dtype

  @property
  def __numpy_dtype__(self) -> np.dtype:
    # __numpy_dtype__ protocol added in NumPy v2.4.0.
    return self.dtype

  def __hash__(self) -> int:
    return hash(self.dtype.type)

  def __eq__(self, other: Any) -> bool:
    return self is other or self.dtype.type == other

  def __ne__(self, other: Any) -> bool:
    return not (self == other)

  def __call__(self, x: Any) -> Array:
    return asarray(x, dtype=self.dtype)

  def __instancecheck__(self, instance: Any) -> bool:
    return isinstance(instance, self.dtype.type)

def _abstractify_scalar_meta(x):
  raise TypeError(f"JAX scalar type {x} cannot be interpreted as a JAX array.")
core.pytype_aval_mappings[_ScalarMeta] = _abstractify_scalar_meta


def _make_scalar_class(
    name: str, parent: type[_ScalarMeta] = _ScalarMeta
) -> type[_ScalarMeta]:
  meta = types.new_class(f'_ScalarType{name}', bases=(parent,))
  meta.__module__ = _PUBLIC_MODULE_NAME
  return meta


def _make_scalar_type(
    np_scalar_type: type, scalar_class: type[_ScalarMeta] = _ScalarMeta
) -> _ScalarMeta:
  meta = scalar_class(
      np_scalar_type.__name__,
      (scalar_class or object,),
      {'dtype': np.dtype(np_scalar_type)},
  )

  meta.__module__ = _PUBLIC_MODULE_NAME
  meta.__doc__ =\
  f"""A JAX scalar constructor of type {np_scalar_type.__name__}.

  While NumPy defines scalar types for each data type, JAX represents
  scalars as zero-dimensional arrays.
  """
  return meta


# Define a hierarchy of types.
Number = _make_scalar_class('Number')
Integer = _make_scalar_class('Integer', Number)
SignedInteger = _make_scalar_class('SignedInteger', Integer)
UnsignedInteger = _make_scalar_class('UnsignedInteger', Integer)
Floating = _make_scalar_class('Floating', Number)
RealFloating = _make_scalar_class('RealFloating', Floating)
ComplexFloating = _make_scalar_class('ComplexFloating', Floating)

bool_ = _make_scalar_type(np.bool_, Number)
if dtypes.uint1 is not None:
  uint1 = _make_scalar_type(dtypes.uint1, UnsignedInteger)
uint2 = _make_scalar_type(dtypes.uint2, UnsignedInteger)
uint4 = _make_scalar_type(dtypes.uint4, UnsignedInteger)
uint8 = _make_scalar_type(np.uint8, UnsignedInteger)
uint16 = _make_scalar_type(np.uint16, UnsignedInteger)
uint32 = _make_scalar_type(np.uint32, UnsignedInteger)
uint64 = _make_scalar_type(np.uint64, UnsignedInteger)
if dtypes.int1 is not None:
  int1 = _make_scalar_type(dtypes.int1, SignedInteger)
int2 = _make_scalar_type(dtypes.int2, SignedInteger)
int4 = _make_scalar_type(dtypes.int4, SignedInteger)
int8 = _make_scalar_type(np.int8, SignedInteger)
int16 = _make_scalar_type(np.int16, SignedInteger)
int32 = _make_scalar_type(np.int32, SignedInteger)
int64 = _make_scalar_type(np.int64, SignedInteger)
float4_e2m1fn = _make_scalar_type(dtypes.float4_e2m1fn, RealFloating)
float6_e2m3fn = _make_scalar_type(dtypes.float6_e2m3fn, RealFloating)
float6_e3m2fn = _make_scalar_type(dtypes.float6_e3m2fn, RealFloating)
float8_e3m4 = _make_scalar_type(dtypes.float8_e3m4, RealFloating)
float8_e4m3 = _make_scalar_type(dtypes.float8_e4m3, RealFloating)
float8_e8m0fnu = _make_scalar_type(dtypes.float8_e8m0fnu, RealFloating)
float8_e4m3fn = _make_scalar_type(dtypes.float8_e4m3fn, RealFloating)
float8_e4m3fnuz = _make_scalar_type(dtypes.float8_e4m3fnuz, RealFloating)
float8_e5m2 = _make_scalar_type(dtypes.float8_e5m2, RealFloating)
float8_e5m2fnuz = _make_scalar_type(dtypes.float8_e5m2fnuz, RealFloating)
float8_e4m3b11fnuz = _make_scalar_type(dtypes.float8_e4m3b11fnuz, RealFloating)
bfloat16 = _make_scalar_type(dtypes.bfloat16, RealFloating)
float16 = _make_scalar_type(np.float16, RealFloating)
float32 = single = _make_scalar_type(np.float32, RealFloating)
float64 = double = _make_scalar_type(np.float64, RealFloating)
complex64 = csingle = _make_scalar_type(np.complex64, ComplexFloating)
complex128 = cdouble = _make_scalar_type(np.complex128, ComplexFloating)

int_ = int64
uint = uint64
float_ = float64
complex_ = complex128
