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

from typing import Any

import numpy as np

from jax._src.typing import Array
from jax._src import core
from jax._src import dtypes
from jax._src.numpy.array import asarray


# Some objects below rewrite their __module__ attribute to this name.
_PUBLIC_MODULE_NAME = "jax.numpy"


class _ScalarMeta(type):
  dtype: np.dtype

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

def _make_scalar_type(np_scalar_type: type) -> _ScalarMeta:
  meta = _ScalarMeta(np_scalar_type.__name__, (object,),
                     {"dtype": np.dtype(np_scalar_type)})
  meta.__module__ = _PUBLIC_MODULE_NAME
  meta.__doc__ =\
  f"""A JAX scalar constructor of type {np_scalar_type.__name__}.

  While NumPy defines scalar types for each data type, JAX represents
  scalars as zero-dimensional arrays.
  """
  return meta

bool_ = _make_scalar_type(np.bool_)
uint2 = _make_scalar_type(dtypes.uint2)
uint4 = _make_scalar_type(dtypes.uint4)
uint8 = _make_scalar_type(np.uint8)
uint16 = _make_scalar_type(np.uint16)
uint32 = _make_scalar_type(np.uint32)
uint64 = _make_scalar_type(np.uint64)
int2 = _make_scalar_type(dtypes.int2)
int4 = _make_scalar_type(dtypes.int4)
int8 = _make_scalar_type(np.int8)
int16 = _make_scalar_type(np.int16)
int32 = _make_scalar_type(np.int32)
int64 = _make_scalar_type(np.int64)
float4_e2m1fn = _make_scalar_type(dtypes.float4_e2m1fn)
float8_e3m4 = _make_scalar_type(dtypes.float8_e3m4)
float8_e4m3 = _make_scalar_type(dtypes.float8_e4m3)
float8_e8m0fnu = _make_scalar_type(dtypes.float8_e8m0fnu)
float8_e4m3fn = _make_scalar_type(dtypes.float8_e4m3fn)
float8_e4m3fnuz = _make_scalar_type(dtypes.float8_e4m3fnuz)
float8_e5m2 = _make_scalar_type(dtypes.float8_e5m2)
float8_e5m2fnuz = _make_scalar_type(dtypes.float8_e5m2fnuz)
float8_e4m3b11fnuz = _make_scalar_type(dtypes.float8_e4m3b11fnuz)
bfloat16 = _make_scalar_type(dtypes.bfloat16)
float16 = _make_scalar_type(np.float16)
float32 = single = _make_scalar_type(np.float32)
float64 = double = _make_scalar_type(np.float64)
complex64 = csingle = _make_scalar_type(np.complex64)
complex128 = cdouble = _make_scalar_type(np.complex128)

int_ = int32 if dtypes.int_ == np.int32 else int64
uint = uint32 if dtypes.uint == np.uint32 else uint64
float_: Any = float32 if dtypes.float_ == np.float32 else float64
complex_ = complex64 if dtypes.complex_ == np.complex64 else complex128
