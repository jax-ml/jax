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

from jax._src.lib import _jax

import numpy as np

# TypedInt, TypedFloat, and TypedComplex are subclasses of int, float, and
# complex that carry a JAX dtype. Canonicalization forms these types from int,
# float, and complex. Repeated canonicalization, including under different
# jax_enable_x64 modes, preserves the dtype.


class TypedInt(int):

  dtype: np.dtype

  def __new__(cls, value: int, dtype: np.dtype):
    v = super(TypedInt, cls).__new__(cls, value)
    v.dtype = dtype
    return v

  def __repr__(self):
    return f'TypedInt({int(self)}, dtype={self.dtype.name})'

  def __getnewargs__(self):
    return (int(self), self.dtype)


class TypedFloat(float):

  dtype: np.dtype

  def __new__(cls, value: float, dtype: np.dtype):
    v = super(TypedFloat, cls).__new__(cls, value)
    v.dtype = dtype
    return v

  def __repr__(self):
    return f'TypedFloat({float(self)}, dtype={self.dtype.name})'

  def __str__(self):
    return str(float(self))

  def __getnewargs__(self):
    return (float(self), self.dtype)


class TypedComplex(complex):

  dtype: np.dtype

  def __new__(cls, value: complex, dtype: np.dtype):
    v = super(TypedComplex, cls).__new__(cls, value)
    v.dtype = dtype
    return v

  def __repr__(self):
    return f'TypedComplex({complex(self)}, dtype={self.dtype.name})'

  def __getnewargs__(self):
    return (complex(self), self.dtype)


_jax.set_typed_int_type(TypedInt)
_jax.set_typed_float_type(TypedFloat)
_jax.set_typed_complex_type(TypedComplex)


typed_scalar_types: set[type] = {TypedInt, TypedFloat, TypedComplex}


class TypedNdArray(np.ndarray):
  """A TypedNdArray is a host-side array used by JAX during tracing.

  TypedNdArray is a subclass of np.ndarray that carries additional JAX type
  information:
  * its type is not canonicalized by JAX, irrespective of the jax_enable_x64
    mode
  * it can be weakly typed.
  """

  weak_type: bool

  def __new__(cls, val: np.ndarray, weak_type: bool):
    obj = np.asarray(val).view(cls)
    obj.weak_type = weak_type
    return obj

  def __array_finalize__(self, obj):
    if obj is None:
      return
    self.weak_type = getattr(obj, 'weak_type', False)

  @property
  def val(self) -> np.ndarray:
    return np.asarray(self)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    inputs = tuple(np.asarray(x) if isinstance(x, TypedNdArray) else x
                   for x in inputs)
    if 'out' in kwargs:
      kwargs['out'] = tuple(
          np.asarray(x) if isinstance(x, TypedNdArray) else x
          for x in kwargs['out'])
    return getattr(ufunc, method)(*inputs, **kwargs)

  def __repr__(self):
    prefix = 'TypedNdArray('
    if self.weak_type:
      dtype_str = f'dtype={self.dtype.name}, weak_type=True)'
    else:
      dtype_str = f'dtype={self.dtype.name})'

    line_width = np.get_printoptions()['linewidth']
    if self.size == 0:
      s = f'[], shape={self.shape}'
    else:
      s = np.array2string(
          np.asarray(self),
          prefix=prefix,
          suffix=',',
          separator=', ',
          max_line_width=line_width,
      )
    last_line_len = len(s) - s.rfind('\n') + 1
    sep = ' '
    if last_line_len + len(dtype_str) + 1 > line_width:
      sep = ' ' * len(prefix)
    return f'{prefix}{s},{sep}{dtype_str}'

  def __reduce__(self):
    return (TypedNdArray, (np.asarray(self), self.weak_type))

  def __getnewargs__(self):
    return (np.asarray(self), self.weak_type)

_jax.set_typed_ndarray_type(TypedNdArray)
