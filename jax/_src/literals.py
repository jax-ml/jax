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

from typing import Sequence
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


class TypedNdArray:
  """A TypedNdArray is a host-side array used by JAX during tracing.

  To most intents and purposes a TypedNdArray is a thin wrapper around a numpy
  array and should act like it. The primary differences are that a TypedNdArray
  carries a JAX type:
  * its type is not canonicalized by JAX, irrespective of the jax_enable_x64
    mode
  * it can be weakly typed.
  """

  __slots__ = ('val', 'weak_type')

  val: np.ndarray
  weak_type: bool

  def __init__(self, val: np.ndarray, weak_type: bool):
    self.val = val
    self.weak_type = weak_type

  @property
  def dtype(self) -> np.dtype:
    return self.val.dtype

  @property
  def shape(self) -> tuple[int, ...]:
    return self.val.shape

  @property
  def strides(self) -> Sequence[int]:
    return self.val.strides

  @property
  def ndim(self) -> int:
    return self.val.ndim

  @property
  def size(self) -> int:
    return self.val.size

  def __len__(self) -> int:
    return self.val.__len__()

  def __repr__(self):
    prefix = 'TypedNdArray('
    if self.weak_type:
      dtype_str = f'dtype={self.val.dtype.name}, weak_type=True)'
    else:
      dtype_str = f'dtype={self.val.dtype.name})'

    line_width = np.get_printoptions()['linewidth']
    if self.size == 0:
      s = f'[], shape={self.val.shape}'
    else:
      s = np.array2string(
          self.val,
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

  def __array__(self, dtype=None, copy=None):
    # You might think that we can do the following here:
    # return self.val.__array__(dtype=dtype, copy=copy)
    # Unfortunately __array__ appears to be buggy on NumPy < 2.3 and interprets
    # the "dtype=None" as "the default float type".
    # TODO(phawkins): revert to the above form once NumPy 2.3 is the minimum
    # supported version.
    return np.asarray(self.val, dtype=dtype, copy=copy)  # pytype: disable=wrong-keyword-args

  def __add__(self, other):
    return self.val.__add__(other)

  def __sub__(self, other):
    return self.val.__sub__(other)

  def __mul__(self, other):
    return self.val.__mul__(other)

  def __floordiv__(self, other):
    return self.val.__floordiv__(other)

  def __truediv__(self, other):
    return self.val.__truediv__(other)

  def __mod__(self, other):
    return self.val.__mod__(other)

  def __pow__(self, other):
    return self.val.__pow__(other)

  def __radd__(self, other):
    return self.val.__radd__(other)

  def __rsub__(self, other):
    return self.val.__rsub__(other)

  def __rmul__(self, other):
    return self.val.__rmul__(other)

  def __rtruediv__(self, other):
    return self.val.__rtruediv__(other)

  def __rfloordiv__(self, other):
    return self.val.__rfloordiv__(other)

  def __rmod__(self, other):
    return self.val.__rmod__(other)

  def __rpow__(self, other):
    return self.val.__rpow__(other)

  def __getitem__(self, index):
    return self.val.__getitem__(index)

  def __bool__(self):
    return self.val.__bool__()

  def __int__(self):
    return self.val.__int__()

  def __float__(self):
    return self.val.__float__()

  def __complex__(self):
    return self.val.__complex__()

  def __index__(self):
    return self.val.__index__()

  def __lt__(self, other):
    return self.val.__lt__(other)

  def __le__(self, other):
    return self.val.__le__(other)

  def __eq__(self, other):
    return self.val.__eq__(other)

  def __ne__(self, other):
    return self.val.__ne__(other)

  def __gt__(self, other):
    return self.val.__gt__(other)

  def __ge__(self, other):
    return self.val.__ge__(other)

  def __abs__(self):
    return self.val.__abs__()

  def reshape(self, *args, **kw):
    return self.val.reshape(*args, **kw)

  def item(self, *args):
    return self.val.item(*args)

  @property
  def T(self):
    return self.val.T

  @property
  def mT(self):
    return self.val.mT

  def clip(self, *args, **kwargs):
    return self.val.clip(*args, **kwargs)

  def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
    return self.val.astype(
        dtype, order=order, casting=casting, subok=subok, copy=copy
    )

  def tobytes(self, order='C'):
    return self.val.tobytes(order=order)

_jax.set_typed_ndarray_type(TypedNdArray)
