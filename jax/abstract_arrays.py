# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from itertools import product
import operator as op

import numpy as onp

from . import core
from . import ad_util
from . import dtypes
from .util import prod, partialmethod

def concretization_err_msg(fun):
  fname = getattr(fun, "__name__", fun)
  msg = ("Abstract value passed to `{}`, which requires a concrete value. "
         "The function to be transformed can't be traced at the required level "
         "of abstraction. If using `jit`, try using `static_argnums` or "
         "applying `jit` to smaller subfunctions instead.")
  return msg.format(fname)

def concretization_function_error(fun):
  def error(self, *args):
    raise TypeError(concretization_err_msg(fun))
  return error


class UnshapedArray(core.AbstractValue):
  __slots__ = ['dtype', 'weak_type']
  array_abstraction_level = 2

  def __init__(self, dtype, weak_type=False):
    self.dtype = onp.dtype(dtypes.canonicalize_dtype(dtype))
    self.weak_type = weak_type

  def __eq__(self, other):
    return (type(self) is type(other) and self.dtype == other.dtype and
            self.weak_type == other.weak_type)

  def __ne__(self, other):
    return not self == other

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `onp.zeros(3).dtype is onp.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash((self.dtype, self.weak_type))

  def __repr__(self):
    return '{}({}{})'.format(self.__class__.__name__, self.str_short(),
                             ", weak_type=True" if self.weak_type else "")

  _bool = _nonzero = concretization_function_error(bool)
  _float   = concretization_function_error(float)
  _int     = concretization_function_error(int)
  _complex = concretization_function_error(complex)
  _hex     = concretization_function_error(hex)
  _oct     = concretization_function_error(oct)

  def at_least_vspace(self):
    return self

  def join(self, other):
    if self.dtype == other.dtype:
      if self.weak_type == other.weak_type:
        return self
      else:
        return UnshapedArray(self.dtype, weak_type=False)
    else:
      raise TypeError(self, other)

  def str_short(self):
    return self.dtype.name

  def strip_weak_type(self):
    """Returns a copy of the aval with weak_type=False."""
    return UnshapedArray(self.dtype) if self.weak_type else self


class ShapedArray(UnshapedArray):
  __slots__ = ['shape']
  array_abstraction_level = 1

  def __init__(self, shape, dtype, weak_type=False):
    super(ShapedArray, self).__init__(dtype, weak_type=weak_type)
    self.shape = shape

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: prod(self.shape))

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape
            and self.weak_type == other.weak_type)

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `onp.zeros(3).dtype is onp.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash((self.shape, self.dtype, self.weak_type))

  def at_least_vspace(self):
    return self

  def join(self, other):
    if self.shape == other.shape and self.dtype == other.dtype:
      if self.weak_type == other.weak_type:
        return self
      else:
        return ShapedArray(self.shape, self.dtype, weak_type=False)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype)
    else:
      raise TypeError(self, other)

  def str_short(self):
    shapestr = ','.join(map(str, self.shape))
    return '{}[{}]'.format(self.dtype.name, shapestr)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError:
      raise TypeError("len() of unsized object")  # same as numpy error

  def _len(self, ignored_tracer):
    return len(self)

  def strip_weak_type(self):
    return ShapedArray(self.shape, self.dtype) if self.weak_type else self

def _forward_to_value(self, fun, ignored_tracer, *args):
  return fun(self.val, *args)

class ConcreteArray(ShapedArray):
  __slots__ = ['val']
  array_abstraction_level = 0

  def __init__(self, val, weak_type=False):
    super(ConcreteArray, self).__init__(onp.shape(val), onp.result_type(val),
                                        weak_type=weak_type)
    # Note: canonicalized self.dtype doesn't necessarily match self.val
    self.val = val

    assert self.dtype != onp.dtype('O') or is_polymorphic(val)

  def __eq__(self, other):
    return (type(self) is type(other) and self.dtype == other.dtype
            and self.shape == other.shape and self.weak_type == other.weak_type
            and onp.all(self.val == other.val))

  def __hash__(self):
    return id(self.val)

  def at_least_vspace(self):
    return ShapedArray(self.shape, self.dtype, weak_type=self.weak_type)

  def join(self, other):
    if self == other:
      return self
    elif self.shape == other.shape and self.dtype == other.dtype:
      return ShapedArray(self.shape, self.dtype,
                         weak_type=self.weak_type and other.weak_type)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype,
                           weak_type=self.weak_type and other.weak_type)
    else:
      raise TypeError(self, other)

  def str_short(self):
    return str(self.val)

  def strip_weak_type(self):
    return ConcreteArray(self.val) if self.weak_type else self

  _bool = _nonzero = partialmethod(_forward_to_value, bool)
  _float   = partialmethod(_forward_to_value, float)
  _int     = partialmethod(_forward_to_value, int)
  _complex = partialmethod(_forward_to_value, complex)
  _hex     = partialmethod(_forward_to_value, hex)
  _oct     = partialmethod(_forward_to_value, oct)

class AbstractToken(core.AbstractValue): pass

abstract_token = AbstractToken()


def make_shaped_array(x):
  dtype = dtypes.canonicalize_dtype(dtypes.result_type(x))
  return ShapedArray(onp.shape(x), dtype)

def zeros_like_array(x):
  dtype = dtypes.canonicalize_dtype(dtypes.result_type(x))
  return onp.broadcast_to(onp.array(0, dtype), onp.shape(x))

array_types = {onp.ndarray, onp.bool_,
               onp.int8, onp.int16, onp.int32, onp.int64,
               onp.uint8, onp.uint16, onp.uint32, onp.uint64,
               dtypes.bfloat16, onp.float16, onp.float32, onp.float64,
               onp.complex64, onp.complex128,
               onp.longlong}

for t in array_types:
  core.pytype_aval_mappings[t] = ConcreteArray
  ad_util.jaxval_zeros_likers[t] = zeros_like_array


def zeros_like_shaped_array(aval):
  assert isinstance(aval, ShapedArray)
  return onp.zeros(aval.shape, dtype=aval.dtype)

ad_util.aval_zeros_likers[ShapedArray] = zeros_like_shaped_array

def raise_to_shaped(aval, weak_type=False):
  if isinstance(aval, ShapedArray):
    return ShapedArray(aval.shape, aval.dtype, weak_type=weak_type)
  elif aval is core.abstract_unit:
    return core.abstract_unit
  elif aval is abstract_token:
    return abstract_token
  else:
    raise TypeError(type(aval))

core.literalable_types.update(array_types)

def _zeros_like_python_scalar(x):
  return onp.array(0, dtypes.python_scalar_dtypes[type(x)])

def _make_concrete_python_scalar(x):
  return ConcreteArray(
    onp.array(x, dtype=dtypes.python_scalar_dtypes[type(x)]),
    weak_type=True)

for t in dtypes.python_scalar_dtypes.keys():
  core.pytype_aval_mappings[t] = _make_concrete_python_scalar
  ad_util.jaxval_zeros_likers[t] = _zeros_like_python_scalar

core.literalable_types.update(dtypes.python_scalar_dtypes.keys())

def to_index(x):
  """Like operator.index, but allowing polymorphic dimensions.
  Not implemented as `Poly.__index__`, since operator.index only allows ints."""
  return x if type(x) is Poly else op.index(x)

# TODO remove remaining usages:
def is_polymorphic(shape):
  return any(map(lambda d: type(d) is Poly, shape))

def eval_polymorphic_shape(shape, values_dict):
  return tuple(dim.evaluate(values_dict) if type(dim) is Poly else dim
               for dim in shape)

def _ensure_poly(p):
  if type(p) is Poly:
    return p

  return Poly({Mon(): p})

class Poly(dict):
  """Polynomial with integer coefficients,
  usable as element in a polymorphic shape.

  type Poly = Map Mon Int -- monomials to coeffs
  type Mon = Map Str Int
  """

  def __init__(self, coeffs):
    # Makes sure Polynomials are always in canonical form to simplify operators:
    coeffs = {mon: op.index(coeff) for mon, coeff in coeffs.items() if coeff != 0}
    coeffs = {Mon(): 0} if len(coeffs) == 0 else coeffs
    super().__init__(coeffs)

  def __add__(self, other):
    coeffs = self.copy()

    for mon, coeff in _ensure_poly(other).items():
      coeffs[mon] = coeffs.get(mon, 0) + coeff

    return Poly(coeffs)

  def __sub__(self, other):
    return self + -other

  def __neg__(self):
    return Poly({mon: -coeff for mon, coeff in self.items()})

  def __mul__(self, other):
    coeffs = dict()
    for (mon1, coeff1), (mon2, coeff2) \
            in product(self.items(), _ensure_poly(other).items()):
      mon = mon1 * mon2
      coeffs[mon] = coeffs.get(mon, 0) + coeff1 * coeff2

    return Poly(coeffs)

  def __rmul__(self, other):
    return self * other

  def __radd__(self, other):
    return self + other

  def __rsub__(self, other):
    return self + -other

  def __floordiv__(self, divisor):
    q, _ = divmod(self, divisor)  # pytype: disable=wrong-arg-types
    return q

  def __mod__(self, divisor):
    _, r = divmod(self, divisor)  # pytype: disable=wrong-arg-types
    return r

  def __divmod__(self, divisor):
    if self.is_constant:
      return divmod(int(self), divisor)

    def divided(count):
      q, r = divmod(count, divisor)
      if r != 0:
        raise ValueError('shapecheck currently only supports strides '
                         'that exactly divide the strided axis length.')
      return q

    return Poly(
      {k: coeff // divisor if k.degree == 0 else divided(coeff)
       for k, coeff in self.items()}), self.get(Mon(), 0) % divisor

  def __hash__(self):
    return hash(tuple(sorted(self.items())))

  def __eq__(self, other):
    return dict.__eq__(self, _ensure_poly(other))

  def __ne__(self, other):
    return not self == other

  def __ge__(self, other):
    other = _ensure_poly(other)

    if other.is_constant and self.is_constant:
      return int(self) >= int(other)

    if other.is_constant and int(other) <= 1:
      # Assume polynomials > 0, allowing to use shape rules of binops, conv:
      return True

    if self.is_constant and int(self) <= 0:
      return False  # See above.

    if self == other:
      return True

    raise ValueError('Polynomials comparison "{} >= {}" is inconclusive.'
                     .format(self, other))

  def __le__(self, other):
    return _ensure_poly(other) >= self

  def __lt__(self, other):
    return not (self >= other)

  def __gt__(self, other):
    return not (_ensure_poly(other) >= self)

  def __str__(self):
    return ' + '.join('{} {}'.format(v, k)
                      if (v != 1 or k.degree == 0) else str(k)
                      for k, v in sorted(self.items())).strip()

  def __int__(self):
    assert self.is_constant

    return op.index(next(iter(self.values())))

  def evaluate(self, values_dict):
    return sum(coeff * prod([values_dict[id] ** deg for id, deg in mon.items()])
               for mon, coeff in self.items())

  @property
  def is_constant(self):
    return len(self) == 1 and next(iter(self)).degree == 0

class Mon(dict):  # type Mon = Map Id Int -- ids to degrees
  def __hash__(self):
    return hash(tuple(self.items()))

  def __str__(self):
    return ' '.join('{}**{}'.format(k, v) if v != 1 else str(k)
                    for k, v in sorted(self.items()))

  def __lt__(self, other):
    # sort by total degree, then lexicographically on indets
    self_key = self.degree, tuple(sorted(self))
    other_key = other.degree, tuple(sorted(other))
    return self_key < other_key

  def __mul__(self, other):
    return Mon(Counter(self) + Counter(other))

  @property
  def degree(self):
    return sum(self.values())