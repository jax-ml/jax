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

import numpy as onp
import six

from . import core
from . import ad_util
from . util import prod
from .lib import xla_bridge


def concretization_err_msg(fun):
  fname = getattr(fun, "__name__", fun)
  return ("Abstract value passed to function {} that requires a concrete value. "
          "Possibly tracing Python control flow using abstract values. "
          "If so, try using lax.cond or lax.while instead.").format(fname)

def concretization_function_error(fun):
  def error(self, *args):
    raise TypeError(concretization_err_msg(fun))
  return error


class UnshapedArray(core.AbstractValue):
  __slots__ = ['dtype']
  array_abstraction_level = 3

  def __init__(self, dtype):
    self.dtype = dtype

  def __eq__(self, other):
    return type(self) is type(other) and self.dtype == other.dtype

  def __hash__(self):
    return hash(str(self.dtype))

  def __repr__(self):
    return '{}({})'.format(self.__class__.__name__, self.str_short())

  _bool = _nonzero = concretization_function_error(bool)
  _float   = concretization_function_error(float)
  _int     = concretization_function_error(int)
  if six.PY2:
    _long    = concretization_function_error(long)
  _complex = concretization_function_error(complex)
  _hex     = concretization_function_error(hex)
  _oct     = concretization_function_error(oct)

  def at_least_vspace(self):
    return self

  def join(self, other):
    return self

  def str_short(self):
    return onp.dtype(self.dtype).name


class ShapedArray(UnshapedArray):
  __slots__ = ['shape']
  array_abstraction_level = 2

  def __init__(self, shape, dtype):
    self.dtype = onp.dtype(xla_bridge.canonicalize_dtype(dtype))
    self.shape = shape

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: prod(self.shape))

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape)

  def __hash__(self):
    return hash((self.shape, str(self.dtype)))

  def at_least_vspace(self):
    return self

  def join(self, other):
    if self.shape == other.shape and self.dtype == other.dtype:
      return self
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype)
    else:
      raise TypeError(other)

  def str_short(self):
    dtypestr = onp.dtype(self.dtype).name
    shapestr = ','.join(map(str, self.shape))
    return '{}[{}]'.format(dtypestr, shapestr)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError:
      raise TypeError("len() of unsized object")  # same as numpy error


class ConcreteArray(ShapedArray):
  __slots__ = ['val']
  array_abstraction_level = 0

  def __init__(self, val):
    self.val = val
    self.shape = onp.shape(val)
    # canonicalized self.dtype doesn't necessarily match self.val
    self.dtype = onp.dtype(xla_bridge.canonicalize_dtype(onp.result_type(val)))
    assert self.dtype != onp.dtype('O')

  def __eq__(self, other):
    return (type(self) is type(other) and self.dtype == other.dtype
            and self.shape == other.shape and onp.all(self.val == other.val))

  def __hash__(self):
    return id(self.val)

  def at_least_vspace(self):
    return ShapedArray(self.shape, self.dtype)

  def join(self, other):
    if self == other:
      return self
    elif self.shape == other.shape and self.dtype == other.dtype:
      return ShapedArray(self.shape, self.dtype)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype)
    else:
      raise TypeError(other)

  def str_short(self):
    return str(self.val)


def make_shaped_array(x):
  dtype = xla_bridge.canonicalize_dtype(onp.result_type(x))
  return ShapedArray(onp.shape(x), dtype)

array_types = [onp.ndarray, onp.float64, onp.float32, onp.complex64,
               onp.int64, onp.int32, onp.bool_, onp.uint64, onp.uint32, float,
               int, bool]

for t in array_types:
  core.pytype_aval_mappings[t] = ConcreteArray
