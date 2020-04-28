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

import numpy as onp

from . import ad_util
from . import core
from . import dtypes

_DIMENSION_TYPES = core._DIMENSION_TYPES

UnshapedArray = core.UnshapedArray
ShapedArray = core.ShapedArray
ConcreteArray = core.ConcreteArray
AbstractToken = core.AbstractToken
abstract_token = core.abstract_token
canonicalize_shape = core.canonicalize_shape
raise_to_shaped = core.raise_to_shaped


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
