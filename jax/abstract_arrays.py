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

from functools import partial

import numpy as np

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
  return ShapedArray(np.shape(x), dtype)

def zeros_like_array(x):
  dtype = dtypes.canonicalize_dtype(dtypes.result_type(x))
  return zeros_like_shaped_array(ShapedArray(np.shape(x), dtype))

array_types = {np.ndarray, np.bool_,
               np.int8, np.int16, np.int32, np.int64,
               np.uint8, np.uint16, np.uint32, np.uint64,
               dtypes.bfloat16, np.float16, np.float32, np.float64,
               np.complex64, np.complex128,
               np.longlong}

for t in array_types:
  core.pytype_aval_mappings[t] = ConcreteArray
  ad_util.jaxval_zeros_likers[t] = zeros_like_array


def zeros_like_shaped_array(aval):
  assert isinstance(aval, ShapedArray)
  return np.broadcast_to(np.array(0, aval.dtype), aval.shape)

ad_util.aval_zeros_likers[ShapedArray] = zeros_like_shaped_array

core.literalable_types.update(array_types)

def _zeros_like_python_scalar(t, x):
  return np.array(0, dtypes.python_scalar_dtypes[t])

def _make_concrete_python_scalar(t, x):
  return ConcreteArray(
    np.array(x, dtype=dtypes.python_scalar_dtypes[t]),
    weak_type=True)

for t in dtypes.python_scalar_dtypes.keys():
  core.pytype_aval_mappings[t] = partial(_make_concrete_python_scalar, t)
  ad_util.jaxval_zeros_likers[t] = partial(_zeros_like_python_scalar, t)

core.literalable_types.update(dtypes.python_scalar_dtypes.keys())
