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

from typing import Any

import numpy as np

from jax._src import dtypes
from jax._src.typing import DTypeLike


class WeakTypeNdArray(np.ndarray):
  def __new__(subtype, shape, dtype=float, buffer=None, offset=0, strides=None,
              order=None, weak_type=True):
    obj = super().__new__(subtype, shape, dtype, buffer, offset, strides, order)
    obj.weak_type = weak_type
    return obj

  def __array_finalize__(self, obj):
    if obj is None:
      return
    self.weak_type = getattr(obj, "weak_type", True)


def as_weak_type_ndarray(
    x: Any, dtype: DTypeLike | None = None, weak_type: bool = True
) -> WeakTypeNdArray:
  dtype_ = dtypes.dtype(x) if dtype is None else dtypes.dtype(dtype)
  arr = np.asarray(x, dtype=dtype_)
  arr = arr.view(WeakTypeNdArray)
  arr.weak_type = weak_type
  return arr
