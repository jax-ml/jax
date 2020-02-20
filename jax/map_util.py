# Copyright 2020 Google LLC
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


from .core import abstract_unit
from .abstract_arrays import ShapedArray

import numpy as onp

# TODO(mattjj): use a special sentinel type rather than None
NotMapped = type(None)
not_mapped = None

def shard_aval(aval, batch_dim):
  """Remove a batch dimension"""
  if batch_dim is not_mapped or aval is abstract_unit:
    return aval
  elif type(aval) is ShapedArray:
    assert 0 <= batch_dim < aval.ndim
    new_shape = tuple(onp.delete(aval.shape, batch_dim))
    return ShapedArray(new_shape, aval.dtype)
  else:
    raise TypeError(aval)

def map_aval(size, aval, batch_dim):
  """Add a batch dimension"""
  if batch_dim is not_mapped or aval is abstract_unit:
    return aval
  elif type(aval) is ShapedArray:
    assert 0 <= batch_dim < aval.ndim + 1
    new_shape = tuple(onp.insert(aval.shape, batch_dim, size))
    return ShapedArray(new_shape, aval.dtype)
  else:
    raise TypeError(aval)
