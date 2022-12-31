# Copyright 2018 The JAX Authors.
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

import dataclasses
import warnings
from typing import Callable, Sequence

import numpy as np

from jax._src.tree_util import tree_flatten, tree_unflatten, PyTreeDef
from jax._src.util import safe_zip, unzip2, HashableArrayWrapper
from jax._src.typing import DType, Shape

import jax.numpy as jnp
from jax._src import dtypes
from jax import lax

zip = safe_zip


@dataclasses.dataclass(frozen=True)
class UnravelPyTree:
  unravel_list: Callable
  treedef: PyTreeDef

  def __call__(self, flat):
    return tree_unflatten(self.treedef, self.unravel_list(flat))

  def __hash__(self):
    return hash((self.unravel_list, self.treedef))

  def __eq__(self, other):
    if not isinstance(other, UnravelPyTree):
      return False
    return self.unravel_list == other.unravel_list and self.treedef == other.treedef



def ravel_pytree(pytree):
  """Ravel (flatten) a pytree of arrays down to a 1D array.

  Args:
    pytree: a pytree of arrays and scalars to ravel.

  Returns:
    A pair where the first element is a 1D array representing the flattened and
    concatenated leaf values, with dtype determined by promoting the dtypes of
    leaf values, and the second element is a callable for unflattening a 1D
    vector of the same length back to a pytree of of the same structure as the
    input ``pytree``. If the input pytree is empty (i.e. has no leaves) then as
    a convention a 1D empty array of dtype float32 is returned in the first
    component of the output.

  For details on dtype promotion, see
  https://jax.readthedocs.io/en/latest/type_promotion.html.

  """
  leaves, treedef = tree_flatten(pytree)
  flat, unravel_list = _ravel_list(leaves)
  unravel_pytree = UnravelPyTree(unravel_list, treedef)
  return flat, unravel_pytree

@dataclasses.dataclass(frozen=True)
class UnravelListWithSameDtype:
  shapes: Sequence[Shape]
  indices: np.ndarray

  def __call__(self, arr):
    chunks = jnp.split(arr, self.indices[:-1])
    return [chunk.reshape(shape) for chunk, shape in zip(chunks, self.shapes)]

  def __hash__(self):
    return hash((self.shapes, HashableArrayWrapper(self.indices)))

  def __eq__(self, other):
    if not isinstance(other, UnravelListWithSameDtype):
      return False
    return self.shapes == other.shapes and np.all(self.indices == other.indices)


# When there is more than one distinct input dtype, we perform type
# conversions and produce a dtype-specific unravel function.
@dataclasses.dataclass(frozen=True)
class UnravelListWithDifferentDtypes:
  from_dtypes: Sequence[DType]
  to_dtype: DType
  shapes: Sequence[Shape]
  indices: np.ndarray

  def __call__(self, arr):
    arr_dtype = dtypes.dtype(arr)
    if arr_dtype != self.to_dtype:
      raise TypeError(f"unravel function given array of dtype {arr_dtype}, "
                      f"but expected dtype {self.to_dtype}")
    chunks = jnp.split(arr, self.indices[:-1])
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
      return [lax.convert_element_type(chunk.reshape(shape), dtype)
              for chunk, shape, dtype in zip(chunks, self.shapes,
                                             self.from_dtypes)]

  def __hash__(self):
    return hash((self.from_dtypes, self.to_dtype, self.shapes,
                 HashableArrayWrapper(self.indices)))

  def __eq__(self, other):
    if not isinstance(other, UnravelListWithDifferentDtypes):
      return False
    return (self.from_dtypes == other.from_dtypes
            and self.to_dtype == other.to_dtype
            and self.shapes == other.shapes
            and np.all(self.indices == other.indices))

def _ravel_list(lst):
  if not lst: return jnp.array([], jnp.float32), lambda _: []
  from_dtypes = [dtypes.dtype(l) for l in lst]
  to_dtype = dtypes.result_type(*from_dtypes)
  sizes, shapes = unzip2((jnp.size(x), jnp.shape(x)) for x in lst)
  indices = np.cumsum(sizes)

  if all(dt == to_dtype for dt in from_dtypes):
    # Skip any dtype conversion, resulting in a dtype-polymorphic `unravel`.
    # See https://github.com/google/jax/issues/7809.
    del from_dtypes, to_dtype
    raveled = jnp.concatenate([jnp.ravel(e) for e in lst])
    unravel = UnravelListWithSameDtype(shapes, indices)
    return raveled, unravel

  ravel = lambda e: jnp.ravel(lax.convert_element_type(e, to_dtype))
  raveled = jnp.concatenate([ravel(e) for e in lst])
  unravel = UnravelListWithDifferentDtypes(from_dtypes, to_dtype, shapes, indices)
  return raveled, unravel
