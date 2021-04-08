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

import warnings

import numpy as np

from .tree_util import tree_flatten, tree_unflatten
from ._src.util import safe_zip, unzip2

import jax.numpy as jnp
from jax._src import dtypes
from jax import lax

zip = safe_zip


def ravel_pytree(pytree):
  """Ravel (i.e. flatten) a pytree of arrays down to a 1D array.

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
  unravel_pytree = lambda flat: tree_unflatten(treedef, unravel_list(flat))
  return flat, unravel_pytree

def _ravel_list(lst):
  if not lst: return jnp.array([], jnp.float32), lambda _: []
  from_dtypes = [dtypes.dtype(l) for l in lst]
  to_dtype = dtypes.result_type(*from_dtypes)
  sizes, shapes = unzip2((jnp.size(x), jnp.shape(x)) for x in lst)
  indices = np.cumsum(sizes)

  def unravel(arr):
    chunks = jnp.split(arr, indices[:-1])
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
      return [lax.convert_element_type(chunk.reshape(shape), dtype)
              for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)]

  ravel = lambda e: jnp.ravel(lax.convert_element_type(e, to_dtype))
  raveled = jnp.concatenate([ravel(e) for e in lst])
  return raveled, unravel
