# Copyright 2021 Google LLC
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

import functools
import operator

import numpy as np

from jaxlib import xla_client

try:
  from . import _cuda_linalg
  for _name, _value in _cuda_linalg.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")
except ImportError:
  pass

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)


def lu_pivots_to_permutation(c, pivots, *, permutation_size):
  """Kernel for the transformation of pivots to permutations on GPU."""
  pivots_shape = c.get_shape(pivots)
  dims = pivots_shape.dimensions()
  dtype = np.dtype(np.int32)

  assert pivots_shape.element_type() == dtype

  batch_size = _prod(dims[:-1])
  pivot_size = dims[-1]

  opaque = _cuda_linalg.cuda_lu_pivots_to_permutation_descriptor(
      batch_size, pivot_size, permutation_size)
  pivots_layout = tuple(range(len(dims) - 1, -1, -1))
  pivots_shape_with_layout = xla_client.Shape.array_shape(
      dtype, dims, pivots_layout)

  permutations_layout = pivots_layout
  permutations_dims = list(dims)
  permutations_dims[-1] = permutation_size
  permutations_shape_with_layout = xla_client.Shape.array_shape(
      dtype, permutations_dims, permutations_layout)

  return xla_client.ops.CustomCallWithLayout(
      c,
      b"cuda_lu_pivots_to_permutation",
      operands=(pivots,),
      shape_with_layout=permutations_shape_with_layout,
      operand_shapes_with_layout=(pivots_shape_with_layout,),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING)
