# Copyright 2021 The JAX Authors.
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
from functools import partial
import operator

import jaxlib.mlir.ir as ir

from .hlo_helpers import custom_call

from jaxlib import xla_client

try:
  from .cuda import _linalg as _cuda_linalg  # pytype: disable=import-error
  for _name, _value in _cuda_linalg.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")
except ImportError:
  _cuda_linalg = None

try:
  from .rocm import _linalg as _hip_linalg  # pytype: disable=import-error
  for _name, _value in _hip_linalg.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")
except ImportError:
  _hip_linalg = None

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)


def _lu_pivots_to_permutation_hlo(platform, gpu_linalg, pivots, *, permutation_size):
  """Kernel for the transformation of pivots to permutations on GPU."""
  typ = ir.RankedTensorType(pivots.type)
  dims = typ.shape
  i32_type = ir.IntegerType.get_signless(32)

  assert typ.element_type == i32_type, typ

  batch_size = _prod(dims[:-1])
  pivot_size = dims[-1]

  opaque = gpu_linalg.lu_pivots_to_permutation_descriptor(
      batch_size, pivot_size, permutation_size)
  pivots_layout = tuple(range(len(dims) - 1, -1, -1))
  permutations_layout = pivots_layout
  permutations_dims = list(dims)
  permutations_dims[-1] = permutation_size
  permutations_type = ir.RankedTensorType.get(permutations_dims, i32_type)
  return custom_call(
      f"{platform}_lu_pivots_to_permutation",
      [permutations_type],
      [pivots],
      backend_config=opaque,
      operand_layouts=[pivots_layout],
      result_layouts=[permutations_layout])

cuda_lu_pivots_to_permutation = partial(_lu_pivots_to_permutation_hlo, "cu",
                                        _cuda_linalg)
hip_lu_pivots_to_permutation = partial(
    _lu_pivots_to_permutation_hlo, "hip", _hip_linalg)
