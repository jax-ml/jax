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
import importlib
import numpy as np
import operator

import jaxlib.mlir.ir as ir

from .hlo_helpers import custom_call
from .gpu_common_utils import GpuLibNotLinkedError

from jaxlib import xla_client

for cuda_module_name in [".cuda", "jax_cuda12_plugin"]:
  try:
    _cuda_linalg = importlib.import_module(
        f"{cuda_module_name}._linalg", package="jaxlib"
    )
  except ImportError:
    _cuda_linalg = None
  else:
    break

if _cuda_linalg:
  for _name, _value in _cuda_linalg.registrations().items():
    api_version = (1
                   if _name.endswith("lu_pivots_to_permutation")
                   or _name.endswith("_ffi") else 0)
    xla_client.register_custom_call_target(
        _name, _value, platform="CUDA", api_version=api_version
    )

for rocm_module_name in [".rocm", "jax_rocm60_plugin"]:
  try:
    _hip_linalg = importlib.import_module(
        f"{rocm_module_name}._linalg", package="jaxlib"
    )
  except ImportError:
    _hip_linalg = None
  else:
    break

if _hip_linalg:
  for _name, _value in _hip_linalg.registrations().items():
    api_version = (1
                   if _name.endswith("lu_pivots_to_permutation")
                   or _name.endswith("_ffi") else 0)
    xla_client.register_custom_call_target(
        _name, _value, platform="ROCM", api_version=api_version
    )

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)


def _cholesky_update_hlo(platform, gpu_linalg, r_matrix, w_vector, dtype):
  """Cholesky update."""
  del platform
  r_type = ir.RankedTensorType(r_matrix.type)
  dims = r_type.shape
  assert dims[0] == dims[1]
  n = dims[0]

  if not gpu_linalg:
    raise GpuLibNotLinkedError()

  np_type = np.dtype(dtype)
  opaque = gpu_linalg.build_cholesky_update_descriptor(np_type, n)

  return custom_call(
      "cu_cholesky_update",
      operands = [r_matrix, w_vector],
      result_types=[
          ir.RankedTensorType.get((n, n), r_type.element_type),
          ir.RankedTensorType.get((n,), r_type.element_type),
      ],
      operand_output_aliases={0: 0, 1: 1},
      backend_config=opaque,
  ).results[:1]


cuda_cholesky_update = partial(_cholesky_update_hlo, "cu", _cuda_linalg)
