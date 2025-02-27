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

from jaxlib import xla_client

from .plugin_support import import_from_plugin

_cuda_linalg = import_from_plugin("cuda", "_linalg")
_hip_linalg = import_from_plugin("rocm", "_linalg")

if _cuda_linalg:
  for _name, _value in _cuda_linalg.registrations().items():
    xla_client.register_custom_call_target(
        _name, _value, platform="CUDA", api_version=1
    )
  xla_client.register_custom_call_as_batch_partitionable(
      "cu_lu_pivots_to_permutation")

if _hip_linalg:
  for _name, _value in _hip_linalg.registrations().items():
    xla_client.register_custom_call_target(
        _name, _value, platform="ROCM", api_version=1
    )
  xla_client.register_custom_call_as_batch_partitionable(
      "hip_lu_pivots_to_permutation")
