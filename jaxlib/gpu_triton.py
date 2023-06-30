# Copyright 2023 The JAX Authors.
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

try:
  from .cuda import _triton  # pytype: disable=import-error
  xla_client.register_custom_call_target(
      "triton_kernel_call", _triton.get_custom_call(),
      platform='CUDA')
  TritonKernelCall = _triton.TritonKernelCall
  TritonAutotunedKernelCall = _triton.TritonAutotunedKernelCall
  TritonKernel = _triton.TritonKernel
  create_array_parameter = _triton.create_array_parameter
  create_scalar_parameter = _triton.create_scalar_parameter
  get_compute_capability = _triton.get_compute_capability
  get_custom_call = _triton.get_custom_call
  get_serialized_metadata = _triton.get_serialized_metadata
except ImportError:
  _triton = None
