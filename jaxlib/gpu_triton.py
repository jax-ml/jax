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

from .plugin_support import import_from_plugin

_cuda_triton = import_from_plugin("cuda", "_triton")
_hip_triton = import_from_plugin("rocm", "_triton")

if _cuda_triton:
  xla_client.register_custom_call_target(
      "triton_kernel_call", _cuda_triton.get_custom_call(),
      platform='CUDA')
  TritonKernelCall = _cuda_triton.TritonKernelCall
  TritonAutotunedKernelCall = _cuda_triton.TritonAutotunedKernelCall
  TritonKernel = _cuda_triton.TritonKernel
  create_array_parameter = _cuda_triton.create_array_parameter
  create_scalar_parameter = _cuda_triton.create_scalar_parameter
  get_compute_capability = _cuda_triton.get_compute_capability
  get_arch_details = _cuda_triton.get_arch_details
  get_custom_call = _cuda_triton.get_custom_call
  get_serialized_metadata = _cuda_triton.get_serialized_metadata

if _hip_triton:
  xla_client.register_custom_call_target(
      "triton_kernel_call", _hip_triton.get_custom_call(),
      platform='ROCM')
  TritonKernelCall = _hip_triton.TritonKernelCall
  TritonAutotunedKernelCall = _hip_triton.TritonAutotunedKernelCall
  TritonKernel = _hip_triton.TritonKernel
  create_array_parameter = _hip_triton.create_array_parameter
  create_scalar_parameter = _hip_triton.create_scalar_parameter
  get_compute_capability = _hip_triton.get_compute_capability
  get_arch_details = _hip_triton.get_arch_details
  get_custom_call = _hip_triton.get_custom_call
  get_serialized_metadata = _hip_triton.get_serialized_metadata
