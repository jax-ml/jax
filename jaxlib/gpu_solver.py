# Copyright 2019 The JAX Authors.
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

_cublas = import_from_plugin("cuda", "_blas")
_cusolver = import_from_plugin("cuda", "_solver")
_cuhybrid = import_from_plugin("cuda", "_hybrid")

_hipblas = import_from_plugin("rocm", "_blas")
_hipsolver = import_from_plugin("rocm", "_solver")
_hiphybrid = import_from_plugin("rocm", "_hybrid")

if _cublas:
  for _name, _value in _cublas.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")

if _cusolver:
  for _name, _value in _cusolver.registrations().items():
    # TODO(danfm): Clean up after all legacy custom calls are ported.
    api_version = 0
    if _name.endswith("_ffi"):
      api_version = 1
      xla_client.register_custom_call_as_batch_partitionable(_name)
    xla_client.register_custom_call_target(_name, _value, platform="CUDA",
                                           api_version=api_version)

if _cuhybrid:
  for _name, _value in _cuhybrid.registrations().items():
    xla_client.register_custom_call_as_batch_partitionable(_name)
    xla_client.register_custom_call_target(_name, _value, platform="CUDA",
                                           api_version=1)

if _hipblas:
  for _name, _value in _hipblas.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")

if _hipsolver:
  for _name, _value in _hipsolver.registrations().items():
    # TODO(danfm): Clean up after all legacy custom calls are ported.
    api_version = 0
    if _name.endswith("_ffi"):
      api_version = 1
      xla_client.register_custom_call_as_batch_partitionable(_name)
    xla_client.register_custom_call_target(_name, _value, platform="ROCM",
                                           api_version=api_version)

if _hiphybrid:
  for _name, _value in _hiphybrid.registrations().items():
    xla_client.register_custom_call_as_batch_partitionable(_name)
    xla_client.register_custom_call_target(_name, _value, platform="ROCM",
                                           api_version=1)

def initialize_hybrid_kernels():
  if _cuhybrid:
    _cuhybrid.initialize()
  if _hiphybrid:
    _hiphybrid.initialize()

def has_magma():
  if _cuhybrid:
    return _cuhybrid.has_magma()
  if _hiphybrid:
    return _hiphybrid.has_magma()
  return False
