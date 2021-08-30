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

"""Bazel macros used by the JAX build."""

load("@org_tensorflow//tensorflow/core/platform/default:build_config.bzl", _pyx_library = "pyx_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", _pybind_extension = "pybind_extension")
load("@local_config_cuda//cuda:build_defs.bzl", _cuda_library = "cuda_library", _if_cuda_is_configured = "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", _if_rocm_is_configured = "if_rocm_is_configured")
load("@flatbuffers//:build_defs.bzl", _flatbuffer_cc_library = "flatbuffer_cc_library", _flatbuffer_py_library = "flatbuffer_py_library")

# Explicitly re-exports names to avoid "unused variable" warnings from .bzl
# lint tools.
cuda_library = _cuda_library
pytype_library = native.py_library
pyx_library = _pyx_library
pybind_extension = _pybind_extension
if_cuda_is_configured = _if_cuda_is_configured
if_rocm_is_configured = _if_rocm_is_configured
flatbuffer_cc_library = _flatbuffer_cc_library
flatbuffer_py_library = _flatbuffer_py_library
