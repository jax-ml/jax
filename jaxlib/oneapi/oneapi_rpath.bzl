# Copyright 2026 The JAX Authors.
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

"""Wrapper macros for OneAPI wheel RPATH configuration."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//jaxlib:jax.bzl", "nanobind_extension")

# OneAPI runtime libraries are installed in the interpreter's lib dir.
# Include paths for both native artifact depths
_ONEAPI_WHEEL_RPATHS = [
    "-Wl,-rpath,$$ORIGIN/../../..",
    "-Wl,-rpath,$$ORIGIN/../../../..",
]

# Keep fallback locations last so wheel-local runtime libraries take priority.
_ONEAPI_FALLBACK_RPATHS = []

def oneapi_nanobind_extension(name, features = [], linkopts = [], **kwargs):
    nanobind_extension(
        name = name,
        features = features,
        linkopts = linkopts + _ONEAPI_WHEEL_RPATHS + _ONEAPI_FALLBACK_RPATHS,
        **kwargs
    )

def oneapi_cc_binary(name, features = [], linkopts = [], **kwargs):
    cc_binary(
        name = name,
        features = features,
        linkopts = linkopts + _ONEAPI_WHEEL_RPATHS + _ONEAPI_FALLBACK_RPATHS,
        **kwargs
    )
