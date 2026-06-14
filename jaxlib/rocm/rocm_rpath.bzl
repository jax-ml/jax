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

"""Wrapper macros for ROCm wheel RPATH configuration.

When building wheels with rocm_path_type=link_only, Bazel sandbox rpaths
are stripped and replaced with wheel-relative paths so that .so files
find ROCm libraries at install time.

Usage in BUILD files:

    load("//jaxlib/rocm:rpath.bzl", "rocm_nanobind_extension")

    rocm_nanobind_extension(
        name = "_my_ext",
        srcs = ["my_ext.cc"],
        ...
    )
"""

load("//jaxlib:jax.bzl", "if_oss", "nanobind_extension")

_ROCM_LINK_ONLY = "@local_config_rocm//rocm:link_only"

_WHEEL_RPATHS = [
    "-Wl,-rpath,$$ORIGIN/../rocm/lib",
    "-Wl,-rpath,$$ORIGIN/../../rocm/lib",
    "-Wl,-rpath,/opt/rocm/lib",
]

def _wheel_features():
    return select({
        _ROCM_LINK_ONLY: ["no_solib_rpaths"],
        "//conditions:default": [],
    })

def _wheel_linkopts():
    return select({
        _ROCM_LINK_ONLY: _WHEEL_RPATHS,
        "//conditions:default": [],
    })

def rocm_nanobind_extension(name, features = [], linkopts = [], **kwargs):
    """nanobind_extension that automatically strips solib rpaths and embeds wheel RPATHs.

    When built with --@local_config_rocm//rocm:rocm_path_type=link_only,
    the no_solib_rpaths feature is enabled and wheel-specific RPATHs are
    added. Otherwise the target behaves identically to nanobind_extension.

    Args:
        name: Target name.
        features: Additional features (rpath features are appended automatically).
        linkopts: Additional linkopts (wheel RPATHs are appended automatically).
        **kwargs: Passed through to nanobind_extension.
    """
    nanobind_extension(
        name = name,
        features = features + if_oss(_wheel_features()),
        linkopts = linkopts + if_oss(_wheel_linkopts()),
        **kwargs
    )

def rocm_cc_binary(name, features = [], linkopts = [], **kwargs):
    """cc_binary that automatically strips solib rpaths and embeds wheel RPATHs.

    Args:
        name: Target name.
        features: Additional features (rpath features are appended automatically).
        linkopts: Additional linkopts (wheel RPATHs are appended automatically).
        **kwargs: Passed through to native.cc_binary.
    """
    native.cc_binary(
        name = name,
        features = features + if_oss(_wheel_features()),
        linkopts = linkopts + if_oss(_wheel_linkopts()),
        **kwargs
    )
