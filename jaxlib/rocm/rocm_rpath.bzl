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
load("@rules_cc//cc:defs.bzl", "cc_binary")

_ROCM_LINK_ONLY = "@local_config_rocm//rocm:link_only"

# $ORIGIN-to-site-packages path of each wheel's .so files: the kernel modules
# sit in jax_rocm<v>_plugin/, the PJRT plugin in jax_plugins/xla_rocm<v>/.
_KERNEL_ORIGIN_TO_SITE = "$$ORIGIN/.."
_PJRT_ORIGIN_TO_SITE = "$$ORIGIN/../.."

def _rocm_wheel_rpaths(origin_to_site):
    """Builds the wheel-relative ROCm RUNPATH entries baked into the .so files.

    Covers the TheRock multi-arch pip layout (the `_rocm_sdk_*` packages in
    the same site-packages as the JAX wheel) and absolute installs where
    TheRock is in the system Python but the .so is not under a site-packages
    root (e.g. the Bazel py_import runfiles layout). The legacy `/opt/rocm`
    fallback is appended by the caller so it stays last.

    Args:
        origin_to_site: Path from the .so's $ORIGIN to its site-packages root.
    """

    # `_rocm_sdk_core/lib` holds the non-arch runtime libs; `_rocm_sdk_libraries
    # /lib` holds the math libs + per-arch kernel data. TheRock's multi-arch
    # packages ship both. The legacy per-family `_rocm_sdk_libraries_<family>`
    # packages are deprecated with no ROCm 10 release, so they are not searched.
    # `_rocm_sdk_core/lib/rocm_sysdeps/lib` holds TheRock's bundled system libs
    # (librocm_sysdeps_{drm,drm_amdgpu,numa}), which the .so files can depend on
    # directly (e.g. via libhsakmt.a).
    core_libs = [
        "_rocm_sdk_core/lib",
        "_rocm_sdk_core/lib/rocm_sysdeps/lib",
        "_rocm_sdk_libraries/lib",
    ]

    py_minors = range(9, 16)

    rpaths = ["-Wl,-rpath,%s/%s" % (origin_to_site, lib) for lib in core_libs]

    # Absolute system-Python fallback (Bazel py_import runfiles / dev-image
    # installs, where the .so isn't under a site-packages root reachable from
    # $ORIGIN but TheRock is pip-installed into the system interpreter).
    rpaths += [
        "-Wl,-rpath,/usr/local/lib/python3.%d/dist-packages/%s" % (m, lib)
        for m in py_minors
        for lib in core_libs
    ]
    return rpaths

# /opt/rocm is a path fallback for the transition period. It requires a
# TheRock-based /opt/rocm containing lib/rocm_sysdeps/lib; classic ROCm
# installations cannot satisfy the plugin's librocm_sysdeps_* dependencies.
# Keep it last so wheel-provided ROCm libraries are preferred.
_OPT_ROCM_RPATHS = [
    "-Wl,-rpath,/opt/rocm/lib",
    "-Wl,-rpath,/opt/rocm/lib/rocm_sysdeps/lib",
]

def _wheel_features():
    return select({
        _ROCM_LINK_ONLY: ["no_solib_rpaths"],
        "//conditions:default": [],
    })

def _wheel_linkopts(origin_to_site):
    return select({
        _ROCM_LINK_ONLY: _rocm_wheel_rpaths(origin_to_site) + _OPT_ROCM_RPATHS,
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
        linkopts = linkopts + if_oss(_wheel_linkopts(_KERNEL_ORIGIN_TO_SITE)),
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
    cc_binary(
        name = name,
        features = features + if_oss(_wheel_features()),
        linkopts = linkopts + if_oss(_wheel_linkopts(_PJRT_ORIGIN_TO_SITE)),
        **kwargs
    )
