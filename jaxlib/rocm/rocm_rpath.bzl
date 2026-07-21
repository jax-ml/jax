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

# TheRock per-GPU-family pip package suffixes (per-GPU-family index at
# https://rocm.nightlies.amd.com/v2/<family>/). End users install per-arch, so
# the resulting `_rocm_sdk_libraries_<family>` package dirs must be searchable.
# The multi-arch index instead ships a single `_rocm_sdk_libraries` package.
_THEROCK_TARGET_FAMILIES = [
    "gfx950-dcgpu",
    "gfx94X-dcgpu",
    "gfx90a",
    "gfx90X-dcgpu",
    "gfx908",
    "gfx906",
    "gfx900",
    "gfx120X-all",
    "gfx1153",
    "gfx1152",
    "gfx1151",
    "gfx1150",
    "gfx110X-all",
    "gfx103X-all",
    "gfx101X-dgpu",
]

def _rocm_wheel_rpaths():
    """Builds the wheel-relative ROCm RUNPATH entries baked into the .so files.

    Covers the TheRock tarball/`/opt/rocm-<ver>` layout (`rocm/lib`), the
    TheRock pip layouts (multi-arch and per-GPU-family `_rocm_sdk_*`), and
    cross-Python / absolute installs where TheRock lives under a different
    interpreter than the JAX wheel (e.g. the Bazel py_import runfiles layout
    where TheRock is in the container's system Python). The legacy `/opt/rocm`
    fallback is appended by the caller so it stays last.
    """

    # `_rocm_sdk_core/lib` holds the non-arch runtime libs; `_rocm_sdk_libraries
    # [...]/lib` holds the math libs + per-arch kernel data. site_libs covers the
    # multi-arch + per-family packages; core_libs is the cross-Python/absolute
    # fallback (package roots only).
    core_libs = ["_rocm_sdk_core/lib", "_rocm_sdk_libraries/lib"]
    site_libs = core_libs + [
        "_rocm_sdk_libraries_%s/lib" % f.replace("-", "_")
        for f in _THEROCK_TARGET_FAMILIES
    ]

    # $ORIGIN-to-site-packages depth differs per wheel (kernel .so is 1-deep in
    # jax_rocm<v>_plugin/, pjrt .so is 2-deep in jax_plugins/xla_rocm<v>/).
    # linkopts apply to every target uniformly, so emit both; the loader skips
    # RUNPATH entries whose directories don't exist.
    origin_to_site = ["$$ORIGIN/..", "$$ORIGIN/../.."]
    py_minors = range(9, 16)

    rpaths = [
        # TheRock tarball / extracted /opt/rocm-<ver> layout.
        "-Wl,-rpath,$$ORIGIN/../rocm/lib",
        "-Wl,-rpath,$$ORIGIN/../../rocm/lib",
    ]

    # Pip layouts at each possible $ORIGIN-to-site depth: same-Python (TheRock
    # packages under the site-packages root next to the JAX wheel) and
    # cross-Python (TheRock under a different python3.<m>/dist-packages under
    # the same root).
    for ots in origin_to_site:
        rpaths += ["-Wl,-rpath,%s/%s" % (ots, lib) for lib in site_libs]
        rpaths += [
            "-Wl,-rpath,%s/../../python3.%d/dist-packages/%s" % (ots, m, lib)
            for m in py_minors
            for lib in core_libs
        ]

    # Absolute system-Python fallback (Bazel py_import runfiles / dev-image
    # installs, where the .so isn't under a site-packages root reachable from
    # $ORIGIN but TheRock is pip-installed into the system interpreter).
    rpaths += [
        "-Wl,-rpath,/usr/local/lib/python3.%d/dist-packages/%s" % (m, lib)
        for m in py_minors
        for lib in core_libs
    ]
    return rpaths

# /opt/rocm is the legacy ROCm fallback for the transition period. Keep it last
# so the wheel's own ROCm paths are always preferred over a system install.
_WHEEL_RPATHS = _rocm_wheel_rpaths() + ["-Wl,-rpath,/opt/rocm/lib"]

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
