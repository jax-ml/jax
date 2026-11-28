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

"""BUILD file for LAPACK."""

load("@rules_fortran//:defs.bzl", "fortran_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD-3-Clause

exports_files(["LICENSE"])

# Files that require compilation without optimization
NO_OPT_SRCS = [
    "INSTALL/dlamch.f",
    "INSTALL/slamch.f",
    "SRC/cla_wwaddw.f",
    "SRC/dla_wwaddw.f",
    "SRC/dlaruv.f",
    "SRC/sla_wwaddw.f",
    "SRC/slaruv.f",
    "SRC/zla_wwaddw.f",
]

# Module: la_constants (provides la_constants.mod)
fortran_library(
    name = "la_constants",
    srcs = ["SRC/la_constants.f90"],
)

# Module: la_xisnan (provides la_xisnan.mod, depends on la_constants.mod)
fortran_library(
    name = "la_xisnan",
    srcs = ["SRC/la_xisnan.F90"],
    deps = [":la_constants"],
)

# Modules target - ALLMOD = la_xisnan.mod la_constants.mod
# All other Fortran files depend on these modules
fortran_library(
    name = "modules",
    deps = [
        ":la_constants",
        ":la_xisnan",
    ],
)

fortran_library(
    name = "no_opt",
    srcs = NO_OPT_SRCS,
    copts = ["-O0"],
    deps = [
        ":modules",
        "@eigen_archive//:blas",
    ],
)

fortran_library(
    name = "lapack",
    srcs = glob(
        [
            "SRC/*.f",
            "SRC/*.F",
            "SRC/*.f90",
        ],
        exclude = NO_OPT_SRCS + [
            # Fortran modules compiled separately in :la_constants and :la_xisnan
            "SRC/la_constants.f90",
            "SRC/la_xisnan.F90",
            # Provided by @eigen_archive//:blas
            "SRC/xerbla.f",
            "SRC/xerbla_array.f",
            # Extended precision (XBLAS) routines requiring external libxblas
            "SRC/?la_*.f",
            "SRC/*rfsx.f",
            "SRC/*svxx.f",
            "SRC/*lascl2.f",
            "SRC/*larscl2.f",
        ],
    ) + [
        "INSTALL/droundup_lwork.f",
        "INSTALL/dsecnd_INT_CPU_TIME.f",
        "INSTALL/ilaver.f",
        "INSTALL/second_INT_CPU_TIME.f",
        "INSTALL/sroundup_lwork.f",
    ],
    deps = [
        ":modules",
        ":no_opt",
        "@eigen_archive//:blas",
    ],
)
