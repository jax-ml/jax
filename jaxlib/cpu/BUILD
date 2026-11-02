# Copyright 2018 The JAX Authors.
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

# JAX is Autograd and XLA

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//jaxlib:jax.bzl",
    "jax_visibility",
    "nanobind_extension",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

# LAPACK

cc_library(
    name = "lapack_kernels",
    srcs = ["lapack_kernels.cc"],
    hdrs = ["lapack_kernels.h"],
    # compatible with libtpu
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = [
        "//jax:internal",
    ] + jax_visibility("jaxlib/cpu/lapack_kernels"),
    deps = [
        ":thread_pool",
        "//jaxlib:ffi_helpers",
        "//jaxlib:tridiagonal_solve_perturbed",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
    ],
)

cc_library(
    name = "lapack_kernels_using_lapack",
    srcs = ["lapack_kernels_using_lapack.cc"],
    # compatible with libtpu
    deps = [":lapack_kernels"],
    alwayslink = 1,
)

cc_library(
    name = "tridiagonal_solve_kernels",
    srcs = ["tridiagonal_solve_kernels.cc"],
    hdrs = ["tridiagonal_solve_kernels.h"],
    # compatible with libtpu
    deps = [
        "//jaxlib:ffi_helpers",
        "//jaxlib:tridiagonal_solve_perturbed",
        "@xla//xla/ffi/api:ffi",
    ],
)

nanobind_extension(
    name = "_lapack",
    srcs = ["lapack.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    enable_stub_generation = False,  # generated stubs are incorrect
    features = ["-use_header_modules"],
    module_name = "_lapack",
    pytype_srcs = [
        "_lapack/__init__.pyi",
        "_lapack/schur.pyi",
        "_lapack/svd.pyi",
        "_lapack/eig.pyi",
    ],
    deps = [
        ":lapack_kernels",
        ":tridiagonal_solve_kernels",
        "//jaxlib:kernel_nanobind_helpers",
        "@nanobind",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/python:safe_static_init",
    ],
)

cc_library(
    name = "cpu_kernels",
    srcs = ["cpu_kernels.cc"],
    # compatible with libtpu
    visibility = ["//visibility:public"],
    deps = [
        ":digamma_kernels",
        ":lapack_kernels",
        ":lapack_kernels_using_lapack",
        ":lgamma_kernels",
        ":sparse_kernels",
        ":tridiagonal_solve_kernels",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
    ],
    alwayslink = 1,
)

cc_library(
    name = "thread_pool",
    hdrs = ["thread_pool.h"],
    # compatible with libtpu
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
        "@xla//xla/backends/cpu/runtime:work_queue",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/tsl/concurrency:async_value",
        "@xla//xla/tsl/lib/math:math_util",
    ],
)

cc_library(
    name = "lgamma_kernels",
    srcs = ["lgamma_kernels.cc"],
    hdrs = ["lgamma_kernels.h"],
    # compatible with libtpu
    # Highway's `foreach_target.h` re-includes `lgamma_kernels.cc` and the
    # toggle-guarded helper headers once per SIMD target (`HWY_AVX3`, `HWY_AVX2`,
    # etc.), so they cannot be compiled as standalone header modules.
    textual_hdrs = [
        "double_word.h",
        "lgamma_kernels.cc",
        "log.h",
    ],
    deps = [
        ":thread_pool",
        "//jaxlib:lgamma_coeffs",
        "//jaxlib:log_coeffs",
        "@highway//:hwy",
        "@xla//xla/ffi/api:ffi",
    ],
)

cc_library(
    name = "digamma_kernels",
    srcs = ["digamma_kernels.cc"],
    hdrs = ["digamma_kernels.h"],
    # compatible with libtpu
    # See comment on `lgamma_kernels` above.
    textual_hdrs = [
        "digamma_kernels.cc",
        "double_word.h",
        "log.h",
    ],
    deps = [
        ":thread_pool",
        "//jaxlib:digamma_coeffs",
        "//jaxlib:log_coeffs",
        "@highway//:hwy",
        "@xla//xla/ffi/api:ffi",
    ],
)

nanobind_extension(
    name = "_math",
    srcs = ["math.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    enable_stub_generation = False,
    features = ["-use_header_modules"],
    module_name = "_math",
    pytype_srcs = [
        "_math/__init__.pyi",
    ],
    deps = [
        ":digamma_kernels",
        ":lgamma_kernels",
        "//jaxlib:kernel_nanobind_helpers",
        "@nanobind",
        "@xla//xla/ffi/api:ffi",
    ],
)

cc_library(
    name = "sparse_kernels",
    srcs = ["sparse_kernels.cc"],
    hdrs = ["sparse_kernels.h"],
    # compatible with libtpu
    deps = [
        "@eigen_archive//:eigen3",
        "@xla//xla/ffi/api:ffi",
    ],
)

nanobind_extension(
    name = "_sparse",
    srcs = ["sparse.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    enable_stub_generation = False,
    features = ["-use_header_modules"],
    module_name = "_sparse",
    pytype_srcs = [
        "_sparse/__init__.pyi",
    ],
    deps = [
        ":sparse_kernels",
        "//jaxlib:kernel_nanobind_helpers",
        "@com_google_absl//absl/base",
        "@nanobind",
        "@xla//xla/ffi/api:ffi",
    ],
)
