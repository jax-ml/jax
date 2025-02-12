# Copyright 2025 The JAX Authors.
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

load(
    "@tsl//third_party/py:py_manylinux_compliance_test.bzl",
    "verify_manylinux_compliance_test",
)
load(
    "//jaxlib:jax.bzl",
    "AARCH64_MANYLINUX_TAG",
    "PPC64LE_MANYLINUX_TAG",
    "X86_64_MANYLINUX_TAG",
    "jax_wheel",
)

py_binary(
    name = "build_wheel",
    srcs = ["build_wheel.py"],
    data = [
        "AUTHORS",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "//jax:jax_sources",
    ],
    deps = [
        "//jaxlib/tools:build_utils",
        "@bazel_tools//tools/python/runfiles",
        "@pypi_build//:pkg",
        "@pypi_setuptools//:pkg",
        "@pypi_wheel//:pkg",
    ],
)

jax_wheel(
    name = "jax_wheel",
    no_abi = True,
    no_platform = True,
    wheel_binary = ":build_wheel",
    wheel_name = "jax",
)

verify_manylinux_compliance_test(
    name = "jax_manylinux_compliance_test",
    aarch64_compliance_tag = AARCH64_MANYLINUX_TAG,
    ppc64le_compliance_tag = PPC64LE_MANYLINUX_TAG,
    test_tags = [
        "manual",
    ],
    wheel = ":jax_wheel",
    x86_64_compliance_tag = X86_64_MANYLINUX_TAG,
)
