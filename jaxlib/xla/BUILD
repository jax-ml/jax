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

load(
    "//jaxlib:jax.bzl",
    "jax_visibility",
    "pytype_library",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

package_group(
    name = "xla_python",
    includes = [
        "//jax:internal",
    ],
    packages = ["@xla//xla/python/..."],
)

cc_library(
    name = "py_client",
    hdrs = [
        "py_array.h",
        "py_client.h",
        "py_device.h",
        "py_device_list.h",
        "py_executable.h",
        "sharding.h",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = jax_visibility("jaxlib/py_client"),
    deps = [
        "//jaxlib:py_client",
    ],
)

cc_library(
    name = "pytree",
    hdrs = [
        "pytree.h",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = jax_visibility("jaxlib/pytree"),
    deps = [
        "//jaxlib:pytree",
    ],
)

cc_library(
    name = "nb_class_ptr",
    hdrs = [
        "nb_class_ptr.h",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = jax_visibility("jaxlib/nb_class_ptr"),
    deps = [
        "//jaxlib:nb_class_ptr",
    ],
)

cc_library(
    name = "traceback",
    hdrs = [
        "traceback.h",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = jax_visibility("jaxlib/traceback"),
    deps = [
        "//jaxlib:traceback",
    ],
)

cc_library(
    name = "python_ref_manager",
    hdrs = [
        "python_ref_manager.h",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = jax_visibility("jaxlib/python_ref_manager"),
    deps = [
        "//jaxlib:python_ref_manager",
    ],
)

pytype_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    visibility = [":xla_python"],
    deps = [
        ":xla_extension",
        "//jaxlib:xla_client",
    ],
)

pytype_library(
    name = "xla_extension",
    srcs = ["xla_extension.py"],
    visibility = [":xla_python"],
    deps = [
        "//jaxlib:xla_extension",
    ],
)
