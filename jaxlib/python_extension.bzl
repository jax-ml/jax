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

load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@symbol_locations//rules_pywrap:pywrap.bzl", "pybind_extension")

def pywrap_extension(
        name,
        module_name = None,
        srcs = [],
        deps = [],
        pytype_srcs = [],
        pytype_deps = [],
        copts = [],
        linkopts = [],
        visibility = None):
    module_name = name if module_name == None else module_name
    lib_name = name + "_pywrap_library"
    src_cc_name = name + "_pywrap_stub.c"
    native.cc_library(
        name = lib_name,
        srcs = srcs,
        copts = copts,
        deps = deps,
        local_defines = [
            "PyInit_{}=Wrapped_PyInit_{}".format(module_name, module_name),
        ],
        visibility = ["//visibility:private"],
    )
    expand_template(
        name = name + "_pywrap_stub",
        testonly = True,
        out = src_cc_name,
        substitutions = {
            "@MODULE_NAME@": module_name,
        },
        template = "//jaxlib:pyinit_stub.c",
        visibility = ["//visibility:private"],
    )
    pybind_extension(
        name = name,
        # module_name = module_name,
        srcs = [src_cc_name],
        deps = [":" + lib_name],
        linkopts = linkopts,
        visibility = visibility,
        common_lib_packages = [
            "jaxlib",
        ],
    )
