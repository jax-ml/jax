# Copyright 2021 The JAX Authors.
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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_applicable_licenses = [])

licenses(["notice"])

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    tags = ["manual"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/service:hlo_module_config",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tools:hlo_module_loader",
    ],
)
