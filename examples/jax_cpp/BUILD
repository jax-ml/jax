# Copyright 2021 Google LLC
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
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
)

licenses(["notice"])

tf_cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:literal_util",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:status",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:cpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
        "@org_tensorflow//tensorflow/core/platform:logging",
        "@org_tensorflow//tensorflow/core/platform:platform_port",
    ],
)
