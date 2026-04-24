# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//third_party/nccl_device_bitcode:build_defs.bzl", "nccl_device_bitcode")

nccl_device_bitcode(
    name = "libnccl_device",
    srcs = ["@nccl_source//:ir_wrapper_srcs"],
    nccl_hdrs = ["@nccl_source//:nccl_hdrs"],
    visibility = ["//visibility:public"],
)
