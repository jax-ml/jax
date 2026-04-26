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

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "mha_fwd_so",
    shared_library = "@aiter_mha_wheel//:libs/libmha_fwd.so",
    deps = ["@aiter_mha_wheel//:aiter_headers"],
)

cc_import(
    name = "mha_bwd_so",
    shared_library = "@aiter_mha_wheel//:libs/libmha_bwd.so",
    deps = ["@aiter_mha_wheel//:aiter_headers"],
)

cc_library(
    name = "aiter_mha",
    deps = [
        ":mha_fwd_so",
        ":mha_bwd_so",
    ],
)

filegroup(
    name = "aiter_runtime_sos",
    srcs = [
        "@aiter_mha_wheel//:libs/libmha_fwd.so",
        "@aiter_mha_wheel//:libs/libmha_bwd.so",
    ],
)