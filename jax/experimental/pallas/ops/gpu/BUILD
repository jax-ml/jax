# Copyright 2024 The JAX Authors.
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

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:__subpackages__"],
)

exports_files(
    srcs = glob(["*.py"]),
)

filegroup(
    name = "triton_ops",
    srcs = glob(
        ["*.py"],
        exclude = ["*_mgpu.py"],
    ),
)

filegroup(
    name = "mgpu_ops",
    srcs = glob(["*_mgpu.py"]),
)
