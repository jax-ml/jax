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

cc_binary(
    name = "deserialize_and_run",
    srcs = [
        "deserialize_and_run.cc",
    ],
    tags = ["manual"],
    deps = [
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/c:pjrt_c_api_cpu_internal",
    ],
)

py_binary(
    name = "serialize",
    srcs = ["serialize.py"],
    main = "serialize.py",
    python_version = "PY3",
    deps = [
        "//jax",
        "//jax:experimental",
    ],
)
