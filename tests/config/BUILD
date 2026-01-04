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

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

# ----- Flags, Config settings -----

# TSAN
config_setting(
    name = "is_tsan",
    values = {"copt": "-fsanitize=thread"},
)

# Free-threading
bool_flag(
    name = "py_freethreaded",
    build_setting_default = False,
)

config_setting(
    name = "is_freethreading",
    flag_values = {":py_freethreaded": "true"},
)

# RBE
config_setting(
    name = "is_exec_env_rbe",
    define_values = {"exec_env": "rbe"},
)

config_setting(
    name = "is_executor_remote",
    define_values = {"EXECUTOR": "remote"},
)

selects.config_setting_group(
    name = "is_rbe",
    match_any = [
        ":is_exec_env_rbe",
        ":is_executor_remote",
    ],
)

# A composite setting to use for certain tests
# that use a lot of RAM when all these conditions are true.
selects.config_setting_group(
    name = "tsan_freethreading_rbe",
    match_all = [
        ":is_tsan",
        ":is_freethreading",
        ":is_rbe",
    ],
)

# ----- TSAN suppression files -----

filegroup(
    name = "tsan_suppressions_txts",
    srcs = [
        "tsan-suppressions_3.13.txt",
        "tsan-suppressions_3.14.txt",
    ],
)

# ----- Wrapper script -----

# Increases the stack size in case the underlying environment,
# like the typical Ubuntu distro, has it set to something like 4096,
# which is not enough for JAX tests under TSAN.
sh_binary(
    name = "oss_tsan_wrapper_sh",
    srcs = [":oss_tsan_wrapper.sh"],
    data = [":tsan_suppressions_txts"],
    tags = [
        "manual",
        "notap",
    ],
)
