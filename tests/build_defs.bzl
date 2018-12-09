# Copyright 2018 Google LLC
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

"""Helpers for defining multi-platform JAX binary/test targets."""

def jax_test(
        name,
        srcs,
        deps = [],
        args = [],
        disable = [],
        data = [],
        main = None,
        shard_count = None):
    """Defines a set of per-platform JAX test targets."""
    if main == None:
        if len(srcs) == 1:
            main = srcs[0]
        else:
            fail("Only one test source file is currently supported.")

    # Deps that are linked into all test target variants.
    all_test_deps = []
    disabled_tags = ["manual", "notap", "disabled"]
    native.py_test(
        name = name + "_cpu",
        main = main,
        srcs = srcs,
        data = data,
        deps = deps + all_test_deps,
        shard_count = shard_count.get("cpu") if shard_count else None,
        args = args + ["--jax_enable_x64=true --jax_test_dut=cpu --jax_platform_name=Host"],
        tags = (disabled_tags if "cpu" in disable else []),
    )
    # native.py_test(
    #     name = name + "_cpu_x32",
    #     main = main,
    #     srcs = srcs,
    #     data = data,
    #     deps = deps + all_test_deps,
    #     shard_count = shard_count.get("cpu") if shard_count else None,
    #     args = args + ["--jax_enable_x64=false --jax_test_dut=cpu --jax_platform_name=Host"],
    #     tags = (disabled_tags if "cpu" in disable else []),
    # )
    # native.py_test(
    #     name = name + "_gpu",
    #     main = main,
    #     srcs = srcs,
    #     data = data,
    #     args = args + ["--jax_test_dut=gpu --jax_enable_x64=true --jax_platform_name=CUDA"],
    #     deps = deps + all_test_deps,
    #     tags = ["requires-gpu-sm35"] + (disabled_tags if "gpu" in disable else []),
    #     shard_count = shard_count.get("gpu") if shard_count else None,
    # )
    # native.py_test(
    #     name = name + "_gpu_x32",
    #     main = main,
    #     srcs = srcs,
    #     data = data,
    #     args = args + ["--jax_test_dut=gpu --jax_enable_x64=false --jax_platform_name=CUDA"],
    #     deps = deps + all_test_deps,
    #     tags = ["requires-gpu-sm35"] + (disabled_tags if "gpu" in disable else []),
    #     shard_count = shard_count.get("gpu") if shard_count else None,
    # )
