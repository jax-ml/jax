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

_BUILD_CONTENT = """\
package(default_visibility = ["//visibility:public"])

exports_files([
    "libs/libmha_fwd.so",
    "libs/libmha_bwd.so",
])

cc_library(
    name = "aiter_headers",
    hdrs = [
        "include/aiter_hip_common.h",
        "include/aiter_logger.h",
        "include/mha_bwd.h",
        "include/mha_fwd.h",
    ],
    strip_include_prefix = "include",
    include_prefix = "aiter",
)
"""

_AITER_MHA_URL = "https://github.com/ROCm/jax/releases/download/AITER-MHA/aiter_mha-0.1.0-py3-none-linux_x86_64.whl"
_AITER_MHA_SHA256 = "943e8736ded39244acab96c0c4e3901aa0cb166fdbcd1e0bee178799e21868c2"
_AITER_MHA_STRIP_PREFIX = "aiter_mha-0.1.0.data"

_REQUIRED_FILES = [
    "libs/libmha_fwd.so",
    "libs/libmha_bwd.so",
    "include/aiter_hip_common.h",
    "include/aiter_logger.h",
    "include/mha_bwd.h",
    "include/mha_fwd.h",
]

def _aiter_mha_repo_impl(repository_ctx):
    local_path = repository_ctx.os.environ.get("LOCAL_AITER_PATH", "")
    if local_path:
        for f in _REQUIRED_FILES:
            src = local_path + "/" + f
            result = repository_ctx.execute(["test", "-f", src])
            if result.return_code != 0:
                fail("LOCAL_AITER_PATH=%s is missing %s" % (local_path, f))
            repository_ctx.symlink(src, f)
    else:
        repository_ctx.download_and_extract(
            url = [_AITER_MHA_URL],
            sha256 = _AITER_MHA_SHA256,
            type = "zip",
            stripPrefix = _AITER_MHA_STRIP_PREFIX,
        )
    repository_ctx.file("BUILD.bazel", _BUILD_CONTENT)

_aiter_mha_repo = repository_rule(
    implementation = _aiter_mha_repo_impl,
    environ = ["LOCAL_AITER_PATH"],
)

def aiter_mha_whl():
    """Provides @aiter_mha_wheel with libmha_fwd.so, libmha_bwd.so, and headers.

    By default, downloads a pre-built wheel from GitHub. To use local libs:
      a) python build/build.py build --local_aiter_path=/path/to/aiter_mha-0.1.0.data
      b) --bazel_options='--repo_env=LOCAL_AITER_PATH=/path/to/aiter_mha-0.1.0.data'

    The directory must contain libs/{libmha_fwd,libmha_bwd}.so and include/*.h.
    """
    _aiter_mha_repo(name = "aiter_mha_wheel")