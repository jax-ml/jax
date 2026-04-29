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

"""Repository rule that fetches NCCL source matching HERMETIC_NCCL_VERSION."""

load("//third_party:repo.bzl", "tf_mirror_urls")

_MIN_MAJOR = 2
_MIN_MINOR = 28

_STUB_BUILD = """\
# NCCL {reason} — device API bitcode not available.
package(default_visibility = ["//visibility:public"])

filegroup(name = "ir_wrapper_srcs", srcs = [])

cc_library(name = "nccl_hdrs")
"""

def _nccl_source_repo_impl(repository_ctx):
    version = repository_ctx.os.environ.get("HERMETIC_NCCL_VERSION", "")
    if not version:
        repository_ctx.file("BUILD", _STUB_BUILD.format(
            reason = "not configured",
        ))
        return

    parts = version.split(".")
    if len(parts) < 3:
        fail("HERMETIC_NCCL_VERSION must be MAJOR.MINOR.PATCH, got: " + version)

    major, minor, patch = parts[0], parts[1], parts[2]

    if int(major) < _MIN_MAJOR or (int(major) == _MIN_MAJOR and int(minor) < _MIN_MINOR):
        repository_ctx.file("BUILD", _STUB_BUILD.format(
            reason = version + " < {}.{} (no device API)".format(_MIN_MAJOR, _MIN_MINOR),
        ))
        return

    version_code = str(int(major) * 10000 + int(minor) * 100 + int(patch))

    tag = "v{}-1".format(version)
    strip_prefix = "nccl-{}-1".format(version)
    url = "https://github.com/NVIDIA/nccl/archive/refs/tags/{}.tar.gz".format(tag)

    repository_ctx.download_and_extract(
        url = tf_mirror_urls(url),
        stripPrefix = strip_prefix,
    )

    repository_ctx.template(
        "BUILD",
        repository_ctx.attr._build_template,
        substitutions = {
            "%{major}": major,
            "%{minor}": minor,
            "%{patch}": patch,
            "%{version_code}": version_code,
        },
    )

nccl_source_repo = repository_rule(
    implementation = _nccl_source_repo_impl,
    attrs = {
        "_build_template": attr.label(
            default = Label("//third_party/nccl_device_bitcode:BUILD.template"),
        ),
    },
    environ = ["HERMETIC_NCCL_VERSION"],
    doc = "Fetches NCCL source tree matching HERMETIC_NCCL_VERSION.",
)
