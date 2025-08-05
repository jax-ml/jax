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

""" Repository rule to generate a file with NVIDIA wheel versions. """

def _nvidia_wheel_versions_repository_impl(repository_ctx):
    versions_source = repository_ctx.attr.versions_source

    versions_file_content = repository_ctx.read(
        repository_ctx.path(versions_source),
    )
    repository_ctx.file(
        "versions.bzl",
        "NVIDIA_WHEEL_VERSIONS = '''%s'''" % versions_file_content,
    )
    repository_ctx.file("BUILD", "")

nvidia_wheel_versions_repository = repository_rule(
    implementation = _nvidia_wheel_versions_repository_impl,
    attrs = {
        "versions_source": attr.label(mandatory = True, allow_single_file = True),
    },
)
