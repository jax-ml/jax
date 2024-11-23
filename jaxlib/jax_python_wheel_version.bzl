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

""" Repository rule to generate a file with JAX python wheel version. """

def _jax_python_wheel_version_repository_impl(repository_ctx):
    file_content = repository_ctx.read(
        repository_ctx.path(repository_ctx.attr.file_with_version),
    )
    version_line_start_index = file_content.find(repository_ctx.attr.version_key)
    version_line_end_index = version_line_start_index + file_content[version_line_start_index:].find("\n")
    repository_ctx.file(
        "wheel_version.bzl",
        file_content[version_line_start_index:version_line_end_index].replace(
            repository_ctx.attr.version_key,
            "WHEEL_VERSION",
        ),
    )
    repository_ctx.file("BUILD", "")

jax_python_wheel_version_repository = repository_rule(
    implementation = _jax_python_wheel_version_repository_impl,
    attrs = {
        "file_with_version": attr.label(mandatory = True, allow_single_file = True),
        "version_key": attr.string(mandatory = True),
    },
)
