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

""" Repository rule to generate a file with JAX wheel version. """

def _jax_python_wheel_repository_impl(repository_ctx):
    version_source = repository_ctx.attr.version_source
    version_key = repository_ctx.attr.version_key

    version_file_content = repository_ctx.read(
        repository_ctx.path(version_source),
    )
    version_start_index = version_file_content.find(version_key)
    version_end_index = version_start_index + version_file_content[version_start_index:].find("\n")

    wheel_version = version_file_content[version_start_index:version_end_index].replace(
        version_key,
        "WHEEL_VERSION",
    )
    repository_ctx.file(
        "wheel.bzl",
        wheel_version,
    )
    repository_ctx.file("BUILD", "")

jax_python_wheel_repository = repository_rule(
    implementation = _jax_python_wheel_repository_impl,
    attrs = {
        "version_source": attr.label(mandatory = True, allow_single_file = True),
        "version_key": attr.string(mandatory = True),
    },
)
