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

"""Repository rule to configure external test dependencies.

This module provides a repository rule that accepts a list of dependency targets
and generates a .bzl file containing an EXTERNAL_DEPS variable.

Usage:
    init_external_deps_repo(deps = ["@jax_rocm_plugin//:plugin.whl", "@jax_rocm_plugin//:pjrt.whl"])

    # In your BUILD files, access via:
    # load("@rocm_external_test_deps//:external_deps.bzl", "EXTERNAL_DEPS")
"""

def _external_deps_repository_impl(repository_ctx):
    """Implementation of the external_deps_repository rule.

    Generates an external_deps.bzl file containing an EXTERNAL_DEPS variable.

    Args:
        repository_ctx: The repository context.
    """
    deps_list = repository_ctx.attr.deps

    # Generate the external_deps.bzl file using the template
    repository_ctx.template(
        "external_deps.bzl",
        repository_ctx.attr._build_tpl,
        substitutions = {
            "%{EXTERNAL_DEPS}": str(deps_list),
        },
    )

    repository_ctx.file("BUILD.bazel", "# Auto-generated BUILD file\n")

external_deps_repository = repository_rule(
    implementation = _external_deps_repository_impl,
    attrs = {
        "deps": attr.string_list(
            default = [],
            doc = "List of dependency targets.",
        ),
        "_build_tpl": attr.label(
            default = Label("//third_party/external_deps:external_deps.bzl.tpl"),
        ),
    },
    doc = "Repository rule to configure external test dependencies.",
)

def init_external_deps_repo(name, deps = []):
    """Convenience function to create the external deps repository.

    Args:
        name: The name of the repository (default: "rocm_external_test_deps").
        deps: List of dependency targets to include as EXTERNAL_DEPS.

    Example:
        init_external_deps_repo(deps = ["@some_repo//:target1", "@some_repo//:target2"])

        # In your BUILD files, access via:
        # load("@rocm_external_test_deps//:external_deps.bzl", "EXTERNAL_DEPS")
    """
    external_deps_repository(name = name, deps = deps)
