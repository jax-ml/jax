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

"""Repository rule to parse external test dependencies from environment variable.

This module provides a repository rule that reads the EXTERNAL_TEST_DEPS
environment variable and generates a .bzl file containing a list of targets
that can be used as dependencies.

The environment variable should contain a comma-separated list of targets, e.g.:
    EXTERNAL_TEST_DEPS=@jax_rocm_plugin//:plugin.whl,@jax_rocm_plugin//:pjrt.whl
"""

def _external_test_deps_repository_impl(repository_ctx):
    """Implementation of the external_test_deps_repository rule.

    Reads the EXTERNAL_TEST_DEPS environment variable and generates a deps.bzl
    file containing a list of targets.

    Args:
        repository_ctx: The repository context.
    """
    env_var_name = repository_ctx.attr.env_var
    deps_env = repository_ctx.os.environ.get(env_var_name, "")

    # Parse the comma-separated list of targets
    deps_list = []
    if deps_env:
        for dep in deps_env.split(","):
            dep = dep.strip()
            if dep:
                deps_list.append(dep)

    # Format the deps list for the template
    deps_formatted = "\n".join(['    "{}",'.format(dep) for dep in deps_list])

    # Generate the test_deps.bzl file using the template
    repository_ctx.template(
        "test_deps.bzl",
        repository_ctx.attr._build_tpl,
        substitutions = {
            "%{ENV_VAR}": env_var_name,
            "%{DEPS}": deps_formatted,
        },
    )

    repository_ctx.file("BUILD.bazel", "# Auto-generated BUILD file\n")

external_test_deps_repository = repository_rule(
    implementation = _external_test_deps_repository_impl,
    attrs = {
        "env_var": attr.string(
            default = "EXTERNAL_TEST_DEPS",
            doc = "The name of the environment variable containing the comma-separated list of targets.",
        ),
        "_build_tpl": attr.label(
            default = Label("//third_party/testing:BUILD.tpl"),
        ),
    },
    environ = ["EXTERNAL_TEST_DEPS"],
    doc = "Repository rule to parse external test dependencies from an environment variable.",
)

def repo(name = "external_test_deps"):
    """Convenience function to create the external test deps repository.

    Args:
        name: The name of the repository (default: "external_test_deps").
    """
    external_test_deps_repository(name = name)
