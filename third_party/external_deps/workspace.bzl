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

"""Repository rule to parse external test dependencies from environment variables.

This module provides a repository rule that reads a list of environment variables
and generates a .bzl file containing a struct with dependency lists for each
environment variable.

Each environment variable should contain a comma-separated list of targets, e.g.:
    TEST_DEPS=@jax_rocm_plugin//:plugin.whl,@jax_rocm_plugin//:pjrt.whl

The generated struct uses lowercase environment variable names as keys, e.g.:
    external.test_deps  # list of deps from TEST_DEPS
"""

# List of environment variables to parse for external dependencies.
# Each variable should contain a comma-separated list of Bazel targets.
# The generated struct will have fields named after the lowercase variable names.
EXTERNAL_DEPS_ENV_VARS = [
    "TEST_DEPS",
]

def _parse_deps_from_env(repository_ctx, env_var_name):
    """Parses a comma-separated list of deps from an environment variable.

    Args:
        repository_ctx: The repository context.
        env_var_name: The name of the environment variable.

    Returns:
        A list of dependency targets.
    """
    deps_env = repository_ctx.getenv(env_var_name, "")
    deps_list = []
    if deps_env:
        for dep in deps_env.split(","):
            dep = dep.strip()
            if dep:
                deps_list.append(dep)
    return deps_list

def _external_deps_repository_impl(repository_ctx):
    """Implementation of the external_deps_repository rule.

    Reads the specified environment variables and generates an external_deps.bzl
    file containing a struct with dependency lists.

    Args:
        repository_ctx: The repository context.
    """
    env_vars = repository_ctx.attr.env_vars

    struct_fields = []
    env_var_comments = []
    for env_var_name in env_vars:
        deps_list = _parse_deps_from_env(repository_ctx, env_var_name)
        field_name = env_var_name.lower()

        # Format the deps list
        if deps_list:
            deps_formatted = "\n".join(['        "{}",'.format(dep) for dep in deps_list])
            field_content = "    {} = [\n{}\n    ],".format(field_name, deps_formatted)
        else:
            field_content = "    {} = [],".format(field_name)

        struct_fields.append(field_content)
        env_var_comments.append(env_var_name)

    # Generate the external_deps.bzl file using the template
    repository_ctx.template(
        "external_deps.bzl",
        repository_ctx.attr._build_tpl,
        substitutions = {
            "%{ENV_VARS}": ", ".join(env_var_comments),
            "%{STRUCT_FIELDS}": "\n".join(struct_fields),
        },
    )

    repository_ctx.file("BUILD.bazel", "# Auto-generated BUILD file\n")

external_deps_repository = repository_rule(
    implementation = _external_deps_repository_impl,
    attrs = {
        "env_vars": attr.string_list(
            default = EXTERNAL_DEPS_ENV_VARS,
            doc = "List of environment variable names containing comma-separated lists of targets.",
        ),
        "_build_tpl": attr.label(
            default = Label("//third_party/external_deps:BUILD.tpl"),
        ),
    },
    doc = "Repository rule to parse external dependencies from environment variables.",
)

def repo(name = "external_deps", env_vars = EXTERNAL_DEPS_ENV_VARS):
    """Convenience function to create the external deps repository.

    Args:
        name: The name of the repository (default: "external_deps").
        env_vars: List of environment variable names to parse (default: ["EXTERNAL_TEST_DEPS"]).
    """
    external_deps_repository(name = name, env_vars = env_vars)
