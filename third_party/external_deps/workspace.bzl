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

This module provides a repository rule that accepts dependency lists directly
and generates a .bzl file containing a struct with those dependencies.

Usage:
    repo(
        TEST_DEPS = ["@jax_rocm_plugin//:plugin.whl", "@jax_rocm_plugin//:pjrt.whl"],
    )

The generated struct uses lowercase field names as keys, e.g.:
    external.test_deps  # list of deps from TEST_DEPS
"""

# List of declared external dependency names.
# Each name corresponds to a keyword argument that can be passed to repo().
# The generated struct will have fields named after the lowercase names.
EXTERNAL_DEPS_NAMES = [
    "TEST_DEPS",
]

def _external_deps_repository_impl(repository_ctx):
    """Implementation of the external_deps_repository rule.

    Generates an external_deps.bzl file containing a struct with dependency lists.

    Args:
        repository_ctx: The repository context.
    """
    deps_dict = repository_ctx.attr.deps

    struct_fields = []
    field_names = []
    for field_name, deps_list in deps_dict.items():
        struct_fields.append("    {} = {},".format(field_name, deps_list))
        field_names.append(field_name)

    # Generate the external_deps.bzl file using the template
    repository_ctx.template(
        "external_deps.bzl",
        repository_ctx.attr._build_tpl,
        substitutions = {
            "%{ENV_VARS}": ", ".join(field_names),
            "%{STRUCT_FIELDS}": "\n".join(struct_fields),
        },
    )

    repository_ctx.file("BUILD.bazel", "# Auto-generated BUILD file\n")

external_deps_repository = repository_rule(
    implementation = _external_deps_repository_impl,
    attrs = {
        "deps": attr.string_list_dict(
            default = {},
            doc = "Dictionary mapping field names to lists of dependency targets.",
        ),
        "_build_tpl": attr.label(
            default = Label("//third_party/external_deps:external_deps.bzl.tpl"),
        ),
    },
    doc = "Repository rule to configure external dependencies.",
)

def repo(name = "external_deps", **kwargs):
    """Convenience function to create the external deps repository.

    Args:
        name: The name of the repository (default: "external_deps").
        **kwargs: Keyword arguments mapping names from EXTERNAL_DEPS_NAMES
                  (e.g., TEST_DEPS) to lists of dependency targets.
                  Names are converted to lowercase for the generated struct fields.

    Example:
        repo(
            TEST_DEPS = ["@some_repo//:target1", "@some_repo//:target2"],
        )

        # In your BUILD files, access via:
        # load("@external_deps//:external_deps.bzl", "external")
        # external.test_deps  -> ["@some_repo//:target1", "@some_repo//:target2"]
    """
    deps = {}

    for dep_name in EXTERNAL_DEPS_NAMES:
        field_name = dep_name.lower()
        deps[field_name] = []

    for key, value in kwargs.items():
        field_name = key.lower()
        deps[field_name] = value

    external_deps_repository(name = name, deps = deps)
