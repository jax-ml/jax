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

"""Repository rule to download ROCm plugin wheels from a GitHub release.

Queries the GitHub Releases API to discover the latest (or a specific) release,
finds the matching plugin and PJRT wheel assets, and exposes them as py_import
targets.

Usage in MODULE.bazel:

    rocm_wheels_repository = use_repo_rule(
        "//third_party/rocm_wheels:workspace.bzl",
        "rocm_wheels_repository",
    )

    # Fetch the latest release (Python version from HERMETIC_PYTHON_VERSION):
    rocm_wheels_repository(name = "rocm_wheels")

    # Or pin to a specific release tag and/or Python version:
    rocm_wheels_repository(
        name = "rocm_wheels",
        tag = "jax-rocm-v0.5.0",
        python_version = "3.11",
    )

Then reference the targets as:
    @rocm_wheels//:rocm_plugin_py_import
    @rocm_wheels//:rocm_pjrt_py_import

Set the GITHUB_TOKEN environment variable to avoid API rate limits in CI.
"""

_BUILD_TEMPLATE = """\
load("@xla//third_party/py:py_import.bzl", "py_import")

package(default_visibility = ["//visibility:public"])

exports_files([
    "plugin.whl",
    "pjrt.whl",
])

py_import(
    name = "rocm_plugin_py_import",
    wheel = "plugin.whl",
)

py_import(
    name = "rocm_pjrt_py_import",
    wheel = "pjrt.whl",
)
"""

def _get_github_headers(repository_ctx):
    headers = {"Accept": "application/vnd.github+json"}
    github_token = repository_ctx.os.environ.get("GITHUB_TOKEN", "")
    if github_token:
        headers["Authorization"] = "Bearer " + github_token
    return headers

def _fetch_release_metadata(repository_ctx, headers):
    owner = repository_ctx.attr.github_owner
    repo = repository_ctx.attr.github_repo
    tag = repository_ctx.attr.tag

    if tag:
        api_url = "https://api.github.com/repos/{}/{}/releases/tags/{}".format(
            owner,
            repo,
            tag,
        )
    else:
        api_url = "https://api.github.com/repos/{}/{}/releases/latest".format(
            owner,
            repo,
        )

    result = repository_ctx.download(
        url = api_url,
        output = "_release.json",
        headers = headers,
    )

    if not result.success:
        fail("Failed to fetch release metadata from {}".format(api_url))

    content = repository_ctx.read("_release.json")
    repository_ctx.delete("_release.json")
    return json.decode(content)

def _find_wheel_assets(release, python_version):
    """Finds the plugin and pjrt wheel assets in a release.

    Returns (plugin_asset, pjrt_asset) or fails if not found.
    """
    cp_tag = "cp" + python_version.replace(".", "")
    assets = release.get("assets", [])
    tag_name = release.get("tag_name", "unknown")

    plugin_asset = None
    pjrt_asset = None

    for asset in assets:
        name = asset["name"]
        if not name.endswith(".whl"):
            continue
        if "rocm" not in name:
            continue

        is_plugin = "plugin" in name and "pjrt" not in name
        is_pjrt = "pjrt" in name

        if is_plugin and cp_tag in name:
            plugin_asset = asset
        elif is_pjrt:
            pjrt_asset = asset

    if not plugin_asset:
        available = [a["name"] for a in assets if a["name"].endswith(".whl")]
        fail(
            "No ROCm plugin wheel found for Python {} (abi tag '{}') in " +
            "release '{}'. Available wheels: {}".format(
                python_version,
                cp_tag,
                tag_name,
                available,
            ),
        )

    if not pjrt_asset:
        available = [a["name"] for a in assets if a["name"].endswith(".whl")]
        fail(
            "No ROCm PJRT wheel found in release '{}'. " +
            "Available wheels: {}".format(tag_name, available),
        )

    return plugin_asset, pjrt_asset

def _rocm_wheels_repository_impl(repository_ctx):
    python_version = repository_ctx.os.environ.get(
        "HERMETIC_PYTHON_VERSION",
        repository_ctx.attr.python_version,
    )
    if not python_version:
        fail(
            "python_version must be set either via the 'python_version' " +
            "attribute or the HERMETIC_PYTHON_VERSION environment variable.",
        )

    headers = _get_github_headers(repository_ctx)
    release = _fetch_release_metadata(repository_ctx, headers)

    plugin_asset, pjrt_asset = _find_wheel_assets(
        release,
        python_version,
    )

    tag_name = release.get("tag_name", "unknown")

    # buildifier: disable=print
    print("rocm_wheels: using release '{}' — plugin='{}', pjrt='{}'".format(
        tag_name,
        plugin_asset["name"],
        pjrt_asset["name"],
    ))

    repository_ctx.download(
        url = plugin_asset["browser_download_url"],
        output = "plugin.whl",
        headers = headers,
    )

    repository_ctx.download(
        url = pjrt_asset["browser_download_url"],
        output = "pjrt.whl",
        headers = headers,
    )

    repository_ctx.file("BUILD.bazel", _BUILD_TEMPLATE)

rocm_wheels_repository = repository_rule(
    implementation = _rocm_wheels_repository_impl,
    attrs = {
        "github_owner": attr.string(
            default = "ROCm",
            doc = "GitHub repository owner.",
        ),
        "github_repo": attr.string(
            default = "jax",
            doc = "GitHub repository name.",
        ),
        "tag": attr.string(
            default = "",
            doc = "GitHub release tag. If empty, the latest release is used.",
        ),
        "python_version": attr.string(
            default = "",
            doc = "Python version to match wheels against, e.g. '3.11'. " +
                  "If empty, reads from the HERMETIC_PYTHON_VERSION env var.",
        ),
    },
    environ = ["GITHUB_TOKEN", "HERMETIC_PYTHON_VERSION"],
    doc = "Downloads ROCm plugin wheels from a GitHub release and exposes them as py_import targets.",
)
