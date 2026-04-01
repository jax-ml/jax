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

Queries the GitHub Releases API on ROCm/rocm-jax to find wheels matching the
current jaxlib version, Python version, and ROCm version, then exposes them
as py_import targets.

Usage in WORKSPACE (after @jax_wheel and @python_version_repo are initialized):

    load("@jax_wheel//:wheel.bzl", "WHEEL_VERSION")
    load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")

    rocm_wheels_repository(
        name = "rocm_wheels",
        jaxlib_version = WHEEL_VERSION,
        python_version = HERMETIC_PYTHON_VERSION,
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

def _download_release_json(repository_ctx, api_url, headers):
    result = repository_ctx.download(
        url = api_url,
        output = "_release.json",
        headers = headers,
        allow_fail = True,
    )
    if not result.success:
        return None
    content = repository_ctx.read("_release.json")
    repository_ctx.delete("_release.json")
    return json.decode(content)

def _download_releases_list(repository_ctx, api_url, headers):
    """Downloads a page of releases and returns the parsed JSON list."""
    result = repository_ctx.download(
        url = api_url,
        output = "_releases.json",
        headers = headers,
        allow_fail = True,
    )
    if not result.success:
        return []
    content = repository_ctx.read("_releases.json")
    repository_ctx.delete("_releases.json")
    return json.decode(content)

def _major_minor(version):
    """Returns 'MAJOR.MINOR.' from a version string like '0.9.1'."""
    parts = version.split(".")
    if len(parts) < 2:
        return version + "."
    return parts[0] + "." + parts[1] + "."

def _fetch_release_metadata(repository_ctx, headers):
    owner = repository_ctx.attr.github_owner
    repo = repository_ctx.attr.github_repo
    tag = repository_ctx.attr.tag
    base_api = "https://api.github.com/repos/{}/{}".format(owner, repo)

    if tag:
        release = _download_release_json(
            repository_ctx,
            "{}/releases/tags/{}".format(base_api, tag),
            headers,
        )
        if not release:
            fail("Release '{}' not found in {}/{}".format(tag, owner, repo))
        return release

    version_tag = "rocm-jax-v{}".format(repository_ctx.attr.jaxlib_version)

    # Try exact version match first.
    release = _download_release_json(
        repository_ctx,
        "{}/releases/tags/{}".format(base_api, version_tag),
        headers,
    )
    if release:
        return release

    # Exact tag not found — scan all recent releases for the best match.
    # Prefer the highest version with the same major.minor, including RCs
    # (e.g. for jaxlib 0.9.1, match rocm-jax-v0.9.0-rc2).
    mm_prefix = "rocm-jax-v" + _major_minor(repository_ctx.attr.jaxlib_version)
    releases = _download_releases_list(
        repository_ctx,
        "{}/releases?per_page=50".format(base_api),
        headers,
    )
    best = None
    for rel in releases:
        rel_tag = rel.get("tag_name", "")
        if rel_tag.startswith(mm_prefix):
            if not best or rel_tag > best.get("tag_name", ""):
                best = rel

    if best:
        # buildifier: disable=print
        print("rocm_wheels: release '{}' not found, using '{}'".format(
            version_tag,
            best["tag_name"],
        ))
        return best

    # Nothing matched the major.minor — pick the overall newest release.
    if releases:
        newest = releases[0]

        # buildifier: disable=print
        print("rocm_wheels: no release matching '{}', falling back to '{}'".format(
            version_tag,
            newest["tag_name"],
        ))
        return newest

    fail("No releases found in {}/{}".format(owner, repo))

def _find_wheel_assets(release, python_version, rocm_version):
    """Finds the plugin and pjrt wheel assets in a release."""
    cp_tag = "cp" + python_version.replace(".", "")
    assets = release.get("assets", [])
    tag_name = release.get("tag_name", "unknown")

    plugin_asset = None
    pjrt_asset = None

    for asset in assets:
        name = asset["name"]
        if not name.endswith(".whl"):
            continue

        if rocm_version and ("rocm" + rocm_version) not in name:
            continue

        is_plugin = "plugin" in name and "pjrt" not in name
        is_pjrt = "pjrt" in name

        if is_plugin and cp_tag in name:
            plugin_asset = asset
        elif is_pjrt:
            pjrt_asset = asset

    wheel_names = [a["name"] for a in assets if a["name"].endswith(".whl")]

    for label, asset in [("plugin", plugin_asset), ("pjrt", pjrt_asset)]:
        if not asset:
            fail(
                "No ROCm {} wheel found for Python {} (abi '{}') and ".format(label, python_version, cp_tag) +
                "ROCm '{}' in release '{}'. Available wheels:\n  {}".format(
                    rocm_version or "any",
                    tag_name,
                    "\n  ".join(wheel_names),
                ),
            )

    return plugin_asset, pjrt_asset

def _rocm_wheels_repository_impl(repository_ctx):
    python_version = repository_ctx.attr.python_version
    jaxlib_version = repository_ctx.attr.jaxlib_version
    rocm_version = repository_ctx.attr.rocm_version

    py_tag = "cp" + python_version.replace(".", "")
    rocm_tag = "rocm" + rocm_version if rocm_version else ""

    # Check for local wheels in jax/dist/ first in case they
    # are downloaded already from S3 or pip.
    dist_dir = repository_ctx.workspace_root.get_child("dist")
    dl_plugin = None
    dl_pjrt = None

    if dist_dir.exists:
        for entry in dist_dir.readdir():

            name = entry.basename
            if not name.endswith(".whl"):
                continue
            if rocm_tag and rocm_tag not in name:
                continue

            if "plugin" in name and "pjrt" not in name and py_tag in name:
                dl_plugin = entry
            elif "pjrt" in name:
                dl_pjrt = entry

    if dl_plugin and dl_pjrt:

        # buildifier: disable=print
        print("rocm_wheels: dist/ -  plugin='{}', pjrt='{}'".format(
            dl_plugin.basename,
            dl_pjrt.basename,
        ))
        repository_ctx.symlink(dl_plugin, "plugin.whl")
        repository_ctx.symlink(dl_pjrt, "pjrt.whl")

    else:
        headers = _get_github_headers(repository_ctx)
        release = _fetch_release_metadata(repository_ctx, headers)

        plugin_asset, pjrt_asset = _find_wheel_assets(
            release,
            python_version,
            rocm_version,
        )

        tag_name = release.get("tag_name", "unknown")

        # buildifier: disable=print
        print("rocm_wheels: release '{}' - plugin='{}', pjrt='{}'".format(
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
            default = "rocm-jax",
            doc = "GitHub repository name.",
        ),
        "tag": attr.string(
            default = "",
            doc = "GitHub release tag override. If empty, derived as 'rocm-jax-v{jaxlib_version}'.",
        ),
        "jaxlib_version": attr.string(
            mandatory = True,
            doc = "JAX/jaxlib version to match, e.g. '0.9.1'.",
        ),
        "python_version": attr.string(
            mandatory = True,
            doc = "Python version to match wheels against, e.g. '3.11'.",
        ),
        "rocm_version": attr.string(
            default = "",
            doc = "ROCm version to match, e.g. '7.2.0'. If empty, picks the first match.",
        ),
    },
    environ = ["GITHUB_TOKEN"],
    doc = "Downloads ROCm plugin wheels from a GitHub release and exposes them as py_import targets.",
)
