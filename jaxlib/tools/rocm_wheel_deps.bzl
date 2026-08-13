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

"""Build rules for aggregating rocm runtime dependecies into a single target."""

def _collect_rocm_data_files_impl(ctx):
    rocm_repo = ctx.attr.rocm_repo.label.repo_name

    runfiles = depset(transitive = [
        root[DefaultInfo].default_runfiles.files
        for root in ctx.attr.roots
    ])
    files = {}
    for f in runfiles.to_list():
        if f.owner and f.owner.repo_name == rocm_repo:
            files[f] = True
    files_depset = depset(files.keys())
    return [DefaultInfo(
        files = files_depset,
        runfiles = ctx.runfiles(transitive_files = files_depset),
    )]

_collect_rocm_data_files = rule(
    implementation = _collect_rocm_data_files_impl,
    attrs = {
        "roots": attr.label_list(
            mandatory = True,
        ),
        "rocm_repo": attr.label(
            default = Label("@local_config_rocm//rocm:hip_runtime"),
        ),
    },
)

def collect_rocm_data_files(name, roots):
    _collect_rocm_data_files(name = name + "_gather", roots = roots)
    native.cc_library(name = name, data = [":" + name + "_gather"])
