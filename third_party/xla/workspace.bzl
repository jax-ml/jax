# Copyright 2023 The JAX Authors.
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

# buildifier: disable=module-docstring
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/xla:revision.bzl", "XLA_COMMIT", "XLA_SHA256")

def repo():
    tf_http_archive(
        name = "xla",
        sha256 = XLA_SHA256,
        type = "tar.gz",
        strip_prefix = "openxla-xla-{commit}".format(commit = XLA_COMMIT[:7]),
        # We use an automated tool to update the revision.bzl file. GitHub prohibits the crawling of
        # web links (`/archive/`) links so we use the GitHub API endpoint to get the tarball
        # instead.
        urls = tf_mirror_urls("https://api.github.com/repos/openxla/xla/tarball/{commit}".format(commit = XLA_COMMIT)),
        patch_file = ["//third_party/xla:fix.patch"],
    )

    # For development, one often wants to make changes to the TF repository as well
    # as the JAX repository. You can override the pinned repository above with a
    # local checkout by either:
    # a) overriding the TF repository on the build.py command line by passing a flag
    #    like:
    #    python build/build.py build --local_xla_path=/path/to/xla
    #    or
    # b) by commenting out the http_archive above and uncommenting the following:
    # local_repository(
    #    name = "xla",
    #    path = "/path/to/xla",
    # )
