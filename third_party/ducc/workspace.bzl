# Copyright 2020 The JAX Authors.
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

"""Bazel workspace for DUCC (CPU FFTs)."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    # Attention: tools parse and update these lines.
    DUCC_COMMIT = "2b2cead005e08d2632478e831d7f45da754162dc"
    DUCC_SHA256 = "60719aa71d637dba594a03fed682bb6943dfffaa5557f8e8bb51228a295bbd79"

    http_archive(
        name = "ducc",
        strip_prefix = "ducc-{commit}".format(commit = DUCC_COMMIT),
        sha256 = DUCC_SHA256,
        urls = [
            "https://gitlab.mpcdf.mpg.de/mtr/ducc/-/archive/{commit}/ducc-{commit}.tar.gz".format(commit = DUCC_COMMIT),
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.mpcdf.mpg.de/mtr/ducc/-/archive/{commit}/ducc-{commit}.tar.gz".format(commit = DUCC_COMMIT),
        ],
        build_file = "@//third_party/ducc:BUILD.bazel",
    )
