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
    http_archive(
        name = "ducc",
        strip_prefix = "ducc-356d619a4b5f6f8940d15913c14a043355ef23be",
        sha256 = "d23eb2d06f03604867ad40af4fe92dec7cccc2c59f5119e9f01b35b973885c61",
        urls = [
            "https://github.com/mreineck/ducc/archive/356d619a4b5f6f8940d15913c14a043355ef23be.tar.gz",
            "https://storage.googleapis.com/jax-releases/mirror/ducc/ducc-356d619a4b5f6f8940d15913c14a043355ef23be.tar.gz",
        ],
        build_file = "@//third_party/ducc:BUILD.bazel",
    )
