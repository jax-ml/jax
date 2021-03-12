# Copyright 2020 Google LLC
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

"""Bazel workspace for PocketFFT."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "pocketfft",
        sha256 = "66eda977b195965d27aeb9d74f46e0029a6a02e75fbbc47bb554aad68615a260",
        strip_prefix = "pocketfft-f800d91ba695b6e19ae2687dd60366900b928002",
        urls = [
            "https://gitlab.mpcdf.mpg.de/mtr/pocketfft/-/archive/f800d91ba695b6e19ae2687dd60366900b928002/pocketfft-f800d91ba695b6e19ae2687dd60366900b928002.tar.gz",
            "https://storage.googleapis.com/jax-releases/mirror/pocketfft/pocketfft-f800d91ba695b6e19ae2687dd60366900b928002.tar.gz",
        ],
        build_file = "@//third_party/pocketfft:BUILD.bazel",
    )
