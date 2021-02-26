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
        sha256 = "bba6962b9f71a220b4873549bad5e6e5a2630bc465e3f9a9822c4ab2418709a7",
        strip_prefix = "pocketfft-53e9dd4d12f986207c96d97c5183f5a72239c76e",
        urls = [
            "https://gitlab.mpcdf.mpg.de/mtr/pocketfft/-/archive/53e9dd4d12f986207c96d97c5183f5a72239c76e/pocketfft-53e9dd4d12f986207c96d97c5183f5a72239c76e.tar.gz",
            "https://storage.googleapis.com/jax-releases/mirror/pocketfft/pocketfft-53e9dd4d12f986207c96d97c5183f5a72239c76e.tar.gz",
        ],
        build_file = "@//third_party/pocketfft:BUILD.bazel",
    )
