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

"""Loads the Flatbuffers library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "com_github_google_flatbuffers",
        strip_prefix = "flatbuffers-23.5.26",
        sha256 = "1cce06b17cddd896b6d73cc047e36a254fb8df4d7ea18a46acf16c4c0cd3f3f3",
        urls = tf_mirror_urls("https://github.com/google/flatbuffers/archive/v23.5.26.tar.gz"),
        patch_file = [
            "//third_party/flatbuffers:flatbuffers.patch",
        ],
    )
