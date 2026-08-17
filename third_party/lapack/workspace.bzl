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

"""Loads Netlib BLAS and LAPACK libraries."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "blas",
        build_file = "//third_party/lapack:blas.BUILD",
        sha256 = "2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422",
        strip_prefix = "lapack-3.12.1/BLAS",
        urls = tf_mirror_urls("https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz"),
    )

    tf_http_archive(
        name = "lapack",
        build_file = "//third_party/lapack:lapack.BUILD",
        sha256 = "2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422",
        strip_prefix = "lapack-3.12.1",
        urls = tf_mirror_urls("https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz"),
    )
