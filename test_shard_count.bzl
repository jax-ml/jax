# Copyright 2025 The JAX Authors.
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

""" Repository rule to generate a file with USE_MINIMAL_SHARD_COUNT. """

def _test_shard_count_repository_impl(repository_ctx):
    USE_MINIMAL_SHARD_COUNT = repository_ctx.getenv("USE_MINIMAL_SHARD_COUNT", "False")
    repository_ctx.file(
        "test_shard_count.bzl",
        "USE_MINIMAL_SHARD_COUNT = %s" % USE_MINIMAL_SHARD_COUNT,
    )
    repository_ctx.file("BUILD", "")

test_shard_count_repository = repository_rule(
    implementation = _test_shard_count_repository_impl,
)
