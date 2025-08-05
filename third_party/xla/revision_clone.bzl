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

# buildifier: disable=module-docstring

# To update XLA to a new revision,
# a) update XLA_COMMIT to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://api.github.com/repos/openxla/xla/tarball/{git_hash} | sha256sum
#    and update XLA_SHA256 with the result.
# Note: This is a clone of revision.bzl for testing the new automation for updating XLA in JAX.

# buildifier: disable=module-docstring
XLA_COMMIT = "34fa03f5acdd6d92ca8e3c82c8ae5d9a33e174fa"
XLA_SHA256 = "1a27df73ca5f2073e657d369bbddddd3355f8577d13d537f2e28ef1c9bbe1c47"
