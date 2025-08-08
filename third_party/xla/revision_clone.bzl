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
XLA_COMMIT = "f4e3f777e378e0351e7e45c1bbef599a42230a6b"
XLA_SHA256 = "d99466f0d7efd22cfbfdc0bc14a8574aae7921fb5a646da812049f3b1c4cdfcf"
