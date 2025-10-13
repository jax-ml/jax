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

# buildifier: disable=module-docstring
XLA_COMMIT = "f2bbe0a318ce5dc600e466b75d9bd13e931cb1b7"
XLA_SHA256 = "191df5afd46943a50624883b86163d5c368bf99b4749beb57c3c9ce18f9cc3e1"
