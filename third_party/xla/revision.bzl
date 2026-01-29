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
#    curl -L https://github.com/openxla/xla/archive/{git_hash}.tar.gz | sha256sum
#    and update XLA_SHA256 with the result.

# buildifier: disable=module-docstring
XLA_COMMIT = "f6270fd9aac265a44e151fbde30e763fe4e65465"
XLA_SHA256 = "94d55677dffd9b083cdae2fc9f0f2597ca178a4662435510e3614761bf60e936"
