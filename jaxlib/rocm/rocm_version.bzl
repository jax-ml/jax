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

"""ROCm version constants derived from @local_config_rocm.

The rocm_version_number() function from build_defs.bzl returns an integer
encoded as: major * 10000 + minor * 100 + patch (e.g. 70101 for 7.1.1).
"""

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_version_number",
)

_version = rocm_version_number()
ROCM_MAJOR_VERSION = str(_version // 10000)
ROCM_MINOR_VERSION = str((_version % 10000) // 100)
ROCM_PATCH_VERSION = str(_version % 100)
