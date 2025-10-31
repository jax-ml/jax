# Copyright 2019 The JAX Authors.
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

from typing import Any

from .plugin_support import import_from_plugin

_cuda_prng = import_from_plugin("cuda", "_prng")
_hip_prng = import_from_plugin("rocm", "_prng")


def registrations() -> dict[str, list[tuple[str, Any, int]]]:
  registrations: dict[str, list[tuple[str, Any, int]]] = {
      "CUDA": [],
      "ROCM": [],
  }
  for platform, module in [("CUDA", _cuda_prng), ("ROCM", _hip_prng)]:
    if module:
      registrations[platform].extend(
          (name, value, int(name.endswith("_ffi")))
          for name, value in module.registrations().items()
      )
  return registrations
