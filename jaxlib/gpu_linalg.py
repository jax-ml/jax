# Copyright 2021 The JAX Authors.
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

_cuda_linalg = import_from_plugin("cuda", "_linalg")
_hip_linalg = import_from_plugin("rocm", "_linalg")


def registrations() -> dict[str, list[tuple[str, Any, int]]]:
  registrations: dict[str, list[tuple[str, Any, int]]] = {
      "CUDA": [],
      "ROCM": [],
  }
  for platform, module in [("CUDA", _cuda_linalg), ("ROCM", _hip_linalg)]:
    if module:
      registrations[platform].extend(
          (*i, 1) for i in module.registrations().items()
      )
  return registrations  # pytype: disable=bad-return-type


def batch_partitionable_targets() -> list[str]:
  targets = []
  if _cuda_linalg:
    targets.append("cu_lu_pivots_to_permutation")
  if _hip_linalg:
    targets.append("hip_lu_pivots_to_permutation")
  return targets
