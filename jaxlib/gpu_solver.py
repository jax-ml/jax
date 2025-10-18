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

_cusolver = import_from_plugin("cuda", "_solver")
_cuhybrid = import_from_plugin("cuda", "_hybrid")

_hipsolver = import_from_plugin("rocm", "_solver")
_hiphybrid = import_from_plugin("rocm", "_hybrid")


def registrations() -> dict[str, list[tuple[str, Any, int]]]:
  registrations: dict[str, list[tuple[str, Any, int]]] = {
      "CUDA": [],
      "ROCM": [],
  }
  for platform, module in [("CUDA", _cusolver), ("ROCM", _hipsolver)]:
    if module:
      registrations[platform].extend(
          (name, value, int(name.endswith("_ffi")))
          for name, value in module.registrations().items()
      )
  for platform, module in [("CUDA", _cuhybrid), ("ROCM", _hiphybrid)]:
    if module:
      registrations[platform].extend(
          (*i, 1) for i in module.registrations().items()
      )
  return registrations  # pytype: disable=bad-return-type


def batch_partitionable_targets() -> list[str]:
  targets: list[str] = []
  for module in [_cusolver, _hipsolver]:
    if module:
      targets.extend(
          name for name in module.registrations() if name.endswith("_ffi")
      )
  for module in [_cuhybrid, _hiphybrid]:
    if module:
      targets.extend(name for name in module.registrations())
  return targets


def initialize_hybrid_kernels():
  if _cuhybrid:
    _cuhybrid.initialize()
  if _hiphybrid:
    _hiphybrid.initialize()


def has_magma():
  if _cuhybrid:
    return _cuhybrid.has_magma()
  if _hiphybrid:
    return _hiphybrid.has_magma()
  return False
