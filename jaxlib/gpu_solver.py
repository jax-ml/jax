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

from .plugin_support import import_from_plugin, load_pkg_so

# try loading libcudart from python package directory before import cuda plugin.
# the native plugin also trigger libcudart dlopen from its call_init,
# but no python context there to easily lookup package directory.
# it has to rely on RPATH or LD_LIBRARY_PATH to locate libcudart,
# if we can load libcudart from python package,
# it's much more reliable than RPATH or LD_LIBRARY_PATH
# this is also consist with jax_plugins/cuda/__init__.py logic
# so we either gets exactly same libcudart.so,
# or fail in both location.
load_pkg_so("cuda_runtime", ["libcudart.so.12"])
load_pkg_so("cu13", ["libcudart.so.13"])
_cusolver = import_from_plugin("cuda", "_solver")
_cuhybrid = import_from_plugin("cuda", "_hybrid")

_hipsolver = import_from_plugin("rocm", "_solver")
_hiphybrid = import_from_plugin("rocm", "_hybrid")

_oneapisolver = import_from_plugin("oneapi", "_solver")
_oneapihybrid = import_from_plugin("oneapi", "_hybrid")


def registrations() -> dict[str, list[tuple[str, Any, int]]]:
  registrations: dict[str, list[tuple[str, Any, int]]] = {
      "CUDA": [],
      "ROCM": [],
      "ONEAPI": [],
  }
  for platform, module in [("CUDA", _cusolver), ("ROCM", _hipsolver),
                           ("ONEAPI", _oneapisolver)]:
    if module:
      registrations[platform].extend(
          (name, value, int(name.endswith("_ffi")))
          for name, value in module.registrations().items()
      )
  for platform, module in [("CUDA", _cuhybrid), ("ROCM", _hiphybrid),
                            ("ONEAPI", _oneapihybrid)]:
    if module:
      registrations[platform].extend(
          (*i, 1) for i in module.registrations().items()
      )
  return registrations


def batch_partitionable_targets() -> list[str]:
  targets: list[str] = []
  for module in [_cusolver, _hipsolver, _oneapisolver]:
    if module:
      targets.extend(
          name for name in module.registrations() if name.endswith("_ffi")
      )
  for module in [_cuhybrid, _hiphybrid, _oneapihybrid]:
    if module:
      targets.extend(name for name in module.registrations())
  return targets


def initialize_hybrid_kernels():
  if _cuhybrid:
    _cuhybrid.initialize()
  if _hiphybrid:
    _hiphybrid.initialize()
  if _oneapihybrid:
    _oneapihybrid.initialize()


def has_magma():
  if _cuhybrid:
    return _cuhybrid.has_magma()
  if _hiphybrid:
    return _hiphybrid.has_magma()
  return False
