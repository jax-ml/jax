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

"""Exposes GPU hardware information."""

import dataclasses
import enum
import re
from collections.abc import Callable

from jax._src import mesh as mesh_lib
from jax._src import util as jax_util
from jax._src.interpreters import pxla
from jax._src.lib import _gpu_spec


GpuTargetConfig = _gpu_spec.GpuTargetConfig
GpuDeviceInfo = _gpu_spec.GpuDeviceInfo


def gpu_version_from_device_kind(device_kind: str) -> _gpu_spec.GpuTargetConfig | None:
  try:
    return _gpu_spec.get_gpu_spec(device_kind)
  except RuntimeError:
    return None


def is_gpu_device() -> bool:
  return get_device_platform() == "gpu"


registry: dict[str, Callable[[], _gpu_spec.GpuTargetConfig]] = {}


@jax_util.cache(trace_context_in_key=True)
def get_gpu_info() -> _gpu_spec.GpuTargetConfig:
  """Returns the GPU hardware info for the current device."""
  device_kind = get_device_kind()
  gpu_config = gpu_version_from_device_kind(device_kind)
  if gpu_config is not None:
    return gpu_config

  if device_kind in registry:
    return registry[device_kind]()
  raise ValueError(f"Unsupported GPU device kind: {device_kind}")


def get_device_kind() -> str:
  abstract_device = mesh_lib.get_abstract_mesh().abstract_device
  if abstract_device is not None:
    return abstract_device.device_kind
  return pxla.get_default_device().device_kind


_GPU_PLATFORMS = ("gpu", "rocm", "cuda")


def get_device_platform() -> str:
  if abstract_device := mesh_lib.get_abstract_mesh().abstract_device:
    platform = abstract_device.platform
  else:
    platform = pxla.get_default_device().platform
  return "gpu" if platform in _GPU_PLATFORMS else platform
