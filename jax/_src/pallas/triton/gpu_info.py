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
from collections.abc import Callable
from functools import lru_cache

from jax._src import mesh as mesh_lib
from jax._src import util as jax_util
from jax._src.interpreters import pxla
from jax._src.lib import _gpu_spec


@dataclasses.dataclass
class CustomGpuTargetConfig:
  platform_name: str
  device_description_str: str
  arch_name: str
  compute_capability: int
  core_count: int
  shared_memory_per_core: int


GpuTargetConfig = _gpu_spec.GpuTargetConfig | CustomGpuTargetConfig
GpuModel = _gpu_spec.GpuModel


@lru_cache
def gpu_version_from_device_kind(device_kind: str) -> GpuTargetConfig | None:
  for model in GpuModel:
    config = _gpu_spec.get_gpu_spec(model)
    if config.device_description_str == device_kind:
      return config

  if device_kind in registry:
    return registry[device_kind]
  return None


def is_gpu_device() -> bool:
  return get_device_platform() == "gpu"


registry: dict[str, Callable[[], GpuTargetConfig]] = {
  "Tesla T4": CustomGpuTargetConfig("CUDA", "Tesla T4", "7.5", 75, 0, 0),
  "NVIDIA A30": CustomGpuTargetConfig("CUDA", "NVIDIA A30", "8.0", 80, 0, 0),
  "NVIDIA A10": CustomGpuTargetConfig("CUDA", "NVIDIA A10", "8.6", 86, 0, 0),
  "NVIDIA L4": CustomGpuTargetConfig("CUDA", "NVIDIA L4", "8.9", 89, 0, 0),
  "NVIDIA L40": CustomGpuTargetConfig("CUDA", "NVIDIA L40", "8.9", 89, 0, 0),
  "NVIDIA GeForce RTX 4090": CustomGpuTargetConfig("CUDA", "NVIDIA GeForce RTX 4090", "8.9", 89, 0, 0),

  "NVIDIA GH200": CustomGpuTargetConfig("CUDA", "NVIDIA GH200", "9.0", 90, 0, 0),

  "NVIDIA RTX PRO 4500 Blackwell": CustomGpuTargetConfig("CUDA", "NVIDIA RTX PRO 4500 Blackwell", "12.0", 120, 0, 0),
  "NVIDIA RTX PRO 5000 Blackwell": CustomGpuTargetConfig("CUDA", "NVIDIA RTX PRO 5000 Blackwell", "12.0", 120, 0, 0),

  "NVIDIA GB10": CustomGpuTargetConfig("CUDA", "NVIDIA GB10", "12.1", 121, 0, 0),
  "NVIDIA Thor": CustomGpuTargetConfig("CUDA", "NVIDIA Thor", "11.0", 110, 0, 0),
  "NVIDIA VR200": CustomGpuTargetConfig("CUDA", "NVIDIA VR200", "10.7", 107, 0, 0),
}


@jax_util.cache(trace_context_in_key=True)
def get_gpu_info() -> GpuTargetConfig:
  """Returns the GPU hardware info for the current device."""
  device_kind = get_device_kind()
  gpu_config = gpu_version_from_device_kind(device_kind)
  if gpu_config is not None:
    return gpu_config
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
