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
from typing import Any
import dataclasses
from collections.abc import Callable
from functools import lru_cache

from jax._src import mesh as mesh_lib
from jax._src import util as jax_util
from jax._src.interpreters import pxla

from jax._src.lib import _gpu_spec

_GpuTargetConfig = _gpu_spec.GpuTargetConfig
GpuModel = _gpu_spec.GpuModel
get_gpu_spec = _gpu_spec.get_gpu_spec


@dataclasses.dataclass(frozen=True)
class CustomGpuTargetConfig:
  platform_name: str
  device_description_str: str
  arch_name: str
  compute_capability: int
  core_count: int
  shared_memory_per_core: int


GpuTargetConfig = _GpuTargetConfig | CustomGpuTargetConfig


@lru_cache
def gpu_version_from_device_kind(device_kind: str) -> GpuTargetConfig | None:
  for model in GpuModel:
    config = get_gpu_spec(model)
    if config.device_description_str == device_kind:
      return config

  concrete_device = pxla.get_default_device()
  if concrete_device.device_kind not in registry and is_gpu_device(concrete_device.platform):
    # Construct CustomGpuTargetConfig from the current GPU device and add it to the registry.
    # If input device_kind (corresponding to a real or an abstract device) matches the
    # current GPU device, we return built CustomGpuTargetConfig,
    # otherwise, None is returned
    platform_name = (
      "CUDA" if "cuda" in concrete_device.client.platform_version else
      "ROCM" if "rocm" in concrete_device.client.platform_version else
      ""
    )
    core_count = concrete_device.core_count
    arch_name = concrete_device.compute_capability
    if platform_name == "CUDA":
      name = arch_name.replace(".", "")
      if not name[-1].isdigit():
        # remove "a" or "f" from arch name: 12.1a
        name = name[:-1]
      cc = int(name)
    else:
      cc = 0
    shared_memory_per_core = concrete_device.shared_memory_per_block_optin
    registry[concrete_device.device_kind] = lambda: CustomGpuTargetConfig(
      platform_name,
      device_kind,
      arch_name=arch_name,
      compute_capability=cc,
      core_count=core_count,
      shared_memory_per_core=shared_memory_per_core,
    )

  if device_kind in registry:
    return registry[device_kind]()
  return None


def is_gpu_device(platform: str | None = None) -> bool:
  if platform is None:
    platform = get_device_platform()
  return platform == "gpu"


registry: dict[str, Callable[[], GpuTargetConfig]] = {}


@jax_util.cache(trace_context_in_key=True)
def get_gpu_info() -> GpuTargetConfig:
  """Returns the GPU hardware info for the current device."""
  device = _get_device()
  device_kind = device.device_kind
  gpu_config = gpu_version_from_device_kind(device_kind)
  if gpu_config is not None:
    return gpu_config
  raise ValueError(f"Unsupported GPU device kind: {device_kind}")


def _get_device() -> Any:
  if abstract_device := mesh_lib.get_abstract_mesh().abstract_device:
    return abstract_device
  else:
    return pxla.get_default_device()


def get_device_kind() -> str:
  device = _get_device()
  return device.device_kind


_GPU_PLATFORMS = ("gpu", "rocm", "cuda")


def get_device_platform() -> str:
  device = _get_device()
  platform = device.platform
  return "gpu" if platform in _GPU_PLATFORMS else platform
