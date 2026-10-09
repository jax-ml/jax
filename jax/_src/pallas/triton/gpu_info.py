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

try:
  from jax._src.lib._gpu_spec import (
      GpuTargetConfig as _GpuTargetConfig,
      GpuModel,
      get_gpu_spec,
  )
except ModuleNotFoundError:
  # Fallback for older jaxlib
  # TODO: remove this whole except ModuleNotFoundError branch
  # once min jaxlib version >= 0.12.0
  from enum import Enum

  @dataclasses.dataclass
  class _Config:
    platform_name: str
    device_description_str: str
    arch_name: str
    compute_capability: int
    core_count: int
    smem_capacity_bytes: int

  class GpuModel(Enum):
    # This is a fallback structure keeping BC for older jaxlib
    # Do not add any new items in this Enum.
    # Instead, use either registry variable below or add new hardware to XLA
    A100_PCIE_80 = 0
    A100_SXM_40 = 1
    A100_SXM_80 = 2
    A6000 = 3
    B200 = 4
    B200_MIG = 5
    B300 = 6
    BMG_G21 = 7
    H100_PCIE = 8
    H100_SXM = 9
    H100_SXM_MIG = 10
    H200 = 11
    MI200 = 12
    MI300 = 13
    MI350 = 14
    MI450 = 15
    P100 = 16
    PVC = 17
    V100 = 18
    GB200 = 19
    GB300 = 20
    RTX6000PRO = 21

  # This is a workaround to please pyrefly
  _GpuTargetConfig = _Config

  def get_gpu_spec(gpu_model):
    if gpu_model == GpuModel.A100_PCIE_80:
      return _Config(
          "CUDA", "NVIDIA A100 80GB PCIe", "8.0", 80, 108, 167936
      )
    elif gpu_model == GpuModel.A100_SXM_40:
      return _Config(
          "CUDA", "NVIDIA A100-SXM4-40GB", "8.0", 80, 108, 167936
      )
    elif gpu_model == GpuModel.A100_SXM_80:
      return _Config(
          "CUDA", "NVIDIA A100-SXM4-80GB", "8.0", 80, 108, 167936
      )
    elif gpu_model == GpuModel.A6000:
      return _Config(
          "CUDA", "NVIDIA RTX A6000", "8.6", 86, 84, 102400
      )
    elif gpu_model == GpuModel.B200:
      return _Config(
          "CUDA", "NVIDIA B200", "10.0a", 100, 148, 233472
      )
    elif gpu_model == GpuModel.B200_MIG:
      return _Config(
          "CUDA", "NVIDIA B200 MIG 1g.23gb", "10.0a", 100, 18, 233472
      )
    elif gpu_model == GpuModel.B300:
      return _Config(
          "CUDA", "NVIDIA B300", "10.3a", 103, 158, 233472
      )
    elif gpu_model == GpuModel.BMG_G21:
      return _Config(
          "SYCL", "Intel(R) Arc(TM) B580 Graphics", "", 0, 20, 131072
      )
    elif gpu_model == GpuModel.H100_PCIE:
      return _Config(
          "CUDA", "NVIDIA H100 PCIe", "9.0a", 90, 114, 233472
      )
    elif gpu_model == GpuModel.H100_SXM:
      return _Config(
          "CUDA", "NVIDIA H100 80GB HBM3", "9.0a", 90, 132, 233472
      )
    elif gpu_model == GpuModel.H100_SXM_MIG:
      return _Config(
          "CUDA", "NVIDIA H100 80GB HBM3 MIG 1g.10gb", "9.0a", 90, 16, 233472
      )
    elif gpu_model == GpuModel.H200:
      return _Config(
          "CUDA", "NVIDIA H200 141GB HBM3e", "9.0a", 90, 132, 233472
      )
    elif gpu_model == GpuModel.MI200:
      return _Config(
          "ROCM", "MI250", "gfx90a:sramecc+:xnack-", 0, 110, 65536
      )
    elif gpu_model == GpuModel.MI300:
      return _Config(
          "ROCM", "MI300X", "gfx942:sramecc+:xnack-", 0, 304, 65536
      )
    elif gpu_model == GpuModel.MI350:
      return _Config(
          "ROCM", "MI350X", "gfx950:sramecc+:xnack-", 0, 256, 163840
      )
    elif gpu_model == GpuModel.MI450:
      return _Config(
          "ROCM", "GFX1250", "gfx1250", 0, 64, 65536
      )
    elif gpu_model == GpuModel.P100:
      return _Config(
          "CUDA", "Tesla P100-SXM2-16GB", "6.0", 60, 56, 65536
      )
    elif gpu_model == GpuModel.PVC:
      return _Config(
          "SYCL", "Intel(R) Data Center GPU Max 1100", "", 0, 56, 131072
      )
    elif gpu_model == GpuModel.V100:
      return _Config(
          "CUDA", "Tesla V100-SXM2-16GB", "7.0", 70, 80, 98304
      )
    elif gpu_model == GpuModel.GB200:
      return _Config(
          "CUDA", "NVIDIA GB200", "10.0a", 100, 152, 233472
      )
    elif gpu_model == GpuModel.GB300:
      return _Config(
          "CUDA", "NVIDIA GB300", "10.3a", 103, 152, 233472
      )
    elif gpu_model == GpuModel.RTX6000PRO:
      return _Config(
          "CUDA", "NVIDIA RTX PRO 6000 Blackwell", "12.0", 120, 188, 131072
      )
    else:
      raise RuntimeError(f"Unknown gpu model: {gpu_model}")


@dataclasses.dataclass
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

  if device_kind in registry:
    return registry[device_kind]()
  return None


def is_gpu_device() -> bool:
  return get_device_platform() == "gpu"


registry: dict[str, Callable[[], GpuTargetConfig]] = {
  "Tesla T4": lambda: CustomGpuTargetConfig("CUDA", "Tesla T4", "7.5", 75, 0, 0),
  "NVIDIA A30": lambda: CustomGpuTargetConfig("CUDA", "NVIDIA A30", "8.0", 80, 0, 0),
  "NVIDIA A10": lambda: CustomGpuTargetConfig("CUDA", "NVIDIA A10", "8.6", 86, 0, 0),
  "NVIDIA L4": lambda: CustomGpuTargetConfig("CUDA", "NVIDIA L4", "8.9", 89, 0, 0),
  "NVIDIA L40": lambda: CustomGpuTargetConfig("CUDA", "NVIDIA L40", "8.9", 89, 0, 0),
  "NVIDIA GeForce RTX 4090": lambda: CustomGpuTargetConfig("CUDA", "NVIDIA GeForce RTX 4090", "8.9", 89, 0, 0),

  "NVIDIA GH200": lambda: CustomGpuTargetConfig("CUDA", "NVIDIA GH200", "9.0", 90, 0, 0),

  "NVIDIA RTX PRO 4500 Blackwell": lambda: CustomGpuTargetConfig("CUDA", "NVIDIA RTX PRO 4500 Blackwell", "12.0", 120, 0, 0),
  "NVIDIA RTX PRO 5000 Blackwell": lambda: CustomGpuTargetConfig("CUDA", "NVIDIA RTX PRO 5000 Blackwell", "12.0", 120, 0, 0),

  "NVIDIA GB10": lambda: CustomGpuTargetConfig("CUDA", "NVIDIA GB10", "12.1", 121, 0, 0),
  "NVIDIA Thor": lambda: CustomGpuTargetConfig("CUDA", "NVIDIA Thor", "11.0", 110, 0, 0),
  "NVIDIA VR200": lambda: CustomGpuTargetConfig("CUDA", "NVIDIA VR200", "10.7", 107, 0, 0),
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
