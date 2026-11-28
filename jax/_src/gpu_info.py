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
import functools

from jax._src import mesh as mesh_lib
from jax._src import util as jax_util
from jax._src.interpreters import pxla
from jax._src.lib import _gpu_spec
from jax._src.lib import xla_client as xc


@dataclasses.dataclass(frozen=True, kw_only=True)
class GpuInfo:
  platform_name: str
  device_description_str: str
  arch_name: str
  compute_capability: int
  core_count: int
  smem_capacity_bytes: int


@functools.cache
def _specs_by_device_kind() -> dict[str, GpuInfo]:
  if _gpu_spec is None:
    return {}
  aliases: dict[str, GpuInfo] = {}
  canonical: dict[str, GpuInfo] = {}
  for model in _gpu_spec.GpuModel:
    spec = _gpu_spec.get_gpu_spec(model)
    info = GpuInfo(
        platform_name=spec.platform_name,
        device_description_str=spec.device_description_str,
        arch_name=spec.arch_name,
        compute_capability=spec.compute_capability,
        core_count=spec.core_count,
        smem_capacity_bytes=spec.smem_capacity_bytes,
    )
    # Ignore generic bringup names reported for unrecognized PCI IDs.
    for kind in spec.device_descriptions:
      if kind != "NVIDIA Graphics Device":
        aliases[kind] = info
    canonical[spec.device_description_str] = info
  # A spec's own description wins over an alias registered for another spec.
  return aliases | canonical


def _local_gpu_info(device: xc.Device) -> GpuInfo:
  arch_name = device.compute_capability
  if arch_name.startswith("gfx"):
    platform_name, cc = "ROCM", 0
  elif "." in arch_name:
    major, minor = arch_name.split(".", 1)
    platform_name, cc = "CUDA", int(major) * 10 + int(minor)
  else:
    raise ValueError(f"Unsupported GPU compute capability: {arch_name!r}")
  return GpuInfo(
      platform_name=platform_name,
      device_description_str=device.device_kind,
      arch_name=arch_name,
      compute_capability=cc,
      core_count=device.core_count,
      smem_capacity_bytes=device.shared_memory_per_block_optin,
  )


_GPU_PLATFORMS = ("gpu", "rocm", "cuda")


def is_gpu_device() -> bool:
  return _get_device().platform in _GPU_PLATFORMS


@jax_util.cache(trace_context_in_key=True)
def get_gpu_info() -> GpuInfo:
  """Returns the GPU hardware info for the current device."""
  device = _get_device()
  if device.platform not in _GPU_PLATFORMS:
    raise ValueError(f"Unsupported GPU device kind: {device.device_kind}")
  if (
      gpu_info := _specs_by_device_kind().get(device.device_kind)
  ) is not None:
    return gpu_info
  if isinstance(device, mesh_lib.AbstractDevice):
    default_device = pxla.get_default_device()
    if (
        default_device.platform not in _GPU_PLATFORMS
        or default_device.device_kind != device.device_kind
    ):
      raise ValueError(f"Unsupported GPU device kind: {device.device_kind}")
    device = default_device
  return _local_gpu_info(device)


def _get_device() -> xc.Device | mesh_lib.AbstractDevice:
  if abstract_device := mesh_lib.get_abstract_mesh().abstract_device:
    return abstract_device
  return pxla.get_default_device()


def get_device_kind() -> str:
  return _get_device().device_kind
