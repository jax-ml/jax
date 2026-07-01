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

from collections.abc import Callable
import dataclasses
import enum

from jax._src import mesh as mesh_lib
from jax._src import util as jax_util
from jax._src.interpreters import pxla
from jax._src.lib import xla_client as xc


class GpuVersion(enum.Enum):
  """GPU version"""

  # NVIDIA GPUs
  NV_7_5 = "7.5"
  NV_8_0 = "8.0"
  NV_8_6 = "8.6"
  NV_8_9 = "8.9"
  NV_9_0 = "9.0"
  NV_10_0 = "10.0"
  NV_10_3 = "10.3"
  NV_11_0 = "11.0"
  NV_12_0 = "12.0"
  NV_12_1 = "12.1"

  # AMD GPUs
  GFX908 = "gfx908"
  GFX90A = "gfx90a"
  GFX942 = "gfx942"
  GFX950 = "gfx950"
  GFX1030 = "gfx1030"
  GFX1100 = "gfx1100"
  GFX1101 = "gfx1101"
  GFX1103 = "gfx1103"
  GFX1150 = "gfx1150"
  GFX1151 = "gfx1151"
  GFX1200 = "gfx1200"
  GFX1201 = "gfx1201"

  def __str__(self) -> str:
    return self.value


def gpu_version_from_device_kind(device_kind: str) -> GpuVersion | None:
  if device_kind in GpuVersion:
    return GpuVersion(device_kind)
  return None


@dataclasses.dataclass(frozen=True, kw_only=True)
class GpuInfo:
  """GPU hardware information"""
  arch_name: str
  compute_capability: int
  gpu_version: GpuVersion | None

  def get_arch_major_minor(self) -> tuple[int, int]:
    return (
        self.compute_capability // 10, self.compute_capability % 10
    )


def is_gpu_device() -> bool:
  return _get_device().platform == "gpu"


registry: dict[str, Callable[[], GpuInfo]] = {}


def _get_gpu_info_impl(gpu_version: GpuVersion) -> GpuInfo:
  """Returns the GPU hardware info for the given its version.

  Args:
    gpu_version: The GPU version.
  """
  def get_cc(arch_name):
    try:
      return int(arch_name.replace(".", ""))
    except ValueError:
      return 0

  return GpuInfo(
      arch_name=gpu_version.value,
      compute_capability=get_cc(gpu_version.value),
      gpu_version=gpu_version,
  )


@jax_util.cache(trace_context_in_key=True)
def get_gpu_info() -> GpuInfo:
  """Returns the GPU hardware info for the current device."""
  device = _get_device()
  if device.platform != "gpu":
    raise RuntimeError(f"Can not get GPU info on the platform: {device.platform}")

  if isinstance(device, mesh_lib.AbstractDevice):
    device_kind = device.device_kind
    gpu_version = gpu_version_from_device_kind(device.device_kind)
    if gpu_version is not None:
      return _get_gpu_info_impl(gpu_version)

    if device_kind in registry:
      return registry[device_kind]()

    raise RuntimeError(f"Unsupported GPU device kind: {device_kind}")
  else:
    # Real device case.
    arch_name = str(device.compute_capability)
    if "cuda" in device.client.platform_version:
      compute_capability = int(arch_name.replace(".", ""))
    else:
      # Other cases, e.g. rocm.
      compute_capability = 0

    return GpuInfo(
      arch_name=arch_name,
      compute_capability=compute_capability,
      gpu_version=None,
    )


@jax_util.cache(trace_context_in_key=True)
def get_gpu_info_from_version(
    gpu_version: GpuVersion
) -> GpuInfo:
  """Returns the GPU hardware info for the given GPU version.

  Args:
    gpu_version: The GPU version.
  """
  return _get_gpu_info_impl(gpu_version)


def _get_device() -> mesh_lib.AbstractDevice | xc.Device:
  if abstract_device := mesh_lib.get_abstract_mesh().abstract_device:
    return abstract_device
  return pxla.get_default_device()


def get_device_kind() -> str:
  return _get_device().device_kind
