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


class GpuVersion(enum.Enum):
  """NVidia GPU version"""

  A10 = "NVIDIA A10"
  A30 = "NVIDIA A30"
  A100 = "NVIDIA A100"
  H100 = "NVIDIA H100"
  H200 = "NVIDIA H200"
  GH200 = "NVIDIA GH200"
  B200 = "NVIDIA B200"
  GB200 = "NVIDIA GB200"
  B300 = "NVIDIA B300"
  GB300 = "NVIDIA GB300"
  GB10 = "NVIDIA GB10"
  L4 = "NVIDIA L4"
  L40 = "NVIDIA L40"
  T4 = "Tesla T4"
  RTX_4090 = "NVIDIA GeForce RTX 4090"
  RTX_PRO_4500 = "NVIDIA RTX PRO 4500"
  RTX_PRO_5000 = "NVIDIA RTX PRO 5000"
  RTX_PRO_6000 = "NVIDIA RTX PRO 6000"
  THOR = "NVIDIA Thor"

  def __str__(self) -> str:
    return self.value


_GPU_VERSION_RE = re.compile(r"\b(" + "|".join(e.value for e in GpuVersion) + r")\b")


def gpu_version_from_device_kind(device_kind: str) -> GpuVersion | None:
  if m := _GPU_VERSION_RE.match(device_kind):
    return GpuVersion(m.group())
  return None


@dataclasses.dataclass(frozen=True, kw_only=True)
class GpuInfo:
  """GPU hardware information"""
  gpu_version: GpuVersion
  arch_name: str
  compute_capability: int


def is_gpu_device() -> bool:
  return gpu_version_from_device_kind(get_device_kind()) is not None


registry: dict[str, Callable[[], GpuInfo]] = {}


def _get_gpu_info_impl(gpu_version: GpuVersion) -> GpuInfo:
  """Returns the GPU hardware info for the given its version.

  Args:
    gpu_version: The GPU version.
  """
  # https://developer.nvidia.com/cuda/gpus
  match gpu_version:
    case GpuVersion.T4:
      return GpuInfo(
          gpu_version=gpu_version,
          arch_name="7.5",
          compute_capability=75,
      )
    case GpuVersion.A100 | GpuVersion.A30:
      return GpuInfo(
          gpu_version=gpu_version,
          arch_name="8.0",
          compute_capability=80,
      )
    case GpuVersion.A10:
      return GpuInfo(
          gpu_version=gpu_version,
          arch_name="8.6",
          compute_capability=86,
      )
    case GpuVersion.L4 | GpuVersion.L40 | GpuVersion.RTX_4090:
      return GpuInfo(
          gpu_version=gpu_version,
          arch_name="8.9",
          compute_capability=89,
      )
    case GpuVersion.H100 | GpuVersion.H200 | GpuVersion.GH200:
      return GpuInfo(
          gpu_version=gpu_version,
          arch_name="9.0",
          compute_capability=90,
      )
    case GpuVersion.B200 | GpuVersion.GB200:
      return GpuInfo(
          gpu_version=gpu_version,
          arch_name="10.0",
          compute_capability=100,
      )
    case GpuVersion.B300 | GpuVersion.GB300:
      return GpuInfo(
          gpu_version=gpu_version,
          arch_name="10.3",
          compute_capability=103,
      )
    case GpuVersion.RTX_PRO_4500 | GpuVersion.RTX_PRO_5000 | GpuVersion.RTX_PRO_6000:
      return GpuInfo(
          gpu_version=gpu_version,
          arch_name="12.0",
          compute_capability=120,
      )
    case GpuVersion.GB10:
      return GpuInfo(
          gpu_version=gpu_version,
          arch_name="12.1",
          compute_capability=121,
      )
    case GpuVersion.THOR:
      return GpuInfo(
          gpu_version=gpu_version,
          arch_name="11.0",
          compute_capability=110,
      )
    case _:
      raise ValueError(f"Unsupported GPU version: {gpu_version}")


@jax_util.cache(trace_context_in_key=True)
def get_gpu_info() -> GpuInfo:
  """Returns the GPU hardware info for the current device."""
  device_kind = get_device_kind()
  gpu_version = gpu_version_from_device_kind(device_kind)
  if gpu_version is None:
    if device_kind in registry:
      return registry[device_kind]()
    raise ValueError(f"Unsupported GPU device kind: {device_kind}")
  return _get_gpu_info_impl(gpu_version)


@jax_util.cache(trace_context_in_key=True)
def get_gpu_info_from_version(
    gpu_version: GpuVersion
) -> GpuInfo:
  """Returns the GPU hardware info for the given GPU version.

  Args:
    gpu_version: The GPU version.
  """
  return _get_gpu_info_impl(gpu_version)


def get_device_kind() -> str:
  if abstract_device := mesh_lib.get_abstract_mesh().abstract_device:
    return abstract_device.device_kind
  return pxla.get_default_device().device_kind
