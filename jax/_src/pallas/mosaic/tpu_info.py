# Copyright 2025 The JAX Authors.
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

"""Exposes TPU hardware information."""

import dataclasses
import enum
from typing import Callable

from jax import numpy as jnp
from jax._src import dtypes
from jax._src import util as jax_util
from jax._src.pallas.mosaic import core


class ChipVersionBase:
  pass


class ChipVersion(ChipVersionBase, enum.Enum):
  TPU_V2 = "v2"
  TPU_V3 = "v3"
  TPU_V4I = "v4i"
  TPU_V4 = "v4"
  TPU_V5E = "v5e"
  TPU_V5P = "v5p"
  TPU_V6E = "v6e"
  TPU_7X = "7x"

  def __str__(self) -> str:
    return self.value


@dataclasses.dataclass(frozen=True, kw_only=True)
class SparseCoreInfo:
  """SparseCore-specific information."""

  num_cores: int
  num_subcores: int
  num_lanes: int
  dma_granule_size_bytes: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class TpuInfo:
  """TPU hardware information.

  Note that all information is per-TensorCore so you would need to multiply
  by `num_cores` to obtain the total for the chip.
  """

  chip_version: ChipVersionBase
  generation: int
  num_cores: int
  num_lanes: int
  num_sublanes: int
  mxu_column_size: int
  vmem_capacity_bytes: int
  cmem_capacity_bytes: int
  smem_capacity_bytes: int
  hbm_capacity_bytes: int
  mem_bw_bytes_per_second: int
  bf16_ops_per_second: int
  int8_ops_per_second: int
  fp8_ops_per_second: int
  int4_ops_per_second: int

  sparse_core: SparseCoreInfo | None = None

  @property
  def is_lite(self) -> bool:
    return self.chip_version in {
        ChipVersion.TPU_V4I,
        ChipVersion.TPU_V5E,
        ChipVersion.TPU_V6E,
    }

  @property
  def is_split_chip(self) -> bool:
    # Is this a multi-core chip being used in single-core mode?
    return self.num_cores == 1 and not self.is_lite

  def is_matmul_supported(
      self,
      lhs_dtype: jnp.dtype | str,
      rhs_dtype: jnp.dtype | str,
  ) -> bool:
    """Returns whether the given matmul input dtypes are supported on the chip."""
    lhs_dt = jnp.dtype(lhs_dtype) if isinstance(lhs_dtype, str) else lhs_dtype
    rhs_dt = jnp.dtype(rhs_dtype) if isinstance(rhs_dtype, str) else rhs_dtype

    F32 = jnp.float32
    BF16 = jnp.bfloat16
    S8 = jnp.int8
    U8 = jnp.uint8
    F8E4M3B11FNUZ = jnp.float8_e4m3b11fnuz
    F8E4M3FN = jnp.float8_e4m3fn
    F8E5M2 = jnp.float8_e5m2
    S4 = jnp.int4
    U4 = jnp.uint4

    match self.generation:
      case 2 | 3:
        return lhs_dt == rhs_dt == F32
      case 4:
        return lhs_dt in {F32, BF16} and rhs_dt in {F32, BF16, S8}
      case 5 | 6:
        return (
            (
                lhs_dt in {F32, BF16, F8E5M2, F8E4M3B11FNUZ}
                and rhs_dt in {F32, BF16, F8E5M2, F8E4M3B11FNUZ}
            )
            or (lhs_dt in {U8, S8} and rhs_dt in {U8, S8})
            or (lhs_dt in {U4, S4} and rhs_dt in {U4, S4})
        )
      case 7:
        return (lhs_dt in {F32, BF16} and rhs_dt in {F32, BF16}) or (
            lhs_dt in {F32, BF16, F8E5M2, F8E4M3FN}
            and rhs_dt in {F8E5M2, F8E4M3FN}
        )
      case _:
        return False

  def get_sublane_tiling(self, dtype: jnp.dtype) -> int:
    """Returns the sublane tiling for the given itemsize.

    Note that this is a heurustic and depends on the settings of the XLA flags.
    """
    bitwidth = dtypes.itemsize_bits(dtype)
    if self.generation < 7:
      # Caveat: before TPU7x, by default XLA does not use large 2nd minor tiling
      # but it can be enabled by setting the flag
      # xla_tpu_enable_large_2nd_minor_layout_for_x16.
      if bitwidth == 16 or bitwidth == 32:
        return self.num_sublanes
      else:
        # Large 2nd minor tiling is enabled for other types.
        return self.num_sublanes * (32 // bitwidth)
    # XLA allows large 2nd minor tiling by default starting with TPU7x.
    if self.generation == 7:
      return self.num_sublanes * (32 // bitwidth)
    raise NotImplementedError("TPU generation is not supported")


def is_tpu_device() -> bool:
  """Returns whether the current device is a TPU."""
  return core.get_device_kind() in {
      "TPU v2",
      "TPU v3",
      "TPU v4",
      "TPU v4 lite",
      "TPU v5e",
      "TPU v5 lite",
      "TPU v5",
      "TPU v5p",
      "TPU v6 lite",
      "TPU v6e",
      "TPU7x",
  }


registry: dict[str, Callable[[], TpuInfo]] = {}


@jax_util.cache(trace_context_in_key=True)
def get_tpu_info() -> TpuInfo:
  """Returns the TPU hardware information for the current device.

  Note that all information is *per-TensorCore* so you would need to multiply by
  `num_cores` to obtain the total for the chip.

  Returns:
    A TpuInfo object containing the hardware information for the current device.
  """
  device_kind = core.get_device_kind()

  # Common parameters for all TensorCores
  NUM_LANES = 128
  NUM_SUBLANES = 8
  MXU_COLUMN_SIZE_GEN_LT_6 = 128
  MXU_COLUMN_SIZE_GEN_GE_6 = 256

  match device_kind:
    case "TPU v2":  # 2 TensorCores per chip
      num_chip_cores = 2
      return TpuInfo(
          chip_version=ChipVersion.TPU_V2,
          generation=2,
          num_cores=core.get_num_device_cores(),
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          vmem_capacity_bytes=16 * 1024 * 1024,  # 16 MiB per core
          cmem_capacity_bytes=0,
          smem_capacity_bytes=16 * 1024,  # 16 KiB per core
          hbm_capacity_bytes=int(16_000_000_000 // num_chip_cores),
          mem_bw_bytes_per_second=int(7.16e11 // num_chip_cores),
          bf16_ops_per_second=int(4.6e13 // num_chip_cores),
          int8_ops_per_second=0,  # Not Available
          fp8_ops_per_second=0,  # Not Available
          int4_ops_per_second=0,  # Not Available
      )
    case "TPU v3":  # 2 TensorCores per chip
      num_chip_cores = 2
      return TpuInfo(
          chip_version=ChipVersion.TPU_V3,
          generation=3,
          num_cores=core.get_num_device_cores(),
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          vmem_capacity_bytes=16 * 1024 * 1024,  # 16 MiB per core
          cmem_capacity_bytes=0,
          smem_capacity_bytes=16 * 1024,  # 16 KiB per core
          hbm_capacity_bytes=34_400_000_000 // num_chip_cores,
          mem_bw_bytes_per_second=int(8.25e11 // num_chip_cores),
          bf16_ops_per_second=int(1.40e14 // num_chip_cores),
          int8_ops_per_second=0,  # Not Available
          fp8_ops_per_second=0,  # Not Available
          int4_ops_per_second=0,  # Not Available
      )
    case "TPU v4 lite":  # 1 TensorCore per chip
      return TpuInfo(
          chip_version=ChipVersion.TPU_V4I,
          generation=4,
          num_cores=core.get_num_device_cores(),
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          vmem_capacity_bytes=16 * 1024 * 1024,  # 16 MiB per core
          cmem_capacity_bytes=134_000_000,
          smem_capacity_bytes=1024 * 1024,  # 1 MiB per core
          hbm_capacity_bytes=8_590_000_000,
          mem_bw_bytes_per_second=int(6.14e11),
          bf16_ops_per_second=int(1.37e14),
          int8_ops_per_second=0,  # Not Available
          fp8_ops_per_second=0,  # Not Available
          int4_ops_per_second=0,  # Not Available
      )
    case "TPU v4":  # 2 TensorCores per chip
      num_chip_cores = 2
      return TpuInfo(
          chip_version=ChipVersion.TPU_V4,
          generation=4,
          num_cores=core.get_num_device_cores(),
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          vmem_capacity_bytes=16 * 1024 * 1024,  # 16 MiB per core
          cmem_capacity_bytes=134_000_000 // num_chip_cores,
          smem_capacity_bytes=1024 * 1024,  # 1 MiB per core
          hbm_capacity_bytes=34_400_000_000 // num_chip_cores,
          mem_bw_bytes_per_second=int(1.23e12 // num_chip_cores),
          bf16_ops_per_second=int(2.75e14 // num_chip_cores),
          int8_ops_per_second=0,  # Not Available
          fp8_ops_per_second=0,  # Not Available
          int4_ops_per_second=0,  # Not Available
      )
    case "TPU v5 lite" | "TPU v5e":  # 1 TensorCore per chip
      return TpuInfo(
          chip_version=ChipVersion.TPU_V5E,
          generation=5,
          num_cores=core.get_num_device_cores(),
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          vmem_capacity_bytes=128 * 1024 * 1024,  # 128 MiB per core
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 1 MiB per core
          hbm_capacity_bytes=17_200_000_000,
          mem_bw_bytes_per_second=int(8.20e11),
          bf16_ops_per_second=int(1.97e14),
          int8_ops_per_second=int(3.94e14),
          fp8_ops_per_second=0,  # Not Available
          int4_ops_per_second=int(7.88e14),
      )
    case "TPU v5" | "TPU v5p":  # 2 TensorCores per chip
      num_chip_cores = 2
      return TpuInfo(
          chip_version=ChipVersion.TPU_V5P,
          generation=5,
          num_cores=core.get_num_device_cores(),
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          vmem_capacity_bytes=64 * 1024 * 1024,  # 64 MiB per core
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 1 MiB per core
          hbm_capacity_bytes=103_000_000_000 // num_chip_cores,
          mem_bw_bytes_per_second=int(2.46e12 // num_chip_cores),
          bf16_ops_per_second=int(4.59e14 // num_chip_cores),
          int8_ops_per_second=int(9.18e14 // num_chip_cores),
          fp8_ops_per_second=0,  # Not Available
          int4_ops_per_second=int(1.84e15 // num_chip_cores),
          sparse_core=SparseCoreInfo(
              num_cores=4,
              num_subcores=16,
              num_lanes=8,
              dma_granule_size_bytes=32,
          ),
      )
    case "TPU v6 lite" | "TPU v6e":  # 1 TensorCore per chip
      return TpuInfo(
          chip_version=ChipVersion.TPU_V6E,
          generation=6,
          num_cores=core.get_num_device_cores(),
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_GE_6,
          vmem_capacity_bytes=128 * 1024 * 1024,  # 128 MiB per core
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 1 MiB per core
          hbm_capacity_bytes=34_400_000_000,
          mem_bw_bytes_per_second=int(1.64e12),
          bf16_ops_per_second=int(9.20e14),
          int8_ops_per_second=int(1.84e15),
          fp8_ops_per_second=int(9.20e14),
          int4_ops_per_second=int(3.68e15),
          sparse_core=SparseCoreInfo(
              num_cores=2,
              num_subcores=16,
              num_lanes=8,
              dma_granule_size_bytes=32,
          ),
      )
    case "TPU7x":
      num_cores = core.get_num_device_cores()
      num_chip_cores = 2
      return TpuInfo(
          chip_version=ChipVersion.TPU_7X,
          generation=7,
          num_cores=num_cores,
          num_lanes=128,
          num_sublanes=8,
          mxu_column_size=256,
          vmem_capacity_bytes=64 * 1024 * 1024,  # 64 MiB per core
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 1 MiB per core
          hbm_capacity_bytes=206_000_000_000 // num_chip_cores,
          mem_bw_bytes_per_second=int(7.40e12 // num_chip_cores),
          bf16_ops_per_second=int(2.31e15 // num_chip_cores),
          int8_ops_per_second=0,  # Not Available
          fp8_ops_per_second=int(4.60e15 // num_chip_cores),
          int4_ops_per_second=0,  # Not Available
          sparse_core=SparseCoreInfo(
              num_cores=2,
              num_subcores=16,
              num_lanes=16,
              dma_granule_size_bytes=64,
          ),
      )
    case _ as d:
      if d in registry:
        return registry[d]()
      raise ValueError(f"Unsupported TPU device kind: {device_kind}")
