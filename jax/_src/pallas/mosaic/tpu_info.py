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
from typing import Callable, cast

from jax import numpy as jnp
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import mesh as mesh_lib
from jax._src import util as jax_util
from jax._src.interpreters import pxla
from jax._src.pallas import utils as pallas_utils


class ChipVersionBase:
  pass


class ChipVersion(ChipVersionBase, enum.Enum):
  """TPU chip version.

  The following table summarizes the differences between TPU versions:

  +---------+-------------------------------+-----------+------------------+
  | Version | Physical TensorCores per chip | Lite chip | Megacore support |
  +=========+===============================+===========+==================+
  | v2      | 2                             | No        | No               |
  +---------+-------------------------------+-----------+------------------+
  | v3      | 2                             | No        | No               |
  +---------+-------------------------------+-----------+------------------+
  | v4i     | 1                             | Yes       | No               |
  +---------+-------------------------------+-----------+------------------+
  | v4      | 2                             | No        | Yes              |
  +---------+-------------------------------+-----------+------------------+
  | v5e     | 1                             | Yes       | No               |
  +---------+-------------------------------+-----------+------------------+
  | v5p     | 2                             | No        | Yes              |
  +---------+-------------------------------+-----------+------------------+
  | v6e     | 1                             | Yes       | No               |
  +---------+-------------------------------+-----------+------------------+
  | 7       | 2                             | No        | No               |
  +---------+-------------------------------+-----------+------------------+
  | 7x      | 2                             | No        | No               |
  +---------+-------------------------------+-----------+------------------+
  """

  TPU_V2 = "v2"
  TPU_V3 = "v3"
  TPU_V4I = "v4i"
  TPU_V4 = "v4"
  TPU_V5E = "v5e"
  TPU_V5P = "v5p"
  TPU_V6E = "v6e"
  TPU_7 = "7"
  TPU_7X = "7x"

  def __str__(self) -> str:
    return self.value

  @property
  def num_physical_tensor_cores_per_chip(self) -> int:
    match self:
      case (
          ChipVersion.TPU_V2
          | ChipVersion.TPU_V3
          | ChipVersion.TPU_V4
          | ChipVersion.TPU_V5P
          | ChipVersion.TPU_7
          | ChipVersion.TPU_7X
      ):
        return 2
      case ChipVersion.TPU_V4I | ChipVersion.TPU_V5E | ChipVersion.TPU_V6E:
        return 1

  @property
  def supports_megacore(self) -> bool:
    match self:
      case ChipVersion.TPU_V4 | ChipVersion.TPU_V5P:
        return True
      case _:
        return False

  @property
  def is_lite(self) -> bool:
    match self:
      case ChipVersion.TPU_V4I | ChipVersion.TPU_V5E | ChipVersion.TPU_V6E:
        return True
      case _:
        return False


def chip_version_from_device_kind(device_kind: str) -> ChipVersion | None:
  match device_kind:
    case "TPU v2":
      return ChipVersion.TPU_V2
    case "TPU v3":
      return ChipVersion.TPU_V3
    case "TPU v4":
      return ChipVersion.TPU_V4
    case "TPU v4 lite":
      return ChipVersion.TPU_V4I
    case "TPU v5e" | "TPU v5 lite":
      return ChipVersion.TPU_V5E
    case "TPU v5" | "TPU v5p":
      return ChipVersion.TPU_V5P
    case "TPU v6e" | "TPU v6 lite":
      return ChipVersion.TPU_V6E
    case "TPU7":
      return ChipVersion.TPU_7
    case "TPU7x":
      return ChipVersion.TPU_7X
    case _:
      return None


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
    return cast(ChipVersion, self.chip_version).is_lite

  @property
  def is_split_chip(self) -> bool:
    """Returns True if the chip is a multi-core chip being used in single-core mode.

    Some TPU generations (e.g. v4, v5p) have multiple TensorCores per chip.
    These chips can be used in two modes:
    1. "Megacore" mode, where the cores are combined into a single logical
    device (if supported).
    2. "Split" mode, where each core is treated as an independent logical
    device.

    This property returns True if the chip is in "split" mode (case 2).
    """
    return self.num_cores == 1 and (
        cast(ChipVersion, self.chip_version).num_physical_tensor_cores_per_chip
        > 1
    )

  @property
  def is_megacore(self) -> bool:
    """Returns True if the chip is configured in Megacore mode.

    Megacore mode means the two physical TensorCores are combined into a single
    logical device.
    """
    return self.num_cores > 1

  def is_matmul_supported(
      self,
      lhs_dtype: dtypes.DTypeLike,
      rhs_dtype: dtypes.DTypeLike,
  ) -> bool:
    """Returns whether the chip natively supports matmul on the given input dtypes (no casting needed)."""
    lhs_dtype = dtypes.dtype(lhs_dtype)
    rhs_dtype = dtypes.dtype(rhs_dtype)

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
        return lhs_dtype == rhs_dtype == F32
      case 4:
        return lhs_dtype in (F32, BF16) and rhs_dtype in (F32, BF16, S8)
      case 5 | 6:
        return (
            (
                lhs_dtype in (F32, BF16, F8E5M2, F8E4M3B11FNUZ)
                and rhs_dtype in (F32, BF16, F8E5M2, F8E4M3B11FNUZ)
            )
            or (lhs_dtype in (U8, S8) and rhs_dtype in (U8, S8))
            or (lhs_dtype in (U4, S4) and rhs_dtype in (U4, S4))
        )
      case 7:
        return (lhs_dtype in (F32, BF16) and rhs_dtype in (F32, BF16)) or (
            lhs_dtype in (F32, BF16, F8E5M2, F8E4M3FN)
            and rhs_dtype in (F8E5M2, F8E4M3FN)
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
  return chip_version_from_device_kind(get_device_kind()) is not None


registry: dict[str, Callable[[], TpuInfo]] = {}


def _get_tpu_info_impl(chip_version: ChipVersion, num_cores: int) -> TpuInfo:
  """Returns the TPU hardware info for the given chip version and core count.

  Note that all information is *per-TensorCore* so you would need to multiply by
  `num_cores` to obtain the total for the chip.

  Args:
    chip_version: The TPU chip version.
    num_cores: The number of TensorCores per chip for this configuration. This
      is influenced by the TPU version and whether Megacore is enabled.
  """
  # Common parameters for all TensorCores
  NUM_LANES = 128
  NUM_SUBLANES = 8
  MXU_COLUMN_SIZE_GEN_LT_6 = 128
  MXU_COLUMN_SIZE_GEN_GE_6 = 256
  tensor_cores_per_chip = chip_version.num_physical_tensor_cores_per_chip
  match chip_version:
    case ChipVersion.TPU_V2:
      return TpuInfo(
          chip_version=chip_version,
          generation=2,
          num_cores=num_cores,
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          vmem_capacity_bytes=16 * 1024 * 1024,  # 16 MiB per core
          cmem_capacity_bytes=0,
          smem_capacity_bytes=16 * 1024,  # 16 KiB per core
          hbm_capacity_bytes=int(16_000_000_000 // tensor_cores_per_chip),
          mem_bw_bytes_per_second=int(7.16e11 // tensor_cores_per_chip),
          bf16_ops_per_second=int(4.6e13 // tensor_cores_per_chip),
          int8_ops_per_second=0,  # Not Available
          fp8_ops_per_second=0,  # Not Available
          int4_ops_per_second=0,  # Not Available
      )
    case ChipVersion.TPU_V3:
      return TpuInfo(
          chip_version=chip_version,
          generation=3,
          num_cores=num_cores,
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          vmem_capacity_bytes=16 * 1024 * 1024,  # 16 MiB per core
          cmem_capacity_bytes=0,
          smem_capacity_bytes=16 * 1024,  # 16 KiB per core
          hbm_capacity_bytes=34_400_000_000 // tensor_cores_per_chip,
          mem_bw_bytes_per_second=int(8.25e11 // tensor_cores_per_chip),
          bf16_ops_per_second=int(1.40e14 // tensor_cores_per_chip),
          int8_ops_per_second=0,  # Not Available
          fp8_ops_per_second=0,  # Not Available
          int4_ops_per_second=0,  # Not Available
      )
    case ChipVersion.TPU_V4I:
      return TpuInfo(
          chip_version=chip_version,
          generation=4,
          num_cores=num_cores,
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
    case ChipVersion.TPU_V4:
      return TpuInfo(
          chip_version=chip_version,
          generation=4,
          num_cores=num_cores,
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          vmem_capacity_bytes=16 * 1024 * 1024,  # 16 MiB per core
          cmem_capacity_bytes=134_000_000 // tensor_cores_per_chip,
          smem_capacity_bytes=1024 * 1024,  # 1 MiB per core
          hbm_capacity_bytes=34_400_000_000 // tensor_cores_per_chip,
          mem_bw_bytes_per_second=int(1.23e12 // tensor_cores_per_chip),
          bf16_ops_per_second=int(2.75e14 // tensor_cores_per_chip),
          int8_ops_per_second=0,  # Not Available
          fp8_ops_per_second=0,  # Not Available
          int4_ops_per_second=0,  # Not Available
      )
    case ChipVersion.TPU_V5E:
      return TpuInfo(
          chip_version=chip_version,
          generation=5,
          num_cores=num_cores,
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
    case ChipVersion.TPU_V5P:
      return TpuInfo(
          chip_version=chip_version,
          generation=5,
          num_cores=num_cores,
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          vmem_capacity_bytes=64 * 1024 * 1024,  # 64 MiB per core
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 1 MiB per core
          hbm_capacity_bytes=103_000_000_000 // tensor_cores_per_chip,
          mem_bw_bytes_per_second=int(2.46e12 // tensor_cores_per_chip),
          bf16_ops_per_second=int(4.59e14 // tensor_cores_per_chip),
          int8_ops_per_second=int(9.18e14 // tensor_cores_per_chip),
          fp8_ops_per_second=0,  # Not Available
          int4_ops_per_second=int(1.84e15 // tensor_cores_per_chip),
          sparse_core=SparseCoreInfo(
              num_cores=4,
              num_subcores=16,
              num_lanes=8,
              dma_granule_size_bytes=32,
          ),
      )
    case ChipVersion.TPU_V6E:
      return TpuInfo(
          chip_version=chip_version,
          generation=6,
          num_cores=num_cores,
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
    case ChipVersion.TPU_7 | ChipVersion.TPU_7X:
      return TpuInfo(
          chip_version=chip_version,
          generation=7,
          num_cores=num_cores,
          num_lanes=128,
          num_sublanes=8,
          mxu_column_size=256,
          vmem_capacity_bytes=64 * 1024 * 1024,  # 64 MiB per core
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 1 MiB per core
          hbm_capacity_bytes=206_000_000_000 // tensor_cores_per_chip,
          mem_bw_bytes_per_second=int(7.40e12 // tensor_cores_per_chip),
          bf16_ops_per_second=int(2.31e15 // tensor_cores_per_chip),
          int8_ops_per_second=0,  # Not Available
          fp8_ops_per_second=int(4.60e15 // tensor_cores_per_chip),
          int4_ops_per_second=0,  # Not Available
          sparse_core=SparseCoreInfo(
              num_cores=2,
              num_subcores=16,
              num_lanes=16,
              dma_granule_size_bytes=64,
          ),
      )
    case _:
      raise ValueError(f"Unsupported TPU chip version: {chip_version}")


@jax_util.cache(trace_context_in_key=True)
def get_tpu_info() -> TpuInfo:
  """Returns the TPU hardware info for the current device.

  Note that all information is *per-TensorCore* so you would need to multiply by
  `num_cores` to obtain the total for the chip.
  """
  device_kind = get_device_kind()
  chip_version = chip_version_from_device_kind(device_kind)
  if chip_version is None:
    if device_kind in registry:
      return registry[device_kind]()
    raise ValueError(f"Unsupported TPU device kind: {device_kind}")
  return _get_tpu_info_impl(chip_version, get_num_device_cores())


@jax_util.cache(trace_context_in_key=True)
def get_tpu_info_for_chip(
    chip_version: ChipVersion, num_tensor_cores_per_logical_device: int
) -> TpuInfo:
  """Returns the TPU hardware info for the given TPU chip version.

  Note that all information is *per-TensorCore* so you would need to multiply by
  `num_tensor_cores_per_logical_device` to obtain the total for the chip.

  Args:
    chip_version: The TPU chip version.
    num_tensor_cores_per_logical_device: The number of TensorCores per logical
      device in the requested configuration. Should be 1 for single-core chips
      (TPU_V4I, TPU_V5E, TPU_V6E). For dual-core chips that support Megacore
      (TPU_V4, TPU_V5P), this can be 2 (Megacore mode) or 1 (split mode). For
      dual-core chips that do not support Megacore (TPU_V2, TPU_V3, TPU_7X),
      this must be 1.
  """
  if (
      chip_version.is_lite
      or chip_version
      in {
          ChipVersion.TPU_V2,
          ChipVersion.TPU_V3,
          ChipVersion.TPU_7,
          ChipVersion.TPU_7X,
      }
  ) and num_tensor_cores_per_logical_device != 1:
    raise ValueError(
        "Lite chips and dual-core chips that do not support Megacore must "
        "have num_tensor_cores_per_logical_device=1, but got"
        f" {num_tensor_cores_per_logical_device}."
    )

  return _get_tpu_info_impl(chip_version, num_tensor_cores_per_logical_device)


# TODO(sharadmv): Generalize Tiling to capture the various options
# (compact 2nd minor, large 2nd minor, regular tiling)
class Tiling(enum.Enum):
  COMPACT = enum.auto()
  SPARSE_CORE = enum.auto()

  @property
  def shape(self) -> tuple[int, ...]:
    # TODO(slebedev): Use ``get_tpu_info()`` instead of hardcoding the values.
    match self:
      case Tiling.COMPACT:
        return (8, 128)
      case Tiling.SPARSE_CORE:
        return (8,)


def _get_tiling_factor(src: int, max_tiling: int, packing: int) -> int:
  # This roughly mirrors ``getTilingFactor`` in infer-memref-layout.
  tpu_generation = get_tpu_info().generation
  tiling = (1 + int(tpu_generation < 4)) * packing
  while tiling < min(src, max_tiling):
    tiling *= 2
  return tiling


def infer_tiling(
    ty: jax_core.AbstractValue, tiling: Tiling | None = None
) -> tuple[int | None, ...]:
  """Compute a tiling for the given shape and type.

  For an n-dimensional shape, returns the tiling for the last
  ``len(tiling.shape)`` dimensions and 1 for the leading dims. For example:
  - 2D tiling: (256, 256) -> (8, 128) and (2, 3, 128, 128) -> (1, 1, 8, 128).
  - 1D tiling: (16,) -> (8,) and (2, 3, 8) -> (1, 1, 8).

  Types are not required to have a dtype, so for such types we return None for
  all dimensions because their tiling is unknown.
  """
  assert hasattr(ty, "shape")
  shape = ty.shape
  if not hasattr(ty, "dtype"):
    return (None,) * len(shape)
  if ty.dtype == jnp.dtype("int4"):
    packing = 8
  else:
    packing = 4 // ty.dtype.itemsize

  if tiling is None:
    tiling = Tiling.COMPACT
  tiling_rank = len(tiling.shape)
  if len(shape) == 1 and tiling == Tiling.COMPACT:
    sublane_count, lane_count = tiling.shape
    src_sublane = pallas_utils.cdiv(shape[0], lane_count)
    max_tiling = max(sublane_count, packing)
    factor = _get_tiling_factor(src_sublane, max_tiling, packing)
    return (factor * lane_count,)
  if len(shape) < tiling_rank:
    raise ValueError(
        f"Shape must have at least {tiling_rank} dimensions: {shape=}"
    )

  leading_dims, final_dims = shape[:-tiling_rank], shape[-tiling_rank:]
  match tiling:
    case Tiling.COMPACT:
      second_minor, _ = final_dims
      factor = _get_tiling_factor(second_minor, tiling.shape[0], packing)
      return (*(1,) * len(leading_dims), factor, tiling.shape[1])
    case Tiling.SPARSE_CORE:
      [tile_size] = tiling.shape
      return (*(1,) * len(leading_dims), tile_size * packing)  # pytype: disable=bad-return-type


def get_device_kind() -> str:
  if abstract_device := mesh_lib.get_abstract_mesh().abstract_device:
    return abstract_device.device_kind
  return pxla.get_default_device().device_kind


def get_num_device_cores() -> int:
  if abstract_device := mesh_lib.get_abstract_mesh().abstract_device:
    return abstract_device.num_cores
  return pxla.get_default_device().num_cores
