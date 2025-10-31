# Copyright 2025 The JAX Authors. All Rights Reserved.
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
# ==============================================================================

import enum
import jax
from jax._src.lib.mlir import ir
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jax.experimental.mosaic.gpu import layouts
from jax.experimental.mosaic.gpu import tcgen05
from jax.experimental.mosaic.gpu import utils


class RegisterLayout(enum.Enum):
  """The list of supported register layouts."""

  WGMMA = enum.auto()
  WG_SPLAT = enum.auto()
  WG_STRIDED = enum.auto()
  TCGEN05 = enum.auto()
  TCGEN05_M64_COLLECTIVE = enum.auto()
  TCGEN05_TMEM_NATIVE = enum.auto()
  SMEM_GMEM_COPY = enum.auto()
  TMA_GATHER_INDICES = enum.auto()

  def to_mgpu(
      self, shape: tuple[int, int], dtype: jax.typing.DTypeLike | ir.Type
  ) -> fa.FragmentedLayout:
    if not isinstance(dtype, ir.Type):
      dtype = utils.dtype_to_ir_type(dtype)
    match self:
      case RegisterLayout.WGMMA:
        return fa.WGMMA_LAYOUT
      case RegisterLayout.WG_SPLAT:
        return fa.WGSplatFragLayout(shape)
      case RegisterLayout.WG_STRIDED:
        ty = ir.VectorType.get(shape, dtype)
        layout = fa.WGStridedFragLayout.from_shaped_type(ty)
        assert layout is not None
        return layout
      case RegisterLayout.TCGEN05:
        return fa.TCGEN05_LAYOUT
      case RegisterLayout.TCGEN05_M64_COLLECTIVE:
        return tcgen05.fa_m64_collective_layout(shape[1])
      case RegisterLayout.TCGEN05_TMEM_NATIVE:
        return fa.TMEM_NATIVE_LAYOUT
      case RegisterLayout.SMEM_GMEM_COPY:
        swizzle = 128
        bitwidth = utils.bitwidth(dtype)
        tiling = (8, 8 * swizzle // bitwidth)
        row_tiles, col_tiles = utils.tile_shape(shape, tiling)[-4:-2]
        return fa.tiled_copy_smem_gmem_layout(
            row_tiles, col_tiles, swizzle, bitwidth
        )
      case RegisterLayout.TMA_GATHER_INDICES:
        return fa.TMA_GATHER_INDICES_LAYOUT

  def to_layout_attr(
      self, shape: tuple[int, int], dtype: jax.typing.DTypeLike | ir.Type
  ) -> ir.Attribute:
    return layouts.to_layout_attr(self.to_mgpu(shape, dtype))
