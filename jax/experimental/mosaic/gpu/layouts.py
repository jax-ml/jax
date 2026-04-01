# Copyright 2024 The JAX Authors.
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

"""Layout utilities."""

from typing import assert_never

from jax._src.lib import mosaic_gpu_dialect as mgpu
from jax._src.lib.mlir import ir

from . import fragmented_array as fa
from . import launch_context


# TODO(b/415721295): Refine return type once minimum jaxlib version is 0.10.0
def _to_splat_fragmented_layout_attr(
    layout: fa.WGSplatFragLayout,
) -> ir.Attribute:
  """Constructs a #mosaic_gpu.WGSplatFragLayout attribute from a WGSplatFragLayout."""
  shape = ir.DenseI64ArrayAttr.get(layout.shape)
  return mgpu.WGSplatFragLayoutAttr.get(shape)


def _from_splat_fragmented_layout_attr(
    attr: ir.Attribute,
) -> fa.WGSplatFragLayout:
  # TODO(b/415721295): Refine arg type once minimum jaxlib version is 0.10.0
  assert isinstance(attr, mgpu.WGSplatFragLayoutAttr)
  return fa.WGSplatFragLayout(shape=tuple(attr.shape))


# TODO(b/415721295): Refine return type once minimum jaxlib version is 0.10.0
def _to_strided_fragmented_layout_attr(
    layout: fa.WGStridedFragLayout,
) -> ir.Attribute:
  """Constructs a #mosaic_gpu.WGStridedFragLayout attribute from a WGStridedFragLayout."""
  shape = ir.DenseI64ArrayAttr.get(layout.shape)
  return mgpu.WGStridedFragLayoutAttr.get(shape, layout.vec_size)


def _from_strided_fragmented_layout_attr(
    attr: ir.Attribute,
) -> fa.WGStridedFragLayout:
  """Constructs a WGStridedFragLayout from a #mosaic_gpu.WGStridedFragLayout attribute."""
  # TODO(b/415721295): Refine arg type once minimum jaxlib version is 0.10.0
  assert isinstance(attr, mgpu.WGStridedFragLayoutAttr)
  return fa.WGStridedFragLayout(
      shape=tuple(attr.shape),
      vec_size=attr.vector_size,
  )


# TODO(b/415721295): Refine return type once minimum jaxlib version is 0.10.0
def _to_tiled_layout_attr(
    layout: fa.TiledLayout,
) -> ir.Attribute:
  """Constructs a #mosaic_gpu.TiledLayout attribute from a TiledLayout."""
  i64 = ir.IntegerType.get_signless(64)

  def _int_or_replicated(d: int | fa.Replicated) -> ir.Attribute:
    if isinstance(d, fa.Replicated):
      return mgpu.ReplicatedAttr.get(d.times)
    return ir.IntegerAttr.get(i64, d)

  def _tile_attr(tile):
    return ir.ArrayAttr.get([ir.IntegerAttr.get(i64, d) for d in tile])

  tiling_attr = ir.ArrayAttr.get(
      [_tile_attr(tile) for tile in layout.tiling.tiles]
  )
  warp_dims_attr = ir.ArrayAttr.get(
      [_int_or_replicated(d) for d in layout.warp_dims]
  )
  lane_dims_attr = ir.ArrayAttr.get(
      [_int_or_replicated(d) for d in layout.lane_dims]
  )

  return mgpu.TiledLayoutAttr.get(
      tiling_attr, warp_dims_attr, lane_dims_attr, layout.vector_dim
  )


def _from_tiled_layout_attr(
    attr: ir.Attribute,
) -> fa.TiledLayout:
  """Constructs a TiledLayout from a #mosaic_gpu.TiledLayout attribute."""
  # TODO(allanrenucci): Refine arg type once minimum jaxlib version is 0.10.0
  assert isinstance(attr, mgpu.TiledLayoutAttr)

  def _from_int_or_replicated_attr(d_attr: ir.Attribute) -> int | fa.Replicated:
    if isinstance(d_attr, mgpu.ReplicatedAttr):
      return fa.Replicated(times=mgpu.ReplicatedAttr(d_attr).times)
    return ir.IntegerAttr(d_attr).value

  tiles = tuple(
      tuple(ir.IntegerAttr(d).value for d in ir.ArrayAttr(tile))
      for tile in attr.tiling
  )
  warp_dims = tuple(_from_int_or_replicated_attr(d) for d in attr.warp_dims)
  lane_dims = tuple(_from_int_or_replicated_attr(d) for d in attr.lane_dims)

  return fa.TiledLayout(
      tiling=fa.Tiling(tiles),
      warp_dims=warp_dims,
      lane_dims=lane_dims,
      vector_dim=attr.vector_dim,
  )


def to_layout_attr(layout: fa.FragmentedLayout) -> ir.Attribute:
  """Constructs an MLIR attribute that corresponds to the given layout."""
  match layout:
    case fa.WGSplatFragLayout():
      return _to_splat_fragmented_layout_attr(layout)
    case fa.WGStridedFragLayout():
      return _to_strided_fragmented_layout_attr(layout)
    case fa.TiledLayout():
      return _to_tiled_layout_attr(layout)
    case _:
      assert_never(layout)


def from_layout_attr(attr: ir.Attribute) -> fa.FragmentedLayout:
  """Constructs a layout from an MLIR attribute."""
  if isinstance(attr, mgpu.WGSplatFragLayoutAttr):
    return _from_splat_fragmented_layout_attr(attr)
  elif isinstance(attr, mgpu.WGStridedFragLayoutAttr):
    return _from_strided_fragmented_layout_attr(attr)
  elif isinstance(attr, mgpu.TiledLayoutAttr):
    return _from_tiled_layout_attr(attr)
  else:
    raise NotImplementedError(
        f"Unsupported layout for conversion from MLIR attribute: {attr}"
    )


def splat_is_compatible_with_tiled(
    l1: fa.WGSplatFragLayout, l2: fa.TiledLayout
) -> bool:
  # A splat layout is compatible with a tiled layout up to replication if each
  # dimension in the shape of the splat layout is divisible by the corresponding
  # dimension in the base tile shape.
  s1, s2 = l1.shape, l2.base_tile_shape
  return all(d1 % d2 == 0 for d1, d2 in zip(s1, s2))


def to_transform_attr(
    transform: launch_context.MemRefTransform | mgpu.SwizzlingMode,
) -> ir.Attribute:
  if isinstance(transform, launch_context.TileTransform):
    return mgpu.TileTransformAttr.get(transform.tiling)
  elif isinstance(transform, launch_context.TransposeTransform):
    return mgpu.TransposeTransformAttr.get(transform.permutation)
  elif isinstance(transform, mgpu.SwizzlingMode):
    return mgpu.SwizzleTransformAttr.get(transform)
  else:
    raise NotImplementedError(f"Unsupported transform {transform}")


def from_transform_attr(
    transform: ir.Attribute,
) -> launch_context.MemRefTransform | mgpu.SwizzlingMode:
  if isinstance(transform, mgpu.TileTransformAttr):
    return launch_context.TileTransform(
        tuple(mgpu.TileTransformAttr(transform).tiling)
    )
  elif isinstance(transform, mgpu.TransposeTransformAttr):
    return launch_context.TransposeTransform(
        tuple(mgpu.TransposeTransformAttr(transform).permutation)
    )
  elif isinstance(transform, mgpu.SwizzleTransformAttr):
    return mgpu.SwizzlingMode(mgpu.SwizzleTransformAttr(transform).swizzle)
  else:
    raise NotImplementedError(f"Unsupported transform {transform}")
