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

import re

from jax._src.lib.mlir import ir
from . import fragmented_array as fa


_splat_fragmented_layout_attr_pattern = re.compile(
    r"^#mosaic_gpu.WGSplatFragLayout<\[(?P<shape>.*)\]>$"
)


def to_splat_fragmented_layout_attr(layout: fa.WGSplatFragLayout) -> ir.Attribute:
  """Constructs a #mosaic_gpu.WGSplatFragLayout attribute from a WGSplatFragLayout."""
  return ir.Attribute.parse(
      f"#mosaic_gpu.WGSplatFragLayout<{list(layout.shape)}>"
  )


def from_splat_fragmented_layout_attr(attr: ir.Attribute) -> fa.WGSplatFragLayout:
  """Constructs a WGSplatFragLayout from a #mosaic_gpu.WGSplatFragLayout attribute.

  Raises:
    ValueError: If the attribute is not a #mosaic_gpu.WGSplatFragLayout
      attribute.
  """
  match = _splat_fragmented_layout_attr_pattern.fullmatch(str(attr))
  if not match:
    raise ValueError(
        f"Expected a #mosaic_gpu.WGSplatFragLayout attribute, got {attr}"
    )

  return fa.WGSplatFragLayout(
      shape=tuple(int(s) for s in match.group("shape").split(","))
  )


def is_splat_fragmented_layout(attr: ir.Attribute) -> bool:
  return bool(_splat_fragmented_layout_attr_pattern.search(str(attr)))


_strided_fragmented_layout_attr_pattern = re.compile(
    r"^#mosaic_gpu.WGStridedFragLayout<\[(?P<shape>.*)\],"
    r" (?P<vector_size>\d+)>$"
)

def to_strided_fragmented_layout_attr(
    layout: fa.WGStridedFragLayout,
) -> ir.Attribute:
  """Constructs a #mosaic_gpu.WGStridedFragLayout attribute from a WGStridedFragLayout."""
  return ir.Attribute.parse(
      f"#mosaic_gpu.WGStridedFragLayout<{list(layout.shape)},"
      f" {layout.vec_size}>"
  )


def from_strided_fragmented_layout_attr(
    attr: ir.Attribute,
) -> fa.WGStridedFragLayout:
  """Constructs a WGStridedFragLayout from a #mosaic_gpu.WGStridedFragLayout attribute.

  Raises:
    ValueError: If the attribute is not a #mosaic_gpu.WGStridedFragLayout
      attribute.
  """
  match = _strided_fragmented_layout_attr_pattern.fullmatch(str(attr))
  if not match:
    raise ValueError(
        f"Expected a #mosaic_gpu.WGStridedFragLayout attribute, got {attr}"
    )

  return fa.WGStridedFragLayout(
      shape=tuple(int(s) for s in match.group("shape").split(",")),
      vec_size=int(match.group("vector_size")),
  )


def is_strided_fragmented_layout(attr: ir.Attribute) -> bool:
  return bool(_strided_fragmented_layout_attr_pattern.search(str(attr)))


_tiled_layout_attr_pattern = re.compile(
    r"^#mosaic_gpu.TiledLayout<\[(?P<tiling>.*)\],"
    r" warp_dim\s*=\s*(?P<warp_dim>.+),"
    r" lane_dims\s*=\s*\[(?P<lane_dims>.*)\],"
    r" vector_dim\s*=\s*(?P<vector_dim>[-\d]+)>$"
)


def to_tiled_layout_attr(
    layout: fa.TiledLayout,
) -> ir.Attribute:
  """Constructs a #mosaic_gpu.TiledLayout attribute from a TiledLayout."""

  def _int_or_replicated(d: int | fa.Replicated) -> str:
    if isinstance(d, fa.Replicated):
      return f"#mosaic_gpu.Replicated<times={d.times}>"
    return str(d)

  tile_str = lambda tile: "[" + ", ".join(str(d) for d in tile) + "]"
  tiling = "[" + ", ".join(tile_str(tile) for tile in layout.tiling.tiles) + "]"
  lane_dims = (
      "[" + ",".join(_int_or_replicated(d) for d in layout.lane_dims) + "]"
  )

  return ir.Attribute.parse(
      f"#mosaic_gpu.TiledLayout<{tiling},"
      f" warp_dim={_int_or_replicated(layout.warp_dim)},"
      f" lane_dims={lane_dims}, vector_dim={layout.vector_dim}>"
  )


_list_of_lists_delimiter = re.compile(r"\]\s*,\s*\[")
_int_pattern = re.compile(r"^(?P<num>[-\d]+)(\s*:\s*\w+)?$")
_replicated_pattern = re.compile(
    r"^#mosaic_gpu.Replicated<\s*times\s*=\s*(?P<times>\d+)\s*>\s*$"
)


def from_tiled_layout_attr(
    attr: ir.Attribute,
) -> fa.TiledLayout:
  """Constructs a TiledLayout from a #mosaic_gpu.TiledLayout attribute.

  Raises:
    ValueError: If the attribute is not a #mosaic_gpu.TiledLayout
      attribute.
  """
  match = _tiled_layout_attr_pattern.fullmatch(str(attr))
  if not match:
    raise ValueError(
        f"Expected a #mosaic_gpu.TiledLayout attribute, got {attr}"
    )

  def _int_or_replicated(replicated_dim: str) -> int | fa.Replicated:
    match = _replicated_pattern.fullmatch(replicated_dim)
    if match:
      return fa.Replicated(int(match.group("times")))
    match = _int_pattern.fullmatch(replicated_dim)
    if match:
      return int(match.group("num"))
    raise ValueError(f"Unexpected format for replicated dim {replicated_dim}")

  tiling_str = match.group("tiling")
  tile_strings = []
  if len(tiling_str) > 2:
    tile_strings = _list_of_lists_delimiter.split(tiling_str[1:-1])
  tiles = tuple(tuple(map(int, ts.split(","))) for ts in tile_strings)
  return fa.TiledLayout(
      tiling=fa.Tiling(tiles),
      warp_dim=_int_or_replicated(match.group("warp_dim")),
      lane_dims=tuple(
          _int_or_replicated(s.strip())
          for s in match.group("lane_dims").split(",")
      ),
      vector_dim=int(match.group("vector_dim")),
  )


def is_tiled_layout(attr: ir.Attribute) -> bool:
  return bool(_tiled_layout_attr_pattern.search(str(attr)))


def to_layout_attr(
    layout: (
        fa.WGSplatFragLayout
        | fa.WGStridedFragLayout
        | fa.TiledLayout
    ),
) -> ir.Attribute:
  """Constructs an MLIR attribute that corresponds to the given layout."""
  match layout:
    case fa.WGSplatFragLayout():
      return to_splat_fragmented_layout_attr(layout)
    case fa.WGStridedFragLayout():
      return to_strided_fragmented_layout_attr(layout)
    case fa.TiledLayout():
      return to_tiled_layout_attr(layout)
    case _:
      raise NotImplementedError(
          f"Unsupported layout for conversion to MLIR attribute: {layout}"
      )


def from_layout_attr(
    attr: ir.Attribute,
) -> (
    fa.WGSplatFragLayout
    | fa.WGStridedFragLayout
    | fa.TiledLayout
):
  """Constructs a layout from an MLIR attribute."""
  if is_splat_fragmented_layout(attr):
    return from_splat_fragmented_layout_attr(attr)
  elif is_strided_fragmented_layout(attr):
    return from_strided_fragmented_layout_attr(attr)
  elif is_tiled_layout(attr):
    return from_tiled_layout_attr(attr)
  else:
    raise NotImplementedError(
        f"Unsupported layout for conversion from MLIR attribute: {attr}"
    )
