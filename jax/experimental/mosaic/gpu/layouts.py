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
from typing import assert_never

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
    r" warp_dims\s*=\s*\[(?P<warp_dims>.*)\],"
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
  warp_dims = (
      "[" + ",".join(_int_or_replicated(d) for d in layout.warp_dims) + "]"
  )
  lane_dims = (
      "[" + ",".join(_int_or_replicated(d) for d in layout.lane_dims) + "]"
  )

  return ir.Attribute.parse(
      f"#mosaic_gpu.TiledLayout<{tiling}, warp_dims={warp_dims},"
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
      warp_dims=tuple(
          _int_or_replicated(s.strip())
          for s in match.group("warp_dims").split(",")
      ),
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


def splat_is_compatible_with_tiled(
    l1: fa.WGSplatFragLayout, l2: fa.TiledLayout
) -> bool:
  # A splat layout is compatible with a tiled layout up to replication if each
  # dimension in the shape of the splat layout is divisible by the corresponding
  # dimension in the base tile shape.
  s1, s2 = l1.shape, l2.base_tile_shape
  return all(d1 % d2 == 0 for d1, d2 in zip(s1, s2))


def meet_layouts(
    layout1: fa.FragmentedLayout, layout2: fa.FragmentedLayout
) -> fa.FragmentedLayout | None:
  """Returns the "meet" of two layouts that are compatible up to replication.

  The "meet" of the two layouts is the most replicated layout that is still
  less replicated than the arguments.

  This is the dual of `join_layouts`.

  Returns:
    The "meet" of the two layouts if both layouts are compatible up to
    replication.

  Raises:
    ValueError: if the two layouts are not compatible up to replication.
  """
  if layout1 == layout2:
    return layout1

  match (layout1, layout2):
    case (fa.WGSplatFragLayout(), _):
      if isinstance(layout2, fa.TiledLayout):
        if splat_is_compatible_with_tiled(layout1, layout2):
          return layout2
      elif layout1.shape == layout2.shape:
        return layout2
    case (_, fa.WGSplatFragLayout()):
      if isinstance(layout1, fa.TiledLayout):
        if splat_is_compatible_with_tiled(layout2, layout1):
          return layout1
      elif layout1.shape == layout2.shape:
        return layout1
    case (fa.TiledLayout(), fa.TiledLayout()):
      # TODO(bchetioui): handle `TiledLayout` replication.
      raise NotImplementedError("TiledLayout replication not supported yet")

  # Layouts are not compatible up to replication.
  return None

# NOTE: We say that two layouts are compatible up to replication if the two
# layouts satisfy at least one of the following conditions together:
#
# - The two layouts are equal;
# - One of the layouts is a `WGSplatFragLayout`, and
#   * The other layout is a `WGStridedFragLayout` with the same shape;
#   * The other layout is a `TiledLayout` that can be used to tile the shape
#     embedded in the `WGSplatFragLayout`.
#
# If any of these conditions hold, then we are always able to substitute one
# layout with the other without having to reorder any data in the underlying
# array---i.e. a relayout is free.
#
# Note that there are other combinations of layouts for which relayout is free,
# but we voluntarily narrowed down our definition to span a small, useful
# subset.

def join_layouts(
    layout1: fa.FragmentedLayout, layout2: fa.FragmentedLayout
) -> fa.FragmentedLayout | None:
  """Returns the "join" of two layouts that are compatible up to replication.

  The "join" of the two layouts is the least replicated layout that is still
  more replicated than the arguments.

  This is the dual of `meet_layouts`.

  Returns:
    The "join" of the two layouts if both layouts are compatible up to
    replication.

  Raises:
    ValueError: if the two layouts are not compatible up to replication.
  """
  if layout1 == layout2:
    return layout1

  match (layout1, layout2):
    case (fa.WGSplatFragLayout(), _):
      if isinstance(layout2, fa.TiledLayout):
        if splat_is_compatible_with_tiled(layout1, layout2):
          return layout1
      elif layout1.shape == layout2.shape:
        return layout1
    case (_, fa.WGSplatFragLayout()):
      if isinstance(layout1, fa.TiledLayout):
        if splat_is_compatible_with_tiled(layout2, layout1):
          return layout2
      elif layout1.shape == layout2.shape:
        return layout2
    case (fa.TiledLayout(), fa.TiledLayout()):
      # TODO(bchetioui): handle `TiledLayout` replication.
      raise NotImplementedError("TiledLayout replication not supported yet")

  # Layouts are not compatible up to replication.
  return None


def has_any_replication(layout: fa.FragmentedLayout) -> bool:
  match layout:
    case fa.WGSplatFragLayout():
      return True
    case fa.WGStridedFragLayout():
      return False
    case fa.TiledLayout():
      is_warp_replicated = any(isinstance(d, fa.Replicated) for d in layout.warp_dims)
      is_lane_replicated = any(isinstance(d, fa.Replicated) for d in layout.lane_dims)
      return is_warp_replicated or is_lane_replicated
    case _ as unreachable:
      return assert_never(unreachable)  # pytype: disable=wrong-arg-types
