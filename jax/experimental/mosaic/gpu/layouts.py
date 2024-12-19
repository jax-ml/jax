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

import itertools
import re

from jax._src.lib.mlir import ir

from .fragmented_array import WGSplatFragLayout, WGStridedFragLayout


_strided_fragmented_layout_attr_pattern = re.compile(
    r"^#mosaic_gpu.WGStridedFragLayout<\[(?P<shape>.*)\],"
    r" (?P<vector_size>\d+)>$"
)


def to_strided_fragmented_layout_attr(
    layout: WGStridedFragLayout,
) -> ir.Attribute:
  """Constructs a #mosaic_gpu.WGStridedFragLayout attribute from a WGStridedFragLayout."""
  return ir.Attribute.parse(
      f"#mosaic_gpu.WGStridedFragLayout<{list(layout.shape)},"
      f" {layout.vec_size}>"
  )


def from_strided_fragmented_layout_attr(
    attr: ir.Attribute,
) -> WGStridedFragLayout:
  """Constructs a WGStridedFragLayout from a #mosaic_gpu.WGStridedFragLayout attribute.

  Raises a ValueError if the attribute is not a #mosaic_gpu.WGStridedFragLayout
  attribute.
  """
  match = re.fullmatch(_strided_fragmented_layout_attr_pattern, str(attr))
  if not match:
    raise ValueError(
        f"Expected a #mosaic_gpu.WGStridedFragLayout attribute, got {attr}"
    )

  return WGStridedFragLayout(
      shape=tuple(int(s) for s in match.group("shape").split(",")),
      vec_size=int(match.group("vector_size")),
  )


def is_strided_fragmented_layout(attr: ir.Attribute) -> bool:
  return bool(re.search(_strided_fragmented_layout_attr_pattern, str(attr)))


def to_splat_fragmented_layout_attr(layout: WGSplatFragLayout) -> ir.Attribute:
  """Constructs a #mosaic_gpu.WGSplatFragLayout attribute from a WGSplatFragLayout."""
  return ir.Attribute.parse(
      f"#mosaic_gpu.WGSplatFragLayout<{list(layout.shape)}>"
  )


def should_have_layout(op: ir.OpView) -> bool:
  """Returns 'true' if the operation should be assigned a layout."""

  is_array = lambda v: ir.VectorType.isinstance(v.type)
  return any(map(is_array, itertools.chain(op.operands, op.results)))  # type: ignore


def has_any_layout_set(op: ir.OpView) -> bool:
  return "in_layouts" in op.attributes or "out_layouts" in op.attributes
