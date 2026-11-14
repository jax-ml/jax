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
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import enum
from typing import Any, Literal

import numpy as np

from jax._src.typing import DTypeLike


### Enums & leaf orders

class _NamedEnum(enum.Enum):
  def __repr__(self) -> str:
    return self.name


class ReductionOrder(_NamedEnum):
  unordered = enum.auto()
  linear = enum.auto()
  unreduced = enum.auto()


class Split(_NamedEnum):
  block = enum.auto()
  strided = enum.auto()


class PadVal(_NamedEnum):
  zero = enum.auto()
  absent = enum.auto()


class RingDir(_NamedEnum):
  asc = enum.auto()
  desc = enum.auto()
  bidir = enum.auto()


@dataclass(frozen=True)
class Intrinsic:
  name: str

  def __post_init__(self):
    if not isinstance(self.name, str) or not self.name:
      raise ValueError(
          f"Intrinsic.name must be a non-empty str, got {self.name!r}")


@dataclass(frozen=True)
class Unordered:
  scope: str | None = None

  def __post_init__(self):
    if self.scope is not None and (
        not isinstance(self.scope, str) or not self.scope
    ):
      raise ValueError(
          "Unordered.scope must be None or non-empty str, got"
          f" {self.scope!r}")


LeafOrder = ReductionOrder | Unordered | Intrinsic | None
SplitKind = Split | Literal["block", "lanes", "strided"]
PadMode = PadVal | Literal["zero", "absent"] | None
RingDirection = RingDir | Literal["asc", "desc", "bidir", "bidi"]
_INTRINSIC_IMPLS: dict[str, Callable[..., Any]] = {}


def register_intrinsic(name: str, impl: Callable[..., Any]) -> None:
  Intrinsic(name)
  _INTRINSIC_IMPLS[name] = impl


### Composite order combinators

@dataclass(frozen=True)
class Tile:
  tile: int
  inner: OrderLike = ReductionOrder.linear
  outer: OrderLike = ReductionOrder.linear
  split: SplitKind = Split.block
  pad: PadMode = PadVal.zero

  def __post_init__(self):
    sp = _coerce_enum(self.split, Split, _SPLIT_STRS, "split")
    pv = (
        None if self.pad is None
        else _coerce_enum(self.pad, PadVal, _PAD_STRS, "pad")
    )
    object.__setattr__(self, "split", sp)
    object.__setattr__(self, "pad", pv)
    object.__setattr__(self, "inner", normalize_order(self.inner))
    object.__setattr__(self, "outer", normalize_order(self.outer))


@dataclass(frozen=True)
class Pad:
  multiple: int
  order: OrderLike = ReductionOrder.linear
  pad: PadVal | Literal["zero", "absent"] = PadVal.zero

  def __post_init__(self):
    pad = _coerce_enum(self.pad, PadVal, _PAD_STRS, "pad")
    object.__setattr__(self, "pad", pad)
    object.__setattr__(self, "order", normalize_order(self.order))


@dataclass(frozen=True)
class Cast:
  dtype: DTypeLike
  order: OrderLike = ReductionOrder.linear

  def __post_init__(self):
    object.__setattr__(self, "dtype", np.dtype(self.dtype))
    object.__setattr__(self, "order", normalize_order(self.order))


@dataclass(frozen=True)
class Ring:
  direction: RingDirection = RingDir.bidir
  split_axis: int | None = None

  def __post_init__(self):
    direction = _coerce_enum(
        self.direction, RingDir, _RING_DIR_STRS, "direction")
    object.__setattr__(self, "direction", direction)


Order = LeafOrder | Tile | Pad | Cast | Ring
OrderLike = Order | Literal["unordered", "linear", "unreduced"]

tiled = Tile
padded = Pad
cast = Cast
ring = Ring


def xlane(
    lane_order: OrderLike = ReductionOrder.unordered,
    cross_lane_order: OrderLike = ReductionOrder.linear, *,
    num_lanes: int = 128, pad: PadMode = PadVal.zero,
) -> Tile:
  return Tile(
      tile=num_lanes, inner=lane_order, outer=cross_lane_order,
      split=Split.strided, pad=pad,
  )


### Normalization & validation helpers

_REDUCTION_ORDER_STRS = {
    "unordered": ReductionOrder.unordered,
    "linear": ReductionOrder.linear,
    "unreduced": ReductionOrder.unreduced,
}
_SPLIT_STRS = {
    "block": Split.block, "strided": Split.strided, "lanes": Split.strided,
}
_PAD_STRS = {"zero": PadVal.zero, "absent": PadVal.absent}
_RING_DIR_STRS = {
    "asc": RingDir.asc,
    "desc": RingDir.desc,
    "bidir": RingDir.bidir,
    "bidi": RingDir.bidir,
}


def is_leaf_order(order: Any) -> bool:
  return (
      order is None
      or order in (ReductionOrder.unordered, ReductionOrder.linear)
      or isinstance(order, (Intrinsic, Unordered))
  )


def normalize_order(order: Any) -> Order:
  if order is None or isinstance(
      order, (ReductionOrder, Intrinsic, Unordered, Tile, Pad, Cast, Ring)
  ):
    return order
  if isinstance(order, str) and (res := _REDUCTION_ORDER_STRS.get(order)):
    return res
  raise ValueError(f"Invalid reduction order: {order!r}")


def _coerce_enum(val: Any, cls: type, table: dict[str, Any], field: str):
  if isinstance(val, cls):
    return val
  if isinstance(val, str) and (res := table.get(val)) is not None:
    return res
  raise ValueError(f"Invalid {field}={val!r}.")
