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
from typing import Any, Literal
import numpy as np
from jax._src import dtypes
from jax._src import util
from jax._src.lax import lax
from jax._src.typing import Array, ArrayLike, DTypeLike

@dataclass(frozen=True)
class Intrinsic:
  name: str
  def __post_init__(self):
    if not isinstance(self.name, str) or not self.name:
      raise ValueError(f"Intrinsic.name must be a non-empty str, got {self.name!r}")

@dataclass(frozen=True)
class Unordered:
  scope: str | None = None
  def __post_init__(self):
    if self.scope is not None and (not isinstance(self.scope, str) or not self.scope):
      raise ValueError(f"Unordered.scope must be None or a non-empty str, got {self.scope!r}")

Order = Literal["unordered", "linear"] | Unordered | Intrinsic | None
_INTRINSIC_IMPLS: dict[str, Callable[..., Any]] = {}

def normalize_order(order: Any) -> Order:
  if order is None or isinstance(order, (Intrinsic, Unordered)):
    return order
  if isinstance(order, str) and order in ("unordered", "linear"):
    return order  # type: ignore[return-value]
  raise ValueError(f"Invalid reduction order: {order!r}")

def register_intrinsic(name: str, impl: Callable[..., Any]) -> None:
  Intrinsic(name)
  _INTRINSIC_IMPLS[name] = impl

def _get_sum_pad_value(dtype: DTypeLike, pad: Literal["zero", "absent"]) -> np.ndarray:
  if not isinstance(pad, str) or pad not in ("zero", "absent"):
    raise ValueError(f"Invalid pad={pad!r}; expected 'zero' or 'absent'.")
  if pad == "absent":
    if dtypes.issubdtype(dtype, np.floating):
      return np.array(-0.0, dtype=dtype)
    if dtypes.issubdtype(dtype, np.complexfloating):
      return np.array(complex(-0.0, -0.0), dtype=dtype)
  return lax._get_sum_identity(dtype)

def _pad_axis(arr: Array, ax: int, pad_len: int, pad_val: Any) -> Array:
  return lax.pad(arr, pad_val, [(0, pad_len if i == ax else 0, 0) for i in range(arr.ndim)])

def _prep(reduce_fn, x: ArrayLike, axis: int, dtype: DTypeLike | None) -> tuple[Array, int, int]:
  arr = lax.convert_element_type(x, dtype) if dtype is not None else lax.asarray(x)
  ax = util.canonicalize_axis(axis, arr.ndim)
  if (k := arr.shape[ax]) == 0 and reduce_fn is lax.reduce_max:
    raise ValueError("zero-size array to reduction operation max")
  return arr, ax, k

def _tiled_reduce(reduce_fn, pad_fn, x: ArrayLike, tile: int, *, axis: int = -1,
                  split: Literal["block", "lanes"] = "block", inner_order: Order = "linear",
                  outer_order: Order = "linear", dtype: DTypeLike | None = None,
                  allow_pad: bool = True, keepdims: bool = False) -> Array:
  if tile <= 0 or split not in ("block", "lanes"):
    raise ValueError(f"Invalid tile={tile!r} or split={split!r}")
  arr, ax, k = _prep(reduce_fn, x, axis, dtype)
  pad_val = pad_fn(arr.dtype)
  if k % tile:
    if not allow_pad:
      raise ValueError(f"Axis size {k} not divisible by tile {tile}")
    arr = _pad_axis(arr, ax, (-k) % tile, pad_val)
  arr = lax.reshape(arr, arr.shape[:ax] + (arr.shape[ax] // tile, tile) + arr.shape[ax + 1:])
  out = reduce_fn(reduce_fn(arr, (ax + int(split == "block"),), reduction_order=inner_order), (ax,), reduction_order=outer_order)
  return lax.expand_dims(out, (ax,)) if keepdims else out

def _xlane_reduce(reduce_fn, pad_fn, x: ArrayLike, axis: int = -1, *,
                  dtype: DTypeLike | None = np.float32, keepdims: bool = False) -> Array:
  arr, ax, k = _prep(reduce_fn, x, axis, dtype)
  pad_val = pad_fn(arr.dtype)
  if k > 128:
    return _tiled_reduce(reduce_fn, lambda _: pad_val, arr, 128, axis=ax, split="lanes",
                         outer_order=Intrinsic("tpu_xlane_128"), keepdims=keepdims)
  out = reduce_fn(_pad_axis(arr, ax, 128 - k, pad_val) if k < 128 else arr,
                  (ax,), reduction_order=Intrinsic("tpu_xlane_128"))
  return lax.expand_dims(out, (ax,)) if keepdims else out

def tiled_sum(x: ArrayLike, tile: int, *, axis: int = -1,
              split: Literal["block", "lanes"] = "block", inner_order: Order = "linear",
              outer_order: Order = "linear", dtype: DTypeLike | None = None,
              pad: Literal["zero", "absent"] = "zero", keepdims: bool = False) -> Array:
  return _tiled_reduce(lax.reduce_sum, lambda dt: _get_sum_pad_value(dt, pad), x, tile,
                       axis=axis, split=split, inner_order=inner_order,
                       outer_order=outer_order, dtype=dtype, keepdims=keepdims)

def tiled_max(x: ArrayLike, tile: int, *, axis: int = -1,
              split: Literal["block", "lanes"] = "block", inner_order: Order = "linear",
              outer_order: Order = "linear", dtype: DTypeLike | None = None,
              pad: bool = True, keepdims: bool = False) -> Array:
  return _tiled_reduce(lax.reduce_max, lax._get_max_identity, x, tile,
                       axis=axis, split=split, inner_order=inner_order,
                       outer_order=outer_order, dtype=dtype, allow_pad=pad, keepdims=keepdims)

def xlane_sum(x: ArrayLike, axis: int = -1, *, dtype: DTypeLike | None = np.float32,
              pad: Literal["zero", "absent"] = "zero", keepdims: bool = False) -> Array:
  return _xlane_reduce(lax.reduce_sum, lambda dt: _get_sum_pad_value(dt, pad), x, axis=axis,
                       dtype=dtype, keepdims=keepdims)

def xlane_max(x: ArrayLike, axis: int = -1, *, dtype: DTypeLike | None = np.float32,
              keepdims: bool = False) -> Array:
  return _xlane_reduce(lax.reduce_max, lax._get_max_identity, x, axis=axis,
                       dtype=dtype, keepdims=keepdims)

def blocked_dot(lhs: ArrayLike, rhs: ArrayLike, bk: int, *, dimension_numbers: Any = None,
                inner_order: Order = "linear", outer_order: Order = "linear",
                preferred_element_type: DTypeLike | None = np.float32,
                pad: Literal["zero", "absent"] = "zero") -> Array:
  a, b = lax.asarray(lhs), lax.asarray(rhs)
  a_pad = _get_sum_pad_value(a.dtype, pad)
  if bk <= 0 or a.ndim < 1 or b.ndim < 1:
    raise ValueError(f"Invalid bk={bk} or operand shapes {a.shape}, {b.shape}")
  (lc, rc), (lb, rb) = ((((a.ndim - 1,), (0 if b.ndim == 1 else b.ndim - 2,)), ((), ()))
                        if dimension_numbers is None else dimension_numbers)
  if len(lc) != 1 or len(rc) != 1:
    raise ValueError(f"Invalid contracting dims {(lc, rc)}")
  ca, cb = util.canonicalize_axis(lc[0], a.ndim), util.canonicalize_axis(rc[0], b.ndim)
  lb = tuple(util.canonicalize_axis(i, a.ndim) for i in lb)
  rb = tuple(util.canonicalize_axis(i, b.ndim) for i in rb)
  if (k := a.shape[ca]) != b.shape[cb]:
    raise ValueError(f"Mismatched contracting sizes {k} vs {b.shape[cb]}")
  if k % bk:
    b_pad = (_get_sum_pad_value(b.dtype, pad)
             if pad == "absent" and not dtypes.issubdtype(a.dtype, np.floating)
             else lax._get_sum_identity(b.dtype))
    a = _pad_axis(a, ca, (-k) % bk, a_pad)
    b = _pad_axis(b, cb, (-k) % bk, b_pad)
  if a.shape[ca] == bk:
    return lax.dot_general(a, b, (((ca,), (cb,)), (lb, rb)),
                           preferred_element_type=preferred_element_type, reduction_order=inner_order)
  a = lax.reshape(a, a.shape[:ca] + (a.shape[ca] // bk, bk) + a.shape[ca + 1:])
  b = lax.reshape(b, b.shape[:cb] + (b.shape[cb] // bk, bk) + b.shape[cb + 1:])
  bdims = ((ca, *(d + int(d > ca) for d in lb)), (cb, *(d + int(d > cb) for d in rb)))
  partials = lax.dot_general(a, b, (((ca + 1,), (cb + 1,)), bdims),
                             preferred_element_type=preferred_element_type, reduction_order=inner_order)
  return lax.reduce_sum(partials, (0,), reduction_order=outer_order)
