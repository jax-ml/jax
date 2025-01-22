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

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from jax._src import core
from jax._src.layout import DeviceLocalLayout
from jax._src.typing import Shape

FfiLayoutOptions = Sequence[int] | DeviceLocalLayout | None


def aval_shape(aval: core.AbstractValue) -> Shape:
  return () if aval is core.abstract_token else aval.shape  # pytype: disable=attribute-error


def convert_layout_for_lowering(
    aval: core.AbstractValue, layout: FfiLayoutOptions = None
) -> Sequence[int]:
  """Convert a layout to the minor-to-major order used by the custom call API."""
  if layout is None:
    return tuple(reversed(range(len(aval_shape(aval)))))
  elif isinstance(layout, DeviceLocalLayout):
    if layout._tiling is not None:
      raise ValueError("The FFI does not support layouts with tiling")
    return layout.major_to_minor[::-1]
  else:
    return tuple(layout)


def unwrap_kwargs_hashable(kwargs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
  unwrapped_kwargs: dict[str, Any] = {}
  for k, v in kwargs:
    if isinstance(v, HashableArray):
      unwrapped_kwargs[k] = v.val
    elif isinstance(v, HashableDict):
      unwrapped_kwargs[k] = dict(v.val)
    else:
      unwrapped_kwargs[k] = v
  return unwrapped_kwargs


class HashableArray:
  __slots__ = ["val"]

  def __init__(self, val):
    assert isinstance(val, np.ndarray)
    self.val = np.copy(val)
    self.val.setflags(write=False)

  def __repr__(self):
    return f"HashableArray({self.val})"

  def __hash__(self):
    return hash((self.val.shape, self.val.dtype, self.val.tobytes()))

  def __eq__(self, other):
    return isinstance(other, HashableArray) and np.array_equal(
        self.val, other.val
    )


class HashableDict:
  __slots__ = ["val"]

  def __init__(self, val):
    assert isinstance(val, dict)
    self.val = tuple(sorted(val.items()))

  def __repr__(self):
    return f"HashableDict({dict(self.val)})"

  def __hash__(self):
    return hash(self.val)

  def __eq__(self, other):
    return isinstance(other, HashableDict) and self.val == other.val
