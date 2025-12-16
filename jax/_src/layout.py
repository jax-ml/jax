# Copyright 2023 The JAX Authors.
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
# See the License for the ific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Union

import numpy as np
from jax._src.dtypes import iinfo, issubdtype
from jax._src.sharding import Sharding
from jax._src.named_sharding import AUTO as AutoSharding
from jax._src.util import tuple_insert
from jax._src.lib import xla_client as xc

Shape = tuple[int, ...]

class AutoLayout:

  def __repr__(self):
    return "AUTO"


class Layout:
  major_to_minor: tuple[int, ...]
  tiling: tuple[tuple[int, ...], ...] | None
  sub_byte_element_size_in_bits: int

  AUTO = AutoLayout()

  def __init__(self, major_to_minor: tuple[int, ...],
                tiling: tuple[tuple[int, ...], ...] | None = None,
                sub_byte_element_size_in_bits: int = 0):
    self.major_to_minor = tuple(major_to_minor)
    self.tiling = None if tiling is None else tuple(map(tuple, tiling))
    self._sub_byte_element_size_in_bits = sub_byte_element_size_in_bits

  @staticmethod
  def from_pjrt_layout(pjrt_layout: xc.PjRtLayout):
    xla_layout = pjrt_layout._xla_layout()
    return Layout(xla_layout.minor_to_major()[::-1],  # pytype: disable=wrong-arg-types
                  xla_layout.tiling(),  # type: ignore[arg-type]
                  xla_layout.element_size_in_bits())

  def __repr__(self):
    return (
        f'Layout(major_to_minor={self.major_to_minor},'
        f' tiling={self.tiling},'
        f' sub_byte_element_size_in_bits={self._sub_byte_element_size_in_bits})'
    )

  def __hash__(self):
    return hash((self.major_to_minor, self.tiling,
                  self._sub_byte_element_size_in_bits))

  def __eq__(self, other):
    if not isinstance(other, Layout):
      return False
    return (self.major_to_minor == other.major_to_minor and
            self.tiling == other.tiling and
            self._sub_byte_element_size_in_bits == other._sub_byte_element_size_in_bits)

  def update(self, **kwargs):
    if 'major_to_minor' not in kwargs:
      kwargs['major_to_minor'] = self.major_to_minor
    if 'tiling' not in kwargs:
      kwargs['tiling'] = self.tiling
    if 'sub_byte_element_size_in_bits' not in kwargs:
      kwargs['sub_byte_element_size_in_bits'] = self._sub_byte_element_size_in_bits
    return Layout(kwargs['major_to_minor'], kwargs['tiling'],
                  kwargs['sub_byte_element_size_in_bits'])

  def _to_xla_layout(self, dtype) -> xc.Layout:
    if self.tiling is None:
      xla_layout = xc.Layout(self.major_to_minor[::-1])
    else:
      if self._sub_byte_element_size_in_bits != 0:
        sub_byte_size = self._sub_byte_element_size_in_bits
      elif issubdtype(dtype, np.integer):
        sub_byte_size = iinfo(dtype).bits if iinfo(dtype).bits < 8 else 0
      else:
        sub_byte_size = 0
      xla_layout = xc.Layout(self.major_to_minor[::-1], self.tiling,
                              sub_byte_size)
    return xla_layout

  def check_compatible_aval(self, aval_shape: Shape):
    if len(self.major_to_minor) != len(aval_shape):
      raise ValueError(
          f'Length of major_to_minor and the rank of the value should match.'
          f' Got major_to_minor={self.major_to_minor} and shape={aval_shape}')


LayoutOptions = Union[Layout, None, AutoLayout]  # pytype: disable=invalid-annotation
ShardingOptions = Union[Sharding, None, AutoSharding]


class Format:
  __slots__ = ['layout', 'sharding']

  def __init__(self, layout: LayoutOptions = None,
               sharding: ShardingOptions = None):
    # If layout is concrete and sharding is not, error.
    if (isinstance(layout, Layout) and
        (sharding is None or isinstance(sharding, AutoSharding))):
      raise ValueError(
          'Sharding has to be concrete when layout is of type'
          f' {type(layout)}. Please pass a'
          ' `jax.sharding.NamedSharding` or'
          ' `jax.sharding.SingleDeviceSharding` to the sharding argument. Got'
          f' sharding {sharding}'
      )
    if not isinstance(
        layout, (Layout, type(None), AutoLayout)):
      raise TypeError(
          'Invalid value received for the layout argument.'
          ' Expected values are `None`, `Layout.AUTO` or an'
          f' instance of `Layout`. Got {layout} of'
          f' type {type(layout)}'
      )
    if not isinstance(
        sharding, (Sharding, type(None), AutoSharding)):
      raise TypeError(
          'Invalid value received for the sharding argument. Expected values'
          ' are `None`, `pjit.AUTO` or an instance of `jax.Sharding`. Got'
          f' {sharding} of type {type(sharding)}')

    self.layout = layout
    self.sharding = sharding

  def __repr__(self):
    return f'Format(layout={self.layout}, sharding={self.sharding})'

  def __hash__(self):
    return hash((self.layout, self.sharding))

  def __eq__(self, other):
    if not isinstance(other, Format):
      return False
    return (self.layout == other.layout and
            self.sharding == other.sharding)


def get_layout_for_vmap(dim: int, layout: Layout) -> Layout:
  # Make the new dim major-most and shift all other dims by 1 in major_to_minor
  new_m2m = tuple(m + 1 for m in layout.major_to_minor)
  vmapped_major_to_minor = tuple_insert(new_m2m, dim, 0)
  return layout.update(major_to_minor=vmapped_major_to_minor)
