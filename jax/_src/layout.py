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

from jax._src.sharding import Sharding
from jax._src.sharding_impls import AUTO as AutoSharding, is_auto
from jax._src.lib import xla_client as xc


class DeviceLocalLayout:
  layout: xc.PjRtLayout

  def __init__(self, layout: xc.PjRtLayout):
    self._layout = layout
    self._layout_str = str(self._layout)

  def __repr__(self):
    return f'DeviceLocalLayout({self._layout_str})'

  def __hash__(self):
    return hash(self._layout)

  def __eq__(self, other):
    if not isinstance(other, DeviceLocalLayout):
      return False
    return self._layout == other._layout

  def _to_xla_layout(self) -> str:
    return self._layout_str


class AutoLayout:

  def __repr__(self):
    return "AUTO"

AUTO = AutoLayout()


LayoutOptions = Union[DeviceLocalLayout, None, AutoLayout]
ShardingOptions = Union[Sharding, None, AutoSharding]


class Layout:
  __slots__ = ['device_local_layout', 'sharding']

  def __init__(self, device_local_layout: LayoutOptions,
               sharding: ShardingOptions):
    # If layout is concrete and sharding is not, error.
    if (isinstance(device_local_layout, DeviceLocalLayout) and
        (sharding is None or is_auto(sharding))):
      raise ValueError(
          'Sharding has to be concrete when layout is of type'
          f' {type(device_local_layout)}. Please pass a'
          ' `jax.sharding.NamedSharding`, `jax.sharding.PositionalSharding` or'
          ' `jax.sharding.SingleDeviceSharding` to the sharding argument. Got'
          f' sharding {sharding}'
      )
    self.device_local_layout = device_local_layout
    self.sharding = sharding

  def __repr__(self):
    return (f'Layout(device_local_layout={self.device_local_layout},'
            f' sharding={self.sharding})')

  def __hash__(self):
    return hash((self.device_local_layout, self.sharding))

  def __eq__(self, other):
    if not isinstance(other, Layout):
      return False
    return (self.device_local_layout == other.device_local_layout and
            self.sharding == other.sharding)
