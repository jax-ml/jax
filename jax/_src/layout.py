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

from jax._src.lib import xla_client as xc


class Layout:
  pass


class XLACompatibleLayout(Layout):
  @classmethod
  def _from_xla_layout(cls, xla_layout) -> XLACompatibleLayout:
    raise NotImplementedError("Subclasses should implement this method.")

  def _to_xla_layout(self) -> str:
    raise NotImplementedError("Subclasses should implement this method.")


class SpecifiedLayout(XLACompatibleLayout):
  minor_to_major: tuple[int, ...]

  def __init__(self, minor_to_major: tuple[int, ...]):
    self.minor_to_major = minor_to_major

  def __repr__(self):
    return f'SpecifiedLayout(minor_to_major={self.minor_to_major})'

  def __hash__(self):
    return hash(self.minor_to_major)

  def __eq__(self, other):
    if not isinstance(other, SpecifiedLayout):
      return False
    return self.minor_to_major == other.minor_to_major

  @classmethod
  def _from_xla_layout(cls, xla_layout: xc.Layout) -> XLACompatibleLayout:
    return cls(xla_layout.minor_to_major())

  def _to_xla_layout(self) -> str:
    return xc.Layout(self.minor_to_major).to_string()


class LayoutRequest:

  def __repr__(self):
    return "Request a layout from the compiler"

AUTO = LayoutRequest()
