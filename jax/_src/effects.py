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
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

class Effect:
  """A generic side-effect."""

Effects = set[Effect]

class JaxprInputEffect(Effect):
  """A side-effect associated with the input of a jaxpr.

  Note that the `input_index` includes constvars.
  """

  def __init__(self, input_index: Any):
    self.input_index = input_index

  def replace(self, *, input_index: Any | None = None):
    if input_index is None:
      input_index = self.input_index
    return self.__class__(input_index)

  def __eq__(self, other):
    if not isinstance(other, JaxprInputEffect):
      return NotImplemented
    return self.input_index == other.input_index

  def __hash__(self):
    return hash((self.__class__, self.input_index))

  def __repr__(self):
    return f"{self.__class__.__name__}({self.input_index})"

class EffectTypeSet:

  def __init__(self):
    self._effect_types: set[type[Effect]] = set()

  def add_type(self, effect_type: type[Effect]):
    self._effect_types.add(effect_type)

  def contains(self, eff: Effect) -> bool:
    return any(isinstance(eff, eff_type) for eff_type in self._effect_types)

  def filter_in(self, effects: Iterable[Effect]) -> list[Effect]:
    return [eff for eff in effects if self.contains(eff)]

  def filter_not_in(self, effects: Iterable[Effect]) -> list[Effect]:
    return [eff for eff in effects if not self.contains(eff)]


no_effects: Effects = set()
ordered_effects: EffectTypeSet = EffectTypeSet()
lowerable_effects: EffectTypeSet = EffectTypeSet()
control_flow_allowed_effects: EffectTypeSet = EffectTypeSet()
custom_derivatives_allowed_effects: EffectTypeSet = EffectTypeSet()
remat_allowed_effects: EffectTypeSet = EffectTypeSet()
