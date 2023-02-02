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

from typing import Type, Set


class Effect:
  """A generic side-effect."""

Effects = Set[Effect]

class EffectTypeSet:

  def __init__(self):
    self._effect_types: Set[Type[Effect]] = set()

  def add_type(self, effect_type: Type[Effect]):
    self._effect_types.add(effect_type)

  def contains(self, eff: Effect) -> bool:
    return any(isinstance(eff, eff_type) for eff_type in self._effect_types)

  def filter_in(self, effects: Effects) -> Effects:
    return {eff for eff in effects if self.contains(eff)}

  def filter_not_in(self, effects: Effects) -> Effects:
    return {eff for eff in effects if not self.contains(eff)}

no_effects: Effects = set()
ordered_effects: EffectTypeSet = EffectTypeSet()
lowerable_effects: EffectTypeSet = EffectTypeSet()
control_flow_allowed_effects: EffectTypeSet = EffectTypeSet()
custom_derivatives_allowed_effects: EffectTypeSet = EffectTypeSet()
remat_allowed_effects: EffectTypeSet = EffectTypeSet()
