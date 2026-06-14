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
"""Internal APIs and data structures for the custom pipelining API."""
from collections.abc import Hashable, Sequence
import dataclasses

from jax._src import core as jax_core
from jax._src.state import types as state_types


ReadEffect = state_types.ReadEffect
WriteEffect = state_types.WriteEffect
RefEffect = state_types.ReadEffect | state_types.WriteEffect
BufferIndex = int | str


def filter_write_effects(effects: set[RefEffect]) -> set[WriteEffect]:
  return {effect for effect in effects if isinstance(effect, WriteEffect)}


def filter_read_effects(effects: set[RefEffect]) -> set[ReadEffect]:
  return {effect for effect in effects if isinstance(effect, ReadEffect)}


def filter_tokens(effects: set[RefEffect]) -> set[RefEffect]:
  return {effect for effect in effects if isinstance(effect.input_index, str)}


@dataclasses.dataclass(frozen=True)
class SchedulingProperties:
  max_in_flight: int
  is_async_start: bool
  is_async_done: bool

  def __post_init__(self):
    if self.is_async_start and self.is_async_done:
      raise ValueError(
          "Async start and async done are mutually exclusive.")


@dataclasses.dataclass(frozen=True)
class PipelineStage:
  """An internal representation of a pipeline stage."""
  jaxpr: jax_core.ClosedJaxpr
  effects: set[RefEffect]
  properties: SchedulingProperties
  name: str

  def get_read_idxs(self) -> set[BufferIndex]:
    """Returns the buffer indices that this stage reads from."""
    return {
        effect.input_index
        for effect in filter_read_effects(self.effects)
    }

  def get_write_idxs(self) -> set[BufferIndex]:
    """Returns the buffer indices that this stage writes to."""
    return {
        effect.input_index
        for effect in filter_write_effects(self.effects)
    }

  def __str__(self):
    return self.name

  def __repr__(self):
    return f"{self.name}[effs={self.effects}]"


@dataclasses.dataclass(frozen=True)
class NDLoopStruct:
  stages: Sequence[PipelineStage]
  grid: Sequence[int]


def make_token(obj: Hashable) -> str:
  """Returns a fake input ID used to thread data dependencies."""
  return f"token_{hash(obj)}"
