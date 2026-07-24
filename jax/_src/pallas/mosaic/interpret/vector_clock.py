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

from typing import Protocol, Self
import numpy as np


class VectorClockProto(Protocol):

  def __init__(self, vector_clock_size: int):
    ...

  def copy(self) -> Self:
    ...

  def update(self, other: Self) -> None:
    ...

  def inc(self, position: int) -> None:
    ...


class NpVectorClock(VectorClockProto):
  clock: np.ndarray

  def __init__(self, vector_clock_size: int):
    self.clock = np.zeros(vector_clock_size, dtype=np.int32)

  def copy(self) -> Self:
    new = self.__new__(self.__class__)
    new.clock[:] = self.clock[:]
    return new

  def update(self, other: Self) -> None:
    self.clock[:] = np.maximum(self.clock[:], other.clock[:])

  def lt(self, other: Self) -> bool:
    return bool((self.clock <= other.clock).all() & (self.clock < other.clock).any())

  def ordered(self, other: Self) -> bool:
    return self.lt(other) or other.lt(self)

  def inc(self, position: int) -> None:
    if position >= len(self.clock):
      raise ValueError(f'position {position} is out of range for clock {self.clock}')
    assert position < len(self.clock)
    self.clock[position] += 1
