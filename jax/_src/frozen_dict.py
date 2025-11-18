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

from typing import Any, TypeVar
from collections.abc import Iterator, Mapping

K = TypeVar("K")
V = TypeVar("V")


class FrozenDict(Mapping[K, V]):

  def __init__(self, d: Mapping[K, V]):
    self._d = dict(d.items())

  def __repr__(self) -> str:
    return f"FrozenDict({self._d!r})"

  def __str__(self) -> str:
    return f"FrozenDict({self._d})"

  def __getitem__(self, key: K) -> V:
    return self._d[key]

  def __hash__(self) -> int:
    # This assumes that the values are hashable.
    return hash(frozenset(self._d.items()))

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, FrozenDict):
      return False
    return self._d == other._d

  def __iter__(self) -> Iterator[K]:
    return iter(self._d)

  def __len__(self) -> int:
    return len(self._d)

  def get(self, key: K) -> V | None:  # type: ignore
    return self._d.get(key, None)
