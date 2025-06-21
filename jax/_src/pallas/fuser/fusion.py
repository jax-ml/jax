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

"""Fusion classes."""

from __future__ import annotations

import dataclasses
from typing import Any, Generic, ParamSpec, TypeVar
from collections.abc import Callable

import jax
from jax._src import util

safe_map = util.safe_map

A = ParamSpec("A")
K = TypeVar("K")


@dataclasses.dataclass
class Fusion(Generic[A, K]):

  func: Callable[A, K]
  in_type: tuple[tuple[Any, ...], dict[str, Any]]
  out_type: Any

  def __call__(self, *args: A.args, **kwargs: A.kwargs) -> K:
    return self.func(*args, **kwargs)

  @property
  def shape(self):
    return jax.tree.map(lambda x: x.shape, self.out_type)

  @property
  def dtype(self):
    return jax.tree.map(lambda x: x.dtype, self.out_type)

  @property
  def type(self):
    return self.out_type

  @property
  def in_shape(self):
    return jax.tree.map(lambda x: x.shape, self.in_type)

  @property
  def in_dtype(self):
    return jax.tree.map(lambda x: x.dtype, self.in_type)
