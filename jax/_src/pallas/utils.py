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

"""Pallas utility functions."""
import math
import numpy as np
from typing import Tuple

from jax import lax


def when(condition):
  def _wrapped(f):
    if isinstance(condition, bool):
      if condition:
        f()
    else:
      lax.cond(condition, f, lambda: None)
  return _wrapped


def cdiv(a: int, b: int) -> int:
  return (a + b - 1) // b


def strides_from_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
  size = np.prod(shape)
  strides = []
  for s in shape:
    size = size // s
    strides.append(int(size))
  return tuple(strides)


def next_power_of_2(x: int) -> int:
  if x == 0:
    return 1
  return int(2 ** math.ceil(math.log2(x)))
