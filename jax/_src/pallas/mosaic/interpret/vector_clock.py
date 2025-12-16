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

import numpy as np

VectorClock = np.ndarray


def make_vector_clock(vector_clock_size: int) -> VectorClock:
  return np.zeros(vector_clock_size, dtype=np.int32)


def copy_vector_clock(x: VectorClock) -> VectorClock:
  if x is None:
    return None
  return x.copy()


def update_vector_clock(x: VectorClock, y: VectorClock):
  x[:] = np.maximum(x[:], y[:])


def lt(x: VectorClock, y: VectorClock) -> bool:
  return bool((x <= y).all() & (x < y).any())


def ordered(x: VectorClock, y: VectorClock) -> bool:
  return lt(x, y) | lt(y, x)


def inc_vector_clock(x: VectorClock, global_core_id: int):
  if global_core_id >= len(x):
    raise ValueError(f'device_id={global_core_id} is out of range for x={x}')
  assert global_core_id < len(x)
  x[global_core_id] += 1
