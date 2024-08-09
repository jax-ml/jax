# Copyright 2024 The JAX Authors.
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

"""Contains GPU-specific Pallas abstractions."""

import dataclasses
import enum
from jax import core as jax_core
from jax._src.pallas import core as pallas_core
import jax.numpy as jnp

AbstractMemoryRef = pallas_core.AbstractMemoryRef


class GPUMemorySpace(enum.Enum):
  GMEM = "gmem"
  SMEM = "smem"
  REGS = "regs"

  def __str__(self) -> str:
    return self.value

  def __call__(self, shape: tuple[int, ...], dtype: jnp.dtype):
    # A convenience function for constructing MemoryRef types.
    return MemoryRef(shape, dtype, self)


# TODO(b/354568887): Cosolidate this with TPU's MemoryRef.
@dataclasses.dataclass(frozen=True)
class MemoryRef:
  """Like jax.ShapeDtypeStruct but with memory spaces."""

  shape: tuple[int, ...]
  dtype: jnp.dtype
  memory_space: GPUMemorySpace

  def get_aval(self) -> AbstractMemoryRef:
    return AbstractMemoryRef(
        jax_core.ShapedArray(self.shape, self.dtype), self.memory_space
    )

GMEM = GPUMemorySpace.GMEM
SMEM = GPUMemorySpace.SMEM
REGS = GPUMemorySpace.REGS
