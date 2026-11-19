# Copyright 2026 The JAX Authors.
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

import dataclasses
from typing import Any

import numpy as np

from jax._src import core
from jax._src import dtypes
from jax._src import mesh as mesh_lib
from jax._src.mesh import AxisType, MeshAxis as MeshAxis, ShardedMeshAxis
from jax._src.named_sharding import NamedSharding
from jax._src.partition_spec import PartitionSpec as P
from jax._src.typing import DTypeLike
from jax._src.util import set_module


@set_module('jax.avals')
@dataclasses.dataclass(frozen=True, slots=True)
class AvalDType:
  dtype: DTypeLike

  def __getitem__(self, elts: Any) -> core.ShapedArray:
    elts = elts if isinstance(elts, tuple) else (elts,)
    mesh = mesh_lib.get_abstract_mesh()
    if bad := [n for d in elts if isinstance(d, ShardedMeshAxis)
               for n in d.names if mesh._name_to_type.get(n) != AxisType.Explicit]:
      raise ValueError(
          f"Sharded axis {bad[0]!r} is not an Explicit axis in the active "
          f"context mesh {mesh} (did you call `jax.set_mesh` or "
          f"`jax.use_abstract_mesh`?)")
    shape = [d.size if isinstance(d, ShardedMeshAxis) else d for d in elts]
    spec = P(*(d.names if isinstance(d, ShardedMeshAxis) else None
               for d in elts))
    return core.ShapedArray(shape, self.dtype, sharding=NamedSharding(mesh, spec))


bool_ = AvalDType(np.bool_)
i2 = AvalDType(dtypes.int2)
i4 = AvalDType(dtypes.int4)
i8 = AvalDType(np.int8)
i16 = AvalDType(np.int16)
i32 = AvalDType(np.int32)
i64 = AvalDType(np.int64)
u2 = AvalDType(dtypes.uint2)
u4 = AvalDType(dtypes.uint4)
u8 = AvalDType(np.uint8)
u16 = AvalDType(np.uint16)
u32 = AvalDType(np.uint32)
u64 = AvalDType(np.uint64)
bf16 = AvalDType(dtypes.bfloat16)
f16 = AvalDType(np.float16)
f32 = AvalDType(np.float32)
f64 = AvalDType(np.float64)
c64 = AvalDType(np.complex64)
c128 = AvalDType(np.complex128)
float0 = AvalDType(dtypes.float0)
