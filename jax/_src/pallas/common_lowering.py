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

"""Common lowering utils for Pallas backends."""

import functools
from jax._src import tree_util
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.pallas.primitives import DeviceIdType

def _device_id_to_logical(
    ctx, device_id,
    device_id_type: DeviceIdType):
  if device_id_type is DeviceIdType.MESH:
    # Mesh means we are passed the mesh coordinates for the device
    device_ids = tree_util.tree_leaves(device_id)
    mesh_strides = ctx.lowering_context.mesh_context.mesh_strides

    i32 = ir.IntegerType.get_signless(32)
    if len(device_ids) == 0:
      return arith.constant(i32, 0)
    return functools.reduce(
        arith.addi,
        (
            arith.muli(a, arith.constant(i32, b))
            for a, b in zip(device_ids, mesh_strides)
        ),
    )
  elif device_id_type is DeviceIdType.LOGICAL:
    return device_id
  raise NotImplementedError(f"Unsupported device id type: {device_id_type}")
