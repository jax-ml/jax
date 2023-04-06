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
"""Sharding utilities"""

from typing import Sequence, Tuple
from jax._src.lib import xla_client as xc


def get_num_ways_dim_sharded(
    op_sharding: xc.OpSharding) -> Tuple[Sequence[int], int]:
  partitions = op_sharding.tile_assignment_dimensions
  if op_sharding.last_tile_dims == [xc.OpSharding.Type.REPLICATED]:
    replicate_on_last_tile_dim = True
  else:
    replicate_on_last_tile_dim = op_sharding.replicate_on_last_tile_dim
    if op_sharding.last_tile_dims:
      raise NotImplementedError(
          "Unhandled OpSharding type. Please open a bug report!")
  num_replicas = 1
  if replicate_on_last_tile_dim:
    num_replicas = partitions[-1]
    partitions = partitions[:-1]
  return partitions, num_replicas


def is_op_sharding_replicated(op: xc.OpSharding) -> bool:
  if len(op.tile_assignment_devices) == 1:
    return True
  return xc.HloSharding.from_proto(op).is_replicated()  # type: ignore
