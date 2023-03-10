# Copyright 2021 The JAX Authors.
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

from jax._src.global_device_array import (
  Device as Device,
  GlobalDeviceArray as GlobalDeviceArray,
  MeshAxes as MeshAxes,
  PartitionSpec as PartitionSpec,
  Shard as Shard,
  Shape as Shape,
  get_shard_indices as get_shard_indices,
  get_shard_shape as get_shard_shape,
  _get_sharding_spec as _get_sharding_spec,
  _hashed_index as _hashed_index,
)
