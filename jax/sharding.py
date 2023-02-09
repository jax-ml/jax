# Copyright 2022 The JAX Authors.
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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/google/jax/issues/7570

from jax._src.sharding import (
    Sharding as Sharding,
    XLACompatibleSharding as XLACompatibleSharding,
    # TODO(yashkatariya): Deprecate MeshPspecSharding in 3 months.
    MeshPspecSharding as MeshPspecSharding,
    # New name of MeshPspecSharding to match PositionalSharding below.
    NamedSharding as NamedSharding,
    PartitionSpec as PartitionSpec,
    SingleDeviceSharding as SingleDeviceSharding,
    PmapSharding as PmapSharding,
    OpShardingSharding as OpShardingSharding,
    PositionalSharding as PositionalSharding,
)

from jax._src.interpreters.pxla import Mesh as Mesh
