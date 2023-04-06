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

from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    XLACompatibleSharding as XLACompatibleSharding,
    NamedSharding as NamedSharding,
    SingleDeviceSharding as SingleDeviceSharding,
    PmapSharding as PmapSharding,
    GSPMDSharding as GSPMDSharding,
    # TODO(yashkatariya): Remove OpShardingSharding in 3 months from
    # Feb 17, 2023.
    GSPMDSharding as _deprecated_OpShardingSharding,
    PositionalSharding as PositionalSharding,
)
from jax._src.partition_spec import (
    PartitionSpec as PartitionSpec,
)
from jax._src.interpreters.pxla import Mesh as Mesh


_deprecations = {
    "OpShardingSharding": (
        (
            "jax.sharding.OpShardingSharding is deprecated. Please use"
            " jax.sharding.GSPMDSharding."
        ),
        _deprecated_OpShardingSharding,
    ),
    "MeshPspecSharding": (
        (
            "jax.sharding.MeshPspecSharding has been removed. Please use"
            " jax.sharding.NamedSharding."
        ),
        None,
    ),
}

import typing
if typing.TYPE_CHECKING:
  from jax._src.sharding_impls import GSPMDSharding as OpShardingSharding
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
