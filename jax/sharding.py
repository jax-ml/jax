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
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    SingleDeviceSharding as _deprecated_SingleDeviceSharding,
    set_mesh as set_mesh,
    get_mesh as get_mesh,
    make_single_device_sharding as make_single_device_sharding,
    is_single_device_sharding as is_single_device_sharding,
)
from jax._src.partition_spec import (
    PartitionSpec as PartitionSpec,
    UnreducedKind as UnreducedKind,
)
from jax._src.mesh import (
    Mesh as Mesh,
    AbstractDevice as AbstractDevice,
    AbstractMesh as AbstractMesh,
    AxisType as AxisType,
    get_abstract_mesh as get_abstract_mesh,
    use_abstract_mesh as use_abstract_mesh,
)
from jax._src.core import ManualAxisType as ManualAxisType
from jax._src.config import no_collectives as no_collectives
from jax._src.pjit import (
    reshard as reshard,
    auto_axes as auto_axes,
    explicit_axes as explicit_axes,
)

_deprecations = {
    # Added for v0.12.1
    "SingleDeviceSharding": (
        (
            "jax.sharding.SingleDeviceSharding is deprecated; use"
            " jax.sharding.make_single_device_sharding or"
            " jax.sharding.is_single_device_sharding instead."
        ),
        _deprecated_SingleDeviceSharding,
    ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  SingleDeviceSharding = _deprecated_SingleDeviceSharding
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _deprecated_SingleDeviceSharding
