# Copyright 2020 The JAX Authors.
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

from jax._src.maps import (
  AxisName as AxisName,
  ResourceSet as ResourceSet,
  SerialLoop as SerialLoop,
  make_xmap_callable as make_xmap_callable,
  serial_loop as serial_loop,
  xmap as xmap,
  xmap_p as xmap_p,
  _prepare_axes as _prepare_axes,
)
from jax._src.mesh import (
  EMPTY_ENV as EMPTY_ENV,
  ResourceEnv as ResourceEnv,
  thread_resources as thread_resources,
)

# Deprecations

from jax._src.maps import (
  Mesh as _deprecated_Mesh,
)

import typing
if typing.TYPE_CHECKING:
  from jax._src.maps import (
    Mesh as Mesh,
  )
del typing

_deprecations = {
  # Added Feb 13, 2023:
  "Mesh": (
    "jax.experimental.maps.Mesh is deprecated. Use jax.sharding.Mesh.",
    _deprecated_Mesh,
  ),
}

from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
