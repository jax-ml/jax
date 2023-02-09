# Copyright 2018 The JAX Authors.
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

# TODO(https://github.com/google/jax/issues/13487): Remove PartitionSpec in
# 3 months from `jax.experimental.PartitionSpec`.
from jax.experimental.x64_context import (
  enable_x64 as enable_x64,
  disable_x64 as disable_x64,
)
from jax._src.callback import (
  io_callback as io_callback
)

# Deprecations

from jax._src.interpreters.pxla import (
  PartitionSpec as _deprecated_PartitionSpec,
)

import typing
if typing.TYPE_CHECKING:
  from jax._src.interpreters.pxla import (
    PartitionSpec as PartitionSpec,
  )
del typing

_deprecations = {
  # Added Feb 8, 2023:
  "PartitionSpec": (
    ("jax.experimental.PartitionSpec is deprecated. Use "
     "jax.sharding.PartitionSpec."),
    _deprecated_PartitionSpec,
  ),
}

from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr, _deprecations
