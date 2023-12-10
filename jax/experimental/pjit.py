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

# ruff: noqa

from jax._src.pjit import (
  pjit as pjit,
  pjit_p as pjit_p,
  with_sharding_constraint as _deprecated_with_sharding_constraint,
)
from jax._src.sharding_impls import (
  AUTO as AUTO,
  UNSPECIFIED as _UNSPECIFIED,
)

from jax._src.pjit import _pjit_lower_cached, _pjit_lower

_deprecations = {
    # Added September 14, 2023
    "with_sharding_constraint": (
        ("jax.experimental.pjit.with_sharding_constraint is deprecated."
         " Please use jax.lax.with_sharding_constraint."),
        _deprecated_with_sharding_constraint,
    )
}

import typing
if typing.TYPE_CHECKING:
  with_sharding_constraint = _deprecated_with_sharding_constraint
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
