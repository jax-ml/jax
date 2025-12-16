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
  pjit as _deprecated_pjit,
)
from jax._src.sharding_impls import (
  AUTO as AUTO,
)

_deprecations = {
    # Added Oct 13, 2025
    "pjit": (
        (
            "jax.experimental.pjit.pjit has been deprecated. Please use"
            " `jax.jit` instead."
        ),
        _deprecated_pjit,
    )
}

import typing as _typing
if _typing.TYPE_CHECKING:
  pjit = _deprecated_pjit
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
