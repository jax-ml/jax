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

"""Experimental GPU backend for Pallas targeting Ampere, Hopper and Blackwell architectures.

This module is an alias of :mod:`jax.pallas.mosaic_gpu`, which is the preferred
import path.

These APIs are highly unstable and can change weekly. Use at your own risk.
"""

from jax.pallas.mosaic_gpu import *  # noqa: F403


_deprecations = {
    # Finalized in JAX v0.12.1
    # TODO(allanrenucci): remove for JAX v0.13.0.
    "transpose_ref": (
        (
            "jax.experimental.pallas.mosaic_gpu.transpose_ref was deprecated in"
            " JAX v0.11.2, and removed in JAX v0.12.1. Use ref.transpose(...)"
            " directly instead."
        ),
        None,
    ),
}

import typing

if not typing.TYPE_CHECKING:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr

  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
