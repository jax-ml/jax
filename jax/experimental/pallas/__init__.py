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

"""Module for Pallas, a JAX kernel language.

This module is an alias of :mod:`jax.pallas`, which is the preferred import
path.

See the Pallas documentation at
https://docs.jax.dev/en/latest/pallas/index.html.
"""

from jax.pallas import *  # noqa: F403

# Deprecated APIs not exported from jax.pallas.
from jax._src.pallas.core import core_map as _deprecated_core_map
from jax._src.pallas.primitives import DeviceIdType as _deprecated_DeviceIdType
from jax._src.pallas.primitives import reciprocal as _deprecated_reciprocal


_deprecations = {
    # Added September 28, 2026
    "DeviceIdType": (
        (
            "jax.experimental.pallas.DeviceIdType is deprecated and will be"
            " removed along with the device_id_type= parameter in semaphore and"
            " DMA operations."
        ),
        _deprecated_DeviceIdType,
    ),
    # Added August 17, 2026
    "reciprocal": (
        (
            "jax.experimental.pallas.reciprocal was moved to"
            " jax.experimental.pallas.tpu. Accessing it via"
            " jax.experimental.pallas is deprecated."
        ),
        _deprecated_reciprocal,
    ),
    # Added August 11, 2026
    "core_map": (
        (
            "jax.experimental.pallas.core_map is deprecated, use"
            " jax.experimental.pallas.kernel instead."
        ),
        _deprecated_core_map,
    ),
    # Finalized in JAX v0.12.0
    # TODO(slebedev): remove these for JAX v0.13.0.
    "dot": (
        (
            "pl.dot was deprecated in JAX v0.11.0, and removed in JAX v0.12.0."
            " Use jax.numpy.dot, jax.numpy.einsum or the @ operator instead in"
            " TPU or MGPU kernels."
        ),
        None,
    ),
    "debug_checks_enabled": (
        (
            "pl.debug_checks_enabled was deprecated in JAX v0.11.0, and removed"
            " in JAX v0.12.0. Use pl.enable_debug_checks instead."
        ),
        None,
    ),
}

import typing
if typing.TYPE_CHECKING:
  DeviceIdType = _deprecated_DeviceIdType
  core_map = _deprecated_core_map
  reciprocal = _deprecated_reciprocal
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
