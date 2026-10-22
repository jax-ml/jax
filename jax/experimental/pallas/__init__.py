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

"""Module for Pallas, a JAX extension for custom kernels.

The stable subset of this module is available as ``jax.pallas``, which is the
preferred import path. APIs that are not exported from ``jax.pallas`` remain
experimental.

See the Pallas documentation at
https://docs.jax.dev/en/latest/pallas/index.html.
"""

from jax.pallas import *  # noqa: F403

# Experimental APIs which are not (yet) exported from jax.pallas.
from jax._src.pallas.core import core_map as _deprecated_core_map
from jax._src.pallas.core import enable_poison_buffers as enable_poison_buffers
from jax._src.pallas.core import GridSpec as GridSpec
from jax._src.pallas.core import lower_as_mlir as lower_as_mlir
from jax._src.pallas.core import pallas_export_experimental as pallas_export_experimental
from jax._src.pallas.core import RevisitMode as RevisitMode
from jax._src.pallas.core import semaphore as semaphore
from jax._src.pallas.core import squeezed as squeezed
from jax._src.pallas.pallas_call import pallas_call as pallas_call
from jax._src.pallas.pallas_call import pallas_call_p as pallas_call_p
from jax._src.pallas.primitives import DeviceIdType as _deprecated_DeviceIdType
from jax._src.pallas.primitives import get_global as get_global
from jax._src.pallas.primitives import reciprocal as _deprecated_reciprocal
from jax._src.pallas.utils import next_power_of_2 as next_power_of_2
from jax._src.pallas.utils import strides_from_shape as strides_from_shape
from jax._src.state.indexing import dslice as dslice
from jax._src.state.primitives import broadcast_to as broadcast_to


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
