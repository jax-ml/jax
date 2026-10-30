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

"""Mosaic-specific Pallas APIs.

This module is an alias of :mod:`jax.pallas.tpu`, which is the preferred import
path.
"""
import typing

from jax.pallas.tpu import *  # noqa: F403

# Deprecated APIs not exported from jax.pallas.tpu.
from jax._src.pallas.mosaic.core import create_tensorcore_mesh as _deprecated_create_tensorcore_mesh
from jax._src.pallas.mosaic.pipeline import BufferedRefBase as _deprecated_BufferedRefBase


_deprecations = {
    # Added October 13, 2026
    "BufferedRefBase": (
        "pltpu.BufferedRefBase is deprecated. It is internal to the pipeline"
        " implementation and should not be used.",
        _deprecated_BufferedRefBase,
    ),
    # Added June 30, 2026
    "create_tensorcore_mesh": (
        "pltpu.create_tensorcore_mesh is deprecated, use pltpu.TensorCoreMesh"
        " instead.",
        _deprecated_create_tensorcore_mesh,
    ),
    # Finalized in JAX v0.12.0
    # TODO(slebedev): remove these for JAX v0.13.0.
    "HOST": (
        "pltpu.HOST was deprecated in JAX v0.10.0, and removed in JAX v0.12.0."
        " Use pl.HOST instead.",
        None,
    ),
}

if typing.TYPE_CHECKING:
  BufferedRefBase = _deprecated_BufferedRefBase
  create_tensorcore_mesh = _deprecated_create_tensorcore_mesh
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
