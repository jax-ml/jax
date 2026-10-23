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

The stable subset of this module is available as ``jax.pallas.tpu``, which is
the preferred import path. APIs that are not exported from ``jax.pallas.tpu``
remain experimental.
"""
import typing

from jax.pallas.tpu import *  # noqa: F403

# Experimental APIs which are not (yet) exported from jax.pallas.tpu.
from jax._src.pallas.mosaic import core as core
from jax._src.pallas.mosaic.core import CoreType as CoreType
from jax._src.pallas.mosaic.core import dma_semaphore as dma_semaphore
from jax._src.pallas.mosaic.core import GridDimensionSemantics as _GridDimensionSemantics
from jax._src.pallas.mosaic.core import MemorySpace as _MemorySpace
from jax._src.pallas.mosaic.core import PrefetchScalarGridSpec as PrefetchScalarGridSpec
from jax._src.pallas.mosaic.helpers import join_bits as join_bits
from jax._src.pallas.mosaic.helpers import pack_fp6 as pack_fp6
from jax._src.pallas.mosaic.helpers import split_bits as split_bits
from jax._src.pallas.mosaic.helpers import unpack_fp6 as unpack_fp6
from jax._src.pallas.mosaic.lowering import LoweringException as LoweringException
from jax._src.pallas.mosaic.pipeline import BufferedRefBase as _deprecated_BufferedRefBase
from jax._src.pallas.mosaic.prefetch import emit_pipeline_with_async_prefetch as emit_pipeline_with_async_prefetch
from jax._src.pallas.mosaic.primitives import conv as conv
from jax._src.pallas.mosaic.primitives import has_memory_space as has_memory_space
from jax._src.pallas.mosaic.primitives import load as load
from jax._src.pallas.mosaic.primitives import matmul_acc_lhs as matmul_acc_lhs
from jax._src.pallas.mosaic.primitives import matmul_lhs_fifo as matmul_lhs_fifo
from jax._src.pallas.mosaic.primitives import matmul_pop_fifo as matmul_pop_fifo
from jax._src.pallas.mosaic.primitives import with_memory_space_constraint as with_memory_space_constraint
from jax._src.pallas.mosaic.random import stateful_bernoulli as stateful_bernoulli
from jax._src.pallas.mosaic.random import stateful_bits as stateful_bits
from jax._src.pallas.mosaic.random import stateful_normal as stateful_normal
from jax._src.pallas.primitives import reciprocal as reciprocal

CORE_PARALLEL = _GridDimensionSemantics.CORE_PARALLEL
SUBCORE_PARALLEL = _GridDimensionSemantics.SUBCORE_PARALLEL
CMEM = _MemorySpace.CMEM


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
        core.create_tensorcore_mesh,
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
  from jax._src.pallas.mosaic.core import create_tensorcore_mesh as create_tensorcore_mesh
  BufferedRefBase = _deprecated_BufferedRefBase
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
