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

"""Mosaic-specific Pallas APIs."""
import typing

from jax._src.pallas.mosaic import core as core
from jax._src.pallas.mosaic.core import create_tensorcore_mesh as create_tensorcore_mesh
from jax._src.pallas.mosaic.core import dma_semaphore as dma_semaphore
from jax._src.pallas.mosaic.core import GridDimensionSemantics as GridDimensionSemantics
from jax._src.pallas.mosaic.core import KernelType as KernelType
from jax._src.pallas.mosaic.core import PrefetchScalarGridSpec as PrefetchScalarGridSpec
from jax._src.pallas.mosaic.core import SemaphoreType as SemaphoreType
from jax._src.pallas.mosaic.core import SideEffectType as SideEffectType
from jax._src.pallas.mosaic.core import MemorySpace as MemorySpace
from jax._src.pallas.mosaic.core import CompilerParams as CompilerParams
from jax._src.pallas.mosaic.helpers import sync_copy as sync_copy
from jax._src.pallas.mosaic.helpers import core_barrier as core_barrier
from jax._src.pallas.mosaic.helpers import run_on_first_core as run_on_first_core
from jax._src.pallas.mosaic.interpret.interpret_pallas_call import InterpretParams as InterpretParams
from jax._src.pallas.mosaic.interpret.interpret_pallas_call import force_tpu_interpret_mode as force_tpu_interpret_mode
from jax._src.pallas.mosaic.interpret.interpret_pallas_call import reset_tpu_interpret_mode_state as reset_tpu_interpret_mode_state
from jax._src.pallas.mosaic.interpret.interpret_pallas_call import set_tpu_interpret_mode as set_tpu_interpret_mode
from jax._src.pallas.mosaic.lowering import LoweringException as LoweringException
from jax._src.pallas.mosaic.pipeline import BufferedRef as BufferedRef
from jax._src.pallas.mosaic.pipeline import BufferedRefBase as BufferedRefBase
from jax._src.pallas.mosaic.pipeline import emit_pipeline as emit_pipeline
from jax._src.pallas.mosaic.pipeline import emit_pipeline_with_allocations as emit_pipeline_with_allocations
from jax._src.pallas.mosaic.pipeline import get_pipeline_schedule as get_pipeline_schedule
from jax._src.pallas.mosaic.pipeline import make_pipeline_allocations as make_pipeline_allocations
from jax._src.pallas.mosaic.pipeline import Tiling as Tiling
from jax._src.pallas.mosaic.primitives import async_copy as async_copy
from jax._src.pallas.mosaic.primitives import async_remote_copy as async_remote_copy
from jax._src.pallas.mosaic.primitives import bitcast as bitcast
from jax._src.pallas.mosaic.primitives import get_barrier_semaphore as get_barrier_semaphore
from jax._src.pallas.mosaic.primitives import load as load
from jax._src.pallas.mosaic.primitives import make_async_copy as make_async_copy
from jax._src.pallas.mosaic.primitives import make_async_remote_copy as make_async_remote_copy
from jax._src.pallas.mosaic.primitives import pack_elementwise as pack_elementwise
from jax._src.pallas.mosaic.primitives import prng_random_bits as prng_random_bits
from jax._src.pallas.mosaic.primitives import prng_seed as prng_seed
from jax._src.pallas.mosaic.primitives import repeat as repeat
from jax._src.pallas.mosaic.primitives import roll as roll
from jax._src.pallas.mosaic.primitives import stochastic_round as stochastic_round
from jax._src.pallas.mosaic.primitives import store as store
from jax._src.pallas.mosaic.primitives import touch as touch
from jax._src.pallas.mosaic.primitives import unpack_elementwise as unpack_elementwise
from jax._src.pallas.mosaic.primitives import with_memory_space_constraint as with_memory_space_constraint
from jax._src.pallas.mosaic.random import sample_block as sample_block
from jax._src.pallas.mosaic.random import stateful_bernoulli as stateful_bernoulli
from jax._src.pallas.mosaic.random import stateful_bits as stateful_bits
from jax._src.pallas.mosaic.random import stateful_normal as stateful_normal
from jax._src.pallas.mosaic.random import stateful_uniform as stateful_uniform
from jax._src.pallas.mosaic.random import to_pallas_key as to_pallas_key
from jax._src.pallas.mosaic.tpu_info import ChipVersion as ChipVersion
from jax._src.pallas.mosaic.tpu_info import get_tpu_info as get_tpu_info
from jax._src.pallas.mosaic.tpu_info import is_tpu_device as is_tpu_device
from jax._src.pallas.mosaic.tpu_info import TpuInfo as TpuInfo

# Those primitives got moved to Pallas core. Keeping the updated imports
# here for backward compatibility.
from jax._src.pallas import primitives as pl_primitives
from jax._src.pallas.core import semaphore as semaphore
from jax._src.pallas.core import MemorySpace as GeneralMemorySpace
from jax._src.pallas.primitives import DeviceIdType as DeviceIdType
from jax._src.pallas.primitives import semaphore_read as semaphore_read
from jax._src.pallas.primitives import semaphore_signal as semaphore_signal
from jax._src.pallas.primitives import semaphore_wait as semaphore_wait

PARALLEL = GridDimensionSemantics.PARALLEL
CORE_PARALLEL = GridDimensionSemantics.CORE_PARALLEL
SUBCORE_PARALLEL = GridDimensionSemantics.SUBCORE_PARALLEL
ARBITRARY = GridDimensionSemantics.ARBITRARY

CMEM = MemorySpace.CMEM
SMEM = MemorySpace.SMEM
VMEM = MemorySpace.VMEM
VMEM_SHARED = MemorySpace.VMEM_SHARED
HBM = MemorySpace.HBM
HOST = MemorySpace.HOST
SEMAPHORE = MemorySpace.SEMAPHORE

_deprecations = {
    # Added Oct 31, 2025
    "delay": (
      "pltpu.delay is deprecated, use pl.delay instead.",
      pl_primitives.delay
    ),
    # Added Dec 10, 2025
    "ANY": (
        "pltpu.ANY is deprecated, use pl.ANY instead.",
        GeneralMemorySpace.ANY
    ),
}

if typing.TYPE_CHECKING:
  delay = pl_primitives.delay
  ANY = GeneralMemorySpace.ANY
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del pl_primitives
del GeneralMemorySpace
