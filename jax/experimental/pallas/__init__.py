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

See the Pallas documentation at
https://docs.jax.dev/en/latest/pallas.html.
"""

from jax._src.pallas.core import BlockDim as BlockDim
from jax._src.pallas.core import Blocked as Blocked
from jax._src.pallas.core import BlockSpec as BlockSpec
from jax._src.pallas.core import BoundedSlice as BoundedSlice
from jax._src.pallas.core import Buffered as Buffered
from jax._src.pallas.core import CompilerParams as CompilerParams
from jax._src.pallas.core import core_map as core_map
from jax._src.pallas.core import CostEstimate as CostEstimate
from jax._src.pallas.core import Element as Element
from jax._src.pallas.core import GridSpec as GridSpec
from jax._src.pallas.core import lower_as_mlir as lower_as_mlir
from jax._src.pallas.core import MemorySpace as MemorySpace
from jax._src.pallas.core import no_block_spec as no_block_spec
from jax._src.pallas.core import semaphore as semaphore
from jax._src.pallas.core import Squeezed as Squeezed
from jax._src.pallas.core import squeezed as squeezed
from jax._src.pallas.cost_estimate import estimate_cost as estimate_cost
from jax._src.pallas.helpers import debug_check as debug_check
from jax._src.pallas.helpers import debug_checks_enabled as debug_checks_enabled
from jax._src.pallas.helpers import empty as empty
from jax._src.pallas.helpers import empty_like as empty_like
from jax._src.pallas.helpers import empty_ref_like as empty_ref_like
from jax._src.pallas.helpers import enable_debug_checks as enable_debug_checks
from jax._src.pallas.helpers import loop as loop
from jax._src.pallas.helpers import when as when
from jax._src.pallas.pallas_call import pallas_call as pallas_call
from jax._src.pallas.pallas_call import pallas_call_p as pallas_call_p
from jax._src.pallas.primitives import atomic_add as _deprecated_atomic_add
from jax._src.pallas.primitives import atomic_and as _deprecated_atomic_and
from jax._src.pallas.primitives import atomic_cas as _deprecated_atomic_cas
from jax._src.pallas.primitives import atomic_max as _deprecated_atomic_max
from jax._src.pallas.primitives import atomic_min as _deprecated_atomic_min
from jax._src.pallas.primitives import atomic_or as _deprecated_atomic_or
from jax._src.pallas.primitives import atomic_xchg as _deprecated_atomic_xchg
from jax._src.pallas.primitives import atomic_xor as _deprecated_atomic_xor
from jax._src.pallas.primitives import debug_print as debug_print
from jax._src.pallas.primitives import DeviceIdType as DeviceIdType
from jax._src.pallas.primitives import dot as dot
from jax._src.pallas.primitives import load as _deprecated_load
from jax._src.pallas.primitives import max_contiguous as max_contiguous
from jax._src.pallas.primitives import multiple_of as multiple_of
from jax._src.pallas.primitives import num_programs as num_programs
from jax._src.pallas.primitives import program_id as program_id
from jax._src.pallas.primitives import reciprocal as reciprocal
from jax._src.pallas.primitives import run_scoped as run_scoped
from jax._src.pallas.primitives import semaphore_read as semaphore_read
from jax._src.pallas.primitives import semaphore_signal as semaphore_signal
from jax._src.pallas.primitives import semaphore_wait as semaphore_wait
from jax._src.pallas.primitives import store as _deprecated_store
from jax._src.pallas.primitives import swap as swap
from jax._src.pallas.utils import cdiv as cdiv
from jax._src.pallas.utils import next_power_of_2 as next_power_of_2
from jax._src.pallas.utils import strides_from_shape as strides_from_shape
from jax._src.state.discharge import run_state as run_state
from jax._src.state.indexing import ds as ds
from jax._src.state.indexing import dslice as dslice
from jax._src.state.indexing import Slice as Slice
from jax._src.state.primitives import broadcast_to as broadcast_to


ANY = MemorySpace.ANY
HOST = MemorySpace.HOST


import typing as _typing  # pylint: disable=g-import-not-at-top
if _typing.TYPE_CHECKING:
  atomic_add = _deprecated_atomic_add
  atomic_and = _deprecated_atomic_and
  atomic_cas = _deprecated_atomic_cas
  atomic_max = _deprecated_atomic_max
  atomic_min = _deprecated_atomic_min
  atomic_or = _deprecated_atomic_or
  atomic_xchg = _deprecated_atomic_xchg
  atomic_xor = _deprecated_atomic_xor
  load = _deprecated_load
  store = _deprecated_store
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  _deprecations = {
      # Deprecated on July 25th 2025.
      "load": (
          "pl.load is deprecated, use ``ref[idx]`` or a backend-specific loading API instead.",
          _deprecated_load,
      ),
      "store": (
          "pl.store is deprecated, use ``ref[idx] = value`` or a backend-specific storing API instead.",
          _deprecated_store,
      ),
      # Deprecated on July 23rd 2025.
      "atomic_add": (
          "pl.atomic_add is deprecated, access it through jax.experimental.pallas.triton.",
          _deprecated_atomic_add,
      ),
      "atomic_and": (
          "pl.atomic_and is deprecated, access it through jax.experimental.pallas.triton.",
          _deprecated_atomic_and,
      ),
      "atomic_cas": (
          "pl.atomic_cas is deprecated, access it through jax.experimental.pallas.triton.",
          _deprecated_atomic_cas,
      ),
      "atomic_max": (
          "pl.atomic_max is deprecated, access it through jax.experimental.pallas.triton.",
          _deprecated_atomic_max,
      ),
      "atomic_min": (
          "pl.atomic_min is deprecated, access it through jax.experimental.pallas.triton.",
          _deprecated_atomic_min,
      ),
      "atomic_or": (
          "pl.atomic_or is deprecated, access it through jax.experimental.pallas.triton.",
          _deprecated_atomic_or,
      ),
      "atomic_xchg": (
          "pl.atomic_xchg is deprecated, access it through jax.experimental.pallas.triton.",
          _deprecated_atomic_xchg,
      ),
      "atomic_xor": (
          "pl.atomic_xor is deprecated, access it through jax.experimental.pallas.triton.",
          _deprecated_atomic_xor,
      ),
  }
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
