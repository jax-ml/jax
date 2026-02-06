# Copyright 2025 The JAX Authors.
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
"""TPU SparseCore Extensions to Pallas."""

from jax._src.pallas.mosaic.sc_core import BlockSpec as BlockSpec
from jax._src.pallas.mosaic.sc_core import get_sparse_core_info as get_sparse_core_info
from jax._src.pallas.mosaic.sc_core import MemoryRef as MemoryRef
from jax._src.pallas.mosaic.sc_core import ScalarSubcoreMesh as ScalarSubcoreMesh
from jax._src.pallas.mosaic.sc_core import VectorSubcoreMesh as VectorSubcoreMesh
from jax._src.pallas.mosaic.sc_primitives import addupdate as addupdate
from jax._src.pallas.mosaic.sc_primitives import addupdate_compressed as addupdate_compressed
from jax._src.pallas.mosaic.sc_primitives import addupdate_scatter as addupdate_scatter
from jax._src.pallas.mosaic.sc_primitives import all_reduce_ffs as all_reduce_ffs
from jax._src.pallas.mosaic.sc_primitives import all_reduce_population_count as all_reduce_population_count
from jax._src.pallas.mosaic.sc_primitives import bitcast as bitcast
from jax._src.pallas.mosaic.sc_primitives import cummax as cummax
from jax._src.pallas.mosaic.sc_primitives import cumsum as cumsum
from jax._src.pallas.mosaic.sc_primitives import fetch_and_add as fetch_and_add
from jax._src.pallas.mosaic.sc_primitives import load_expanded as load_expanded
from jax._src.pallas.mosaic.sc_primitives import load_gather as load_gather
from jax._src.pallas.mosaic.sc_primitives import pack as pack
from jax._src.pallas.mosaic.sc_primitives import PackFormat as PackFormat
from jax._src.pallas.mosaic.sc_primitives import parallel_loop as parallel_loop
from jax._src.pallas.mosaic.sc_primitives import scan_count as scan_count
from jax._src.pallas.mosaic.sc_primitives import sort_key_val as sort_key_val
from jax._src.pallas.mosaic.sc_primitives import store_compressed as store_compressed
from jax._src.pallas.mosaic.sc_primitives import store_scatter as store_scatter
from jax._src.pallas.mosaic.sc_primitives import subcore_barrier as subcore_barrier
from jax._src.pallas.mosaic.sc_primitives import unpack as unpack
