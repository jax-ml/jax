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

"""Experimental GPU backend for Pallas targeting H100.

These APIs are highly unstable and can change weekly. Use at your own risk.
"""

from jax._src.pallas.mosaic_gpu.core import Barrier as Barrier
from jax._src.pallas.mosaic_gpu.core import ClusterBarrier as ClusterBarrier
from jax._src.pallas.mosaic_gpu.core import BlockSpec as BlockSpec
from jax._src.pallas.mosaic_gpu.core import CompilerParams as CompilerParams
from jax._src.pallas.mosaic_gpu.core import Layout as Layout
from jax._src.pallas.mosaic_gpu.core import layout_cast as layout_cast
from jax._src.pallas.mosaic_gpu.core import Mesh as Mesh
from jax._src.pallas.mosaic_gpu.core import MemorySpace as MemorySpace
from jax._src.pallas.mosaic_gpu.core import kernel as kernel
from jax._src.pallas.mosaic_gpu.core import PeerMemRef as PeerMemRef
from jax._src.pallas.mosaic_gpu.core import RefUnion as RefUnion
from jax._src.pallas.mosaic_gpu.core import remote_ref as remote_ref
from jax._src.pallas.mosaic_gpu.core import SemaphoreType as SemaphoreType
from jax._src.pallas.mosaic_gpu.core import SwizzleTransform as SwizzleTransform
from jax._src.pallas.mosaic_gpu.core import TilingTransform as TilingTransform
from jax._src.pallas.mosaic_gpu.core import TMEMLayout as TMEMLayout
from jax._src.pallas.mosaic_gpu.core import transform_ref as transform_ref
from jax._src.pallas.mosaic_gpu.core import transpose_ref as transpose_ref
from jax._src.pallas.mosaic_gpu.core import untile_ref as untile_ref
from jax._src.pallas.mosaic_gpu.core import unswizzle_ref as unswizzle_ref
from jax._src.pallas.mosaic_gpu.core import TransposeTransform as TransposeTransform
from jax._src.pallas.mosaic_gpu.core import WarpMesh as WarpMesh
from jax._src.pallas.mosaic_gpu.core import WGMMAAccumulatorRef as ACC  # noqa: F401
from jax._src.pallas.mosaic_gpu.core import WGMMAAccumulatorRef as WGMMAAccumulatorRef
from jax._src.pallas.mosaic_gpu.helpers import nd_loop as nd_loop
from jax._src.pallas.mosaic_gpu.helpers import find_swizzle as find_swizzle
from jax._src.pallas.mosaic_gpu.helpers import format_tcgen05_sparse_metadata as format_tcgen05_sparse_metadata
from jax._src.pallas.mosaic_gpu.pipeline import emit_pipeline as emit_pipeline
from jax._src.pallas.mosaic_gpu.pipeline import emit_pipeline_warp_specialized as emit_pipeline_warp_specialized
from jax._src.pallas.mosaic_gpu.primitives import async_copy_scales_to_tmem as async_copy_scales_to_tmem
from jax._src.pallas.mosaic_gpu.primitives import async_copy_sparse_metadata_to_tmem as async_copy_sparse_metadata_to_tmem
from jax._src.pallas.mosaic_gpu.primitives import async_load_tmem as async_load_tmem
from jax._src.pallas.mosaic_gpu.primitives import async_store_tmem as async_store_tmem
from jax._src.pallas.mosaic_gpu.primitives import barrier_arrive as barrier_arrive
from jax._src.pallas.mosaic_gpu.primitives import barrier_wait as barrier_wait
from jax._src.pallas.mosaic_gpu.primitives import broadcasted_iota as broadcasted_iota
from jax._src.pallas.mosaic_gpu.primitives import commit_smem as commit_smem
from jax._src.pallas.mosaic_gpu.primitives import commit_smem_to_gmem_group as commit_smem_to_gmem_group
from jax._src.pallas.mosaic_gpu.primitives import ShapeDtypeStruct as ShapeDtypeStruct
from jax._src.pallas.mosaic_gpu.primitives import copy_gmem_to_smem as copy_gmem_to_smem
from jax._src.pallas.mosaic_gpu.primitives import copy_smem_to_gmem as copy_smem_to_gmem
from jax._src.pallas.mosaic_gpu.primitives import inline_mgpu as inline_mgpu
from jax._src.pallas.mosaic_gpu.primitives import load as load
from jax._src.pallas.mosaic_gpu.primitives import print_layout as print_layout
from jax._src.pallas.mosaic_gpu.primitives import RefType as RefType
from jax._src.pallas.mosaic_gpu.primitives import semaphore_signal_parallel as semaphore_signal_parallel
from jax._src.pallas.mosaic_gpu.primitives import SemaphoreSignal as SemaphoreSignal
from jax._src.pallas.mosaic_gpu.primitives import set_max_registers as set_max_registers
from jax._src.pallas.mosaic_gpu.primitives import wait_load_tmem as wait_load_tmem
from jax._src.pallas.mosaic_gpu.primitives import wait_smem_to_gmem as wait_smem_to_gmem
from jax._src.pallas.mosaic_gpu.primitives import wgmma as wgmma
from jax._src.pallas.mosaic_gpu.primitives import wgmma_wait as wgmma_wait
from jax._src.pallas.mosaic_gpu.primitives import tcgen05_mma as tcgen05_mma
from jax._src.pallas.mosaic_gpu.primitives import tcgen05_commit_arrive as tcgen05_commit_arrive
from jax._src.pallas.mosaic_gpu.primitives import commit_tmem as commit_tmem
from jax.experimental.mosaic.gpu.core import LoweringSemantics as LoweringSemantics


#: Alias of :data:`jax.experimental.pallas.mosaic_gpu.MemorySpace.GMEM`.
GMEM = MemorySpace.GMEM
#: Alias of :data:`jax.experimental.pallas.mosaic_gpu.MemorySpace.SMEM`.
SMEM = MemorySpace.SMEM
#: Alias of :data:`jax.experimental.pallas.mosaic_gpu.MemorySpace.TMEM`.
TMEM = MemorySpace.TMEM
#: Alias of :data:`jax.experimental.pallas.mosaic_gpu.MemorySpace.REGS`.
REGS = MemorySpace.REGS
