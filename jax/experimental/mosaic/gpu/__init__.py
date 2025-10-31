# Copyright 2024 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from jax import ShapeDtypeStruct as ShapeDtypeStruct
from jax._src.lib import mosaic_gpu_dialect as dialect  # noqa: F401

# The imports below shadow the module, so we need to rename it.
from . import wgmma as _wgmma  # noqa: F401

from .core import (
    Barrier as Barrier,
    ClusterBarrier as ClusterBarrier,
    TMABarrier as TMABarrier,
    LoweringSemantics as LoweringSemantics,
    TMEM as TMEM,
    Union as Union,
    as_gpu_kernel as as_gpu_kernel,
    as_torch_gpu_kernel as as_torch_gpu_kernel,
    supports_cross_device_collectives as supports_cross_device_collectives,
)

from .launch_context import (
    AsyncCopyImplementation as AsyncCopyImplementation,
    GLOBAL_BROADCAST as GLOBAL_BROADCAST,
    LaunchContext as LaunchContext,
    MemRefTransform as MemRefTransform,
    TMAReductionOp as TMAReductionOp,
    Rounding as Rounding,
    TileTransform as TileTransform,
    TransposeTransform as TransposeTransform,
)

from .dialect_lowering import (
    lower_mgpu_dialect as lower_mgpu_dialect,
)

from .layout_inference import (
    infer_layout as infer_layout,
)

from .layouts import (
    to_layout_attr as to_layout_attr,
)

from .fragmented_array import (
    FragmentedArray as FragmentedArray,
    FragmentedLayout as FragmentedLayout,
    TCGEN05_LAYOUT as TCGEN05_LAYOUT,
    TCGEN05_TRANSPOSED_LAYOUT as TCGEN05_TRANSPOSED_LAYOUT,
    TCGEN05_ROW_LAYOUT as TCGEN05_ROW_LAYOUT,
    TCGEN05_COL_LAYOUT as TCGEN05_COL_LAYOUT,
    TiledLayout as TiledLayout,
    WGMMA_LAYOUT as WGMMA_LAYOUT,
    WGMMA_ROW_LAYOUT as WGMMA_ROW_LAYOUT,
    WGMMA_COL_LAYOUT as WGMMA_COL_LAYOUT,
    WGMMA_TRANSPOSED_LAYOUT as WGMMA_TRANSPOSED_LAYOUT,
    WGMMA_LAYOUT_UPCAST_2X as WGMMA_LAYOUT_UPCAST_2X,
    WGMMA_LAYOUT_UPCAST_4X as WGMMA_LAYOUT_UPCAST_4X,
    TMEM_NATIVE_LAYOUT as TMEM_NATIVE_LAYOUT,
    TMA_GATHER_INDICES_LAYOUT as TMA_GATHER_INDICES_LAYOUT,
    tmem_native_layout as tmem_native_layout,
    WGSplatFragLayout as WGSplatFragLayout,
    WGStridedFragLayout as WGStridedFragLayout,
    copy_tiled as copy_tiled,
    optimization_barrier as optimization_barrier,
)
from .utils import (
    BarrierRef as BarrierRef,
    DialectBarrierRef as DialectBarrierRef,
    CollectiveBarrierRef as CollectiveBarrierRef,
    DynamicSlice as DynamicSlice,
    Partition as Partition,
    Partition1D as Partition1D,
    SemaphoreRef as SemaphoreRef,
    ThreadSubset as ThreadSubset,
    MultimemReductionOp as MultimemReductionOp,
    bitwidth as bitwidth,
    bytewidth as bytewidth,
    c as c,
    commit_shared as commit_shared,
    debug_print as debug_print,
    ds as ds,
    fori as fori,
    is_known_divisible as is_known_divisible,
    memref_fold as memref_fold,
    memref_slice as memref_slice,
    memref_reshape as memref_reshape,
    memref_transpose as memref_transpose,
    memref_unfold as memref_unfold,
    memref_unsqueeze as memref_unsqueeze,
    nanosleep as nanosleep,
    query_cluster_cancel as query_cluster_cancel,
    single_thread as single_thread,
    single_thread_predicate as single_thread_predicate,
    system_memory_barrier as system_memory_barrier,
    thread_idx as thread_idx,
    tile_shape as tile_shape,
    try_cluster_cancel as try_cluster_cancel,
    warp_idx as warp_idx,
    warpgroup_barrier as warpgroup_barrier,
    warpgroup_idx as warpgroup_idx,
    when as when,
)
from .mma import (
    MMALayouts as MMALayouts,
    mma as mma,
)
from .wgmma import (
    WGMMAAccumulator as WGMMAAccumulator,
    wgmma as wgmma,
)

from . import tcgen05 as tcgen05
