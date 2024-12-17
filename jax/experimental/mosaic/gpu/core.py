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

from collections.abc import Callable, Sequence
import contextlib
import ctypes
import dataclasses
import enum
import functools
import hashlib
import math
import os
import pathlib
import time
from typing import Any, Generic, TypeVar
import weakref

import jax
from jax._src.interpreters import mlir
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import builtin
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
import numpy as np

from jax._src.lib import mosaic_gpu_dialect as dialect  # noqa: F401

if dialect is not None:
  from . import dialect_lowering
  from . import layout_inference
else:
  dialect_lowering = None
  layout_inference = None

from . import profiler
from . import utils

# mypy: ignore-errors

# MLIR can't find libdevice unless we point it to the CUDA path
# TODO(apaszke): Unify with jax._src.lib.cuda_path
CUDA_ROOT = "/usr/local/cuda"
if os.environ.get("CUDA_ROOT") is None:
  os.environ["CUDA_ROOT"] = CUDA_ROOT
else:
  CUDA_ROOT = os.environ["CUDA_ROOT"]

PTXAS_PATH = os.path.join(CUDA_ROOT, "bin/ptxas")
NVDISASM_PATH = os.path.join(CUDA_ROOT, "bin/nvdisasm")

TMA_DESCRIPTOR_BYTES = 128
TMA_DESCRIPTOR_ALIGNMENT = 64


c = utils.c  # This is too common to fully qualify.


RUNTIME_PATH = None
try:
  from jax._src.lib import mosaic_gpu as mosaic_gpu_lib

  RUNTIME_PATH = (
      pathlib.Path(mosaic_gpu_lib._mosaic_gpu_ext.__file__).parent
      / "libmosaic_gpu_runtime.so"
  )
except ImportError:
  pass

if RUNTIME_PATH and RUNTIME_PATH.exists():
  # Set this so that the custom call can find it
  os.environ["MOSAIC_GPU_RUNTIME_LIB_PATH"] = str(RUNTIME_PATH)


mosaic_gpu_p = jax._src.core.Primitive("mosaic_gpu_p")
mosaic_gpu_p.multiple_results = True


@mosaic_gpu_p.def_abstract_eval
def _mosaic_gpu_abstract_eval(*_, module, out_types):
  del module # Unused.
  return [jax._src.core.ShapedArray(t.shape, t.dtype) for t in out_types]

# TODO(apaszke): Implement a proper system for managing kernel lifetimes
KNOWN_KERNELS = {}


def _mosaic_gpu_lowering_rule(
    ctx,
    *args,
    module,
    out_types,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
):
  assert len(out_types) == len(ctx.avals_out)
  kernel_id = hashlib.sha256(module).digest()
  # Note that this is technically only a half measure. Someone might load a
  # compiled module with a hash collision from disk. But that's so unlikely with
  # SHA256 that it shouldn't be a problem.
  if (kernel_text := KNOWN_KERNELS.get(kernel_id, None)) is not None:
    if kernel_text != module:
      raise RuntimeError("Hash collision!")
  else:
    KNOWN_KERNELS[kernel_id] = module
  op = mlir.custom_call(
      "mosaic_gpu",
      result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
      operands=args,
      operand_layouts=[list(reversed(range(a.ndim))) for a in ctx.avals_in],
      result_layouts=[list(reversed(range(a.ndim))) for a in ctx.avals_out],
      backend_config=kernel_id + module,
      operand_output_aliases=dict(input_output_aliases),
  )
  return op.results


mlir.register_lowering(mosaic_gpu_p, _mosaic_gpu_lowering_rule, "cuda")


@dataclasses.dataclass(frozen=True)
class MemRefTransform:
  def apply(self, ref: ir.Value) -> ir.Value:
    raise NotImplementedError("Subclasses should override this method")

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    raise NotImplementedError("Subclasses should override this method")

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    raise NotImplementedError("Subclasses should override this method")

  def batch(self, leading_rank: int) -> 'MemRefTransform':
    """Returns a transform that accepts a ref with the extra `leading_rank` dims.

    The returned transform should leave the leading dimensions unchanged and
    only apply to the suffix of the shape.
    """
    raise NotImplementedError("Subclasses should override this method")


@dataclasses.dataclass(frozen=True)
class TileTransform(MemRefTransform):
  """Tiles a suffix of memref dimensions.

  For example, given a memref of shape (5, 128, 128) and a tiling of (64, 32),
  the shape of the result will be (5, 2, 4, 64, 32). The shape always ends with
  the tile shape, and the size of tiled dimensions is divided by the tile size.
  This is especially useful for swizzled WGMMA, which expect tiled layouts in
  shared memory.
  """
  tiling: tuple[int, ...]

  def apply(self, ref: ir.Value) -> ir.Value:
    untiled_rank = ir.MemRefType(ref.type).rank
    tiling_rank = len(self.tiling)
    tiled_rank = untiled_rank + tiling_rank
    for t, d in zip(self.tiling[::-1], range(untiled_rank)[::-1]):
      s = ir.MemRefType(ref.type).shape[d]
      if s % t and s > t:
        raise ValueError(
            f"Dimension {d} must have size smaller or a multiple of its tiling"
            f" {t}, but got {s}"
        )
      ref = utils.memref_unfold(ref, d, (None, min(t, s)))
    permutation = (
        *range(untiled_rank - tiling_rank),
        *range(untiled_rank - tiling_rank, tiled_rank, 2),
        *range(untiled_rank - tiling_rank + 1, tiled_rank, 2),
    )
    return utils.memref_transpose(ref, permutation)

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    index = ir.IndexType.get()
    tiling_rank = len(self.tiling)
    return (
        *idx[:-tiling_rank],
        *(
            arith.divui(i, c(t, index))
            for i, t in zip(idx[-tiling_rank:], self.tiling)
        ),
        *(
            arith.remui(i, c(t, index))
            for i, t in zip(idx[-tiling_rank:], self.tiling)
        ),
    )

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    # Note that this also checks that tiled dims are not squeezed. Their slice
    # size would be 1 if so.
    tiling_rank = len(self.tiling)
    for size, tile_size in zip(shape[-tiling_rank:], self.tiling):
      if size % tile_size:
        raise ValueError(
            f"Expected GMEM slice shape {shape} suffix to be a multiple of"
            f" tiling {self.tiling}.\nIf you're using padded async copies, your"
            " slice might need to extend out of bounds of the GMEM buffer (OOB"
            " accesses will be skipped)."
        )
    return (
        *shape[:-tiling_rank],
        *(s // t for s, t in zip(shape[-tiling_rank:], self.tiling)),
        *self.tiling,
    )

  def batch(self, leading_rank: int) -> MemRefTransform:
    return self


@dataclasses.dataclass(frozen=True)
class TransposeTransform(MemRefTransform):
  """Transposes memref dimensions."""
  permutation: tuple[int, ...]

  def __post_init__(self):
    if len(self.permutation) != len(set(self.permutation)):
      raise ValueError("Permutation must be a permutation")

  def apply(self, ref: ir.Value) -> ir.Value:
    return utils.memref_transpose(ref, self.permutation)

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    return tuple(idx[p] for p in self.permutation)

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(shape[p] for p in self.permutation)

  def batch(self, leading_rank: int) -> MemRefTransform:
    return TransposeTransform(
        (*range(leading_rank), *(d + leading_rank for d in self.permutation))
    )


@dataclasses.dataclass(frozen=True)
class CollapseLeadingIndicesTransform(MemRefTransform):
  """Collapses leading indices into one."""
  strides: tuple[int, ...]

  @functools.cached_property
  def common_stride(self) -> int:
    return math.gcd(*self.strides)

  def apply(self, ref: ir.Value) -> ir.Value:
    ref_ty = ir.MemRefType(ref.type)
    strides, offset = ref_ty.get_strides_and_offset()
    if offset == ir.ShapedType.get_dynamic_stride_or_offset():
      raise NotImplementedError("Dynamic offsets are not supported")
    max_bound = sum(
        (d - 1) * s // self.common_stride
        for d, s in zip(
            ref_ty.shape[: len(self.strides)], strides[: len(self.strides)]
        )
    ) + 1
    new_shape = [max_bound, *ref_ty.shape[len(self.strides):]]
    new_strides = [self.common_stride, *strides[len(self.strides):]]
    new_layout = ir.StridedLayoutAttr.get(offset, new_strides)
    new_ref_ty = ir.MemRefType.get(
        new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
    )
    return memref.reinterpret_cast(
        new_ref_ty, ref, [], [], [],
        static_offsets=[offset],
        static_sizes=new_shape,
        static_strides=new_strides,
    )

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    index = ir.IndexType.get()
    flat_idx = c(0, index)
    for i, s in zip(idx[:len(self.strides)], self.strides):
      flat_idx = arith.addi(
          flat_idx, arith.muli(i, c(s // self.common_stride, index))
      )
    return (flat_idx, *idx[len(self.strides):])

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    if any(s != 1 for s in shape[:len(self.strides)]):
      raise ValueError("Expected leading indices to be squeezed")
    return (1, *shape[len(self.strides):])

  def batch(self, leading_rank: int) -> MemRefTransform:
    raise NotImplementedError  # Unused


OnDeviceProfiler = profiler.OnDeviceProfiler


@dataclasses.dataclass()
class LaunchContext:
  launch_op: gpu.LaunchOp
  gmem_scratch_ptr: ir.Value
  cluster_size: tuple[int, int, int]
  profiler: OnDeviceProfiler | None = None
  next_scratch_offset: int = 0
  host_scratch_init: list[Callable[[ir.Value], None]] = dataclasses.field(
      default_factory=list, init=False
  )
  tma_descriptors: dict[
      tuple[ir.Value, tuple[int, ...], int | None, tuple[MemRefTransform, ...]],
      ir.Value,
  ] = dataclasses.field(default_factory=dict, init=False)

  @contextlib.contextmanager
  def named_region(self, *args, **kwargs):
    if self.profiler is not None:
      with self.profiler.record(*args, **kwargs):
        yield
    else:
      yield

  def _alloc_scratch(
      self,
      size: int,
      alignment: int | None = None,
      host_init: Callable[[ir.Value], None] = lambda _: None,
      device_init: Callable[[ir.Value], Any] = lambda x: x,
  ) -> ir.Value:
    """Allocates a GMEM scratch buffer.

    The buffer is initialized on the host and then copied to GMEM before the
    kernel launch.
    """
    i8 = ir.IntegerType.get_signless(8)
    ptr_ty = ir.Type.parse("!llvm.ptr")
    if alignment is None:
      alignment = size
    if self.next_scratch_offset % alignment:
      raise NotImplementedError  # TODO(apaszke): Pad to match alignment
    alloc_base = self.next_scratch_offset
    self.next_scratch_offset += size
    def host_init_wrapped(host_ptr):
      host_init(
          llvm.getelementptr(ptr_ty, host_ptr, [], [alloc_base], i8)
      )
    self.host_scratch_init.append(host_init_wrapped)
    # with ir.InsertionPoint(self.gmem_scratch_ptr.owner):
    # There is no way to create an insertion point after an operation...
    gep = llvm.GEPOp(
        ptr_ty, self.gmem_scratch_ptr, [], [alloc_base], i8
    )
    gep.move_after(self.gmem_scratch_ptr.owner)
    return device_init(gep.result)

  def _get_tma_desc(
      self,
      gmem_ref,
      gmem_transform: tuple[MemRefTransform, ...],
      transformed_slice_shape: tuple[int, ...],
      swizzle: int | None,
  ):
    tma_desc_key = (gmem_ref, transformed_slice_shape, swizzle, gmem_transform)
    if (tma_desc := self.tma_descriptors.get(tma_desc_key, None)) is None:
      i64 = ir.IntegerType.get_signless(64)
      ptr_ty = ir.Type.parse("!llvm.ptr")
      def init_tma_desc(host_ptr):
        ref = gmem_ref
        for t in gmem_transform:
          ref = t.apply(ref)
        ref_ty = ir.MemRefType(ref.type)
        # TODO(apaszke): Use utils.memref_ptr to compute base_ptr
        _, offset, *sizes_and_strides = memref.extract_strided_metadata(ref)
        aligned_ptr_idx = memref.extract_aligned_pointer_as_index(ref)
        as_i64 = lambda i: arith.index_cast(i64, i)
        alloc_ptr = llvm.inttoptr(ptr_ty, as_i64(aligned_ptr_idx))
        llvm_dyn = -2147483648  # TODO(apaszke): Improve the MLIR bindings...
        base_ptr = llvm.getelementptr(
            ptr_ty, alloc_ptr, [as_i64(offset)], [llvm_dyn], ref_ty.element_type,
        )
        rank = ref_ty.rank
        assert rank * 2 == len(sizes_and_strides)
        args = [
            host_ptr,
            base_ptr,
            c(utils.bytewidth(ref_ty.element_type), i64),
            c(rank, i64),
            utils.pack_array([as_i64(i) for i in sizes_and_strides[:rank]]),
            utils.pack_array([as_i64(i) for i in sizes_and_strides[rank:]]),
            c(0 if swizzle is None else swizzle, i64),
            utils.pack_array([c(v, i64) for v in transformed_slice_shape]),
        ]
        func.call([], "mosaic_gpu_init_tma_desc", args)
      def cast_tma_desc(device_ptr):
        # TODO(apaszke): Investigate why prefetching can cause launch failures
        # nvvm.prefetch_tensormap(device_ptr)
        return device_ptr
      tma_desc = self._alloc_scratch(
          TMA_DESCRIPTOR_BYTES,
          alignment=TMA_DESCRIPTOR_ALIGNMENT,
          host_init=init_tma_desc,
          device_init=cast_tma_desc,
      )
      self.tma_descriptors[tma_desc_key] = tma_desc
    return tma_desc

  def async_copy(
      self,
      *,
      src_ref,
      dst_ref,
      gmem_slice: Any = (),
      gmem_transform: MemRefTransform | tuple[MemRefTransform, ...] = (),
      barrier: utils.BarrierRef | None = None,
      swizzle: int | None = None,
      arrive: bool | None = None,
      uniform: bool = True,
      collective: Sequence[gpu.Dimension] | gpu.Dimension | None = None,
      predicate: ir.Value | None = None,  # Should select 0 or 1 threads from the WG.
  ):
    index = ir.IndexType.get()
    i16 = ir.IntegerType.get_signless(16)
    i32 = ir.IntegerType.get_signless(32)
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
    src_ref_ty = ir.MemRefType(src_ref.type)
    dst_ref_ty = ir.MemRefType(dst_ref.type)
    element_type = src_ref_ty.element_type
    element_bytewidth = utils.bytewidth(element_type)
    if element_type != dst_ref_ty.element_type:
      raise ValueError(
          f"Expected same element type, got {element_type} and"
          f" {dst_ref_ty.element_type}"
      )
    if predicate is not None and not uniform:
      raise ValueError("Predicate can only be defined when uniform is True")
    if not isinstance(gmem_transform, tuple):
      gmem_transform = (gmem_transform,)

    if src_ref_ty.memory_space is None and dst_ref_ty.memory_space == smem:
      gmem_ref, smem_ref = src_ref, dst_ref
      if barrier is None:
        raise ValueError("Barriers are required for GMEM -> SMEM copies")
      if arrive is None:
        arrive = True  # Arrive by default
    elif src_ref_ty.memory_space == smem and dst_ref_ty.memory_space is None:
      gmem_ref, smem_ref = dst_ref, src_ref
      if barrier is not None:
        raise ValueError("Barriers are unsupported for SMEM -> GMEM copies")
      if arrive is not None:
        raise ValueError("arrive is unsupported for SMEM -> GMEM copies")
    else:
      raise ValueError("Only SMEM <-> GMEM copies supported")
    # TODO(apaszke): This is a very approximate check. Improve it!
    expected_name = "builtin.unrealized_conversion_cast"
    if (
        gmem_ref.owner is None
        or gmem_ref.owner.opview.OPERATION_NAME != expected_name
    ):
      raise ValueError("GMEM reference in async_copy must be a kernel argument")
    gmem_ref_ty = ir.MemRefType(gmem_ref.type)
    gmem_strides, _ = gmem_ref_ty.get_strides_and_offset()
    if gmem_strides != utils.get_contiguous_strides(gmem_ref_ty.shape):
      raise NotImplementedError(
          "async_copy assumes the GMEM reference is contiguous"
      )
    if any(s * element_bytewidth % 16 != 0 for s in gmem_strides[:-1]):
      raise ValueError(
          "async_copy requires all GMEM strides except the last one to be a"
          " multiple of 16 bytes"
      )

    base_indices, slice_shape, is_squeezed = utils.parse_indices(
        gmem_slice, ir.MemRefType(gmem_ref.type).shape
    )
    dyn_base_indices = tuple(
        c(i, index) if not isinstance(i, ir.Value) else i for i in base_indices
    )
    squeezed_dims = [i for i, squeezed in enumerate(is_squeezed) if squeezed]
    sliced_dims = [i for i, squeezed in enumerate(is_squeezed) if not squeezed]
    # Indexing is really slicing + squeezing, and user transforms are meant to
    # apply after that. However, we actually have to apply the indexing last
    # (it's fused into the TMA) and so we need to commute it with all the user
    # transforms. For slicing this is done using transform_index and
    # transform_shape. For squeezing we actually move all the squeezed dims to
    # the front, and then batch each transform, making it ignore the extra dims.
    if squeezed_dims:
      gmem_transform = (TransposeTransform((*squeezed_dims, *sliced_dims)),
                        *(t.batch(len(squeezed_dims)) for t in gmem_transform))

    slice_shape = tuple(slice_shape)
    for t in gmem_transform:
      dyn_base_indices = t.transform_index(dyn_base_indices)
      slice_shape = t.transform_shape(slice_shape)

    num_squeezed_dims = len(squeezed_dims)
    if len(slice_shape) > 5:
      # We can try to collapse all squeezed dims into one.
      if len(slice_shape) - num_squeezed_dims + 1 > 5:
        raise ValueError(
            "Async copies only support striding up to 5 dimensions"
        )
      collapse = CollapseLeadingIndicesTransform(
          tuple(gmem_strides[d] for d in squeezed_dims)
      )
      gmem_transform = (*gmem_transform, collapse)
      dyn_base_indices = collapse.transform_index(dyn_base_indices)
      slice_shape = collapse.transform_shape(slice_shape)
      num_squeezed_dims = 1
    del squeezed_dims, sliced_dims  # Those no longer make sense.

    smem_ref_ty = ir.MemRefType(smem_ref.type)
    # We moved all squeezed dims to the front.
    if slice_shape[num_squeezed_dims:] != tuple(smem_ref_ty.shape):
      raise ValueError(
          "Expected the SMEM reference to have the same shape as the"
          f" transformed slice: {tuple(smem_ref_ty.shape)} != {slice_shape}"
      )
    smem_strides, _ = smem_ref_ty.get_strides_and_offset()
    if smem_strides != utils.get_contiguous_strides(smem_ref_ty.shape):
      raise ValueError(
          "async_copy needs the SMEM reference to be contiguous, but got"
          f" strides {smem_strides} for shape {smem_ref_ty.shape}"
      )

    dyn_base_indices = list(dyn_base_indices)
    slice_shape = list(slice_shape)
    assert all(d == 1 for d in slice_shape[:num_squeezed_dims])
    collective_size = 1
    if collective is not None:
      if isinstance(collective, gpu.Dimension):
        collective = (collective,)
      collective_size = math.prod(self.cluster_size[d] for d in collective)
    if collective_size > 1:
      def partition_dim(dim: int, idx: ir.Value, num_chunks: int):
        # No need to partition squeezed dims. They don't even exist in smem_ref.
        assert dim >= num_squeezed_dims
        nonlocal smem_ref
        slice_shape[dim] //= num_chunks
        block_offset = arith.muli(idx, c(slice_shape[dim], index))
        dyn_base_indices[dim] = arith.addi(dyn_base_indices[dim], block_offset)
        smem_ref = utils.memref_slice(
            smem_ref,
            (slice(None),) * (dim - num_squeezed_dims)
            + (utils.ds(block_offset, slice_shape[dim]),),
        )
      stride = 1
      idx = c(0, index)
      for d in sorted(collective):
        if self.cluster_size[d] == 1:  # Optimize a multiply by 0.
          continue
        idx = arith.addi(idx, arith.muli(gpu.cluster_block_id(d), c(stride, index)))
        stride *= self.cluster_size[d]
      rem_collective_size = collective_size
      for dim, slice_size in enumerate(slice_shape[:-1]):
        if slice_size % rem_collective_size == 0:
          partition_dim(dim, idx, rem_collective_size)
          rem_collective_size = 1
          break
        elif rem_collective_size % slice_size == 0:
          # This is an optimization and it lets us skip squeezed dims.
          if slice_size > 1:
            dim_idx = arith.remui(idx, c(slice_size, index))
            partition_dim(dim, dim_idx, slice_size)
            idx = arith.divui(idx, c(slice_size, index))
            rem_collective_size //= slice_size
        else:
          break  # We failed to partition the leading dimensions.
      del idx  # We overwrote the block index in the loop.
      if rem_collective_size > 1:
        raise ValueError(
            "None of the leading dimensions in the transformed slice shape"
            f" {slice_shape} is divisible by the collective size"
            f" {collective_size}"
        )
      # Make each block load a smaller slice, adjust the GMEM indices and slice
      # the SMEM reference accordingly.
      multicast_mask = arith.trunci(
          i16, utils.cluster_collective_mask(self.cluster_size, collective)
      )
    else:
      multicast_mask = None

    tma_desc = self._get_tma_desc(
        gmem_ref, gmem_transform, tuple(slice_shape), swizzle,
    )

    # We constuct TMA descriptors in column-major order.
    rev_dyn_base_indices = [
        arith.index_cast(i32, idx) for idx in reversed(dyn_base_indices)
    ]

    uniform_ctx = (
        functools.partial(utils.single_thread, per_block=False)
        if uniform and predicate is None
        else contextlib.nullcontext
    )

    if max(slice_shape) > 256:
      raise ValueError(
          "Async copies only support copying <=256 elements along each"
          " dimension"
      )
    if (zeroth_bw := slice_shape[-1] * element_bytewidth) % 16 != 0:
      raise ValueError(
          "Async copies require the number of bytes copied along the last"
          f" dimension to be divisible by 16, but got {zeroth_bw}"
      )
    if swizzle is not None and slice_shape[-1] != swizzle // element_bytewidth:
      raise ValueError(
          f"Async copies with {swizzle=} require last dimension of the slice to"
          f" be exactly {swizzle} bytes"
          f" ({swizzle // element_bytewidth} elements), but got"
          f" {slice_shape[-1]}"
      )
    smem_ptr = utils.memref_ptr(smem_ref, memory_space=3)
    if gmem_ref is src_ref:
      assert barrier is not None  # for pytype
      transfer_bytes = c(
          np.prod(slice_shape) * element_bytewidth * collective_size, i32
      )
      barrier_ptr = barrier.get_ptr()
      with uniform_ctx():
        if arrive:
          nvvm.mbarrier_arrive_expect_tx_shared(
              barrier_ptr, transfer_bytes, predicate=predicate
          )
        nvvm.cp_async_bulk_tensor_shared_cluster_global(
            smem_ptr, tma_desc, rev_dyn_base_indices, barrier_ptr, [],
            multicast_mask=multicast_mask, predicate=predicate
        )
    else:
      with uniform_ctx():
        nvvm.cp_async_bulk_tensor_global_shared_cta(
            tma_desc, smem_ptr, rev_dyn_base_indices, predicate=predicate
        )
        nvvm.cp_async_bulk_commit_group()

  def await_async_copy(
      self, allow_groups: int, await_read_only: bool = False
  ):
    nvvm.cp_async_bulk_wait_group(allow_groups, read=await_read_only)
    utils.warpgroup_barrier()


# ShapeTrees currently can not contain unions.
ShapeTree = Any
RefTree = Any
T = TypeVar('T')


@dataclasses.dataclass(frozen=True)
class Union(Generic[T]):
  members: Sequence[T]

  def __iter__(self):
    return iter(self.members)

@dataclasses.dataclass(frozen=True)
class TMABarrier:
  num_barriers: int = 1

@dataclasses.dataclass(frozen=True)
class Barrier:
  arrival_count: int
  num_barriers: int = 1

@dataclasses.dataclass(frozen=True)
class ClusterBarrier:
  collective_dims: Sequence[gpu.Dimension]
  num_barriers: int = 1


def _count_buffer_bytes(shape_dtype: jax.ShapeDtypeStruct) -> int:
  return np.prod(shape_dtype.shape) * np.dtype(shape_dtype.dtype).itemsize


def _construct_smem_reftree(
    cluster_shape: tuple[int, int, int],
    dynamic_smem: ir.Value,
    smem_buffers: ShapeTree,
    dynamic_smem_offset: int = 0,
) -> RefTree:
  index = ir.IndexType.get()
  i8 = ir.IntegerType.get_signless(8)
  ptr = ir.Type.parse("!llvm.ptr")
  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
  flat_ref_tys, smem_buffer_tree = jax.tree.flatten(
      smem_buffers, is_leaf=lambda x: isinstance(x, Union)
  )
  smem_refs = []
  for ref_ty in flat_ref_tys:
    def get_barrier_ptr(num_barriers: int) -> ir.Value:
      nonlocal dynamic_smem_offset
      smem_base_ptr = utils.memref_ptr(dynamic_smem, memory_space=3)
      barrier_base_ptr = llvm.getelementptr(
          ptr, smem_base_ptr, [], [dynamic_smem_offset], i8
      )
      dynamic_smem_offset += num_barriers * MBARRIER_BYTES
      return barrier_base_ptr
    match ref_ty:
      case Union(members):
        member_trees = [
            _construct_smem_reftree(cluster_shape, dynamic_smem, m, dynamic_smem_offset)
            for m in members
        ]
        # TODO(apaszke): This is quadratic, but it shouldn't matter for now...
        dynamic_smem_offset += _smem_tree_size(ref_ty)
        ref = Union(member_trees)
      case TMABarrier(num_barriers):
        ref = utils.BarrierRef.initialize(
            get_barrier_ptr(num_barriers), num_barriers, arrival_count=1
        )
      case Barrier(arrival_count, num_barriers):
        ref = utils.BarrierRef.initialize(
            get_barrier_ptr(num_barriers),
            num_barriers,
            arrival_count=arrival_count,
        )
      case ClusterBarrier(collective_dims, num_barriers):
        ref = utils.CollectiveBarrierRef.initialize(
            get_barrier_ptr(num_barriers),
            num_barriers,
            collective_dims,
            cluster_shape,
        )
      case _:
        mlir_dtype = utils.dtype_to_ir_type(ref_ty.dtype)
        tile_smem = memref.view(
            ir.MemRefType.get(ref_ty.shape, mlir_dtype, memory_space=smem),
            dynamic_smem, c(dynamic_smem_offset, index), [],
        )
        dynamic_smem_offset += _count_buffer_bytes(ref_ty)
        ref = tile_smem
    smem_refs.append(ref)
  return jax.tree.unflatten(smem_buffer_tree, smem_refs)


MBARRIER_BYTES = 8


def _smem_tree_size(smem_buffers: ShapeTree) -> int:
  leaves = jax.tree.leaves(
      smem_buffers, is_leaf=lambda x: isinstance(x, Union)
  )
  size = 0
  for l in leaves:
    match l:
      case Union(members):
        size += max(_smem_tree_size(s) for s in members)
      case (
          TMABarrier(num_barriers)
          | ClusterBarrier(_, num_barriers=num_barriers)
          | Barrier(_, num_barriers=num_barriers)
      ):
        if size % MBARRIER_BYTES:
          raise NotImplementedError("Misaligned barrier allocation")
        size += num_barriers * MBARRIER_BYTES
      case _:
        size += _count_buffer_bytes(l)
  return size


# TODO(apaszke): Inline this
@contextlib.contextmanager
def _launch(
    token,
    grid: tuple[int, int, int],
    cluster: tuple[int, int, int],
    block: tuple[int, int, int],
    scratch_arr,
    smem_buffers: ShapeTree | Union[ShapeTree],
    profiler_spec: profiler.ProfilerSpec | None = None,
    maybe_prof_buffer: ir.Value | None = None,
):
  if (profiler_spec is None) != (maybe_prof_buffer is None):
    raise ValueError
  index = ir.IndexType.get()
  i32 = ir.IntegerType.get_signless(32)
  i8 = ir.IntegerType.get_signless(8)
  grid_vals = [c(i, index) for i in grid]
  block_vals = [c(i, index) for i in block]

  user_smem_bytes = _smem_tree_size(smem_buffers)

  smem_bytes = user_smem_bytes
  if profiler_spec is not None:
    smem_bytes += profiler_spec.smem_bytes(block=block)

  # TODO(cperivol): Query the shared memory size programmatically.
  if smem_bytes > 228 * 1024:
    raise ValueError(f"Mosaic GPU kernel exceeds available shared memory {smem_bytes=} > 228000")
  if math.prod(cluster) != 1:
    if len(cluster) != 3:
      raise ValueError("Clusters must be 3D")
    cluster_kwargs = {
        "clusterSize" + d: c(s, index) for s, d in zip(cluster, "XYZ")
    }
    for d, grid_size, cluster_size in zip("xyz", grid, cluster):
      if grid_size % cluster_size != 0:
        raise ValueError(
            f"Grid dimension {d} must be divisible by cluster dimension:"
            f" {grid_size} % {cluster_size} != 0"
        )
  else:
    cluster_kwargs = {}
  launch_op = gpu.LaunchOp(
      token.type, [token], *grid_vals, *block_vals,
      dynamicSharedMemorySize=c(smem_bytes, i32), **cluster_kwargs)
  launch_op.body.blocks.append(*([index] * (12 + 2 * len(cluster_kwargs))))  # Append an empty block
  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
  with ir.InsertionPoint(launch_op.body.blocks[0]):
    dynamic_smem = gpu.dynamic_shared_memory(
        ir.MemRefType.get(
            (ir.ShapedType.get_dynamic_size(),), i8, memory_space=smem
        )
    )

    if profiler_spec:
      prof_smem = memref.view(
          ir.MemRefType.get(
              (profiler_spec.smem_i32_elements(block=block),),
              i32, memory_space=smem,
          ),
          dynamic_smem, c(user_smem_bytes, index), [],
      )
      prof = profiler.OnDeviceProfiler(
          profiler_spec, prof_smem, maybe_prof_buffer
      )
    else:
      prof = None

    ptr_ty = ir.Type.parse("!llvm.ptr")
    scratch_ptr = builtin.unrealized_conversion_cast([ptr_ty], [scratch_arr])
    ctx = LaunchContext(launch_op, scratch_ptr, cluster, prof)
    with ctx.named_region("Init"):
      smem_ref_tree = _construct_smem_reftree(
          cluster, dynamic_smem, smem_buffers
      )
      # TODO(apaszke): Skip the following if no barriers were initialized.
      nvvm.fence_mbarrier_init()
      if math.prod(cluster) != 1:
        nvvm.cluster_arrive_relaxed(aligned=ir.UnitAttr.get())
        nvvm.cluster_wait(aligned=ir.UnitAttr.get())
      gpu.barrier()

    yield ctx, smem_ref_tree
    if prof is not None:
      prof.finalize(grid=grid, block=block)
    gpu.terminator()


def _lower_as_gpu_kernel(
    body,
    grid: tuple[int, int, int],
    cluster: tuple[int, int, int],
    block: tuple[int, int, int],
    in_shapes: tuple[Any, ...],
    out_shape,
    smem_scratch_shape: ShapeTree | Union[ShapeTree],
    module_name: str,
    kernel_name: str | None = None,
    prof_spec: profiler.ProfilerSpec | None = None,
):
  ptr_ty = ir.Type.parse("!llvm.ptr")
  token_ty = ir.Type.parse("!gpu.async.token")
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)

  def _shape_to_ref_ty(shape: jax.ShapeDtypeStruct) -> ir.MemRefType:
    return ir.MemRefType.get(shape.shape, utils.dtype_to_ir_type(shape.dtype))

  in_ref_tys = [_shape_to_ref_ty(t) for t in in_shapes]

  unwrap_output_tuple = False
  if isinstance(out_shape, list):
    out_shape = tuple(out_shape)
  elif not isinstance(out_shape, tuple):
    out_shape = (out_shape,)
    unwrap_output_tuple = True
  out_ref_tys = [_shape_to_ref_ty(t) for t in out_shape]
  if prof_spec is not None:
    out_shape = (*out_shape, prof_spec.jax_buffer_type(grid, block))
    out_ref_tys.append(prof_spec.mlir_buffer_type(grid, block))

  module = ir.Module.create()
  attrs = module.operation.attributes
  attrs["sym_name"] = ir.StringAttr.get(module_name)
  if kernel_name is None:
    kernel_name = getattr(body, "__name__", "anonymous")
  with ir.InsertionPoint(module.body):
    _declare_runtime_functions()
    gmem_scratch_bytes = 0
    global_scratch = llvm.GlobalOp(
        ir.Type.parse("!llvm.array<0 x i8>"),  # We don't know the shape yet.
        "global_scratch",
        ir.Attribute.parse("#llvm.linkage<external>"),
        addr_space=ir.IntegerAttr.get(i32, 4),  # GPU constant memory.
    )
    @func.FuncOp.from_py_func(ptr_ty, ptr_ty, name=f"mosaic_gpu_{kernel_name}")
    def main(token_ptr, buffers):
      nonlocal gmem_scratch_bytes
      token = builtin.unrealized_conversion_cast([token_ty], [token_ptr])
      arg_refs = []
      for i, ref_ty in enumerate([*in_ref_tys, *out_ref_tys]):
        ptr = llvm.LoadOp(ptr_ty, llvm.GEPOp(ptr_ty, buffers, [], [i], ptr_ty))
        arg_refs.append(utils.ptr_as_memref(ptr, ir.MemRefType(ref_ty)))
      in_refs = arg_refs[:len(in_ref_tys)]
      out_refs = arg_refs[len(in_ref_tys):]
      prof_buffer = out_refs.pop() if prof_spec is not None else None
      empty_arr_ty = ir.Type.parse("!llvm.array<0 x i8>")
      scratch_alloc = llvm.AllocaOp(
          ptr_ty, c(1, i64), empty_arr_ty, alignment=TMA_DESCRIPTOR_ALIGNMENT
      )
      scratch_arr = llvm.load(empty_arr_ty, scratch_alloc.result)
      with _launch(
          token, grid, cluster, block, scratch_arr, smem_scratch_shape,
          prof_spec, prof_buffer
      ) as (launch_ctx, smem_refs):
        body(launch_ctx, *in_refs, *out_refs, smem_refs)
        gmem_scratch_bytes = launch_ctx.next_scratch_offset
      # Allocate and initialize the host buffer right before the launch.
      # Note that we couldn't do that before, because we had to run the body
      # to learn what the scratch contains.
      with ir.InsertionPoint(scratch_arr.owner):
        scratch_arr_ty = ir.Type.parse(f"!llvm.array<{gmem_scratch_bytes} x i8>")
        scratch_alloc.elem_type = ir.TypeAttr.get(scratch_arr_ty)
        scratch_arr.set_type(scratch_arr_ty)
        for init_callback in launch_ctx.host_scratch_init:
          init_callback(scratch_alloc.result)
    main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
  sym_tab = ir.SymbolTable(module.operation)
  sym_tab.insert(main.func_op)
  sym_tab.insert(global_scratch)
  module.operation.verify()

  return module, out_shape, unwrap_output_tuple


def _declare_runtime_functions():
  """Declares the runtime functions that can be used by the generated code."""
  ptr_ty = ir.Type.parse("!llvm.ptr")
  i64 = ir.IntegerType.get_signless(64)
  arg_tys = [ptr_ty, ptr_ty, i64, i64, ptr_ty, ptr_ty, i64, ptr_ty]
  init_tma_desc_type = ir.FunctionType.get(arg_tys, [])
  func.FuncOp(
      "mosaic_gpu_init_tma_desc", init_tma_desc_type, visibility="private"
  )
  memcpy_async_type = ir.FunctionType.get([ptr_ty, ptr_ty, i64, ptr_ty], [])
  func.FuncOp(
      "mosaic_gpu_memcpy_async_h2d", memcpy_async_type, visibility="private"
  )


class ThreadSemantics(enum.Enum):
  """Semantics for the kernel's instruction stream."""

  Lane = enum.auto()
  Warpgroup = enum.auto()


def as_gpu_kernel(
    body,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    in_shape,
    out_shape,
    smem_scratch_shape: ShapeTree | Union[ShapeTree],
    prof_spec: profiler.ProfilerSpec | None = None,
    cluster: tuple[int, int, int] = (1, 1, 1),
    module_name: str = "unknown",
    kernel_name: str | None = None,
    thread_semantics: ThreadSemantics = ThreadSemantics.Lane,
):
  if isinstance(in_shape, list):
    in_shape = tuple(in_shape)
  elif not isinstance(in_shape, tuple):
    in_shape = (in_shape,)

  module, out_shape, unwrap_output_tuple = (
      _lower_as_gpu_kernel(
          body, grid, cluster, block, in_shape, out_shape, smem_scratch_shape,
          module_name, kernel_name, prof_spec
      )
  )

  if thread_semantics == ThreadSemantics.Warpgroup and dialect is not None:
    # Run Python lowering passes. The remaining passes will be run in C++ in
    # jax/jaxlib/mosaic/gpu/custom_call.cc
    layout_inference.infer_layout(module)  # pytype: disable=attribute-error
    dialect_lowering.lower_mgpu_dialect(module)  # pytype: disable=attribute-error

  expected_arg_treedef = jax.tree.structure(in_shape)
  def _check_args(*args):
    arg_treedef = jax.tree.structure(args)
    if arg_treedef != expected_arg_treedef:
      raise ValueError(
          f"Invalid argument structure: expected {expected_arg_treedef}, got"
          f" {arg_treedef}, ({args=})"
      )

  module_asm = module.operation.get_asm(binary=True, enable_debug_info=True)
  def bind(*args):
    return mosaic_gpu_p.bind(
        *args,
        out_types=out_shape,
        module=module_asm,
    )

  if prof_spec is not None:
    @jax.jit
    def prof_kernel(*args):
      _check_args(*args)
      *results, prof_buffer = bind(*args)
      def dump_profile(prof_buffer):
        out_file = os.path.join(
            os.getenv("TEST_UNDECLARED_OUTPUTS_DIR"),
            f"{time.time_ns()}-trace.json",
        )
        try:
          with open(out_file, "x") as f:
            prof_spec.dump(prof_buffer, f, grid=grid, block=block)
        except FileExistsError:
          pass  # TODO: Retry
      jax.debug.callback(dump_profile, prof_buffer)
      return results[0] if unwrap_output_tuple else results
    return prof_kernel
  else:
    @jax.jit
    def kernel(*args):
      _check_args(*args)
      results = bind(*args)
      return results[0] if unwrap_output_tuple else results
    return kernel


def as_torch_gpu_kernel(
    body,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    in_shape,
    out_shape,
    smem_scratch_shape: ShapeTree | Union[ShapeTree],
    prof_spec: profiler.ProfilerSpec | None = None,
    cluster: tuple[int, int, int] = (1, 1, 1),
    module_name: str = "unknown",
    kernel_name: str | None = None,
):
  try:
    import torch
  except ImportError:
    raise RuntimeError("as_torch_gpu_kernel requires PyTorch")
  torch.cuda.init()  # Make sure CUDA context is set up.

  if isinstance(in_shape, list):
    in_shape = tuple(in_shape)
  elif not isinstance(in_shape, tuple):
    in_shape = (in_shape,)

  flat_out_types, out_treedef = jax.tree.flatten(out_shape)
  expected_arg_treedef = jax.tree.structure(in_shape)

  module, out_shape, unwrap_output_tuple = (
      _lower_as_gpu_kernel(
          body, grid, cluster, block, in_shape, out_shape, smem_scratch_shape,
          module_name, kernel_name, prof_spec
      )
  )

  # Get our hands on the compilation and unload functions
  try:
    import jax_plugins.xla_cuda12 as cuda_plugin
  except ImportError:
    raise RuntimeError("as_torch_gpu_kernel only works with recent jaxlib builds "
                       "that use backend plugins")
  dll = ctypes.CDLL(cuda_plugin._get_library_path())
  compile_func = dll.MosaicGpuCompile
  compile_func.argtypes = [ctypes.c_void_p]
  compile_func.restype = ctypes.POINTER(ctypes.c_void_p)
  unload_func = dll.MosaicGpuUnload
  unload_func.argtypes = [compile_func.restype]
  unload_func.restype = None

  module_asm = module.operation.get_asm(binary=True, enable_debug_info=True)
  compiled = compile_func(ctypes.c_char_p(module_asm))
  if compiled is None:
    raise RuntimeError("Failed to compile the module")
  ctx, launch_ptr = compiled[0], compiled[1]
  ctx_ptr_ptr = ctypes.pointer(ctypes.c_void_p(ctx))
  launch = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(launch_ptr)

  def as_torch_dtype(dtype):
    # torch contains NumPy-compatible dtypes in its top namespace
    return getattr(torch, np.dtype(dtype).name)

  def apply(*args):
    flat_args, arg_treedef = jax.tree.flatten(args)
    if arg_treedef != expected_arg_treedef:
      raise ValueError(
          f"Invalid argument structure: expected {expected_arg_treedef}, got"
          f" {arg_treedef}, ({args=})"
      )

    # Construct a device pointer list like in the XLA calling convention
    buffers = (ctypes.c_void_p * (arg_treedef.num_leaves + out_treedef.num_leaves))()
    i = -1  # Define i in case there are no args
    device = 'cuda'
    for i, arg in enumerate(flat_args):
      buffers[i] = arg.data_ptr()
      device = arg.device
    flat_outs = []
    for i, t in enumerate(flat_out_types, i + 1):
      out = torch.empty(t.shape, dtype=as_torch_dtype(t.dtype), device=device)
      flat_outs.append(out)
      buffers[i] = out.data_ptr()
    # Allocate another buffer for args of the host-side program. This is sadly
    # the default MLIR calling convention.
    args_ptr = (ctypes.POINTER(ctypes.c_void_p) * 3)()
    args_ptr[0] = ctx_ptr_ptr
    args_ptr[1] = ctypes.pointer(torch.cuda.default_stream(device)._as_parameter_)
    args_ptr[2] = ctypes.cast(ctypes.pointer(ctypes.pointer(buffers)),
                              ctypes.POINTER(ctypes.c_void_p))
    launch(args_ptr)
    return jax.tree.unflatten(out_treedef, flat_outs)

  # Unload the compiled code when the Python function is destroyed.
  def unload(_):
    unload_func(compiled)
  apply.destructor = weakref.ref(apply, unload)

  return apply
