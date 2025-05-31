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
import dataclasses
import enum
import functools
import math
from typing import Any, Literal

from jax._src.lib import mosaic_gpu_dialect as mgpu_dialect
from jax._src import lib as jaxlib
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import builtin
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
import numpy as np

from . import profiler
from . import utils
# mypy: ignore-errors

TMA_DESCRIPTOR_BYTES = 128
TMA_DESCRIPTOR_ALIGNMENT = 64

c = utils.c  # This is too common to fully qualify.

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


class Rounding(enum.Enum):
  UP = enum.auto()
  DOWN = enum.auto()


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
  rounding: Rounding | None = None

  def apply(self, ref: ir.Value) -> ir.Value:
    untiled_rank = ir.MemRefType(ref.type).rank
    tiling_rank = len(self.tiling)
    tiled_rank = untiled_rank + tiling_rank
    for t, d in zip(self.tiling[::-1], range(untiled_rank)[::-1]):
      ref_shape = ir.MemRefType(ref.type).shape
      s = ref_shape[d]
      if s > t:
        if s % t:
          match self.rounding:
            case None:
              raise ValueError(
                  f"When no rounding mode is specified, dimension {d} must have"
                  f" size smaller or a multiple of its tiling {t}, but got {s}"
              )
            case Rounding.UP:
              raise NotImplementedError
            case Rounding.DOWN:
              slices = [slice(None)] * d
              slices.append(slice(0, s // t * t))
              ref = utils.memref_slice(ref, tuple(slices))
            case _:
              raise ValueError(f"Unknown rounding mode: {self.rounding}")
      else:
        t = s
      ref = utils.memref_unfold(ref, d, (None, t))
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
    if self.rounding is None:
      for size, tile_size in zip(shape[-tiling_rank:], self.tiling):
        if size % tile_size:
          raise ValueError(
              f"Expected GMEM slice shape {shape} suffix to be a multiple of"
              f" tiling {self.tiling}.\nIf you're using padded async copies,"
              " your slice might need to extend out of bounds of the GMEM"
              " buffer (OOB accesses will be skipped)."
          )
    elif self.rounding != Rounding.DOWN:
      raise NotImplementedError(self.rounding)
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
      raise ValueError("All elements of `permutation` must be unique")

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

ReductionOp = Literal["add", "min", "max", "inc", "dec", "and", "or", "xor"]

class Scratch:
  """Manages ops handling the GMEM scratch that contains the TMA descriptors.

  TMA descriptors are created on the host and then copied to GMEM. So there
  needs to be some code on the host to allocate and initialize the TMA
  descriptors. However, we only know what descriptors we need after we have
  lowered the entire kernel. This class helps manage everything needed to
  correctly allocate and initialize the scratch.

  To help reconcile the needs of kernels that use the dialect lowering with
  those that use MGPU APIs directly, this class only creates the relevant ops
  lazily. Eager creation would make them appear dead before dialect lowering
  and MLIR's DCE would remove them.

  During the lowering, we collect information about how many bytes are needed
  and also how each descriptor should be initialized on the host. At the end
  of the lowering, the finalize_size() method should be called to add the
  necessary code on the host to allocate and initialize all descriptors.
  """
  def __init__(self, gpu_launch_op: gpu.LaunchOp):
    self.next_offset: int = 0
    self.host_init: list[Callable[[ir.Value], None]] = []
    self._alloc_op = None
    self._load_op = None
    self._scratch_ptr = None

    # Ideally, we would store the gpu.launch op directly. However, it gets
    # invalidated by passes like "canonicalize". Thus we store the module and
    # find the gpu.launch op from there when needed.
    op = gpu_launch_op
    while op.name != "builtin.module":
      op = op.parent.opview
    assert op is not None
    self._module_op = op

  def _find_gpu_launch_op(self, block: ir.Block) -> ir.OpView | None:
    for op in block:
      if op.name == "gpu.launch":
        return op
      for region in op.regions:
        for block in region:
          child_op = self._find_gpu_launch_op(block)
          if child_op is not None:
            return child_op
    return None

  def _create_ops_if_none(self):
    if self._alloc_op is not None:
      return

    gpu_launch_op = self._find_gpu_launch_op(self._module_op.body)
    assert gpu_launch_op is not None
    ptr_ty = ir.Type.parse("!llvm.ptr")
    with ir.InsertionPoint(gpu_launch_op):
      empty_arr_ty = ir.Type.parse("!llvm.array<0 x i8>")
      i64 = ir.IntegerType.get_signless(64)
      self._alloc_op = llvm.AllocaOp(
          ptr_ty, c(1, i64), empty_arr_ty,
          alignment=TMA_DESCRIPTOR_ALIGNMENT
      )
      self._load_op = llvm.LoadOp(empty_arr_ty, self._alloc_op)

    with ir.InsertionPoint.at_block_begin(gpu_launch_op.body.blocks[0]):
      self._scratch_ptr = builtin.unrealized_conversion_cast(
          [ptr_ty], [self._load_op]
      )

  def device_ptr(self) -> ir.Value:
    self._create_ops_if_none()
    return self._scratch_ptr

  def finalize_size(self):
    """
    Allocates and initializes the host buffer. This needs to be done after
    lowering, i.e. after all TMA descriptors have been recorded. Only then we
    know what the scratch contains.
    """
    if self.next_offset == 0:
      return
    assert self._alloc_op is not None
    with ir.InsertionPoint(self._load_op):
      gmem_scratch_bytes = self.next_offset
      scratch_arr_ty = ir.Type.parse(f"!llvm.array<{gmem_scratch_bytes} x i8>")
      self._alloc_op.elem_type = ir.TypeAttr.get(scratch_arr_ty)
      self._load_op.result.set_type(scratch_arr_ty)
      for init_callback in self.host_init:
        init_callback(self._alloc_op.result)


class _DefaultPredicate:
  pass


@dataclasses.dataclass()
class LaunchContext:
  module: ir.Module
  scratch: Scratch
  cluster_size: tuple[int, int, int]
  profiler: OnDeviceProfiler | None = None
  tma_descriptors: dict[
      tuple[ir.Value, tuple[int, ...], int | None, tuple[MemRefTransform, ...]],
      ir.Value,
  ] = dataclasses.field(default_factory=dict, init=False)
  is_device_collective: bool = False

  @contextlib.contextmanager
  def named_region(self, *args, **kwargs):
    if self.profiler is not None:
      with self.profiler.record(*args, **kwargs):
        yield
    else:
      yield

  def cluster_idx(
      self, dim: gpu.Dimension | Sequence[gpu.Dimension] | None = None
  ) -> ir.Value:
    """Returns the index of a block within a subset of the cluster spanned by the given dimensions."""
    if dim is None:
      dim = gpu.Dimension
    elif isinstance(dim, gpu.Dimension):
      dim = (dim,)
    index = ir.IndexType.get()
    stride = 1
    idx = c(0, index)
    for d in sorted(dim):
      if self.cluster_size[d] == 1:  # Optimize a multiply by 0.
        continue
      idx = arith.addi(idx, arith.muli(gpu.cluster_block_id(d), c(stride, index)))
      stride *= self.cluster_size[d]
    return idx

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
    if self.scratch.next_offset % alignment:
      raise NotImplementedError  # TODO(apaszke): Pad to match alignment
    alloc_base = self.scratch.next_offset
    self.scratch.next_offset += size
    def host_init_wrapped(host_ptr):
      host_init(
          llvm.getelementptr(ptr_ty, host_ptr, [], [alloc_base], i8, llvm.GEPNoWrapFlags.none)
      )
    self.scratch.host_init.append(host_init_wrapped)
    # with ir.InsertionPoint(self.gmem_scratch_ptr.owner):
    # There is no way to create an insertion point after an operation...
    gep = llvm.GEPOp(
        ptr_ty, self.scratch.device_ptr(), [], [alloc_base], i8, llvm.GEPNoWrapFlags.none
    )
    gep.move_after(self.scratch.device_ptr().owner)
    return device_init(gep.result)

  def _get_tma_desc(
      self,
      gmem_ref,
      gmem_transform: tuple[MemRefTransform, ...],
      gmem_peer_id: int | ir.Value | None,
      transformed_slice_shape: tuple[int, ...],
      swizzle: int | None,
      reduction_op: Literal[
        "add","min","max","inc","dec","and","or","xor"
      ] | None,
  ):
    # Using ir.Values in cache keys is a little sketchy, but I think it should
    # be fine. Having it in the key will keep it alive, and if comparison and
    # hashing is by identity then it should work out.
    tma_desc_key = (gmem_ref, transformed_slice_shape, swizzle, gmem_transform, gmem_peer_id)
    if (tma_desc := self.tma_descriptors.get(tma_desc_key, None)) is None:
      i32 = ir.IntegerType.get_signless(32)
      i64 = ir.IntegerType.get_signless(64)
      ptr_ty = ir.Type.parse("!llvm.ptr")
      def init_tma_desc(host_ptr):
        ref = gmem_ref
        for t in gmem_transform:
          ref = t.apply(ref)
        ref_ty = ir.MemRefType(ref.type)
        # TODO(apaszke): Use utils.memref_ptr to compute base_ptr
        strides, _ = ref_ty.get_strides_and_offset()
        if strides[-1] != 1:
          raise ValueError(
              "TMA requires the stride of the last dimension after"
              " transforming the GMEM reference to be 1, but it is"
              f" {strides[-1]}."
          )

        _, offset, *sizes_and_strides = memref.extract_strided_metadata(ref)
        aligned_ptr_idx = memref.extract_aligned_pointer_as_index(ref)
        as_i64 = lambda i: arith.index_cast(i64, i)
        alloc_ptr = llvm.inttoptr(ptr_ty, as_i64(aligned_ptr_idx))
        llvm_dyn = -2147483648  # TODO(apaszke): Improve the MLIR bindings...
        base_ptr = llvm.getelementptr(
            ptr_ty, alloc_ptr, [as_i64(offset)], [llvm_dyn], ref_ty.element_type, llvm.GEPNoWrapFlags.none,
        )
        if gmem_peer_id is not None:
          if not isinstance(gmem_peer_id, ir.Value):
            peer_id = c(gmem_peer_id, i32)
          else:
            try:
              # We try to reproduce the gmem_peer_id computation on the host.
              peer_id = _recompute_peer_id(gmem_peer_id)
            except ReplicationError as e:
              raise ValueError(
                  "Failed to recompute the async_copy peer id on the host"
              ) from e
          self._ensure_nvshmem_decls()
          base_ptr = llvm.call(
              base_ptr.type,
              [base_ptr, peer_id],
              [],
              [],
              callee="nvshmem_ptr",
          )
        rank = ref_ty.rank
        assert rank * 2 == len(sizes_and_strides)
        swizzle_arg = (
            mgpu_dialect.SwizzlingMode.kNoSwizzle
            if swizzle is None
            else swizzle
        )
        # TODO(apaszke): Better verification (e.g. slice is non-zero)
        # TODO(apaszke): We always know strides statically.
        if jaxlib.version < (0, 5, 4):
          dtype_or_bitwidth = c(utils.bitwidth(ref_ty.element_type), i64)
        else:
          if isinstance(ref_ty.element_type, ir.IntegerType):
            if reduction_op is not None:
              raise ValueError(
                  f"TMA with reduction_op={reduction_op} is not supported with Integers"
              )
            bitwidth = utils.bitwidth_impl(ref_ty.element_type)
            if bitwidth == 4:
              tma_dtype = 0
            elif bitwidth == 8:
              tma_dtype = 1
            elif bitwidth == 16:
              tma_dtype = 2
            elif bitwidth == 32:
              tma_dtype = 3
            elif bitwidth == 64:
              tma_dtype = 4
            else:
              raise ValueError(f"Unsupported integer bitwidth: {bitwidth}")
          elif ir.F16Type.isinstance(ref_ty.element_type):
            tma_dtype = 5
          elif ir.F32Type.isinstance(ref_ty.element_type):
            tma_dtype = 6
          elif ir.BF16Type.isinstance(ref_ty.element_type):
            tma_dtype = 7
          # We treat 8 bit floats as 8 bit integers
          elif ir.Float8E5M2Type.isinstance(ref_ty.element_type):
            tma_dtype = 1
          elif ir.Float8E4M3FNType.isinstance(ref_ty.element_type):
            tma_dtype = 1
          else:
            raise ValueError(f"unsupported TMA dtype {ref_ty.element_type}")
          dtype_or_bitwidth = c(tma_dtype, i64)
        args = [
            host_ptr,
            base_ptr,
            dtype_or_bitwidth,
            c(rank, i64),
            utils.pack_array([as_i64(i) for i in sizes_and_strides[:rank]]),
            utils.pack_array([as_i64(i) for i in sizes_and_strides[rank:]]),
            c(swizzle_arg, i64),
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
      gmem_peer_id: int | ir.Value | None = None,
      barrier: utils.BarrierRef | None = None,
      swizzle: int | None = None,
      arrive: bool | None = None,
      collective: Sequence[gpu.Dimension] | gpu.Dimension | None = None,
      partitioned: int | None = None,
      # Should select 0 or 1 threads from the WG.
      predicate: ir.Value | None | _DefaultPredicate = _DefaultPredicate(),
      reduction_op: ReductionOp | None = None,
  ):
    """Initiates an async copy between GMEM and SMEM.

    Exactly one of `src_ref` and `dst_ref` must be in GMEM and in SMEM, and the
    SMEM reference must be contiguous. The GMEM window that is read or written
    to is specified by the `gmem_slice`. The copy can change the order in which
    the data appears in the window by applying a sequence of transforms to the
    GMEM reference (as specified by `gmem_transform`).

    When `collective` is specified (only allowed for GMEM -> SMEM copies), the
    identical async_copy must be scheduled by all blocks that share the same
    coordinates along collective dimensions within a cluster. The behavior is
    undefined otherwise. The semantics of collective loads depend further on the
    `partitioned` argument:

    - If `partitioned` is not specified, all blocks load the same data into
      their shared memory and all receive the update in their barriers, unless
      `arrive` is False. If `arrive` is False, you should expect the barrier to
      have expect_tx incremented by the same amount of bytes as if `collective`
      was not specified.
    - If `partitioned` is specified, each block only loads a separate slice of
      the data into SMEM, partitioned into equal tiles along the `partitioned`
      dimension. In this case only the barrier of the first block in the
      collective will have its expect_tx incremented by the total size of the
      transfer across all blocks involved in the collective. Barriers supplied
      by other blocks will be ignored (even if `arrive` is True).
    """
    index = ir.IndexType.get()
    i16 = ir.IntegerType.get_signless(16)
    i32 = ir.IntegerType.get_signless(32)
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
    src_ref_ty = ir.MemRefType(src_ref.type)
    dst_ref_ty = ir.MemRefType(dst_ref.type)
    element_type = src_ref_ty.element_type
    element_bitwidth = utils.bitwidth(element_type)
    if element_type != dst_ref_ty.element_type:
      raise ValueError(
          f"Expected same element type, got {element_type} and"
          f" {dst_ref_ty.element_type}"
      )
    if isinstance(predicate, _DefaultPredicate):
      predicate = utils.single_thread_predicate(utils.ThreadSubset.WARPGROUP)
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
      if arrive is None:
        arrive = True  # Commit this copy to the async group by default
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
    if any(s * element_bitwidth % 128 != 0 for s in gmem_strides[:-1]):
      raise ValueError(
          "async_copy requires all GMEM strides except the last one to be a"
          " multiple of 16 bytes"
      )

    if reduction_op is not None:
      if not any(
          t.isinstance(gmem_ref_ty.element_type)
          for t in (ir.F32Type, ir.BF16Type, ir.F16Type)
      ):
        raise ValueError(
            "TMA with reduction is only supported with f32, f16 and bf16"
        )
      if reduction_op != "add":
        raise ValueError(
            "TMA with reduction is only supported with add operation"
        )

    # NOTE: TMA supports OOB indices, so we skip the check.
    base_indices, slice_shape, is_squeezed = utils.parse_indices(
        gmem_slice, ir.MemRefType(gmem_ref.type).shape, check_oob=False
    )
    dyn_base_indices = tuple(
        c(i, index) if not isinstance(i, ir.Value) else i for i in base_indices
    )
    del base_indices  # Use the dynamic indices from now on!

    collective_size = 1
    if collective is not None:
      if isinstance(collective, gpu.Dimension):
        collective = (collective,)
      collective_size = math.prod(self.cluster_size[d] for d in collective)
      if gmem_ref is dst_ref:
        raise ValueError("Only GMEM -> SMEM copies can be collective")
    if partitioned is not None:
      if collective is None:
        raise ValueError("Only collective loads can be partitioned")
      if collective_size > 1 and partitioned is not None:
        if math.prod(self.cluster_size) != 2:
          raise NotImplementedError(
              "Partitioned loads only supported for clusters of size 2"
          )
        if slice_shape[partitioned] % collective_size != 0:
          raise ValueError(
              f"The collective size ({collective_size}) must divide the slice"
              " shape along the partitioned dimension, but it has size"
              f" {slice_shape[partitioned]}"
          )
        slice_shape[partitioned] //= collective_size
        dyn_base_indices = list(dyn_base_indices)
        dyn_base_indices[partitioned] = arith.addi(
            dyn_base_indices[partitioned],
            arith.muli(
                self.cluster_idx(collective), c(slice_shape[partitioned], index)
            ),
        )
        dyn_base_indices = tuple(dyn_base_indices)

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
    if any(
        s != cs and d != 1  # Strides don't matter for dims of size 1.
        for s, cs, d in zip(
            smem_strides,
            utils.get_contiguous_strides(smem_ref_ty.shape),
            smem_ref_ty.shape,
        )
    ):
      raise ValueError(
          "async_copy needs the SMEM reference to be contiguous, but got"
          f" strides {smem_strides} for shape {smem_ref_ty.shape}"
      )

    dyn_base_indices = list(dyn_base_indices)
    slice_shape = list(slice_shape)
    assert all(d == 1 for d in slice_shape[:num_squeezed_dims])

    # Partitioned loads have already been processed (before transforms).
    if collective_size > 1 and partitioned is None:
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
      idx = self.cluster_idx(collective)
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
        gmem_ref, gmem_transform, gmem_peer_id,
        tuple(slice_shape), swizzle, reduction_op,
    )

    # We construct TMA descriptors in column-major order.
    rev_dyn_base_indices = [
        arith.index_cast(i32, idx) for idx in reversed(dyn_base_indices)
    ]

    if max(slice_shape) > 256:
      raise ValueError(
          "Async copies only support copying <=256 elements along each"
          " dimension"
      )
    if (zeroth_bw := slice_shape[-1] * element_bitwidth) % 128 != 0:
      raise ValueError(
          "Async copies require the number of bits copied along the last"
          f" dimension to be divisible by 128, but got {zeroth_bw}"
      )
    if (
        swizzle is not None
        and swizzle != mgpu_dialect.SwizzlingMode.kNoSwizzle
        and slice_shape[-1] != (swizzle * 8) // element_bitwidth
    ):
      raise ValueError(
          f"Async copies with {swizzle=} require the last dimension of the"
          f" slice to be exactly {swizzle} bytes i.e. "
          f" {(swizzle * 8) // element_bitwidth} elements, but got"
          f" {slice_shape[-1]} elements."
      )
    smem_ptr = utils.memref_ptr(smem_ref, memory_space=3)
    if gmem_ref is src_ref:
      assert barrier is not None  # for pytype
      assert np.prod(slice_shape) * element_bitwidth * collective_size % 8 == 0
      transfer_bytes = c(
          np.prod(slice_shape) * element_bitwidth * collective_size // 8, i32
      )
      barrier_ptr = barrier.get_ptr()
      assert reduction_op is None
      if collective_size > 1 and partitioned is not None:
        if predicate is None:
          predicate = c(1, ir.IntegerType.get_signless(1))
        if arrive:
          first_block = arith.cmpi(
              arith.CmpIPredicate.eq, self.cluster_idx(collective), c(0, index),
          )
          arrive_predicate = arith.andi(predicate, first_block)
          nvvm.mbarrier_arrive_expect_tx_shared(
              barrier_ptr, transfer_bytes, predicate=arrive_predicate
          )
        rank = len(slice_shape)
        idx_operands = ",".join(f"${i}" for i in range(4, 4 + rank))
        llvm.inline_asm(
            ir.Type.parse("!llvm.void"),
            [predicate, smem_ptr, tma_desc, barrier_ptr, *rev_dyn_base_indices],
            f"""
            {{
            .reg .b32 mapped_addr;
            @$0 mapa.shared::cluster.u32 mapped_addr, $3, 0;
            @$0 cp.async.bulk.tensor.{rank}d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2
                                  [$1], [$2, {{{idx_operands}}}], [mapped_addr];
            }}
            """,
            "b,r,l,r" + ",r" * rank,
            has_side_effects=True,
        )
      else:
        if arrive:
          nvvm.mbarrier_arrive_expect_tx_shared(
              barrier_ptr, transfer_bytes, predicate=predicate
          )
        nvvm.cp_async_bulk_tensor_shared_cluster_global(
            smem_ptr, tma_desc, rev_dyn_base_indices, barrier_ptr, [],
            multicast_mask=multicast_mask, predicate=predicate
        )
    else:
      assert multicast_mask is None
      if reduction_op is not None:
        if predicate is None:
          predicate = c(1, ir.IntegerType.get_signless(1))
        rank = len(slice_shape)
        idx_operands = ",".join(f"${i}" for i in range(3, 3 + rank))
        llvm.inline_asm(
          ir.Type.parse("!llvm.void"),
          [predicate,smem_ptr,tma_desc,*rev_dyn_base_indices],
          f"@$0 cp.reduce.async.bulk.tensor.{rank}d.global.shared::cta.{reduction_op}.tile.bulk_group [$2,{{{idx_operands}}}], [$1];",
          "b,r,l" + ",r" * rank,
          has_side_effects=True,
        )
        if arrive:
          nvvm.cp_async_bulk_commit_group()
      else:
        nvvm.cp_async_bulk_tensor_global_shared_cta(
            tma_desc, smem_ptr, rev_dyn_base_indices, predicate=predicate
        )
        if arrive:
          nvvm.cp_async_bulk_commit_group()

  def await_async_copy(
      self, allow_groups: int, await_read_only: bool = False
  ):
    nvvm.cp_async_bulk_wait_group(allow_groups, read=await_read_only)
    utils.warpgroup_barrier()

  def _ensure_nvshmem_decls(self):
    if self.is_device_collective:
      return
    self.is_device_collective = True
    with ir.InsertionPoint(self.module.body):
      nvshmem_my_pe_type = ir.TypeAttr.get(ir.Type.parse("!llvm.func<i32()>"))
      llvm.LLVMFuncOp(
          "nvshmem_my_pe", nvshmem_my_pe_type, sym_visibility="private"
      )
      nvshmem_ptr_type = ir.TypeAttr.get(
          ir.Type.parse("!llvm.func<!llvm.ptr(!llvm.ptr,i32)>")
      )
      llvm.LLVMFuncOp("nvshmem_ptr", nvshmem_ptr_type, sym_visibility="private")

  def to_remote(self, ref: ir.Value, peer: ir.Value):
    self._ensure_nvshmem_decls()
    if ir.MemRefType.isinstance(ref.type):
      # We replace the offset in the ref type by 0, because memref_ptr always
      # folds the offset into the pointer.
      ref_ty = ir.MemRefType(ref.type)
      strides, _ = ref_ty.get_strides_and_offset()
      result_type = ir.MemRefType.get(
          ref_ty.shape,
          ref_ty.element_type,
          ir.StridedLayoutAttr.get(0, strides),
          ref_ty.memory_space,
      )
      return utils.ptr_as_memref(
          self.to_remote(utils.memref_ptr(ref), peer), result_type
      )
    if ref.type != ir.Type.parse("!llvm.ptr"):
      raise ValueError(f"Unsupported type for to_remote: {ref.type}")
    if peer.type != ir.IntegerType.get_signless(32):
      raise ValueError(f"peer index must be an i32, got {peer.type}")
    return llvm.call(ref.type, [ref, peer], [], [], callee="nvshmem_ptr")

  def device_id(self) -> ir.Value:
    self._ensure_nvshmem_decls()
    i32 = ir.IntegerType.get_signless(32)
    return llvm.call(i32, [], [], [], callee="nvshmem_my_pe")


class ReplicationError(Exception):
  pass

def _recompute_peer_id(peer_id: ir.Value, fuel=8) -> ir.Value:
  if fuel == 0:
    raise ReplicationError(
        "gmem_peer_id computation is too complicated to recompute on the host"
    )
  if isinstance(peer_id, ir.BlockArgument):
    raise ReplicationError("Can't recompute a value that's a block argument")
  op = peer_id.owner.opview
  # We accept all arith ops
  if op.OPERATION_NAME.startswith("arith."):
    new_operands = [_recompute_peer_id(x, fuel - 1) for x in op.operands]
    result_types = [r.type for r in op.results]
    new_attributes = {na.name: na.attr for na in op.attributes}
    new_op = ir.Operation.create(
        op.OPERATION_NAME, result_types, new_operands, new_attributes
    )
    return new_op.results if len(new_op.results) > 1 else new_op.result
  # nvshmem_my_pe queries the device id of the current process and works on both
  # the host and the device.
  if isinstance(op, llvm.CallOp) and op.callee.value == "nvshmem_my_pe":
    i32 = ir.IntegerType.get_signless(32)
    return llvm.call(i32, [], [], [], callee="nvshmem_my_pe")
  raise ReplicationError(
      f"Unrecognized op can't be recomputed on the host: {op}"
  )
