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
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import _gpu_ops_gen
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import builtin
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
import numpy as np

from . import fragmented_array as fa
from . import profiler
from . import utils

TMA_DESCRIPTOR_BYTES = 128
TMA_DESCRIPTOR_ALIGNMENT = 64
TMAReductionOp = Literal[
    "add",
    "min",
    "max",
    "inc",
    "dec",
    "and",
    "or",
    "xor",
    "umin",
    "umax",
    "smin",
    "smax",
]

def _reduction_op_to_ptx(reduction_op: TMAReductionOp) -> str:
  # convert [s|u]min|max to min|max
  return reduction_op[-3:]

c = utils.c  # This is too common to fully qualify.

class GlobalBroadcast:
  pass

GLOBAL_BROADCAST = GlobalBroadcast()


@dataclasses.dataclass(frozen=True)
class MemRefTransform:
  def apply(self, ref: ir.Value) -> ir.Value:
    raise NotImplementedError("Subclasses should override this method")

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    raise NotImplementedError("Subclasses should override this method")

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    raise NotImplementedError("Subclasses should override this method")

  def transform_strides(self, shape: Sequence[int]) -> tuple[int, ...]:
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

  def transform_strides(self, strides: Sequence[int]) -> tuple[int, ...]:
    tiling_rank = len(self.tiling)
    return (
        *strides[:-tiling_rank],
        *(s * t for s, t in zip(strides[-tiling_rank:], self.tiling)),
        *strides[-tiling_rank:],
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

  def transform_strides(self, strides: Sequence[int]) -> tuple[int, ...]:
    return tuple(strides[p] for p in self.permutation)

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

MOSAIC_GPU_SMEM_ALLOC_ATTR = "mosaic_gpu_smem_alloc"

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

  Here's how the IR looks after the initial ops are created for the first time:


  %1 = llvm.alloc_op {elem_type = !llvm.array<0 x i8>} -> !llvm.ptr
  %2 = llvm.load_op (%1) : (!llvm.ptr) -> !llvm.array<0 x i8>
  ...
  %3 = gpu.launch async
    ^bb0:
      %4 = builtin.unrealized_conversion_cast_op(%2)
             : (!llvm.array<256 x i8>) -> !llvm.ptr


  And here is an example of how the IR could look like after finalize_size() is
  called:


  %11 = llvm.alloc_op {elem_type = !llvm.array<256 x i8>} -> !llvm.ptr
  %22 = llvm.load_op (%11) : (!llvm.ptr) -> !llvm.array<256 x i8>
  ...
  # Ops inserted to initialize the tma descriptors on the host:
  ...
  %33 = llvm.getelementptr %11[0] : (!llvm.ptr) -> !llvm.ptr, i8
  call @mosaic_gpu_init_tma_desc (%33, ...)
  ...
  %44 = llvm.getelementptr %11[128] : (!llvm.ptr) -> !llvm.ptr, i8
  call @mosaic_gpu_init_tma_desc (%44, ...)
  ...
  %55 = gpu.launch async
    ^bb0:
      %66 = builtin.unrealized_conversion_cast_op(%22)
             : (!llvm.array<256 x i8>) -> !llvm.ptr

  """
  def __init__(self, gpu_launch_op: _gpu_ops_gen.LaunchOp):
    self.next_offset: int = 0
    self.host_init: list[Callable[[ir.Value], None]] = []
    self._ops_created = False

    # Ideally, we would store the gpu.launch op directly. However, it gets
    # invalidated by passes like "canonicalize". Thus we store the module and
    # find the gpu.launch op from there when needed.
    op = gpu_launch_op
    while op.name != "builtin.module":
      op = op.parent.opview
    assert op is not None
    self._module_op = op

  def _find_first_op(
      self, op_name: str, block: ir.Block, tag_attribute_name: str | None = None
  ) -> ir.OpView | None:
    for op in block:
      if op.name == op_name and (
          tag_attribute_name is None or tag_attribute_name in op.attributes
      ):
        return op
      for region in op.regions:
        for block in region:
          child_op = self._find_first_op(op_name, block, tag_attribute_name)
          if child_op is not None:
            return child_op
    return None

  def _create_ops(self):
    if self._ops_created:
      return
    self._ops_created = True

    gpu_launch_op = self._find_first_op("gpu.launch", self._module_op.body)
    assert gpu_launch_op is not None

    ptr_ty = ir.Type.parse("!llvm.ptr")
    empty_arr_ty = ir.Type.parse("!llvm.array<0 x i8>")
    i64 = ir.IntegerType.get_signless(64)

    with ir.InsertionPoint(gpu_launch_op):
      alloc_op = llvm.AllocaOp(
          ptr_ty, c(1, i64), empty_arr_ty,
          alignment=TMA_DESCRIPTOR_ALIGNMENT
      )
      # Tag the alloc op with an attribute so that we can find it later.
      alloc_op.attributes[MOSAIC_GPU_SMEM_ALLOC_ATTR] = ir.UnitAttr.get()
      load_op = llvm.LoadOp(empty_arr_ty, alloc_op)

    with ir.InsertionPoint.at_block_begin(gpu_launch_op.body.blocks[0]):
      builtin.unrealized_conversion_cast([ptr_ty], [load_op])

  def _find_alloc_load_and_device_ptr(
      self,
  ) -> tuple[llvm.AllocaOp, llvm.LoadOp, ir.Value]:
    if not self._ops_created:
      self._create_ops()

    alloc_op = self._find_first_op(
        "llvm.alloca", self._module_op.body, MOSAIC_GPU_SMEM_ALLOC_ATTR
    )
    assert alloc_op is not None
    [alloc_user] = alloc_op.result.uses
    load_op = alloc_user.owner
    assert load_op.operation.name == "llvm.load"
    [load_op_user] = load_op.result.uses
    device_ptr = load_op_user.owner
    assert device_ptr.operation.name == "builtin.unrealized_conversion_cast"
    return alloc_op, load_op, device_ptr.result

  def device_ptr(self) -> ir.Value:
    _, _, device_ptr = self._find_alloc_load_and_device_ptr()
    return device_ptr

  def finalize_size(self):
    """
    Allocates and initializes the host buffer. This needs to be done after
    lowering, i.e. after all TMA descriptors have been recorded. Only then we
    know what the scratch contains.
    """
    if self.next_offset == 0:
      return
    alloc_op, load_op, _ = self._find_alloc_load_and_device_ptr()

    with ir.InsertionPoint(load_op):
      gmem_scratch_bytes = self.next_offset
      scratch_arr_ty = ir.Type.parse(f"!llvm.array<{gmem_scratch_bytes} x i8>")
      alloc_op.elem_type = ir.TypeAttr.get(scratch_arr_ty)
      load_op.result.set_type(scratch_arr_ty)
      for init_callback in self.host_init:
        init_callback(alloc_op.result)


class _DefaultPredicate:
  pass


def _find_kernel_argument_for_gmem_ref(
    gmem_ref: ir.Value,
) -> builtin.UnrealizedConversionCastOp:
  """Returns the kernel argument value for a given gmem_ref.

  The kernel argument is expected to be an unrealized conversion cast. This
  function will recursively go up block arguments in case of nested blocks.
  """
  if not isinstance(gmem_ref.type, ir.MemRefType):
    raise ValueError(f"Expected {gmem_ref} to have a memref type.")

  while isinstance(gmem_ref, ir.BlockArgument):
    gmem_ref = gmem_ref.owner.owner.operands[gmem_ref.arg_number]

  # TODO(apaszke): This is a very approximate check. Improve it!
  if not isinstance(gmem_ref.owner.opview, builtin.UnrealizedConversionCastOp):
    raise NotImplementedError(
        f"Expected {gmem_ref.owner} to be an unrealized conversion cast"
        " corresponding to a GMEM kernel argument."
    )
  return gmem_ref


def _is_tma_reduction_op_supported(
    reduction_op: TMAReductionOp | None, dtype: ir.Type,
) -> bool:
  """Returns whether the given TMA reduction op supports the given dtype.

  This function essentially implements the table at:
  https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor
  with the following differences:
  - For `add` reductions, we also support int64, treating it as uint64.
  - For `and`, `or`, and `xor` reductions, we support signed integer types.
  - For `inc` and `dec` reductions, we support both signed and unsigned i32
    treating both as unsigned.
  """
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  f16 = ir.F16Type.get()
  f32 = ir.F32Type.get()
  bf16 = ir.BF16Type.get()

  match reduction_op:
    case None:
      return True
    case "add":
      return dtype in (f16, f32, bf16, i32, i64)
    case "max" | "min":
      return dtype in (f16, bf16)
    case "umax" | "umin" | "smax" | "smin":
      return dtype in (i32, i64)
    case "inc" | "dec":
      return dtype == i32
    case "and" | "or" | "xor":
      return dtype in (i32, i64)


def _tma_dma_type(
    element_type: ir.Type,
    reduction_op: TMAReductionOp | None,
) -> int:
  """Returns the TMA DMA type for the given element type and signedness."""
  if ir.IntegerType.isinstance(element_type):
    bitwidth = utils.bitwidth_impl(element_type)
    if bitwidth == 2:
      tma_dtype = 8
    elif bitwidth == 4:
      tma_dtype = 0
    elif bitwidth == 8:
      tma_dtype = 1
    elif bitwidth == 16:
      tma_dtype = 2
    elif bitwidth == 32:
      tma_dtype = 9 if reduction_op in ("smin", "smax") else 3
    elif bitwidth == 64:
      tma_dtype = 10 if reduction_op in ("smin", "smax") else 4
    else:
      raise ValueError(f"Unsupported integer bitwidth: {bitwidth}")
  elif ir.F16Type.isinstance(element_type):
    tma_dtype = 5
  elif ir.F32Type.isinstance(element_type):
    tma_dtype = 6
  elif ir.BF16Type.isinstance(element_type):
    tma_dtype = 7
  # We treat narrow floats as integers
  elif ir.Float8E5M2Type.isinstance(element_type):
    tma_dtype = 1
  elif ir.Float8E4M3FNType.isinstance(element_type):
    tma_dtype = 1
  elif ir.Float8E8M0FNUType.isinstance(element_type):
    tma_dtype = 1
  elif ir.Float4E2M1FNType.isinstance(element_type):
    tma_dtype = 0
  else:
    raise ValueError(f"unsupported TMA dtype {element_type}")
  return tma_dtype


class AsyncCopyImplementation(enum.Enum):
  TMA = enum.auto()
  CP_ASYNC = enum.auto()


@dataclasses.dataclass()
class LaunchContext:
  module: ir.Module
  scratch: Scratch
  cluster_size: tuple[int, int, int]
  profiler: OnDeviceProfiler | None = None
  tma_descriptors: dict[
      tuple[ir.Value, tuple[int, ...], int | None, tuple[MemRefTransform, ...], Any, int],
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
      self,
      dim: gpu.Dimension | Sequence[gpu.Dimension] | None = None,
      dim_idx: ir.Value | Sequence[ir.Value] | None = None,
  ) -> ir.Value:
    """Returns the index of a block within a subset of the cluster spanned by the given dimensions."""
    if dim is None:
      dim = gpu.Dimension
    elif isinstance(dim, gpu.Dimension):
      dim = (dim,)
    if dim_idx is None:
      dim_idx = [gpu.cluster_block_id(d) for d in dim]
    elif isinstance(dim_idx, ir.Value):
      if len(dim) != 1:
        raise ValueError(
            "Expected a single dimension when passing a single index"
        )
      dim_idx = [dim_idx]
    index = ir.IndexType.get()
    stride = 1
    lin_idx = c(0, index)
    for d, idx in sorted(zip(dim, dim_idx), key=lambda x: x[0]):
      if self.cluster_size[d] == 1:  # Optimize a multiply by 0.
        continue
      lin_idx = arith.addi(lin_idx, arith.muli(idx, c(stride, index)))
      stride *= self.cluster_size[d]
    return lin_idx

  def get_cluster_ref(self, ref: ir.Value, dim: gpu.Dimension, idx: ir.Value):
    i32 = ir.IntegerType.get_signless(32)
    # We replace the offset in the ref type by 0, because memref_ptr always
    # folds the offset into the pointer.
    ref_ty = ir.MemRefType(ref.type)
    strides, _ = ref_ty.get_strides_and_offset()
    result_type = ir.MemRefType.get(
        ref_ty.shape,
        ref_ty.element_type,
        ir.StridedLayoutAttr.get(0, strides),
        None,
    )
    if ref_ty.memory_space != ir.Attribute.parse("#gpu.address_space<workgroup>"):
      raise ValueError(f"Expected SMEM but got: {ref.memory_space}")
    idxs = [gpu.cluster_block_id(d) for d in gpu.Dimension]
    idxs[dim] = idx
    flat_block = arith.index_cast(i32, self.cluster_idx(gpu.Dimension, idxs))  # type: ignore
    return utils.ptr_as_memref(
        utils.get_cluster_ptr(
            utils.memref_ptr(ref, memory_space=3), flat_block
        ),
        result_type,
    )

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
      gmem_ref: ir.Value,
      gmem_transform: tuple[MemRefTransform, ...],
      gmem_peer_id: int | ir.Value | GlobalBroadcast | None,
      transformed_slice_shape: tuple[int, ...],
      swizzle: int | None,
      reduction_op: TMAReductionOp | None,
  ):
    gmem_ref = _find_kernel_argument_for_gmem_ref(gmem_ref)
    tma_dtype = _tma_dma_type(ir.MemRefType(gmem_ref.type).element_type, reduction_op)
    # Using ir.Values in cache keys is a little sketchy, but I think it should
    # be fine. Having it in the key will keep it alive, and if comparison and
    # hashing is by identity then it should work out.
    tma_desc_key = (gmem_ref, transformed_slice_shape, swizzle, gmem_transform, gmem_peer_id, tma_dtype)
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
        if isinstance(gmem_peer_id, GlobalBroadcast):
          self._ensure_nvshmem_decls()
          world_team = arith.constant(i32, 0)
          base_ptr = llvm.call(
              base_ptr.type,
              [world_team, base_ptr],
              [],
              [],
              callee="nvshmemx_mc_ptr",
          )
        elif gmem_peer_id is not None:
          if not isinstance(gmem_peer_id, ir.Value):
            peer_id = c(gmem_peer_id, i32)
          else:
            try:
              # We try to reproduce the gmem_peer_id computation on the host.
              peer_id = _recompute_peer_id(gmem_peer_id, fuel=16)
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

  def _prepare_async_copy(
      self,
      gmem_ref: ir.Value,
      gmem_slice: Any,
      gmem_transform: tuple[MemRefTransform, ...],
      collective: Sequence[gpu.Dimension] | None,
      partitioned: int | None,
      implementation: AsyncCopyImplementation,
  ):
    """Performs setup common to TMA and CP_ASYNC implementations."""
    index = ir.IndexType.get()

    gmem_ref_ty = ir.MemRefType(gmem_ref.type)
    gmem_strides, _ = gmem_ref_ty.get_strides_and_offset()
    if gmem_strides != utils.get_contiguous_strides(gmem_ref_ty.shape):
      raise NotImplementedError(
          "async_copy assumes the GMEM reference is contiguous"
      )

    # Look for and verify gather indices in gmem_slice.
    is_gathered_dim = [isinstance(s, fa.FragmentedArray) for s in gmem_slice]
    gather_indices: fa.FragmentedArray | None = None
    if any(is_gathered_dim):
      if is_gathered_dim != [True, False]:
        raise NotImplementedError(
            "Gathers/scatters only supported along the first dimension of 2D"
            " arrays"
        )
      gather_indices = gmem_slice[0]
      if not isinstance(gather_indices, fa.FragmentedArray):
        raise ValueError("Gather/scatter indices must be a FragmentedArray")
      if len(gather_indices.shape) != 1:
        raise ValueError("Gather/scatter indices must be 1D")
      idx_dtype = gather_indices.mlir_dtype
      if not ir.IntegerType.isinstance(idx_dtype) or utils.bitwidth(idx_dtype) > 32:
        raise ValueError("Gather/scatter indices must be integers that are at most 32-bit wide")
      if gather_indices.is_signed:
        raise ValueError("Gather/scatter indices must be unsigned")
      gmem_slice = (slice(None), *gmem_slice[1:])

    # Analyze the slice (taking gathers into account).
    base_indices, slice_shape, is_squeezed = utils.parse_indices(
        gmem_slice,
        ir.MemRefType(gmem_ref.type).shape,
        # NOTE: TMA supports OOB indices, so we skip the check.
        check_oob=implementation != AsyncCopyImplementation.TMA,
    )
    if gather_indices is not None:
      slice_shape = [gather_indices.shape[0], *slice_shape[1:]]
    del gmem_slice  # Use slice_shape, base_indices and is_squeezed from now on!
    dyn_base_indices = tuple(
        c(i, index) if not isinstance(i, ir.Value) else i for i in base_indices
    )
    del base_indices  # Use the dynamic indices from now on!

    # Deal with collective and partitioned loads.
    if collective:
      if implementation != AsyncCopyImplementation.TMA:
        raise ValueError("Only the TMA implementation supports collective copies")
      if gather_indices is not None:
        raise NotImplementedError("Collective copies with gather/scatter unsupported")
    if partitioned is not None:
      # Increment partitioned by the number of preceding squeezed dimensions.
      partitioned = np.where(
          np.cumsum(~np.array(is_squeezed)) == partitioned+1)[0][0]
      # Partitioning happens on the logical slice we extract from GMEM, so we do
      # it before we apply transforms.
      if not collective:  # This implies non-gather TMA already.
        raise ValueError("Only collective loads can be partitioned")
      collective_size = math.prod(self.cluster_size[d] for d in collective)
      if collective_size > 1:
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
        dyn_base_indices = list(dyn_base_indices)  # type: ignore[assignment]
        dyn_base_indices[partitioned] = arith.addi(  # type: ignore[index]
            dyn_base_indices[partitioned],
            arith.muli(
                self.cluster_idx(collective), c(slice_shape[partitioned], index)
            ),
        )
        dyn_base_indices = tuple(dyn_base_indices)

    squeezed_dims = tuple(
        i for i, squeezed in enumerate(is_squeezed) if squeezed
    )
    # Indexing is really slicing + squeezing, and user transforms are meant to
    # apply after that. However, we actually have to apply the indexing last
    # (it's fused into the TMA) and so we need to commute it with all the user
    # transforms. For slicing this is done using transform_index and
    # transform_shape. For squeezing we actually move all the squeezed dims to
    # the front, and then batch each transform, making it ignore the extra dims.
    if squeezed_dims and implementation != AsyncCopyImplementation.CP_ASYNC:
      sliced_dims = [i for i, squeezed in enumerate(is_squeezed) if not squeezed]
      gmem_transform = (TransposeTransform((*squeezed_dims, *sliced_dims)),
                        *(t.batch(len(squeezed_dims)) for t in gmem_transform))

    slice_shape = tuple(slice_shape)
    for t in gmem_transform:
      dyn_base_indices = t.transform_index(dyn_base_indices)
      slice_shape = t.transform_shape(slice_shape)

    return (
        list(slice_shape),
        dyn_base_indices,
        squeezed_dims,
        gather_indices,
        gmem_transform,
    )

  def _prepare_tma(
      self,
      gmem_ref: ir.Value,
      smem_ref: ir.Value | None,
      swizzle: int | None,
      slice_shape: list[int],
      dyn_base_indices: tuple[ir.Value, ...],
      gather_indices,
      squeezed_dims: tuple[int, ...],
      gmem_transform: tuple[MemRefTransform, ...],
      collective: Sequence[gpu.Dimension],
      partitioned: int | None,
  ):
    """Finalizes setup specific to the TMA implementation of async_copy."""
    index = ir.IndexType.get()
    # The function below is called only to verify the GMEM ref. The output
    # is meant to be ignored.
    _find_kernel_argument_for_gmem_ref(gmem_ref)
    gmem_ref_ty = ir.MemRefType(gmem_ref.type)
    element_bitwidth = utils.bitwidth(gmem_ref_ty.element_type)
    gmem_strides, _ = gmem_ref_ty.get_strides_and_offset()
    if any(s * element_bitwidth % 128 != 0 for s in gmem_strides[:-1]):
      raise ValueError(
          "async_copy requires all GMEM strides except the last one to be a"
          " multiple of 16 bytes"
      )
    # We don't need to do this for gather TMAs, because we'll unroll the
    # transfers ourselves anyway.
    num_squeezed_dims = len(squeezed_dims)
    if len(slice_shape) > 5 and gather_indices is None:
      # We can try to collapse all squeezed dims into one.
      if len(slice_shape) - num_squeezed_dims + 1 > 5:
        raise ValueError(
            "Async copies only support striding up to 5 dimensions"
        )
      squeezed_dim_strides = tuple(gmem_strides[d] for d in squeezed_dims)
      collapse = CollapseLeadingIndicesTransform(squeezed_dim_strides)
      gmem_transform = (*gmem_transform, collapse)
      dyn_base_indices = collapse.transform_index(dyn_base_indices)
      slice_shape = list(collapse.transform_shape(tuple(slice_shape)))
      num_squeezed_dims = 1

    dyn_base_indices = list(dyn_base_indices)
    slice_shape = list(slice_shape)
    assert all(d == 1 for d in slice_shape[:num_squeezed_dims])

    # Partitioned loads have already been processed (before transforms).
    # We process non-partitioned collective loads here, because only here are we
    # able to know in what order the data will be written to SMEM. Transposes
    # and tiling change that order and if we picked a partition based on the
    # untransformed slice shape, we might have ended up with a non-contiguous
    # SMEM window, which would no longer be realizable in a single TMA transfer.
    collective_size = math.prod(self.cluster_size[d] for d in collective)  # type: ignore
    if collective_size > 1 and partitioned is None:
      assert gather_indices is None  # Checked above.
      def partition_dim(dim: int, idx: ir.Value, num_chunks: int):
        # No need to partition squeezed dims. They don't even exist in smem_ref.
        assert dim >= num_squeezed_dims
        nonlocal smem_ref
        slice_shape[dim] //= num_chunks
        block_offset = arith.muli(idx, c(slice_shape[dim], index))
        dyn_base_indices[dim] = arith.addi(dyn_base_indices[dim], block_offset)  # type: ignore[index]
        if smem_ref is not None:
          smem_ref = utils.memref_slice(
              smem_ref,
              (slice(None),) * (dim - num_squeezed_dims)
              + (utils.ds(block_offset, slice_shape[dim]),),
          )
      idx = self.cluster_idx(collective)
      rem_collective_size = collective_size
      has_swizzle = (
          swizzle is not None
          and swizzle != mgpu_dialect.SwizzlingMode.kNoSwizzle
      )
      # We can partition the minormost dim if there's no swizzling.
      for dim, slice_size in enumerate(
          slice_shape[:-1] if has_swizzle else slice_shape
      ):
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

    if max(slice_shape) > 256:
      raise ValueError(
          "Async copies only support copying <=256 elements along each"
          f" dimension, got {tuple(slice_shape)}"
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
    return (smem_ref, slice_shape, dyn_base_indices, gmem_transform)

  def async_copy(
      self,
      *,
      src_ref: ir.Value,
      dst_ref: ir.Value,
      gmem_slice: Any = (),
      gmem_transform: MemRefTransform | tuple[MemRefTransform, ...] = (),
      gmem_peer_id: int | ir.Value | GlobalBroadcast | None = None,
      barrier: utils.BarrierRef | None = None,
      swizzle: int | None = None,
      arrive: bool | None = None,
      collective: Sequence[gpu.Dimension] | gpu.Dimension | None = None,
      partitioned: int | None = None,
      # Should select 0 or 1 threads from the WG.
      predicate: ir.Value | None | _DefaultPredicate = _DefaultPredicate(),
      reduction_op: TMAReductionOp | None = None,
      implementation: AsyncCopyImplementation = AsyncCopyImplementation.TMA,
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
    i8 = ir.IntegerType.get_signless(8)
    i16 = ir.IntegerType.get_signless(16)
    i32 = ir.IntegerType.get_signless(32)

    src_ref_ty = ir.MemRefType(src_ref.type)
    dst_ref_ty = ir.MemRefType(dst_ref.type)
    element_type = src_ref_ty.element_type
    element_bitwidth = utils.bitwidth(element_type)
    if element_type != dst_ref_ty.element_type:
      raise ValueError(
          f"Expected same element type, got {element_type} and"
          f" {dst_ref_ty.element_type}"
      )

    if isinstance(collective, gpu.Dimension):
      collective = (collective,)
    elif collective is None:
      collective = ()
    if not isinstance(gmem_transform, tuple):
      gmem_transform = (gmem_transform,)
    if not isinstance(gmem_slice, tuple):
      gmem_slice = (gmem_slice,)

    if reduction_op is not None:
      if implementation != AsyncCopyImplementation.TMA:
        raise ValueError("Only the TMA implementation supports reductions")
      if not _is_tma_reduction_op_supported(reduction_op, element_type):
        raise ValueError(
            f"Reduction op {reduction_op} not supported by the TMA"
            f" implementation for element type {element_type}"
        )

    if src_ref_ty.memory_space is None and utils.is_smem_ref(dst_ref_ty):
      gmem_ref, smem_ref = src_ref, dst_ref
      if implementation == AsyncCopyImplementation.TMA:
        if barrier is None:
          raise ValueError("Barriers are required for TMA GMEM -> SMEM copies")
      else:
        assert implementation == AsyncCopyImplementation.CP_ASYNC
        if barrier is not None:
          raise NotImplementedError(
              "Barriers are unsupported for CP_ASYNC GMEM -> SMEM copies"
          )
      if arrive is None:
        arrive = True  # Arrive by default
    elif utils.is_smem_ref(src_ref_ty) and dst_ref_ty.memory_space is None:
      gmem_ref, smem_ref = dst_ref, src_ref
      if barrier is not None:
        raise ValueError("Barriers are unsupported for SMEM -> GMEM copies")
      if arrive is None:
        arrive = True  # Commit this copy to the async group by default
    else:
      raise ValueError("Only SMEM <-> GMEM copies supported")

    if collective and gmem_ref is dst_ref:
      raise ValueError("Only GMEM -> SMEM copies can be collective")

    (
        slice_shape,
        dyn_base_indices,
        squeezed_dims,
        gather_indices,
        gmem_transform,
    ) = self._prepare_async_copy(
        gmem_ref,
        gmem_slice,
        gmem_transform,
        collective,
        partitioned,
        implementation,
    )
    del gmem_slice  # Use slice_shape, dyn_base_indices and squeezed_dims instead.

    gmem_ref_ty = ir.MemRefType(gmem_ref.type)
    smem_ref_ty = ir.MemRefType(smem_ref.type)
    # TODO(apaszke): Support squeezed dims for CP_ASYNC.
    if implementation == AsyncCopyImplementation.CP_ASYNC and squeezed_dims:
      raise NotImplementedError(
          "Integer indexing in gmem_slice not supported for CP_ASYNC"
      )
    # We moved all squeezed dims to the front in _prepare_async_copy.
    assert all(d == 1 for d in slice_shape[:len(squeezed_dims)])
    if slice_shape[len(squeezed_dims):] != smem_ref_ty.shape:
      raise ValueError(
          "Expected the SMEM reference to have the same shape as the"
          f" transformed slice: {tuple(smem_ref_ty.shape)} !="
          f" {tuple(slice_shape[len(squeezed_dims):])}"
      )

    if implementation == AsyncCopyImplementation.CP_ASYNC:
      assert not collective
      assert partitioned is None
      if not isinstance(predicate, _DefaultPredicate):
        raise NotImplementedError(
            "CP_ASYNC needs to be performed by the whole warpgroup and does not"
            " support the predicate argument"
        )
      # TODO(apaszke): This should be quite easy? The only complication is that
      # the indices array needs to have a layout compatible with the way we
      # assign lanes to rows/cols.
      if gather_indices is not None:
        raise NotImplementedError("Gather/scatter unsupported for the CP_ASYNC implementation")
      if smem_ref is src_ref:
        raise ValueError("CP_ASYNC implementation only supports GMEM -> SMEM copies")
      assert swizzle is not None
      swizzle_elems = 8 * swizzle // element_bitwidth
      if gmem_transform != (TileTransform((8, swizzle_elems)),):
        raise NotImplementedError(gmem_transform)
      layout = fa.tiled_copy_smem_gmem_layout(
          *smem_ref_ty.shape[-4:-2], swizzle, element_bitwidth  # type: ignore[call-arg]
      )
      gmem_strides = gmem_ref_ty.get_strides_and_offset()[0]
      dst_tiled_strides = [
          arith.constant(i32, s)
          for s in layout.tiling.tile_strides(gmem_strides)[gmem_ref_ty.rank :]
      ]
      lane_offset = utils.dyn_dot(layout.lane_indices(), dst_tiled_strides)
      warp_offset = utils.dyn_dot(layout.warp_indices(), dst_tiled_strides)
      dyn_offset = arith.addi(lane_offset, warp_offset)
      offset_scale = 1 if element_bitwidth >= 8 else 8 // element_bitwidth
      if element_bitwidth < 8:
        gep_type = i8
      elif ir.FloatType.isinstance(element_type) and ir.FloatType(element_type).width == 8:
        gep_type = i8  # LLVM has no support for f8.
      else:
        gep_type = element_type
      dyn_offset = arith.divui(dyn_offset, c(offset_scale, i32))
      if gmem_ref_ty.rank != 2:
        raise NotImplementedError("Only 2D copies implemented")
      transfers = fa.FragmentedArray.transfer_tiled(
          smem_ref, swizzle, layout, tuple(gmem_ref_ty.shape), optimized=False
      )
      gmem_base_ptr = utils.getelementptr(utils.memref_ptr(gmem_ref), [dyn_offset], gep_type)
      gmem_base_ptr = llvm.addrspacecast(ir.Type.parse("!llvm.ptr<1>"), gmem_base_ptr)
      bytes_per_transfer = layout.vector_length * element_bitwidth // 8
      # Only 16-byte transfers can skip the L1 cache (this is what CG means).
      cache_modifier = (
          nvvm.LoadCacheModifierKind.CG
          if bytes_per_transfer == 16
          else nvvm.LoadCacheModifierKind.CA
      )
      for _get, _update, get_base_idx, smem_ptr in transfers:
        constant_offset = sum(i * s for i, s in zip(get_base_idx(), gmem_strides, strict=True))
        gmem_ptr = utils.getelementptr(gmem_base_ptr, [constant_offset // offset_scale], gep_type)
        nvvm.cp_async_shared_global(smem_ptr, gmem_ptr, bytes_per_transfer, cache_modifier)
      if barrier is None:
        nvvm.cp_async_commit_group()
      else:
        raise NotImplementedError
      return

    assert implementation == AsyncCopyImplementation.TMA

    (smem_ref, slice_shape, dyn_base_indices, gmem_transform) = (
        self._prepare_tma(
            gmem_ref,
            smem_ref,
            swizzle,
            slice_shape,
            dyn_base_indices,
            gather_indices,
            squeezed_dims,
            gmem_transform,
            collective,
            partitioned,
        )
    )
    assert smem_ref is not None  # For type checkers.

    smem_strides, _ = ir.MemRefType(smem_ref.type).get_strides_and_offset()
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

    collective_size = math.prod(self.cluster_size[d] for d in collective)
    assert math.prod(slice_shape) * element_bitwidth * collective_size % 8 == 0
    transfer_bytes = c(
        math.prod(slice_shape) * element_bitwidth * collective_size // 8, i32
    )

    if gather_indices is not None:
      import builtins
      zips = functools.partial(builtins.zip, strict=True)
      # The gather TMA instruction is limited to 2D GMEM references. That means
      # that we can't apply the transforms to the GMEM reference and have the
      # TMA engine deal with permuting the data, like we do for non-gather TMA.
      # Instead, we have to break up the transfer into multiple 2D gathers
      # ourselves, which requires us to do more complicated stride math etc.
      #
      # The minor transformed dim should be a contiguous transfer dim.
      # The second minor should be a gather dim of size divisible by 4.
      # The rest can be anything, and we will unroll the transfers over them.
      if smem_ref is src_ref:
        raise NotImplementedError("Scatter unsupported for the TMA implementation")
      assert barrier is not None  # for pytype
      barrier_ptr = barrier.get_ptr()
      if squeezed_dims:
        raise NotImplementedError("Gather/scatter unsupported when using integer indexing")
      if reduction_op is not None:
        raise ValueError("Gather/scatter TMA can't perform reductions")
      if not isinstance(predicate, _DefaultPredicate):
        raise ValueError("Gather/scatter TMA can't use a predicate")
      if gather_indices.layout != fa.TMA_GATHER_INDICES_LAYOUT:
        raise ValueError(f"Unsupported gather indices layout: {gather_indices.layout}")
      ROWS_PER_INSTR = 4
      # Make sure we'll always be accessing SMEM with sufficient alignment.
      single_tma_bits = ROWS_PER_INSTR * slice_shape[-1] * element_bitwidth
      if single_tma_bits % 1024:
        raise ValueError(
            "Gather/scatter TMA would require breaking it up into transfers of"
            f" {single_tma_bits // 8} bytes, but need a multiple of 128 bytes"
        )

      if arrive:
        arrive_predicate = utils.single_thread_predicate(utils.ThreadSubset.WARPGROUP)
        utils.nvvm_mbarrier_arrive_expect_tx(
            barrier_ptr,
            transfer_bytes,
            predicate=arrive_predicate,
        )

      gmem_strides, _ = gmem_ref_ty.get_strides_and_offset()
      assert len(gmem_strides) == 2
      _, gmem_cols = gmem_ref_ty.shape
      slice_gather_strides: tuple[int, ...] = (1, 0)  # Each row gets a new index, column has no effect.
      for t in gmem_transform:
        gmem_strides = t.transform_strides(gmem_strides)
        slice_gather_strides = t.transform_strides(slice_gather_strides)
      is_gather_dim = [bool(s) for s in slice_gather_strides]

      tma_desc = self._get_tma_desc(
          gmem_ref, (), gmem_peer_id, (1, slice_shape[-1]), swizzle, reduction_op,
      )

      # Indices are split over 4 warps, and replicated within each warp.
      assert fa.TMA_GATHER_INDICES_LAYOUT.vector_length == ROWS_PER_INSTR
      # Index 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 ...
      # Warp  <--- 0 ---> <--- 1 ---> <--- 2 ---> <--- 3 ---> <--- 0 --
      warp_idx = arith.remui(
          utils.warp_idx(sync=True),
          arith.constant(i32, utils.WARPS_IN_WARPGROUP),
      )
      gather_linear_idx_warp = arith.muli(warp_idx, c(ROWS_PER_INSTR, i32))

      # Since the TMA instruction is limited to 2D gathers, we flatten all
      # non-gather dims into the column index.
      max_non_gather_linear_index = sum(
          (d - 1) * s
          for g, d, s in zip(is_gather_dim[:-1], slice_shape[:-1], gmem_strides[:-1])
          if not g
      )
      # If we ever exceed this then we need to change the size of the GMEM ref,
      # to prevent the TMA engine from clipping our indices.
      if max_non_gather_linear_index > gmem_cols:
        raise NotImplementedError("Non-gather dims don't fit into the columns")
      col_base_offset = functools.reduce(
          arith.addi,
          (
              arith.muli(idx, arith.constant(index, stride))
              for g, idx, stride in zips(
                  is_gather_dim, dyn_base_indices, gmem_strides
              )
              if not g
          ),
          arith.constant(index, 0),
      )
      col_base_offset = arith.index_cast(i32, col_base_offset)
      # TMA instructions are uniform, so we can't use multiple lanes.
      predicate = utils.single_thread_predicate(utils.ThreadSubset.WARP)
      # We need to unroll over all non-gather dimensions other than the last one
      non_gather_slice_shape = tuple(
          1 if g else d for d, g in zips(slice_shape[:-1], is_gather_dim[:-1])
      )
      # First, iterate over gather index registers we have available.
      for i, reg in enumerate(gather_indices.registers.flat):
        if utils.bitwidth(gather_indices.mlir_dtype) != 32:
          reg = arith.extui(ir.VectorType.get((4,), i32), reg)
        # Compute which rows within the 2D slice we'll be gathering.
        gather_linear_idx_reg = i * ROWS_PER_INSTR * utils.WARPS_IN_WARPGROUP
        gather_linear_idx = arith.addi(
            gather_linear_idx_warp, arith.constant(i32, gather_linear_idx_reg)
        )
        # Transform row indices to align with the transformed SMEM shape.
        gather_slice_idx = [
            arith.remui(arith.divui(gather_linear_idx, c(s, i32)), c(d, i32))
            for g, d, s in zip(is_gather_dim, slice_shape, slice_gather_strides)
            if g
        ]
        gather_slice_idx = [arith.index_cast(index, i) for i in gather_slice_idx]
        gather_rows = [
            llvm.extractelement(reg, c(i, i32)) for i in range(ROWS_PER_INSTR)
        ]
        # Second, step over non-gather slice indices.
        for non_gather_idxs in np.ndindex(non_gather_slice_shape):
          gather_slice_idx_it = iter(gather_slice_idx)
          smem_indices = tuple(
              next(gather_slice_idx_it) if g else i
              for g, i in zip(is_gather_dim[:-1], non_gather_idxs)
          )
          # We should really take a slice here, but it doesn't matter. We're
          # just going to take the base pointer anyway.
          transfer_smem_ref = utils.memref_slice(smem_ref, smem_indices)
          smem_ptr = utils.memref_ptr(transfer_smem_ref, memory_space=3)
          # The slice index needs to be folded into the gather col index.
          col_slice_offset = sum(
              idx * stride
              for g, idx, stride in zips(
                  is_gather_dim[:-1], non_gather_idxs, gmem_strides[:-1]
              )
              if not g
          )
          col_offset = arith.addi(col_base_offset, arith.constant(i32, col_slice_offset))
          llvm.inline_asm(
              ir.Type.parse("!llvm.void"),
              [predicate, smem_ptr, tma_desc, barrier_ptr, col_offset, *gather_rows],
              "@$0 cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [$1], [$2, {$4, $5, $6, $7, $8}], [$3];",
              "b,r,l,r" + ",r" * (ROWS_PER_INSTR + 1),
              has_side_effects=True,
          )
      return

    assert gather_indices is None  # Only tiled TMA handled below.
    tma_desc = self._get_tma_desc(
        gmem_ref, gmem_transform, gmem_peer_id,
        tuple(slice_shape), swizzle, reduction_op,
    )
    # We construct TMA descriptors in column-major order.
    rev_dyn_base_indices = [
        arith.index_cast(i32, idx) for idx in reversed(dyn_base_indices)
    ]
    if isinstance(predicate, _DefaultPredicate):
      predicate = utils.single_thread_predicate(utils.ThreadSubset.WARPGROUP)
    if predicate is None:
      predicate = c(1, ir.IntegerType.get_signless(1))
    smem_ptr = utils.memref_ptr(smem_ref, memory_space=3)
    if gmem_ref is src_ref:
      assert barrier is not None  # for pytype
      barrier_ptr = barrier.get_ptr()
      assert reduction_op is None
      if collective_size > 1 and partitioned is not None:
        assert collective_size == 2
        if arrive:
          first_block = arith.cmpi(
              arith.CmpIPredicate.eq, self.cluster_idx(collective), c(0, index),
          )
          arrive_predicate = arith.andi(predicate, first_block)
          utils.nvvm_mbarrier_arrive_expect_tx(
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
          utils.nvvm_mbarrier_arrive_expect_tx(
              barrier_ptr, transfer_bytes, predicate=predicate
          )
        if collective_size > 1:
          multicast_mask = arith.trunci(
              i16, utils.cluster_collective_mask(self.cluster_size, collective)
          )
        else:
          multicast_mask = None
        nvvm.cp_async_bulk_tensor_shared_cluster_global(
            smem_ptr, tma_desc, rev_dyn_base_indices, barrier_ptr, [],
            multicast_mask=multicast_mask, predicate=predicate
        )
    else:
      if reduction_op is not None:
        rank = len(slice_shape)
        idx_operands = ",".join(f"${i}" for i in range(3, 3 + rank))
        llvm.inline_asm(
          ir.Type.parse("!llvm.void"),
          [predicate,smem_ptr,tma_desc,*rev_dyn_base_indices],
          f"@$0 cp.reduce.async.bulk.tensor.{rank}d.global.shared::cta.{_reduction_op_to_ptx(reduction_op)}.tile.bulk_group [$2,{{{idx_operands}}}], [$1];",
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

  def async_prefetch(
    self,
    *,
    gmem_ref: ir.Value,
    gmem_slice: Any = (),
    gmem_transform: MemRefTransform | tuple[MemRefTransform, ...] = (),
    gmem_peer_id: int | ir.Value | None = None,
    swizzle: int | None = None,
    collective: Sequence[gpu.Dimension] | gpu.Dimension | None = None,
    partitioned: int | None = None,
    # Should select 0 or 1 threads from the WG.
    predicate: ir.Value | None | _DefaultPredicate = _DefaultPredicate(),
  ):
    i32 = ir.IntegerType.get_signless(32)

    if isinstance(collective, gpu.Dimension):
      collective = (collective,)
    elif collective is None:
      collective = ()
    if not isinstance(gmem_transform, tuple):
      gmem_transform = (gmem_transform,)
    if not isinstance(gmem_slice, tuple):
      gmem_slice = (gmem_slice,)

    impl =  AsyncCopyImplementation.TMA
    (
        slice_shape,
        dyn_base_indices,
        squeezed_dims,
        gather_indices,
        gmem_transform,
    ) = self._prepare_async_copy(
        gmem_ref, gmem_slice, gmem_transform, collective, partitioned, impl
    )
    del gmem_slice  # Use slice_shape, dyn_base_indices and squeezed_dims instead.

    (_, slice_shape, dyn_base_indices, gmem_transform) = (
        self._prepare_tma(
            gmem_ref,
            None,
            swizzle,
            slice_shape,
            dyn_base_indices,
            gather_indices,
            squeezed_dims,
            gmem_transform,
            collective,
            partitioned,
        )
    )

    if gather_indices is not None:
      raise NotImplementedError("Gather/scatter prefetch not implemented yet")

    tma_desc = self._get_tma_desc(
        gmem_ref, gmem_transform, gmem_peer_id,
        tuple(slice_shape), swizzle, reduction_op=None,
    )
    # We construct TMA descriptors in column-major order.
    rev_dyn_base_indices = [
        arith.index_cast(i32, idx) for idx in reversed(dyn_base_indices)
    ]
    if isinstance(predicate, _DefaultPredicate):
      predicate = utils.single_thread_predicate(utils.ThreadSubset.WARPGROUP)
    if predicate is None:
      predicate = c(1, ir.IntegerType.get_signless(1))
    rank = len(slice_shape)
    idx_operands = ",".join(f"${i}" for i in range(2, 2 + rank))
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [predicate, tma_desc, *rev_dyn_base_indices],
        f"@$0 cp.async.bulk.prefetch.tensor.{rank}d.L2.global.tile [$1, {{{idx_operands}}}];",
        "b,l" + ",r" * rank,
        has_side_effects=True,
    )

  def await_async_copy(
      self, allow_groups: int, await_read_only: bool = False,
      scope: utils.ThreadSubset = utils.ThreadSubset.WARPGROUP,
  ):
    nvvm.cp_async_bulk_wait_group(allow_groups, read=await_read_only)
    if scope == utils.ThreadSubset.WARPGROUP:
      utils.warpgroup_barrier()
    elif scope == utils.ThreadSubset.WARP:
      utils.warp_barrier()
    else:
      raise ValueError(f"Unsupported scope: {scope}")

  def await_cp_async_copy(self, allow_groups: int):
    nvvm.cp_async_wait_group(allow_groups)
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
      nvshmemx_mc_ptr_type = ir.TypeAttr.get(
          ir.Type.parse("!llvm.func<!llvm.ptr(i32,!llvm.ptr)>")
      )
      llvm.LLVMFuncOp(
          "nvshmemx_mc_ptr", nvshmemx_mc_ptr_type, sym_visibility="private"
      )

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

  def to_remote_multicast(self, ref: ir.Value):
    i32 = ir.IntegerType.get_signless(32)
    self._ensure_nvshmem_decls()
    if not ir.MemRefType.isinstance(ref.type):
      raise ValueError(f"Unsupported type for to_remote_multicast: {ref.type}")
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
    world_team = arith.constant(i32, 0)
    ptr = utils.memref_ptr(ref)
    mc_ptr = llvm.call(
        ptr.type, [world_team, ptr], [], [], callee="nvshmemx_mc_ptr",
    )
    return utils.MultimemRef(utils.ptr_as_memref(mc_ptr, result_type))

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
    new_attributes = {na: op.attributes[na] for na in op.attributes}
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
