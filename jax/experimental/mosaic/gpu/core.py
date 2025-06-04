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

from collections.abc import Sequence
import contextlib
import ctypes
import dataclasses
import enum
import hashlib
import math
import os
import pathlib
import time
from typing import Any, Callable, Generic, TypeVar
import weakref

import itertools
import jax
from jax._src import sharding_impls
from jax._src.interpreters import mlir
from jax._src.lib import mosaic_gpu_dialect as dialect
from jaxlib.mlir import ir
from jaxlib.mlir import passmanager
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import builtin
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
import numpy as np


# mypy: ignore-errors

from . import dialect_lowering
from . import launch_context
from . import layout_inference
from . import profiler
from . import tcgen05
from . import transform_inference
from . import utils

# MLIR can't find libdevice unless we point it to the CUDA path
# TODO(apaszke): Unify with jax._src.lib.cuda_path
CUDA_ROOT = "/usr/local/cuda"
if os.environ.get("CUDA_ROOT") is None:
  os.environ["CUDA_ROOT"] = CUDA_ROOT
else:
  CUDA_ROOT = os.environ["CUDA_ROOT"]

PTXAS_PATH = os.path.join(CUDA_ROOT, "bin/ptxas")
NVDISASM_PATH = os.path.join(CUDA_ROOT, "bin/nvdisasm")

# This tracks the latest Mosaic GPU IR version with a monthly delay.
FWD_COMPAT_IR_VERSION = 1

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


try:
  from nvidia import nvshmem
except ImportError:
  pass
else:
  if os.environ.get("MOSAIC_GPU_NVSHMEM_BC_PATH") is None:
    os.environ["MOSAIC_GPU_NVSHMEM_BC_PATH"] = os.path.join(
        nvshmem.__path__[0], "lib/libnvshmem_device.bc"
    )
  if os.environ.get("MOSAIC_GPU_NVSHMEM_SO_PATH") is None:
    os.environ["MOSAIC_GPU_NVSHMEM_SO_PATH"] = os.path.join(
        nvshmem.__path__[0], "lib/libnvshmem_host.so.3"
    )


def supports_cross_device_collectives():
  try:
    nvshmem_bc_path = os.environ["MOSAIC_GPU_NVSHMEM_BC_PATH"]
  except KeyError:
    return False
  if nvshmem_so_path := os.environ.get("MOSAIC_GPU_NVSHMEM_SO_PATH", ""):
    try:
      # This both ensures that the file exists, and it populates the dlopen
      # cache, helping XLA find the library even if the RPATH is not right...
      ctypes.CDLL(nvshmem_so_path)
    except OSError:
      return False
  xla_flags = os.environ.get("XLA_FLAGS", "")
  return (
      os.path.exists(nvshmem_bc_path)
      and "--xla_gpu_experimental_enable_nvshmem" in xla_flags
  )


mosaic_gpu_p = jax._src.core.Primitive("mosaic_gpu_p")
mosaic_gpu_p.multiple_results = True


@mosaic_gpu_p.def_abstract_eval
def _mosaic_gpu_abstract_eval(*_, module, out_types, inout_types):
  del module  # Unused.
  return [
      jax._src.core.ShapedArray(t.shape, t.dtype)
      for t in itertools.chain(out_types, inout_types)
  ]


def _has_communication(module, **_):
  empty_str_attr = ir.StringAttr.get("")
  for op in module.body:
    if "nvshmem" in getattr(op, "sym_name", empty_str_attr).value:
      return True
  return False


# TODO(apaszke): Implement a proper system for managing kernel lifetimes
KNOWN_KERNELS = {}


def _mosaic_gpu_lowering_rule(
    ctx,
    *args,
    module,
    out_types,
    inout_types,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
    use_custom_barrier: bool = False,
):
  axis_context = ctx.module_context.axis_context
  if _has_communication(module):
    # Those checks are trying to ensure that the logical device ids are
    # consistent with the NVSHMEM PE ids that Mosaic will be using for
    # communication. Any divergence here would require us to implement a logical
    # to physical translation, which is currently not implemented.
    if isinstance(axis_context, sharding_impls.SPMDAxisContext):
      mesh = axis_context.mesh
      if not np.array_equal(mesh.device_ids.ravel(), np.arange(mesh.size)):
        raise NotImplementedError(
            "Mosaic GPU only supports meshes with device ordering that follows"
            " row-major device ids."
        )
    elif isinstance(axis_context, sharding_impls.ShardingContext):
      if axis_context.num_devices != 1:
        raise NotImplementedError(
            "Mosaic GPU only supports single-device meshes in ShardingContext."
        )
    else:
      raise NotImplementedError(f"Unsupported sharding context: {axis_context}")

  if inout_types:
    if input_output_aliases:
      raise ValueError(
          "input_output_aliases and inout_types are mutually exclusive"
      )
    num_inputs = len(ctx.avals_in)
    num_outputs = len(ctx.avals_out)
    input_output_aliases = tuple(
        (num_inputs - 1 - i, num_outputs - 1 - i)
        for i in range(len(inout_types))
    )
  assert len(ctx.avals_in) == len(args)
  assert len(ctx.avals_out) == len(out_types) + len(inout_types)
  module = _run_serde_pass(
      module,
      serialize=True,
      ir_version=FWD_COMPAT_IR_VERSION if ctx.is_forward_compat() else None,
  )
  module_asm = module.operation.get_asm(binary=True, enable_debug_info=True)
  kernel_id = hashlib.sha256(module_asm).digest()
  # Note that this is technically only a half measure. Someone might load a
  # compiled module with a hash collision from disk. But that's so unlikely with
  # SHA256 that it shouldn't be a problem.
  if (kernel_text := KNOWN_KERNELS.get(kernel_id, None)) is not None:
    if kernel_text != module_asm:
      raise RuntimeError("Hash collision!")
  else:
    KNOWN_KERNELS[kernel_id] = module_asm

  if ctx.is_forward_compat():
    if use_custom_barrier:
      raise ValueError("Barrier semaphore is not supported in forward compatibility mode. "
                       "Please, use 'export_ignore_forward_compatibility=True'.")
    op = mlir.custom_call(
        "mosaic_gpu",
        result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
        operands=args,
        operand_layouts=[list(reversed(range(a.ndim))) for a in ctx.avals_in],
        result_layouts=[list(reversed(range(a.ndim))) for a in ctx.avals_out],
        backend_config=kernel_id + module_asm,
        operand_output_aliases=dict(input_output_aliases),
    )
  else:
    op = mlir.custom_call(
        "mosaic_gpu_v2",
        result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
        operands=args,
        operand_layouts=[list(reversed(range(a.ndim))) for a in ctx.avals_in],
        result_layouts=[list(reversed(range(a.ndim))) for a in ctx.avals_out],
        backend_config=dict(
            kernel_hash=ir.StringAttr.get(kernel_id),
            module=ir.StringAttr.get(module_asm),
            use_custom_barrier=ir.BoolAttr.get(use_custom_barrier),
        ),
        operand_output_aliases=dict(input_output_aliases),
        api_version=4,
    )
  return op.results


mlir.register_lowering(mosaic_gpu_p, _mosaic_gpu_lowering_rule, "cuda")


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

  def __post_init__(self):
    if self.arrival_count < 1:
      raise ValueError(
          f"Arrival count must be at least 1, but got {self.arrival_count}"
      )

@dataclasses.dataclass(frozen=True)
class ClusterBarrier:
  collective_dims: Sequence[gpu.Dimension]
  num_barriers: int = 1

@dataclasses.dataclass(frozen=True)
class TMEM:
  shape: tuple[int, int]
  dtype: Any
  _: dataclasses.KW_ONLY
  layout: tcgen05.TMEMLayout | None = None
  collective: bool = False
  packing: int | None = None

  def __post_init__(self):
    if self.layout is not None:
      self.layout.check_type(self.shape, utils.dtype_to_ir_type(self.dtype))
      if self.packing is not None:
        raise ValueError("Cannot specify both layout and packing")


def _count_buffer_bytes(shape_dtype: jax.ShapeDtypeStruct) -> int:
  return math.prod(shape_dtype.shape) * np.dtype(shape_dtype.dtype).itemsize


class LoweringSemantics(enum.Enum):
  """Semantics for the kernel's instruction stream."""

  Lane = enum.auto()
  Warpgroup = enum.auto()


@dataclasses.dataclass(frozen=True)
class _TMEMAlloc:
  addr_ref: ir.Value
  num_cols: int
  collective: bool

  def alloc(self):
    tcgen05.tmem_alloc(
        self.addr_ref, self.num_cols, collective=self.collective, exact=False
    )

  def dealloc(self):
    addr = memref.load(self.addr_ref, [])
    tcgen05.tmem_dealloc(
        addr, self.num_cols, collective=self.collective, exact=False
    )


def _slice_smem(
    result: ir.Type,
    smem_base: ir.Value,
    offset: ir.Value,  # This should be an ir.IndexType.
    lowering_semantics: LoweringSemantics,
) -> ir.Value:
  if lowering_semantics == LoweringSemantics.Warpgroup:
    offset = arith.index_cast(ir.IntegerType.get_signless(32), offset)
    return dialect.slice_smem(result, offset)
  else:
    return memref.view(result, smem_base, offset, [])


def _construct_smem_reftree(
    cluster_shape: tuple[int, int, int],
    dynamic_smem: ir.Value,
    smem_buffers: ShapeTree,
    tmem_allocs: list[_TMEMAlloc],  # Mutated by this function!
    lowering_semantics: LoweringSemantics,
    dynamic_smem_offset: int = 0,
) -> Callable[[], RefTree]:
  index = ir.IndexType.get()
  i8 = ir.IntegerType.get_signless(8)
  i32 = ir.IntegerType.get_signless(32)
  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
  flat_ref_tys, smem_buffer_tree = jax.tree.flatten(
      smem_buffers, is_leaf=lambda x: isinstance(x, Union)
  )
  smem_refs = []

  for ref_ty in flat_ref_tys:
    def get_barrier_ptr(num_barriers: int) -> ir.Value:
      nonlocal dynamic_smem_offset
      workgroup_nvptx_address_space = (
          utils.gpu_address_space_to_nvptx(gpu.AddressSpace.Workgroup)
      )
      smem_base_ptr = utils.memref_ptr(
          dynamic_smem, memory_space=workgroup_nvptx_address_space
      )
      smem_ptr_ty = ir.Type.parse(f"!llvm.ptr<{workgroup_nvptx_address_space}>")
      barrier_base_ptr = llvm.getelementptr(
          smem_ptr_ty, smem_base_ptr, [], [dynamic_smem_offset], i8,
          llvm.GEPNoWrapFlags.none
      )
      dynamic_smem_offset += num_barriers * utils.MBARRIER_BYTES
      return barrier_base_ptr
    match ref_ty:
      case Union(members):
        member_thunks = [
            _construct_smem_reftree(
                cluster_shape,
                dynamic_smem,
                m,
                tmem_allocs,
                lowering_semantics,
                dynamic_smem_offset,
            )
            for m in members
        ]
        # TODO(apaszke): This is quadratic, but it shouldn't matter for now...
        dynamic_smem_offset += _smem_tree_size(ref_ty)

        def ref(member_thunks=member_thunks):
          return Union([t() for t in member_thunks])

      case TMABarrier(num_barriers):
        init_fn = utils.DialectBarrierRef.initialize if (
            lowering_semantics == LoweringSemantics.Warpgroup
        ) else utils.BarrierRef.initialize
        ref = init_fn(
            get_barrier_ptr(num_barriers), num_barriers, arrival_count=1
        )
      case Barrier(arrival_count, num_barriers):
        init_fn = utils.DialectBarrierRef.initialize if (
            lowering_semantics == LoweringSemantics.Warpgroup
        ) else utils.BarrierRef.initialize
        ref = init_fn(
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
      case TMEM(shape, dtype, layout=layout, collective=collective, packing=packing):
        addr_ref = _slice_smem(
            ir.MemRefType.get([], i32, memory_space=smem),
            dynamic_smem,
            c(dynamic_smem_offset, index),
            lowering_semantics,
        )
        if layout is None:
          layout = tcgen05._infer_tmem_layout(
              shape, 1 if packing is None else packing
          )
        num_cols = layout.cols_in_shape(shape)
        tmem_allocs.append(_TMEMAlloc(addr_ref, num_cols, collective))
        def ref(addr_ref=addr_ref, shape=shape, dtype=dtype, layout=layout):
          addr = memref.load(addr_ref, [])
          return tcgen05.TMEMRef(
              addr, shape, utils.dtype_to_ir_type(dtype), layout
          )
        dynamic_smem_offset += 4  # i32 takes up 4 bytes
      case _:
        mlir_dtype = utils.dtype_to_ir_type(ref_ty.dtype)
        tile_smem = _slice_smem(
            ir.MemRefType.get(ref_ty.shape, mlir_dtype, memory_space=smem),
            dynamic_smem,
            c(dynamic_smem_offset, index),
            lowering_semantics,
        )
        dynamic_smem_offset += _count_buffer_bytes(ref_ty)
        ref = tile_smem
    smem_refs.append(ref)
  def ref_tree_thunk():
    refs = []
    for ref in smem_refs:
      if callable(ref):
        ref = ref()
      refs.append(ref)
    return jax.tree.unflatten(smem_buffer_tree, refs)
  return ref_tree_thunk


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
        if size % utils.MBARRIER_BYTES:
          raise NotImplementedError("Misaligned barrier allocation")
        size += num_barriers * utils.MBARRIER_BYTES
      case TMEM(_):
        # TODO(justinfu): This can trigger misaligned barrier allocations
        # if TMEM is requested before barriers b/c it's not divisible by 8.
        size += 4  # i32 takes up 4 bytes
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
    smem_buffers: ShapeTree | Union[ShapeTree],
    lowering_semantics: LoweringSemantics,
    module: ir.Module,
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
    # Profiler array stores values in 64 bit chunks (vectors of size 2
    # of 32-bit elements), and so the starting address needs to be 64
    # bit = 8 byte aligned.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands:~:text=The%20address%20must%20be%20naturally%20aligned%20to%20a%20multiple%20of%20the%20access%20size.
    align = 8
    profiler_start = (smem_bytes + align - 1) & ~(align - 1)
    smem_bytes = profiler_start + profiler_spec.smem_bytes(block=block)

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
        ir.MemRefType.get((utils.DYNAMIC,), i8, memory_space=smem)
    )

    if profiler_spec:
      prof_smem = _slice_smem(
          ir.MemRefType.get(
              (profiler_spec.smem_i32_elements(block=block),),
              i32,
              memory_space=smem,
          ),
          dynamic_smem,
          c(profiler_start, index),
          lowering_semantics,
      )
      prof = profiler.OnDeviceProfiler(
          profiler_spec, prof_smem, maybe_prof_buffer
      )
    else:
      prof = None

    ctx = launch_context.LaunchContext(
        module, launch_context.Scratch(launch_op), cluster, prof
    )
    with ctx.named_region("Init"):
      tmem_allocs: list[_TMEMAlloc] = []
      smem_ref_tree_thunk = _construct_smem_reftree(
          cluster, dynamic_smem, smem_buffers, tmem_allocs, lowering_semantics
      )
      # TODO(apaszke): Skip fences if no barriers or TMEM is initialized.
      # TODO(apaszke): Only initialize cluster barriers before the cluster wait.
      nvvm.fence_mbarrier_init()
      if math.prod(cluster) != 1:
        nvvm.cluster_arrive_relaxed(aligned=ir.UnitAttr.get())
        nvvm.cluster_wait(aligned=ir.UnitAttr.get())
      if tmem_allocs:
        eq = arith.CmpIPredicate.eq
        is_init_warp = arith.cmpi(eq, utils.warp_idx(sync=False), c(0, i32))
        with utils.when(is_init_warp):
          for alloc in tmem_allocs:
            alloc.alloc()
          if any(alloc.collective for alloc in tmem_allocs):
            tcgen05.tmem_relinquish_alloc_permit(collective=True)
          if any(not alloc.collective for alloc in tmem_allocs):
            tcgen05.tmem_relinquish_alloc_permit(collective=False)
      gpu.barrier()  # Make sure the init is visible to all threads.
      smem_ref_tree = smem_ref_tree_thunk()

    yield ctx, smem_ref_tree

    if tmem_allocs:
      gpu.barrier()  # Make sure everyone is done before we release TMEM.
      if any(alloc.collective for alloc in tmem_allocs):
        nvvm.cluster_arrive_relaxed(aligned=ir.UnitAttr.get())
        nvvm.cluster_wait(aligned=ir.UnitAttr.get())
      with utils.when(is_init_warp):
        for alloc in tmem_allocs:
          alloc.dealloc()
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
    inout_shape,
    smem_scratch_shape: ShapeTree | Union[ShapeTree],
    lowering_semantics: LoweringSemantics,
    module_name: str,
    kernel_name: str | None = None,
    prof_spec: profiler.ProfilerSpec | None = None,
):
  ptr_ty = ir.Type.parse("!llvm.ptr")
  token_ty = ir.Type.parse("!gpu.async.token")
  i32 = ir.IntegerType.get_signless(32)

  def _shape_to_ref_ty(shape: jax.ShapeDtypeStruct) -> ir.MemRefType:
    return ir.MemRefType.get(shape.shape, utils.dtype_to_ir_type(shape.dtype))

  in_ref_tys = [_shape_to_ref_ty(t) for t in in_shapes]
  inout_ref_tys = [_shape_to_ref_ty(t) for t in inout_shape]

  unwrap_output_tuple = False
  if isinstance(out_shape, list):
    out_shape = tuple(out_shape)
  elif not isinstance(out_shape, tuple):
    out_shape = (out_shape,)
    unwrap_output_tuple = not inout_shape
  out_ref_tys = [_shape_to_ref_ty(t) for t in out_shape]
  if prof_spec is not None:
    out_shape = (*out_shape, prof_spec.jax_buffer_type(grid, block))
    out_ref_tys.append(prof_spec.mlir_buffer_type(grid, block))

  module = ir.Module.create()
  dialect.register_dialect(module.context)
  attrs = module.operation.attributes
  attrs["sym_name"] = ir.StringAttr.get(module_name)
  if kernel_name is None:
    kernel_name = getattr(body, "__name__", "anonymous")

  # These are needed as nonlocal below.
  launch_ctx = None
  with ir.InsertionPoint(module.body):
    _declare_runtime_functions()
    global_scratch = llvm.GlobalOp(
        ir.Type.parse("!llvm.array<0 x i8>"),  # We don't know the shape yet.
        "global_scratch",
        ir.Attribute.parse("#llvm.linkage<external>"),
        addr_space=ir.IntegerAttr.get(i32, 4),  # GPU constant memory.
    )
    @func.FuncOp.from_py_func(ptr_ty, ptr_ty, name=f"mosaic_gpu_{kernel_name}")
    def main(token_ptr, buffers):
      nonlocal launch_ctx
      token = builtin.unrealized_conversion_cast([token_ty], [token_ptr])
      arg_refs = []
      # XLA will pass in inout refs again as outputs, but we ignore them.
      for i, ref_ty in enumerate([*in_ref_tys, *inout_ref_tys, *out_ref_tys]):
        ptr = llvm.LoadOp(ptr_ty, llvm.GEPOp(ptr_ty, buffers, [], [i], ptr_ty, llvm.GEPNoWrapFlags.none))
        arg_refs.append(utils.ptr_as_memref(ptr, ir.MemRefType(ref_ty)))
      prof_buffer = arg_refs.pop() if prof_spec is not None else None
      with _launch(
          token, grid, cluster, block, smem_scratch_shape,
          lowering_semantics, module, prof_spec, prof_buffer
      ) as (_launch_ctx, smem_refs):
        nonlocal launch_ctx
        launch_ctx = _launch_ctx
        body(launch_ctx, *arg_refs, smem_refs)
    main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
  sym_tab = ir.SymbolTable(module.operation)
  sym_tab.insert(main.func_op)
  sym_tab.insert(global_scratch)
  module.operation.verify()

  return module, out_shape, unwrap_output_tuple, launch_ctx


def _run_serde_pass(
    module: ir.Module, *, serialize: bool, ir_version: int | None = None
) -> ir.Module:
  module = ir.Module.parse(
      module.operation.get_asm(binary=True, enable_debug_info=True),
      context=module.context,
  )
  pipeline = passmanager.PassManager.parse(
      "builtin.module(mosaic_gpu-serde{serialize="
      + str(serialize).lower()
      + (f" target-version={ir_version}" if ir_version is not None else "")
      + "})",
      module.context,
  )
  allow_unregistered_dialects = module.context.allow_unregistered_dialects
  module.context.allow_unregistered_dialects = True
  try:
    pipeline.run(module.operation)
    module.operation.verify()
  finally:
    module.context.allow_unregistered_dialects = allow_unregistered_dialects
  return module


def _declare_runtime_functions():
  """Declares the runtime functions that can be used by the generated code."""
  ptr_ty = ir.Type.parse("!llvm.ptr")
  i64 = ir.IntegerType.get_signless(64)
  arg_tys = [ptr_ty, ptr_ty, i64, i64, ptr_ty, ptr_ty, i64, ptr_ty]
  init_tma_desc_type = ir.FunctionType.get(arg_tys, [])
  func.FuncOp(
      "mosaic_gpu_init_tma_desc", init_tma_desc_type, visibility="private"
  )


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
    ir_version: int | None = None,
    thread_semantics: LoweringSemantics = LoweringSemantics.Lane,
    inout_shape = (),
):
  if isinstance(in_shape, list):
    in_shape = tuple(in_shape)
  elif not isinstance(in_shape, tuple):
    in_shape = (in_shape,)
  if isinstance(inout_shape, list):
    inout_shape = tuple(inout_shape)
  elif not isinstance(inout_shape, tuple):
    inout_shape = (inout_shape,)

  module, out_shape, unwrap_output_tuple, launch_ctx = (
      _lower_as_gpu_kernel(
          body, grid, cluster, block, in_shape, out_shape, inout_shape,
          smem_scratch_shape, thread_semantics, module_name, kernel_name,
          prof_spec
      )
  )

  if thread_semantics == LoweringSemantics.Warpgroup and dialect is not None:
    # We need to run a pass that removes dead-code for which layout inference
    # does not work.
    pm = mlir.passmanager.PassManager.parse("builtin.module(canonicalize)", module.context)
    pm.run(module.operation)

    # Run Python lowering passes. The remaining passes will be run in C++ in
    # jax/jaxlib/mosaic/gpu/custom_call.cc
    layout_inference.infer_layout(module)  # pytype: disable=attribute-error
    transform_inference.infer_transforms(module)  # pytype: disable=attribute-error
    dialect_lowering.lower_mgpu_dialect(module, launch_ctx)  # pytype: disable=attribute-error

  launch_ctx.scratch.finalize_size()
  module.operation.verify()

  if launch_ctx.is_device_collective and not supports_cross_device_collectives():
    raise RuntimeError("Kernel is a cross-device collective but no support is available.")

  expected_arg_tys, expected_arg_treedef = jax.tree.flatten((*in_shape, *inout_shape))
  def _check_args(*args):
    arg_treedef = jax.tree.structure(args)
    if arg_treedef != expected_arg_treedef:
      raise ValueError(
          f"Invalid argument structure: expected {expected_arg_treedef}, got"
          f" {arg_treedef}, ({args=})"
      )
    for arg, expected_ty in zip(args, expected_arg_tys):
      if arg.shape != expected_ty.shape:
        raise ValueError(
            f"Argument shape mismatch: expected {expected_ty.shape}, got"
            f" {arg.shape}"
        )
      if arg.dtype != expected_ty.dtype:
        hint = ""
        if not arg.shape:
          hint = f". Hint: cast the scalar to {expected_ty.dtype} explicitly."
        raise ValueError(
            f"Argument dtype mismatch: expected {expected_ty.dtype}, got"
            f" {arg.dtype}{hint}"
        )

  def bind(*args) -> Any:
    return mosaic_gpu_p.bind(*args, module=module, out_types=out_shape, inout_types=inout_shape)

  if prof_spec is not None:
    @jax.jit
    def prof_kernel(*args):
      _check_args(*args)
      *results, prof_buffer = bind(*args)
      def dump_profile(prof_buffer):
        out_file = os.path.join(
            os.getenv("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"),
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
    lowering_semantics: LoweringSemantics = LoweringSemantics.Lane,
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

  module, out_shape, unwrap_output_tuple, launch_ctx = (
      _lower_as_gpu_kernel(
          body, grid, cluster, block, in_shape, out_shape, smem_scratch_shape,
          lowering_semantics, module_name, kernel_name, prof_spec
      )
  )

  if lowering_semantics == LoweringSemantics.Warpgroup and dialect is not None:
    # We need to run a pass that removes dead-code for which layout inference
    # does not work.
    pm = mlir.passmanager.PassManager.parse("builtin.module(canonicalize)", module.context)
    pm.run(module.operation)

    # Run Python lowering passes. The remaining passes will be run in C++ in
    # jax/jaxlib/mosaic/gpu/custom_call.cc
    layout_inference.infer_layout(module)  # pytype: disable=attribute-error
    transform_inference.infer_transforms(module)  # pytype: disable=attribute-error
    dialect_lowering.lower_mgpu_dialect(module, launch_ctx)  # pytype: disable=attribute-error

  launch_ctx.scratch.finalize_size()
  module.operation.verify()

  if launch_ctx.is_device_collective:
    raise RuntimeError("Kernel is a cross-device collective but no support is available.")

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
