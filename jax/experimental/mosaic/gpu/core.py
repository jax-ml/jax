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

from collections.abc import Callable, Iterable, Sequence
import contextlib
import ctypes
import dataclasses
import enum
import functools
import hashlib
import io
import itertools
import math
import os
import pathlib
import time
from typing import Any, Generic, TypeVar
import weakref

import jax
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import lib
from jax._src import mesh as mesh_lib
from jax._src import sharding_impls
from jax._src import util as jax_util
from jax._src.interpreters import mlir
from jax._src.lib import mosaic_gpu_dialect as dialect
from jax.extend import backend as jex_backend
from jaxlib.mlir import ir
from jaxlib.mlir import passmanager
from jaxlib.mlir.dialects import _gpu_ops_gen
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import builtin
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
import numpy as np

from . import dialect_lowering
from . import launch_context
from . import layout_inference
from . import layouts
from . import profiler
from . import tcgen05
from . import utils

# MLIR can't find libdevice unless we point it to the CUDA path
cuda_root = lib.cuda_path or "/usr/local/cuda"
os.environ["CUDA_ROOT"] = cuda_root
PYTHON_RUNFILES = os.environ.get("PYTHON_RUNFILES")

_SMEM_SIZE_BOUND = None  # For test purposes.

@contextlib.contextmanager
def artificial_shared_memory_limit(limit):
    global _SMEM_SIZE_BOUND
    old_limit = _SMEM_SIZE_BOUND
    _SMEM_SIZE_BOUND = limit
    try:
        yield
    finally:
        _SMEM_SIZE_BOUND = old_limit

# This tracks the latest Mosaic GPU IR version with a monthly delay.
FWD_COMPAT_IR_VERSION = 2

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
  from nvidia import nvshmem  # pytype: disable=import-error
except ImportError:
  # Try to find the nvshmem library in Bazel runfiles.
  if PYTHON_RUNFILES:
    libdevice_path = os.path.join(
        PYTHON_RUNFILES, "nvidia_nvshmem", "lib", "libnvshmem_device.bc"
    )
    if os.path.exists(libdevice_path):
      os.environ["MOSAIC_GPU_NVSHMEM_BC_PATH"] = libdevice_path
    for root, _, files in os.walk(os.getcwd()):
      if "/_solib" in root and "libnvshmem_host.so.3" in files:
        os.environ["MOSAIC_GPU_NVSHMEM_SO_PATH"] = os.path.join(
            root, "libnvshmem_host.so.3"
        )
        break
  else:
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


def is_nvshmem_available():
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


def is_single_process_multi_device_topology():
  return (jax.device_count() > 1
          and jax.device_count() == jax.local_device_count())


def supports_cross_device_collectives():
  return ((is_nvshmem_available() and jax.local_device_count() == 1)
          or is_single_process_multi_device_topology())


mosaic_gpu_p = jax_core.Primitive("mosaic_gpu_p")
mosaic_gpu_p.multiple_results = True


@mosaic_gpu_p.def_abstract_eval
def _mosaic_gpu_abstract_eval(*_, module, out_types, inout_types):
  del module # Unused.
  return [
      jax_core.ShapedArray(t.shape, t.dtype)
      for t in itertools.chain(out_types, inout_types)
  ]


def _has_communication(module, **_):
  if launch_context.uses_collective_metadata(module):
    return True
  empty_str_attr = ir.StringAttr.get("")
  for op in module.body:
    if "nvshmem" in getattr(op, "sym_name", empty_str_attr).value:
      return True
  return False


# TODO(apaszke): Implement a proper system for managing kernel lifetimes
# Maps kernel ID to the compiled kernel ASM.
KNOWN_KERNELS: dict[bytes, bytes] = {}


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
  replica_ids = []
  if is_multi_device_module := _has_communication(module):
    # Those checks are trying to ensure that the logical device ids are
    # consistent with the NVSHMEM PE ids that Mosaic will be using for
    # communication. Any divergence here would require us to implement a logical
    # to physical translation, which is currently not implemented.
    if isinstance(axis_context, sharding_impls.SPMDAxisContext):
      mesh = axis_context.mesh
      if isinstance(mesh, mesh_lib.Mesh):
        replica_ids = mesh.device_ids.ravel()
        # Skip the check for AbstractMesh
        if not np.array_equal(mesh.device_ids.ravel(), np.arange(mesh.size)):
          raise NotImplementedError(
              "Mosaic GPU only supports meshes with device ordering that follows"
              f" row-major device ids. Got: {mesh.device_ids.ravel()} device ids."
          )
    elif isinstance(axis_context, sharding_impls.ShardingContext):
      if axis_context.num_devices != 1:
        raise NotImplementedError(
            "Mosaic GPU only supports single-device meshes in ShardingContext."
            f" Got: {axis_context.num_devices} devices."
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
  bytecode_buffer = io.BytesIO()
  module.operation.write_bytecode(bytecode_buffer, desired_version=0)
  module_asm = bytecode_buffer.getvalue()
  kernel_id = hashlib.sha256(module_asm).digest()
  # Note that this is technically only a half measure. Someone might load a
  # compiled module with a hash collision from disk. But that's so unlikely with
  # SHA256 that it shouldn't be a problem.
  if (kernel_text := KNOWN_KERNELS.get(kernel_id, None)) is not None:
    if kernel_text != module_asm:
      raise RuntimeError("Kernel hash collision!")
  else:
    KNOWN_KERNELS[kernel_id] = module_asm

  backend_config = dict(
      kernel_hash=ir.StringAttr.get(kernel_id),
      module=ir.StringAttr.get(module_asm),
      use_custom_barrier=ir.BoolAttr.get(use_custom_barrier),
      uses_xla_collective_metadata=ir.BoolAttr.get(
          launch_context.uses_collective_metadata(module)
      ),
  )
  custom_call_kwargs = {
      "call_target_name": "mosaic_gpu_v2",
      "result_types": [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
      "operands": args,
      "operand_layouts": [list(reversed(range(a.ndim))) for a in ctx.avals_in],
      "result_layouts": [list(reversed(range(a.ndim))) for a in ctx.avals_out],
      "backend_config": backend_config,
      "operand_output_aliases": dict(input_output_aliases),
      "api_version": 4
  }

  if not is_multi_device_module or not is_single_process_multi_device_topology():
    return mlir.custom_call(**custom_call_kwargs).results  # type: ignore

  # Add collective metadata as additional output buffer.
  # Collective metadata stores pointers to both input and output parameters.
  num_params = len(ctx.avals_out) + len(ctx.avals_in)
  num_peers = axis_context.mesh.size
  collective_metadata_size = (
      launch_context.COLLECTIVE_METADATA_SIZE + num_peers * num_params
  )

  custom_call_kwargs["result_layouts"] += [[0]]  # type: ignore
  custom_call_kwargs["result_types"] += [  # type: ignore
      ir.RankedTensorType.get((collective_metadata_size,),
                              ir.IntegerType.get_signless(64))]
  backend_config["xla_replica_ids"] = ir.StringAttr.get(
      ",".join(map(str, replica_ids))
  )
  # Drop the collective metadata buffer from the results.
  return mlir.custom_call(**custom_call_kwargs).results[:-1]  # type: ignore

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
  arrival_count: int = 1
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
      self.layout.check_type(
          self.shape, utils.bitwidth(utils.dtype_to_ir_type(self.dtype))
      )
      if self.packing is not None:
        raise ValueError("Cannot specify both layout and packing")


def _count_buffer_bytes(shape_dtype: jax.ShapeDtypeStruct) -> int:
  return math.prod(shape_dtype.shape) * dtypes.itemsize_bits(dtypes.dtype(shape_dtype.dtype)) // 8


class LoweringSemantics(enum.Enum):
  """Semantics for the kernel's instruction stream."""

  Lane = enum.auto()
  Warpgroup = enum.auto()


@dataclasses.dataclass(frozen=True)
class _TMEMAlloc:
  addr_ref: ir.Value
  num_cols: int
  collective: bool

  def alloc(self) -> int:
    """Allocates TMEM and returns the number of columns allocated."""
    _, cols = tcgen05.tmem_alloc(
        self.addr_ref, self.num_cols, collective=self.collective, exact=False
    )
    return cols

  def dealloc(self):
    addr = memref.load(self.addr_ref, [])
    tcgen05.tmem_dealloc(
        addr, self.num_cols, collective=self.collective, exact=False
    )


@dataclasses.dataclass()
class _TMEMDialectAlloc:
  addr_ref: ir.Value
  shape: tuple[int, int]
  dtype: ir.Type
  packing: int
  collective: bool
  tmem_ref: ir.Value | None = dataclasses.field(init=False, default=None)

  def alloc(self) -> int:
    """Allocates TMEM and returns the number of columns allocated."""
    result_type = ir.MemRefType.get(
        self.shape,
        self.dtype,
        memory_space=utils.tmem(),
    )
    self.tmem_ref = dialect.tmem_alloc(
        result_type,
        self.addr_ref,
        collective=self.collective,
        packing=self.packing,
    )
    ncols = self.shape[1] // self.packing
    return tcgen05.tmem_alloc_exact_ncols(ncols, exact=False)

  def dealloc(self):
    assert self.tmem_ref is not None
    dialect.tmem_dealloc(self.tmem_ref)


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
    tmem_allocs: list[
        _TMEMAlloc | _TMEMDialectAlloc
    ],  # Mutated by this function!
    lowering_semantics: LoweringSemantics,
    dynamic_smem_offset: int = 0,
) -> Callable[[], RefTree]:
  index = ir.IndexType.get()
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  flat_ref_tys, smem_buffer_tree = jax.tree.flatten(
      smem_buffers, is_leaf=lambda x: isinstance(x, Union)
  )
  smem_refs = []

  for ref_ty in flat_ref_tys:
    def barrier_memref(num_barriers: int) -> ir.Value:
      nonlocal dynamic_smem_offset
      barrier_ty = ir.MemRefType.get(
          (num_barriers,),
          ir.Type.parse("!mosaic_gpu.barrier")
          if lowering_semantics == LoweringSemantics.Warpgroup
          else i64,
          memory_space=utils.smem(),
      )
      barrier_memref = _slice_smem(
            barrier_ty,
            dynamic_smem,
            c(dynamic_smem_offset, index),
            lowering_semantics,
        )
      dynamic_smem_offset += num_barriers * utils.MBARRIER_BYTES
      return barrier_memref
    ref: Any
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
        init_fn: Callable[..., Any] = (
            utils.DialectBarrierRef.initialize
            if lowering_semantics == LoweringSemantics.Warpgroup
            else utils.BarrierRef.initialize
        )
        ref = init_fn(barrier_memref(num_barriers), arrival_count=1)
      case Barrier(arrival_count, num_barriers):
        init_fn = (
            utils.DialectBarrierRef.initialize
            if lowering_semantics == LoweringSemantics.Warpgroup
            else utils.BarrierRef.initialize
        )
        ref = init_fn(barrier_memref(num_barriers), arrival_count=arrival_count)
      case ClusterBarrier(collective_dims, arrival_count, num_barriers):
        ref = utils.CollectiveBarrierRef.initialize(
            barrier_memref(num_barriers), arrival_count, collective_dims, cluster_shape
        )
      case TMEM(shape, dtype, layout=layout, collective=collective, packing=packing):
        addr_ref = _slice_smem(
            ir.MemRefType.get([], i32, memory_space=utils.smem()),
            dynamic_smem,
            c(dynamic_smem_offset, index),
            lowering_semantics,
        )
        packing = 1 if packing is None else packing
        ir_dtype = utils.dtype_to_ir_type(dtype)
        if lowering_semantics == LoweringSemantics.Warpgroup:
          if layout is not None:
            packing = layout.vector_length

          alloc = _TMEMDialectAlloc(
              addr_ref, shape, ir_dtype, packing, collective
          )
          tmem_allocs.append(alloc)
          def ref(alloc=alloc, layout=layout):
            assert alloc.tmem_ref is not None
            if layout is not None:
              layout_attr = layouts.to_layout_attr(layout)
              return dialect.tmem_layout_cast(alloc.tmem_ref, layout_attr)
            else:
              return alloc.tmem_ref

        else:
          if layout is None:
            layout = tcgen05._infer_tmem_layout(shape, collective, packing)
          num_cols = layout.cols_in_shape(shape, utils.bitwidth(ir_dtype))
          tmem_allocs.append(_TMEMAlloc(addr_ref, num_cols, collective))
          def ref(addr_ref=addr_ref, shape=shape, ir_dtype=ir_dtype, layout=layout):
            addr = memref.load(addr_ref, [])
            return tcgen05.TMEMRef(addr, shape, ir_dtype, layout)

        dynamic_smem_offset += 4  # i32 takes up 4 bytes
      case _:
        mlir_dtype = utils.dtype_to_ir_type(ref_ty.dtype)
        tile_smem = _slice_smem(
            ir.MemRefType.get(ref_ty.shape, mlir_dtype, memory_space=utils.smem()),
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
          | ClusterBarrier(_, _, num_barriers=num_barriers)
          | Barrier(_, num_barriers=num_barriers)
      ):
        if size % utils.MBARRIER_BYTES:
          raise NotImplementedError(
              "Misaligned barrier allocation. Expected smem size"
              f" ({size} bytes) to be divisible by the size of the barrier:"
              f" {utils.MBARRIER_BYTES} bytes."
          )
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
    device_collective_metadata: ir.Value | None = None,
    host_collective_metadata: ir.Value | None = None,
    num_peers: int = 0,
):
  if (profiler_spec is None) != (maybe_prof_buffer is None):
    raise ValueError(
        "Both profiler_spec and maybe_prof_buffer must be specified or"
        " left unspecified."
    )
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

  device = jax.local_devices()[0]
  # For ahead-of-time compilation purposes, that is when a CUDA device
  # isn't available to query directly, we default to 227 KB, the
  # maximum amount of shared memory per thread block available in
  # compute capabilities 9.0 and 10.x:
  # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
  # Note in either case we assume all devices have the same amount of
  # shared memory.
  max_smem_bytes = getattr(device, "shared_memory_per_block_optin", 227 * 1024)
  if _SMEM_SIZE_BOUND is not None:
    max_smem_bytes = min(max_smem_bytes, _SMEM_SIZE_BOUND)
  if smem_bytes > max_smem_bytes:
    raise ValueError("Mosaic GPU kernel exceeds available shared memory: "
                     f"{smem_bytes=} > {max_smem_bytes=}")
  if math.prod(cluster) != 1:
    if len(cluster) != 3:
      raise ValueError(f"Clusters must be 3D. Got: {cluster}")
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
  # `gpu.LaunchOp` is missing the clusterSize{X,Y,Z} arguments.
  launch_op = _gpu_ops_gen.LaunchOp(
      token.type,
      [token],
      *grid_vals,
      *block_vals,
      dynamicSharedMemorySize=c(smem_bytes, i32),
      **cluster_kwargs,
  )
  launch_op.body.blocks.append(*([index] * (12 + 2 * len(cluster_kwargs))))  # Append an empty block
  with ir.InsertionPoint(launch_op.body.blocks[0]):
    dynamic_smem = gpu.dynamic_shared_memory(
        ir.MemRefType.get((utils.DYNAMIC,), i8, memory_space=utils.smem())
    )

    if profiler_spec:
      prof_smem = _slice_smem(
          ir.MemRefType.get(
              (profiler_spec.smem_i32_elements(block=block),),
              i32,
              memory_space=utils.smem(),
          ),
          dynamic_smem,
          c(profiler_start, index),
          lowering_semantics,
      )
      if lowering_semantics == LoweringSemantics.Warpgroup:
        prof_smem = dialect.with_transforms(prof_smem, ir.ArrayAttr.get([]))
        wrap_in_custom_primitive = True
      else:
        wrap_in_custom_primitive = False
      prof = profiler.OnDeviceProfiler(
          profiler_spec,
          prof_smem,
          maybe_prof_buffer,
          wrap_in_custom_primitive,
      )
    else:
      prof = None

    ctx = launch_context.LaunchContext(
        module,
        launch_context.Scratch(launch_op),
        cluster,
        prof,
        device_collective_metadata=device_collective_metadata,
        host_collective_metadata=host_collective_metadata,
        num_peers=num_peers,
    )
    with ctx.named_region("Init"):
      tmem_allocs: list[_TMEMAlloc | _TMEMDialectAlloc] = []
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
        init_warp_ctx: contextlib.AbstractContextManager
        if lowering_semantics == LoweringSemantics.Warpgroup:
          init_warp_ctx = contextlib.nullcontext()
        else:
          eq = arith.CmpIPredicate.eq
          is_init_warp = arith.cmpi(eq, utils.warp_idx(sync=False), c(0, i32))
          init_warp_ctx = utils.when(is_init_warp)
        with init_warp_ctx:
          cols_used = 0
          for alloc in tmem_allocs:
            cols_used += alloc.alloc()
          if cols_used > tcgen05.TMEM_MAX_COLS:
            raise ValueError(
                "Total TMEM allocation exceeds memory limit. "
                f"Requested {cols_used} columns which exceeds limit of "
                f"{tcgen05.TMEM_MAX_COLS}."
            )
          collective_types = {alloc.collective for alloc in tmem_allocs}
          if len(collective_types) > 1:
            raise ValueError(
                "Can't mix collective and non-collective TMEM allocations"
                " within the same kernel."
            )
          collective = True in collective_types
          if collective and math.prod(cluster) % 2:
            raise ValueError(
                "Collective TMEM allocations are only supported for clusters"
                " with an even number of blocks in them. Got cluster:"
                f" {cluster}"
            )
          if lowering_semantics == LoweringSemantics.Warpgroup:
            dialect.tmem_relinquish_alloc_permit(collective=collective)
          else:
            tcgen05.tmem_relinquish_alloc_permit(collective=collective)
      gpu.barrier()  # Make sure the init is visible to all threads.
      smem_ref_tree = smem_ref_tree_thunk()

    yield ctx, smem_ref_tree

    if tmem_allocs:
      gpu.barrier()  # Make sure everyone is done before we release TMEM.
      if any(alloc.collective for alloc in tmem_allocs):
        nvvm.cluster_arrive_relaxed(aligned=ir.UnitAttr.get())
        nvvm.cluster_wait(aligned=ir.UnitAttr.get())
      if lowering_semantics == LoweringSemantics.Warpgroup:
        init_warp_ctx = contextlib.nullcontext()
      else:
        init_warp_ctx = utils.when(is_init_warp)
      with init_warp_ctx:
        for alloc in tmem_allocs:
          alloc.dealloc()
    if prof is not None:
      prof.finalize(grid=grid, block=block)
    gpu.terminator()


def _infer_arch() -> tuple[int, int]:
  device: Any = jax.sharding.get_abstract_mesh().abstract_device
  if device is None:
    device = jex_backend.get_default_device()
  if not hasattr(device, "compute_capability"):
    return (9, 0)  # TODO(apaszke): Remove this once we figure out the export story.
  arch_name = device.compute_capability
  # Handle ROCm devices that return architecture strings like "gfxXXX".
  if arch_name.startswith("gfx"):
    raise ValueError(
        f"Mosaic GPU does not yet support AMD ROCm devices. "
        f"Got compute_capability: {arch_name}"
    )
  return tuple(map(int, arch_name.split(".")))  # type: ignore


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
    kernel_name: str,
    prof_spec: profiler.ProfilerSpec | None = None,
    jax_mesh: mesh_lib.Mesh | None = None,
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
  arch_major, arch_minor = _infer_arch()
  attrs["mosaic_gpu.arch_major"] = ir.IntegerAttr.get(i32, arch_major)
  attrs["mosaic_gpu.arch_minor"] = ir.IntegerAttr.get(i32, arch_minor)

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
    @func.FuncOp.from_py_func(ptr_ty, ptr_ty, name=f"{kernel_name}_mosaic_gpu")
    def main(token_ptr, buffers):
      nonlocal launch_ctx
      token = builtin.unrealized_conversion_cast([token_ty], [token_ptr])
      arg_refs = []
      # XLA will pass in inout refs again as outputs, but we ignore them.
      for i, ref_ty in enumerate([*in_ref_tys, *inout_ref_tys, *out_ref_tys]):
        ptr = llvm.load(ptr_ty, utils.getelementptr(buffers, [i], ptr_ty))
        arg_memref = utils.ptr_as_memref(ptr, ir.MemRefType(ref_ty))
        # Annotate so we can find the corresponding kernel argument during the
        # lowering.
        arg_memref.owner.attributes[launch_context.KERNEL_ARG_ID_ATTR] = (
            ir.IntegerAttr.get(i32, i)
        )
        arg_memref.owner.attributes[launch_context.ORIGINAL_KERNEL_ARG_ATTR] = (
            ir.UnitAttr.get()
        )
        arg_refs.append(arg_memref)

      collective_metadata = None
      host_collective_metadata = None
      num_peers = 0

      # Collective metadata parameter is used to lower collective operations
      # in a single-process setup.
      if (
          jax_mesh is not None
          and jax_mesh.size > 1
          and is_single_process_multi_device_topology()
      ):
        num_args = len(arg_refs)
        num_peers = jax_mesh.size
        metadata_ptr = llvm.load(
            ptr_ty, utils.getelementptr(buffers, [num_args], ptr_ty)
        )
        metadata_ty = ir.MemRefType.get(
            (launch_context.COLLECTIVE_METADATA_SIZE + num_args * num_peers,),
            ir.IntegerType.get_signless(64)
        )
        collective_metadata = utils.ptr_as_memref(metadata_ptr, metadata_ty)

        host_metadata_ptr = llvm.load(
            ptr_ty, utils.getelementptr(buffers, [num_args + 1], ptr_ty)
        )
        host_collective_metadata = utils.ptr_as_memref(
            host_metadata_ptr, metadata_ty
        )

      prof_buffer = arg_refs.pop() if prof_spec is not None else None

      with _launch(
          token,
          grid,
          cluster,
          block,
          smem_scratch_shape,
          lowering_semantics,
          module,
          prof_spec,
          prof_buffer,
          collective_metadata,
          host_collective_metadata,
          num_peers,
      ) as (_launch_ctx, smem_refs):
        nonlocal launch_ctx
        launch_ctx = _launch_ctx
        body(launch_ctx, *arg_refs, smem_refs)
    main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
  sym_tab = ir.SymbolTable(module.operation)
  sym_tab.insert(main.func_op)
  sym_tab.insert(global_scratch)
  module.operation.verify()

  assert launch_ctx is not None
  return module, out_shape, unwrap_output_tuple, launch_ctx


def _run_serde_pass(
    module: ir.Module, *, serialize: bool, ir_version: int | None = None
) -> ir.Module:
  bytecode_buffer = io.BytesIO()
  module.operation.write_bytecode(bytecode_buffer)
  module = ir.Module.parse(bytecode_buffer.getvalue(), context=module.context)
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


def _kernel_to_module(
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
  if kernel_name is None:
    kernel_name = jax_util.fun_name(body, "anonymous")

  inout_shape = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                             inout_shape)
  out_shape = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                           out_shape)
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
    dialect_lowering.lower_mgpu_dialect(module, launch_ctx)  # pytype: disable=attribute-error

  launch_ctx.scratch.finalize_size()
  module.operation.verify()

  return (
      module,
      in_shape,
      inout_shape,
      out_shape,
      unwrap_output_tuple,
      launch_ctx.is_device_collective,
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
  module, in_shape, inout_shape, out_shape, unwrap_output_tuple, is_device_collective = _kernel_to_module(
      body, grid, block, in_shape, out_shape, smem_scratch_shape, prof_spec,
      cluster, module_name, kernel_name, thread_semantics, inout_shape
  )

  if is_device_collective and not supports_cross_device_collectives():
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
    return mosaic_gpu_p.bind(
        *args,
        module=module,
        out_types=out_shape,
        inout_types=inout_shape,
    )

  if prof_spec is not None:
    @jax.jit
    def prof_kernel(*args):
      _check_args(*args)
      *results, prof_buffer = bind(*args)
      def dump_profile(prof_buffer):
        out_file = os.path.join(prof_spec.dump_path, f"{time.time_ns()}-trace.json")
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
    thread_semantics: LoweringSemantics = LoweringSemantics.Lane,
    inout_shape=(),
):
  (
      module,
      in_shape,
      inout_shape,
      out_shape,
      unwrap_output_tuple,
      is_device_collective,
  ) = _kernel_to_module(
      body,
      grid,
      block,
      in_shape,
      out_shape,
      smem_scratch_shape,
      prof_spec,
      cluster,
      module_name,
      kernel_name,
      thread_semantics,
      inout_shape,
  )
  module = _run_serde_pass(module, serialize=True, ir_version=None)
  bytecode_buffer = io.BytesIO()
  module.operation.write_bytecode(bytecode_buffer)
  return _as_torch_gpu_kernel(
      bytecode_buffer.getvalue(),
      in_shape,
      out_shape,
      inout_shape,
      unwrap_output_tuple=unwrap_output_tuple,
  )


def _compile_as_torch_gpu_kernel(module_asm: bytes):
  try:
    import torch  # type: ignore[import-not-found]  # pytype: disable=import-error
  except ImportError:
    raise RuntimeError("Can't compile for PyTorch: import torch failed") from None

  torch.cuda.init()  # Make sure CUDA context is set up.

  # Get our hands on the compilation and unload functions
  try:
    try:
      import jax_plugins.xla_cuda13 as cuda_plugin  # type: ignore[import-not-found]  # pytype: disable=import-error
    except ImportError:
      import jax_plugins.xla_cuda12 as cuda_plugin  # type: ignore[import-not-found]  # pytype: disable=import-error
  except ImportError:
    dll = ctypes.CDLL(None)
  else:
    dll = ctypes.CDLL(cuda_plugin._get_library_path())
  compile_func = dll.MosaicGpuCompile
  compile_func.argtypes = [ctypes.c_void_p]
  compile_func.restype = ctypes.POINTER(ctypes.c_void_p)
  unload_func = dll.MosaicGpuUnload
  unload_func.argtypes = [compile_func.restype]
  unload_func.restype = None

  compiled = compile_func(ctypes.c_char_p(module_asm), ctypes.c_int(len(module_asm)))
  if not compiled:
    raise RuntimeError("Failed to compile the module")
  ctx, launch_ptr = compiled[0], compiled[1]
  ctx_ptr_ptr = ctypes.pointer(ctypes.c_void_p(ctx))
  launch_c = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(launch_ptr)

  def launch(arg_ptrs, device):
    # Allocate another buffer for args of the host-side program. This is sadly
    # the default MLIR calling convention.
    launch_args_ptr = (ctypes.POINTER(ctypes.c_void_p) * 3)()
    launch_args_ptr[0] = ctx_ptr_ptr
    launch_args_ptr[1] = ctypes.pointer(
        torch.cuda.default_stream(device)._as_parameter_
    )
    launch_args_ptr[2] = ctypes.cast(
        ctypes.pointer(ctypes.pointer(arg_ptrs)),
        ctypes.POINTER(ctypes.c_void_p),
    )
    launch_c(launch_args_ptr)

  return launch, functools.partial(unload_func, compiled)


def _as_torch_gpu_kernel(
    module_asm: bytes,
    in_shape: Iterable[object],
    out_shape: Iterable[object],
    inout_shape: Iterable[object] = (),
    *,
    unwrap_output_tuple: bool = False,
    _prepare_args = None,
    _prepare_results = None,
):
  flat_arg_types, expected_arg_treedef = jax.tree.flatten((*in_shape, *inout_shape))
  flat_out_types, _ = jax.tree.flatten(out_shape)
  out_treedef = jax.tree.structure((*out_shape, *inout_shape))

  launch, unload = _compile_as_torch_gpu_kernel(module_asm)
  # _compile_as_torch_gpu_kernel checks that this succeeds
  import torch  # type: ignore[import-not-found]  # pytype: disable=import-error

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
    for arg, expected_ty in zip(flat_args, flat_arg_types):
      if arg.shape != expected_ty.shape:
        raise ValueError(
            f"Argument shape mismatch: expected {expected_ty.shape}, got"
            f" {arg.shape}"
        )
      if arg.dtype != as_torch_dtype(expected_ty.dtype):
        raise ValueError(
            "Argument dtype mismatch: expected"
            f" {as_torch_dtype(expected_ty.dtype)}, got {arg.dtype}"
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
    if num_inout_args := jax.tree.structure(inout_shape).num_leaves:
      flat_outs += flat_args[-num_inout_args:]
    launch(buffers, device)
    out = jax.tree.unflatten(out_treedef, flat_outs)
    return out[0] if unwrap_output_tuple else out

  # Unload the compiled code when the Python function is destroyed.
  apply.destructor = weakref.ref(apply, lambda _weak_ref: unload)

  return apply
