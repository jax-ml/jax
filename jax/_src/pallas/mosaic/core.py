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

"""Contains TPU-specific Pallas abstractions."""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import contextlib
import dataclasses
import enum
import math
from typing import Any, Literal

import jax
from jax._src import core as jax_core
from jax._src import deprecations
from jax._src import state
from jax._src import util
from jax._src.frozen_dict import FrozenDict
from jax._src.pallas import core as pallas_core
from jax._src.tpu_custom_call import OptLevel
import jax.numpy as jnp


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

no_block_spec = pallas_core.no_block_spec
_out_shape_to_aval_mapping = pallas_core._out_shape_to_aval_mapping


class CoreType(enum.Enum):
  TC = "tc"
  SC_SCALAR_SUBCORE = "sc_scalar_subcore"
  SC_VECTOR_SUBCORE = "sc_vector_subcore"

  def __str__(self) -> str:
    return self.value

  def __repr__(self) -> str:
    return self.name


class GridDimensionSemantics(enum.Enum):
  PARALLEL = "parallel"
  CORE_PARALLEL = "core_parallel"
  SUBCORE_PARALLEL = "subcore_parallel"
  ARBITRARY = "arbitrary"


PARALLEL = GridDimensionSemantics.PARALLEL
CORE_PARALLEL = GridDimensionSemantics.CORE_PARALLEL
SUBCORE_PARALLEL = GridDimensionSemantics.SUBCORE_PARALLEL
ARBITRARY = GridDimensionSemantics.ARBITRARY


LiteralDimensionSemantics = Literal[
    "parallel", "core_parallel", "subcore_parallel", "arbitrary"
]
DimensionSemantics = LiteralDimensionSemantics | GridDimensionSemantics


class SideEffectType(enum.Enum):
  # No side effects, can be deduplicated / removed if unused.
  PURE = "pure"
  # Cannot be deduplicated, but can be removed if unused.
  DATAFLOW_SIDE_EFFECTING = "dataflow_side_effecting"
  # Cannot be deduplicated or removed.
  SIDE_EFFECTING = "side_effecting"


@dataclasses.dataclass(frozen=True)
class CompilerParams:
  """Mosaic TPU compiler parameters.

  Attributes:
    dimension_semantics: A list of dimension semantics for each grid dimension
      of the kernel. Either "parallel" for dimensions that can execute in any
      order, or "arbitrary" for dimensions that must be executed sequentially.
    allow_input_fusion: A list of booleans indicating whether input fusion is
      allowed for each argument.
    vmem_limit_bytes: Overrides the default VMEM limit for a kernel. Note that
      this must be used in conjunction with the
      --xla_tpu_scoped_vmem_limit_kib=N flag with N*1kib > vmem_limit_bytes.
    collective_id: Indicates which barrier semaphore to use for the kernel. Note
      that using the same collective_id does not guarantee that the same barrier
      semaphore will be allocated between kernels.
    has_side_effects: Set to True to prevent kernel being CSEd by XLA.
    flags: A dictionary of command line flags for the kernel.
    internal_scratch_in_bytes: The size of the internal scratch space used by
      Mosaic.
    serialization_format: The serialization format for the kernel body.
    disable_bounds_checks: Disable bounds checks in the kernel.
    disable_semaphore_checks: Disable semaphore checks in the kernel.
    skip_device_barrier: Skip the default device barrier for the kernel.
    allow_collective_id_without_custom_barrier: Allow the use of collective_id
      without a custom barrier.
    use_tc_tiling_on_sc: Use TensorCore tiling for SparseCore. This flag is only
      used for ``SC_*_SUBCORE`` kernels and it implicitly defaults to True.
    needs_layout_passes: Whether to use vector layout inference passes. This
      flag is temporary and will eventually be removed.
    fuse_transposed_lhs_in_matmul: Hint to compilers to attempt to fuse
      transposed LHS in MXU if users specify the transposed layout of LHS in
      matmul operations, e.g., `jnp.einsum('km,kn->mn', lhs, rhs)`; on the other
      hand, When transposition is performed separately from multiplication (e.g.
      jnp.matmul(lhs.T, rhs)), this flag does not affect the compiler's decision
      (it might still decide to do it if obviously profitable). Note that this
      flag is at the best-effort basis, and the fusion will only be performed
      when compilers determine it is feasible. Also, the fusion is not always
      profitable and therefore should be used sparingly.
    opt_level: Optimization level. This flag is only used for ``SC_*_SUBCORE``
      kernels and it implicitly defaults to O3.
  """

  dimension_semantics: tuple[DimensionSemantics, ...] | None = None
  allow_input_fusion: tuple[bool, ...] | None = None
  vmem_limit_bytes: int | None = None
  collective_id: int | None = None
  has_side_effects: bool | SideEffectType = False
  flags: dict[str, Any] | None = None
  internal_scratch_in_bytes: int | None = None
  serialization_format: int = 1
  disable_bounds_checks: bool = False
  disable_semaphore_checks: bool = False
  skip_device_barrier: bool = False
  allow_collective_id_without_custom_barrier: bool = False
  shape_invariant_numerics: bool = True
  use_tc_tiling_on_sc: bool | None = None
  needs_layout_passes: bool = True
  fuse_transposed_lhs_in_matmul: bool = False
  opt_level: OptLevel | None = None

  def __init__(
      self,
      dimension_semantics: Sequence[DimensionSemantics] | None = None,
      allow_input_fusion: Sequence[bool] | None = None,
      vmem_limit_bytes: int | None = None,
      collective_id: int | None = None,
      has_side_effects: bool | SideEffectType = False,
      flags: Mapping[str, Any] | None = None,
      internal_scratch_in_bytes: int | None = None,
      serialization_format: int = 1,
      disable_bounds_checks: bool = False,
      disable_semaphore_checks: bool = False,
      skip_device_barrier: bool = False,
      allow_collective_id_without_custom_barrier: bool = False,
      shape_invariant_numerics: bool = True,
      use_tc_tiling_on_sc: bool | None = None,
      needs_layout_passes: bool = True,
      fuse_transposed_lhs_in_matmul: bool = False,
      opt_level: OptLevel | None = None,
  ):
    object.__setattr__(
        self,
        "dimension_semantics",
        None if dimension_semantics is None else tuple(dimension_semantics),
    )
    object.__setattr__(
        self,
        "allow_input_fusion",
        None if allow_input_fusion is None else tuple(allow_input_fusion),
    )
    object.__setattr__(self, "vmem_limit_bytes", vmem_limit_bytes)
    object.__setattr__(self, "collective_id", collective_id)
    object.__setattr__(self, "has_side_effects", has_side_effects)
    object.__setattr__(
        self, "flags", None if flags is None else FrozenDict(flags)
    )
    object.__setattr__(
        self, "internal_scratch_in_bytes", internal_scratch_in_bytes
    )
    object.__setattr__(self, "serialization_format", serialization_format)
    object.__setattr__(self, "disable_bounds_checks", disable_bounds_checks)
    object.__setattr__(
        self, "disable_semaphore_checks", disable_semaphore_checks
    )
    object.__setattr__(self, "skip_device_barrier", skip_device_barrier)
    object.__setattr__(
        self,
        "allow_collective_id_without_custom_barrier",
        allow_collective_id_without_custom_barrier,
    )
    object.__setattr__(
        self, "shape_invariant_numerics", shape_invariant_numerics
    )
    object.__setattr__(self, "use_tc_tiling_on_sc", use_tc_tiling_on_sc)
    object.__setattr__(self, "needs_layout_passes", needs_layout_passes)
    object.__setattr__(
        self,
        "fuse_transposed_lhs_in_matmul",
        fuse_transposed_lhs_in_matmul,
    )
    object.__setattr__(self, "opt_level", opt_level)

  # Replace is a method, not a field.
  replace = dataclasses.replace


def check_accumulator_ref(shape: tuple[int, ...], dtype: jax.typing.DTypeLike, mxu_id: int):
  from jax._src.pallas.mosaic import tpu_info  # pyrefly: ignore[missing-module-attribute]
  if len(shape) < 2:
    raise ValueError(f"Acc ref must be at least 2D, got shape {shape}")

  if dtype not in (jnp.float32, jnp.int32):
    raise ValueError(
        f"Acc ref dtype must be float32 or int32, got {dtype}")

  info = tpu_info.get_tpu_info()
  if not info.num_accumulators:
    raise ValueError(
        f"Accumulators are not available on TPU {info.chip_version}"
    )

  if mxu_id < 0 or mxu_id >= info.num_mxus:
    raise ValueError(f"mxu_id must be in [0, {info.num_mxus}), got {mxu_id=}")

  m, n = math.prod(shape[:-1]), shape[-1]
  if n != info.mxu_column_size:
    raise ValueError(
        f"The minor dimension size of an accumulator ref must be "
        f"{info.mxu_column_size} but got {n}"
    )
  if m <= 0 or m % info.num_sublanes != 0:
    raise ValueError(
        f"The product of the major dimensions must be a multiple of "
        f"{info.num_sublanes}, but got {m}"
    )


class MemoryRef(pallas_core.MemoryRef):

  def __matmul__(self, other, /):
    if not isinstance(other, pallas_core.Mesh):
      return NotImplemented
    return dataclasses.replace(self, memory_space=self.memory_space @ other)


class MemorySpace(enum.Enum):
  VMEM = "vmem"
  VMEM_SHARED = "vmem_shared"
  SMEM = "smem"
  SREG = "sreg"
  CMEM = "cmem"
  SEMAPHORE = "semaphore_mem"
  HBM = "hbm"

  @property
  def memory_kind(self) -> str:
    return "device"

  def __getattr__(self, name):
    if name == "HOST":
      # Deprecated on June 4, 2026.
      deprecations.warn(
          "pltpu-memory-space-host",
          "pltpu.MemorySpace.HOST is deprecated. Use pl.HOST instead.",
          stacklevel=2,
      )
      return jax_core.MemorySpace.Host
    super().__getattr__(name)  # pyrefly: ignore[missing-attribute]

  def __str__(self) -> str:
    return self.value

  def __repr__(self) -> str:
    return self.name

  def from_type(self, ty):
    return MemoryRef(ty, memory_space=self)

  def __call__(self, shape: Sequence[int], dtype: jax.typing.DTypeLike):
    # A convenience function for constructing MemoryRef types of ShapedArrays.
    return self.from_type(jax_core.ShapedArray(tuple(shape), dtype))

  def like(self, shape_dtype_like):
    if isinstance(shape_dtype_like, jax_core.AbstractValue):
      return self.from_type(shape_dtype_like)
    return self.from_type(jax.typeof(shape_dtype_like))

  def __matmul__(self, other, /):
    if not isinstance(other, pallas_core.Mesh):
      return NotImplemented
    return pallas_core.CoreMemorySpace(self, other)


@dataclasses.dataclass(frozen=True)
class AccMemorySpace:
  mxu_id: int

  def __call__(self, shape: Sequence[int], dtype: jax.typing.DTypeLike):
    shape = tuple(shape)
    check_accumulator_ref(shape, dtype, self.mxu_id)
    return MemoryRef(
        jax_core.ShapedArray(shape, dtype),
        memory_space=self,
    )

  def __matmul__(self, other, /):
    if not isinstance(other, pallas_core.Mesh):
      return NotImplemented
    return pallas_core.CoreMemorySpace(self, other)

  def __str__(self) -> str:
    return f"ACC(mxu_id={self.mxu_id})"


class dma_semaphore(pallas_core.semaphore_dtype):
  pass


class DMASemaphore(pallas_core.AbstractSemaphoreTy):
  type = dma_semaphore
  name = "dma_sem"


class SemaphoreType(enum.Enum):
  REGULAR = "regular"
  DMA = "dma"
  BARRIER = "barrier"

  @property
  def dtype(self) -> Any:
    if self == SemaphoreType.DMA:
      return DMASemaphore()
    elif self == SemaphoreType.BARRIER:
      return pallas_core.BarrierSemaphore()
    else:
      return pallas_core.Semaphore()

  def __call__(self, shape: tuple[int, ...]):
    return MemoryRef(jax_core.ShapedArray(shape, self.dtype), MemorySpace.SEMAPHORE)

  def __matmul__(self, other, /):
    if not isinstance(other, pallas_core.Mesh):
      return NotImplemented
    return pallas_core.CoreMemorySpace(MemorySpace.SEMAPHORE, other)(
        (), self.dtype
    )

  def get_array_aval(self) -> jax_core.ShapedArray:
    return self(()).get_array_aval()

  def get_ref_aval(self) -> state.AbstractRef:
    return self(()).get_ref_aval()


@dataclasses.dataclass(frozen=True)
class AbstractSemaphore(jax_core.AbstractValue):
  sem_type: SemaphoreType


@dataclasses.dataclass(init=False, kw_only=True, unsafe_hash=True)
class PrefetchScalarGridSpec(pallas_core.GridSpec):
  num_scalar_prefetch: int

  def __init__(
      self,
      num_scalar_prefetch: int,
      grid: pallas_core.Grid = (),
      in_specs: pallas_core.BlockSpecTree = no_block_spec,
      out_specs: pallas_core.BlockSpecTree = no_block_spec,
      scratch_shapes: pallas_core.ScratchShapeTree = (),
  ):
    super().__init__(grid, in_specs, out_specs, scratch_shapes)
    self.num_scalar_prefetch = num_scalar_prefetch
    self.scratch_shapes = tuple(scratch_shapes)

  def _make_scalar_ref_aval(self, aval):
    return state.AbstractRef(
        jax_core.ShapedArray(aval.shape, aval.dtype), MemorySpace.SMEM
    )


def _get_default_num_cores() -> int:
  abstract_device = jax.sharding.get_abstract_mesh().abstract_device
  if abstract_device is None:
    device = jax.devices()[0]
  else:
    device = abstract_device
  return device.num_cores


@dataclasses.dataclass(frozen=True, kw_only=True)
class TensorCoreMesh(pallas_core.Mesh):
  """A mesh of TensorCores."""

  axis_name: str
  num_cores: int = dataclasses.field(
      default_factory=_get_default_num_cores
  )

  @property
  def core_type(self) -> CoreType:
    return CoreType.TC

  @property
  def default_memory_space(self) -> pallas_core.MemorySpace:
    return pallas_core.MemorySpace.ANY

  @property
  def shape(self):
    return collections.OrderedDict({self.axis_name: self.num_cores})

  @property
  def dimension_semantics(self) -> Sequence[DimensionSemantics]:
    return [GridDimensionSemantics.PARALLEL]

  def discharges_effect(self, effect: jax_core.Effect) -> Literal[False]:
    del effect
    return False

  def check_is_compatible_with(self, other_mesh):
    if isinstance(other_mesh, TensorCoreMesh) and self != other_mesh:
      raise ValueError("You can't use two different TensorCoreMeshes.")
    # TODO: Add support for mpmd with SparseCore meshes.
    return super().check_is_compatible_with(other_mesh)

  @property
  def supported_memory_spaces(self) -> Sequence[Any]:
    return [
        MemorySpace.VMEM,
        MemorySpace.SMEM,
        MemorySpace.CMEM,
        MemorySpace.SEMAPHORE,
    ]

  @contextlib.contextmanager
  def tracing_context(self):
    yield


def create_tensorcore_mesh(
    axis_name: str,
    devices: Sequence[jax.Device] | None = None,
    num_cores: int | None = None,
) -> TensorCoreMesh:
  if devices is not None and num_cores is not None:
    raise ValueError("cannot specify both devices and num_cores")
  if num_cores is None:
    if devices is None:
      num_cores = _get_default_num_cores()
    else:
      num_cores = devices[0].num_cores
  return TensorCoreMesh(axis_name=axis_name, num_cores=num_cores)


def _convert_semaphore_type_to_aval(
    out_shape: SemaphoreType,
) -> jax_core.AbstractValue:
  return out_shape.get_array_aval()


pallas_core._out_shape_to_aval_mapping[SemaphoreType] = (
    _convert_semaphore_type_to_aval
)


def memory_space_to_tpu_memory_space(
    memory_space: (
        MemorySpace
        | pallas_core.MemorySpace
        | pallas_core.CoreMemorySpace
        | jax_core.MemorySpace
        | None
    ),
    core_type: CoreType,
) -> (
    MemorySpace
    | pallas_core.MemorySpace
    | pallas_core.CoreMemorySpace
    | jax_core.MemorySpace
):
  match memory_space:
    case None:
      match core_type:
        case CoreType.TC:
          return pallas_core.MemorySpace.ANY
        case CoreType.SC_SCALAR_SUBCORE | CoreType.SC_VECTOR_SUBCORE:
          return MemorySpace.HBM
    case pallas_core.MemorySpace.DEFAULT:
      match core_type:
        case CoreType.TC | CoreType.SC_VECTOR_SUBCORE:
          return MemorySpace.VMEM
        case CoreType.SC_SCALAR_SUBCORE:
          return MemorySpace.SMEM
        case _:
          raise ValueError(f"Unsupported core type: {core_type}")
    case pallas_core.MemorySpace.ANY | jax_core.MemorySpace.Host:
      return memory_space
    case (
        pallas_core.MemorySpace.ERROR
        | pallas_core.MemorySpace.INDEX
        | pallas_core.MemorySpace.KEY
    ):
      return MemorySpace.SMEM
    case pallas_core.CoreMemorySpace():
      return (
          memory_space.memory_space
          if memory_space.mesh.core_type is core_type
          else memory_space
      )
    case acc if isinstance(acc, AccMemorySpace):
      return acc
    case MemorySpace():
      return memory_space
    case _:
      raise ValueError(f"Invalid memory space: {memory_space!r}")
