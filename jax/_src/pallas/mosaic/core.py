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
from collections.abc import Sequence
import dataclasses
import enum
import functools
from typing import Any, ClassVar, Literal

import jax
from jax._src import core as jax_core
from jax._src import util
from jax._src.pallas import core as pallas_core
import jax.numpy as jnp
import numpy as np


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

partial = functools.partial
Grid = pallas_core.Grid
TupleGrid = pallas_core.TupleGrid
BlockSpec = pallas_core.BlockSpec
BlockSpecTree = pallas_core.BlockSpecTree
GridMapping = pallas_core.GridMapping
NoBlockSpec = pallas_core.NoBlockSpec
ScratchShapeTree = pallas_core.ScratchShapeTree
AbstractMemoryRef = pallas_core.AbstractMemoryRef
no_block_spec = pallas_core.no_block_spec
_convert_block_spec_to_block_mapping = pallas_core._convert_block_spec_to_block_mapping
_out_shape_to_aval_mapping = pallas_core._out_shape_to_aval_mapping
split_list = util.split_list


class KernelType(enum.Enum):
  TC = 0
  SC_SCALAR_SUBCORE = 1
  SC_VECTOR_SUBCORE = 2


class GridDimensionSemantics(enum.Enum):
  PARALLEL = "parallel"
  ARBITRARY = "arbitrary"

PARALLEL = GridDimensionSemantics.PARALLEL
ARBITRARY = GridDimensionSemantics.ARBITRARY


DimensionSemantics = Literal["parallel", "arbitrary"] | GridDimensionSemantics


@dataclasses.dataclass(frozen=True)
class CompilerParams(pallas_core.CompilerParams):
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
    internal_scratch_in_bytes: The size of the internal scratch space used by
      Mosaic.
    flags: A dictionary of command line flags for the kernel.
    serialization_format: The serialization format for the kernel body.
    disable_bounds_checks: Disable bounds checks in the kernel.
  """
  BACKEND: ClassVar[pallas_core.Backend] = "mosaic_tpu"
  dimension_semantics: Sequence[DimensionSemantics] | None = None
  allow_input_fusion: Sequence[bool] | None = None
  vmem_limit_bytes: int | None = None
  collective_id: int | None = None
  has_side_effects: bool = False
  flags: dict[str, Any] | None = None
  internal_scratch_in_bytes: int | None = None
  serialization_format: int = 1
  kernel_type: KernelType = KernelType.TC
  disable_bounds_checks: bool = False

  # Replace is a method, not a field.
  replace = dataclasses.replace

class MemorySpace(enum.Enum):
  ANY = "any"  # TODO(b/368401328): Remove this and just use pl.ANY.
  VMEM = "vmem"
  SMEM = "smem"
  CMEM = "cmem"
  SEMAPHORE = "semaphore_mem"

  def __str__(self) -> str:
    return self.value

  def __call__(self, shape: tuple[int, ...], dtype: jnp.dtype):
    # A convenience function for constructing MemoryRef types.
    return pallas_core.MemoryRef(shape, dtype, self)

class dma_semaphore(pallas_core.semaphore_dtype): pass

class DMASemaphore(pallas_core.AbstractSemaphoreTy):
  type = dma_semaphore
  name = "dma_sem"

class SemaphoreType(enum.Enum):
  REGULAR = "regular"
  DMA = "dma"
  BARRIER = "barrier"

  def __call__(self, shape: tuple[int, ...]):
    dtype: Any
    if self == SemaphoreType.DMA:
      dtype = DMASemaphore()
    elif self == SemaphoreType.BARRIER:
      dtype = pallas_core.BarrierSemaphore()
    else:
      dtype = pallas_core.Semaphore()
    return pallas_core.MemoryRef(shape, dtype, MemorySpace.SEMAPHORE)

  def get_array_aval(self) -> pallas_core.ShapedArrayWithMemorySpace:
    return self(()).get_array_aval()

  def get_ref_aval(self) -> AbstractMemoryRef:
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
      grid: Grid = (),
      in_specs: BlockSpecTree = no_block_spec,
      out_specs: BlockSpecTree = no_block_spec,
      scratch_shapes: ScratchShapeTree = ()
  ):
    super().__init__(grid, in_specs, out_specs, scratch_shapes)
    self.num_scalar_prefetch = num_scalar_prefetch
    self.scratch_shapes = tuple(scratch_shapes)

  def _make_scalar_ref_aval(self, aval):
    return AbstractMemoryRef(jax_core.ShapedArray(aval.shape, aval.dtype),
                             MemorySpace.SMEM)


@dataclasses.dataclass(frozen=True)
class TensorCore:
  id: int


@dataclasses.dataclass(frozen=True)
class TensorCoreMesh:
  """A mesh of TensorCores."""
  devices: np.ndarray
  axis_names: Sequence[str]

  @property
  def backend(self) -> str:
    return "mosaic_tpu"

  @property
  def shape(self):
    return collections.OrderedDict(zip(self.axis_names, self.devices.shape))

  def discharges_effect(self, effect: jax_core.Effect):
    del effect
    return False


def create_tensorcore_mesh(
    axis_name: str,
    devices: Sequence[jax.Device] | None = None,
    num_cores: int | None = None,
) -> TensorCoreMesh:
  if devices is not None and num_cores is not None:
    raise ValueError('cannot specify both devices and num_cores')
  if num_cores is None:
    if devices is None:
      devices = jax.devices()
    num_cores = devices[0].num_cores
  return TensorCoreMesh(
      np.array([TensorCore(i) for i in range(num_cores)]),
      [axis_name],
  )


def _tensorcore_mesh_discharge_rule(
    in_avals,
    out_avals,
    *args,
    mesh,
    jaxpr,
    compiler_params: Any | None,
    interpret: Any,
    debug: bool,
    cost_estimate: pallas_core.CostEstimate | None,
    name: str,
):
  assert isinstance(mesh, TensorCoreMesh)
  if compiler_params and not isinstance(compiler_params, CompilerParams):
    raise ValueError(
        "compiler_params must be a pltpu.CompilerParams"
    )
  if not compiler_params:
    compiler_params = CompilerParams()
  if len(mesh.shape) > 1:
    raise NotImplementedError("Mesh must be 1D")
  if compiler_params.dimension_semantics is not None:
    raise ValueError(
        "dimension_semantics must be None for TensorCoreMesh"
    )
  return pallas_core.default_mesh_discharge_rule(
      in_avals,
      out_avals,
      *args,
      jaxpr=jaxpr,
      mesh=mesh,
      compiler_params=compiler_params.replace(
          dimension_semantics=(PARALLEL,)
      ),
      debug=debug,
      interpret=interpret,
      cost_estimate=cost_estimate,
      name=name,
  )

pallas_core._core_map_mesh_rules[TensorCoreMesh] = (
    _tensorcore_mesh_discharge_rule
)


def _convert_semaphore_type_to_aval(
    out_shape: SemaphoreType,
) -> jax_core.AbstractValue:
  return out_shape.get_array_aval()


pallas_core._out_shape_to_aval_mapping[SemaphoreType] = (
    _convert_semaphore_type_to_aval
)
