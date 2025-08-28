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
"""Contains SparseCore-specific Pallas abstractions."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools

from jax._src import core as jax_core
from jax._src import tree_util
from jax._src.pallas import core as pallas_core
from jax._src.pallas import helpers as pallas_helpers
from jax._src.pallas import pallas_call
from jax._src.pallas import primitives as pallas_primitives
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.state import discharge as state_discharge
from jax.extend import backend as jex_backend


@dataclasses.dataclass
class BlockSpec(pallas_core.BlockSpec):
  """A BlockSpec for SparseCore.

  Attributes:
    indexed_by: The optional index of a parameter to use as the indexer. If set,
      the pipeline emitter will issue and indirect stream indexing into the
      value of this parameter as part of the pipeline.
    indexed_dim: The dimension to index into. Optional unless ``indexed_by`` is
      set.

  See also:
    :class:`jax.experimental.pallas.BlockSpec`
  """

  # TODO(slebedev): Can we infer these from the ``index_map``?
  indexed_by: int | None = None
  indexed_dim: int | None = None

  def __post_init__(self):
    if (self.indexed_by is None) != (self.indexed_dim is None):
      raise ValueError(
          "indexed_by and indexed_dim must both be set or both unset"
      )

  def to_block_mapping(
      self,
      origin: pallas_core.OriginStr,
      array_aval: jax_core.ShapedArray,
      *,
      index_map_avals: Sequence[jax_core.AbstractValue],
      index_map_tree: tree_util.PyTreeDef,
      grid: pallas_core.GridMappingGrid,
      vmapped_dims: tuple[int, ...],
      debug: bool = False,
  ) -> BlockMapping:
    bm = super().to_block_mapping(
        origin,
        array_aval,
        index_map_avals=index_map_avals,
        index_map_tree=index_map_tree,
        grid=grid,
        vmapped_dims=vmapped_dims,
        debug=debug,
    )
    return BlockMapping(
        **{f.name: getattr(bm, f.name) for f in dataclasses.fields(bm)},
        indexed_by=self.indexed_by,
        indexed_dim=self.indexed_dim,
    )


@dataclasses.dataclass(frozen=True)
class BlockMapping(pallas_core.BlockMapping):
  indexed_by: int | None = None
  indexed_dim: int | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScalarSubcoreMesh:
  axis_name: str
  num_cores: int

  @property
  def backend(self) -> str:
    return "mosaic_tpu"

  @property
  def shape(self):
    return dict(core=self.num_cores)

  def discharges_effect(self, effect):
    del effect  # Unused.
    return False


def _num_available_cores():
  """Returns the number of SparseCores on the current TPU chip."""
  device = jex_backend.get_default_device()
  match device.device_kind:
    case "TPU v5" | "TPU v5p" | "TPU v6" | "TPU7x":
      return 4
    case "TPU v6 lite":
      return 2
    case _:
      raise NotImplementedError(
          f"Unsupported device kind: {device.device_kind}"
      )


def _scalar_subcore_mesh_discharge_rule(
    in_avals,
    out_avals,
    *args,
    mesh,
    jaxpr,
    compiler_params,
    interpret,
    debug,
    cost_estimate,
    name,
    metadata,
):
  if not isinstance(mesh, ScalarSubcoreMesh):
    raise TypeError(f"Mesh must be a ScalarSubcoreMesh, got {type(mesh)}")
  assert len(mesh.shape) == 1
  if mesh.num_cores > (num_expected := _num_available_cores()):
    raise ValueError(
        f"Mesh has {mesh.num_cores} cores, but the current TPU chip has only"
        f" {num_expected} SparseCores"
    )
  if compiler_params is None:
    compiler_params = tpu_core.CompilerParams()
  if compiler_params.dimension_semantics is not None:
    raise ValueError("ScalarSubcoreMesh does not support dimension_semantics=")
  return pallas_core.default_mesh_discharge_rule(
      in_avals,
      out_avals,
      *args,
      mesh=mesh,
      jaxpr=jaxpr,
      compiler_params=dataclasses.replace(
          compiler_params,
          dimension_semantics=["core_parallel"],
          kernel_type=tpu_core.KernelType.SC_SCALAR_SUBCORE,
      ),
      interpret=interpret,
      debug=debug,
      cost_estimate=cost_estimate,
      name=name,
      memory_space=tpu_core.MemorySpace.HBM,
      metadata=metadata,
  )


pallas_core._core_map_mesh_rules[ScalarSubcoreMesh] = (
    _scalar_subcore_mesh_discharge_rule
)


def vector_subcore_kernel(**kwargs):
  # We currently ignore kernel_type= provided by the user, because
  # the default kernel_type= is not None.
  # TODO(slebedev): Set the default kernel_type= to None and update this.
  compiler_params = kwargs.pop("compiler_params", tpu_core.CompilerParams())
  compiler_params = dataclasses.replace(
      compiler_params, kernel_type=tpu_core.KernelType.SC_VECTOR_SUBCORE
  )
  return functools.partial(
      pallas_call.pallas_call, compiler_params=compiler_params, **kwargs
  )


def scalar_subcore_kernel(
    out_shape: object,
    *,
    mesh: pallas_core.Mesh,
    scratch_shapes: pallas_core.ScratchShapeTree = (),
    **kwargs: object,
):
  if unwrap_out := not isinstance(out_shape, (tuple, list)):
    out_shape = (out_shape,)

  def decorator(body):
    def wrapper(*args):
      def stateful(operand_and_out_refs):
        arg_refs, out_refs = operand_and_out_refs

        def cmap_body():
          return pallas_primitives.run_scoped(
              lambda *scratch_refs: body(*arg_refs, *out_refs, *scratch_refs),
              *scratch_shapes,
          )

        pallas_core.core_map(mesh, **kwargs)(cmap_body)

      _, outs = state_discharge.run_state(stateful)(
          (args, pallas_helpers.empty_like(out_shape, backend="mosaic_tpu"))
      )
      return outs[0] if unwrap_out else outs

    return wrapper

  return decorator
