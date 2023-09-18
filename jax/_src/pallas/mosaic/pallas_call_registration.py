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

"""Contains registrations for pallas_call on TPU."""

from __future__ import annotations

from typing import Any

import jax
from jax import core as jax_core
from jax.experimental import mosaic
from jax.experimental.mosaic.dialects import tpu
from jax.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.pallas import core
from jax._src.pallas.mosaic import lowering
from jax._src.pallas.pallas_call import pallas_call_p


def pallas_call_tpu_lowering_rule(
    ctx: mlir.LoweringRuleContext, *in_nodes,
    jaxpr: jax_core.Jaxpr,
    name: str,
    which_linear: tuple[bool, ...],
    grid_mapping: core.GridMapping,
    input_output_aliases: tuple[tuple[int, int], ...],
    in_shapes: tuple[jax.ShapeDtypeStruct, ...],
    out_shapes: tuple[jax.ShapeDtypeStruct, ...],
    debug: bool,
    interpret: bool,
    mosaic_params: dict[str, Any] | None = None,
    **compiler_params: Any):
  """Lowers a pallas_call to a Mosaic TPU custom call."""
  if input_output_aliases:
    raise NotImplementedError(
        "`input_output_aliases` not supported on TPU backend.")
  if interpret:
    return mlir.lower_fun(pallas_call_p.impl, multiple_results=True)(
        ctx, *in_nodes, jaxpr=jaxpr, name=name, out_shapes=out_shapes,
        in_shapes=in_shapes,
        which_linear=which_linear,
        interpret=interpret, debug=debug,
        input_output_aliases=input_output_aliases,
        grid_mapping=grid_mapping, **compiler_params)
  if debug:
    print(jaxpr)
  with ir.Context() as mlir_ctx, ir.Location.unknown(mlir_ctx):
    tpu.register_dialect(mlir_ctx)
    if mosaic_params is None:
      mosaic_params = {}
    dimension_semantics = mosaic_params.get("dimension_semantics", None)
    kernel_regeneration_metadata = mosaic_params.get(
        "kernel_regeneration_metadata"
    )
    mosaic_module = lowering.lower_jaxpr_to_module(
        mlir_ctx, grid_mapping, jaxpr, dimension_semantics=dimension_semantics)
    if debug:
      print(mosaic_module)
  out_avals = [jax_core.ShapedArray(s.shape, s.dtype) for s in out_shapes]
  return mlir.lower_fun(
      mosaic.as_tpu_kernel(
          mosaic_module,
          out_avals,
          backend=ctx.module_context.backend,
          kernel_name=name,
          kernel_regeneration_metadata=kernel_regeneration_metadata,
          cost_estimate=mosaic_params.get('cost_estimate', None),
      ),
      multiple_results=True,
  )(ctx, *in_nodes)
mlir.register_lowering(pallas_call_p, pallas_call_tpu_lowering_rule,
                       platform="tpu")
