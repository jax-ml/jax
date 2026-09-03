# Copyright 2026 The JAX Authors.
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

"""Contains MGPU layout rules."""

import dataclasses

from jax._src import core as jax_core
from jax._src import source_info_util
from jax._src.interpreters import mlir
from jax._src.lib import mosaic_gpu_dialect as dialect
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jax.experimental.mosaic.gpu import layout_inference
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import vector


def _from_fa_layout(layout: fa.FragmentedLayout):
  if not isinstance(layout, fa.TiledLayout):
    raise NotImplementedError(
        f"Unsupported layout type: {type(layout)}"
    )
  return GPUTiledLayout(
      layout.tiling, layout.warp_dims, layout.lane_dims, layout.vector_dim
  )


@dataclasses.dataclass(frozen=True)
class GPUTiledLayout:
  tiling: fa.Tiling
  warp_dims: tuple[int | fa.Replicated, ...]  # major-to-minor
  lane_dims: tuple[int | fa.Replicated, ...]  # major-to-minor
  vector_dim: int

  @staticmethod
  def for_array(x: jax_core.Array, core_layout: gpu_core.Layout, *args, **kwargs):
    layout = core_layout.to_mgpu(*args, **kwargs)
    if not isinstance(layout, fa.TiledLayout):
      raise NotImplementedError(
          f"Unsupported layout type: {type(layout)}"
      )
    tile_ndim = len(layout.base_tile_shape)
    if tile_ndim == 0:
      raise ValueError("0-dimensional tiles are not supported.")
    if x.ndim < tile_ndim:
      raise ValueError(
          f"Array of dimension {x.ndim} can not be tiled with a "
          f"{tile_ndim}-dimensional tile"
      )
    for s, t in zip(x.shape[-tile_ndim:], layout.base_tile_shape, strict=True):
      if s % t != 0:
        raise ValueError(
            f"Array shape {x.shape} is not divisible by tile shape"
            f" {layout.base_tile_shape}"
        )
    return _from_fa_layout(layout)


def aval_to_ir_type(aval):
  if isinstance(aval, jax_core.ShapedArray):
    return ir.VectorType.get(aval.shape, mgpu_utils.dtype_to_ir_type(aval.dtype))
  raise NotImplementedError(f"Unsupported aval: {aval}")


def assume_layout(arg, layout: GPUTiledLayout):
  fa_layout = fa.TiledLayout(
      layout.tiling, layout.warp_dims, layout.lane_dims, layout.vector_dim
  )
  return dialect.layout_cast(arg, mgpu.layouts.to_layout_attr(fa_layout))


def mgpu_layout_rule(prim, in_avals, out_avals, **kwargs):
  if not all(isinstance(a.layout, GPUTiledLayout) for a in in_avals):
    raise TypeError(
        f"Input layouts passed to {prim.name} must be of type `GPUTiledLayout`.")

  with mlir.make_ir_context() as ir_ctx, ir.UnknownLoc.get():
    dialect.register_dialect(ir_ctx)
    ir_ctx.allow_unregistered_dialects = True

    optimization_barrier = None

    def body(launch_ctx, *_):
      module_ctx = lowering.ModuleContext(
          name="",
          axis_names=lowering._AxisNames(()),
          program_ids=None,
          approx_math=False,
          single_wg_lane_predicate=mgpu.single_thread_predicate(
              scope=mgpu.ThreadSubset.WARPGROUP
          ),
          single_warp_lane_predicate=mgpu.single_thread_predicate(
              scope=mgpu.ThreadSubset.WARP
          ),
          smem_requested_bytes=0,
          smem_used_bytes=0,
          tmem_requested_cols=0,
          tmem_used_cols=0,
          tmem_base=None,
          scoped_gmem_used_semaphores=dict(),
          scoped_gmem_semaphore_base_ptr=dict(),
          runtime_barriers={},
          squashed_dims=(),
          lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
          primitive_semantics=gpu_core.PrimitiveSemantics.Warpgroup,
          mesh_info=None,
          name_stack=source_info_util.NameStack(),
          traceback_caches=mlir.TracebackCaches(),
          auto_barriers=False,
          reduction_scratch_bytes=0,
          outer_traceback=None,
      )
      lowering_ctx = lowering.LoweringRuleContext(
          module_ctx=module_ctx,
          launch_ctx=launch_ctx,
          prim=prim,
          avals_in=in_avals,
          avals_out=out_avals,
          out_layout_hint=None,
      )

      arg_types = [aval_to_ir_type(aval) for aval in in_avals]
      mlir_args = [
          vector.broadcast(arg_type, mgpu_utils.c(0, arg_type.element_type))
          for arg_type in arg_types
      ]

      mlir_args = [assume_layout(arg, aval.layout)
                  for arg, aval in zip(mlir_args, in_avals)]
      outs = lowering.mosaic_lowering_rules[
          (module_ctx.lowering_semantics, module_ctx.primitive_semantics)
      ][prim](lowering_ctx, *mlir_args, **kwargs)
      if not prim.multiple_results:
        outs = (outs,)
      nonlocal optimization_barrier
      optimization_barrier = dialect.OptimizationBarrierOp(outs)

    module, _, _, _ = mgpu_core._lower_as_gpu_kernel(
        body,
        grid=(1, 1, 1),
        cluster=(1, 1, 1),
        block=(1, 1, 1),
        in_shapes=(),
        out_shape=(),
        inout_shape=(),
        smem_scratch_shape=(),
        lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        module_name="mgpu_layout_rule",
        kernel_name="mgpu_layout_rule",
        prof_spec=None,
        jax_mesh=None,
        base_loc=None,
        uses_pdl=False,
    )

    pm = mlir.passmanager.PassManager.parse("builtin.module(canonicalize,cse)", module.context)
    pm.run(module.operation)
    try:
      mgpu.infer_layout(module)
    except ValueError as ex:
      if "Failed to infer a possible set of layouts" in str(ex):
        layouts_str = ", ".join(f"{aval.layout}" for aval in in_avals)
        raise ValueError(
            f"Failed to infer layouts. The input layouts were {layouts_str}"
        )
      else:
        raise
    layout_inference.check_for_expensive_relayout(module)
    assert optimization_barrier is not None
    layouts = optimization_barrier.attributes["in_layouts"]
    out = tuple(_from_fa_layout(mgpu.layouts.from_layout_attr(l)) for l in layouts)
    return out if prim.multiple_results else out[0]
