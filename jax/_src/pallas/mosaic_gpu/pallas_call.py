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

"""A testing shim for ``pl.pallas_call``."""

import dataclasses
import functools
from typing import Any

import jax
from jax import lax
from jax._src import util
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import pipeline as pipeline


def pallas_call(
    fn,
    out_shape,
    *,
    grid=(),
    in_specs=(),
    out_specs=(),
    scratch_shapes=(),
    compiler_params=gpu_core.CompilerParams(),
    kernel_fn=gpu_core.kernel,
):
  if isinstance(out_shape, list):
    out_shape = tuple(out_shape)
  if isinstance(grid, int):
    grid = (grid,)

  dimension_semantics = compiler_params.dimension_semantics
  if dimension_semantics is None:
    dimension_semantics = ("parallel",) * len(grid)
  which_parallel = [ds == "parallel" for ds in dimension_semantics]
  sequential_grid, parallel_grid = util.partition_list(which_parallel, grid)

  def _make_pipeline_spec(spec):
    if spec.index_map is None:
      return spec
    parallel_indices = [
        lax.axis_index(f"d{i}") for i, _ in enumerate(parallel_grid)
    ]
    return dataclasses.replace(
        spec,
        index_map=lambda *indices: spec.index_map(
            *util.merge_lists(
                which_parallel,
                indices[: len(sequential_grid)],
                parallel_indices,
            )
        ),
    )

  def _normalize_specs(specs, shapes):
    if not specs:
      return [pallas_core.BlockSpec()] * len(shapes)
    if not isinstance(specs, (list, tuple)):
      specs = [specs] * len(shapes)
    result = []
    for spec, shape in zip(specs, shapes):
      if spec.block_shape is None and spec.index_map is not None:
        spec = dataclasses.replace(spec, block_shape=shape)
      result.append(spec)
    return result

  @jax.jit
  @kernel_fn(
      out_type=out_shape,
      scratch_types=scratch_shapes,
      compiler_params=dataclasses.replace(
          compiler_params, dimension_semantics=None
      ),
      grid=tuple(parallel_grid),
      grid_names=tuple(f"d{i}" for i, _ in enumerate(parallel_grid)),
  )
  def wrapper(*args_gmem):
    gmem_refs, scratch_refs = util.split_list(
        args_gmem, [len(args_gmem) - len(scratch_shapes)]
    )

    num_inputs = len(gmem_refs) - len(jax.tree.leaves(out_shape))
    in_shapes = [ref.shape for ref in gmem_refs[:num_inputs]]
    out_shapes = [leaf.shape for leaf in jax.tree.leaves(out_shape)]

    @functools.partial(
        pipeline.emit_pipeline,
        grid=sequential_grid,
        in_specs=[
            _make_pipeline_spec(s)
            for s in _normalize_specs(in_specs, in_shapes)
        ],
        out_specs=[
            _make_pipeline_spec(s)
            for s in _normalize_specs(out_specs, out_shapes)
        ],
        max_concurrent_steps=compiler_params.max_concurrent_steps,
    )
    def pipelined_fn(indices, *args):
      grid_env = util.merge_lists(
          which_parallel,
          [*map(pallas_core.GridAxis, indices, sequential_grid)],
          parallel_grid_env,
      )
      with pallas_core.grid_env(grid_env):
        fn(*args, *scratch_refs)

    parallel_grid_env = [
        pallas_core.GridAxis(lax.axis_index(f"d{i}"), lax.axis_size(f"d{i}"))
        for i, _ in enumerate(parallel_grid)
    ]
    grid_env: list[Any] = util.merge_lists(
        which_parallel, [None] * len(sequential_grid), parallel_grid_env
    )
    with pallas_core.grid_env(grid_env):
      pipelined_fn(*gmem_refs)

  return wrapper
