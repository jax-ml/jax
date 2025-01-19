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
"""Callback utilities."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jax._src import core
from jax._src import deprecations
from jax._src import util
from jax._src.interpreters import batching
from jax._src.lax import lax
from jax._src.lax.control_flow.loops import map as lax_map
from jax._src.typing import DeprecatedArg


def callback_batching_rule(
    prim,
    args,
    dims,
    *,
    vectorized: bool | None | DeprecatedArg,
    vmap_method: str | None,
    result_avals: Sequence[core.ShapedArray],
    **kwargs: Any,
):
  if isinstance(vectorized, DeprecatedArg) and vmap_method is None:
    deprecations.warn(
        "jax-callback-vectorized",
        f"The default behavior of {prim.name} under vmap will soon "
        "change. Currently, the default behavior is to generate a sequential "
        "vmap (i.e. a loop), but in the future the default will be to raise "
        "an error. To keep the current default, set vmap_method='sequential'.",
        stacklevel=6,
    )
    vmap_method = "sequential"

  (axis_size,) = {
      a.shape[d] for a, d in zip(args, dims) if d is not batching.not_mapped
  }
  new_args = [
      arg if dim is batching.not_mapped else batching.moveaxis(arg, dim, 0)
      for arg, dim in zip(args, dims)
  ]
  batched_result_avals = tuple(
      core.unmapped_aval(axis_size, core.no_axis_name, 0, aval)
      for aval in result_avals
  )

  # For FFI calls we must update the layouts. We handle the output layouts
  # here, but the input layout updates depend on the vmap_method parameter.
  if vmap_method != "sequential" and kwargs.get("output_layouts") is not None:
    kwargs["output_layouts"] = tuple(
        None if layout is None else tuple(n + 1 for n in layout) + (0,)
        for layout in kwargs["output_layouts"]
    )

  if vmap_method == "legacy_vectorized":
    # This method is kept to support the behavior that was previously exposed
    # when using `vectorized=True`.
    if kwargs.get("input_layouts") is not None:
      kwargs["input_layouts"] = tuple(
          layout
          if d is batching.not_mapped
          else (None if layout is None else tuple(n + 1 for n in layout) + (0,))
          for layout, d in zip(kwargs["input_layouts"], dims)
      )
    outvals = prim.bind(
        *new_args,
        vectorized=vectorized,
        vmap_method=vmap_method,
        result_avals=batched_result_avals,
        **kwargs,
    )
  elif vmap_method == "expand_dims" or vmap_method == "broadcast_all":
    size = axis_size if vmap_method == "broadcast_all" else 1
    bcast_args = [
        lax.broadcast(x, (size,)) if d is batching.not_mapped else x
        for x, d in zip(new_args, dims)
    ]
    if kwargs.get("input_layouts") is not None:
      kwargs["input_layouts"] = tuple(
          None if layout is None else tuple(n + 1 for n in layout) + (0,)
          for layout in kwargs["input_layouts"]
      )
    outvals = prim.bind(
        *bcast_args,
        vectorized=vectorized,
        vmap_method=vmap_method,
        result_avals=batched_result_avals,
        **kwargs,
    )
  elif vmap_method == "sequential":
    is_batched = [d is not batching.not_mapped for d in dims]
    unbatched_args, batched_args = util.partition_list(is_batched, new_args)

    def _batch_fun(batched_args):
      merged_args = util.merge_lists(is_batched, unbatched_args, batched_args)
      return prim.bind(
          *merged_args,
          result_avals=result_avals,
          vectorized=vectorized,
          vmap_method=vmap_method,
          **kwargs,
      )

    outvals = lax_map(_batch_fun, batched_args)
  else:
    raise NotImplementedError(
        f"vmap is only supported for the {prim.name} primitive when vmap_method"
        " is one of 'sequential', 'expand_dims', 'broadcast_all', or"
        " 'legacy_vectorized'."
    )
  return tuple(outvals), (0,) * len(outvals)

