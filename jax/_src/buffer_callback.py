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

from collections.abc import Callable
import functools
from typing import Any

import numpy as np

from jax._src import core
from jax._src import dispatch
from jax._src import ffi
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.lib import ffi as ffi_lib

Buffer = ffi_lib.Buffer
ExecutionStage = ffi_lib.ExecutionStage
ExecutionContext = ffi_lib.ExecutionContext


def buffer_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
):
  flat_shape_dtypes, out_tree = tree_util.tree_flatten(result_shape_dtypes)
  flat_result_avals = tuple(
      core.ShapedArray(x.shape, x.dtype) for x in flat_shape_dtypes
  )

  def wrapped_callback(*args, **kwargs):
    flat_args, in_tree = tree_util.tree_flatten(args)
    out_flat = buffer_callback_p.bind(
        *flat_args,
        callback=functools.partial(callback, **kwargs),
        result_avals=flat_result_avals,
        in_tree=in_tree,
        out_tree=out_tree,
    )
    return tree_util.tree_unflatten(out_tree, out_flat)

  return wrapped_callback


buffer_callback_p = core.Primitive("buffer_callback")
buffer_callback_p.multiple_results = True
# dispatch.prim_requires_devices_during_lowering.add(buffer_callback_p)
dispatch.simple_impl(buffer_callback_p)


@buffer_callback_p.def_abstract_eval
def _buffer_callback_abstract_eval(
    *args, result_avals: tuple[core.ShapedArray, ...], **_
):
  del args
  return result_avals


def _buffer_callback_lowering(
    ctx: mlir.LoweringRuleContext,
    *args: Any,
    callback,
    result_avals: tuple[core.ShapedArray, ...],
    in_tree: Any,
    out_tree: Any,
):
  del result_avals

  if len(ctx.module_context.platforms) > 1:
    raise NotImplementedError("multi-platform lowering for buffer_callback")
  platform = ctx.module_context.platforms[0]
  if platform not in {"cpu", "cuda", "rocm"}:
    raise ValueError(f"`buffer_callback` not supported on {platform} backend.")

  def wrapped_callback(exec_ctx: ExecutionContext, *args: Buffer):
    args_in, args_out = util.split_list(args, [in_tree.num_leaves])
    args_in = tree_util.tree_unflatten(in_tree, args_in)
    args_out = tree_util.tree_unflatten(out_tree, args_out)
    if callback(exec_ctx, args_out, *args_in) is not None:
      raise ValueError("buffer_callback callback must not return any values.")
    return ()

  backend = ctx.module_context.get_backend()
  ifrt_callback = backend.get_emit_python_callback(wrapped_callback)
  ctx.module_context.add_host_callback(ifrt_callback)
  index = np.uint64(len(ctx.module_context.host_callbacks) - 1)
  return ffi.ffi_lowering("xla_python_buffer_callback")(ctx, *args, index=index)


mlir.register_lowering(buffer_callback_p, _buffer_callback_lowering)
