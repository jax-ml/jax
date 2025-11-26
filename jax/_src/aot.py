# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
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
"""JAX AOT API"""

from collections.abc import Hashable
import functools
import traceback
from typing import Any, Callable, Sequence


from absl import logging
from jax._src import aot_util
from jax._src import api
from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import mesh as mesh_lib
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect


UserKey = Hashable | Callable[..., Hashable]
ComponentKey = aot_util.ComponentKey
get_cache = aot_util.get_cache


def component(
  key: UserKey = None,
) -> Callable[..., Any]:
  def _component(fun: Callable[..., Any]):
    # TODO(dsuo): Need to consider static args, etc if fun is jitted.
    component_key = ComponentKey(key)

    if component_key in aot_util._wrapper_cache.cache_keys():
      return aot_util._wrapper_cache.get(component_key)

    @api.jit
    @util.wraps(fun)
    @traceback_util.api_boundary
    def wrapper(*args, **kwargs):
      args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
      wrapped_fun = aot_util.wrap_init(fun, "component")
      flat_fun, out_tree = api_util.flatten_fun(wrapped_fun, in_tree)
      # TODO(dsuo): We need this in vmap(vmap(f)) because otherwise we create a
      # new flat_fun and may not have called it yet; store will be empty.
      flat_fun = aot_util.cached_flat_fun(flat_fun)
      out_flat = component_p.bind(
        *args_flat,
        fun=api.jit(flat_fun.f_transformed),
        component_key=component_key,
      )
      return tree_util.tree_unflatten(out_tree(), out_flat)

    wrapper.key = component_key
    wrapper.fun = fun
    aot_util._wrapper_cache.put(component_key, wrapper)
    return wrapper

  return _component


def component_impl(*args, fun: Callable[..., Any], **_):
  return fun(*args)


def component_abstract_eval(
  *args,
  fun: Callable[..., Any],
  component_key: ComponentKey,
) -> Sequence[core.AbstractValue] | None:
  # TODO(dsuo): Is this an effectful rule since we read/write to disk?
  entry = aot_util.get_entry(component_key)
  if entry is None:
    # TODO(dsuo): By the time we get to lowering, our trace context has picked
    # up an empty AbstractMesh. Don't know why.
    with mesh_lib.use_abstract_mesh(mesh_lib.AbstractMesh((), (), ())):
      avals_out = tree_util.tree_map(
        lambda x: core.ShapedArray(x.shape, x.dtype), api.eval_shape(fun, *args)
      )
    aot_util.put_entry(component_key, entry := aot_util.CacheEntry(avals_out))
  return entry.abstract_eval


def component_lowering(
  ctx: mlir.LoweringRuleContext,
  *args,
  fun: Callable[..., Any],
  component_key: ComponentKey,
) -> Sequence[ir.Value]:
  module_name = aot_util.get_module_name(ctx.module_context.module)
  entry = aot_util.get_entry(component_key, ctx.module_context.context)
  if entry is None:
    raise ValueError("Should hit abstract_eval already, which would populate.")

  if (module := entry.module) is None:
    entry.module = module = aot_util.lower_component_to_module(
      ctx, fun, module_name, component_key
    )
    aot_util.put_entry(component_key, entry, update=True)

  return aot_util.get_module_results(ctx, module, module_name, *args)


def component_batcher(
  axis_data,
  vals_in,
  dims_in,
  fun: Callable[..., Any],
  component_key: ComponentKey,
):
  # Missing from batching process_call:
  # TODO(dsuo): Ignore ragged.
  # TODO(dsuo): Ignore updating annotations.

  # TODO(dsuo): This doesn't handle nesting.
  batched_component_key = ComponentKey.vmap(component_key)
  entry = aot_util.get_entry(batched_component_key)
  if entry is not None:
    return entry.batcher

  wrapped_fun = aot_util.wrap_init(fun, "vmap(component)")

  # ????(dsuo): I don't understand trace tags.
  batched_fun, dims_out = batching.batch_subtrace(
    wrapped_fun, core.TraceTag(), axis_data, tuple(dims_in)
  )
  # TODO(dsuo): We might need to reset stores because we may be calling a cached
  # wrapped fun.
  batched_fun = aot_util.maybe_reset_stores(batched_fun)

  vals_out = component_p.bind(
    *vals_in,
    fun=batched_fun.f_transformed,
    component_key=batched_component_key,
  )

  batcher_outs = vals_out, dims_out()
  entry = aot_util.get_entry(batched_component_key)
  entry.batcher = batcher_outs
  assert False
  aot_util.put_entry(batched_component_key, entry, update=True)

  return batcher_outs


def component_jvp(arg_values, arg_tangents):
  pass


component_p = core.Primitive("component")
component_p.multiple_results = True
component_p.def_impl(component_impl)
component_p.def_abstract_eval(component_abstract_eval)
mlir.register_lowering(component_p, component_lowering)
batching.fancy_primitive_batchers[component_p] = component_batcher
