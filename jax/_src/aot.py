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
    # TODO(dsuo): Do we have all the information we need at this point to make
    # the component key?
    component_key = ComponentKey(key)

    if component_key in aot_util._wrapper_cache.cache_keys():
      logging.info("hit wrapper_cache: %s", component_key)
      return aot_util._wrapper_cache.get(component_key)

    @api.jit
    @util.wraps(fun)
    @traceback_util.api_boundary
    def wrapper(*args, **kwargs):
      args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
      wrapped_fun = lu.wrap_init(
        fun, debug_info=api_util.debug_info("component", fun, args, kwargs)
      )
      flat_fun, out_tree = api_util.flatten_fun(wrapped_fun, in_tree)
      # TODO(dsuo): do we need this cached?
      flat_fun = aot_util.cached_flat_fun(flat_fun)
      logging.info("miss component flat_fun %s:", id(flat_fun))
      flat_fun = flat_fun.f_transformed
      flat_fun.__name__ = "wrapped(flat_fun)"
      jitted_fun = api.jit(flat_fun)
      logging.info("miss component jitted_fun %s:", id(jitted_fun))

      out_flat = component_p.bind(
        *args,
        fun=jitted_fun,
        component_key=component_key,
      )
      return tree_util.tree_unflatten(out_tree(), out_flat)

    wrapper.key = component_key
    wrapper.fun = fun
    logging.info("jit(wrapper(fun)) wrapper id %s", id(wrapper))
    logging.info("wrapper(fun) wrapper._fun id %s", id(wrapper._fun))
    logging.info(
      "fun wrapper._fun.__wrapped__ id %s", id(wrapper._fun.__wrapped__)
    )
    logging.info("user fun id %s", id(fun))
    aot_util._wrapper_cache.put(component_key, wrapper)
    return wrapper

  return _component


def component_impl(*args, fun: Callable[..., Any], **_):
  if isinstance(fun, lu.WrappedFun):
    return fun.call_wrapped(*args)
  return fun(*args)


def component_abstract_eval(
  *args,
  fun: Callable[..., Any],
  component_key: ComponentKey,
) -> Sequence[core.AbstractValue] | None:
  # ????(dsuo): Is this an effectful rule?
  entry = aot_util.get_entry(component_key)
  logging.info("component_abstract_eval got entry %s", component_key)
  if entry is None:
    logging.info("missed abstract_eval %s %s", component_key, type(fun))
    # TODO(dsuo): By the time we get to lowering, our trace context has picked
    # up an empty AbstractMesh. Don't know why.
    if isinstance(fun, functools.partial):
      logging.info("abstract_eval partial %s", fun.func.__name__)
    if isinstance(fun, lu.WrappedFun):
      logging.info("abstract_eval lu.WrappedFun")
      fun = aot_util.maybe_reset_stores(fun).call_wrapped
    with mesh_lib.use_abstract_mesh(mesh_lib.AbstractMesh((), (), ())):
      avals_out = tree_util.tree_map(
        lambda x: core.ShapedArray(x.shape, x.dtype), api.eval_shape(fun, *args)
      )
    aot_util.put_entry(component_key, entry := aot_util.CacheEntry(avals_out))
  else:
    logging.info("hit abstract_eval %s", component_key)
  return entry.avals_out


def component_lowering(
  ctx: mlir.LoweringRuleContext,
  *args,
  fun: Callable[..., Any],
  component_key: ComponentKey,
) -> Sequence[ir.Value]:
  with ctx.module_context.context as ir_ctx:
    entry = aot_util.get_entry(component_key, ir_ctx)
  logging.info("component_lowering got entry %s", component_key)
  if entry is None:
    raise ValueError("Should hit abstract_eval already, which would populate.")

  module_name = f"{component_key}.module"
  if (module := entry.module) is None:
    logging.info("missed lowering: %s", component_key)
    if isinstance(fun, lu.WrappedFun):
      fun = aot_util.maybe_reset_stores(fun).call_wrapped
    module = aot_util.lower_component_to_module(
      ctx, fun, module_name, component_key
    )
    # TODO(dsuo): We have this to ensure the source and destination modules have
    # the same context, but is it necessary? Perhaps yes, since we need to get
    # rid of the submodule context before merging. Could we just create it with
    # the right context?
    entry.module = module = ir.Module.parse(mlir.module_to_bytecode(module))
    aot_util.put_entry(component_key, entry, update=True)
  else:
    logging.info("hit lowering: %s", component_key)

  symtab = ir.SymbolTable(module.operation)
  module = mlir.merge_mlir_modules(
    ctx.module_context.module,
    f"component_{module_name}",
    module,
    dst_symtab=ctx.module_context.symbol_table,
  )
  # TODO(dsuo): There's quite a bit of logic from jax.export, but we just strip
  # away most of that for this demo. e.g., ordered effects, platforms.
  # submodule_args = [mlir.aval_to_ir_type(x) for x in ctx.avals_in]
  results = symtab["main"].type.results
  call = func_dialect.CallOp(results, ir.FlatSymbolRefAttr.get(module), args)

  return call.results


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

  # TODO(dsuo): Dummy debug info.
  if isinstance(fun, functools.partial):
    name = fun.func.__name__
  else:
    name = fun.__name__
  if isinstance(fun, lu.WrappedFun):
    fun = aot_util.maybe_reset_stores(fun)
  wrapped_fun = lu.wrap_init(
    fun, debug_info=lu.DebugInfo("vmap(component)", name, None, None)
  )

  # ????(dsuo): I don't understand trace tags.
  batched_fun, dims_out = batching.batch_subtrace(
    wrapped_fun, core.TraceTag(), axis_data, tuple(dims_in)
  )
  batched_fun = aot_util.maybe_reset_stores(batched_fun)

  vals_out = component_p.bind(
    *vals_in,
    fun=batched_fun.f_transformed,
    component_key=ComponentKey.vmap(component_key),
  )
  return vals_out, dims_out()


component_p = core.Primitive("component")
component_p.multiple_results = True
component_p.def_impl(component_impl)
component_p.def_abstract_eval(component_abstract_eval)
mlir.register_lowering(component_p, component_lowering)
batching.fancy_primitive_batchers[component_p] = component_batcher
