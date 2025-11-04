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
from typing import Any, Callable, Sequence


from absl import logging
from jax._src import aot_util
from jax._src import api
from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect


UserKey = Hashable | Callable[..., Hashable]
ComponentKey = aot_util.ComponentKey
get_cache = aot_util.get_cache


def component(
  key: UserKey = None,
  # TODO(dsuo): This is really only useful for if we have return of length 1.
  # I.e., interpret (x,) as x. If this is True, then interpret (x,) as (x,).
  # If we see more than one return value, then of course you have multiple
  # results. Need to think about this a bit more since component_p is similar to
  # a call primitive, which requires multiple_results=True, but want to
  # distinguish between (x,) and x.
  multiple_results: bool = False,
) -> Callable[..., Any]:
  def _component(fun: Callable[..., Any]):
    # TODO(dsuo): Do we have all the information we need at this point to make
    # the component key?
    component_key = ComponentKey(key)

    # TODO(dsuo): Jit your function if it isn't. This is so we can produce the
    # debug_info object we need in order to wrap fun later on in batching, but
    # might be the wrong way of doing things.
    if not isinstance(fun, xc._xla.PjitFunction):
      fun = api.jit(fun)

    @api.jit
    @util.wraps(fun)
    @traceback_util.api_boundary
    def wrapper(*args, **kwargs):
      # TODO(dsuo): Flatten function as in shard_map pmap.
      # TODO(dsuo): Do something about kwargs.
      # TODO(dsuo): Need to consider static args.
      return component_p.bind(
        *args,
        fun=fun,
        component_key=component_key,
        multiple_results=multiple_results,
      )

    wrapper.component_key = component_key
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
  multiple_results: bool,
) -> Sequence[core.AbstractValue] | None:
  entry = aot_util.get_entry(component_key)
  logging.info("component_abstract_eval got entry %s", component_key)
  if entry is None:
    traced = aot_util.get_traced(component_key, fun, *args)
    avals_out = tree_util.tree_map(
      lambda x: core.ShapedArray(x.shape, x.dtype), traced.out_info
    )
    aot_util.put_entry(component_key, entry := aot_util.CacheEntry(avals_out))
  return entry.avals_out


def component_lowering(
  ctx,
  *args,
  fun: Callable[..., Any],
  component_key: ComponentKey,
  multiple_results: bool,
) -> Sequence[ir.Value]:
  with ctx.module_context.context as ir_ctx:
    entry = aot_util.get_entry(component_key, ir_ctx)
  logging.info("component_lowering got entry %s", component_key)
  if entry is None:
    raise ValueError("Should hit abstract_eval already, which would populate.")

  module_name = f"{component_key}.module"
  if (module := entry.module) is None:
    logging.info("missed lowering: %s", fun)
    traced = aot_util.get_traced(component_key, fun, *ctx.avals_in)
    lowering_result = mlir.lower_jaxpr_to_module(
      module_name=module_name,
      jaxpr=traced.jaxpr,
      num_const_args=traced._num_consts,
      in_avals=ctx.avals_in,
      # TODO(dsuo): What are ordered effects vs effects?
      ordered_effects=traced.jaxpr.effects,
      # TODO(dsuo): Figure out why ctx.platforms=None.
      platforms=["cpu"],
      backend=ctx.module_context.backend,
      axis_context=ctx.module_context.axis_context,
      donated_args=tuple(
        x.donated for x in tree_util.tree_leaves(traced.args_info)
      ),
      lowering_parameters=mlir.LoweringParameters(),
      # TODO(dsuo): Presumably we need to forward the rest of the arguments to
      # lower_jaxpr_to_module?
    )
    # TODO(dsuo): What should we do about the other attributes on
    # LoweringResult?
    # - keepalive: probably not supported.
    # - host_callbacks: probably not supported.
    # - shape_poly_state: talk to necula@
    module = lowering_result.module
    # TODO(dsuo): We have this to ensure the source and destination modules have
    # the same context, but is it necessary? Perhaps yes, since we need to get
    # rid of the submodule context before merging. Could we just create it with
    # the right context?
    entry.module = module = ir.Module.parse(mlir.module_to_bytecode(module))
    aot_util.put_entry(component_key, entry, update=True)

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
  multiple_results: bool,
):
  # Missing from batching process_call:
  # TODO(dsuo): Ignore ragged.
  # TODO(dsuo): Ignore updating annotations.

  ji = fun._jit_info

  # TODO(dsuo): Dummy debug info
  debug_info = lu.DebugInfo(
    "component_batcher", fun.__name__, arg_names=None, result_paths=None
  )
  # ????(dsuo): Should we be wrapping fun?
  f_ = lu.wrap_init(fun, debug_info=debug_info)

  # ????(dsuo): I don't understand trace tags.
  f_, dims_out = batching.batch_subtrace(
    f_, core.TraceTag(), axis_data, tuple(dims_in)
  )

  # TODO(dsuo): Derp how to actually do this.
  # with core.set_current_trace(vals_in[0]._trace.parent_trace):
  # assert False
  component_key.vmap()
  vals_out = component_p.bind(
    *vals_in,
    fun=f_,
    component_key=component_key,
    multiple_results=multiple_results,
  )
  # if not multiple_results and len(vals_out) == 1:
  #   vals_out = vals_out[0]
  #   dims_out = dims_out[0]
  # logging.info("component_batcher vals_out %s, %s", vals_out, dims_out)
  return vals_out, dims_out


def clear_caches():
  aot_util.component_cache.value.clear()
  aot_util._traced_cache.clear()


component_p = core.Primitive("component")
component_p.def_impl(component_impl)
component_p.def_abstract_eval(component_abstract_eval)
# TODO(dsuo): Figure out multiple_results i.e., distinguishing between (1,) and
# 1.
mlir.register_lowering(component_p, component_lowering)
batching.fancy_primitive_batchers[component_p] = component_batcher
