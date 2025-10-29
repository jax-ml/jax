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

import functools
from collections.abc import Hashable
from typing import Any, Callable, NamedTuple


from absl import logging
from jax._src import aot_util
from jax._src import api
from jax._src import core
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect

component_p = core.Primitive("component")


ComponentKey = Hashable | Callable[..., Hashable]


def component(component_key: ComponentKey = None) -> Callable[..., Any]:
  def _component(fun: Callable[..., Any]):
    @api.jit
    @util.wraps(fun)
    @traceback_util.api_boundary
    def wrapper(*args):
      logging.info("wrapper: %s", args)
      # TODO(dsuo): Flatten function as in shard_map pmap.
      # TODO(dsuo): Need to consider static args.
      return component_p.bind(*args, fun=fun, component_key=component_key)

    # NOTE(dsuo): Using a component means we'll jit you in this dummy
    # implementation.
    return wrapper

  return _component


def component_impl(*args, fun: Callable[..., Any], **_):
  # TODO(dsuo): Call should not re-trace.
  logging.info("component_impl")
  return fun(*args)


def component_abstract_eval(
  *args, fun: Callable[..., Any], component_key: ComponentKey
):
  logging.info("component_abstract_eval: %s", component_key)
  key = aot_util.make_abstract_eval_key(component_key)

  def abstract_eval():
    logging.info("component_abstract_eval args: %s", args)
    # NOTE(dsuo): The claim is tracing cache will handle caching jaxprs for us.
    # However, we'll need to convert ir.Values in the lowering rule to avals to
    # trace in lowering with args. There are two further downsides:
    # 1. `fun` must have the same id (in addition to same everything else) in
    # order for us to use this cache within the same process.
    # 2. It's not easy to inspect the _infer_params_cached.cache_info() to
    # understand if we've gotten a cache hit or not (more relevant for testing).
    return tree_util.tree_map(
      lambda x: core.ShapedArray(x.shape, x.dtype), api.eval_shape(fun, *args)
    )

  return aot_util.get_cached_or_put(
    key,
    abstract_eval,
    aot_util.serialize_abstract_eval,
    aot_util.deserialize_abstract_eval,
  )


def component_lowering(
  ctx, *args, fun: Callable[..., Any], component_key: ComponentKey
):
  logging.info("component_lowering: %s", component_key)
  key = aot_util.make_lowering_key(component_key)

  # TODO(dsuo): Is this something we can grab from LoweringRuleContext or
  # traced?
  module_name = f"{component_key}.module"
  traced_key = aot_util.make_abstract_eval_key(component_key)
  # TODO(dsuo): Expect entry exists. TBD for transformations.
  logging.info("component_lowering avals_in: %s", ctx.avals_in)

  def lower_jaxpr_to_module():
    # with ctx.module_context.module.context:
    traced = api.trace(fun, *ctx.avals_in)
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
    submodule = lowering_result.module
    # TODO(dsuo): We have this to ensure the source and destination modules have
    # the same context, but is it necessary? Perhaps yes, since we need to get
    # rid of the submodule context before merging. Could we just create it with
    # the right context?
    submodule = ir.Module.parse(mlir.module_to_bytecode(submodule))
    return submodule

  submodule = aot_util.get_cached_or_put(
    key,
    lower_jaxpr_to_module,
    aot_util.serialize_lowering,
    aot_util.deserialize_lowering,
  )

  symtab = ir.SymbolTable(submodule.operation)
  module = mlir.merge_mlir_modules(
    ctx.module_context.module,
    f"component_{module_name}",
    submodule,
    dst_symtab=ctx.module_context.symbol_table,
  )
  # TODO(dsuo): There's quite a bit of logic from jax.export, but we just strip
  # away most of that for this demo. e.g., ordered effects, platforms.
  # submodule_args = [mlir.aval_to_ir_type(x) for x in ctx.avals_in]
  results = symtab["main"].type.results
  call = func_dialect.CallOp(results, ir.FlatSymbolRefAttr.get(module), args)

  return call.results


def component_batcher(
  vals_in, dims_in, fun: Callable[..., Any], component_key: ComponentKey
):
  return fun(vals_in[0]), dims_in[0]


component_p.def_impl(component_impl)
component_p.def_abstract_eval(component_abstract_eval)
# TODO(dsuo): Figure out multiple_results i.e., distinguishing between (1,) and
# 1.
mlir.register_lowering(component_p, component_lowering)
batching.primitive_batchers[component_p] = component_batcher
