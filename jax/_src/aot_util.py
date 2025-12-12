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
"""JAX AOT API utilities."""

from collections.abc import Hashable
import functools
import pickle
import traceback
from typing import Any, Callable, NamedTuple, Self, Sequence

from absl import logging
from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import linear_util as lu
from jax._src import stages
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect


# For now, we don't worry about serialization.
SerializedType = bytes | Any


class ComponentKey:
  def __init__(self, user_key: Hashable):
    self.user_key = user_key

  def __hash__(self):
    return hash(self.user_key)

  def __eq__(self, other):
    return isinstance(other, ComponentKey) and self.user_key == other.user_key

  def __str__(self):
    return self.user_key

  def __repr__(self):
    return self.__str__()

  # TODO(dsuo): This is just a hack for now.
  @classmethod
  def vmap(cls, key):
    return ComponentKey(f"vmap({key.user_key})")


def _validate_component_cache(val):
  logging.info("Validating component cache config.")
  assert val is None or isinstance(val, Cache)
  if val is not None:
    util.register_cache(val, "aot_cache")


component_cache = config.string_or_object_state(
  name="jax_component_cache",
  default=None,
  help="Cache dir for components. Components won't be cached if None.",
  validator=_validate_component_cache,
)


class CacheEntry:
  def __init__(
    self,
    abstract_eval: Sequence[core.AbstractValue] | None,
    # TODO(dsuo): What are the vals_out of batcher, DynamicJaxprTrace?
    module: ir.Module | None = None,
    batcher: tuple[Any, tuple[int]] = None,
  ):
    self.abstract_eval = abstract_eval
    self.module = module
    self.batcher = batcher

  def serialize(self) -> SerializedType:
    module_bytecode = None
    if self.module is not None:
      module_bytecode = mlir.module_to_bytecode(self.module)
    return pickle.dumps((self.abstract_eval, module_bytecode, self.batcher))

  @classmethod
  def deserialize(
    cls, blob: SerializedType, ctx: ir.Context | None = None
  ) -> Self:
    abstract_eval, module_bytecode, batcher = pickle.loads(blob)
    if module_bytecode is None or ctx is None:
      module = None
    else:
      with ctx:
        module = ir.Module.parse(module_bytecode)
    return cls(abstract_eval, module, batcher)


# TODO(dsuo): This should be a protocol.
class Cache:
  def __init__(self):
    self._in_memory_cache: dict[ComponentKey, SerializedType] = {}
    self._in_memory_cache_info: dict[ComponentKey, dict[str, Any]] = {}

  def get(self, key: ComponentKey) -> SerializedType | None:
    entry = self._in_memory_cache.get(key, None)
    if entry is not None:
      self._in_memory_cache_info[key] = dict(
        hits=self._in_memory_cache_info[key]["hits"] + 1
      )
    return entry

  def put(self, key: ComponentKey, data: SerializedType, update: bool):
    self._in_memory_cache[key] = data
    if not update:
      self._in_memory_cache_info[key] = dict(hits=0)

  def cache_keys(
    self,
  ) -> list[ComponentKey]:
    return list(self._in_memory_cache.keys())

  def cache_clear(self) -> None:
    self._in_memory_cache.clear()
    self._in_memory_cache_info.clear()

  def cache_info(self, key: ComponentKey) -> dict[str, Any]:
    if key not in self._in_memory_cache_info:
      raise ValueError(f"`{key}` not found in self._in_memory_cache_info")
    return self._in_memory_cache_info[key]


def get_cache() -> Cache | None:
  return component_cache.value


def get_entry(
  key: ComponentKey, ctx: ir.Context | None = None
) -> CacheEntry | None:
  if (cache := get_cache()) is not None:
    if (blob := cache.get(key)) is not None:
      return CacheEntry.deserialize(blob, ctx)
  return None  # sigh pytype


def put_entry(
  key: ComponentKey, entry: CacheEntry, update: bool = False
) -> None:
  if (cache := get_cache()) is not None:
    cache.put(key, entry.serialize(), update)


@lu.cache
def cached_flat_fun(flat_fun):
  return maybe_reset_stores(flat_fun)


# TODO(dsuo): Share logic with pmap.
def maybe_reset_stores(fun):
  # TODO(dsuo): Hack to clear lu.Store borrowed from pmap.
  f_transformed = fun.f_transformed

  # TODO(dsuo): Add this as a transformation.
  def reset_stores_f_transformed(*args, **kwargs):
    for store in fun.stores:
      if store is not None:
        store.reset()
    return f_transformed(*args, **kwargs)

  fun.f_transformed = reset_stores_f_transformed
  return fun


class WrapperCache:
  def __init__(self):
    self.data = dict()
    self.info = dict()

  def get(self, key: ComponentKey) -> xc._xla.PjitFunction | None:
    fun = self.data.get(key, None)
    if fun is not None:
      self.info[key]["hits"] += 1
    return fun

  def put(self, key: ComponentKey, fun: xc._xla.PjitFunction):
    fun = self.data.setdefault(key, fun)
    info = self.info.setdefault(key, dict(hits=0))
    self.info[key]["hits"] = 0

  def cache_info(self):
    return self.info

  def cache_clear(self):
    self.data.clear()

  def cache_keys(self):
    return self.data.keys()


_wrapper_cache = WrapperCache()
util.register_cache(_wrapper_cache, "aot_wrapper_cache")


def lower_component_to_module(
  ctx: mlir.LoweringRuleContext,
  fun: Callable[..., Any],
  module_name: str,
  component_key: ComponentKey,
) -> ir.Module:
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
  module = lowering_result.module
  # TODO(dsuo): Do we need to do this step to strip context?
  module = ir.Module.parse(mlir.module_to_bytecode(module))

  return module


def get_module_name(module: ir.Module) -> str:
  # TODO(dsuo): Is this reasonable?
  module_name = str(module.operation.attributes["sym_name"].value)
  logging.info("module_name: %s", module_name)
  return module_name


def get_module_results(
  ctx: mlir.LoweringRuleContext, module: ir.Module, module_name: str, *args
) -> Sequence[ir.Value]:
  symtab = ir.SymbolTable(module.operation)
  module = mlir.merge_mlir_modules(
    ctx.module_context.module,
    f"component_{module_name}",
    module,
    dst_symtab=ctx.module_context.symbol_table,
  )
  results = symtab["main"].type.results
  call = func_dialect.CallOp(results, ir.FlatSymbolRefAttr.get(module), args)

  return call.results


def wrap_init(fun: Callable[..., Any], traced_for: str) -> lu.WrappedFun:
  # TODO(dsuo): Dummy debug info.
  if isinstance(fun, functools.partial):
    name = fun.func.__name__
  else:
    name = fun.__name__
  return lu.wrap_init(
    fun, debug_info=lu.DebugInfo(traced_for, name, None, None)
  )
