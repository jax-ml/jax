# Copyright 2024 The JAX Authors.
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
"""Colocated Python function API implementation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import inspect
import random
import threading
from typing import Any
import uuid
import weakref

import jax
from jax._src import api
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import pxla
from jax._src.lib import xla_client as xc
from jax._src.traceback_util import api_boundary
from jax._src.util import wraps
from jax.experimental.colocated_python import func_backend
from jax.experimental.colocated_python.serialization import _deserialize, _deserialize_specs, _make_specs_for_serialized_specs, _serialize, _serialize_specs
from jax.extend.backend import register_backend_cache as jax_register_backend_cache
from jax.extend.ifrt_programs import ifrt_programs

ShapeDtypeStructTree = Any  # PyTree[api.ShapeDtypeStruct]


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionInfo:
  """User function wrapped by colocated_python."""

  fun: Callable[..., Any]
  fun_sourceinfo: str | None
  fun_signature: inspect.Signature | None


@dataclasses.dataclass(frozen=True, slots=True)
class Specialization:
  """Specialization for a colocated_python function."""

  in_specs_treedef: tree_util.PyTreeDef | None = None
  in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None
  out_specs_fn: Callable[..., ShapeDtypeStructTree] | None = None
  out_specs_treedef: tree_util.PyTreeDef | None = None
  out_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None
  devices: xc.DeviceList | None = None

  def update(
      self,
      *,
      in_specs_treedef: tree_util.PyTreeDef | None = None,
      in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None,
      out_specs_fn: Callable[..., ShapeDtypeStructTree] | None = None,
      out_specs_treedef: tree_util.PyTreeDef | None = None,
      out_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None,
      devices: Sequence[jax.Device] | xc.DeviceList | None = None,
  ):
    """Creates a new specialization with overrides."""
    if in_specs_treedef is None:
      in_specs_treedef = self.in_specs_treedef
    elif self.in_specs_treedef is not None:
      raise ValueError("in_specs already specified")
    if in_specs_leaves is None:
      in_specs_leaves = self.in_specs_leaves
    elif self.in_specs_leaves is not None:
      raise ValueError("in_specs already specified")

    if out_specs_fn is None:
      out_specs_fn = self.out_specs_fn
    elif self.out_specs_fn is not None:
      raise ValueError("out_specs_fn already specified")

    if out_specs_treedef is None:
      out_specs_treedef = self.out_specs_treedef
    elif self.out_specs_treedef is not None:
      raise ValueError("out_specs already specified")
    if out_specs_leaves is None:
      out_specs_leaves = self.out_specs_leaves
    elif self.out_specs_leaves is not None:
      raise ValueError("out_specs already specified")

    if devices is None:
      devices = self.devices
    elif self.devices is not None:
      raise ValueError("devices already specified")
    elif not isinstance(devices, xc.DeviceList):
      devices = xc.DeviceList(tuple(devices))

    return Specialization(
        in_specs_treedef,
        in_specs_leaves,
        out_specs_fn,
        out_specs_treedef,
        out_specs_leaves,
        devices,
    )


def _get_spec(x: Any) -> api.ShapeDtypeStruct:
  """Extracts a spec for a value, which must be a JAX Array."""
  # TODO(hyeontaek): Allow Python values and automatically apply `shard_arg`
  # with a suitable sharding and layout.
  if not isinstance(x, jax.Array):
    raise ValueError(
        "colocated_python only supports jax.Array as input and output, but got"
        f" {type(x)}."
    )
  return api.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)


def _infer_devices_from_args(args: Sequence[Any]) -> xc.DeviceList | None:
  """Returns a representative device list from function call arguments."""
  device_list_set: set[xc.DeviceList] = set()
  for x in args:
    sharding = getattr(x, "sharding", None)
    if sharding is not None:
      device_list_set.add(x.sharding._internal_device_list)
  if not device_list_set:
    return None
  if len(device_list_set) != 1:
    raise ValueError(
        "All arguments must use the same device list, but got"
        f" multiple device lists: {device_list_set}."
    )
  return device_list_set.pop()


def _compile_to_executable(
    name: str,
    fun: Callable[..., Any],
    in_specs_treedef: tree_util.PyTreeDef,
    in_specs_leaves: tuple[api.ShapeDtypeStruct, ...],
    out_specs_treedef: tree_util.PyTreeDef,
    out_specs_leaves: tuple[api.ShapeDtypeStruct, ...],
    devices: xc.DeviceList,
) -> Callable[..., Any]:
  """Compiles a Python function into a runtime executable."""
  fun_and_specialization = (
      fun,
      in_specs_treedef,
      in_specs_leaves,
      out_specs_treedef,
      out_specs_leaves,
      devices,
  )
  pickled_function = _serialize(fun_and_specialization)
  program = ifrt_programs.make_colocated_python_program(
      name, pickled_function, devices, in_specs_leaves, out_specs_leaves
  )
  ifrt_client = devices[0].client
  out_sdss = tuple(
      jax.core.ShapedArray(sds.shape, sds.dtype) for sds in out_specs_leaves
  )
  out_shardings = tuple(sds.sharding for sds in out_specs_leaves)
  try:
    compile_options = ifrt_programs.make_colocated_python_compile_options()
    loaded_executable = ifrt_client.compile_ifrt_program(
        program, compile_options
    )
    out_handlers = pxla.global_avals_to_results_handler(
        out_sdss, out_shardings, committed=True  # type: ignore
    ).handlers

    def call(*args, **kwargs):
      args_leaves = tree_util.tree_leaves((args, kwargs))
      execute_result = loaded_executable.execute_sharded(
          args_leaves, with_tokens=False
      )
      results = execute_result.consume_with_handlers(out_handlers)
      return tree_util.tree_unflatten(out_specs_treedef, results)

    return call
  except jax.errors.JaxRuntimeError as e:
    # TODO(hyeontaek): Implement colocated Python support in McJAX and remove
    # this fallback path.
    if "PjRtCompiler requires an HloProgram" in str(e):
      return _deserialize(pickled_function)[0]
    raise


def _make_output_specs_and_push_result_fun(
    info: FunctionInfo,
    specialization: Specialization,
    uid: int,
) -> Callable[..., Any]:
  """Creates a function that computes output specs and pushes the result to the result store."""
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.out_specs_treedef is None
  assert specialization.out_specs_leaves is None
  assert specialization.devices is not None

  devices = specialization.devices

  def lowered_fun(*args, **kwargs) -> jax.Array:
    result = info.fun(*args, **kwargs)
    result_leaves, out_treedef = tree_util.tree_flatten(result)
    out_spec_leaves = tuple(_get_spec(x) for x in result_leaves)
    func_backend.SINGLETON_RESULT_STORE.push(uid, result_leaves)
    return _serialize_specs(out_treedef, out_spec_leaves, devices)

  out_specs_leaves, out_specs_treedef = tree_util.tree_flatten(
      _make_specs_for_serialized_specs(specialization.devices),
  )
  name = getattr(info.fun, "__name__", "unknown")
  name = f"{name}_output_specs_and_push_result"
  return _compile_to_executable(
      name=name,
      fun=lowered_fun,
      in_specs_treedef=specialization.in_specs_treedef,
      in_specs_leaves=specialization.in_specs_leaves,
      out_specs_treedef=out_specs_treedef,
      out_specs_leaves=tuple(out_specs_leaves),
      devices=specialization.devices,
  )


def _make_pop_result_fun(
    info: FunctionInfo,
    specialization: Specialization,
    uid: int,
) -> Callable[..., Any]:
  """Makes a function that pops results from the result store."""
  assert specialization.out_specs_treedef is not None
  assert specialization.out_specs_leaves is not None
  assert specialization.devices is not None

  out_specs_treedef = specialization.out_specs_treedef

  def lowered_fun():
    result_leaves = func_backend.SINGLETON_RESULT_STORE.pop(uid)
    return tree_util.tree_unflatten(out_specs_treedef, result_leaves)

  in_specs_leaves, in_specs_treedef = tree_util.tree_flatten((
      # args
      (),
      # kwargs
      {},
  ))
  name = getattr(info.fun, "__name__", "unknown")
  name = f"{name}_pop_result"
  return _compile_to_executable(
      name=name,
      fun=lowered_fun,
      in_specs_treedef=in_specs_treedef,
      in_specs_leaves=tuple(in_specs_leaves),
      out_specs_treedef=specialization.out_specs_treedef,
      out_specs_leaves=specialization.out_specs_leaves,
      devices=specialization.devices,
  )


def _make_async_execution_fun(
    info: FunctionInfo,
    specialization: Specialization,
) -> Callable[..., Any]:
  """Makes a function that asynchronously executes the function."""
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.out_specs_treedef is not None
  assert specialization.out_specs_leaves is not None
  assert specialization.devices is not None

  name = getattr(info.fun, "__name__", "unknown")
  return _compile_to_executable(
      name=name,
      fun=info.fun,
      in_specs_treedef=specialization.in_specs_treedef,
      in_specs_leaves=specialization.in_specs_leaves,
      out_specs_treedef=specialization.out_specs_treedef,
      out_specs_leaves=specialization.out_specs_leaves,
      devices=specialization.devices,
  )


def _uncached_get_specialized_func(
    info: FunctionInfo,
    specialization: Specialization,
) -> Callable[..., Any]:
  """Returns a specialized function for the given specialization."""
  util.test_event("colocated_python_func._get_specialized_func")
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.devices is not None
  uid = random.getrandbits(63)

  mutex = threading.Lock()
  # Asynchronous execution function that has known output_specs.
  async_execution_func = None

  def specialized_func(*args, **kwargs):
    """Specialized function to be executed with given args and kwargs."""
    nonlocal specialization, async_execution_func
    with mutex:
      if async_execution_func is None:
        if specialization.out_specs_treedef is None:
          if specialization.out_specs_fn is None:
            output_specs_and_push_result_fun = (
                _make_output_specs_and_push_result_fun(
                    info, specialization, uid
                )
            )
            serialized_out_specs = output_specs_and_push_result_fun(
                *args, **kwargs
            )

            # Waits for the output_specs. This may block.
            out_specs_treedef, out_specs_leaves = _deserialize_specs(
                serialized_out_specs
            )

            # Subsequent calls would use async_execution_func with discovered
            # output_specs.
            specialization = specialization.update(
                out_specs_treedef=out_specs_treedef,
                out_specs_leaves=out_specs_leaves,
            )
            async_execution_func = _make_async_execution_fun(
                info, specialization
            )

            # Hold the PyExecutable until async_execution_fun is called at
            # least once, so the number of _OBJECT_STORE references at the
            # backend does not drop to 0.
            async_execution_func.output_specs_and_push_result_fun = (
                output_specs_and_push_result_fun
            )

            return _make_pop_result_fun(info, specialization, uid)()
          else:
            # Compute out_specs using out_specs_fn and inputs.
            args_specs, kwargs_specs = tree_util.tree_map(
                _get_spec, (args, kwargs)
            )
            out_specs = specialization.out_specs_fn(*args_specs, **kwargs_specs)
            # Type checking is ignored to silence mypy error: Incompatible types
            # in assignment (expression has type "list[Any]", variable has type
            # "tuple[ShapeDtypeStruct, ...]")  [assignment]
            out_specs_leaves, out_specs_treedef = tree_util.tree_flatten(  # type: ignore[assignment]
                out_specs
            )
            specialization = specialization.update(
                out_specs_treedef=out_specs_treedef,
                out_specs_leaves=tuple(out_specs_leaves),
            )
            async_execution_func = _make_async_execution_fun(
                info, specialization
            )
            # Fall-through.
        else:
          async_execution_func = _make_async_execution_fun(info, specialization)
          # Fall-through.

    # Asynchronous execution runs outside of the mutex to allow concurrent
    # execution for inline executors.
    result = async_execution_func(*args, **kwargs)
    with mutex:
      async_execution_func.output_specs_and_push_result_fun = None
    return result

  return specialized_func


class _SpecializedCollection:
  """Collection of specialized functions for a single unspecialized function.

  The `get()` method retrieves the specialized function for the provided input
  spec, either by looking up a cache or by compiling the specialized function.

  Looking up a cache with an input spec as a key can be slow, because
  `Sharding`'s equivalence comparison is slow. Instead, we maintain two caches
  for the same value: we use the ID of the sharding object (via `WeakSpec`) as
  the key in one cache, and the corresponding strong references to the sharding
  object (via `StrongSpec`) as the key in another cache. Looking up the
  `WeakSpec`-keyed cache is fast. Note that the ID integer in the `WeakSpec`
  cache will remain valid as long as a strong-ref exists in the `StrongSpec`
  cache.

  The `StrongSpec`-keyed cache is unbounded, while the `WeakSpec`-keyed cache
  is LRU(1): if there is a miss in the `WeakSpec` cache but a hit in the
  `StrongSpec` cache, the strong-ref is the `StrongSpec` cache and the ID
  integer in the `WeakSpec` cache are both updated.
  """

  @dataclasses.dataclass(slots=True, unsafe_hash=True)
  class WeakSpec:
    """WeakSpec stores just the `id()` of the input spec sharding."""

    dtypes: tuple[jax.numpy.dtype, ...]
    shapes: tuple[tuple[int, ...], ...]
    sharding_ids: tuple[int, ...]
    treedef: tree_util.PyTreeDef

    def __init__(
        self, args_leaves: Sequence[jax.Array], treedef: tree_util.PyTreeDef
    ):
      self.dtypes = tuple(x.dtype for x in args_leaves)
      self.shapes = tuple(x.shape for x in args_leaves)
      self.sharding_ids = tuple(id(x.sharding) for x in args_leaves)
      self.treedef = treedef

  @dataclasses.dataclass(slots=True, unsafe_hash=True)
  class StrongSpec:
    """StrongSpec stores the full input spec sharding."""

    in_specs_treedef: tree_util.PyTreeDef | None = None
    in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None

    def __init__(
        self, args_leaves: Sequence[jax.Array], pytreedef: tree_util.PyTreeDef
    ):
      self.in_specs_leaves = tuple(_get_spec(x) for x in args_leaves)
      self.in_specs_treedef = pytreedef

  def __init__(self):
    CompiledId = int

    self._weak_to_id: dict[_SpecializedCollection.WeakSpec, CompiledId] = {}
    self._id_to_weak: dict[CompiledId, _SpecializedCollection.WeakSpec] = {}
    self._strong_to_id: dict[_SpecializedCollection.StrongSpec, CompiledId] = {}
    self._id_to_compiled: dict[CompiledId, Callable[..., Any]] = {}

    self._counter = 0
    self._mu = threading.Lock()

  def get(
      self,
      args_leaves: Sequence[jax.Array],
      pytreedef: tree_util.PyTreeDef,
      func_info: FunctionInfo,
      specialization: Specialization,
  ) -> Callable[..., Any]:
    # TODO(hyeontaek): Allow Python values in args_leaves, similar to the todo
    # in _get_spec().

    # Attempt fast-path cache hit.
    weak_spec = _SpecializedCollection.WeakSpec(args_leaves, pytreedef)
    compiled_id = self._weak_to_id.get(weak_spec)
    if compiled_id is not None:
      return self._id_to_compiled[compiled_id]

    with self._mu:
      # Attempt slow-path cache hit.
      strong_spec = _SpecializedCollection.StrongSpec(args_leaves, pytreedef)
      compiled_id = self._strong_to_id.pop(strong_spec, None)
      if compiled_id is not None:
        # Update the caches so that the fast-path cache stores the `id()` of the
        # shardings presented by the current invocation.
        old_weak = self._id_to_weak.pop(compiled_id)
        del self._weak_to_id[old_weak]

        self._strong_to_id[strong_spec] = compiled_id
        self._weak_to_id[weak_spec] = compiled_id
        self._id_to_weak[compiled_id] = weak_spec

        return self._id_to_compiled[compiled_id]

      # Cache-miss: compile.
      if specialization.devices is None:
        result = _uncached_get_specialized_func(
            func_info,
            specialization.update(
                in_specs_treedef=strong_spec.in_specs_treedef,
                in_specs_leaves=strong_spec.in_specs_leaves,
                devices=_infer_devices_from_args(args_leaves),
            ),
        )
      else:
        result = _uncached_get_specialized_func(
            func_info,
            specialization.update(
                in_specs_treedef=strong_spec.in_specs_treedef,
                in_specs_leaves=strong_spec.in_specs_leaves,
            ),
        )

      compiled_id = self._counter
      self._counter += 1

      self._weak_to_id[weak_spec] = compiled_id
      self._strong_to_id[strong_spec] = compiled_id
      self._id_to_weak[compiled_id] = weak_spec
      self._id_to_compiled[compiled_id] = result
      return result


class _JaxSecondLevelCaches:
  """Manages second-level caches registered as a single cache with JAX."""

  def __init__(self, name: str):
    self._lock = threading.Lock()
    self._callbacks: dict[int, Callable[..., Any]] = {}
    jax_register_backend_cache(self, name)

  def cache_clear(self):
    """Meant to be invoked by JAX internals."""
    for callback in self._callbacks.values():
      callback()
    self._callbacks.clear()

  def register_second_level(
      self, uid: int, cache_clear_callback: Callable[..., Any]
  ):
    self._callbacks[uid] = cache_clear_callback

  def remove_second_level(self, uid: int):
    try:
      self._callbacks.pop(uid)
    except KeyError:
      pass


class _CachedColocatedFunctionMaker:
  """Function maker for colocated Python functions.

  Generated functions are stored (cached) indefinitely so that they can be
  reused, until the cache is dropped.
  """

  JAX_CACHE = _JaxSecondLevelCaches("colocated_python_specialized_func_cache")

  def __init__(self, held_by: int | None):
    self.held_by = held_by if held_by is not None else uuid.uuid4().int
    specialized_collections: list[_SpecializedCollection] = []
    specialized_functions: list[Callable[..., Any]] = []

    def clear_caches():
      specialized_collections.clear()
      specialized_functions.clear()

    _CachedColocatedFunctionMaker.JAX_CACHE.register_second_level(
        self.held_by,
        clear_caches,
    )
    self.specialized_collections = specialized_collections
    self.specialized_functions = specialized_functions

  def __del__(self):
    self.specialized_collections.clear()
    self.specialized_functions.clear()
    try:
      _CachedColocatedFunctionMaker.JAX_CACHE.remove_second_level(self.held_by)
    except AttributeError:
      # Ignore error during python finalization.
      pass

  def _make_callable(
      self,
      info: FunctionInfo,
      specialization: Specialization,
  ):
    """Internal implementation of make_callable."""

    def specialize(
        in_specs: ShapeDtypeStructTree | None = None,
        out_specs_fn: Callable[..., ShapeDtypeStructTree] | None = None,
        devices: Sequence[jax.Device] | None = None,
    ):
      """Returns a colocated Python callable with extra specialization.

      Args:
        in_specs: Optionally specifies the expected input specs. Input specs are
          expressed as a `PyTree[ShapeDtypeStruct]` for `(args, kwargs)` of a
          function call.
        out_specs_fn: Optionally specifies a function that computes the output
          specs from input specs. If unspecified, colocated Python will compute
          the output specs during the very first execution, and this execution
          will be synchronous.
        devices: Optionally specifies the devices to execute the function on.
          Must be provided if `in_specs` has no leaves because devices cannot be
          inferred from input specs or arguments.

      Returns:
        A colocated Python callable with extra specialization.
      """
      # TODO(hyeontaek): Allow unspecified devices for zero-leaf `in_specs` if
      # `out_specs_fn(in_specs)` returns at least one leaf that we can use for
      # inferring `devices`.
      if in_specs is None:
        in_specs_leaves, in_specs_treedef = None, None
      else:
        in_specs_leaves_list, in_specs_treedef = tree_util.tree_flatten(
            in_specs
        )
        in_specs_leaves = tuple(in_specs_leaves_list)
      return self._make_callable(
          info,
          specialization.update(
              in_specs_treedef=in_specs_treedef,
              in_specs_leaves=in_specs_leaves,
              out_specs_fn=out_specs_fn,
              devices=devices,
          ),
      )

    # Caches for a collection of specialized functions or a specialized function
    # itself. The latter is used as a performance optimization when the input
    # spec is explicitly specified and can skip a collection lookup. The caches
    # use weakrefs so that we avoid creating cyclic references.
    specialized_collections_wref = lambda: None
    specialized_functions_wref = lambda: None
    wref_mu = threading.Lock()

    @api_boundary
    def __call__(*args, **kwargs):
      """Executes the given Python function on the same devices as the arguments or as specialized.

      If the callable has not been specialized with output shapes and shardings
      (see `specialize` above), the very first call will run synchronously to
      discover output shapes and shardings, and will run asynchronously after.
      If specialized with output shapes and shardings, every execution of the
      callable will be asynchronous.
      """
      args_leaves, in_specs_treedef = tree_util.tree_flatten((args, kwargs))

      no_input = len(args_leaves) == 0
      if no_input and specialization.devices is None:
        raise ValueError(
            "No devices found. colocated_python function without input"
            " arguments must be first specialized with devices."
        )

      fully_specified_in_spec = (
          specialization.in_specs_treedef is not None
          and specialization.in_specs_leaves is not None
      )

      if not fully_specified_in_spec and not no_input:
        # We need to handle input polymorphism
        nonlocal specialized_collections_wref
        with wref_mu:
          collection: _SpecializedCollection = specialized_collections_wref()
          if collection is None:
            collection = _SpecializedCollection()
            self.specialized_collections.append(collection)
            specialized_collections_wref = weakref.ref(collection)
        result = collection.get(
            args_leaves, in_specs_treedef, info, specialization
        )(*args, **kwargs)
        del collection
        return result

      # No input polymorphism -- exactly one compiled function is possible.
      with wref_mu:
        nonlocal specialized_functions_wref
        func: Callable[..., Any] = specialized_functions_wref()
        if func is None:
          if fully_specified_in_spec and specialization.devices is not None:
            func = _uncached_get_specialized_func(info, specialization)
          elif fully_specified_in_spec:
            func = _uncached_get_specialized_func(
                info,
                specialization.update(
                    devices=_infer_devices_from_args(args_leaves)
                ),
            )
          elif no_input:
            func = _uncached_get_specialized_func(
                info,
                specialization.update(
                    in_specs_leaves=tuple(),
                    in_specs_treedef=in_specs_treedef,
                ),
            )
          self.specialized_functions.append(func)
          specialized_functions_wref = weakref.ref(func)
      result = func(*args, **kwargs)
      del func
      return result

    __call__ = wraps(info.fun)(__call__)
    __call__.specialize = specialize
    return __call__

  def make_callable(
      self,
      fun: Callable[..., Any],
      fun_sourceinfo: str | None,
      fun_signature: inspect.Signature | None,
  ):
    """Makes a colocated Python callable."""
    return self._make_callable(
        FunctionInfo(fun, fun_sourceinfo, fun_signature), Specialization()
    )


_DEFAULT_FUNCTION_MAKER = _CachedColocatedFunctionMaker(None)


def make_callable(
    fun: Callable[..., Any],
    fun_sourceinfo: str | None,
    fun_signature: inspect.Signature | None,
):
  return _DEFAULT_FUNCTION_MAKER.make_callable(
      fun, fun_sourceinfo, fun_signature
  )
