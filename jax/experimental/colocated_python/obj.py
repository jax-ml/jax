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
"""Colocated Python object API implementation."""

from __future__ import annotations

from collections.abc import Callable
import inspect
import random
import threading
from typing import Any
import weakref

import jax
from jax._src import api_util
from jax._src import config
from jax._src import tree_util
from jax._src.traceback_util import api_boundary
from jax._src.util import wraps
from jax.experimental.colocated_python import func
from jax.experimental.colocated_python import obj_backend

# TODO(madthanu): Remove the following config option and make its behavior the
# default, once the behavior has been declared stable.
_USE_WEAKREFS = config.bool_state(
    'jax_experimental_colocated_python_object_use_weakrefs_at_backend',
    False,
    help=(
        'Unstable in-development feature that switches the colocated-python'
        ' implementation to internally use reference counting for destructing'
        ' objects at the colocated backend, instead of invoking an explicit'
        ' delete-object function from the frontend.'
    ),
)


class _InstanceRegistry:
  """Registry of object instances."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._storage: dict[int, set[jax.Device]] = {}

  def new_instance(self) -> int:
    """Returns a new unique identifier for an instance on the controller."""
    uid = random.getrandbits(63)
    with self._lock:
      assert uid not in self._storage
      self._storage[uid] = set()
    return uid

  def update_devices(self, uid: int, device_set: set[jax.Device]) -> None:
    """Updates the set of devices on which it is live."""
    with self._lock:
      self._storage[uid] |= device_set

  def pop_instance(self, uid: int) -> set[jax.Device]:
    """Removes the instance and returns the set of devices on which it is live."""
    with self._lock:
      return self._storage.pop(uid)


SINGLETON_INSTANCE_REGISTRY = _InstanceRegistry()


@jax._src.util.cache(max_size=4096)
def _update_instance_devices(
    uid: int, shardings: tuple[jax.sharding.Sharding, ...]
) -> None:
  """Caching version of _InstanceRegistry.update_devices()."""
  device_set = set()
  for sharding in shardings:
    device_set |= sharding.device_set
  SINGLETON_INSTANCE_REGISTRY.update_devices(uid, device_set)


def _make_method(
    cls: type[object],
    cls_sourceinfo: str | None,
    uid: int,
    init_args: tuple[Any, ...],
    init_kwargs: dict[str, Any],
    method_name: str,
    original_method: Callable[..., Any],
    func_maker: func._CachedColocatedFunctionMaker,
    use_weakrefs: bool,
):

  class MethodCallerAtBackend:

    def __init__(self):
      self._lock = threading.Lock()

    def __reduce__(self):
      return type(self), ()

    def _first_call(self):
      # Temporarily hold a strong reference to a new object if it is created
      # using initializer.
      temp_strong_ref = None

      def initializer():
        if not use_weakrefs:
          return obj_backend._ClassWrapperForGarbageCollection(  # pylint: disable=protected-access
              cls(*init_args, **init_kwargs)
          )
        nonlocal temp_strong_ref
        temp_strong_ref = cls(*init_args, **init_kwargs)
        return weakref.ref(temp_strong_ref)

      retrieved = obj_backend.SINGLETON_OBJECT_STORE.get_or_create(
          uid, initializer
      )

      if use_weakrefs:
        self.obj = temp_strong_ref
      else:
        self.obj = retrieved

    def __call__(self, *args, **kwargs):
      with self._lock:
        if not hasattr(self, 'obj'):
          self._first_call()

      if use_weakrefs:
        return getattr(self.obj, method_name)(*args, **kwargs)
      else:
        assert isinstance(
            self.obj, obj_backend._ClassWrapperForGarbageCollection
        )
        return getattr(self.obj.obj, method_name)(*args, **kwargs)

  # Colocated Python callable for the controller.
  callable = func_maker.make_callable(
      MethodCallerAtBackend(),
      cls_sourceinfo,
      api_util.fun_signature(original_method),
  )

  # Outer wrapper of the method for the controller. It tracks devices that have
  # been used with any method call.
  def make_method_wrapper(callable):
    @api_boundary
    def method_wrapper(*args, **kwargs):
      # TODO(hyeontaek): Instead of inspecting argument/result shardings, get
      # shardings from final specialization of the function. This may require
      # lowering `_update_instance_devices` into the function API.

      args_leaves = tree_util.tree_leaves((args, kwargs))
      args_shardings_leaves = tuple(
          func._get_spec(x).sharding for x in args_leaves
      )
      if args_shardings_leaves:
        _update_instance_devices(uid, args_shardings_leaves)

      result = callable(*args, **kwargs)

      # If args had any array, we can skip incorporating devices from the result
      # because results will not use any new devices.
      if not args_shardings_leaves:
        result_leaves = tree_util.tree_leaves(result)
        result_shardings_leaves = tuple(
            func._get_spec(x).sharding for x in result_leaves
        )
        _update_instance_devices(uid, result_shardings_leaves)
      return result

    def specialize(*args, **kwargs):
      return make_method_wrapper(callable.specialize(*args, **kwargs))

    method_wrapper = wraps(original_method)(method_wrapper)
    method_wrapper.specialize = specialize
    return method_wrapper

  method_wrapper = make_method_wrapper(callable)
  return method_wrapper


def wrap_class(
    cls: type[object],
    cls_sourceinfo: str | None,
) -> type[object]:
  class WrappedClass:

    @wraps(cls.__init__)
    def __init__(self, *init_args, **init_kwargs) -> None:
      uid = self._colocated_python_uid = (
          SINGLETON_INSTANCE_REGISTRY.new_instance()
      )
      self.func_maker = func._CachedColocatedFunctionMaker(uid)
      self.use_weakrefs = _USE_WEAKREFS.value
      for attr_name in dir(cls):
        original_member = getattr(cls, attr_name)
        if not inspect.isfunction(original_member):
          continue

        # WrappedClass defines lazy initialization and colocated deletion logic.
        # WrappedClass is not serializable even if the original class may be
        # serializable.
        if attr_name in ('__init__', '__del__', '__reduce__', '__reduce_ex__'):
          continue

        method = _make_method(
            cls,
            cls_sourceinfo,
            uid,
            init_args,
            init_kwargs,
            attr_name,
            original_member,
            self.func_maker,
            self.use_weakrefs,
        )
        # TODO(hyeontaek): Support method specialization similar to function
        # specialization.
        setattr(self, attr_name, method)

    def __del__(self):
      del self.func_maker
      if self.use_weakrefs:
        return
      uid = self._colocated_python_uid
      devices = SINGLETON_INSTANCE_REGISTRY.pop_instance(uid)
      if devices:

        def remove_object() -> None:
          obj_backend.SINGLETON_OBJECT_STORE.remove(uid)

        destructor = func.make_callable(
            remove_object,
            cls_sourceinfo,
            None,
        )
        destructor = destructor.specialize(  # type: ignore[attribute-error]
            devices=sorted(devices, key=lambda device: device.id)
        )
        destructor()

  WrappedClass.__name__ = cls.__name__
  WrappedClass.__doc__ = cls.__doc__
  return WrappedClass
