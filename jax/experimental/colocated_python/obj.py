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

import inspect
import random
import threading
from typing import Any, Callable, Type

import jax
from jax._src import api_util
from jax._src import tree_util
from jax._src.traceback_util import api_boundary
from jax._src.util import wraps
from jax.experimental.colocated_python import func
from jax.experimental.colocated_python import obj_backend


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


@jax.util.cache(max_size=4096)
def _update_instance_devices(
    uid: int, shardings: tuple[jax.sharding.Sharding, ...]
) -> None:
  """Caching version of _InstanceRegistry.update_devices()."""
  device_set = set()
  for sharding in shardings:
    device_set |= sharding.device_set
  SINGLETON_INSTANCE_REGISTRY.update_devices(uid, device_set)


def _make_method(
    cls: Type[object],
    cls_sourceinfo: str | None,
    uid: int,
    init_args: tuple[Any, ...],
    init_kwargs: dict[str, Any],
    method_name: str,
    original_method: Callable[..., Any],
):
  # Initializer to use when the object is not present in the backend.
  def initializer() -> object:
    return cls(*init_args, **init_kwargs)

  # Method to call on the backend.
  def method(*args, **kwargs):
    obj = obj_backend.SINGLETON_OBJECT_STORE.get_or_create(uid, initializer)
    return getattr(obj, method_name)(*args, **kwargs)

  # Colocated Python callable for the controller.
  callable = func.make_callable(
      method,
      cls_sourceinfo,
      api_util.fun_signature(original_method),
  )

  # Outer wrapper of the method for the controller. It tracks
  @api_boundary
  def method_wrapper(*args, **kwargs):
    if not args:
      raise NotImplementedError(
          'Method calls with no arguments are not yet supported.'
      )
    # TODO(hyeontaek): Instead of inspecting argument shardings, get shardings
    # from final specialization of the function. This may require lowering
    # `_update_instance_devices` into the function API.
    args_leaves = tree_util.tree_leaves((args, kwargs))
    shardings_leaves = tuple(func._get_spec(x).sharding for x in args_leaves)
    _update_instance_devices(uid, shardings_leaves)
    return callable(*args, **kwargs)

  method_wrapper = wraps(original_method)(method_wrapper)
  return method_wrapper


def wrap_class(
    cls: Type[object],
    cls_sourceinfo: str | None,
) -> Type[object]:
  class WrappedClass:

    @wraps(cls.__init__)
    def __init__(self, *init_args, **init_kwargs) -> None:
      uid = self._colocated_python_uid = (
          SINGLETON_INSTANCE_REGISTRY.new_instance()
      )
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
        )
        # TODO(hyeontaek): Support method specialization similar to function
        # specialization.
        setattr(self, attr_name, method)

    def __del__(self) -> None:
      uid = self._colocated_python_uid
      devices = SINGLETON_INSTANCE_REGISTRY.pop_instance(uid)
      if devices:

        def remove_object() -> None:
          obj_backend.SINGLETON_OBJECT_STORE.remove(uid)

        # TODO(hyeontaek): Request "best-effort" non-SPMD execution that tries
        # to run this function on any healthy processes instead of failing when
        # any process of the execution is unhealthy.
        destructor = func.make_callable(
            remove_object,
            cls_sourceinfo,
            None,
        )
        destructor = destructor.specialize(  # type: ignore[attribute-error]
            devices=devices
        )
        destructor()

  WrappedClass.__name__ = cls.__name__
  WrappedClass.__doc__ = cls.__doc__
  return WrappedClass
