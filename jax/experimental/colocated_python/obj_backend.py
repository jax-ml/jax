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
"""Backend for colocated_python.obj."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import threading
from typing import Any
import weakref


@dataclasses.dataclass
class _ClassWrapperForGarbageCollection:
  obj: Any


@dataclasses.dataclass
class _ConsumableRef:
  """Stores a strong ref initially, but switches to a weak ref once consumed.

  We consider a _ConsumableRef to have been consumed once the __call__ method
  has been called once, and the _ConsumableRef is no longer storing a strong
  ref. We consider a _ConsumableRef to have expired once it has been consumed
  and the resulting weak ref has expired.

  Pickling and unpickling an unexpired _ConsumableRef will create a new,
  unconsumed _ConsumableRef. Pickling and unpickling an expired _ConsumableRef
  will create an expired _ConsumableRef.
  """

  strong_ref: Any | None = None
  weak_ref: weakref.ref | None = None
  _mutex = threading.Lock()

  def __init__(self, obj: Any) -> None:
    self.strong_ref = obj

  def __call__(self, *args, **kwargs):
    with self._mutex:
      if self.strong_ref is not None:
        assert self.weak_ref is None
        result = self.strong_ref
        self.strong_ref = None
        self.weak_ref = weakref.ref(result)
        return result
      elif self.weak_ref is not None:
        return self.weak_ref()
      else:
        return None

  def __reduce__(self):
    with self._mutex:
      if self.strong_ref is not None:
        return type(self), (self.strong_ref,)
      elif self.weak_ref is not None:
        return type(self), (self.weak_ref(),)
      else:
        return type(self), (None,)


@dataclasses.dataclass(frozen=True)
class _ObjectState:
  is_being_initialized: bool
  exc: Exception | None = None
  obj: Any = None


class _ObjectStore:
  """Stores live objects.

  TODO(madthanu): Currently the dictionary never removes entries that are
  expired refs.
  """

  def __init__(self) -> None:
    self._lock = threading.Condition()
    self._storage: dict[int, _ObjectState] = {}

  def get_or_create(self, uid: int, initializer: Callable[[], Any]) -> Any:
    """Returns the object associated with the given uid, or creates it if it does not exist."""
    with self._lock:
      if uid in self._storage:
        while True:
          state = self._storage[uid]
          if state.is_being_initialized:
            # Another thread is initializing the object. Wait for it to finish.
            self._lock.wait()
          else:
            break

        if state.exc is not None:
          raise state.exc
        return state.obj

      self._storage[uid] = _ObjectState(is_being_initialized=True)

    try:
      obj = initializer()
    except Exception as exc:
      with self._lock:
        self._storage[uid] = _ObjectState(is_being_initialized=False, exc=exc)
        self._lock.notify_all()
      raise

    with self._lock:
      self._storage[uid] = _ObjectState(is_being_initialized=False, obj=obj)
      self._lock.notify_all()
      return obj

  def remove(self, uid: int) -> None:
    """Removes the object associated with the given uid."""
    with self._lock:
      state = self._storage.pop(uid)

    # The object will be deleted without holding the lock.
    if isinstance(state.obj, _ClassWrapperForGarbageCollection):
      del state.obj.obj
    del state


SINGLETON_OBJECT_STORE = _ObjectStore()
