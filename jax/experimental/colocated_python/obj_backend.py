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

import dataclasses
import threading
from typing import Any, Callable


@dataclasses.dataclass(frozen=True)
class _ObjectState:
  is_being_initialized: bool
  exc: Exception | None = None
  obj: Any = None


class _ObjectStore:
  """Stores live objects."""

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
    del state


SINGLETON_OBJECT_STORE = _ObjectStore()
