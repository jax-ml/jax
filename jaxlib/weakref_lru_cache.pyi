# Copyright 2026 The JAX Authors.
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

from collections.abc import Callable
import types

class WeakrefLRUCache:
  def evict_weakref(self, arg: object, /) -> None: ...
  def cache_keys(self) -> list[object]: ...
  def cache_info(self) -> WeakrefLRUCache.WeakrefLRUCacheInfo: ...
  def cache_clear(self) -> None: ...

  __call__: types.MethodDescriptorType = ...

  class WeakrefLRUCacheInfo:
    @property
    def hits(self) -> int: ...
    @property
    def misses(self) -> int: ...
    @property
    def maxsize(self) -> int: ...
    @property
    def currsize(self) -> int: ...
    def __repr__(self) -> str: ...

def weakref_lru_cache(
    cache_context_fn: Callable,
    fn: Callable,
    maxsize: int | None = ...,
    explain: Callable | None = ...,
    num_weakrefs: int = ...,
) -> WeakrefLRUCache: ...

class MultiWeakrefLRUCache:
  def cache_keys(self) -> list[object]: ...
  def cache_info(self) -> WeakrefLRUCache.WeakrefLRUCacheInfo: ...
  def cache_clear(self) -> None: ...

  __call__: types.MethodDescriptorType = ...

def multi_weakref_lru_cache(
    cache_context_fn: Callable,
    fn: Callable,
    *,
    maxsize: int | None = ...,
    explain: Callable | None = ...,
    registry: object,
    weak_types: set,
) -> MultiWeakrefLRUCache: ...

class WeakKeyWeakValueCache:
  __call__: types.MethodDescriptorType = ...

def weak_key_weak_value_cache(fn: Callable) -> WeakKeyWeakValueCache: ...
