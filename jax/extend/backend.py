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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.api import (
  clear_backends as clear_backends,
)
from jax._src.xla_bridge import (
  backends as backends,
  backend_xla_version as backend_xla_version,
  get_backend as get_backend,
  register_backend_factory as register_backend_factory,
)
from jax._src.interpreters.pxla import (
  get_default_device as get_default_device
)
from jax._src import util as _util

def add_clear_backends_callback(cache_clear):
  # DEPRECATED: use `util.register_cache`.
  class _BackendCacheDeprecated:
    def __init__(self, cache_clear):
      self._cache_clear = cache_clear

    def cache_clear(self):
      self._cache_clear()
  register_backend_cache(_BackendCacheDeprecated(cache_clear),  # type: ignore
                         str(cache_clear))

def register_backend_cache(cache, for_what: str):
  """Registers a cache, to be cleared when JAX clears the caches.

  Args:
    cache: an object supporting `cache_clear()`, `cache_info()`, and
    `cache_keys()`, like the result of `functools.lru_cache()`.
    for_what: a string to identify what this cache is used for. This is
     used for debugging.
  """
  _util.register_cache(cache, for_what)
