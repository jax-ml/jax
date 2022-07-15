# Copyright 2021 Google LLC
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

import jax
from jax._src import api
from jax._src import dispatch
from jax._src.lib import xla_bridge


def clear_backends():
  """Clear all backend clients so that new backend clients can be created later.

  This is currently experimental and test-only.
  """
  xla_bridge._clear_backends()
  jax.lib.xla_bridge._backends = {}
  dispatch.xla_callable.cache_clear()  # type: ignore
  dispatch.xla_primitive_callable.cache_clear()
  api._cpp_jit_cache.clear()
