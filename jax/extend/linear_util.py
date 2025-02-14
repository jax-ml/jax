# Copyright 2023 The JAX Authors.
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

from typing import Callable

from jax._src.linear_util import (
  StoreException as StoreException,
  WrappedFun as WrappedFun,
  cache as cache,
  merge_linear_aux as merge_linear_aux,
  transformation as transformation,
  transformation_with_aux as transformation_with_aux,
  transformation2 as transformation2,
  transformation_with_aux2 as transformation_with_aux2,
  # TODO(b/396086979): remove this once we pass debug_info everywhere.
  wrap_init as _wrap_init,
  _missing_debug_info as _missing_debug_info,
)

# Version of wrap_init that does not require a DebugInfo object.
# This usage is deprecated, use api_util.debug_info() to construct a proper
# DebugInfo object.
def wrap_init(f: Callable, params=None, *, debug_info=None) -> WrappedFun:
  debug_info = debug_info or _missing_debug_info("linear_util.wrap_init")
  return _wrap_init(f, params, debug_info=debug_info)
