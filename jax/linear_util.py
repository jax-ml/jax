# Copyright 2018 The JAX Authors.
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
# See PEP 484 & https://github.com/google/jax/issues/7570

# TODO(jakevdp): deprecate these and remove this module.

from jax._src.linear_util import (
  EmptyStoreValue as EmptyStoreValue,
  Store as Store,
  StoreException as StoreException,
  WrappedFun as WrappedFun,
  _EMPTY_STORE_VALUE as _EMPTY_STORE_VALUE,
  _check_input_type as _check_input_type,
  _copy_main_traces as _copy_main_traces,
  annotate as annotate,
  annotations as annotations,
  cache as cache,
  config as config,
  core as core,
  curry as curry,
  fun_name as fun_name,
  hashable_partial as hashable_partial,
  merge_linear_aux as merge_linear_aux,
  traceback_util as traceback_util,
  transformation as transformation,
  transformation_with_aux as transformation_with_aux,
  tree_map as tree_map,
  wrap_init as wrap_init,
)
