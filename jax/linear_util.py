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

from jax._src import linear_util as _lu


_deprecations = {
    # Added August 29, 2023:
    "StoreException": (
      "jax.linear_util.StoreException is deprecated. Use jax.extend.linear_util.StoreException instead.",
      _lu.StoreException,
    ),
    "WrappedFun": (
      "jax.linear_util.WrappedFun is deprecated. Use jax.extend.linear_util.WrappedFun instead.",
    _lu.WrappedFun,
    ),
    "cache": (
      "jax.linear_util.cache is deprecated. Use jax.extend.linear_util.cache instead.",
      _lu.cache,
    ),
    "merge_linear_aux": (
      "jax.linear_util.merge_linear_aux is deprecated. Use jax.extend.linear_util.merge_linear_aux instead.",
      _lu.merge_linear_aux
    ),
    "transformation": (
      "jax.linear_util.transformation is deprecated. Use jax.extend.linear_util.transformation instead.",
      _lu.transformation
    ),
    "transformation_with_aux": (
      "jax.linear_util.transformation_with_aux is deprecated. Use jax.extend.linear_util.transformation_with_aux instead.",
      _lu.transformation_with_aux
    ),
    "wrap_init": (
      "jax.linear_util.wrap_init is deprecated. Use jax.extend.linear_util.wrap_init instead.",
      _lu.wrap_init
    ),
}

import typing
if typing.TYPE_CHECKING:
  StoreException = _lu.StoreException
  WrappedFun = _lu.WrappedFun
  cache = _lu.cache
  merge_linear_aux = _lu.merge_linear_aux
  transformation = _lu.transformation
  transformation_with_aux = _lu.transformation_with_aux
  wrap_init = _lu.wrap_init
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del _lu
