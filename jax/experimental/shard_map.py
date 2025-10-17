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

from jax._src import traceback_util
from jax._src import shard_map as jshmap

@traceback_util.api_boundary
def shard_map(f, mesh, in_specs, out_specs, check_rep=True):
  """Please use `jax.shard_map`. `jax.experimental.shard_map.shard_map`
  has been deprecated."""
  return jshmap._shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                           axis_names=set(), check_vma=check_rep)

_deprecations = {
    # Deprecated in v0.8.0; we plan to keep this as a deprecated legacy API.
    "shard_map": (
      "jax.experimental.shard_map is deprecated in v0.8.0. Used jax.shard_map instead.",
      shard_map
    )
}

import typing
if typing.TYPE_CHECKING:
  pass
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
  del shard_map
del typing
