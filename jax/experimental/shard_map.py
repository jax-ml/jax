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
def shard_map(f, mesh, in_specs, out_specs, check_rep=True, auto=frozenset()):
  """Please use :func:`jax.shard_map`.
  :func:`jax.experimental.shard_map.shard_map` is a legacy API"""
  axis_names = frozenset(mesh.axis_names) - auto
  return jshmap._shard_map(
      f, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
      check_vma=check_rep, axis_names=axis_names, _skip_mesh_check=True)
