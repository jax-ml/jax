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

from collections.abc import Mapping, Sequence, Set

from jax.jaxlib._jax import Client

def _transfer_to_shardings(
    arrays: Sequence, out_shardings: Sequence, donate: bool = ...
) -> list: ...
def _split_by_mesh_axis(
    arrays: object,
    sharded_dim_idxs: Sequence[int],
    mesh_axis_sizes: Sequence[int],
    mesh_axis_idx: int,
    mesh_axis_sections: Sequence[int],
    submesh_shardings: Sequence[Sequence[object]],
    donate: bool,
) -> list[list[object]]: ...
def _concatenate_by_mesh_axis(
    arrays: object,
    sharded_dim_idxs: Sequence[int],
    mesh_axis_sizes: Sequence[int],
    mesh_axis_idx: int,
    mesh_axis_sections: Sequence[int],
    out_shardings: Sequence[object],
    donate: bool,
) -> list[object]: ...
def _create_cpu_client(
    addressable_devices: Set[int], device_id_to_process_index: Mapping[int, int]
) -> Client: ...
