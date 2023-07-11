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

from jax._src.interpreters.pxla import (
  AvalDimSharding as AvalDimSharding,
  EmapInfo as EmapInfo,
  ExecuteReplicated as ExecuteReplicated,
  Index as Index,
  InputsHandler as InputsHandler,
  MapTrace as MapTrace,
  MapTracer as MapTracer,
  MeshComputation as MeshComputation,
  MeshDimAssignment as MeshDimAssignment,
  MeshExecutable as MeshExecutable,
  ParallelCallableInfo as ParallelCallableInfo,
  PmapComputation as PmapComputation,
  PmapExecutable as PmapExecutable,
  PxlaResultHandler as PxlaResultHandler,
  ReplicaInfo as ReplicaInfo,
  ResultsHandler as ResultsHandler,
  SPMDBatchTrace as SPMDBatchTrace,
  ShardInfo as ShardInfo,
  TileManual as TileManual,
  TileVectorize as TileVectorize,
  TilingMethod as TilingMethod,
  UnloadedMeshExecutable as UnloadedMeshExecutable,
  UnloadedPmapExecutable as UnloadedPmapExecutable,
  WeakRefList as WeakRefList,
  _create_pmap_sharding_spec as _create_pmap_sharding_spec,
  _get_and_check_device_assignment as _get_and_check_device_assignment,
  _pmap_sharding_spec as _pmap_sharding_spec,
  array_types as array_types,
  custom_resource_typing_rules as custom_resource_typing_rules,
  find_replicas as find_replicas,
  full_to_shard_p as full_to_shard_p,
  global_aval_to_result_handler as global_aval_to_result_handler,
  global_avals_to_results_handler as global_avals_to_results_handler,
  global_result_handlers as global_result_handlers,
  local_aval_to_result_handler as local_aval_to_result_handler,
  local_avals_to_results_handler as local_avals_to_results_handler,
  local_result_handlers as local_result_handlers,
  lower_mesh_computation as lower_mesh_computation,
  lower_parallel_callable as lower_parallel_callable,
  lower_sharding_computation as lower_sharding_computation,
  maybe_extend_axis_env as maybe_extend_axis_env,
  mesh_sharding_specs as mesh_sharding_specs,
  multi_host_supported_collectives as multi_host_supported_collectives,
  parallel_callable as parallel_callable,
  replicate as replicate,
  resource_typecheck as resource_typecheck,
  shard_arg_handlers as shard_arg_handlers,
  shard_args as shard_args,
  shard_arg as shard_arg,
  shard_aval as shard_aval,
  shard_aval_handlers as shard_aval_handlers,
  shard_to_full_p as shard_to_full_p,
  spmd_primitive_batchers as spmd_primitive_batchers,
  stage_parallel_callable as stage_parallel_callable,
  tile_aval_nd as tile_aval_nd,
  untile_aval_nd as untile_aval_nd,
  vtile_by_mesh as vtile_by_mesh,
  vtile_manual as vtile_manual,
  wrap_name as wrap_name,
  xb as xb,
  xla_pmap as xla_pmap,
  xla_pmap_impl as xla_pmap_impl,
  xla_pmap_impl_lazy as xla_pmap_impl_lazy,
  xla_pmap_p as xla_pmap_p,
)
from jax._src.mesh import (
  MeshAxisName as MeshAxisName,
  thread_resources as thread_resources,
)

from jax._src.op_shardings import (
  are_op_shardings_equal as are_op_shardings_equal,
  is_op_sharding_replicated as is_op_sharding_replicated,
  op_sharding_to_indices as op_sharding_to_indices,
)

from jax._src.sharding_impls import (
  ArrayMapping as ArrayMapping,
  ArrayMappingOrAutoOrUnspecified as ArrayMappingOrAutoOrUnspecified,
  AUTO as AUTO,
  UNSPECIFIED as _UNSPECIFIED,
  array_mapping_to_axis_resources as array_mapping_to_axis_resources,
  is_unspecified as _is_unspecified,
)

from jax._src.sharding_specs import (
  Chunked as Chunked,
  NoSharding as NoSharding,
  OpShardingType as OpShardingType,
  Replicated as Replicated,
  ShardedAxis as ShardedAxis,
  ShardingSpec as ShardingSpec,
  Unstacked as Unstacked,
  new_mesh_sharding_specs as new_mesh_sharding_specs,
  sharding_spec_sharding_proto as sharding_spec_sharding_proto,
  spec_to_indices as spec_to_indices,
)
