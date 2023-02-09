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
  AUTO as AUTO,
  ArrayMapping as ArrayMapping,
  ArrayMappingOrAutoOrUnspecified as ArrayMappingOrAutoOrUnspecified,
  AvalDimSharding as AvalDimSharding,
  Chunked as Chunked,
  ContextDecorator as ContextDecorator,
  DynamicAxisEnv as DynamicAxisEnv,
  DynamicAxisEnvFrame as DynamicAxisEnvFrame,
  EMPTY_ENV as EMPTY_ENV,
  EmapInfo as EmapInfo,
  ExecuteReplicated as ExecuteReplicated,
  Index as Index,
  InputsHandler as InputsHandler,
  MapTrace as MapTrace,
  MapTracer as MapTracer,
  MeshAxisName as MeshAxisName,
  MeshComputation as MeshComputation,
  MeshDimAssignment as MeshDimAssignment,
  MeshExecutable as MeshExecutable,
  NoSharding as NoSharding,
  OpShardingType as OpShardingType,
  OrderedDictType as OrderedDictType,
  OutputType as OutputType,
  ParallelCallableInfo as ParallelCallableInfo,
  PartitionInfo as PartitionInfo,
  PartitionsOrReplicated as PartitionsOrReplicated,
  PmapComputation as PmapComputation,
  PmapExecutable as PmapExecutable,
  PxlaResultHandler as PxlaResultHandler,
  ReplicaInfo as ReplicaInfo,
  Replicated as Replicated,
  ResourceAxisName as ResourceAxisName,
  ResourceEnv as ResourceEnv,
  ResultsHandler as ResultsHandler,
  SPMDBatchTrace as SPMDBatchTrace,
  ShardInfo as ShardInfo,
  ShardedAxis as ShardedAxis,
  ShardedDeviceArray as ShardedDeviceArray,
  ShardedDeviceArrayBase as ShardedDeviceArrayBase,
  ShardingSpec as ShardingSpec,
  TileManual as TileManual,
  TileVectorize as TileVectorize,
  TilingMethod as TilingMethod,
  UnloadedMeshExecutable as UnloadedMeshExecutable,
  UnloadedPmapExecutable as UnloadedPmapExecutable,
  Unstacked as Unstacked,
  WeakRefList as WeakRefList,
  _PositionalSemantics as _PositionalSemantics,
  _ShardedDeviceArray as _ShardedDeviceArray,
  _UNSPECIFIED as _UNSPECIFIED,
  _create_pmap_sharding_spec as _create_pmap_sharding_spec,
  _get_and_check_device_assignment as _get_and_check_device_assignment,
  _get_sharding_specs as _get_sharding_specs,
  _is_unspecified as _is_unspecified,
  _one_replica_buffer_indices as _one_replica_buffer_indices,
  _pmap_sharding_spec as _pmap_sharding_spec,
  are_op_shardings_equal as are_op_shardings_equal,
  array_mapping_to_axis_resources as array_mapping_to_axis_resources,
  array_types as array_types,
  custom_resource_typing_rules as custom_resource_typing_rules,
  device_put as device_put,
  find_partitions as find_partitions,
  find_replicas as find_replicas,
  full_to_shard_p as full_to_shard_p,
  get_global_aval as get_global_aval,
  get_local_aval as get_local_aval,
  get_num_partitions as get_num_partitions,
  global_aval_to_result_handler as global_aval_to_result_handler,
  global_avals_to_results_handler as global_avals_to_results_handler,
  global_result_handlers as global_result_handlers,
  is_op_sharding_replicated as is_op_sharding_replicated,
  local_aval_to_result_handler as local_aval_to_result_handler,
  local_avals_to_results_handler as local_avals_to_results_handler,
  local_result_handlers as local_result_handlers,
  lower_mesh_computation as lower_mesh_computation,
  lower_parallel_callable as lower_parallel_callable,
  lower_sharding_computation as lower_sharding_computation,
  make_sharded_device_array as make_sharded_device_array,
  maybe_extend_axis_env as maybe_extend_axis_env,
  mesh_sharding_specs as mesh_sharding_specs,
  multi_host_supported_collectives as multi_host_supported_collectives,
  new_mesh_sharding_specs as new_mesh_sharding_specs,
  new_name_stack as new_name_stack,
  op_sharding_to_indices as op_sharding_to_indices,
  parallel_callable as parallel_callable,
  partitioned_sharding_spec as partitioned_sharding_spec,
  reconcile_num_partitions as reconcile_num_partitions,
  replicate as replicate,
  resource_typecheck as resource_typecheck,
  sda_array_result_handler as sda_array_result_handler,
  shard_arg_handlers as shard_arg_handlers,
  shard_args as shard_args,
  shard_aval as shard_aval,
  shard_aval_handlers as shard_aval_handlers,
  shard_to_full_p as shard_to_full_p,
  sharding_internal as sharding_internal,
  sharding_spec_sharding_proto as sharding_spec_sharding_proto,
  show_axes as show_axes,
  spec_to_indices as spec_to_indices,
  spmd_primitive_batchers as spmd_primitive_batchers,
  stage_parallel_callable as stage_parallel_callable,
  thread_resources as thread_resources,
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

# Deprecations

from jax._src.interpreters.pxla import (
  Mesh as _deprecated_Mesh,
  PartitionSpec as _deprecated_PartitionSpec,
)

import typing
if typing.TYPE_CHECKING:
  from jax._src.interpreters.pxla import (
    Mesh as Mesh,
    PartitionSpec as PartitionSpec,
  )
del typing

_deprecations = {
  # Added Feb 8, 2023:
  "Mesh": (
    "jax.interpreters.pxla.Mesh is deprecated. Use jax.sharding.Mesh.",
    _deprecated_Mesh,
  ),
  "PartitionSpec": (
    ("jax.interpreters.pxla.PartitionSpec is deprecated. Use "
     "jax.sharding.PartitionSpec."),
     _deprecated_PartitionSpec,
  ),
}

from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr, _deprecations
