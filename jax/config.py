# Copyright 2025 The JAX Authors.
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

from jax._src.config import (
    # jax2tf_default_native_serialization as jax2tf_default_native_serialization,
    # backend_target as backend_target,
    bcoo_cusparse_lowering as bcoo_cusparse_lowering,
    captured_constants_warn_bytes as captured_constants_warn_bytes,
    check_tracer_leaks as check_tracer_leaks,
    checking_leaks as checking_leaks,
    compilation_cache_dir as compilation_cache_dir,
    compilation_cache_include_metadata_in_key as compilation_cache_include_metadata_in_key,
    cpu_collectives_implementation as cpu_collectives_implementation,
    debug_infs as debug_infs,
    # cpu_enable_async_dispatch as cpu_enable_async_dispatch,
    debug_key_reuse as debug_key_reuse,
    debug_log_modules as debug_log_modules,
    debug_nans as debug_nans,
    default_device as default_device,
    default_matmul_precision as default_matmul_precision,
    default_prng_impl as default_prng_impl,
    disable_jit as disable_jit,
    # disable_most_optimizations as disable_most_optimizations,
    disable_vmap_shmap_error as disable_vmap_shmap_error,
    distributed_debug as distributed_debug,
    # distributed_initialization_timeout as distributed_initialization_timeout,
    # dump_xla_aot as dump_xla_aot,
    enable_checks as enable_checks,
    enable_compilation_cache as enable_compilation_cache,
    enable_custom_prng as enable_custom_prng,
    enable_x64 as enable_x64,
    explain_cache_misses as explain_cache_misses,
    export_ignore_forward_compatibility as export_ignore_forward_compatibility,
    hlo_source_file_canonicalization_regex as hlo_source_file_canonicalization_regex,
    jax2tf_associative_scan_reductions as jax2tf_associative_scan_reductions,
    jax_mosaic_allow_hlo as jax_mosaic_allow_hlo,
    # platform_name as platform_name,
    jax_platforms as platforms,  # noqa: F401
    # xla_backend as xla_backend,
    # TODO(phawkins): rename
    jax_xla_profile_version as xla_profile_version,  # noqa: F401
    legacy_prng_key as legacy_prng_key,
    log_compiles as log_compiles,
    logging_level as logging_level,
    make_user_context as make_user_context,
    no_execution as no_execution,
    no_tracing as no_tracing,
    numpy_dtype_promotion as numpy_dtype_promotion,
    # num_cpu_devices as num_cpu_devices,
    numpy_rank_promotion as numpy_rank_promotion,
    persistent_cache_enable_xla_caches as persistent_cache_enable_xla_caches,
    persistent_cache_min_compile_time_secs as persistent_cache_min_compile_time_secs,
    persistent_cache_min_entry_size_bytes as persistent_cache_min_entry_size_bytes,
    # ragged_dot_use_ragged_dot_instruction as ragged_dot_use_ragged_dot_instruction,
    raise_persistent_cache_errors as raise_persistent_cache_errors,
    remove_size_one_mesh_axis_from_type as remove_size_one_mesh_axis_from_type,
    send_traceback_to_runtime as send_traceback_to_runtime,
    # serialization_version as serialization_version,
    # skip_xla_dumps as skip_xla_dumps,
    softmax_custom_jvp as softmax_custom_jvp,
    threefry_partitionable as threefry_partitionable,
    traceback_filtering as traceback_filtering,
    transfer_guard_device_to_device as transfer_guard_device_to_device,
    transfer_guard_device_to_host as transfer_guard_device_to_host,
    transfer_guard_host_to_device as transfer_guard_host_to_device,
    transfer_guard as transfer_guard,
    use_direct_linearize as use_direct_linearize,
    use_shardy_partitioner as use_shardy_partitioner,
    # mock_num_gpu_processes as mock_num_gpu_processes,
    # mock_num_gpus as mock_num_gpus,
)
from jax._src.config import config as _config

# TODO(phawkins): deprecate update and read, in lieu of calling .set and .value
# on flag/state objects.
update = _config.update
read = _config.read

config_with_absl = _config.config_with_absl
parse_flags_with_absl = _config.parse_flags_with_absl

_old_configs = {
    'check_vma',
    'eager_constant_folding',
    'jax2tf_associative_scan_reductions',
    'jax2tf_default_native_serialization',
    'jax_array_garbage_collection_guard',
    'jax_bcoo_cusparse_lowering',
    'jax_captured_constants_report_frames',
    'jax_captured_constants_warn_bytes',
    'jax_check_tracer_leaks',
    'jax_compilation_cache_dir',
    'jax_compilation_cache_expect_pgle',
    'jax_compilation_cache_include_metadata_in_key',
    'jax_compilation_cache_max_size',
    'jax_compiler_enable_remat_pass',
    'jax_cpu_collectives_implementation',
    'jax_cpu_get_global_topology_timeout_minutes',
    'jax_cpu_get_local_topology_timeout_minutes',
    'jax_custom_vjp_disable_shape_check',
    'jax_debug_infs',
    'jax_debug_key_reuse',
    'jax_debug_log_modules',
    'jax_debug_nans',
    'jax_default_device',
    'jax_default_matmul_precision',
    'jax_default_prng_impl',
    'jax_disable_jit',
    'jax_disable_vmap_shmap_error',
    'jax_disallow_mesh_context_manager',
    'jax_distributed_debug',
    'jax_enable_checks',
    'jax_enable_compilation_cache',
    'jax_enable_custom_prng',
    'jax_enable_pgle',
    'jax_enable_recoverability',
    'jax_enable_x64',
    'jax_error_checking_behavior_divide',
    'jax_error_checking_behavior_nan',
    'jax_error_checking_behavior_oob',
    'jax_exec_time_optimization_effort',
    'jax_experimental_unsafe_xla_runtime_errors',
    'jax_explain_cache_misses',
    'jax_explicit_x64_dtypes',
    'jax_export_calling_convention_version',
    'jax_export_ignore_forward_compatibility',
    'jax_high_dynamic_range_gumbel',
    'jax_hlo_source_file_canonicalization_regex',
    'jax_include_full_tracebacks_in_locations',
    'jax_legacy_prng_key',
    'jax_log_checkpoint_residuals',
    'jax_log_compiles',
    'jax_logging_level',
    'jax_memory_fitting_effort',
    'jax_memory_fitting_level',
    'jax_mutable_array_checks',
    'jax_no_execution',
    'jax_no_tracing',
    'jax_num_cpu_devices',
    'jax_numpy_dtype_promotion',
    'jax_numpy_rank_promotion',
    'jax_optimization_level',
    'jax_persistent_cache_enable_xla_caches',
    'jax_persistent_cache_min_compile_time_secs',
    'jax_persistent_cache_min_entry_size_bytes',
    'jax_pgle_aggregation_percentile',
    'jax_pgle_profiling_runs',
    'jax_pjrt_client_create_options',
    'jax_platforms',
    'jax_pmap_no_rank_reduction',
    'jax_pprint_use_color',
    'jax_ragged_dot_use_ragged_dot_instruction',
    'jax_raise_persistent_cache_errors',
    'jax_random_seed_offset',
    'jax_refs_to_pins',
    'jax_remove_custom_partitioning_ptr_from_cache_key',
    'jax_remove_size_one_mesh_axis_from_type',
    'jax_safer_randint',
    'jax_serialization_version',
    'jax_share_binary_between_hosts',
    'jax_share_binary_between_hosts_timeout_ms',
    'jax_softmax_custom_jvp',
    'jax_threefry_gpu_kernel_lowering',
    'jax_threefry_partitionable',
    'jax_traceback_filtering',
    'jax_traceback_in_locations_limit',
    'jax_transfer_guard',
    'jax_transfer_guard_device_to_device',
    'jax_transfer_guard_device_to_host',
    'jax_transfer_guard_host_to_device',
    'jax_use_direct_linearize',
    'jax_use_magma',
    'jax_use_shardy_partitioner',
    'jax_use_simplified_jaxpr_constants',
    'jax_vjp3',
    'jax_vmap_primitive',
    'jax_xla_profile_version',
    'x64_enabled',
}


def __getattr__(name):
  if name == 'values':
    return _config.values
  elif name == 'x64_enabled':
    return enable_x64.value
  elif name in _old_configs:
    return _config._read(name)
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
