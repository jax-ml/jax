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
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.interpreters import pxla as _deprecated_pxla
from jax._src import mesh as _deprecated_mesh
from jax._src import op_shardings as _deprecated_op_shardings
from jax._src import sharding_impls as _deprecated_sharding_impls

from jax._src.interpreters.pxla import (
  create_compile_options as create_compile_options,
)

_deprecations = {
    # deprecated as of JAX v0.8.2 (Dec 2025)
    "Index": (
        "jax.interpreters.pxla.Index is deprecated as of JAX v0.8.2.",
        _deprecated_pxla.Index,
    ),
    "MeshAxisName": (
        (
            "jax.interpreters.pxla.MeshAxisName is deprecated as of JAX v0.8.2."
            " Use jax.sharding.Mesh axis names directly."
        ),
        _deprecated_pxla.MeshAxisName,
    ),
    "MeshComputation": (
        "jax.interpreters.pxla.MeshComputation is deprecated as of JAX v0.8.2.",
        _deprecated_pxla.MeshComputation,
    ),
    "MeshExecutable": (
        "jax.interpreters.pxla.MeshExecutable is deprecated as of JAX v0.8.2.",
        _deprecated_pxla.MeshExecutable,
    ),
    "global_aval_to_result_handler": (
        (
            "jax.interpreters.pxla.global_aval_to_result_handler is deprecated"
            " as of JAX v0.8.2."
        ),
        _deprecated_pxla.global_aval_to_result_handler,
    ),
    "global_avals_to_results_handler": (
        (
            "jax.interpreters.pxla.global_avals_to_results_handler is"
            " deprecated as of JAX v0.8.2."
        ),
        _deprecated_pxla.global_avals_to_results_handler,
    ),
    "global_result_handlers": (
        (
            "jax.interpreters.pxla.global_result_handlers is deprecated as of"
            " JAX v0.8.2."
        ),
        _deprecated_pxla.global_result_handlers,
    ),
    "thread_resources": (
        (
            "jax.interpreters.pxla.thread_resources is deprecated as of JAX"
            " v0.8.2. Please switch to using `with jax.set_mesh(mesh)` instead"
            " of `with mesh:` and then use `jax.sharding.get_abstract_mesh()`"
            " to get the current mesh."
        ),
        _deprecated_mesh.thread_resources,
    ),
    "are_hlo_shardings_equal": (
        (
            "jax.interpreters.pxla.are_hlo_shardings_equal is deprecated as of"
            " JAX v0.8.2."
        ),
        _deprecated_op_shardings.are_hlo_shardings_equal,
    ),
    "is_hlo_sharding_replicated": (
        (
            "jax.interpreters.pxla.is_hlo_sharding_replicated is deprecated as"
            " of JAX v0.8.2."
        ),
        _deprecated_op_shardings.is_hlo_sharding_replicated,
    ),
    "op_sharding_to_indices": (
        (
            "jax.interpreters.pxla.op_sharding_to_indices is deprecated as of"
            " JAX v0.8.2."
        ),
        _deprecated_op_shardings.op_sharding_to_indices,
    ),
    "ArrayMapping": (
        "jax.interpreters.pxla.ArrayMapping is deprecated as of JAX v0.8.2.",
        _deprecated_sharding_impls.ArrayMapping,
    ),
    "_UNSPECIFIED": (
        "jax.interpreters.pxla._UNSPECIFIED is deprecated as of JAX v0.8.2.",
        _deprecated_sharding_impls.UNSPECIFIED,
    ),
    "array_mapping_to_axis_resources": (
        (
            "jax.interpreters.pxla.array_mapping_to_axis_resources is"
            " deprecated as of JAX v0.8.2."
        ),
        _deprecated_sharding_impls.array_mapping_to_axis_resources,
    ),
    # Deprecated as of JAX v0.8.2; finalized in JAX v0.10.0; remove in v0.11.0.
    "MapTracer": (
        "jax.interpreters.pxla.MapTracer was removed in JAX v0.10.0.",
        None,
    ),
    "PmapExecutable": (
        "jax.interpreters.pxla.PmapExecutable was removed in JAX v0.10.0.",
        None,
    ),
    "parallel_callable": (
        (
            "jax.interpreters.pxla.parallel_callable was removed in JAX"
            " v0.10.0."
        ),
        None,
    ),
    "shard_args": (
        "jax.interpreters.pxla.shard_args was removed in JAX v0.10.0.",
        None,
    ),
    "xla_pmap_p": (
        "jax.interpreters.pxla.xla_pmap_p was removed in JAX v0.10.0.",
        None,
    ),
    "Chunked": (
        (
            "jax.interpreters.pxla.Chunked was removed in JAX v0.10.0."
            " Please use `jax.shard_map` instead of `jax.pmap`."
        ),
        None,
    ),
    "NoSharding": (
        (
            "jax.interpreters.pxla.NoSharding was removed in JAX v0.10.0."
            " Please use `jax.shard_map` instead of `jax.pmap`."
        ),
        None,
    ),
    "Replicated": (
        (
            "jax.interpreters.pxla.Replicated was removed in JAX v0.10.0."
            " Please use `jax.shard_map` instead of `jax.pmap`."
        ),
        None,
    ),
    "ShardedAxis": (
        (
            "jax.interpreters.pxla.ShardedAxis was removed in JAX v0.10.0."
            " Please use `jax.shard_map` instead of `jax.pmap`."
        ),
        None,
    ),
    "ShardingSpec": (
        (
            "jax.interpreters.pxla.ShardingSpec was removed in JAX v0.10.0."
            " Please use `jax.shard_map` instead of `jax.pmap`."
        ),
        None,
    ),
    "Unstacked": (
        (
            "jax.interpreters.pxla.Unstacked was removed in JAX v0.10.0."
            " Please use `jax.shard_map` instead of `jax.pmap`."
        ),
        None,
    ),
    "spec_to_indices": (
        (
            "jax.interpreters.pxla.spec_to_indices was removed in JAX"
            " v0.10.0. Please use `jax.shard_map` instead of `jax.pmap`."
        ),
        None,
    ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  Index = _deprecated_pxla.Index
  MeshAxisName = _deprecated_pxla.MeshAxisName
  MeshComputation = _deprecated_pxla.MeshComputation
  MeshExecutable = _deprecated_pxla.MeshExecutable
  global_aval_to_result_handler = _deprecated_pxla.global_aval_to_result_handler
  global_avals_to_results_handler = _deprecated_pxla.global_avals_to_results_handler
  global_result_handlers = _deprecated_pxla.global_result_handlers
  thread_resources = _deprecated_mesh.thread_resources
  are_hlo_shardings_equal = _deprecated_op_shardings.are_hlo_shardings_equal
  is_hlo_sharding_replicated = _deprecated_op_shardings.is_hlo_sharding_replicated
  op_sharding_to_indices = _deprecated_op_shardings.op_sharding_to_indices
  ArrayMapping = _deprecated_sharding_impls.ArrayMapping
  _UNSPECIFIED = _deprecated_sharding_impls.UNSPECIFIED
  array_mapping_to_axis_resources = _deprecated_sharding_impls.array_mapping_to_axis_resources
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
