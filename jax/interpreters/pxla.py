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

from jax._src.interpreters.pxla import (
  create_compile_options as create_compile_options,
)

_deprecations = {
    # deprecated in JAX v0.8.2 and finalized in JAX v0.11.0
    "Index": (
        (
            "jax.interpreters.pxla.Index was deprecated as of JAX v0.8.2 and"
            " removed in JAX v0.11.0."
        ),
        None,
    ),
    "MeshAxisName": (
        (
            "jax.interpreters.pxla.MeshAxisName was deprecated in JAX v0.8.2"
            " and removed in JAX v0.11.0. Use jax.sharding.Mesh axis names"
            " directly."
        ),
        None,
    ),
    "MeshExecutable": (
        (
            "jax.interpreters.pxla.MeshExecutable was deprecated in JAX v0.8.2"
            " and removed in JAX v0.11.0."
        ),
        None,
    ),
    "global_aval_to_result_handler": (
        (
            "jax.interpreters.pxla.global_aval_to_result_handler was deprecated"
            " in JAX v0.8.2 and removed in JAX v0.11.0."
        ),
        None,
    ),
    "global_result_handlers": (
        (
            "jax.interpreters.pxla.global_result_handlers was deprecated in"
            " JAX v0.8.2 and removed in JAX v0.11.0."
        ),
        None,
    ),
    "are_hlo_shardings_equal": (
        (
            "jax.interpreters.pxla.are_hlo_shardings_equal was deprecated in"
            " JAX v0.8.2 and removed in JAX v0.11.0."
        ),
        None,
    ),
    "is_hlo_sharding_replicated": (
        (
            "jax.interpreters.pxla.is_hlo_sharding_replicated is deprecated as"
            " of JAX v0.8.2."
        ),
        None,
    ),
    "ArrayMapping": (
        (
            "jax.interpreters.pxla.ArrayMapping was deprecated in JAX v0.8.2 and"
            " removed in JAX v0.11.0."
        ),
        None,
    ),
    "_UNSPECIFIED": (
        (
            "jax.interpreters.pxla._UNSPECIFIED was deprecated in JAX v0.8.2 and"
            " removed in JAX v0.11.0."
        ),
        None,
    ),
    "array_mapping_to_axis_resources": (
        (
            "jax.interpreters.pxla.array_mapping_to_axis_resources was"
            " deprecated in JAX v0.8.2 and removed in JAX v0.11.0."
        ),
        None,
    ),
    # Deprecated in JAX v0.8.2, TODO(jakevdp) finalize after v0.11.0
    "MeshComputation": (
        "jax.interpreters.pxla.MeshComputation is deprecated as of JAX v0.8.2.",
        _deprecated_pxla.MeshComputation,
    ),
    "global_avals_to_results_handler": (
        (
            "jax.interpreters.pxla.global_avals_to_results_handler is"
            " deprecated as of JAX v0.8.2."
        ),
        _deprecated_pxla.global_avals_to_results_handler,
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
    "op_sharding_to_indices": (
        (
            "jax.interpreters.pxla.op_sharding_to_indices is deprecated as of"
            " JAX v0.8.2."
        ),
        _deprecated_op_shardings.op_sharding_to_indices,
    ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  MeshComputation = _deprecated_pxla.MeshComputation
  global_avals_to_results_handler = _deprecated_pxla.global_avals_to_results_handler
  thread_resources = _deprecated_mesh.thread_resources
  op_sharding_to_indices = _deprecated_op_shardings.op_sharding_to_indices
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
