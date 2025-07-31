# Copyright 2024 The JAX Authors.
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

import jax._src.lib
from jax._src.lib import ifrt_proxy as _ifrt_proxy
from jax._src.lib import _jax

_deprecations = {
    # Finalized for JAX v0.7.0
    "Device": (
        (
            "jax.lib.xla_extension.Device was deprecated in JAX v0.6.0"
            " and removed in JAX v0.7.0; use jax.Device instead."
        ),
        None,
    ),
    "DistributedRuntimeClient": (
        (
            "jax.lib.xla_extension.DistributedRuntimeClient deprecated in JAX"
            " v0.6.0 and removed in JAX v0.7.0; use jax.distributed instead."
        ),
        None,
    ),
    "HloModule": (
        (
            "jax.lib.xla_extension.HloModule deprecated in JAX v0.6.0"
            " and removed in JAX v0.7.0."
        ),
        None,
    ),
    "OpSharding": (
        (
            "jax.lib.xla_extension.OpSharding deprecated in JAX v0.6.0"
            " and removed in JAX v0.7.0."
        ),
        None,
    ),
    "PjitFunctionCache": (
        (
            "jax.lib.xla_extension.PjitFunctionCache was deprecated in JAX v0.6.0"
            " and removed in JAX v0.7.0."
        ),
        None,
    ),
    "get_distributed_runtime_client": (
        (
            "jax.lib.xla_extension.get_distributed_runtime_client was deprecated"
            " in JAX v0.6.0 and removed in JAX v0.7.0; use jax.distributed instead."
        ),
       None,
    ),
    "get_distributed_runtime_service": (
        (
            "jax.lib.xla_extension.get_distributed_runtime_service was deprecated"
            " in JAX v0.6.0 and removed in JAX v0.7.0; use jax.distributed instead."
        ),
        None,
    ),
    "jax_jit": (
        "jax.lib.xla_extension.jax_jit deprecated in JAX v0.6.0 and removed in JAX v0.7.0.",
        None,
    ),
    "pmap_lib": (
        "jax.lib.xla_extension.pmap_lib deprecated in JAX v0.6.0 and removed in JAX v0.7.0.",
       None
    ),
    "pytree": (
        "jax.lib.xla_extension.pytree deprecated in JAX v0.6.0 and removed in JAX v0.7.0.",
        None,
    ),
    # Deprecated March 26 2025.
    "ifrt_proxy": (
        "jax.lib.xla_extension.ifrt_proxy is deprecated.",
        _ifrt_proxy,
    ),
    "mlir": ("jax.lib.xla_extension.mlir is deprecated.", _jax.mlir),
    "profiler": (
        "jax.lib.xla_extension.profiler is deprecated.",
        jax._src.lib._profiler,
    ),
    "hlo_module_cost_analysis": (
        "jax.lib.xla_extension.hlo_module_cost_analysis is deprecated.",
        _jax.hlo_module_cost_analysis,
    ),
    "hlo_module_to_dot_graph": (
        "jax.lib.xla_extension.hlo_module_to_dot_graph is deprecated.",
        _jax.hlo_module_to_dot_graph,
    ),
    "HloPrintOptions": (
        "jax.lib.xla_extension.HloPrintOptions is deprecated.",
        _jax.HloPrintOptions,
    ),
    "PjitFunction": (
        "jax.lib.xla_extension.PjitFunction is deprecated.",
        _jax.PjitFunction,
    ),
    "PmapFunction": (
        "jax.lib.xla_extension.PmapFunction is deprecated.",
        _jax.PmapFunction,
    ),
}

import typing as _typing

if _typing.TYPE_CHECKING:
  HloPrintOptions = _jax.HloPrintOptions
  PjitFunction = _jax.PjitFunction
  PmapFunction = _jax.PmapFunction
  hlo_module_cost_analysis = _jax.hlo_module_cost_analysis
  hlo_module_to_dot_graph = _jax.hlo_module_to_dot_graph
  ifrt_proxy = _ifrt_proxy
  mlir = _jax.mlir
  profiler = jax._src.lib._profiler
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr

  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _ifrt_proxy
del _jax
