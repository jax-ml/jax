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
from jax._src.lib import _jax

_deprecations = {
    "ArrayImpl": (
        (
            "jax.lib.xla_extension.ArrayImpl has been removed; use jax.Array"
            " instead."
        ),
        None,
    ),
    "XlaRuntimeError": (
        (
            "jax.lib.xla_extension.XlaRuntimeError has been removed; use"
            " jax.errors.JaxRuntimeError instead."
        ),
        None,
    ),
    # Deprecated March 26 2025.
    "DistributedRuntimeClient": (
        (
            "jax.lib.xla_extension.DistributedRuntimeClient is"
            " deprecated; use jax.distributed instead."
        ),
        _jax.DistributedRuntimeClient,
    ),
    "get_distributed_runtime_client": (
        (
            "jax.lib.xla_extension.get_distributed_runtime_client is"
            " deprecated; use jax.distributed instead."
        ),
        _jax.get_distributed_runtime_client,
    ),
    "get_distributed_runtime_service": (
        (
            "jax.lib.xla_extension.get_distributed_runtime_service is"
            " deprecated; use jax.distributed instead."
        ),
        _jax.get_distributed_runtime_service,
    ),
    "Device": (
        "jax.lib.xla_extension.Device is deprecated; use jax.Device instead.",
        _jax.Device,
    ),
    "PjitFunctionCache": (
        "jax.lib.xla_extension.PjitFunctionCache is deprecated.",
        _jax.PjitFunctionCache,
    ),
    "ifrt_proxy": (
        "jax.lib.xla_extension.ifrt_proxy is deprecated.",
        _jax.ifrt_proxy,
    ),
    "jax_jit": (
        "jax.lib.xla_extension.jax_jit is deprecated.",
        _jax.jax_jit,
    ),
    "mlir": ("jax.lib.xla_extension.mlir is deprecated.", _jax.mlir),
    "pmap_lib": ("jax.lib.xla_extension.pmap_lib is deprecated.", _jax.pmap_lib),
    "profiler": (
        "jax.lib.xla_extension.profiler is deprecated.",
        jax._src.lib._profiler,
    ),
    "pytree": (
        "jax.lib.xla_extension.pytree is deprecated.",
        _jax.pytree,
    ),
    "hlo_module_cost_analysis": (
        "jax.lib.xla_extension.hlo_module_cost_analysis is deprecated.",
        _jax.hlo_module_cost_analysis,
    ),
    "hlo_module_to_dot_graph": (
        "jax.lib.xla_extension.hlo_module_to_dot_graph is deprecated.",
        _jax.hlo_module_to_dot_graph,
    ),
    "HloModule": (
        "jax.lib.xla_extension.HloModule is deprecated.",
        _jax.HloModule,
    ),
    "HloPrintOptions": (
        "jax.lib.xla_extension.HloPrintOptions is deprecated.",
        _jax.HloPrintOptions,
    ),
    "OpSharding": (
        "jax.lib.xla_extension.OpSharding is deprecated.",
        _jax.OpSharding,
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
  Device = _jax.Device
  DistributedRuntimeClient = _jax.DistributedRuntimeClient
  HloModule = _jax.HloModule
  HloPrintOptions = _jax.HloPrintOptions
  OpSharding = _jax.OpSharding
  PjitFunction = _jax.PjitFunction
  PjitFunctionCache = _jax.PjitFunctionCache
  PmapFunction = _jax.PmapFunction

  get_distributed_runtime_client = _jax.get_distributed_runtime_client
  get_distributed_runtime_service = _jax.get_distributed_runtime_service
  hlo_module_cost_analysis = _jax.hlo_module_cost_analysis
  hlo_module_to_dot_graph = _jax.hlo_module_to_dot_graph
  ifrt_proxy = _jax.ifrt_proxy
  jax_jit = _jax.jax_jit
  mlir = _jax.mlir
  pmap_lib = _jax.pmap_lib
  profiler = jax._src.lib._profiler
  pytree = _jax.pytree

else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr

  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _jax
