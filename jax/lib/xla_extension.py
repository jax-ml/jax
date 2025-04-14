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

from jax._src.lib import xla_extension as _xe

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
        _xe.DistributedRuntimeClient,
    ),
    "get_distributed_runtime_client": (
        (
            "jax.lib.xla_extension.get_distributed_runtime_client is"
            " deprecated; use jax.distributed instead."
        ),
        _xe.get_distributed_runtime_client,
    ),
    "get_distributed_runtime_service": (
        (
            "jax.lib.xla_extension.get_distributed_runtime_service is"
            " deprecated; use jax.distributed instead."
        ),
        _xe.get_distributed_runtime_service,
    ),
    "Device": (
        "jax.lib.xla_extension.Device is deprecated; use jax.Device instead.",
        _xe.Device,
    ),
    "PjitFunctionCache": (
        "jax.lib.xla_extension.PjitFunctionCache is deprecated.",
        _xe.PjitFunctionCache,
    ),
    "ifrt_proxy": (
        "jax.lib.xla_extension.ifrt_proxy is deprecated.",
        _xe.ifrt_proxy,
    ),
    "jax_jit": (
        "jax.lib.xla_extension.jax_jit is deprecated.",
        _xe.jax_jit,
    ),
    "mlir": ("jax.lib.xla_extension.mlir is deprecated.", _xe.mlir),
    "pmap_lib": ("jax.lib.xla_extension.pmap_lib is deprecated.", _xe.pmap_lib),
    "profiler": (
        "jax.lib.xla_extension.profiler is deprecated.",
        _xe.profiler,
    ),
    "pytree": (
        "jax.lib.xla_extension.pytree is deprecated.",
        _xe.pytree,
    ),
    "hlo_module_cost_analysis": (
        "jax.lib.xla_extension.hlo_module_cost_analysis is deprecated.",
        _xe.hlo_module_cost_analysis,
    ),
    "hlo_module_to_dot_graph": (
        "jax.lib.xla_extension.hlo_module_to_dot_graph is deprecated.",
        _xe.hlo_module_to_dot_graph,
    ),
    "HloModule": (
        "jax.lib.xla_extension.HloModule is deprecated.",
        _xe.HloModule,
    ),
    "HloPrintOptions": (
        "jax.lib.xla_extension.HloPrintOptions is deprecated.",
        _xe.HloPrintOptions,
    ),
    "OpSharding": (
        "jax.lib.xla_extension.OpSharding is deprecated.",
        _xe.OpSharding,
    ),
    "PjitFunction": (
        "jax.lib.xla_extension.PjitFunction is deprecated.",
        _xe.PjitFunction,
    ),
    "PmapFunction": (
        "jax.lib.xla_extension.PmapFunction is deprecated.",
        _xe.PmapFunction,
    ),
}

import typing as _typing

if _typing.TYPE_CHECKING:
  Device = _xe.Device
  DistributedRuntimeClient = _xe.DistributedRuntimeClient
  HloModule = _xe.HloModule
  HloPrintOptions = _xe.HloPrintOptions
  OpSharding = _xe.OpSharding
  PjitFunction = _xe.PjitFunction
  PjitFunctionCache = _xe.PjitFunctionCache
  PmapFunction = _xe.PmapFunction

  get_distributed_runtime_client = _xe.get_distributed_runtime_client
  get_distributed_runtime_service = _xe.get_distributed_runtime_service
  hlo_module_cost_analysis = _xe.hlo_module_cost_analysis
  hlo_module_to_dot_graph = _xe.hlo_module_to_dot_graph
  ifrt_proxy = _xe.ifrt_proxy
  jax_jit = _xe.jax_jit
  mlir = _xe.mlir
  pmap_lib = _xe.pmap_lib
  profiler = _xe.profiler
  pytree = _xe.pytree

else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr

  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _xe
