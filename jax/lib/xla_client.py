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

import gzip as _gzip
from jax._src.lib import xla_client as _xc

def _heap_profile(client):
  return _gzip.compress(client.heap_profile())

_deprecations = {
    # Finalized 2025-03-25; remove after 2025-06-25
    "FftType": (
        (
            "jax.lib.xla_client.FftType was removed in JAX v0.6.0; use"
            " jax.lax.FftType."
        ),
        None,
    ),
    "PaddingType": (
        (
            "jax.lib.xla_client.PaddingType was removed in JAX v0.6.0;"
            " this type is unused by JAX so there is no replacement."
        ),
        None,
    ),
    "dtype_to_etype": (
        "dtype_to_etype was removed in JAX v0.6.0; use StableHLO instead.",
        None,
    ),
    "shape_from_pyval": (
        "shape_from_pyval was removed in JAX v0.6.0; use StableHLO instead.",
        None,
    ),
    # Added Oct 11 2024, finalized 2025-04-09
    "ops": (
        "ops has been removed in JAX v0.6.0; use StableHLO instead.",
        None,
    ),
    "register_custom_call_target": (
        (
            "register_custom_call_target has been removed in JAX v0.6.0; use"
            " the JAX FFI instead (https://docs.jax.dev/en/latest/ffi.html)"
        ),
        None,
    ),
    "PrimitiveType": (
        "PrimitiveType has been removed in JAX v0.6.0; use StableHLO instead.",
        None,
    ),
    "Shape": (
        "Shape has been removed in JAX v0.6.0; use StableHLO instead.",
        None,
    ),
    "XlaBuilder": (
        "XlaBuilder has been removed in JAX v0.6.0; use StableHLO instead.",
        None,
    ),
    "XlaComputation": (
        "XlaComputation has been removed in JAX v0.6.0; use StableHLO instead.",
        None,
    ),
    # Added Nov 20 2024, finalized 2025-04-09
    "ArrayImpl": (
        (
            "jax.lib.xla_client.ArrayImpl has been removed in JAX v0.6.0; use"
            " jax.Array instead."
        ),
        None,
    ),
    # Added April 4 2025.
    "get_topology_for_devices": (
        (
            "jax.lib.xla_client.get_topology_for_devices was deprecated in JAX"
            " v0.6.0 and will be removed in JAX v0.7.0"
        ),
        _xc.get_topology_for_devices,
    ),
    "heap_profile": (
        (
            "jax.lib.xla_client.heap_profile was deprecated in JAX v0.6.0 and"
            " will be removed in JAX v0.7.0"
        ),
        _heap_profile,
    ),
    "mlir_api_version": (
        (
            "jax.lib.xla_client.mlir_api_version was deprecated in JAX v0.6.0"
            " and will be removed in JAX v0.7.0"
        ),
        58,
    ),
    "Client": (
        (
            "jax.lib.xla_client.Client was deprecated in JAX v0.6.0 and will be"
            " removed in JAX v0.7.0"
        ),
        _xc.Client,
    ),
    "CompileOptions": (
        (
            "jax.lib.xla_client.CompileOptions was deprecated in JAX v0.6.0 and"
            " will be removed in JAX v0.7.0"
        ),
        _xc.CompileOptions,
    ),
    "DeviceAssignment": (
        (
            "jax.lib.xla_client.DeviceAssignment was deprecated in JAX v0.6.0"
            " and will be removed in JAX v0.7.0"
        ),
        _xc.DeviceAssignment,
    ),
    "Frame": (
        (
            "jax.lib.xla_client.Frame was deprecated in JAX v0.6.0 and will be"
            " removed in JAX v0.7.0"
        ),
        _xc.Frame,
    ),
    "HloSharding": (
        (
            "jax.lib.xla_client.HloSharding was deprecated in JAX v0.6.0 and"
            " will be removed in JAX v0.7.0"
        ),
        _xc.HloSharding,
    ),
    "OpSharding": (
        (
            "jax.lib.xla_client.OpSharding was deprecated in JAX v0.6.0 and"
            " will be removed in JAX v0.7.0"
        ),
        _xc.OpSharding,
    ),
    "Traceback": (
        (
            "jax.lib.xla_client.Traceback was deprecated in JAX v0.6.0 and will"
            " be removed in JAX v0.7.0"
        ),
        _xc.Traceback,
    ),
}

import typing as _typing

if _typing.TYPE_CHECKING:
  get_topology_for_devices = _xc.get_topology_for_devices
  heap_profile = _heap_profile
  mlir_api_version = 58
  Client = _xc.Client
  CompileOptions = _xc.CompileOptions
  DeviceAssignment = _xc.DeviceAssignment
  Frame = _xc.Frame
  HloSharding = _xc.HloSharding
  OpSharding = _xc.OpSharding
  Traceback = _xc.Traceback
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr

  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _heap_profile
del _xc
