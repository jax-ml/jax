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

from jax._src.lib import xla_client as _xc

_deprecations = {
    # Finalized for JAX v0.7.0
    "heap_profile": (
        (
            "jax.lib.xla_client.heap_profile was deprecated in JAX v0.6.0 and"
            " removed in JAX v0.7.0"
        ),
        None,
    ),
    "get_topology_for_devices": (
        (
            "jax.lib.xla_client.get_topology_for_devices was deprecated in JAX"
            " v0.6.0 and removed in JAX v0.7.0"
        ),
        None,
    ),
    "mlir_api_version": (
        (
            "jax.lib.xla_client.mlir_api_version was deprecated in JAX v0.6.0"
            " and removed in JAX v0.7.0"
        ),
        None,
    ),
    "DeviceAssignment": (
        (
            "jax.lib.xla_client.DeviceAssignment was deprecated in JAX v0.6.0"
            " and removed in JAX v0.7.0"
        ),
        None,
    ),
    # Added April 4 2025.
    "Client": (
        (
            "jax.lib.xla_client.Client was deprecated in JAX v0.6.0 and will be"
            " removed in JAX v0.8.0"
        ),
        _xc.Client,
    ),
    "CompileOptions": (
        (
            "jax.lib.xla_client.CompileOptions was deprecated in JAX v0.6.0 and"
            " will be removed in JAX v0.8.0"
        ),
        _xc.CompileOptions,
    ),
    "Frame": (
        (
            "jax.lib.xla_client.Frame was deprecated in JAX v0.6.0 and will be"
            " removed in JAX v0.8.0"
        ),
        _xc.Frame,
    ),
    "HloSharding": (
        (
            "jax.lib.xla_client.HloSharding was deprecated in JAX v0.6.0 and"
            " will be removed in JAX v0.8.0"
        ),
        _xc.HloSharding,
    ),
    "OpSharding": (
        (
            "jax.lib.xla_client.OpSharding was deprecated in JAX v0.6.0 and"
            " will be removed in JAX v0.8.0"
        ),
        _xc.OpSharding,
    ),
    "Traceback": (
        (
            "jax.lib.xla_client.Traceback was deprecated in JAX v0.6.0 and will"
            " be removed in JAX v0.8.0"
        ),
        _xc.Traceback,
    ),
}

import typing as _typing

if _typing.TYPE_CHECKING:
  Client = _xc.Client
  CompileOptions = _xc.CompileOptions
  Frame = _xc.Frame
  HloSharding = _xc.HloSharding
  OpSharding = _xc.OpSharding
  Traceback = _xc.Traceback
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _xc
