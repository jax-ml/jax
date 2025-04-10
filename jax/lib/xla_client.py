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

get_topology_for_devices = _xc.get_topology_for_devices
heap_profile = _xc.heap_profile
mlir_api_version = _xc.mlir_api_version
Client = _xc.Client
CompileOptions = _xc.CompileOptions
DeviceAssignment = _xc.DeviceAssignment
Frame = _xc.Frame
HloSharding = _xc.HloSharding
OpSharding = _xc.OpSharding
Traceback = _xc.Traceback

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
        "Shape is deprecated; use StableHLO instead.",
        _xc.Shape,
    ),
    "XlaBuilder": (
        "XlaBuilder has been removed in JAX v0.6.0; use StableHLO instead.",
        None,
    ),
    "XlaComputation": (
        "XlaComputation is deprecated; use StableHLO instead.",
        _xc.XlaComputation,
    ),
    # Added Nov 20 2024, finalized 2025-04-09
    "ArrayImpl": (
        (
            "jax.lib.xla_client.ArrayImpl has been removed in JAX v0.6.0; use"
            " jax.Array instead."
        ),
        None,
    ),
}

import typing as _typing

if _typing.TYPE_CHECKING:
  Shape = _xc.Shape
  XlaComputation = _xc.XlaComputation
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr

  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _xc
