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

"""MLIR/XLA integration"""

from typing import overload

from .import XlaComputation as _XlaComputation

def hlo_to_stablehlo(computation: bytes) -> bytes: ...
def xla_computation_to_mlir_module(computation: _XlaComputation) -> str: ...
@overload
def mlir_module_to_xla_computation(
    mlir_module: bytes, use_tuple_args: bool = ..., return_tuple: bool = ...
) -> _XlaComputation: ...
@overload
def mlir_module_to_xla_computation(
    mlir_module: str, use_tuple_args: bool = ..., return_tuple: bool = ...
) -> _XlaComputation: ...
@overload
def mhlo_to_stablehlo(mlir_module: bytes) -> bytes: ...
@overload
def mhlo_to_stablehlo(mlir_module: str) -> bytes: ...
@overload
def serialize_portable_artifact(
    mlir_module: bytes, target: str, use_mixed_serialization: bool = ...
) -> bytes: ...
@overload
def serialize_portable_artifact(
    mlir_module: str, target: str, use_mixed_serialization: bool = ...
) -> bytes: ...
def deserialize_portable_artifact(mlir_module: bytes) -> str: ...
def refine_polymorphic_shapes(
    mlir_module: bytes,
    enable_shape_assertions: bool = ...,
    validate_static_shapes: bool = ...,
    enable_shardy: bool = ...,
) -> bytes:
  """Refines the dynamic shapes for a module.

  The "main" function must have static shapes and all the
  intermediate dynamic shapes depend only on the input static
  shapes. Optionally, also validates that the resulting module has
  only static shapes.
  """
