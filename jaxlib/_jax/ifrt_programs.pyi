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

from collections.abc import Sequence
from typing import Any, overload

import jax.jaxlib._jax
import typing_extensions

import jaxlib._xla

class Program:
  pass

class CompileOptions:
  pass

@overload
def make_hlo_program(mlir_module: str) -> Program: ...
@overload
def make_hlo_program(mlir_module: bytes) -> Program: ...
def make_colocated_python_program(
    name: str,
    picked_function: bytes,
    devices: Sequence[jax.jaxlib._jax.Device] | jax.jaxlib._jax.DeviceList,
    input_avals: Sequence[Any],
    output_avals: Sequence[Any],
) -> Program: ...
@overload
def make_plugin_program(data: str) -> Program: ...
@overload
def make_plugin_program(data: bytes) -> Program: ...
def make_xla_compile_options(
    options: CompileOptions,
    executable_devices: Sequence[jax.jaxlib._jax.Device],
    host_callbacks: Sequence[typing_extensions.CapsuleType],
) -> jaxlib._xla.CompileOptions: ...
def make_colocated_python_compile_options() -> CompileOptions: ...
def make_plugin_compile_options() -> CompileOptions: ...
