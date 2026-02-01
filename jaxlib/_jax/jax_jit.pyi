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

"""Jax C++ jit library"""

from collections.abc import Callable, Sequence
from .config import Config as _Config
from .pytree import (
    PyTreeDef as _PyTreeDef,
    PyTreeRegistry as _PyTreeRegistry,
)
import numpy

def set_disable_jit_state(config: _Config) -> None: ...
def set_enable_x64_state(config: _Config) -> None: ...
def set_post_hook_state(config: _Config) -> None: ...
def set_thread_local_state_initialization_callback(
    f: Callable[[], None],
) -> None: ...

class PyArgSignature:
  @property
  def dtype(self) -> numpy.dtype: ...
  @property
  def shape(self) -> tuple[int, ...]: ...
  @property
  def weak_type(self) -> bool: ...

def _ArgSignatureOfValue(arg0: object, arg1: bool, /) -> PyArgSignature: ...

class ArgumentSignature:
  @property
  def static_args(self) -> list[object]: ...
  @property
  def static_arg_names(self) -> list[str]: ...
  @property
  def dynamic_arg_names(self) -> list[str]: ...
  @property
  def dynamic_arg_treedefs(self) -> Sequence[_PyTreeDef]: ...
  def __repr__(self) -> str: ...
  def __str__(self) -> str: ...
  def __hash__(self) -> int: ...
  def __eq__(self, arg: object, /) -> bool: ...
  def __ne__(self, arg: object, /) -> bool: ...

def parse_arguments(
    positional_args: Sequence[object],
    keyword_args: Sequence[object],
    kwnames: tuple[str, ...],
    static_argnums: Sequence[int],
    static_argnames: Sequence[str],
    pytree_registry: _PyTreeRegistry,
) -> tuple[ArgumentSignature, list[object]]:
  """Parses the arguments to a function as jax.jit would.

  Returns a ArgumentSignature and the flattened dynamic arguments.

  Args:
    positional_args: The positional arguments.
    keyword_args: The keyword arguments.
    kwnames: The keyword names.
    static_argnums: The static argument numbers.
    static_argnames: The static argument names.
    pytree_registry: The pytree registry.
  """
