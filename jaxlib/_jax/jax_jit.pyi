# Copyright 2021 The JAX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Any
from collections.abc import Callable, Sequence

import numpy as np
from jaxlib import _jax

from . import pytree

Client = _jax.Client
Device = _jax.Device


class JitState:
  disable_jit: bool | None
  enable_x64: bool | None
  default_device: Any | None
  extra_jit_context: Any | None
  post_hook: Callable[..., Any] | None

def global_state() -> JitState: ...
def thread_local_state() -> JitState: ...

def get_enable_x64() -> bool: ...
def set_thread_local_state_initialization_callback(
    function: Callable[[], None]): ...

def swap_thread_local_state_disable_jit(
    value: bool | None) -> bool | None: ...

class ArgSignature:
  dtype: np.dtype
  shape: tuple[int, ...]
  weak_type: bool

def _ArgSignatureOfValue(
    __arg: Any,
    __jax_enable_x64: bool) -> ArgSignature: ...

def _is_float0(__arg: Any) -> bool: ...


class ArgumentSignature:
  static_args: Sequence[Any]
  static_arg_names: Sequence[str]
  dynamic_arg_names: Sequence[str]
  dynamic_arg_treedefs: Sequence[pytree.PyTreeDef]

  def __eq__(self, value, /): ...
  def __ne__(self, value, /): ...
  def __hash__(self, /): ...
  def __str__(self): ...
  def __repr__(self): ...


def parse_arguments(
    positional_args: Sequence[Any],
    keyword_args: Sequence[Any],
    kwnames: tuple[str, ...],
    static_argnums: Sequence[int],
    static_argnames: Sequence[str],
    pytree_registry: pytree.PyTreeRegistry,
) -> tuple[ArgumentSignature, Sequence[Any]]: ...
