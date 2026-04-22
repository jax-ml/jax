# Copyright 2026 The JAX Authors.
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

"""Registers upstream MLIR dialects used by JAX."""

from collections.abc import Callable, Sequence
from jax.jaxlib._jax import Traceback
from mlir import ir

def register_dialects(arg: ir.DialectRegistry, /) -> None: ...
def enter_multi_threaded_execution(arg: ir.Context, /) -> None: ...
def exit_multi_threaded_execution(arg: ir.Context, /) -> None: ...
def inlined_func_call(
    callee: ir.Operation,
    args: Sequence[ir.Value],
    block: ir.Block,
    loc: ir.Location | None = ...,
) -> list[ir.Value]:
  """Makes an inlined call to a function containing a single block with a single return op."""

def arith_constant(value: int | float | bool, type: ir.Type, /) -> ir.Value:
  """Creates an arith.constant operation."""

class TracebackToLocationCache:
  def __init__(
      self,
      code_to_filename: Callable,
      frame_limit: int,
      context: ir.Context | None = ...,
  ) -> None: ...
  def get(self, traceback: Traceback, /) -> ir.Location: ...
