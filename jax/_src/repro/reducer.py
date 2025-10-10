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

from absl import app
from absl import flags
import os
from typing import Any, Callable

from jax._src.repro.tracker import (
  Func,
  ReproError,
)
from jax._src.repro.emitter import (
  EmitFuncContext,
)

def emit_call_preprocessor(ctx: EmitFuncContext,
                           func, args, kwargs, results_with_paths):


  if (not isinstance(func, Func) or
      (len(results_with_paths) == 1 and
       func.api_name != "flax.core.axes_scan.scan")):
    return None

  return func, args, kwargs, results_with_paths


def undefined_value_handler(ctx: EmitFuncContext, v: Any) -> str:
  from jax._src.random import prng
  def fake_array(a) -> str:
    if isinstance(a.dtype, prng.KeyTy):
      return f"fake_prng_key({ctx.emit_operand_atom(a.dtype._impl)}, {a.shape})"
    else:
      return f"np.ones({a.shape}, dtype={ctx.emit_operand_atom(a.dtype)})"

  if hasattr(v, "shape") and hasattr(v, "dtype"):
    return fake_array(v)

  raise ReproError(f"undefined value handler for {type(v)}: {v}")

# Delta Debugging, implementation from https://www.debuggingbook.org/html/DeltaDebugger.html
def ddmin(test: Callable[[tuple[Any, ...]], bool | None],
          inp: tuple[Any, ...]) -> tuple[Any, ...]:
  """
  Reduce `inp` to a 1-minimal failing subset, using the outcome
  of `test(inp)`, which should be True (the test is "interesting"), False
  (not interesting), or None (unresolved).
  """
  assert test(inp) == False

  n = 2  # Initial granularity
  while len(inp) >= 2:
    start: int = 0  # Where to start the next subset
    subset_length: int = int(len(inp) / n)
    some_complement_is_interesting: bool = False

    while start < len(inp):
      # Cut out inp[start:(start + subset_length)]
      complement = inp[:start] + inp[start + subset_length:]

      if test(complement) == True:
        # Continue with reduced input
        inp = complement
        n = max(n - 1, 2)
        some_complement_is_interesting = True
        break

      # Continue with next subset
      start += subset_length

    if not some_complement_is_interesting:
      # Increase granularity
      if n == len(inp):
        break
      n = min(n * 2, len(inp))

  return inp
