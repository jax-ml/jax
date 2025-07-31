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

from typing import Any, Callable

import jax
import jax.numpy as jnp


def unfold_args(
    conditions: tuple[jax.Array | bool, ...],
    fn_conditions: tuple[bool, ...],
    fn: Callable[..., Any],
):
  """Minimize run-time branching of fn by converting jnp.bool to python bool."""
  if len(conditions) == 0:
    fn(*fn_conditions)
  else:
    arg = conditions[0]
    if isinstance(arg, bool):
      unfold_args(conditions[1:], fn_conditions + (arg,), fn)
    else:
      assert arg.dtype == jnp.bool and arg.size == 1
      jax.lax.cond(
          arg,
          lambda: unfold_args(conditions[1:], fn_conditions + (True,), fn),
          lambda: unfold_args(conditions[1:], fn_conditions + (False,), fn),
      )


def quantize_tensor(x: jax.Array, n_bits: int = 8, dim: int = -1):
  max_val = jnp.max(jnp.abs(x), axis=dim, keepdims=True)
  int_min = -(2 ** (n_bits - 1))
  int_max = 2 ** (n_bits - 1) - 1
  scale = max_val / int_max
  x_int = jnp.clip(jnp.rint(x / scale), int_min, int_max).astype(jnp.int8)
  return x_int, scale.astype(jnp.float32)


def next_multiple(x, multiple):
  return ((x + multiple - 1) // multiple) * multiple


def get_kernel_name(bs_block_size, out_block_size, in_block_size):
  return f'quantized_matmul_kernel_{bs_block_size}_{out_block_size}_{in_block_size}'
