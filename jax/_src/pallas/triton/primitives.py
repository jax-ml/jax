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

"""Module for GPU-specific JAX primitives."""

from __future__ import annotations

import jax
from jax import core as jax_core
from jax._src.lib.triton import dialect as tt_dialect
from jax._src.pallas.triton import lowering
import jax.numpy as jnp


def approx_tanh(x: jax.typing.ArrayLike) -> jax.Array:
  r"""Elementwise approximate hyperbolic tangent: :math:`\mathrm{tanh}(x)`.

  See
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh.
  """
  return approx_tanh_p.bind(x)


approx_tanh_p = jax_core.Primitive("approx_tanh_p")


@approx_tanh_p.def_abstract_eval
def _approx_tanh_abstract_eval(
    x_aval: jax_core.ShapedArray,
) -> jax_core.ShapedArray:
  if jnp.dtype(x_aval.dtype) not in (jnp.float16, jnp.bfloat16, jnp.float32):
    raise TypeError(f"approx_tanh does not accept {x_aval.dtype} arrays")
  return x_aval


@lowering.register_lowering(approx_tanh_p)
def _approx_tanh_lowering(ctx: lowering.LoweringContext, x):
  [x_aval] = ctx.avals_in
  if x_aval.dtype == jnp.float16:
    asm = "tanh.approx.f16 $0, $1;"
    constraint = "h"
  elif x_aval.dtype == jnp.bfloat16:
    asm = "tanh.approx.bf16 $0, $1;"
    constraint = "h"
  elif x_aval.dtype == jnp.float32:
    asm = "tanh.approx.f32 $0, $1;"
    constraint = "f"
  else:
    raise NotImplementedError(f"Unsupported dtype: {x_aval.dtype}")
  return tt_dialect.elementwise_inline_asm(
      [x.type],
      asm,
      constraints=f"={constraint},{constraint}",
      pure=True,
      packed_element=1,
      args=[x],
  )
