# Copyright 2023 The JAX Authors.
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

"""Pallas utility functions."""

from __future__ import annotations
from typing import overload

import jax
from jax import lax
from jax._src import core as jax_core
from jax._src.util import split_list
import jax.numpy as jnp
import numpy as np


def when(condition):
  def _wrapped(f):
    if isinstance(condition, bool):
      if condition:
        f()
    else:
      lax.cond(condition, f, lambda: None)
  return _wrapped

@overload
def cdiv(a: int, b: int) -> int:
  ...

@overload
def cdiv(a: int, b: jax.Array) -> jax.Array:
  ...

@overload
def cdiv(a: jax.Array, b: int) -> jax.Array:
  ...

@overload
def cdiv(a: jax.Array, b: jax.Array) -> jax.Array:
  ...

def cdiv(a: int | jax.Array, b: int | jax.Array) -> int | jax.Array:
  if isinstance(a, int) and isinstance(b, int):
    return (a + b - 1) // b
  return lax.div(a + b - 1, b)


def strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
  size = np.prod(shape)
  strides = []
  for s in shape:
    size = size // s
    strides.append(int(size))
  return tuple(strides)


def next_power_of_2(x: int) -> int:
  """Returns the next power of two greater than or equal to `x`."""
  if x < 0:
    raise ValueError("`next_power_of_2` requires a non-negative integer.")
  return 1 if x == 0 else 2 ** (x - 1).bit_length()

def dtype_bitwidth(dtype: np.dtype | jnp.dtype) -> int:
  if isinstance(dtype, jnp.integer):
    return jnp.iinfo(dtype).bits
  return np.dtype(dtype).itemsize * 8

def pattern_match_scan_to_fori_loop(
    jaxpr: jax_core.Jaxpr, num_consts: int, num_carry: int
) -> tuple[jax_core.Jaxpr, bool]:
  if num_carry > 0:
    # Pattern match onto fori_loop:
    # We expect the first carry argument to the jaxpr to be the loop index and
    # for the loop index + 1 to be returned as the first value out of the loop.
    in_index_var = jaxpr.invars[num_consts]
    out_index_var = jaxpr.outvars[0]
    # Check that the loop index argument is an int32 scalar
    if (in_index_var.aval.shape or
        in_index_var.aval.dtype not in (jnp.int32, jnp.int64)):
      raise NotImplementedError(
          f"not a fori_loop index in: {in_index_var.aval} {jaxpr=}")
    if (out_index_var.aval.shape or
        out_index_var.aval.dtype not in (jnp.int32, jnp.int64)):
      raise NotImplementedError(
          f"not a fori_loop index out: {out_index_var.aval} {jaxpr=}")
    # Look for the equation that increments the loop index
    for i, eqn in enumerate(jaxpr.eqns):
      if eqn.primitive == lax.add_p:
        if eqn.invars[0] == in_index_var:
          if isinstance(eqn.invars[1], jax_core.Literal):
            if eqn.invars[1].val == 1:
              if eqn.outvars[0] == out_index_var:
                eqn_index = i
                break
    else:
      raise NotImplementedError("Unable to match fori_loop pattern")
    # Delete the equation that increments and remove the loop index from the
    # output. Incrementing the loop index will be done implicitly.
    jaxpr = jaxpr.replace(
        eqns=jaxpr.eqns[:eqn_index] + jaxpr.eqns[eqn_index + 1:],
        outvars=jaxpr.outvars[1:])
    has_loop_index = True
  else:
    # If there's no carry, the loop index has been DCEd and the body does *not*
    # expect a loop index as an argument.
    has_loop_index = False
  return jaxpr, has_loop_index


def pattern_match_while_to_fori_loop(
    cond_jaxpr: jax_core.Jaxpr,
    cond_nconsts: int,
    body_jaxpr: jax_core.Jaxpr,
    body_nconsts: int,
) -> tuple[jax_core.Jaxpr | None, str | None]:
  # Try to pattern match to fori loop.
  # Successful matches produce (jaxpr, None), while failures use the str
  # component of the return tuple to capture information about the failure.
  if cond_nconsts:
    return (None, "Conditional jaxpr can't contain consts.")
  _, cond_invars = split_list(cond_jaxpr.jaxpr.invars, [cond_nconsts])
  cond_in_avals = [v.aval for v in cond_invars]
  if len(cond_in_avals) < 2:
    return (None, "Conditional jaxpr have only two carry args.")
  # Check that the first two carry values are scalar ints
  a1, a2 = cond_in_avals[:2]
  if a1.shape or a1.dtype not in (jnp.int32, jnp.int64):
    return (None, "First conditional jaxpr carry arg is not a scalar int.")
  if a2.shape or a2.dtype not in (jnp.int32, jnp.int64):
    return (None, "Second conditional jaxpr carry arg is not a scalar int.")
  # Check that the only eqn in the cond checks the loop index condition
  v1, v2 = cond_invars[:2]
  outvar = cond_jaxpr.jaxpr.outvars[0]
  assert outvar.aval.dtype == jnp.bool_
  if len(cond_jaxpr.jaxpr.eqns) != 1:
    return (None, "Non-trivial conditional jaxprs not supported.")
  eqn = cond_jaxpr.jaxpr.eqns[0]
  if eqn.primitive != lax.lt_p:
    return (None, "Non-trivial conditional jaxprs not supported.")
  if eqn.outvars != [outvar]:
    return (None, "Non-trivial conditional jaxprs not supported.")
  if eqn.invars != [v1, v2]:
    return (None, "Non-trivial conditional jaxprs not supported.")
  # Check that the carry is updated in the body appropriately
  _, body_invars = split_list(body_jaxpr.jaxpr.invars, [body_nconsts])
  v1, v2 = body_invars[:2]
  vo1, vo2 = body_jaxpr.jaxpr.outvars[:2]
  # Upper bound should be constant
  if v2 is not vo2:
    return (None, "Loop upper bound is not constant.")
  # Check that we increment the loop index in the body
  for i, eqn in enumerate(body_jaxpr.jaxpr.eqns):
    if eqn.primitive is lax.add_p:
      if eqn.invars[0] is v1:
        if isinstance(eqn.invars[1], jax_core.Literal):
          if eqn.invars[1].val == 1:
            if eqn.outvars[0] == vo1:
              eqn_index = i
              break
  else:
    return (None, "Loop index not incremented in body.")
  jaxpr = body_jaxpr.jaxpr
  new_invars = (
      *jaxpr.invars[:body_nconsts],
      jaxpr.invars[body_nconsts],
      *jaxpr.invars[body_nconsts + 2 :],
  )
  new_outvars = tuple(jaxpr.outvars[2:])
  jaxpr = jaxpr.replace(
      eqns=jaxpr.eqns[:eqn_index] + jaxpr.eqns[eqn_index + 1 :],
      invars=new_invars,
      outvars=new_outvars,
  )
  return jaxpr, None


# based on https://github.com/openxla/xla/blob/a7a09d56c3599123f8148bbf3e44c9ebc04624b9/xla/mlir_hlo/mhlo/transforms/chlo_legalize_to_hlo/chlo_legalize_to_hlo.cc#L644-L802
def erf_inv_32_lowering_helper(x):
  k_degree = 9
  w_lt_5_constants = [
    2.81022636e-08,  3.43273939e-07, -3.5233877e-06,
    -4.39150654e-06, 0.00021858087,  -0.00125372503,
    -0.00417768164,  0.246640727,    1.50140941,
  ]
  w_gt_5_constants = [
    -0.000200214257, 0.000100950558, 0.00134934322,
    -0.00367342844,  0.00573950773,  -0.0076224613,
    0.00943887047,   1.00167406,     2.83297682,
  ]

  w = -jnp.log1p(x * -x)
  w_lt_5 = w < 5.0

  w = jnp.where(w_lt_5, w - 2.5, jnp.sqrt(w) - 3.0)

  p = jnp.where(w_lt_5, w_lt_5_constants[0], w_gt_5_constants[0])
  for i in range(1, k_degree):
    c = jnp.where(w_lt_5, w_lt_5_constants[i], w_gt_5_constants[i])
    p = c + p * w

  return jnp.where(jnp.abs(x) == 1.0, jnp.inf * x, p * x)


def sign_lowering_helper(x):
  if jnp.issubdtype(x.dtype, jnp.unsignedinteger):
    return (x != 0).astype(x.dtype)

  if jnp.issubdtype(x.dtype, jnp.integer):
    return (x > 0).astype(x.dtype) - (x < 0).astype(x.dtype)

  if jnp.issubdtype(x.dtype, jnp.floating):
    out = (x > 0.).astype(x.dtype) - (x < 0.).astype(x.dtype)
    return jnp.where(jnp.isnan(x), jnp.nan, out)

  raise NotImplementedError(f"sign_lowering_helper not implemented for {x.dtype}")
