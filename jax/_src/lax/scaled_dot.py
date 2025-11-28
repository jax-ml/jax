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

from functools import partial
from typing import Sequence
import jax
from jax._src import core
from jax._src import dtypes
from jax._src import numpy as jnp
from jax._src.interpreters import mlir
from jax._src.lax import lax
from jax._src.typing import Array, DTypeLike
import numpy as np


def _validate_operand_scale(
    side, operand, scale, contracting_dims: Sequence[int]
):
  for i, size in enumerate(operand.shape):
    if i in contracting_dims:
      if size % scale.shape[i] != 0:
        raise TypeError(
            f"{side} contracting dim {i} of size {size} must be divisible by "
            f"its scale's dim size {scale.shape[i]}."
        )
      s = size // scale.shape[i]
      if s < 2:
        raise TypeError(
            f"The ratio of {side} contracting dim {i} to its scale's dim size"
            f" ({s}) must be at least 2."
        )
    elif scale.shape[i] != size:
      raise TypeError(
          f"{side} dim {i} of size {size} does not match scale dim size "
          f"{scale.shape[i]}."
      )


def _scaled_dot_validate_inputs(
    lhs: Array,
    rhs: Array,
    lhs_scale: Array | None,
    rhs_scale: Array | None,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    preferred_element_type: DTypeLike | None,
):
  """Validates the inputs to scaled_dot."""
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers

  if preferred_element_type is not None:
    if preferred_element_type not in (dtypes.bfloat16, np.float32):
      raise TypeError(
          "preferred_element_type must be one of bfloat16 or float32, got"
          f" {preferred_element_type}"
      )

  if lhs.dtype == dtypes.bfloat16:
    if lhs_scale is not None:
      raise ValueError("lhs_scale must be None if lhs dtype is bfloat16.")
  elif lhs.dtype in (dtypes.float8_e4m3fn, dtypes.float8_e5m2):
    if lhs_scale is None:
      raise ValueError("lhs_scale must be provided if lhs dtype is float8.")
  else:
    raise TypeError(
        "lhs dtype must be float8_e4m3fn, float8_e5m2 or bfloat16, got"
        f" {lhs.dtype}"
    )

  if rhs.dtype == dtypes.bfloat16:
    if rhs_scale is not None:
      raise ValueError("rhs_scale must be None if rhs dtype is bfloat16.")
  elif rhs.dtype in (dtypes.float8_e4m3fn, dtypes.float8_e5m2):
    if rhs_scale is None:
      raise ValueError("rhs_scale must be provided if rhs dtype is float8.")
  else:
    raise TypeError(
        "rhs dtype must be float8_e4m3fn, float8_e5m2 or bfloat16, got"
        f" {rhs.dtype}"
    )

  if lhs_scale is not None and lhs_scale.dtype != dtypes.float8_e8m0fnu:
    raise TypeError(
        f"lhs_scale dtype must be float8_e8m0fnu, got {lhs_scale.dtype}"
    )
  if rhs_scale is not None and rhs_scale.dtype != dtypes.float8_e8m0fnu:
    raise TypeError(
        f"rhs_scale dtype must be float8_e8m0fnu, got {rhs_scale.dtype}"
    )

  ndims = [lhs.ndim, rhs.ndim]
  if lhs_scale is not None:
    ndims.append(lhs_scale.ndim)
  if rhs_scale is not None:
    ndims.append(rhs_scale.ndim)

  if max(ndims) != min(ndims):
    raise TypeError(
        "All input tensors must have the same rank. Got lhs rank:"
        f" {lhs.ndim} rhs rank: {rhs.ndim} lhs_scale rank:"
        f" {lhs_scale.ndim if lhs_scale is not None else 'N/A'} rhs_scale"
        f" rank: {rhs_scale.ndim if rhs_scale is not None else 'N/A'}."
    )

  if len(lhs_batch) != len(rhs_batch):
    raise TypeError(
        "LHS and RHS must have the same number of batch dimensions, got"
        f" {len(lhs_batch)} and {len(rhs_batch)}."
    )
  if len(lhs_contracting) != len(rhs_contracting):
    raise TypeError(
        "LHS and RHS must have the same number of contracting dimensions, got"
        f" {len(lhs_contracting)} and {len(rhs_contracting)}."
    )

  for i_lhs, i_rhs in zip(lhs_batch, rhs_batch):
    batch_dims_sizes = [
        lhs.shape[i_lhs],
        rhs.shape[i_rhs],
    ]
    if lhs_scale is not None:
      batch_dims_sizes.append(lhs_scale.shape[i_lhs])
    if rhs_scale is not None:
      batch_dims_sizes.append(rhs_scale.shape[i_rhs])
    if max(batch_dims_sizes) != min(batch_dims_sizes):
      raise TypeError(
          "All input tensors must have the same batch dimension size for"
          f" batch dims ({i_lhs}, {i_rhs})."
      )

  # Check contracting dimensions are the same.
  for i, j in zip(lhs_contracting, rhs_contracting):
    if lhs.shape[i] != rhs.shape[j]:
      raise TypeError(
          f"LHS contracting dim {i} of size"
          f" {lhs.shape[i]} does not match RHS"
          f" contracting dim {j} of size"
          f" {rhs.shape[j]}."
      )

  if lhs_scale is not None:
    _validate_operand_scale("LHS", lhs, lhs_scale, lhs_contracting)

  if rhs_scale is not None:
    _validate_operand_scale("RHS", rhs, rhs_scale, rhs_contracting)


def _scaled_dot_abstract_eval(
    lhs,
    rhs,
    lhs_scale,
    rhs_scale,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    preferred_element_type: DTypeLike | None = None,
):
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_shape, rhs_shape = lhs.shape, rhs.shape

  batch_dims_shape = [lhs_shape[i] for i in lhs_batch]

  lhs_kept = sorted(
      i
      for i in range(len(lhs_shape))
      if i not in lhs_contracting and i not in lhs_batch
  )
  rhs_kept = sorted([
      i
      for i in range(len(rhs_shape))
      if i not in rhs_contracting and i not in rhs_batch
  ])
  output_shape = tuple(
      batch_dims_shape
      + [lhs_shape[i] for i in lhs_kept]
      + [rhs_shape[i] for i in rhs_kept]
  )

  if preferred_element_type is not None:
    output_dtype = preferred_element_type
  else:
    output_dtype = dtypes.bfloat16

  return core.ShapedArray(output_shape, output_dtype)


def _scale_broadcast(
    scale: Array,
    operand_shape: tuple[int, ...],
    contracting_dims: Sequence[int],
) -> Array:
  for i in contracting_dims:
    if scale.shape[i] != operand_shape[i]:
      multiplier = operand_shape[i] // scale.shape[i]
      new_broadcast_shape = list(scale.shape)
      new_broadcast_shape.insert(i + 1, multiplier)
      scale = jnp.expand_dims(scale, axis=i + 1)
      scale = jnp.broadcast_to(scale, new_broadcast_shape)
      new_reshape_shape = list(scale.shape)
      new_reshape_shape[i] = new_reshape_shape[i] * new_reshape_shape[i + 1]
      new_reshape_shape.pop(i + 1)
      scale = scale.reshape(new_reshape_shape)
  return scale


# 4. Primal Implementation
@partial(lax.composite, name="xla.scaled_dot")
def _scaled_dot_impl(
    lhs: Array,
    rhs: Array,
    lhs_scale: Array,
    rhs_scale: Array,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    preferred_element_type: DTypeLike | None = None,
) -> Array:
  """Implementation of scaled_dot that could be replaced by XLA."""

  (lhs_contracting, rhs_contracting), _ = dimension_numbers

  # If lhs is not bfloat16, broadcast lhs_scale to lhs.shape along
  # lhs_contracting, cast lhs to bfloat16 and multiply them.
  if lhs.dtype != dtypes.bfloat16:
    lhs_scale = _scale_broadcast(lhs_scale, lhs.shape, lhs_contracting)
    lhs = lhs.astype(dtypes.bfloat16)
    lhs_scale = lhs_scale.astype(dtypes.bfloat16)
    lhs_scaled = lhs * lhs_scale
  else:
    # If lhs is bfloat16, just use lhs and lhs_scale directly.
    lhs_scaled = lhs

  # If rhs is not bfloat16, broadcast rhs_scale to rhs.shape along
  # rhs_contracting, cast rhs to bfloat16 and multiply them.
  if rhs.dtype != dtypes.bfloat16:
    rhs_scale = _scale_broadcast(rhs_scale, rhs.shape, rhs_contracting)
    rhs = rhs.astype(dtypes.bfloat16)
    rhs_scale = rhs_scale.astype(dtypes.bfloat16)
    rhs_scaled = rhs * rhs_scale
  else:
    # If rhs is bfloat16, just use rhs and rhs_scale directly.
    rhs_scaled = rhs

  result = jax.lax.dot_general(
      lhs_scaled,
      rhs_scaled,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
  )

  return result


scaled_dot_p = core.Primitive("scaled_dot")
scaled_dot_p.def_abstract_eval(_scaled_dot_abstract_eval)

scaled_dot_lowering = mlir.lower_fun(_scaled_dot_impl, multiple_results=False)
mlir.register_lowering(scaled_dot_p, scaled_dot_lowering)
scaled_dot_p.def_impl(_scaled_dot_impl)


def scaled_dot(
    lhs: Array,
    rhs: Array,
    *,
    lhs_scale: Array | None = None,
    rhs_scale: Array | None = None,
    dimension_numbers: lax.DotDimensionNumbers | None = None,
    preferred_element_type: DTypeLike | None = None,
):
  """Computes a scaled dot product.

  This function computes `(lhs * lhs_scale) @ (rhs * rhs_scale)` in
  `preferred_element_type` precision, where `@` denotes `jax.lax.dot_general`.
  If `lhs` or `rhs` are of type `bfloat16`, their corresponding scales must
  be `None`, and they are used directly in the computation. If `lhs` or `rhs`
  are  of type `dtypes.float8_e4m3fn` or `float8_e5m2`, their corresponding
  scales must be provided with dtype `float8_e8m0fnu`.

  Non-contracting dimensions of the operand and scale must have the same size.
  Contracting dimension size of the operand must be an integer multiple of the
  scale contracting dimension size (subchannel size). Latency of the op depends
  on what subchannel sizes are natively supported on your platform.

  Example:
    B = 32
    M = 16384
    N = 16
    K = 4096
    subchannel_size = 32

    lhs_shape = (B, M, K)
    rhs_shape = (B, K, N)
    lhs_scales_shape = (B, M, K // subchannel_size)
    rhs_scales_shape = (B, K // subchannel_size, N)

    key = jax.random.key(42)

    lhs = jax.random.normal(key, lhs_shape, dtype=jnp.float8_e4m3fn)
    rhs = jax.random.normal(key, rhs_shape, dtype=jnp.float8_e4m3fn)
    lhs_scales = jax.random.normal(
        key, lhs_scales_shape, dtype=jnp.float8_e8m0fnu
    )
    rhs_scales = jax.random.normal(
        key, rhs_scales_shape, dtype=jnp.float8_e8m0fnu
    )

    @jax.jit
    def scaled_dot_fn(lhs, rhs, lhs_scale, rhs_scale):
      return jax.lax.scaled_dot(
          lhs,
          rhs,
          lhs_scale=lhs_scale,
          rhs_scale=rhs_scale,
          preferred_element_type=jnp.bfloat16,
      )

    result = scaled_dot_fn(
        lhs,
        rhs,
        lhs_scale=lhs_scales,
        rhs_scale=rhs_scales,
    )

  This function only supports one contracting dimension and at most one batch
  dimension per operand.

  Args:
    lhs: The left-hand side operand of the dot product. dtype must be one of
      `float8_e4m3fn`, `float8_e5m2`, or `bfloat16`.
    rhs: The right-hand side operand of the dot product.dtype must be one of
      `float8_e4m3fn`, `float8_e5m2`, or `bfloat16`.
    lhs_scale: The scale factor for `lhs`. Must be provided if `lhs` is float8,
      and must be `None` if `lhs` is bfloat16. If provided, must be of dtype
      `float8_e8m0fnu` and have the same rank as `lhs`.
    rhs_scale: The scale factor for `rhs`. Must be provided if `rhs` is float8,
      and must be `None` if `rhs` is bfloat16. If provided, must be of dtype
      `float8_e8m0fnu` and have the same rank as `rhs`.
    dimension_numbers: A tuple of tuples of the form `((lhs_contracting_dims,
      rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))`. If not
      provided, default is `(((1,), (0,)), ((), ()))` for 2D inputs which is
      lhs_contracting_dim=1, rhs_contracting_dim=0, and `(((2,), (1,)), ((0,),
      (0,)))` for 3D inputs which is lhs_contracting_dim=2,
      rhs_contracting_dim=1 and lhs_batch_dim=0, rhs_batch_dim=0.
    preferred_element_type: The desired dtype of the output and intermediate
      accumulations, can be `bfloat16` or `float32`. Defaults to `bfloat16`.

  Returns:
    The result of the scaled dot product.
  """

  # Syntax sugar for dimension numbers it allows for None to be passed for the
  # default case.
  if dimension_numbers is None:
    if lhs.ndim == 3:
      dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    else:
      dimension_numbers = (((1,), (0,)), ((), ()))

  _scaled_dot_validate_inputs(
      lhs,
      rhs,
      lhs_scale,
      rhs_scale,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
  )

  element_type = (
      preferred_element_type
      if preferred_element_type is not None
      else dtypes.bfloat16
  )
  element_type = dtypes.check_and_canonicalize_user_dtype(
      element_type, "scaled_dot"
  )
  if lhs_scale is None:
    if lhs.dtype != dtypes.bfloat16:
      raise ValueError("lhs_scale must be provided if lhs dtype is float8.")
    lhs_scale = jnp.ones([1] * lhs.ndim, dtype=dtypes.bfloat16)
  else:
    if lhs.dtype == dtypes.bfloat16:
      raise ValueError("lhs_scale must be None if lhs dtype is bfloat16.")

  if rhs_scale is None:
    if rhs.dtype != dtypes.bfloat16:
      raise ValueError("rhs_scale must be provided if rhs dtype is float8.")
    rhs_scale = jnp.ones([1] * rhs.ndim, dtype=dtypes.bfloat16)
  else:
    if rhs.dtype == dtypes.bfloat16:
      raise ValueError("rhs_scale must be None if rhs dtype is bfloat16.")

  return scaled_dot_p.bind(
      lhs,
      rhs,
      lhs_scale,
      rhs_scale,
      dimension_numbers=dimension_numbers,
      preferred_element_type=element_type,
  )
