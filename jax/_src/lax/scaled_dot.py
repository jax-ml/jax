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
# limitations under the License.from functools import partial

from functools import partial
import jax
from jax._src import core
from jax._src import dtypes
from jax._src import numpy as jnp
from jax._src.interpreters import mlir
from jax._src.lax import lax
from jax._src.typing import Array, DTypeLike
import numpy as np


def _validate_operand_scale(side, operand, scale, contracting_dim_index):
  for i, size in enumerate(operand.shape):
    if i == contracting_dim_index:
      if size % scale.shape[i] != 0:
        raise TypeError(
            f"{side} contracting dim {i} of size {size} must be divisible by "
            f"its scale's dim size {scale.shape[i]}."
        )
      s = size // scale.shape[i]
      if s % 32 != 0:
        raise TypeError(
            f"The ratio of {side} contracting dim {i} to its scale's dim size"
            f" ({s}) must be a multiple of 32."
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
    lhs_contracting_dim_index: int,
    rhs_contracting_dim_index: int,
    lhs_batch_dim_index: int | None,
    rhs_batch_dim_index: int | None,
    preferred_element_type: DTypeLike | None,
):
  """Validates the inputs to scaled_dot."""

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

  # Check that the rank is 2 or 3.
  if lhs.ndim != 2 and lhs.ndim != 3:
    raise TypeError("All input tensors must have a rank 2 or 3.")

  if lhs_batch_dim_index is not None and rhs_batch_dim_index is not None:
    batch_dims = [
        lhs.shape[lhs_batch_dim_index],
        rhs.shape[rhs_batch_dim_index],
    ]
    if lhs_scale is not None:
      batch_dims.append(lhs_scale.shape[lhs_batch_dim_index])
    if rhs_scale is not None:
      batch_dims.append(rhs_scale.shape[rhs_batch_dim_index])
    if max(batch_dims) != min(batch_dims):
      raise TypeError(
          "All input tensors must have the same batch dimension size."
      )

  # Check contracting dimensions are the same.
  if (
      lhs.shape[lhs_contracting_dim_index]
      != rhs.shape[rhs_contracting_dim_index]
  ):
    raise TypeError(
        f"LHS contracting dim {lhs_contracting_dim_index} of size"
        f" {lhs.shape[lhs_contracting_dim_index]} does not match RHS"
        f" contracting dim {rhs_contracting_dim_index} of size"
        f" {rhs.shape[rhs_contracting_dim_index]}."
    )

  if lhs_scale is not None:
    _validate_operand_scale("LHS", lhs, lhs_scale, lhs_contracting_dim_index)

  if rhs_scale is not None:
    _validate_operand_scale("RHS", rhs, rhs_scale, rhs_contracting_dim_index)


def _scaled_dot_abstract_eval(
    lhs,
    rhs,
    lhs_scale,
    rhs_scale,
    *,
    lhs_contracting_dim_index: int,
    rhs_contracting_dim_index: int,
    lhs_batch_dim_index: int | None,
    rhs_batch_dim_index: int | None,
    preferred_element_type: DTypeLike | None = None,
):
  lhs_shape, rhs_shape = lhs.shape, rhs.shape

  batch_dims_shape = []
  if lhs_batch_dim_index is not None:
    batch_dims_shape.append(lhs_shape[lhs_batch_dim_index])

  lhs_kept = sorted(
      i
      for i in range(len(lhs_shape))
      if i != lhs_contracting_dim_index and i != lhs_batch_dim_index
  )
  rhs_kept = sorted([
      i
      for i in range(len(rhs_shape))
      if i != rhs_contracting_dim_index and i != rhs_batch_dim_index
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


def _broadcast_to_shape(
    tensor: Array,
    new_shape: tuple[int, ...],
    dim_idx: int,
) -> Array:
  """Broadcasts a tensor to a shape with a given multiplier."""
  multiplier = new_shape[dim_idx] // tensor.shape[dim_idx]
  new_broadcast_shape = list(tensor.shape)
  new_broadcast_shape.insert(dim_idx + 1, multiplier)
  tensor = jnp.expand_dims(tensor, axis=dim_idx + 1)
  tensor = jnp.broadcast_to(tensor, new_broadcast_shape)
  return tensor.reshape(new_shape)


def _broadcast_scale_to_subchannel_32(
    operand: Array,
    scale: Array,
    contracting_dim_index: int,
) -> Array:
  """Reshapes and broadcasts the scale tensor if needed to the contracing

  dimension of the operand divided by 32.
  """
  broadcast_multiplier = (
      operand.shape[contracting_dim_index] // scale.shape[contracting_dim_index]
  )
  if broadcast_multiplier % 32 == 0 and broadcast_multiplier > 1:
    new_scale_shape = list(scale.shape)
    new_scale_shape[contracting_dim_index] = (
        scale.shape[contracting_dim_index] * broadcast_multiplier // 32
    )
    scale = _broadcast_to_shape(
        scale, tuple(new_scale_shape), contracting_dim_index
    )
  return scale


# 4. Primal Implementation
@partial(lax.composite, name="xla.scaled_dot")
def _scaled_dot_impl(
    lhs: Array,
    rhs: Array,
    lhs_scale: Array,
    rhs_scale: Array,
    *,
    lhs_contracting_dim_index: int,
    rhs_contracting_dim_index: int,
    lhs_batch_dim_index: int | None,
    rhs_batch_dim_index: int | None,
    preferred_element_type: DTypeLike | None = None,
) -> Array:

  # If the lhs is bfloat16 then there is no scale was passed and we created one
  # with value 1.0 a  nd shape identical to lhs.
  if lhs.dtype != dtypes.bfloat16:
    lhs_scale = _broadcast_to_shape(
        lhs_scale, lhs.shape, lhs_contracting_dim_index
    )
    lhs = lhs.astype(dtypes.bfloat16)
    lhs_scale = lhs_scale.astype(dtypes.bfloat16)
    lhs_scaled = lhs * lhs_scale
  else:
    lhs_scaled = lhs

  # The same applies to rhs if it is bfloat16.
  if rhs.dtype != dtypes.bfloat16:
    rhs_scale = _broadcast_to_shape(
        rhs_scale, rhs.shape, rhs_contracting_dim_index
    )
    rhs = rhs.astype(dtypes.bfloat16)
    rhs_scale = rhs_scale.astype(dtypes.bfloat16)
    rhs_scaled = rhs * rhs_scale
  else:
    rhs_scaled = rhs

  dimension_numbers = (
      ((lhs_contracting_dim_index,), (rhs_contracting_dim_index,)),
      (
          (lhs_batch_dim_index,) if lhs_batch_dim_index is not None else (),
          (rhs_batch_dim_index,) if rhs_batch_dim_index is not None else (),
      ),
  )
  result = jax.lax.dot_general(
      lhs_scaled,
      rhs_scaled,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
  )

  # Cast to preferred element type if specified
  if preferred_element_type is not None:
    result = result.astype(preferred_element_type)

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

  Scales are applied along contracting dimensions. Contracting dimensions of
  `lhs` and `rhs` must be identical. For float8 inputs, operand contracting
  dimensions must be at least 32 times larger than the corresponding scale
  dimensions: `operand.shape[c] >= scale.shape[c] * 32 * S` where S is the
  integer > 0 for each contracting dimension `c`. For non-contracting
  dimensions, operand and scale dimension sizes must match.

  This function only supports one contracting dimension and at most one batch
  dimension per operand.

  Args:
    lhs: The left-hand side operand of the dot product. Must be of rank 2 or 3,
      and dtype must be one of `float8_e4m3fn`, `float8_e5m2`, or `bfloat16`.
    rhs: The right-hand side operand of the dot product. Must be of rank 2 or 3,
      and dtype must be one of `float8_e4m3fn`, `float8_e5m2`, or `bfloat16`.
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

  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  if len(lhs_contracting) > 1 or len(rhs_contracting) > 1:
    raise TypeError("Only one contracting dimension is supported.")
  if len(lhs_batch) > 1 or len(rhs_batch) > 1:
    raise TypeError("Only one batch dimension is supported.")

  # Pass the dimension numbers separately as the named arguments. As a result
  # they will be captured as the composite call attributes with the same names.
  (lhs_contracting_dim_index,) = lhs_contracting
  (rhs_contracting_dim_index,) = rhs_contracting

  lhs_batch_dim_index = lhs_batch[0] if lhs_batch else None
  rhs_batch_dim_index = rhs_batch[0] if rhs_batch else None

  _scaled_dot_validate_inputs(
      lhs,
      rhs,
      lhs_scale,
      rhs_scale,
      lhs_contracting_dim_index=lhs_contracting_dim_index,
      rhs_contracting_dim_index=rhs_contracting_dim_index,
      lhs_batch_dim_index=lhs_batch_dim_index,
      rhs_batch_dim_index=rhs_batch_dim_index,
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
    lhs_scale = _broadcast_scale_to_subchannel_32(
        lhs, lhs_scale, lhs_contracting_dim_index
    )

  if rhs_scale is None:
    if rhs.dtype != dtypes.bfloat16:
      raise ValueError("rhs_scale must be provided if rhs dtype is float8.")
    rhs_scale = jnp.ones([1] * rhs.ndim, dtype=dtypes.bfloat16)
  else:
    if rhs.dtype == dtypes.bfloat16:
      raise ValueError("rhs_scale must be None if rhs dtype is bfloat16.")
    rhs_scale = _broadcast_scale_to_subchannel_32(
        rhs, rhs_scale, rhs_contracting_dim_index
    )

  return scaled_dot_p.bind(
      lhs,
      rhs,
      lhs_scale,
      rhs_scale,
      lhs_contracting_dim_index=lhs_contracting_dim_index,
      rhs_contracting_dim_index=rhs_contracting_dim_index,
      lhs_batch_dim_index=lhs_batch_dim_index,
      rhs_batch_dim_index=rhs_batch_dim_index,
      preferred_element_type=element_type,
  )
