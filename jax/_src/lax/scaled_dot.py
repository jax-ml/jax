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
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import lax
from jax._src.typing import Array, DTypeLike


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

  _scaled_dot_validate_inputs(
      lhs,
      rhs,
      lhs_scale,
      rhs_scale,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
  )

  (lhs_contracting, rhs_contracting), _ = dimension_numbers

  lhs_scale = _scale_broadcast(lhs_scale, lhs.shape, lhs_contracting)
  lhs = lhs.astype(dtypes.bfloat16)
  lhs_scale = lhs_scale.astype(dtypes.bfloat16)
  lhs_scaled = lhs * lhs_scale

  rhs_scale = _scale_broadcast(rhs_scale, rhs.shape, rhs_contracting)
  rhs = rhs.astype(dtypes.bfloat16)
  rhs_scale = rhs_scale.astype(dtypes.bfloat16)
  rhs_scaled = rhs * rhs_scale

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


def _create_dummy_scale(operand, contracting_dims):
  shape = list(operand.shape)
  for d in contracting_dims:
    shape[d] = 1
  return jnp.ones(shape, dtype=jnp.bfloat16).astype(dtypes.float8_e8m0fnu)


def _scaled_dot_batching_rule(
    batched_args, batch_dims, *, dimension_numbers, preferred_element_type
):
  # Unpack arguments and batch dimensions for inputs.
  lhs, rhs, lhs_scale, rhs_scale = batched_args
  lhs_bdim, rhs_bdim, lhs_scale_bdim, rhs_scale_bdim = batch_dims

  # Determine the batch size from the first argument that has a batch dimension.
  # We iterate through args and corresponding batch dims; if bdim is not None,
  # it means that argument is batched, so we take its size at that dimension.
  size = next(
      x.shape[d] for x, d in zip(batched_args, batch_dims) if d is not None
  )

  # Ensure the batch dimension is at the front (index 0) for all inputs.
  # If an input is broadcasted (bdim is None), this broadcasts it to include
  # the batch dimension at the front. If it is already batched but at a
  # different index, it moves it to 0.
  lhs = batching.bdim_at_front(lhs, lhs_bdim, size)
  rhs = batching.bdim_at_front(rhs, rhs_bdim, size)
  if lhs_scale is not None:
    lhs_scale = batching.bdim_at_front(lhs_scale, lhs_scale_bdim, size)
  if rhs_scale is not None:
    rhs_scale = batching.bdim_at_front(rhs_scale, rhs_scale_bdim, size)

  # Unpack the original dimension numbers.
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

  # Since we moved the batch dimension to index 0 for all inputs, all existing
  # dimension indices must be shifted by 1.
  lhs_contract = tuple(d + 1 for d in lhs_contract)
  rhs_contract = tuple(d + 1 for d in rhs_contract)
  lhs_batch = tuple(d + 1 for d in lhs_batch)
  rhs_batch = tuple(d + 1 for d in rhs_batch)

  # Add the new leading batch dimension (index 0) to the set of batch dimensions
  # for both LHS and RHS. This effectively batches the operation.
  new_lhs_batch = (0,) + lhs_batch
  new_rhs_batch = (0,) + rhs_batch

  # Reconstruct dimension_numbers with the shifted and new indices.
  new_dimension_numbers = (
      (lhs_contract, rhs_contract),
      (new_lhs_batch, new_rhs_batch),
  )

  # Bind the primitive with the batched operands and updated dimension numbers.
  # This creates the batched scaled_dot operation in the jaxpr.
  result = scaled_dot_p.bind(
      lhs,
      rhs,
      lhs_scale,
      rhs_scale,
      dimension_numbers=new_dimension_numbers,
      preferred_element_type=preferred_element_type,
  )

  # Return the result and the index of the batch dimension in the result (0).
  return result, 0


batching.primitive_batchers[scaled_dot_p] = _scaled_dot_batching_rule


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

  Non-contracting dimensions of the operand and scale must have the same size.
  Contracting dimension size of the operand must be an integer multiple of the
  scale contracting dimension size (subchannel size). Latency of the op depends
  on what subchannel sizes are natively supported on your platform.

  .. note::
    This currently isn't differentiable (no transpose rule).

  Example:
    ::

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

  Args:
    lhs: The left-hand side operand of the dot product.
    rhs: The right-hand side operand of the dot product.
    lhs_scale: The scale factor for `lhs`. It should be at least 2x smaller
      along the contracting dimension as compared to the operand.
    rhs_scale: The scale factor for `rhs`. It should be at least 2x smaller
      along the contracting dimension as compared to the operand.
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

  (lhs_contracting, rhs_contracting), _ = dimension_numbers

  if lhs_scale is None:
    lhs_scale = _create_dummy_scale(lhs, lhs_contracting)

  if rhs_scale is None:
    rhs_scale = _create_dummy_scale(rhs, rhs_contracting)

  element_type = (
      preferred_element_type
      if preferred_element_type is not None
      else dtypes.bfloat16
  )
  element_type = dtypes.check_and_canonicalize_user_dtype(
      element_type, "scaled_dot"
  )
  return scaled_dot_p.bind(
      lhs,
      rhs,
      lhs_scale,
      rhs_scale,
      dimension_numbers=dimension_numbers,
      preferred_element_type=element_type,
  )
