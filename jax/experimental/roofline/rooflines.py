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
from collections import defaultdict
from dataclasses import replace
import itertools as it
from collections.abc import Sequence
import numpy as np

from jax._src import api
from jax._src import ad_checkpoint
from jax._src import ad_util
from jax._src import core, util
from jax._src import dispatch
from jax._src import ops
from jax._src import pjit
from jax._src import prng
from jax._src import random
from jax._src import shard_map
from jax._src.lax import (
  ann,
  control_flow,
  convolution,
  fft,
  lax,
  linalg,
  parallel as lax_parallel,
  slicing,
  special,
  windowed_reductions,
)
from jax.experimental import roofline

# One FMA (Fused Multiply Add) takes 2 flops to compute.
_FMA_FLOPS_FACTOR = 2

for prim in it.chain(
  ad_checkpoint.__dict__.values(),
  ad_util.__dict__.values(),
  ann.__dict__.values(),
  control_flow.__dict__.values(),
  convolution.__dict__.values(),
  dispatch.__dict__.values(),
  fft.__dict__.values(),
  lax.__dict__.values(),
  linalg.__dict__.values(),
  ops.__dict__.values(),
  [pjit.sharding_constraint_p],
  prng.__dict__.values(),
  random.__dict__.values(),
  shard_map.__dict__.values(),
  slicing.__dict__.values(),
  special.__dict__.values(),
  windowed_reductions.__dict__.values(),
):
  if isinstance(prim, core.Primitive):
    roofline.register_standard_roofline(prim)


def _unary_p_roofline(
    ctx: roofline.RooflineRuleContext,
    *args,
    **kw,
) -> roofline.RooflineResult:
  (x,) = (roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in)
  out = roofline.RooflineShape.from_aval(ctx.avals_out[0])
  return roofline.RooflineResult(
      unfused_flops=x.size,
      unfused_hbm_bytes=(
          x.dtype.itemsize * x.size + out.dtype.itemsize * out.size
      ),
  )

roofline.register_roofline(lax.abs_p)(_unary_p_roofline)
roofline.register_roofline(lax.acos_p)(_unary_p_roofline)
roofline.register_roofline(lax.asin_p)(_unary_p_roofline)
roofline.register_roofline(lax.atan_p)(_unary_p_roofline)
roofline.register_roofline(lax.cbrt_p)(_unary_p_roofline)
roofline.register_roofline(lax.ceil_p)(_unary_p_roofline)
roofline.register_roofline(lax.conj_p)(_unary_p_roofline)
roofline.register_roofline(lax.cos_p)(_unary_p_roofline)
roofline.register_roofline(lax.cosh_p)(_unary_p_roofline)
roofline.register_roofline(lax.exp_p)(_unary_p_roofline)
roofline.register_roofline(lax.expm1_p)(_unary_p_roofline)
roofline.register_roofline(lax.floor_p)(_unary_p_roofline)
roofline.register_roofline(lax.imag_p)(_unary_p_roofline)
roofline.register_roofline(lax.integer_pow_p)(_unary_p_roofline)
roofline.register_roofline(lax.is_finite_p)(_unary_p_roofline)
roofline.register_roofline(lax.log_p)(_unary_p_roofline)
roofline.register_roofline(lax.log1p_p)(_unary_p_roofline)
roofline.register_roofline(lax.logistic_p)(_unary_p_roofline)
roofline.register_roofline(lax.neg_p)(_unary_p_roofline)
roofline.register_roofline(lax.not_p)(_unary_p_roofline)
roofline.register_roofline(lax.real_p)(_unary_p_roofline)
roofline.register_roofline(lax.round_p)(_unary_p_roofline)
roofline.register_roofline(lax.rsqrt_p)(_unary_p_roofline)
roofline.register_roofline(lax.sign_p)(_unary_p_roofline)
roofline.register_roofline(lax.sin_p)(_unary_p_roofline)
roofline.register_roofline(lax.sinh_p)(_unary_p_roofline)
roofline.register_roofline(lax.sqrt_p)(_unary_p_roofline)
roofline.register_roofline(lax.square_p)(_unary_p_roofline)
roofline.register_roofline(lax.tan_p)(_unary_p_roofline)
roofline.register_roofline(special.bessel_i0e_p)(_unary_p_roofline)
roofline.register_roofline(special.bessel_i1e_p)(_unary_p_roofline)
roofline.register_roofline(special.digamma_p)(_unary_p_roofline)
roofline.register_roofline(special.erf_inv_p)(_unary_p_roofline)
roofline.register_roofline(special.erf_p)(_unary_p_roofline)
roofline.register_roofline(special.erfc_p)(_unary_p_roofline)
roofline.register_roofline(special.lgamma_p)(_unary_p_roofline)

roofline.register_standard_roofline(core.pvary_p)

def _binary_p_roofline(
    ctx: roofline.RooflineRuleContext,
    *args,
    **kw,
) -> roofline.RooflineResult:
  lhs, rhs = (roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in)
  broadcasted_shape = [
      max(l, r) for l, r in it.zip_longest(lhs.shape, rhs.shape, fillvalue=1)
  ]
  out = roofline.RooflineShape.from_aval(ctx.avals_out[0])
  return roofline.RooflineResult(
      unfused_flops=int(np.prod(broadcasted_shape)),
      unfused_hbm_bytes=(
          lhs.dtype.itemsize * lhs.size
          + rhs.dtype.itemsize * rhs.size
          + out.dtype.itemsize * out.size
      ),
  )


roofline.register_roofline(lax.add_p)(_binary_p_roofline)
roofline.register_roofline(lax.sub_p)(_binary_p_roofline)
roofline.register_roofline(lax.mul_p)(_binary_p_roofline)
roofline.register_roofline(lax.div_p)(_binary_p_roofline)
roofline.register_roofline(lax.rem_p)(_binary_p_roofline)
roofline.register_roofline(lax.and_p)(_binary_p_roofline)
roofline.register_roofline(lax.or_p)(_binary_p_roofline)
roofline.register_roofline(lax.xor_p)(_binary_p_roofline)
roofline.register_roofline(lax.gt_p)(_binary_p_roofline)
roofline.register_roofline(lax.lt_p)(_binary_p_roofline)
roofline.register_roofline(lax.ge_p)(_binary_p_roofline)
roofline.register_roofline(lax.le_p)(_binary_p_roofline)
roofline.register_roofline(lax.eq_p)(_binary_p_roofline)
roofline.register_roofline(lax.ne_p)(_binary_p_roofline)
roofline.register_roofline(lax.min_p)(_binary_p_roofline)
roofline.register_roofline(lax.max_p)(_binary_p_roofline)

def _cumulative_p_roofline(
    ctx: roofline.RooflineRuleContext,
    *args,
    axis: int,
    **kw,
) -> roofline.RooflineResult:
  (x,) = (roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in)
  out = roofline.RooflineShape.from_aval(ctx.avals_out[0])
  return roofline.RooflineResult(
      # `cum{max, min, prod, sum}` only calculate values for one axis.
      unfused_flops=x.shape[axis],
      unfused_hbm_bytes=(
          x.dtype.itemsize * x.size + out.dtype.itemsize * out.size
      ),
  )

roofline.register_roofline(control_flow.cummax_p)(_cumulative_p_roofline)
roofline.register_roofline(control_flow.cummin_p)(_cumulative_p_roofline)
roofline.register_roofline(control_flow.cumprod_p)(_cumulative_p_roofline)
roofline.register_roofline(control_flow.cumsum_p)(_cumulative_p_roofline)

@roofline.register_roofline(control_flow.cumlogsumexp_p)
def _cumlogsumexp_p_roofline(
    ctx: roofline.RooflineRuleContext,
    *args,
    axis: int,
    **kw,
) -> roofline.RooflineResult:
  (x,) = (roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in)
  out = roofline.RooflineShape.from_aval(ctx.avals_out[0])
  return roofline.RooflineResult(
      # Similar to `cum{max, min, prod, sum}`, `cumlogsumexp` only calculates
      # values for one axis. But for `x.shape[axis] = S`, it computes (for a
      # naive implementation):
      #   S `exp` ops.
      #   S-1 `add` ops.
      #   1 log op.
      # Thus, the total number of flops is 2 * S.
      unfused_flops=x.shape[axis] * 2,
      unfused_hbm_bytes=(
          x.dtype.itemsize * x.size + out.dtype.itemsize * out.size
      ),
  )


@roofline.register_roofline(lax.dot_general_p)
def _dot_general_roofline(
  ctx: roofline.RooflineRuleContext,
  *args,
  dimension_numbers: lax.DotDimensionNumbers,
  **kw,
) -> roofline.RooflineResult:
  lhs, rhs = (roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in)
  out = roofline.RooflineShape.from_aval(ctx.avals_out[0])
  (lhs_contract, _), (lhs_batch, _) = dimension_numbers

  flops = (
    _FMA_FLOPS_FACTOR
    * lhs.size
    * rhs.size
    / np.prod([lhs.shape[i] for i in lhs_contract])
    / np.prod([lhs.shape[i] for i in lhs_batch])
  )

  hbm_bytes = 0
  if not ctx.pin_lhs_in_vmem:
    hbm_bytes += lhs.bytes
    hbm_bytes += out.bytes
  if not ctx.pin_rhs_in_vmem:
    hbm_bytes += rhs.bytes

  return roofline.RooflineResult(
      flops=int(flops),
      unfused_flops=int(flops),
      hbm_bytes=hbm_bytes,
      unfused_hbm_bytes=hbm_bytes,
  )


def _get_spatial_valid_position_count_for_one_dim(
    window_dim_stride: int,
    base_dilation: int,
    window_dilation: int,
    kernel_limit: int,
    input_limit: int,
    output_limit: int,
    padding: tuple[int, int],
) -> int:
  """Gets the valid position count for conv for a single spatial dimension.

  Args:
    window_dim_stride: The stride of the window along this dimension.
    base_dilation: The base dilation factor along this dimension.
    window_dilation: The window dilation factor along this dimension.
    kernel_limit: The size of the kernel along this dimension.
    input_limit: The size of the input along this dimension.
    output_limit: The size of the output along this dimension.
    padding: The padding applied to the input along this dimension.
  """
  padding_low = padding[0]
  padding_high = padding[1]

  # These two conditions will create an N^2 iteration pattern with only N
  # valid elements. This is a performance optimization and produces the same
  # result as the whole loop.
  if (
      input_limit == output_limit
      and kernel_limit == output_limit
      and input_limit == base_dilation
      and window_dilation == 1
      and max(1, input_limit - 1) == window_dim_stride
      and padding_low == 0
      and padding_high == 0
  ):
    return input_limit

  if (
      input_limit == 1
      and kernel_limit == output_limit
      and window_dilation == 1
      and base_dilation == 1
      and window_dim_stride == 1
      and padding_low == output_limit - 1
      and padding_high == output_limit - 1
  ):
    return output_limit

  valid_position_count = 0
  # Loop over each point in the kernel
  for kernel_idx in range(kernel_limit):

    # Skip loop for trivial stride and base_dilation
    if window_dim_stride == 1 and base_dilation == 1:
      undilated_index_base = padding_low - kernel_idx * window_dilation
      upper_limit = min(
          input_limit + undilated_index_base,
          output_limit,
      )
      lower_limit = max(0, undilated_index_base)

      valid_position_count += max(upper_limit - lower_limit, 0)
      continue

    # Loop over each point in the output
    for output_idx in range(output_limit):
      # Calculate lhs (input) index without taking base dilation into account
      undilated_index = (
          output_idx * window_dim_stride
          - padding_low
          + kernel_idx * window_dilation
      )
      # Calculate the actual lhs (input) index after dilation
      lhs_spatial_index = int(undilated_index / base_dilation)

      # Skip if the lhs (input) index is to be dilated.
      if undilated_index != lhs_spatial_index * base_dilation:
        continue
      # Skip if input index is not in bound.
      if lhs_spatial_index < 0 or lhs_spatial_index >= input_limit:
        continue

      valid_position_count += 1
  return valid_position_count


def _get_spatial_valid_position_count(
    dnums: convolution.ConvDimensionNumbers,
    lhs: roofline.RooflineShape,
    rhs: roofline.RooflineShape,
    out: roofline.RooflineShape,
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int],
    rhs_dilation: Sequence[int],
) -> int:
  """Gets the number of valid spatial positions for conv_general_dilated.

  Args:
    dnums: The dimension numbers for the convolution.
    lhs: The shape of the left-hand side of the convolution.
    rhs: The shape of the right-hand side of the convolution.
    out: The shape of the output of the convolution.
    window_strides: The stride of the window along each spatial dimension.
    padding: The padding applied to the input along each spatial dimension.
    lhs_dilation: The dilation factor for the left-hand side along each spatial
      dimension.
    rhs_dilation: The dilation factor for the right-hand side along each spatial
      dimension.
  """
  input_spatial_dims, kernel_spatial_dims, out_spatial_dims = (
      dnums.lhs_spec[2:],
      dnums.rhs_spec[2:],
      dnums.out_spec[2:],
  )

  valid_position_counts = 1
  # Loop over each spatial dimension and determine how many valid positions
  # there are for each dimension.
  for d in range(len(input_spatial_dims)):
    valid_position_counts *= _get_spatial_valid_position_count_for_one_dim(
        window_dim_stride=window_strides[d],
        base_dilation=lhs_dilation[d],
        window_dilation=rhs_dilation[d],
        kernel_limit=rhs.shape[kernel_spatial_dims[d]],
        input_limit=lhs.shape[input_spatial_dims[d]],
        output_limit=out.shape[out_spatial_dims[d]],
        padding=padding[d],
    )

  return valid_position_counts


def _calculate_conv_flops(
    lhs: roofline.RooflineShape,
    rhs: roofline.RooflineShape,
    out: roofline.RooflineShape,
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int],
    rhs_dilation: Sequence[int],
    dimension_numbers: convolution.ConvGeneralDilatedDimensionNumbers,
    batch_group_count: int,
) -> int:
  """Calculates roofline unfused flops for Jax's conv_general_dilated primitive.

  See `jax.lax.conv_general_dilated` for details on the arguments.
  """
  dnums = convolution.conv_dimension_numbers(
      lhs.shape, rhs.shape, dimension_numbers
  )

  spatial_valid_position_counts = _get_spatial_valid_position_count(
      dnums, lhs, rhs, out, window_strides, padding, lhs_dilation, rhs_dilation
  )

  batch = lhs.shape[dnums.lhs_spec[0]]
  num_output_features = out.shape[dnums.out_spec[1]]
  num_input_features = rhs.shape[dnums.rhs_spec[1]]
  num_output_batch = batch / batch_group_count

  non_spatial_dims_factor = (
      num_input_features * num_output_features * num_output_batch
  )

  fma_count = non_spatial_dims_factor * spatial_valid_position_counts
  flops = fma_count * _FMA_FLOPS_FACTOR
  return int(flops)


@roofline.register_roofline(convolution.conv_general_dilated_p)
def _conv_general_dilated_roofline(
    ctx: roofline.RooflineRuleContext,
    *args,
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int],
    rhs_dilation: Sequence[int],
    dimension_numbers: convolution.ConvGeneralDilatedDimensionNumbers,
    batch_group_count: int,
    **kw,
) -> roofline.RooflineResult:
  """Roofline for Jax's conv_general_dilated primitive.

  See `jax.lax.conv_general_dilated` for details on the arguments.
  """
  lhs, rhs = (roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in)
  out = roofline.RooflineShape.from_aval(ctx.avals_out[0])

  return roofline.RooflineResult(
      unfused_flops=_calculate_conv_flops(
          lhs,
          rhs,
          out,
          window_strides,
          padding,
          lhs_dilation,
          rhs_dilation,
          dimension_numbers,
          batch_group_count,
      ),
      unfused_hbm_bytes=(
          lhs.dtype.itemsize * lhs.size
          + rhs.dtype.itemsize * rhs.size
          + out.dtype.itemsize * out.size
      ),
  )


def _return_zeros_if_one_sized_axis(
  ctx: roofline.RooflineRuleContext, axes: tuple[str, ...]
) -> roofline.RooflineResult | None:
  assert ctx.mesh
  axes_size = np.prod([ctx.mesh.shape[axis] for axis in axes])
  if axes_size > 1:
    return None
  return roofline.RooflineResult(
    ici_bytes={axis: 0 for axis in axes},
    ici_latency={axis: 0 for axis in axes},
  )


def _ring_collective_roofline(
  ctx: roofline.RooflineRuleContext,
  *args,
  axes: tuple[str, ...],
  is_reduce: bool = True,
  **kw,
) -> roofline.RooflineResult:
  if zeros_result := _return_zeros_if_one_sized_axis(ctx, axes):
    return zeros_result

  assert ctx.mesh
  mesh = ctx.mesh.shape
  current_shard_size = roofline.RooflineShape.total_bytes(ctx.avals_in)
  if is_reduce:
    current_shard_size /= np.prod([mesh[axis] for axis in axes])

  # We model the slowest color as the bottleneck.
  sorted_axes = sorted(axes, key=lambda x: mesh[x], reverse=True)
  num_axes = len(sorted_axes)

  ici_bytes = 0
  # Phase split.
  current_shard_size //= num_axes
  for axis in sorted_axes:
    axis_size = mesh[axis]
    # Do phase.
    ici_bytes += current_shard_size * (axis_size - 1)
    # Increase shard size.
    current_shard_size *= axis_size

  # Bottleneck is the longest axis.
  ici_latency = mesh[sorted_axes[0]] * num_axes

  return roofline.RooflineResult(
    ici_bytes={axis: int(ici_bytes) for axis in sorted_axes},
    ici_latency={axis: int(ici_latency) for axis in sorted_axes},
  )


roofline.register_roofline(lax_parallel.reduce_scatter_p)(
  lambda *args, axis_name, **kw: _ring_collective_roofline(*args, axes=axis_name, **kw)
)
roofline.register_roofline(lax_parallel.all_gather_p)(
  lambda *args, axis_name, **kw: _ring_collective_roofline(
    *args, axes=axis_name, is_reduce=False, **kw
  )
)


def _calculate_gather_flops(
    mode: slicing.GatherScatterMode,
    indices_size: int,
    output_size: int,
) -> int:
  """Calculates roofline unfused flops for Jax's gather primitive."""

  if mode == slicing.GatherScatterMode.FILL_OR_DROP:
    # With FILL_OR_DROP, we have 4 steps to check whether to fill (or drop):
    # 1. Check if the index is within upper bound.
    # 2. Check if the index is within lower bound.
    # 3. Call `and` on #1 and #2 to check the index is "in bounds".
    # 4. `reduce` the result to a single boolean per window.
    # Each of the steps is a single elementwise op on the indices.
    index_check_flops = indices_size * 4

    # Once we know whether to fill or drop (per window), there are 2 steps to
    # mask the output:
    # 1. Broadcast the per-window boolean to the output shape.
    # 2. Choose whether to fill (from `operand`) if in-bounds, or drop if
    #    out-of-bounds.
    # Broadcasting is free, but choosing whether to fill or drop involves an
    # elementwise op the size of the output.
    output_mask_flops = output_size
    return index_check_flops + output_mask_flops

  return 0


@roofline.register_roofline(slicing.gather_p)
def _gather_roofline(
    ctx: roofline.RooflineRuleContext,
    *args,
    mode: slicing.GatherScatterMode,
    **kw,
) -> roofline.RooflineResult:
  _, indices = (roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in)
  out = roofline.RooflineShape.from_aval(ctx.avals_out[0])

  # Gather doesn't read the whole input buffer, it's equivalent to a copy the
  # size of the output shape and a read of the gather indices.
  unfused_hbm_bytes = (
      out.dtype.itemsize * out.size * 2 + indices.dtype.itemsize * indices.size
  )

  return roofline.RooflineResult(
      unfused_flops=_calculate_gather_flops(mode, indices.size, out.size),
      unfused_hbm_bytes=unfused_hbm_bytes,
  )


def _scatter_roofline(
    ctx: roofline.RooflineRuleContext,
    *args,
    **kw,
) -> roofline.RooflineResult:
  """Roofline for Jax's `scatter*` primitives.

  The `scatter` functionality itself is a simple data read and write, which
  contributes 0 flops.

  But, the jaxpr for each `scatter*` function (aside from `jax.lax.scatter`)
  contains an `update_jaxpr` that gets applied to the operand & scattered
  updates (e.g. `add` for `scatter_add`, or arbitrary unary function for
  `scatter_apply`), which *does* contribute flops. This `update_jaxpr` gets
  applied to every element of the scattered updates.

  Thus,
  flops = [# flops for `update_jaxpr` per element] * [# elements in `updates`].

  To calculate # flops for `update_jaxpr`, we convert the `update_jaxpr` back to
  a callable, and then call `roofline` on that callable. `update_jaxpr` does not
  contain any information about input shapes or dtypes; it expects scalars. It
  will therefore give us a # flops-per-element result, which we multiply by
  the size of the updates to get the total flops.
  """
  (_, indices, updates) = (
      roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in
  )

  update_jaxpr = kw.get('update_jaxpr')

  flops = 0
  if update_jaxpr:
    update_fn = lambda *inputs: core.eval_jaxpr(update_jaxpr, [], *inputs)
    # Create dummy scalar inputs.
    dummy_inputs = [
        api.ShapeDtypeStruct((), updates.dtype) for _ in update_jaxpr.invars
    ]
    # Calculate the flops for the `update_jaxpr` on scalar inputs.
    _, roofline_result = roofline.roofline(update_fn)(*dummy_inputs)
    # Multiply by the size of the updates to get the total flops.
    flops = roofline_result.unfused_flops * updates.size

  return roofline.RooflineResult(
      unfused_flops=flops,
      # Scatter accesses the equivalent of 3N update shapes (input, output, and
      # updates), and the scatter indices.
      unfused_hbm_bytes=(
          3 * updates.dtype.itemsize * updates.size
          + indices.dtype.itemsize * indices.size
      ),
  )


roofline.register_roofline(slicing.scatter_add_p)(_scatter_roofline)
roofline.register_roofline(slicing.scatter_max_p)(_scatter_roofline)
roofline.register_roofline(slicing.scatter_min_p)(_scatter_roofline)
roofline.register_roofline(slicing.scatter_mul_p)(_scatter_roofline)
roofline.register_roofline(slicing.scatter_sub_p)(_scatter_roofline)
# Also registers `jax.lax.scatter_apply`, which uses the `scatter_p` primitive.
roofline.register_roofline(slicing.scatter_p)(_scatter_roofline)

def _scalar_collective_roofline(
    ctx: roofline.RooflineRuleContext,
    *args,
    axes: tuple[str, ...],
    **kw,
) -> roofline.RooflineResult:
  shapes = [roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in]
  ctx = replace(ctx, avals_in=[core.ShapedArray((1,), shape.dtype) for shape in shapes])
  return _ring_collective_roofline(ctx, *args, axes=axes, is_reduce=False, **kw)


roofline.register_roofline(lax_parallel.pmin_p)(_scalar_collective_roofline)
roofline.register_roofline(lax_parallel.pmax_p)(_scalar_collective_roofline)


@roofline.register_roofline(lax_parallel.psum_invariant_p)
def _psum2_roofline(
  ctx: roofline.RooflineRuleContext,
  *args,
  axes: tuple[str, ...],
  **kw,
) -> roofline.RooflineResult:
  ring_roofline = _ring_collective_roofline(ctx, *args, axes=axes, **kw)

  def double_dict(d: dict[str, int]) -> dict[str, int]:
    return {k: v * 2 for k, v in d.items()}

  return roofline.RooflineResult(
    ici_bytes=double_dict(ring_roofline.ici_bytes),
    ici_latency=double_dict(ring_roofline.ici_latency),
  )


@roofline.register_roofline(lax_parallel.all_to_all_p)
def _all_to_all_roofline(
  ctx: roofline.RooflineRuleContext,
  *args,
  axis_name: tuple[str, ...],
  **kw,
) -> roofline.RooflineResult:
  if zeros_result := _return_zeros_if_one_sized_axis(ctx, axis_name):
    return zeros_result

  assert ctx.mesh
  mesh = ctx.mesh.shape
  size = roofline.RooflineShape.total_bytes(ctx.avals_in) * np.prod([
    mesh[axis] for axis in axis_name
  ])

  smallest_axis = sorted(axis_name, key=lambda x: mesh[x])[0]
  num_axes = len(axis_name)
  bisection_bw = mesh[smallest_axis] ** (num_axes - 1)
  if mesh[smallest_axis] > 2:
    # Times 2 because of wraparound.
    bisection_bw *= 2

  # Half the data needs to cross the bisection on average.
  ici_bytes = size / 2 / bisection_bw

  # The latency is the max number of hops across the mesh.
  ici_latency = sum(mesh[axis] / 2 for axis in axis_name)

  return roofline.RooflineResult(
    ici_bytes={axis: int(ici_bytes) for axis in axis_name},
    ici_latency={axis: int(ici_latency) for axis in axis_name},
  )


@roofline.register_roofline(lax_parallel.ppermute_p)
def _ppermute_roofline(
  ctx: roofline.RooflineRuleContext,
  *args,
  axis_name: tuple[str, ...],
  perm: tuple[tuple[int, int], ...],
  **kw,
) -> roofline.RooflineResult:
  if zeros_result := _return_zeros_if_one_sized_axis(ctx, axis_name):
    return zeros_result

  assert ctx.mesh
  mesh = ctx.mesh.shape
  mesh_dims: list[int] = [mesh.get(axis, 1) for axis in axis_name]
  shard_size = roofline.RooflineShape.total_bytes(ctx.avals_in)

  ici_contention: dict[tuple[tuple[int, ...], ...], float] = defaultdict(float)
  ici_latency = 0

  for src, dst in perm:
    if src == dst:
      continue
    # Perms are linearized.
    src_coords = tuple(int(i) for i in np.unravel_index(src, mesh_dims))
    dst_coords = tuple(int(i) for i in np.unravel_index(dst, mesh_dims))

    ici_latency_for_perm = 0

    # For each dimension.
    for i in range(len(axis_name)):
      dim_size = mesh_dims[i]
      src_pos = src_coords[i]
      dst_pos = dst_coords[i]

      if src_pos != dst_pos:
        # Calculate distance with wraparound.
        clockwise_dist = (dst_pos - src_pos) % dim_size
        counter_dist = (src_pos - dst_pos) % dim_size
        direction = 1 if clockwise_dist <= counter_dist else -1

        curr_pos = src_pos
        while curr_pos != dst_pos:
          curr_coords = util.tuple_update(src_coords, i, curr_pos)
          next_pos = (curr_pos + direction) % dim_size
          next_coords = util.tuple_update(curr_coords, i, next_pos)
          ici_contention[tuple(sorted([curr_coords, next_coords]))] += 1
          curr_pos = next_pos

        distance = min(clockwise_dist, counter_dist)
        ici_latency_for_perm += distance

    ici_latency = max(ici_latency, ici_latency_for_perm)

  ici_bytes = shard_size * max(ici_contention.values(), default=0)
  return roofline.RooflineResult(
    ici_bytes={axis: int(ici_bytes) for axis in axis_name},
    ici_latency={axis: int(ici_latency) for axis in axis_name},
  )


@roofline.register_roofline(lax.reduce_sum_p)
def _reduce_sum_p_roofline(
    ctx: roofline.RooflineRuleContext,
    *args,
    axes: tuple[int, ...],
    **kw,
) -> roofline.RooflineResult:
  (x,) = (roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in)
  domain_size = np.prod([x.shape[i] for i in axes])
  other_axes = set(range(len(x.shape))) - set(axes)
  result_size = np.prod([x.shape[i] for i in other_axes])

  return roofline.RooflineResult(
      # To add n values, we do n - 1 add operations, and we have to do that
      # for every element in the result.
      unfused_flops=int((domain_size - 1) * result_size),
      # Size of input, plus output. (We assume that the output is also used
      # as accumulator.)
      unfused_hbm_bytes=int(x.dtype.itemsize * (x.size + result_size)),
  )

@roofline.register_roofline(lax.select_n_p)
def _select_n_p_roofline(
    ctx: roofline.RooflineRuleContext,
    *args,
    **kw,
) -> roofline.RooflineResult:
  (x, *_) = (roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in)
  out = roofline.RooflineShape.from_aval(ctx.avals_out[0])

  return roofline.RooflineResult(
      unfused_flops=out.size,
      unfused_hbm_bytes=(
          x.dtype.itemsize * x.size + out.dtype.itemsize * out.size
      ),
  )
