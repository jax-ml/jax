# Copyright 2020 Google LLC
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


from typing import Any, Optional, Sequence, Tuple, Union
from jax._src.numpy import lax_numpy as jnp
from jax._src.util import prod
from . import lax

DType = Any

def conv_general_dilated_patches(
    lhs: lax.Array,
    filter_shape: Sequence[int],
    window_strides: Sequence[int],
    padding: Union[str, Sequence[Tuple[int, int]]],
    lhs_dilation: Sequence[int] = None,
    rhs_dilation: Sequence[int] = None,
    dimension_numbers: lax.ConvGeneralDilatedDimensionNumbers = None,
    precision: lax.PrecisionType = None,
    preferred_element_type: Optional[DType] = None,
) -> lax.Array:
  """Extract patches subject to the receptive field of `conv_general_dilated`.

  Runs the input through a convolution with given parameters. The kernel of the
  convolution is constructed such that the output channel dimension `"C"`
  contains flattened image patches, so instead a single `"C"` dimension
  represents, for example, three dimensions `"chw"` collapsed. The order of
  these dimensions is `"c" + ''.join(c for c in rhs_spec if c not in 'OI')`,
  where `rhs_spec == dimension_numbers[1]`, and the size of this `"C"`
  dimension is therefore the size of each patch, i.e.
  `np.prod(filter_shape) * lhs.shape[lhs_spec.index('C')]`, where
  `lhs_spec == dimension_numbers[0]`.

  Docstring below adapted from `jax.lax.conv_general_dilated`.

  See Also:
    https://www.tensorflow.org/xla/operation_semantics#conv_convolution

  Args:
    lhs: a rank `n+2` dimensional input array.
    filter_shape: a sequence of `n` integers, representing the receptive window
      spatial shape in the order as specified in
      `rhs_spec = dimension_numbers[1]`.
    window_strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence of
      `n` `(low, high)` integer pairs that give the padding to apply before and
      after each spatial dimension.
    lhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `lhs`. LHS dilation
      is also known as transposed convolution.
    rhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `rhs`. RHS dilation
      is also known as atrous convolution.
    dimension_numbers: either `None`, or a 3-tuple
      `(lhs_spec, rhs_spec, out_spec)`, where each element is a string
      of length `n+2`. `None` defaults to `("NCHWD..., OIHWD..., NCHWD...")`.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, or a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``).
    preferred_element_type: Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    A rank `n+2` array containing the flattened image patches in the output
    channel (`"C"`) dimension. For example if
    `dimension_numbers = ("NcHW", "OIwh", "CNHW")`, the output has dimension
    numbers `"CNHW" = "{cwh}NHW"`, with the size of dimension `"C"` equal to
    the size of each patch
    (`np.prod(filter_shape) * lhs.shape[lhs_spec.index('C')]`).

  """
  filter_shape = tuple(filter_shape)
  dimension_numbers = lax.conv_dimension_numbers(
      lhs.shape, (1, 1) + filter_shape, dimension_numbers)

  lhs_spec, rhs_spec, out_spec = dimension_numbers

  spatial_size = prod(filter_shape)
  n_channels = lhs.shape[lhs_spec[1]]

  # Move separate `lhs` spatial locations into separate `rhs` channels.
  rhs = jnp.eye(spatial_size, dtype=lhs.dtype).reshape(filter_shape * 2)

  rhs = rhs.reshape((spatial_size, 1) + filter_shape)
  rhs = jnp.tile(rhs, (n_channels,) + (1,) * (rhs.ndim - 1))
  rhs = jnp.moveaxis(rhs, (0, 1), (rhs_spec[0], rhs_spec[1]))

  out = lax.conv_general_dilated(
      lhs=lhs,
      rhs=rhs,
      window_strides=window_strides,
      padding=padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      dimension_numbers=dimension_numbers,
      precision=None if precision is None else (precision,
                                                lax.Precision.DEFAULT),
      feature_group_count=n_channels,
      preferred_element_type=preferred_element_type
  )
  return out
