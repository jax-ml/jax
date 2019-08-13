# Copyright 2019 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from jax import lax
from jax.numpy import lax_numpy as np


def extract_image_patches(lhs, rhs_shape, window_strides, padding,
                          rhs_dilation, data_format="NHWC"):
    r"""Extract `patches` from `images` and put them in the \"depth\" output dimension.
      Args:
        lhs: A 4-D Tensor with shape `[batch, in_rows, in_cols, depth]
        rhs_shape: The size of the sliding window for each dimension of `images`.
        window_strides: A 1-D Tensor of length 4. How far the centers of two consecutive
          patches are in the images. Must be: `[1, stride_rows, stride_cols, 1]`.
        padding: The type of padding algorithm to use.
          We specify the size-related attributes as: ```python ksizes = [1,
            ksize_rows, ksize_cols, 1] strides = [1, strides_rows, strides_cols, 1]
            rates = [1, rates_rows, rates_cols, 1]```
        rhs_dilation: A 1-D Tensor of length 4. Must be: `[1, rate_rows, rate_cols, 1]`.
          This is the input stride, specifying how far two consecutive patch samples
          are in the input. Equivalent to extracting patches with `patch_sizes_eff =
          patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by subsampling
          them spatially by a factor of `rates`. This is equivalent to `rate` in
          dilated (a.k.a. Atrous) convolutions.
        data_format: The format of the `lhs`. Must be either `'NHWC'` or `'NCHW'`.
      Returns:
        A 4-D Tensor. Has the same type and data format as `lhs`, and with shape
        output depth `O = rhs_shape[1] * rhs_shape[2] * C`.
      """
    num_dims = lhs.ndim
    num_spatial_dims = num_dims - 2

    batch_dim = data_format.index('N')
    feature_dim = data_format.index('C')
    depth = lhs.shape[feature_dim]

    if rhs_shape[batch_dim] != 1 or rhs_shape[feature_dim] != 1:
        raise NotImplementedError((
            "Current implementation does not yet support window sizes > 1 in "
            "the batch and depth dimensions."
        ))
    if window_strides[batch_dim] != 1 or window_strides[feature_dim] != 1:
        raise NotImplementedError((
            "Current implementation does not support strides in the batch "
            "and depth dimensions."
        ))
    if rhs_dilation[batch_dim] != 1 or rhs_dilation[feature_dim] != 1:
        raise NotImplementedError((
            "Current implementation does not support dilations in the batch "
            "and depth dimensions."
        ))

    # replicating tensorflow's implementation
    lhs_perm = lax.conv_general_permutations(
        (data_format, "HWIO", data_format))[0]
    kernel_shape = [rhs_shape[i] for i in lhs_perm[2:]]

    kernel_size = onp.product(kernel_shape)
    kernel_shape.append(1)
    kernel_shape.append(kernel_size * depth)

    iota_kernel_shape = (kernel_size, depth, kernel_size)

    conv_filter = lax.eq(
        lax.broadcasted_iota(np.int32, iota_kernel_shape, 0),
        lax.broadcasted_iota(np.int32, iota_kernel_shape, 2),
    )
    conv_filter = lax.convert_element_type(conv_filter, lhs.dtype)
    conv_filter = lax.reshape(conv_filter, kernel_shape)

    dim_num = lax.conv_dimension_numbers(lhs.shape,
                                         conv_filter.shape,
                                         (data_format, "HWIO", data_format))
    conv_window_strides = [0] * num_spatial_dims
    conv_rhs_dilation = [0] * num_spatial_dims
    for i in range(num_spatial_dims):
        dim = dim_num.lhs_spec[i + 2]
        conv_window_strides[i] = window_strides[dim]
        conv_rhs_dilation[i] = rhs_dilation[dim]

    conv = lax.conv_general_dilated(lhs,
                                    conv_filter,
                                    conv_window_strides,
                                    padding,
                                    None,
                                    conv_rhs_dilation,
                                    dim_num,
                                    depth)
    conv_dims = list(conv.shape)

    conv_dims[-1] = depth
    conv_dims.append(kernel_size)
    conv = lax.reshape(conv, conv_dims)

    permutation = list(range(len(conv_dims)))
    permutation[-2] = permutation[-1]
    permutation[-1] = permutation[-2] - 1
    conv = lax.transpose(conv, permutation)

    conv_dims = conv_dims[:-1]
    conv_dims[-1] *= kernel_size

    return lax.reshape(conv, conv_dims)
