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
from jax.ops.image import util
from jax.numpy import lax_numpy as np


def extract_image_patches(image, ksizes, strides, rates, padding):
    # replicating tensorflow's implementation
    num_dims = image.ndim
    num_spatial_dims = num_dims - 2
    data_format = "NHWC"

    feature_dim = util.get_feature_dim(num_dims, data_format)

    depth = image.shape[feature_dim]

    kernel_shape = [ksizes[util.get_spatial_dim(num_dims, data_format, i)]
                    for i in range(num_spatial_dims)]
    kernel_size = onp.product(kernel_shape)
    kernel_shape.append(1)
    kernel_shape.append(kernel_size * depth)

    iota_kernel_shape = (kernel_size, 1, kernel_size)

    conv_filter = lax.eq(
        lax.broadcasted_iota(np.int64, iota_kernel_shape, 0),
        lax.broadcasted_iota(np.int64, iota_kernel_shape, 2),
    )
    conv_filter = lax.convert_element_type(conv_filter, image.dtype)
    conv_filter = lax.reshape(conv_filter, kernel_shape)

    dim_num = lax.conv_dimension_numbers(image.shape,
                                         conv_filter.shape,
                                         (data_format, "HWIO", data_format))
    window_strides = [0] * num_spatial_dims
    rhs_dilation = [0] * num_spatial_dims
    for i in range(num_spatial_dims):
        dim = util.get_spatial_dim(num_dims, data_format, i)
        window_strides[i] = strides[dim]
        rhs_dilation[i] = rates[dim]

    conv = lax.conv_general_dilated(image,
                                    conv_filter,
                                    window_strides,
                                    padding,
                                    None,
                                    rhs_dilation,
                                    dim_num,
                                    depth)
    conv_dims = list(conv.shape)

    conv_dims[-1] = depth
    conv_dims.append(kernel_size)
    conv = lax.reshape(conv, conv_dims)

    permutation = list(range(len(conv_dims)))
    permutation[-2] = permutation[-1]
    permutation[-1] = permutation[-2] - 1
    # TODO(gehring): check if there is a way to transpose and reshape in one
    #  reshape op.
    conv = lax.transpose(conv, permutation)

    conv_dims = conv_dims[:-1]
    conv_dims[-1] *= kernel_size

    return lax.reshape(conv, conv_dims)
