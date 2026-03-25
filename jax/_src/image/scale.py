# Copyright 2020 The JAX Authors.
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

from __future__ import annotations

from collections.abc import Callable, Sequence
import enum
from typing import Any

import math
import numpy as np

from jax._src import api
from jax._src import core
from jax._src import dtypes
from jax._src import numpy as jnp
from jax._src.lax import lax
from jax._src.numpy import einsum as jnp_einsum
from jax._src.util import canonicalize_axis
from jax._src.numpy.util import promote_dtypes_inexact


def _fill_lanczos_kernel(radius, x):
  y = radius * jnp.sin(np.pi * x) * jnp.sin(np.pi * x / radius)
  #  out = y / (np.pi ** 2 * x ** 2) where x >1e-3, 1 otherwise
  out = jnp.where(x > 1e-3, jnp.divide(y, jnp.where(x != 0, np.pi**2 * x**2, 1)), 1)
  return jnp.where(x > radius, 0., out)

def _fill_keys_cubic_kernel(x):
  # http://ieeexplore.ieee.org/document/1163711/
  # R. G. Keys. Cubic convolution interpolation for digital image processing.
  # IEEE Transactions on Acoustics, Speech, and Signal Processing,
  # 29(6):1153–1160, 1981.
  #
  # https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
  # This is the Keys kernel with A=-0.5.
  #
  # This kernel matches Pillow, TensorFlow, and Pytorch when
  # antialiasing is enabled.
  out = ((1.5 * x - 2.5) * x) * x + 1.
  out = jnp.where(x >= 1., ((-0.5 * x + 2.5) * x - 4.) * x + 2., out)
  return jnp.where(x >= 2., 0., out)


def _fill_opencv_cubic_kernel(x):
  # See https://github.com/jax-ml/jax/issues/15768#issuecomment-1529939102 and
  # https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
  #
  # When antialiasing is disabled, PyTorch uses a cubic kernel with A = -0.75
  # that matches OpenCV.
  # At least some users consider this a bug (opencv/opencv#17720), and that set
  # of parameters suffers from ringing artifacts.
  a = -0.75
  out = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
  out = jnp.where(x >= 1.0, ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a,
                  out)
  return jnp.where(x >= 2.0, 0.0, out)


def _fill_triangle_kernel(x):
  return jnp.maximum(0, 1 - jnp.abs(x))


def compute_weight_mat(input_size: core.DimSize,
                       output_size: core.DimSize,
                       scale,
                       translation,
                       kernel: Callable,
                       antialias: bool,
                       edge_padding: bool,
                       radius: int | None):
  dtype = dtypes.result_type(scale, translation)
  inv_scale = 1. / scale
  # When downsampling the kernel should be scaled since we want to low pass
  # filter and interpolate, but when upsampling it should not be since we only
  # want to interpolate.
  kernel_scale = jnp.maximum(inv_scale, 1.) if antialias else 1.

  # sample_f has shape [output_size] and is the floating-point index in the
  # input image corresponding to the center of each output pixel.
  sample_f = ((jnp.arange(output_size, dtype=dtype) + 0.5) * inv_scale -
              translation * inv_scale - 0.5)

  # Evaluate the kernel for all input/output coordinate pairs. If edge_padding
  # is true, this includes k pixels outside the original image.
  if edge_padding:
    assert radius is not None
    if antialias:
      # This case isn't actually reachable from the public APIs at the time of
      # writing, but we did figure it out, so we may as well leave the code.
      concrete_scale = core.concrete_or_error(
          None, scale,
          context="Antialiasing with edge padding requires a static scale."
      )
      inv_scale_val = 1.0 / float(concrete_scale)
      kernel_scale_val = max(inv_scale_val, 1.0)
      k = math.ceil(radius * kernel_scale_val)
    else:
      k = radius
  else:
    k = 0

  expanded_indices = jnp.arange(-k, input_size + k, dtype=dtype)
  x = jnp.abs(sample_f[np.newaxis, :] - expanded_indices[:, np.newaxis])
  x = x / kernel_scale
  weights = kernel(x)

  if edge_padding:
    # Some of the weights are for indices outside the input image. We use a
    # scatter-add to move their mass onto the relevant edge pixels.
    clamped_indices = jnp.clip(
      expanded_indices.astype(jnp.int32), 0, input_size - 1)
    output_indices = jnp.arange(output_size)
    weight_mat = jnp.zeros((input_size, output_size), dtype=dtype)
    output_indices_expanded = lax.broadcast_in_dim(
        output_indices, (expanded_indices.shape[0], output_size), (1,))
    weight_mat = weight_mat.at[
        clamped_indices[:, np.newaxis], output_indices_expanded
    ].add(weights)
    # Normalize the weights
    total_weight_sum = jnp.sum(weight_mat, axis=0, keepdims=True)
    weights = jnp.where(
        jnp.abs(total_weight_sum) > 1000. * float(np.finfo(np.float32).eps),
        jnp.divide(weight_mat,
                   jnp.where(total_weight_sum != 0, total_weight_sum, 1)),
        0)
  else:
    # Normalize the weights to account for the fact that some or all of the
    # input coordinates might not be in the valid part of the input image.
    total_weight_sum = jnp.sum(weights, axis=0, keepdims=True)
    weights = jnp.where(
        jnp.abs(total_weight_sum) > 1000. * float(np.finfo(np.float32).eps),
        jnp.divide(weights,
                   jnp.where(total_weight_sum != 0, total_weight_sum, 1)),
        0)

    # Zero out weights where the sample location is completely outside the input
    # range. sample_f has already had the 0.5 removed, hence the weird range
    # below.
    input_size_minus_0_5 = core.dimension_as_value(input_size) - 0.5
    weights = jnp.where(
        jnp.logical_and(sample_f >= -0.5,
                        sample_f <= input_size_minus_0_5)[np.newaxis, :],
        weights, 0)

  return weights


def _scale_and_translate(x, output_shape: core.Shape,
                         spatial_dims: Sequence[int], scale, translation,
                         kernel, antialias: bool, precision,
                         edge_padding: bool = False, radius: int | None = None):
  """
  Args:
    edge_padding: if False, pixels that are off the edge of the input
      image will receive zero weight. If True, the edges of the input image are
      repeated.
    radius: the radius of the kernel. May be None if edge_padding is False.
  """
  input_shape = x.shape
  assert len(input_shape) == len(output_shape)
  assert len(spatial_dims) == len(scale)
  assert len(spatial_dims) == len(translation)
  if len(spatial_dims) == 0:
    return x
  contractions = []
  in_indices = list(range(len(output_shape)))
  out_indices = list(range(len(output_shape)))
  for i, d in enumerate(spatial_dims):
    d = canonicalize_axis(d, x.ndim)
    m = input_shape[d]
    n = output_shape[d]
    w = compute_weight_mat(
        m, n, scale[i], translation[i], kernel, antialias,
        edge_padding=edge_padding, radius=radius,
    ).astype(x.dtype)
    contractions.append(w)
    contractions.append([d, len(output_shape) + i])
    out_indices[d] = len(output_shape) + i
  contractions.append(out_indices)
  return jnp_einsum.einsum(x, in_indices, *contractions, precision=precision)


class ResizeMethod(enum.Enum):
  """Image resize method.

  Possible values are:

  NEAREST:
    Nearest-neighbor interpolation.

  LINEAR:
    `Linear interpolation`_.

  LANCZOS3:
    `Lanczos resampling`_, using a kernel of radius 3.

  LANCZOS5:
    `Lanczos resampling`_, using a kernel of radius 5.

  CUBIC:
    `Cubic interpolation`_, using the Keys cubic kernel.

  .. _Linear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
  .. _Cubic interpolation: https://en.wikipedia.org/wiki/Bicubic_interpolation
  .. _Lanczos resampling: https://en.wikipedia.org/wiki/Lanczos_resampling
  """

  NEAREST = 0
  LINEAR = 1
  LANCZOS3 = 2
  LANCZOS5 = 3
  CUBIC = 4
  CUBIC_PYTORCH = 5

  # Caution: The current resize implementation assumes that the resize kernels
  # are interpolating, i.e. for the identity warp the output equals the input.
  # This is not true for, e.g. a Gaussian kernel, so if such kernels are added
  # the implementation will need to be changed.

  @staticmethod
  def from_string(s: str):
    if s == 'nearest':
      return ResizeMethod.NEAREST
    if s in ['linear', 'bilinear', 'trilinear', 'triangle']:
      return ResizeMethod.LINEAR
    elif s == 'lanczos3':
      return ResizeMethod.LANCZOS3
    elif s == 'lanczos5':
      return ResizeMethod.LANCZOS5
    elif s in ['cubic', 'bicubic', 'tricubic']:
      return ResizeMethod.CUBIC
    elif s in ['cubic-pytorch', 'bicubic-pytorch']:
      return ResizeMethod.CUBIC_PYTORCH
    else:
      raise ValueError(f'Unknown resize method "{s}"')

_kernels = {
    ResizeMethod.LINEAR: (1, _fill_triangle_kernel),
    ResizeMethod.LANCZOS3: (3, lambda x: _fill_lanczos_kernel(3., x)),
    ResizeMethod.LANCZOS5: (5, lambda x: _fill_lanczos_kernel(5., x)),
    ResizeMethod.CUBIC: (2, _fill_keys_cubic_kernel),
    ResizeMethod.CUBIC_PYTORCH: (2, _fill_opencv_cubic_kernel),
}


# scale and translation here are scalar elements of an np.array, what is the
# correct type annotation?
def scale_and_translate(image, shape: core.Shape,
                        spatial_dims: Sequence[int],
                        scale, translation,
                        method: str | ResizeMethod,
                        antialias: bool = True,
                        precision=lax.Precision.HIGHEST):
  """Apply a scale and translation to an image.

  Generates a new image of shape 'shape' by resampling from the input image
  using the sampling method corresponding to method. For 2D images, this
  operation transforms a location in the input images, (x, y), to a location
  in the output image according to::

    (x * scale[1] + translation[1], y * scale[0] + translation[0])

  (Note the *inverse* warp is used to generate the sample locations.)
  Assumes half-centered pixels, i.e the pixel at integer location ``row, col``
  has coordinates ``y, x = row + 0.5, col + 0.5``, and similarly for other input
  image dimensions.

  If an output location(pixel) maps to an input sample location that is outside
  the input boundaries then the value for the output location will be set to
  zero.

  This function can be used to imitate the behavior of
  ``torch.nn.functional.interpolate`` with ``align_corners=True`` by setting::

      scale = (n - 1) / (m - 1)
      translation = 0.5 * (1 - scale)

  where ``m`` is the input size and ``n`` is the output size for a given
  dimension.

  The ``method`` argument expects one of the following resize methods:

  ``ResizeMethod.LINEAR``, ``"linear"``, ``"bilinear"``, ``"trilinear"``,
    ``"triangle"`` `Linear interpolation`_. If ``antialias`` is ``True``, uses a
    triangular filter when downsampling.

  ``ResizeMethod.CUBIC``, ``"cubic"``, ``"bicubic"``, ``"tricubic"``
    `Cubic interpolation`_, using the Keys cubic kernel.

  ``ResizeMethod.CUBIC_PYTORCH``, ``"cubic-pytorch"``, ``"bicubic-pytorch"``
    `Cubic interpolation`_, matching PyTorch's bicubic resizing behavior.
    Identical to ``ResizeMethod.CUBIC`` when antialiasing is enabled, but uses
    a different kernel and enables edge padding when antialiasing is disabled.

  ``ResizeMethod.LANCZOS3``, ``"lanczos3"``
    `Lanczos resampling`_, using a kernel of radius 3.

  ``ResizeMethod.LANCZOS5``, ``"lanczos5"``
    `Lanczos resampling`_, using a kernel of radius 5.

  .. _Linear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
  .. _Cubic interpolation: https://en.wikipedia.org/wiki/Bicubic_interpolation
  .. _Lanczos resampling: https://en.wikipedia.org/wiki/Lanczos_resampling

  Args:
    image: a JAX array.
    shape: the output shape, as a sequence of integers with length equal to the
      number of dimensions of `image`.
    spatial_dims: A length K tuple specifying the spatial dimensions that the
      passed scale and translation should be applied to.
    scale: A [K] array with the same number of dimensions as image, containing
      the scale to apply in each dimension.
    translation: A [K] array with the same number of dimensions as image,
      containing the translation to apply in each dimension.
    method: the resizing method to use; either a ``ResizeMethod`` instance or a
      string. Available methods are: LINEAR, LANCZOS3, LANCZOS5, CUBIC, CUBIC_PYTORCH.
    antialias: Should an antialiasing filter be used when downsampling? Defaults
      to ``True``. Has no effect when upsampling.

  Returns:
    The scale and translated image.
  """
  shape = core.canonicalize_shape(shape)
  if len(shape) != image.ndim:
    msg = ('shape must have length equal to the number of dimensions of x; '
           f' {shape} vs {image.shape}')
    raise ValueError(msg)
  if isinstance(method, str):
    method = ResizeMethod.from_string(method)
  if method == ResizeMethod.NEAREST:
    # Nearest neighbor is currently special-cased for straight resize, so skip
    # for now.
    raise ValueError('Nearest neighbor resampling is not currently supported '
                     'for scale_and_translate.')
  assert isinstance(method, ResizeMethod)

  if method == ResizeMethod.CUBIC_PYTORCH and antialias:
    method = ResizeMethod.CUBIC
  radius, kernel = _kernels[method]
  edge_padding = (method == ResizeMethod.CUBIC_PYTORCH and not antialias)
  image, = promote_dtypes_inexact(image)
  scale, translation = promote_dtypes_inexact(scale, translation)
  return _scale_and_translate(
     image, shape, spatial_dims, scale, translation, kernel, antialias,
     precision, edge_padding=edge_padding, radius=radius)


def _resize_nearest(x, output_shape: core.Shape):
  input_shape = x.shape
  assert len(input_shape) == len(output_shape)
  spatial_dims = tuple(i for i in range(len(input_shape))
                       if not core.definitely_equal(input_shape[i], output_shape[i]))
  for d in spatial_dims:
    m = input_shape[d]
    n = output_shape[d]
    offsets = (jnp.arange(n, dtype=np.float32) + 0.5) * core.dimension_as_value(m) / core.dimension_as_value(n)
    # TODO(b/206898375): this computation produces the wrong result on
    # CPU and GPU when using float64. Use float32 until the bug is fixed.
    offsets = jnp.floor(offsets.astype(np.float32)).astype(np.int32)
    indices: list[Any] = [slice(None)] * len(input_shape)
    indices[d] = offsets
    x = x[tuple(indices)]
  return x


@api.jit(static_argnums=(1, 2, 3, 4))
def _resize(image, shape: core.Shape, method: str | ResizeMethod,
            antialias: bool, precision):
  if len(shape) != image.ndim:
    msg = ('shape must have length equal to the number of dimensions of x; '
           f' {shape} vs {image.shape}')
    raise ValueError(msg)
  if isinstance(method, str):
    method = ResizeMethod.from_string(method)
  if method == ResizeMethod.NEAREST:
    return _resize_nearest(image, shape)
  assert isinstance(method, ResizeMethod)

  image, = promote_dtypes_inexact(image)
  # Skip dimensions that have scale=1 and translation=0, this is only possible
  # since all of the current resize methods (kernels) are interpolating, so the
  # output = input under an identity warp.
  spatial_dims = tuple(i for i in range(len(shape))
                       if not core.definitely_equal(image.shape[i], shape[i]))
  if method == ResizeMethod.CUBIC_PYTORCH and antialias:
    method = ResizeMethod.CUBIC
  radius, kernel = _kernels[method]
  scale = [1.0 if core.definitely_equal(shape[d], 0) else core.dimension_as_value(shape[d]) / core.dimension_as_value(image.shape[d])
           for d in spatial_dims]
  edge_padding = (method == ResizeMethod.CUBIC_PYTORCH and not antialias)
  return _scale_and_translate(image, shape, spatial_dims, scale,
                              [0.] * len(spatial_dims), kernel, antialias,
                              precision, edge_padding=edge_padding,
                              radius=radius)


def resize(image, shape: core.Shape, method: str | ResizeMethod,
           antialias: bool = True,
           precision = lax.Precision.HIGHEST):
  """Image resize.

  The ``method`` argument expects one of the following resize methods:

  ``ResizeMethod.NEAREST``, ``"nearest"``
    `Nearest neighbor interpolation`_. The values of ``antialias`` and
    ``precision`` are ignored.

  ``ResizeMethod.LINEAR``, ``"linear"``, ``"bilinear"``, ``"trilinear"``, ``"triangle"``
    `Linear interpolation`_. If ``antialias`` is ``True``, uses a triangular
    filter when downsampling.

  ``ResizeMethod.CUBIC``, ``"cubic"``, ``"bicubic"``, ``"tricubic"``
    `Cubic interpolation`_, using the Keys cubic kernel.

  ``ResizeMethod.CUBIC_PYTORCH``, ``"cubic-pytorch"``, ``"bicubic-pytorch"``
    `Cubic interpolation`_, matching PyTorch's bicubic resizing behavior.
    Identical to ``ResizeMethod.CUBIC`` when antialiasing is enabled, but uses
    a different kernel and enables edge padding when antialiasing is disabled.

  ``ResizeMethod.LANCZOS3``, ``"lanczos3"``
    `Lanczos resampling`_, using a kernel of radius 3.

  ``ResizeMethod.LANCZOS5``, ``"lanczos5"``
    `Lanczos resampling`_, using a kernel of radius 5.

  .. _Nearest neighbor interpolation: https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
  .. _Linear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
  .. _Cubic interpolation: https://en.wikipedia.org/wiki/Bicubic_interpolation
  .. _Lanczos resampling: https://en.wikipedia.org/wiki/Lanczos_resampling

  This function does not support an ``align_corners`` argument like
  ``torch.nn.functional.interpolate``. That behavior can be emulated using
  :func:`scale_and_translate`.

  Args:
    image: a JAX array.
    shape: the output shape, as a sequence of integers with length equal to
      the number of dimensions of `image`. Note that :func:`resize` does not
      distinguish spatial dimensions from batch or channel dimensions, so this
      includes all dimensions of the image. To represent a batch or a channel
      dimension, simply leave that element of the shape unchanged.
    method: the resizing method to use; either a ``ResizeMethod`` instance or a
      string. Available methods are: LINEAR, LANCZOS3, LANCZOS5, CUBIC, CUBIC_PYTORCH.
    antialias: should an antialiasing filter be used when downsampling? Defaults
      to ``True``. Has no effect when upsampling.
  Returns:
    The resized image.
  """
  return _resize(image, core.canonicalize_shape(shape), method, antialias,
                 precision)
