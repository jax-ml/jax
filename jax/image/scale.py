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

from functools import partial
import enum
import math
from typing import Callable, Sequence, Tuple, Union

from jax import jit
from jax import lax
from jax import numpy as jnp
import numpy as np


def _lanczos_kernel(radius: float):
  def kernel(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    y = radius * np.sin(np.pi * x) * np.sin(np.pi * x / radius)
    with np.errstate(divide='ignore', invalid='ignore'):
      out = y / (np.pi ** 2 * x ** 2)
    out = np.where(x <= 1e-3, 1., out)
    return np.where(x > radius, 0., out)

  return radius, kernel


def _triangle_kernel():
  return 1., lambda x: np.maximum(0, 1 - np.abs(x))


def _keys_cubic_kernel():
  # http://ieeexplore.ieee.org/document/1163711/
  # R. G. Keys. Cubic convolution interpolation for digital image processing.
  # IEEE Transactions on Acoustics, Speech, and Signal Processing,
  # 29(6):1153–1160, 1981.
  def kernel(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    out = ((1.5 * x - 2.5) * x) * x + 1.
    out = np.where(x >= 1., ((-0.5* x + 2.5) * x - 4.) * x + 2., out)
    return np.where(x >= 2., 0., out)
  return 2., kernel


def _compute_spans(input_size: int, output_size: int,
                   scale: float, translate: float,
                   kernel: Tuple[float, Callable[[np.ndarray], np.ndarray]],
                   antialias: bool) -> Tuple[np.ndarray, np.ndarray]:
  """
  Computes the locations and weights of the spans of a 1D input image.
  Returns:
    A `(starts, weights)` tuple of ndarrays, where `starts` has shape
    `[output_size]` and `weights` has shape `[output_size, span_size]`.
  """
  radius, kernel_fn = kernel
  inv_scale = 1. / scale
  # When downsampling the kernel should be scaled since we want to low pass
  # filter and interpolate, but when upsampling it should not be since we only
  # want to interpolate.
  kernel_scale = max(inv_scale, 1.) if antialias else 1.
  span_size = min(2 * int(math.ceil(radius * kernel_scale)) + 1, input_size)

  sample_f = (np.arange(output_size) + 0.5) * inv_scale * (1. - translate)
  span_start = np.ceil(sample_f - radius * kernel_scale - 0.5)
  span_start = np.clip(span_start, 0, input_size - span_size)
  kernel_pos = (span_start - sample_f)[:, None] + np.arange(span_size) + 0.5
  weight = kernel_fn(kernel_pos / kernel_scale)
  total_weight_sum = np.sum(weight, axis=1, keepdims=True)
  weights = np.where(
      np.abs(total_weight_sum) > 1000. * np.finfo(np.float32).min,
      weight / total_weight_sum, 0)
  return span_start.astype(np.int32), weights


def _scale_and_translate(x, output_shape, scale, translate, kernel,
                         antialias):
  input_shape = x.shape
  assert len(input_shape) == len(output_shape)
  assert len(input_shape) == len(scale)
  assert len(input_shape) == len(translate)
  spatial_dims = np.nonzero(
      np.not_equal(input_shape, output_shape) |
      np.not_equal(scale, 1) |
      np.not_equal(translate, 0))[0]
  if len(spatial_dims) == 0:
    return x
  output_spatial_shape = tuple(np.array(output_shape)[spatial_dims])
  indices = []
  contractions = []
  slice_shape = list(input_shape)
  in_indices = list(range(len(output_shape) + len(spatial_dims)))
  out_indices = list(range(len(output_shape)))
  for i, d in enumerate(spatial_dims):
    m = input_shape[d]
    n = output_shape[d]
    starts, weights = _compute_spans(m, n, scale[d], translate[d],
                                     kernel, antialias=antialias)
    starts = lax.broadcast_in_dim(starts, output_spatial_shape + (1,), (i,))
    slice_shape[d] = weights.shape[1]
    indices.append(starts.astype(np.int32))
    contractions.append(weights.astype(x.dtype))
    contractions.append([len(output_shape) + i, d])
    out_indices[d] = len(output_shape) + i
  index = lax.concatenate(indices, len(output_spatial_shape))
  dnums = lax.GatherDimensionNumbers(
      offset_dims=tuple(range(len(output_shape))),
      collapsed_slice_dims=(),
      start_index_map=tuple(spatial_dims))
  out = lax.gather(x, index, dnums, slice_shape)
  contractions.append(out_indices)
  return jnp.einsum(out, in_indices, *contractions,
                    precision=lax.Precision.HIGHEST)


class ResizeMethod(enum.Enum):
  LINEAR = 1
  LANCZOS3 = 2
  LANCZOS5 = 3
  CUBIC = 4

  @staticmethod
  def from_string(s: str):
    if s in ['linear', 'bilinear', 'trilinear', 'triangle']:
      return ResizeMethod.LINEAR
    elif s == 'lanczos3':
      return ResizeMethod.LANCZOS3
    elif s == 'lanczos5':
      return ResizeMethod.LANCZOS5
    elif s in ['cubic', 'bicubic', 'tricubic']:
      return ResizeMethod.CUBIC
    else:
      raise ValueError(f'Unknown resize method "{s}"')

_kernels = {}
_kernels[ResizeMethod.LINEAR] = _triangle_kernel()
_kernels[ResizeMethod.LANCZOS3] = _lanczos_kernel(3.)
_kernels[ResizeMethod.LANCZOS5] = _lanczos_kernel(5.)
_kernels[ResizeMethod.CUBIC] = _keys_cubic_kernel()


@partial(jit, static_argnums=(1, 2, 3))
def _resize(image, shape: Sequence[int], method: Union[str, ResizeMethod],
            antialias: bool):
  if len(shape) != image.ndim:
    msg = ('shape must have length equal to the number of dimensions of x; '
           f' {shape} vs {image.shape}')
    raise ValueError(msg)
  kernel = _kernels[ResizeMethod.from_string(method) if isinstance(method, str)
                    else method]
  scale = [float(o) / i for o, i in zip(shape, image.shape)]
  if not jnp.issubdtype(image.dtype, jnp.inexact):
    image = lax.convert_element_type(image, jnp.result_type(image, jnp.float32))
  return _scale_and_translate(image, shape, scale, [0.] * image.ndim, kernel,
                              antialias)

def resize(image, shape: Sequence[int], method: Union[str, ResizeMethod],
           antialias: bool = True):
  """Image resize.

  The ``method`` argument expects one of the following resize methods:

  ``ResizeMethod.LINEAR``, ``"linear"``, ``"bilinear"``, ``"trilinear"``, ``"triangle"``
    `Linear interpolation`_. If ``antialias`` is ``True``, uses a triangular
    filter when downsampling.

  ``ResizeMethod.CUBIC``, ``"cubic"``, ``"bicubic"``, ``"tricubic"``
    `Cubic interpolation`_, using the Keys cubic kernel.

  ``ResizeMethod.LANCZOS3``, ``"lanczos3"``
    `Lanczos resampling`_, using a kernel of radius 3.

  ``ResizeMethod.LANCZOS5``, ``"lanczos5"``
    `Lanczos resampling`_, using a kernel of radius 5.

  .. _Linear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
  .. _Cubic interpolation: https://en.wikipedia.org/wiki/Bicubic_interpolation
  .. _Lanczos resampling: https://en.wikipedia.org/wiki/Lanczos_resampling

  Args:
    image: a JAX array.
    shape: the output shape, as a sequence of integers with length equal to
      the number of dimensions of `image`. Note that :func:`resize` does not
      distinguish spatial dimensions from batch or channel dimensions, so this
      includes all dimensions of the image. To represent a batch or a channel
      dimension, simply leave that element of the shape unchanged.
    method: the resizing method to use; either a ``ResizeMethod`` instance or a
      string. Available methods are: LINEAR, LANCZOS3, LANCZOS5, CUBIC.
    antialias: should an antialiasing filter be used when downsampling? Defaults
      to ``True``. Has no effect when upsampling.
  Returns:
    The resized image.
  """
  return _resize(image, shape, method, antialias)

