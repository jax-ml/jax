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

"""
Common neural network layer initializers, consistent with definitions
used in Keras and Sonnet.
"""

from __future__ import absolute_import
from __future__ import division

from functools import partial

import numpy as onp

from jax import lax
from jax import random
import jax.numpy as np

def zeros(key, shape, dtype=np.float32): return np.zeros(shape, dtype)
def ones(key, shape, dtype=np.float32): return np.ones(shape, dtype)

def uniform(scale=1e-2):
  def init(key, shape, dtype=np.float32):
    return random.uniform(key, shape, dtype) * scale
  return init

def normal(stddev=1e-2):
  def init(key, shape, dtype=np.float32):
    return random.normal(key, shape, dtype) * stddev
  return init

def _compute_fans(shape, in_axis=-2, out_axis=-1):
  receptive_field_size = onp.prod(shape) / shape[in_axis] / shape[out_axis]
  fan_in = shape[in_axis] * receptive_field_size
  fan_out = shape[out_axis] * receptive_field_size
  return fan_in, fan_out

def variance_scaling(scale, mode, distribution, in_axis=-2, out_axis=-1):
  def init(key, shape, dtype=np.float32):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in": denominator = fan_in
    elif mode == "fan_out": denominator = fan_out
    elif mode == "fan_avg": denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = np.array(scale / denominator, dtype=dtype)
    if distribution == "truncated_normal":
      # constant is stddev of standard normal truncated to (-2, 2)
      stddev = np.sqrt(variance) / np.array(.87962566103423978, dtype)
      return random.truncated_normal(key, -2, 2, shape, dtype) * stddev
    elif distribution == "normal":
      return random.normal(key, shape, dtype) * np.sqrt(variance)
    elif distribution == "uniform":
      return random.uniform(key, shape, dtype, -1) * onp.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")
  return init

xavier_uniform = glorot_uniform = partial(variance_scaling, 1.0, "fan_avg", "uniform")
xavier_normal = glorot_normal = partial(variance_scaling, 1.0, "fan_avg", "truncated_normal")
lecun_uniform = partial(variance_scaling, 1.0, "fan_in", "uniform")
lecun_normal = partial(variance_scaling, 1.0, "fan_in", "truncated_normal")
kaiming_uniform = he_uniform = partial(variance_scaling, 2.0, "fan_in", "uniform")
kaiming_normal = he_normal = partial(variance_scaling, 2.0, "fan_in", "truncated_normal")

def orthogonal(scale=1.0, column_axis=-1):
  """
  Construct an initializer for uniformly distributed orthogonal matrices.
  
  If the shape is not square, the matrices will have orthonormal rows or columns
  depending on which side is smaller.
  """
  def init(key, shape, dtype=np.float32):
    if len(shape) < 2:
      raise ValueError("orthogonal initializer requires at least a 2D shape")
    n_rows, n_cols = onp.prod(shape) // shape[column_axis], shape[column_axis]
    matrix_shape = (n_cols, n_rows) if n_rows < n_cols else (n_rows, n_cols)
    A = random.normal(key, matrix_shape, dtype)
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R)) # needed for a uniform distribution
    if n_rows < n_cols: Q = Q.T
    Q = np.reshape(Q, onp.delete(shape, column_axis) + (shape[column_axis],))
    Q = np.moveaxis(Q, -1, column_axis)
    return scale * Q
  return init
