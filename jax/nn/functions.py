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

"""Shared neural network activations and other functions."""

from __future__ import absolute_import
from __future__ import division

import numpy as onp

from jax import lax
from jax import random
from jax.scipy.special import expit
import jax.numpy as np
from jax import jarrett

# activations

def relu(x): return np.maximum(x, 0)
def softplus(x): return np.logaddexp(x, 0)
def soft_sign(x): return x / (np.abs(x) + 1)
def sigmoid(x): return expit(x)
def swish(x): return x * sigmoid(x)
def log_sigmoid(x): return -softplus(-x)

def elu(x, alpha=1.0):
  safe_x = lax.select(x > 0, np.zeros(onp.shape(x)), x)
  return lax.select(x > 0, x, alpha * np.expm1(safe_x))

def leaky_relu(x, negative_slope=1e-2):
  return lax.select(x >= 0, x, negative_slope * x)

def hard_tanh(x):
  shape = onp.shape(x)
  ones = np.full(shape, 1.)
  minus_ones = np.full(shape, -1.)
  return lax.select(x > 1, ones, lax.select(x < -1, minus_ones, x))

def celu(x, alpha=1.0):
  """Continuously-differentiable exponential linear unit activation"""
  return lax.select(x > 0, x, alpha * np.expm1(x / alpha))

def selu(x):
  """Scaled exponential linear unit activation"""
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * elu(x, alpha)

def gelu(x):
  """GELU activation function.

  We explicitly use the approximation rather than the exact formulation for
  speed. See: https://arxiv.org/abs/1606.08415 Section 2.
  """
  cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))))
  return x * cdf

def glu(x, axis=-1):
  """Gated linear unit activation"""
  size = x.shape[axis]
  assert size % 2 == 0, "axis size must be divisible by 2"
  return x[..., :size] * sigmoid(x[..., size:])

# other functions

def log_softmax(x, axis=-1):
  shifted = x - x.max(axis, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis, keepdims=True))

def softmax(x, axis=-1):
  unnormalized = np.exp(x - x.max(axis, keepdims=True))
  return unnormalized / unnormalized.sum(axis, keepdims=True)

def normalize(x, axis=-1, mean=None, variance=None, epsilon=1e-5):
  """Normalize an array by subtracting mean and dividing by sqrt(var)."""
  if mean is None:
    mean = np.mean(x, axis, keepdims=True)
  if variance is None:
    # this definition is traditionally seen as less accurate than np.var's
    # mean((x - mean(x))**2) but may be faster and even, given typical
    # activation distributions and low-precision arithmetic, more accurate
    # when used in neural network normalization layers
    variance = np.mean(x**2, axis, keepdims=True) - mean**2
  return (x - mean) * lax.rsqrt(variance + epsilon)
