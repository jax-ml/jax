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


"""An MNIST example for single-program multiple-data (SPMD) spatial parallelism.

The aim here is to illustrate how to use JAX's `pmap` to express and execute
SPMD programs for data parallelism along a spatial dimension (rather than a
batch dimension).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import itertools

import numpy as onp

import jax.numpy as np
from jax.config import config
from jax import jit, grad
from jax import lax, random
from jax.experimental import stax
from jax.experimental.stax import Relu, Flatten
from examples import datasets


def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -np.mean(preds * targets)

def accuracy(params, batch):
  inputs, targets = batch
  target_class = np.argmax(targets, axis=1)
  predicted_class = np.argmax(predict(params, inputs), axis=1)
  return np.mean(predicted_class == target_class)


# spmd utility functions

def axis_size(axis_name):
  return lax.psum(1, axis_name)

def spmd_logsoftmax(x, axis_name):
  max_x = lax.pmax(x, axis_name)
  log_normalizer = max_x + np.log(lax.psum(np.exp(x - max_x), axis_name))
  return x - log_normalizer

def spmd_glorot(axis_name, out_axis=0, in_axis=1, scale=onp.sqrt(2)):
  def init(rng, shape):
    axis_size = lax.psum(1, axis_name)
    fan_in, fan_out = shape[in_axis] * axis_size, shape[out_axis] * axis_size
    size = onp.prod(onp.delete(shape, [in_axis, out_axis]))
    std = scale / np.sqrt((fan_in + fan_out) / 2. * size)
    return std * random.normal(rng, shape, dtype=np.float32)
  return init

def send_right(x, axis_name):
  left_perm = [(i, (i + 1)) for i in range(device_count - 1)]
  return lax.ppermute(x, perm=left_perm, axis_name=axis_name)

def send_left(x, axis_name):
  left_perm = [((i + 1), i) for i in range(device_count - 1)]
  return lax.ppermute(x, perm=left_perm, axis_name=axis_name)


# spmd layers for spatial sharding

def LogSoftmax(axis_name):
  def init_fun(rng, input_shape):
    return (input_shape, ())
  def apply_fun(_, input_shard, **kwargs):
    return spmd_softmax(input_shard, axis_name)
  return init_fun, apply_fun

def Dense(axis_name, out_dim, W_init=None, b_init=stax.randn()):
  W_init = W_init or spmd_glorot(axis_name)
  def init_fun(rng, input_shape):
    sz = axis_size(axis_name)
    output_shape = input_shape[:-1] + (out_dim,)
    k1, k2 = random.split(random.fold_in(rng, pxla.axis_index(axis_name)))
    W_shard = W_init(k1, (input_shape[-1] // sz, output_shape[-1] // sz))
    b_shard = b_init(k2, (output_shape[-1] // sz,))
    return output_shape, (W_shard, b_shard)
  def apply_fun(params, input_shard, **kwargs):
    W_shard, b_shard = params
    return lax.psum(np.dot(input_shard, W_shard), axis_name) + b_shard
  return init_fun, apply_fun

def Conv(axis_name, out_chan, filter_shape, strides=None, padding='valid',
         W_init=None, b_init=stax.randn(1e-6)):
  kernel_height, kernel_width = filter_shape
  strides = strides or (1, 1)
  ribbon_size = kernel_height // 2
  init_fun, _ = stax.Conv(out_chan, filter_shape, strides, padding, W_init, b_init)
  def apply_fun(params, input_shard, **kwargs):
    # TODO test this code in a separate unit test!
    W, b = params
    left, right = input_shard[:, :ribbon_size], input_shard[:, -ribbon_size:]
    right, left = send_left(left, axis_name), send_right(right, axis_name)
    enlarged_input_shard = np.concatenate([left, input_shard, right], 1)
    return b + lax.conv_general_dilated(
        enlarged_input_shard, W, strides, padding,
        (1, 1), (1, 1), ("NHWC", "HWIO", "NHWC"))
  return init_fun, apply_fun

# def MaxpoolSpatialShard(axis_name, window_shape):
#   pass  # TODO


class AxisName(object): pass
ax = AxisName()

init_random_params, predict = stax.serial(
    # Conv(ax, 32, (3, 3), padding='SAME'), Relu,
    # Conv(ax, 64, (3, 3), padding='SAME'), Relu,
    # MaxpoolSpatialShard(ax, (2, 2)),
    # Flatten,
    Dense(ax, 128), Relu
    # Dense(ax, 10),
    # LogSoftmax(ax),
)

rng = random.PRNGKey(0)
pmap(lambda: init_random_params(rng, (128, 28 * 28)),
     axis_name=ax, axis_size=num_devices)()
