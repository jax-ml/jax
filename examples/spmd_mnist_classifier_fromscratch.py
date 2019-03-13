# Copyright 2018 Google LLC
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

"""A basic MNIST example using Numpy and JAX.

The primary aim here is simplicity and minimal dependencies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import time

import numpy.random as npr

from jax import jit, grad, pmap, replicate, unreplicate
from jax.config import config
from jax.scipy.special import logsumexp
from jax.lib import xla_bridge
from jax import lax
import jax.numpy as np
from examples import datasets


def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def predict(params, inputs):
  activations = inputs
  for w, b in params[:-1]:
    outputs = np.dot(activations, w) + b
    activations = np.tanh(outputs)

  final_w, final_b = params[-1]
  logits = np.dot(activations, final_w) + final_b
  return logits - logsumexp(logits, axis=1, keepdims=True)

def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -np.mean(preds * targets)

@jit
def accuracy(params, batch):
  inputs, targets = batch
  target_class = np.argmax(targets, axis=1)
  predicted_class = np.argmax(predict(params, inputs), axis=1)
  return np.mean(predicted_class == target_class)


if __name__ == "__main__":
  layer_sizes = [784, 1024, 1024, 10]
  param_scale = 0.1
  step_size = 0.001
  num_epochs = 10
  batch_size = 128

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)


  # For this manual SPMD example, we get the number of devices (e.g. GPUs or
  # TPUs) that we're using, and use it to reshape data minibatches.
  num_devices = xla_bridge.device_count()
  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        images, labels = train_images[batch_idx], train_labels[batch_idx]
        # For this SPMD example, we reshape the data batch dimension into two
        # batch dimensions, one of which is mapped over parallel devices.
        batch_size_per_device, ragged = divmod(images.shape[0], num_devices)
        if ragged:
          msg = "batch size must be divisible by device count, got {} and {}."
          raise ValueError(msg.format(batch_size, num_devices))
        shape_prefix = (num_devices, batch_size_per_device)
        images = images.reshape(shape_prefix + images.shape[1:])
        labels = labels.reshape(shape_prefix + labels.shape[1:])
        yield images, labels
  batches = data_stream()

  @partial(pmap, axis_name='batch')
  def spmd_update(params, batch):
    grads = grad(loss)(params, batch)
    # We compute the total gradients, summing across the device-mapped axis,
    # using the `lax.psum` SPMD primitive, which does a fast all-reduce-sum.
    grads = [(lax.psum(dw, 'batch'), lax.psum(db, 'batch')) for dw, db in grads]
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

  # We replicate parameters out across devices. (Check the implementation of
  # replicate; analogous to device_put, it's a simple wrapper around pmap.)
  params = replicate(init_random_params(param_scale, layer_sizes))

  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      params = spmd_update(params, next(batches))
    epoch_time = time.time() - start_time

    # We evaluate using the jitted `accuracy` function (not using pmap) by
    # grabbing just one of the replicated parameter values.
    train_acc = accuracy(unreplicate(params), (train_images, train_labels))
    test_acc = accuracy(unreplicate(params), (test_images, test_labels))
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
