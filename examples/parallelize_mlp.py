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

"""A basic MNIST example, automatically parallelized with `papply`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import time

import jax.numpy as np
from jax import jit, papply, serial_pmap, make_jaxpr
from jax import lax
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
from examples import datasets


def loss(params, inputs, targets):
  preds = predict(params, inputs)
  return -np.mean(preds * targets)

init_random_params, predict = stax.serial(
    Dense(1024), Relu,
    Dense(1024), Relu,
    Dense(10), LogSoftmax)

if __name__ == "__main__":
  step_size = 0.001
  batch_size = 128

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    for i in range(num_batches):
      batch_idx = slice(i * batch_size, (i + 1) * batch_size)
      yield train_images[batch_dx], train_labels[batch_idx]
  batches = data_stream()

  _, params = init_random_params((-1, 28 * 28))
  loss_ = partial(loss, params)
  spmd_loss, axis_name = papply(loss_, train_images.shape[0])
  pmap_loss = serial_pmap(spmd_loss, axis_name)

  print("original:", train_images.shape)
  print(make_jaxpr(loss_)(train_images, train_labels))
  print()
  print("spmd:", train_images[0].shape)
  print(make_jaxpr(spmd_loss)(train_images[0], train_labels[0]))
  print()
  print("pmap:", train_images.shape)
  print(make_jaxpr(pmap_loss)(train_images, train_labels))

  closs = jit(pmap_loss)
  start_time = time.time()
  train_acc = closs(train_images, train_labels)
  test_acc = closs(test_images, test_labels)
  time_delta = time.time() - start_time
  print("Took {:0.2f} sec".format(time_delta))
  print("Training set loss {}".format(train_acc))
  print("Test set loss {}".format(test_acc))
