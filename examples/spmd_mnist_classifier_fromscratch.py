# Copyright 2018 The JAX Authors.
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
import time

from jax import NamedSharding
import numpy as np
import numpy.random as npr
import jax
from jax import jit, grad
from jax.sharding import PartitionSpec as P, AxisType, reshard
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import datasets


def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
  return [
    (scale * rng.randn(m, n), scale * rng.randn(n))
    for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
  ]


def predict(params, inputs):
  activations = inputs
  for w, b in params[:-1]:
    outputs = jnp.dot(activations, w) + b
    activations = jnp.tanh(outputs)

  final_w, final_b = params[-1]
  logits = jnp.dot(activations, final_w) + final_b
  return logits - logsumexp(logits, axis=1, keepdims=True)


def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))


@partial(jax.jit, donate_argnums=0)
def train_step(params, batch):
  grads = grad(loss)(params, batch)
  return [
    (w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)
  ]


@jit
def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)


if __name__ == "__main__":
  layer_sizes = [784, 1024, 1024, 10]
  param_scale = 0.1
  step_size = 0.001
  num_epochs = 10
  batch_size = 128

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]

  num_devices = jax.device_count()
  print(f"Using {num_devices} devices")

  if batch_size % num_devices != 0:
    batch_size = (batch_size // num_devices) * num_devices
    print(f"Adjusting batch size to {batch_size} for divisibility")

  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  devices = np.array(jax.devices())
  mesh = jax.make_mesh(
    (jax.device_count(),), ("batch",), axis_types=(AxisType.Explicit,)
  )

  replicated_sharding = NamedSharding(mesh, P())
  data_sharding = NamedSharding(mesh, P("batch"))

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size : (i + 1) * batch_size]
        images_np, labels_np = train_images[batch_idx], train_labels[batch_idx]

        current_batch_size = images_np.shape[0]
        if current_batch_size < batch_size:
          pad_len = batch_size - current_batch_size
          images_np = np.concatenate([images_np, images_np[:pad_len]], axis=0)
          labels_np = np.concatenate([labels_np, labels_np[:pad_len]], axis=0)

        images = jax.device_put(images_np, data_sharding)
        labels = jax.device_put(labels_np, data_sharding)
        yield images, labels

  batches = data_stream()

  params = init_random_params(param_scale, layer_sizes)
  replicated_params = jax.device_put(params, replicated_sharding)

  for epoch in range(num_epochs):
    start_time = time.time()
    for i in range(num_batches - 1):
      print(f"Batch no {i+1} of {num_batches}")
      batch = next(batches)
      with jax.set_mesh(mesh):
        replicated_params = train_step(replicated_params, batch)
    epoch_time = time.time() - start_time

    # Reshard train_images, train_labels, test_images, test_labels
    sharded_train_images = reshard(train_images, data_sharding)
    sharded_train_labels = reshard(train_labels, data_sharding)
    sharded_test_images = reshard(test_images, data_sharding)
    sharded_test_labels = reshard(test_labels, data_sharding)

    train_acc = accuracy(
      replicated_params, (sharded_train_images, sharded_train_labels)
    )
    test_acc = accuracy(replicated_params, (sharded_test_images, sharded_test_labels))
    print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
    print(f"Training set accuracy {train_acc}")
    print(f"Test set accuracy {test_acc}")

    if epoch < num_epochs - 1:
      batches = data_stream()
      print(f"Batch no {0} of {num_batches}")
      replicated_params = train_step(replicated_params, next(batches))
