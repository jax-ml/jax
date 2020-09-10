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
"""Shows how to mix JAX for training and TensorFlow for evaluation."""

from absl import app
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from jax.config import config
config.config_with_absl()

from jax.experimental import jax2tf

TRAIN_EXAMPLES = 60000
BATCH_SIZE = 1000
NUM_EPOCHS = 20
LEARNING_RATE = 0.05

# ==============================================================================
# Input pipeline := We use TFDS to create a generator of NumPy arrays.
# ==============================================================================


def load_dataset(split, *, is_training, batch_size):
  """Loads the dataset as a generator of batches."""
  ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return tfds.as_numpy(ds)

# ==============================================================================
# Training := We use JAX to define the functions needed for training.
# ==============================================================================


@jax.jit
def predict_jax(params, x):
  x = jnp.reshape(x, [x.shape[0], -1])
  x = x.astype(jnp.float32) / 255.
  for i, (w, b) in enumerate(params):
    if i:
      x = jax.nn.relu(x)
    x = jnp.dot(x, w) + b
  return x


def loss_fn_jax(params, batch):
  logits = predict_jax(params, batch["image"])
  labels = jax.nn.one_hot(batch["label"], 10)
  softmax_xent = -jnp.mean(labels * jax.nn.log_softmax(logits))
  l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
  return softmax_xent + 1e-4 * l2_loss


@jax.jit
def train_step_jax(params, batch):
  loss, grads = jax.value_and_grad(loss_fn_jax)(params, batch)
  params = jax.tree_multimap(lambda p, g: p - g * LEARNING_RATE, params, grads)
  return loss, params


def train_epoch_jax(params, train_dataset):
  for _ in range(TRAIN_EXAMPLES // BATCH_SIZE):
    loss, params = train_step_jax(params, next(train_dataset))
  return loss, params

# ==============================================================================
# Evaluation := While we define our evaluation metric (top-1 accuracy) using JAX
# we actually evaluate it using TensorFlow via go/jax2tf.
# ==============================================================================


@tf.function
@jax2tf.convert
def accuracy_fn_tf(params, batch):
  logits = predict_jax(params, batch["image"])
  return jnp.mean(jnp.argmax(logits, axis=-1) == batch["label"])

# ==============================================================================
# Training loop := is a mixture of JAX code (to initialize the model and train)
# and TF (evaluating the model at every epoch).
# ==============================================================================


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Initial state for a LeNet-300-100 MLP.
  k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
  w_init = (
      lambda k, s: jax.random.truncated_normal(k, -2, 2, s) / np.sqrt(s[-1]))
  params = ((w_init(k1, [28 * 28, 300]), jnp.zeros([300])),
            (w_init(k2, [300, 100]), jnp.zeros([100])),
            (w_init(k3, [100, 10]), jnp.zeros([10])))

  train_dataset = load_dataset("train", is_training=True, batch_size=BATCH_SIZE)
  eval_batch = next(load_dataset("test", is_training=False, batch_size=10000))

  for epoch in range(1, NUM_EPOCHS + 1):
    loss, params = train_epoch_jax(params, train_dataset)
    accuracy = accuracy_fn_tf(params, eval_batch)
    print(f"[Epoch {epoch:02d}] loss: {loss:.5f} test acc: {accuracy:.5f}")

  accuracy = accuracy_fn_tf(params, eval_batch)
  print(f"Final accuracy: {accuracy:.5f}%")

if __name__ == "__main__":
  app.run(main)
