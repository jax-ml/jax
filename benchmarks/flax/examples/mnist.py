# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Default Hyperparameter configuration.

`get_config`
https://github.com/google/flax/blob/a1451a0e55dfc55b4bc9ca42f88bf28744357d8c/examples/mnist/configs/default.py
"""
from typing import Any

from flax.examples.mnist import train
import jax
import jax.numpy as jnp
import ml_collections


def get_fake_batch(batch_size: int) -> tuple[jnp.ndarray, ...]:
  """Returns fake data for the given batch size."""
  rng = jax.random.PRNGKey(0)
  images = jax.random.randint(rng, (batch_size, 28, 28, 1), 0, 255, jnp.uint8)
  labels = jax.random.randint(rng, (batch_size,), 0, 10, jnp.int32)
  return images, labels


def get_apply_fn_and_args() -> tuple[jax.stages.Traced, tuple[Any, ...]]:
  """Returns the apply function and args for the model."""
  config = get_config()
  batch = get_fake_batch(config.batch_size)
  train_state = train.create_train_state(jax.random.key(0), config)
  return train.apply_model, (train_state, *batch)


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.learning_rate = 0.1
  config.momentum = 0.9
  config.batch_size = 128
  config.num_epochs = 10
  return config


def metrics():
  return []
