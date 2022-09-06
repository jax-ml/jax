# Copyright 2022 Google LLC
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
"""Actor-critic model from Flax PPO example, see:

https://github.com/google/flax/tree/main/examples/ppo
"""

from flax import linen as nn
import jax.numpy as jnp


class ActorCritic(nn.Module):
  """Class defining the actor-critic model."""

  num_outputs: int

  @nn.compact
  def __call__(self, x):
    """Define the convolutional network architecture.

    Architecture originates from "Human-level control through deep reinforcement
    learning.", Nature 518, no. 7540 (2015): 529-533.
    Note that this is different than the one from  "Playing atari with deep
    reinforcement learning." arxiv.org/abs/1312.5602 (2013)

    Network is used to both estimate policy (logits) and expected state value;
    in other words, hidden layers' params are shared between policy and value
    networks, see e.g.:
    github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py
    """
    dtype = jnp.float32
    x = x.astype(dtype) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name='conv3',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=512, name='hidden', dtype=dtype)(x)
    x = nn.relu(x)
    logits = nn.Dense(features=self.num_outputs, name='logits', dtype=dtype)(x)
    policy_log_probabilities = nn.log_softmax(logits)
    value = nn.Dense(features=1, name='value', dtype=dtype)(x)
    return policy_log_probabilities, value
