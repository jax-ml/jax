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

"""Common functions for neural network libraries."""

# flake8: noqa: F401

from jax.numpy import tanh
from . import initializers
from jax._src.nn.functions import (
  celu,
  elu,
  gelu,
  glu,
  hard_sigmoid,
  hard_silu,
  hard_swish,
  hard_tanh,
  leaky_relu,
  log_sigmoid,
  log_softmax,
  logsumexp,
  normalize,
  one_hot,
  relu,
  relu6,
  selu,
  sigmoid,
  soft_sign,
  softmax,
  softplus,
  silu,
  swish,
)
