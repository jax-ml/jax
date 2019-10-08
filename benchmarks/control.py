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
Benchmarks for model-predictive linear control
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import jax
from jax import lax
import jax.numpy as np
import jax.ops as jo


def trajectory(dynamics, U, x0):
  '''Unrolls `X[t+1] = dynamics(t, X[t], U[t])`, where `X[0] = x0`.'''
  T, _ = U.shape
  d, = x0.shape

  X = np.zeros((T + 1, d))
  X = jo.index_update(X, jo.index[0], x0)

  def loop(t, X):
    x = dynamics(t, X[t], U[t])
    X = jo.index_update(X, jo.index[t + 1], x)
    return X

  return lax.fori_loop(0, T, loop, X)
