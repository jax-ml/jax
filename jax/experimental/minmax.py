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

"""Optimizers for use with JAX.

This short module contains some convenient optimizer definitions, specifically
initialization and update functions, which can be used with ndarrays or
arbitrarily-nested tuple/list/dicts of ndarrays.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

from jax.core import pack
from jax.tree_util import tree_map, tree_multimap
import jax.numpy as np


def optimizer(opt_maker):
  """Decorator to make an optimizer map over tuple/list/dict containers."""
  def tree_opt_maker(*args, **kwargs):
    init_fun, update_fun = opt_maker(*args, **kwargs)
    def fmapped_init_fun(x0_tree):
      return tree_map(lambda x0: pack(init_fun(x0)), x0_tree)
    def fmapped_update_fun(i, grad_tree, state_tree):
      update = lambda g, state: pack(update_fun(i, g, *state))
      return tree_multimap(update, grad_tree, state_tree)
    return fmapped_init_fun, fmapped_update_fun
  return tree_opt_maker

def iterate(state_tree):
  """Extract the current iterate from an optimizer state."""
  return tree_map(lambda state: tuple(state)[0], state_tree)
get_params = iterate

@optimizer
def sgd(step_size):
  """Init and update step functions for stochastic gradient descent."""
  def init_fun(x0):
    return (x0,)
  def update_fun(i, g, x):
    return (x - step_size * g,)
  return init_fun, update_fun

@optimizer
def momentum(step_size, mass):
  """Init and update step functions for SGD with Nesterov momentum."""
  def init_fun(x0):
    v0 = np.zeros_like(x0)
    return x0, v0
  def update_fun(i, g, x, velocity):
    velocity = mass * velocity - (1. - mass) * g
    x = x + step_size * velocity
    return x, velocity
  return init_fun, update_fun

@optimizer
def rmsprop(step_size, gamma=0.9, eps=1e-8):
  """Init and update step functions for RMSProp."""
  def init_fun(x0):
    avg_sq_grad = np.ones_like(x0)
    return x0, avg_sq_grad
  def update_fun(i, g, x, avg_sq_grad):
    avg_sq_grad = avg_sq_grad * gamma + g**2 * (1. - gamma)
    x = x - step_size * g / (np.sqrt(avg_sq_grad) + eps)
    return x, avg_sq_grad
  return init_fun, update_fun

@optimizer
def adam(step_size, b1=0.9, b2=0.999, eps=1e-8):
  """Init and update step functions for Adam."""
  def init_fun(x0):
    m0 = np.zeros_like(x0)
    v0 = np.zeros_like(x0)
    return x0, m0, v0
  def update_fun(i, g, x, m, v):
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i + 1))
    x = x - step_size*mhat / (np.sqrt(vhat) + eps)
    return x, m, v
  return init_fun, update_fun

def run_optimizer(loss, infeed, update_fun, state):
  """A convenience function for running optimizers with iterated map-reduce.

  Args:
    loss: a scalar-valued loss function taking two aguments, the current iterate
      and a data value.
    infeed: an infeed instance supplying the data stream.
    update_fun: a function that has signature update_fun(i, grad, state) where
      i is the integer iteration count, grad is the gradient of the loss at the
      current iterate, and state is the current optimizer state.
    state: the initial optimizer state.

  Returns:
    A pair (x, state) where is the final iterate and state represents the final
    optimizer state.
  """
  map_fun = lambda _, state, batch: grad(loss)(iterate(state), batch)
  state = fax.iterated_map_reduce(state, map_fun, update_fun, infeed)
  return iterate(state), state
