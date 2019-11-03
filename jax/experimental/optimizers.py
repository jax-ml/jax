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

This module contains some convenient optimizer definitions, specifically
initialization and update functions, which can be used with ndarrays or
arbitrarily-nested tuple/list/dicts of ndarrays.

An optimizer is modeled as an ``(init_fun, update_fun, get_params)`` triple of
functions, where the component functions have these signatures:

::

  init_fun(params)

  Args:
    params: pytree representing the initial parameters.

  Returns:
    A pytree representing the initial optimizer state, which includes the
    initial parameters and may also include auxiliary values like initial
    momentum. The optimizer state pytree structure generally differs from that
    of `params`.

::

  update_fun(step, grads, opt_state)

  Args:
    step: integer representing the step index.
    grads: a pytree with the same structure as `get_params(opt_state)`
      representing the gradients to be used in updating the optimizer state.
    opt_state: a pytree representing the optimizer state to be updated.

  Returns:
    A pytree with the same structure as the `opt_state` argument representing
    the updated optimizer state.

::

  get_params(opt_state)

  Args:
    opt_state: pytree representing an optimizer state.

  Returns:
    A pytree representing the parameters extracted from `opt_state`, such that
    the invariant `params == get_params(init_fun(params))` holds true.


Notice that an optimizer implementation has a lot of flexibility in the form of
opt_state: it just has to be a pytree of JaxTypes (so that it can be passed to
the JAX transforms defined in api.py) and it has to be consumable by update_fun
and get_params.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools
import operator

from six.moves import reduce

import jax.numpy as np
from jax.util import partial, safe_zip, safe_map, unzip2
from jax import tree_util
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten, register_pytree_node)

map = safe_map
zip = safe_zip

# The implementation here basically works by flattening pytrees. There are two
# levels of pytrees to think about: the pytree of params, which we can think of
# as defining an "outer pytree", and a pytree produced by applying init_fun to
# each leaf of the params pytree, which we can think of as the "inner pytrees".
# Since pytrees can be flattened, that structure is isomorphic to a list of
# lists (with no further nesting).

pack = tuple
OptimizerState = namedtuple("OptimizerState", ["packed_state", "tree_def", "subtree_defs"])
register_pytree_node(OptimizerState, lambda xs: ((xs.packed_state,),
                                                 (xs.tree_def, xs.subtree_defs)),
                     lambda data, xs: OptimizerState(xs[0], data[0], data[1]))

def optimizer(opt_maker):
  """Decorator to make an optimizer defined for arrays generalize to containers.

  With this decorator, you can write init, update, and get_params functions that
  each operate only on single arrays, and convert them to corresponding
  functions that operate on pytrees of parameters. See the optimizers defined in
  optimizers.py for examples.

  Args:
    opt_maker: a function that returns an ``(init_fun, update_fun, get_params)``
      triple of functions that might only work with ndarrays, as per

      .. code-block:: haskell

          init_fun :: ndarray -> OptStatePytree ndarray
          update_fun :: OptStatePytree ndarray -> OptStatePytree ndarray
          get_params :: OptStatePytree ndarray -> ndarray

  Returns:
    An ``(init_fun, update_fun, get_params)`` triple of functions that work on
    arbitrary pytrees, as per

    .. code-block:: haskell

          init_fun :: ParameterPytree ndarray -> OptimizerState
          update_fun :: OptimizerState -> OptimizerState
          get_params :: OptimizerState -> ParameterPytree ndarray

    The OptimizerState pytree type used by the returned functions is isomorphic
    to ``ParameterPytree (OptStatePytree ndarray)``, but may store the state
    instead as e.g. a partially-flattened data structure for performance.
  """
  @functools.wraps(opt_maker)
  def tree_opt_maker(*args, **kwargs):
    init, update, get_params = opt_maker(*args, **kwargs)

    @functools.wraps(init)
    def tree_init(x0_tree):
      x0_flat, tree = tree_flatten(x0_tree)
      initial_states = [init(x0) for x0 in x0_flat]
      states_flat, subtrees = unzip2(map(tree_flatten, initial_states))
      packed_state = pack(map(pack, states_flat))
      return OptimizerState(packed_state, tree, subtrees)

    @functools.wraps(update)
    def tree_update(i, grad_tree, opt_state):
      packed_state, tree, subtrees = opt_state
      grad_flat, tree2 = tree_flatten(grad_tree)
      if tree2 != tree:
        msg = ("optimizer update function was passed a gradient tree that did "
               "not match the parameter tree structure with which it was "
               "initialized: parameter tree {} and grad tree {}.")
        raise TypeError(msg.format(tree, tree2))
      states = map(tree_unflatten, subtrees, packed_state)
      new_states = map(partial(update, i), grad_flat, states)
      new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
      for subtree, subtree2 in zip(subtrees, subtrees2):
        if subtree2 != subtree:
          msg = ("optimizer update function produced an output structure that "
                 "did not match its input structure: input {} and output {}.")
          raise TypeError(msg.format(subtree, subtree2))
      new_packed_state = pack(map(pack, new_states_flat))
      return OptimizerState(new_packed_state, tree, subtrees)

    @functools.wraps(get_params)
    def tree_get_params(opt_state):
      packed_state, tree, subtrees = opt_state
      states = map(tree_unflatten, subtrees, packed_state)
      params = map(get_params, states)
      return tree_unflatten(tree, params)

    return tree_init, tree_update, tree_get_params

  return tree_opt_maker

### optimizers
@optimizer
def sgd(step_size):
  """Construct optimizer triple for stochastic gradient descent.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)

  def init(x0):
    return x0

  def update(i, g, x):
    return x - step_size(i) * g

  def get_params(x):
    return x

  return init, update, get_params

@optimizer
def momentum(step_size, mass):
  """Construct optimizer triple for SGD with momentum.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    mass: positive scalar representing the momentum coefficient.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)

  def init(x0):
    v0 = np.zeros_like(x0)
    return x0, v0

  def update(i, g, state):
    x, velocity = state
    velocity = mass * velocity + g
    x = x - step_size(i) * velocity
    return x, velocity

  def get_params(state):
    x, _ = state
    return x

  return init, update, get_params

@optimizer
def nesterov(step_size, mass):
  """Construct optimizer triple for SGD with Nesterov momentum.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    mass: positive scalar representing the momentum coefficient.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)

  def init(x0):
    v0 = np.zeros_like(x0)
    return x0, v0

  def update(i, g, state):
    x, velocity = state
    velocity = mass * velocity + g
    x = x - step_size(i) * (mass * velocity + g)
    return x, velocity

  def get_params(state):
    x, _ = state
    return x

  return init, update, get_params

@optimizer
def adagrad(step_size, momentum=0.9):
  """Construct optimizer triple for Adagrad.

  Adaptive Subgradient Methods for Online Learning and Stochastic Optimization:
  http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    momentum: optional, a positive scalar value for momentum

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)

  def init(x0):
    g_sq = np.zeros_like(x0)
    m = np.zeros_like(x0)
    return x0, g_sq, m

  def update(i, g, state):
    x, g_sq, m = state
    g_sq += g**2
    g_sq_inv_sqrt = np.where(g_sq > 0, 1. / np.sqrt(g_sq), 0.0)
    m = (1. - momentum) * (g * g_sq_inv_sqrt) + momentum * m
    x = x - step_size(i) * m
    return x, g_sq, m

  def get_params(state):
    x, _, _ = state
    return x

  return init, update, get_params

@optimizer
def rmsprop(step_size, gamma=0.9, eps=1e-8):
  """Construct optimizer triple for RMSProp.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
      gamma: Decay parameter.
      eps: Epsilon parameter.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)

  def init(x0):
    avg_sq_grad = np.zeros_like(x0)
    return x0, avg_sq_grad

  def update(i, g, state):
    x, avg_sq_grad = state
    avg_sq_grad = avg_sq_grad * gamma + g**2 * (1. - gamma)
    x = x - step_size(i) * g / np.sqrt(avg_sq_grad + eps)
    return x, avg_sq_grad

  def get_params(state):
    x, _ = state
    return x

  return init, update, get_params

@optimizer
def rmsprop_momentum(step_size, gamma=0.9, eps=1e-8, momentum=0.9):
  """Construct optimizer triple for RMSProp with momentum.

  This optimizer is separate from the rmsprop optimizer because it needs to
  keep track of additional parameters.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    gamma: Decay parameter.
    eps: Epsilon parameter.
    momentum: Momentum parameter.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)

  def init(x0):
    avg_sq_grad = np.zeros_like(x0)
    mom = np.zeros_like(x0)
    return x0, avg_sq_grad, mom

  def update(i, g, state):
    x, avg_sq_grad, mom = state
    avg_sq_grad = avg_sq_grad * gamma + g**2 * (1. - gamma)
    mom = momentum * mom + step_size(i) * g / np.sqrt(avg_sq_grad + eps)
    x = x - mom
    return x, avg_sq_grad, mom

  def get_params(state):
    x, _, _ = state
    return x

  return init, update, get_params

@optimizer
def adam(step_size, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for Adam.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)

  def init(x0):
    m0 = np.zeros_like(x0)
    v0 = np.zeros_like(x0)
    return x0, m0, v0

  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1**(i + 1))  # Bias correction.
    vhat = v / (1 - b2**(i + 1))
    x = x - step_size(i) * mhat / (np.sqrt(vhat) + eps)
    return x, m, v

  def get_params(state):
    x, m, v = state
    return x

  return init, update, get_params

@optimizer
def sm3(step_size, momentum=0.9):
  """Construct optimizer triple for SM3.

  Memory-Efficient Adaptive Optimization for Large-Scale Learning.
  https://arxiv.org/abs/1901.11150

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    momentum: optional, a positive scalar value for momentum

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)

  def splice(seq, i, x):
    lst = list(seq)
    lst[i:i + 1] = x
    return lst

  def broadcast_into(ndim, x, axis):
    idx = splice([None] * ndim, axis, [slice(None)])
    return x[tuple(idx)]

  def init(x0):
    vs = [np.zeros(sz, dtype=x0.dtype) for sz in x0.shape]
    return x0, np.zeros_like(x0), vs

  def update(i, g, state):
    x, m, vs = state
    vs = [broadcast_into(g.ndim, v, i) for i, v in enumerate(vs)]
    accum = reduce(np.minimum, vs) + g**2
    accum_inv_sqrt = np.where(accum > 0, 1. / np.sqrt(accum), 0)
    m = (1. - momentum) * (g * accum_inv_sqrt) + momentum * m
    x = x - step_size(i) * m
    vs = [accum.max(splice(range(x.ndim), j, [])) for j in range(x.ndim)]
    return x, m, vs

  def get_params(state):
    x, _, _ = state
    return x

  return init, update, get_params

### learning rate schedules
def constant(step_size):
  def schedule(i):
    return step_size

  return schedule

def exponential_decay(step_size, decay_steps, decay_rate):
  def schedule(i):
    return step_size * decay_rate**(i / decay_steps)

  return schedule

def inverse_time_decay(step_size, decay_steps, decay_rate, staircase=False):
  if staircase:

    def schedule(i):
      return step_size / (1 + decay_rate * np.floor(i / decay_steps))
  else:

    def schedule(i):
      return step_size / (1 + decay_rate * i / decay_steps)

  return schedule

def polynomial_decay(step_size, decay_steps, final_step_size, power=1.0):
  def schedule(step_num):
    step_num = np.minimum(step_num, decay_steps)
    step_mult = (1 - step_num / decay_steps)**power
    return step_mult * (step_size - final_step_size) + final_step_size

  return schedule

def piecewise_constant(boundaries, values):
  boundaries = np.array(boundaries)
  values = np.array(values)
  if not boundaries.ndim == values.ndim == 1:
    raise ValueError("boundaries and values must be sequences")
  if not boundaries.shape[0] == values.shape[0] - 1:
    raise ValueError("boundaries length must be one longer than values length")

  def schedule(i):
    return values[np.sum(i > boundaries)]

  return schedule

def make_schedule(scalar_or_schedule):
  if callable(scalar_or_schedule):
    return scalar_or_schedule
  elif np.ndim(scalar_or_schedule) == 0:
    return constant(scalar_or_schedule)
  else:
    raise TypeError(type(scalar_or_schedule))

### utilities
def l2_norm(tree):
  """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
  leaves, _ = tree_flatten(tree)
  return np.sqrt(sum(np.vdot(x, x) for x in leaves))

def clip_grads(grad_tree, max_norm):
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(grad_tree)
  normalize = lambda g: np.where(norm < max_norm, g, g * (max_norm / norm))
  return tree_map(normalize, grad_tree)

### serialization utilities
class JoinPoint(object):
  """Marks the boundary between two joined (nested) pytrees."""
  def __init__(self, subtree):
    self.subtree = subtree

  # Since pytrees are containers of numpy arrays, look iterable.
  def __iter__(self):
    yield self.subtree

def unpack_optimizer_state(opt_state):
  """Converts an OptimizerState to a marked pytree.

  Converts an OptimizerState to a marked pytree with the leaves of the outer
  pytree represented as JoinPoints to avoid losing information. This function is
  intended to be useful when serializing optimizer states.

  Args:
    opt_state: An OptimizerState
  Returns:
    A pytree with JoinPoint leaves that contain a second level of pytrees.
  """
  packed_state, tree_def, subtree_defs = opt_state
  subtrees = map(tree_unflatten, subtree_defs, packed_state)
  sentinels = [JoinPoint(subtree) for subtree in subtrees]
  return tree_util.tree_unflatten(tree_def, sentinels)

def pack_optimizer_state(marked_pytree):
  """Converts a marked pytree to an OptimizerState.

  The inverse of unpack_optimizer_state. Converts a marked pytree with the
  leaves of the outer pytree represented as JoinPoints back into an
  OptimizerState. This function is intended to be useful when deserializing
  optimizer states.

  Args:
    marked_pytree: A pytree containing JoinPoint leaves that hold more pytrees.
  Returns:
    An equivalent OptimizerState to the input argument.
  """
  sentinels, tree_def = tree_flatten(marked_pytree)
  assert all(isinstance(s, JoinPoint) for s in sentinels)
  subtrees = [s.subtree for s in sentinels]
  states_flat, subtree_defs = unzip2(map(tree_flatten, subtrees))
  packed_state = pack(map(pack, states_flat))
  return OptimizerState(packed_state, tree_def, subtree_defs)
