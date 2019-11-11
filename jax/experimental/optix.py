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

"""A composable gradient processing and optimization library for JAX.

The `optix` module implements a number of composable gradient transformations,
typically used in the context of optimizing neural nets.

Each transformation defines:

* an `init_fn`, to initialize a (possibly empty) set of statistics, or `state`.
* an `update_fn` to transform an input gradient and update the state.

An (optional) `chain` utility can be used to build custom optimizers by
chaining arbitrary sequences of transformations. For any sequence of
transformations `chain` returns a single `init_fn` and `update_fn`.

An (optional) `apply_updates` function can be used to eventually apply the
transformed gradients to the set of parameters of interest.

Separating gradient transformations from the parameter update allows to flexibly
chain a sequence of transformations of the same gradients, as well as combine
multiple updates to the same parameters (e.g. in multi-task settings where the
different tasks may benefit from different sets of gradient transformations).

Many popular optimizers can be implemented using `optix` as one-liners, and,
for convenience, we provide aliases for some of the most popular ones.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from jax import numpy as jnp
from jax import random as jrandom

from jax.tree_util import tree_leaves
from jax.tree_util import tree_multimap
from jax.tree_util import tree_structure
from jax.tree_util import tree_unflatten


### Composable gradient transformations. ###


ClipState = collections.namedtuple("ClipState", "")


def clip(max_delta):
  """Clip updates element-wise.

  Args:
    max_delta: the maximum size of an update, for each variable

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ClipState()

  def update_fn(updates, state):
    updates = tree_multimap(
        lambda g: jnp.clip_by_value(g, -max_delta, max_delta), updates)
    return updates, state

  return init_fn, update_fn


ClipByGlobalNormState = collections.namedtuple("ClipByGlobalNormState", "")


def global_norm(items):
  return jnp.sqrt(jnp.sum([jnp.sum(x**2) for x in tree_leaves(items)]))
_global_norm = global_norm  # TODO(mtthss): remove when google code updated


def clip_by_global_norm(max_norm):
  """Clip updates using their global norm.

  References:
    [Pascanu et al, 2012](https://arxiv.org/abs/1211.5063)

  Args:
    max_norm: the maximum global norm for an update.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ClipByGlobalNormState()

  def update_fn(updates, state):
    g_norm = global_norm(updates)
    trigger = g_norm < max_norm
    updates = tree_multimap(
        lambda t: jnp.where(trigger, t, t * (max_norm / g_norm)), updates)
    return updates, state

  return init_fn, update_fn


TraceState = collections.namedtuple("TraceState", "trace")


def trace(decay, nesterov):
  """Compute a trace of past updates.

  Args:
    decay: the decay rate for the tracing of past updates.
    nesterov: whether to use nesterov momentum.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    return TraceState(trace=tree_multimap(jnp.zeros_like, params))

  def update_fn(updates, state):
    f = lambda g, t: g + decay * t
    update_trace = tree_multimap(f, updates, state.trace)
    updates = (
        tree_multimap(f, updates, update_trace) if nesterov else update_trace)
    return updates, TraceState(trace=update_trace)

  return init_fn, update_fn


ScaleByRmsState = collections.namedtuple("ScaleByRmsState", "nu")


def _update_moment(updates, moments, decay, order):
  return tree_multimap(
      lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


def scale_by_rms(decay=0.9, eps=1e-8):
  """Rescale updates by the root of the exp. moving avg of the square.

  References:
    [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  Args:
    decay: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    nu = tree_multimap(jnp.zeros_like, params)  # second moment
    return ScaleByRmsState(nu=nu)

  def update_fn(updates, state):
    nu = _update_moment(updates, state.nu, decay, 2)
    updates = tree_multimap(lambda g, n: g / (jnp.sqrt(n + eps)), updates, nu)
    return updates, ScaleByRmsState(nu=nu)

  return init_fn, update_fn


ScaleByRStdDevState = collections.namedtuple("ScaleByRStdDevState", "mu nu")


def scale_by_stddev(decay=0.9, eps=1e-8):
  """Rescale updates by the root of the centered exp. moving average of squares.

  References:
    [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  Args:
    decay: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    mu = tree_multimap(jnp.zeros_like, params)  # First moment
    nu = tree_multimap(jnp.zeros_like, params)  # Second moment
    return ScaleByRStdDevState(mu=mu, nu=nu)

  def update_fn(updates, state):
    mu = _update_moment(updates, state.mu, decay, 1)
    nu = _update_moment(updates, state.nu, decay, 2)
    updates = tree_multimap(
        lambda g, m, n: g / jnp.sqrt(n - m**2 + eps), updates, mu, nu)
    return updates, ScaleByRStdDevState(mu=mu, nu=nu)

  return init_fn, update_fn


ScaleByAdamState = collections.namedtuple("ScaleByAdamState", "count mu nu")


def scale_by_adam(b1=0.9, b2=0.999, eps=1e-8):
  """Rescale updates according to the Adam algorithm.

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    mu = tree_multimap(jnp.zeros_like, params)  # First moment
    nu = tree_multimap(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([]), mu=mu, nu=nu)

  def update_fn(updates, state):
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    mu_hat = tree_multimap(lambda t: t / (1 - b1 ** (state.count + 1)), mu)
    nu_hat = tree_multimap(lambda t: t / (1 - b2 ** (state.count + 1)), nu)
    updates = tree_multimap(
        lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=state.count + 1, mu=mu, nu=nu)

  return init_fn, update_fn


ScaleState = collections.namedtuple("ScaleState", "")


def scale(step_size):
  """Scale updates by some fixed scalar `step_size`.

  Args:
    step_size: a scalar corresponding to a fixed scaling factor for updates.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ScaleState()

  def update_fn(updates, state):
    updates = tree_multimap(lambda g: step_size * g, updates)
    return updates, state

  return init_fn, update_fn


ScaleByScheduleState = collections.namedtuple("ScaleByScheduleState", "count")


def scale_by_schedule(step_size_fn):
  """Scale updates using a custom schedule for the `step_size`.

  Args:
    step_size_fn: a function that takes an update count as input and proposes
      the step_size to multiply the updates by.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ScaleByScheduleState(count=jnp.zeros([]))

  def update_fn(updates, state):
    updates = tree_multimap(lambda g: step_size_fn(state.count) * g, updates)
    return updates, ScaleByScheduleState(count=state.count + 1)

  return init_fn, update_fn


AddNoiseState = collections.namedtuple("AddNoiseState", "count rng_key")


def add_noise(eta, gamma, seed):
  """Add gradient noise.

  References:
    [Neelakantan et al, 2014](https://arxiv.org/abs/1511.06807)

  Args:
    eta: base variance of the gaussian noise added to the gradient.
    gamma: decay exponent for annealing of the variance.
    seed: seed for random number generation.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return AddNoiseState(count=jnp.zeros([]), rng_key=jrandom.PRNGKey(seed))

  def update_fn(updates, state):  # pylint: disable=missing-docstring
    num_vars = len(tree_leaves(updates))
    treedef = tree_structure(updates)
    variance = eta / (1 + state.count) ** gamma
    all_keys = jrandom.split(state.rng_key, num=num_vars + 1)
    noise = tree_multimap(
        lambda g, k: jrandom.normal(k, shape=g.shape),
        updates, tree_unflatten(treedef, all_keys[1:]))
    updates = tree_multimap(
        lambda g, n: g + variance * n, updates, noise)
    return updates, AddNoiseState(count=state.count + 1, rng_key=all_keys[0])

  return init_fn, update_fn


### Utilities for building and using custom optimizers. ###


def chain(*args):
  """Applies a list of chainable update transformations.

  Given a sequence of chainable transforms, `chain` returns an `init_fn`
  that constructs a `state` by concatenating the states of the individual
  transforms, and returns an `update_fn` which chains the update transformations
  feeding the appropriate state to each.

  Args:
    *args: a sequence of chainable (init_fn, update_fn) tuples.

  Returns:
    A single (init_fn, update_fn) tuple.
  """

  init_fns, update_fns = zip(*args)

  def init(params):
    return [fn(params) for fn in init_fns]

  def update(updates, state):
    new_state = []
    for s, fn in zip(state, update_fns):
      updates, new_s = fn(updates, s)
      new_state.append(new_s)
    return updates, new_state

  return init, update


def apply_updates(params, updates):
  """Applies an update to the corresponding parameters.

  This is an (optional) utility functions that applies an update, and returns
  the updated parameters to the caller. The update itself is typically the
  result of applying any number of `chainable` transformations.

  Args:
    params: a tree of parameters.
    updates: a tree of updates, the tree structure and the shape of the leaf
    nodes must match that of `params`.

  Returns:
    Updated parameters, with same structure and shape as `params`.
  """
  return tree_multimap(lambda p, u: p + u, params, updates)


### Aliases for popular optimizers. ###


def sgd(learning_rate, momentum=0., nesterov=False):
  return chain(
      trace(decay=momentum, nesterov=nesterov),
      scale(-learning_rate))


def noisy_sgd(learning_rate, eta=0.01, gamma=0.55, seed=42):
  return chain(
      trace(decay=0., nesterov=False),
      scale(-learning_rate),
      add_noise(eta, gamma, seed))


def adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8):
  return chain(
      scale_by_adam(b1=b1, b2=b2, eps=eps),
      scale(-learning_rate))


def rmsprop(learning_rate, decay=0.9, eps=1e-8, centered=False):
  if not centered:
    return chain(
        scale_by_rms(decay=decay, eps=eps),
        scale(-learning_rate))
  else:
    return chain(
        scale_by_stddev(decay=decay, eps=eps),
        scale(-learning_rate))
