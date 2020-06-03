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

The ``optix`` module implements a number of composable gradient transformations,
typically used in the context of optimizing neural nets.

Each transformation defines:

* an ``init_fn``, to initialize a (possibly empty) set of statistics, or ``state``.
* an ``update_fn`` to transform an input gradient and update the state.

An (optional) ``chain`` utility can be used to build custom optimizers by
chaining arbitrary sequences of transformations. For any sequence of
transformations ``chain`` returns a single ``init_fn`` and ``update_fn``.

An (optional) ``apply_updates`` function can be used to eventually apply the
transformed gradients to the set of parameters of interest.

Separating gradient transformations from the parameter update allows to flexibly
chain a sequence of transformations of the same gradients, as well as combine
multiple updates to the same parameters (e.g. in multi-task settings where the
different tasks may benefit from different sets of gradient transformations).

Many popular optimizers can be implemented using ``optix`` as one-liners, and,
for convenience, we provide aliases for some of the most popular ones.
"""


from typing import Any, Callable, NamedTuple, Sequence, Tuple, Union

from jax import numpy as jnp
from jax import random as jrandom

from jax.tree_util import tree_leaves
from jax.tree_util import tree_multimap
from jax.tree_util import tree_structure
from jax.tree_util import tree_unflatten


###
# Composable gradient transformations.

# TODO(jaslanides): Make these more specific.
OptState = NamedTuple  # Optimizer state is a (possibly empty) namedtuple.
Params = Any  # Parameters are nests of `jnp.ndarrays`.
Updates = Params  # Gradient updates are of the same type as parameters.


InitFn = Callable[[Params], Union[OptState, Sequence[OptState]]]
UpdateFn = Callable[[Updates, OptState], Tuple[Updates, OptState]]


class InitUpdate(NamedTuple):
  """Optix optimizers consists of a pair of functions: (initialiser, update)."""
  init: InitFn
  update: UpdateFn


class ClipState(OptState):
  """The `clip` transformation is stateless."""


def clip(max_delta) -> InitUpdate:
  """Clip updates element-wise, to be between -max_delta and +max_delta.

  Args:
    max_delta: the maximum absolute value for each element in the update.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ClipState()

  def update_fn(updates, state):
    updates = tree_multimap(
        lambda g: jnp.clip(g, -max_delta, max_delta), updates)
    return updates, state

  return InitUpdate(init_fn, update_fn)


def global_norm(updates: Updates) -> Updates:
  return jnp.sqrt(
      sum([jnp.sum(jnp.square(x)) for x in tree_leaves(updates)]))


class ClipByGlobalNormState(OptState):
  """The `clip_by_global_norm` transformation is stateless."""


def clip_by_global_norm(max_norm) -> InitUpdate:
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
        lambda t: jnp.where(trigger, t, (t / g_norm) * max_norm), updates)
    return updates, state

  return InitUpdate(init_fn, update_fn)


class TraceState(OptState):
  """Holds an aggregation of past updates."""
  trace: Params


def trace(decay: float, nesterov: bool) -> InitUpdate:
  """Compute a trace of past updates.

  Args:
    decay: the decay rate for the tracing of past updates.
    nesterov: whether to use Nesterov momentum.

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

  return InitUpdate(init_fn, update_fn)


class ScaleByRmsState(OptState):
  """State for exponential root mean-squared (RMS)-normalized updates."""
  nu: Updates


def _update_moment(updates, moments, decay, order):
  return tree_multimap(
      lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


def scale_by_rms(decay: float = 0.9, eps: float = 1e-8):
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

  return InitUpdate(init_fn, update_fn)


class ScaleByRStdDevState(OptState):
  """State for centered exponential moving average of squares of updates."""
  mu: Updates
  nu: Updates


def scale_by_stddev(decay: float = 0.9, eps: float = 1e-8) -> InitUpdate:
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
        lambda g, m, n: g / jnp.sqrt(n - jnp.square(m) + eps), updates, mu, nu)
    return updates, ScaleByRStdDevState(mu=mu, nu=nu)

  return InitUpdate(init_fn, update_fn)


class ScaleByAdamState(OptState):
  """State for the Adam algorithm."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32.
  mu: Updates
  nu: Updates


def scale_by_adam(b1: float = 0.9,
                  b2: float = 0.999,
                  eps: float = 1e-8,
                  eps_root: float = 0.0) -> InitUpdate:
  """Rescale updates according to the Adam algorithm.

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    mu = tree_multimap(jnp.zeros_like, params)  # First moment
    nu = tree_multimap(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state):
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    mu_hat = tree_multimap(lambda t: t / (1 - b1 ** (state.count + 1)), mu)
    nu_hat = tree_multimap(lambda t: t / (1 - b2 ** (state.count + 1)), nu)
    updates = tree_multimap(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=state.count + 1, mu=mu, nu=nu)

  return InitUpdate(init_fn, update_fn)


class ScaleState(NamedTuple):
  """The scale transformation is stateless."""


def scale(step_size: float) -> InitUpdate:
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

  return InitUpdate(init_fn, update_fn)


class ScaleByScheduleState(OptState):
  """Maintains count for scale scheduling."""
  count: jnp.ndarray  # shape=(), dtype=jnp.int32


def scale_by_schedule(step_size_fn: Callable[[jnp.ndarray], jnp.ndarray]):
  """Scale updates using a custom schedule for the `step_size`.

  Args:
    step_size_fn: a function that takes an update count as input and proposes
      the step_size to multiply the updates by.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state):
    updates = tree_multimap(lambda g: step_size_fn(state.count) * g, updates)
    return updates, ScaleByScheduleState(count=state.count + 1)

  return InitUpdate(init_fn, update_fn)


class AddNoiseState(OptState):
  """State for adding gradient noise. Contains a count for annealing."""
  count: jnp.ndarray
  rng_key: jnp.ndarray


def add_noise(eta: float, gamma: float, seed: int) -> InitUpdate:
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
    return AddNoiseState(count=jnp.zeros([], jnp.int32),
                         rng_key=jrandom.PRNGKey(seed))

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

  return InitUpdate(init_fn, update_fn)


class ApplyEvery(OptState):
  """Contains a counter and a gradient accumulator."""
  count: jnp.ndarray
  grad_acc: Updates


def apply_every(k: int = 1) -> InitUpdate:
  """accumulate gradients and apply them every k steps.

  Args:
    k: apply the update every k steps otherwise accumulate the gradients.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    grad_acc = tree_multimap(jnp.zeros_like, params)
    return ApplyEvery(count=jnp.zeros([], jnp.int32), grad_acc=grad_acc)

  def update_fn(updates, state):

    c = state.count % k
    acc = c != 0
    grad_acc = tree_multimap(
        lambda g, ga: acc * ga + g, updates, state.grad_acc)
    emit = c == (k - 1)
    updates = tree_multimap(lambda ga: emit * ga, grad_acc)
    return updates, ApplyEvery(count=state.count + 1, grad_acc=grad_acc)

  return InitUpdate(init_fn, update_fn)


###
# Utilities for building and using custom optimizers.


def chain(*args: InitUpdate) -> InitUpdate:
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

  def init_fn(params):
    return [fn(params) for fn in init_fns]

  def update_fn(updates, state):
    new_state = []
    for s, fn in zip(state, update_fns):
      updates, new_s = fn(updates, s)
      new_state.append(new_s)
    return updates, new_state

  return InitUpdate(init_fn, update_fn)


def apply_updates(params: Params, updates: Updates) -> Params:
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


###
# Aliases for popular optimizers.


def sgd(learning_rate: float,
        momentum: float = 0.,
        nesterov: bool = False) -> InitUpdate:
  return chain(
      trace(decay=momentum, nesterov=nesterov),
      scale(-learning_rate),
  )


def noisy_sgd(learning_rate: float,
              eta: float = 0.01,
              gamma: float = 0.55,
              seed: int = 0) -> InitUpdate:
  return chain(
      trace(decay=0., nesterov=False),
      scale(-learning_rate),
      add_noise(eta, gamma, seed),
  )


def adam(learning_rate: float,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-8) -> InitUpdate:
  return chain(
      scale_by_adam(b1=b1, b2=b2, eps=eps),
      scale(-learning_rate),
  )


def rmsprop(learning_rate: float,
            decay: float = 0.9,
            eps: float = 1e-8,
            centered: bool = False) -> InitUpdate:
  if centered:
    return chain(
        scale_by_stddev(decay=decay, eps=eps),
        scale(-learning_rate),
    )
  return chain(
      scale_by_rms(decay=decay, eps=eps),
      scale(-learning_rate),
  )
