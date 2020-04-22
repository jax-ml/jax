import numpy as onp
import random as _builtin_random
import jax.random
import jax.numpy as jnp
from jax.numpy.lax_numpy import _wraps

@_wraps(onp.random.RandomState)
class RandomState:
  def __init__(self, seed=None):
    self._rng = _builtin_random.Random(seed)

  @property
  def _key(self):
    return jax.random.PRNGKey(self._rng.randint(0, (1 << 32) - 1))

  @staticmethod
  def _shape(size):
    if size is None:
      return ()
    if not hasattr(size, '__len__'):
      return (size,)
    return size

  @_wraps(onp.random.RandomState.seed)
  def seed(self, seed):
    self._rng = _builtin_random.Random(seed)

  @_wraps(onp.random.RandomState.rand)
  def rand(self, *shape):
    return jax.random.uniform(self._key, shape)

  @_wraps(onp.random.RandomState.random)
  def random(self, size=None):
    return jax.random.uniform(self._key, self._shape(size))

  @_wraps(onp.random.RandomState.randint)
  def randint(self, low, high=None, size=None, dtype=int):
    if high is None:
      low, high = 0, low
    return jax.random.randint(self._key, self._shape(size), low, high, dtype)

  @_wraps(onp.random.RandomState.randn)
  def randn(self, *shape):
    return jax.random.normal(self._key, shape)

  @_wraps(onp.random.RandomState.normal)
  def normal(self, loc=0.0, scale=1.0, size=None):
    return loc + scale * jax.random.normal(self._key, self._shape(size))

_default_rng = RandomState()

seed = _wraps(onp.random.seed)(_default_rng.seed)
rand = _wraps(onp.random.rand)(_default_rng.rand)
random = _wraps(onp.random.random)(_default_rng.random)
randint = _wraps(onp.random.randint)(_default_rng.randint)
randn = _wraps(onp.random.randn)(_default_rng.randn)
normal = _wraps(onp.random.normal)(_default_rng.normal)