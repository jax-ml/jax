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

"""Tests for the optix module."""


from absl.testing import absltest
from jax import numpy as jnp
from jax.experimental import optimizers
from jax.experimental import optix
import jax.test_util as jtu
from jax.tree_util import tree_leaves
import numpy as np

from jax.config import config
config.parse_flags_with_absl()


STEPS = 50
LR = 1e-2


class OptixTest(absltest.TestCase):

  def setUp(self):
    super(OptixTest, self).setUp()
    self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))
    self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

  def test_sgd(self):

    # experimental/optimizers.py
    jax_params = self.init_params
    opt_init, opt_update, get_params = optimizers.sgd(LR)
    state = opt_init(jax_params)
    for i in range(STEPS):
      state = opt_update(i, self.per_step_updates, state)
      jax_params = get_params(state)

    # experimental/optix.py
    optix_params = self.init_params
    sgd = optix.sgd(LR, 0.0)
    state = sgd.init(optix_params)
    for _ in range(STEPS):
      updates, state = sgd.update(self.per_step_updates, state)
      optix_params = optix.apply_updates(optix_params, updates)

    # Check equivalence.
    for x, y in zip(tree_leaves(jax_params), tree_leaves(optix_params)):
      np.testing.assert_allclose(x, y, rtol=1e-5)

  jtu.skip_on_devices("tpu")
  def test_apply_every(self):
    # The frequency of the application of sgd
    k = 4
    zero_update = (jnp.array([0., 0.]), jnp.array([0., 0.]))

    # experimental/optix.py sgd
    optix_sgd_params = self.init_params
    sgd = optix.sgd(LR, 0.0)
    state_sgd = sgd.init(optix_sgd_params)

    # experimental/optix.py sgd apply every
    optix_sgd_apply_every_params = self.init_params
    sgd_apply_every = optix.chain(optix.apply_every(k=k),
                                  optix.trace(decay=0, nesterov=False),
                                  optix.scale(-LR))
    state_sgd_apply_every = sgd_apply_every.init(optix_sgd_apply_every_params)
    for i in range(STEPS):
      # Apply a step of sgd
      updates_sgd, state_sgd = sgd.update(self.per_step_updates, state_sgd)
      optix_sgd_params = optix.apply_updates(optix_sgd_params, updates_sgd)

      # Apply a step of sgd_apply_every
      updates_sgd_apply_every, state_sgd_apply_every = sgd_apply_every.update(
          self.per_step_updates, state_sgd_apply_every)
      optix_sgd_apply_every_params = optix.apply_updates(
          optix_sgd_apply_every_params, updates_sgd_apply_every)
      if i % k == k-1:
        # Check equivalence.
        for x, y in zip(
            tree_leaves(optix_sgd_apply_every_params),
            tree_leaves(optix_sgd_params)):
          np.testing.assert_allclose(x, y, atol=1e-6, rtol=100)
      else:
        # Check updaue is zero.
        for x, y in zip(
            tree_leaves(updates_sgd_apply_every),
            tree_leaves(zero_update)):
          np.testing.assert_allclose(x, y, atol=1e-10, rtol=1e-5)

  def test_adam(self):
    b1, b2, eps = 0.9, 0.999, 1e-8

    # experimental/optimizers.py
    jax_params = self.init_params
    opt_init, opt_update, get_params = optimizers.adam(LR, b1, b2, eps)
    state = opt_init(jax_params)
    for i in range(STEPS):
      state = opt_update(i, self.per_step_updates, state)
      jax_params = get_params(state)

    # experimental/optix.py
    optix_params = self.init_params
    adam = optix.adam(LR, b1, b2, eps)
    state = adam.init(optix_params)
    for _ in range(STEPS):
      updates, state = adam.update(self.per_step_updates, state)
      optix_params = optix.apply_updates(optix_params, updates)

    # Check equivalence.
    for x, y in zip(tree_leaves(jax_params), tree_leaves(optix_params)):
      np.testing.assert_allclose(x, y, rtol=1e-4)

  def test_rmsprop(self):
    decay, eps = .9, 0.1

    # experimental/optimizers.py
    jax_params = self.init_params
    opt_init, opt_update, get_params = optimizers.rmsprop(LR, decay, eps)
    state = opt_init(jax_params)
    for i in range(STEPS):
      state = opt_update(i, self.per_step_updates, state)
      jax_params = get_params(state)

    # experimental/optix.py
    optix_params = self.init_params
    rmsprop = optix.rmsprop(LR, decay, eps)
    state = rmsprop.init(optix_params)
    for _ in range(STEPS):
      updates, state = rmsprop.update(self.per_step_updates, state)
      optix_params = optix.apply_updates(optix_params, updates)

    # Check equivalence.
    for x, y in zip(tree_leaves(jax_params), tree_leaves(optix_params)):
      np.testing.assert_allclose(x, y, rtol=1e-5)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
