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

"""Tests for the minmax optimizer module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import absltest
from jax.config import config
import jax.numpy as np
import jax.test_util as jtu
from jax import jit, grad
from jax.experimental import minmax
from jax.lib import xla_bridge as xla


class OptimizerTests(jtu.JaxTestCase):

  def _CheckOptimizer(self, optimizer, loss, x0, num_steps, *args, **kwargs):
    self._CheckFuns(optimizer, loss, x0, *args)
    self._CheckRun(optimizer, loss, x0, num_steps, *args, **kwargs)

  def _CheckFuns(self, optimizer, loss, x0, *args):
    init_fun, update_fun = optimizer(*args)
    opt_state = init_fun(x0)
    update_fun(0, grad(loss)(x0, None), opt_state)  # doesn't crash

  @jtu.skip_on_devices('gpu')
  def _CheckRun(self, optimizer, loss, x0, num_steps, *args, **kwargs):
    return # TODO(mattjj): bring back fax!
    num_repl = xla.get_replica_count()
    infeeder = fax.make_infeed_from_sequence(
        [np.ones(1, dtype='float32')] * num_steps * num_repl,
        with_pyvals=True)

    def op(infeed, x0):
      opt_init, opt_update = optimizer(*args, **kwargs)
      return minmax.run_optimizer(loss, infeed, opt_update, opt_init(x0))
    cop = jit(op)

    a1, _ = op(infeeder(), x0)
    a2, _ = cop(infeeder(), x0)

    assert loss(a1, None) < 1e-3
    assert loss(a2, None) < 1e-3
    self.assertAllClose(a1, a2, check_dtypes=False)

  def testSgdScalar(self):
    def loss(x, _): return x**2
    x0 = 1.
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(minmax.sgd, loss, x0, num_iters, step_size)

  def testSgdVector(self):
    def loss(x, _): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(minmax.sgd, loss, x0, num_iters, step_size)

  def testSgdNestedTuple(self):
    def loss(xyz, _):
      x, (y, z) = xyz
      return sum(np.dot(a, a) for a in [x, y, z])
    x0 = (np.ones(2), (np.ones(2), np.ones(2)))
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(minmax.sgd, loss, x0, num_iters, step_size)

  def testMomentumVector(self):
    def loss(x, _): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    mass = 0.
    self._CheckOptimizer(minmax.momentum, loss, x0, num_iters, step_size, mass)

  def testRmspropVector(self):
    def loss(x, _): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(minmax.rmsprop, loss, x0, num_iters, step_size)

  @jtu.skip_on_devices('cpu')  # TODO(mattjj): investigate numerical failure
  def testAdamVector(self):
    def loss(x, _): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(minmax.adam, loss, x0, num_iters, step_size)

  def testSgdClosure(self):
    def loss(y, x, _): return y**2 * x**2
    x0 = 1.
    y = 1.
    num_iters = 20
    step_size = 0.1
    partial_loss = functools.partial(loss, y)
    self._CheckRun(minmax.sgd, partial_loss, x0, num_iters, step_size)

  def testSgdVectorExponentialDecaySchedule(self):
    def loss(x, _): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_schedule = minmax.exponential_decay(0.1, 3, 2.)
    self._CheckOptimizer(minmax.sgd, loss, x0, num_iters, step_schedule)

  def testSgdVectorInverseTimeDecaySchedule(self):
    def loss(x, _): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_schedule = minmax.inverse_time_decay(0.1, 3, 2.)
    self._CheckOptimizer(minmax.sgd, loss, x0, num_iters, step_schedule)

  def testAdamVectorInverseTimeDecaySchedule(self):
    def loss(x, _): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_schedule = minmax.inverse_time_decay(0.1, 3, 2.)
    self._CheckOptimizer(minmax.adam, loss, x0, num_iters, step_schedule)

  def testMomentumVectorInverseTimeDecayStaircaseSchedule(self):
    def loss(x, _): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_sched = minmax.inverse_time_decay(0.1, 3, 2., staircase=True)
    mass = 0.9
    self._CheckOptimizer(minmax.momentum, loss, x0, num_iters, step_sched, mass)

  def testRmspropVectorPiecewiseConstantSchedule(self):
    def loss(x, _): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_schedule = minmax.piecewise_constant([25, 75], [1.0, 0.5, 0.1])
    self._CheckOptimizer(minmax.rmsprop, loss, x0, num_iters, step_schedule)

  def testTracedStepSize(self):
    def loss(x, _): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1

    init_fun, _ = minmax.sgd(step_size)
    opt_state = init_fun(x0)

    @jit
    def update(opt_state, step_size):
      _, update_fun = minmax.sgd(step_size)
      x = minmax.get_params(opt_state)
      g = grad(loss)(x, None)
      return update_fun(0, g, opt_state)

    update(opt_state, 0.9)  # doesn't crash


if __name__ == '__main__':
  config.config_with_absl()
  absltest.main()
