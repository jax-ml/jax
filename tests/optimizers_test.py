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

"""Tests for the optimizers module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import absltest
import numpy as onp

import jax.numpy as np
import jax.test_util as jtu
from jax import jit, grad
from jax import core, tree_util
from jax.experimental import optimizers
from jax.interpreters import xla

from jax.config import config
config.parse_flags_with_absl()


class OptimizerTests(jtu.JaxTestCase):

  def _CheckOptimizer(self, optimizer, loss, x0, num_steps, *args, **kwargs):
    self._CheckFuns(optimizer, loss, x0, *args)
    self._CheckRun(optimizer, loss, x0, num_steps, *args, **kwargs)

  def _CheckFuns(self, optimizer, loss, x0, *args):
    init_fun, update_fun, get_params = optimizer(*args)
    opt_state = init_fun(x0)
    self.assertAllClose(x0, get_params(opt_state), check_dtypes=True)
    opt_state2 = update_fun(0, grad(loss)(x0), opt_state)  # doesn't crash
    self.assertEqual(tree_util.tree_structure(opt_state),
                     tree_util.tree_structure(opt_state2))

  @jtu.skip_on_devices('gpu')
  def _CheckRun(self, optimizer, loss, x0, num_steps, *args, **kwargs):
    init_fun, update_fun, get_params = optimizer(*args)

    opt_state = init_fun(x0)
    for i in range(num_steps):
      x = get_params(opt_state)
      g = grad(loss)(x)
      opt_state = update_fun(i, g, opt_state)
    xstar = get_params(opt_state)
    self.assertLess(loss(xstar), 1e-2)

    update_fun_jitted = jit(update_fun)
    opt_state = init_fun(x0)
    for i in range(num_steps):
      x = get_params(opt_state)
      g = grad(loss)(x)
      opt_state = update_fun_jitted(i, g, opt_state)
    xstar = get_params(opt_state)
    self.assertLess(loss(xstar), 1e-2)

  def testSgdScalar(self):
    def loss(x): return x**2
    x0 = 1.
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.sgd, loss, x0, num_iters, step_size)

  def testSgdVector(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.sgd, loss, x0, num_iters, step_size)

  def testSgdNestedTuple(self):
    def loss(xyz):
      x, (y, z) = xyz
      return sum(np.dot(a, a) for a in [x, y, z])
    x0 = (np.ones(2), (np.ones(2), np.ones(2)))
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.sgd, loss, x0, num_iters, step_size)

  def testMomentumVector(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    mass = 0.
    self._CheckOptimizer(optimizers.momentum, loss, x0, num_iters, step_size, mass)

  def testMomentumDict(self):
    def loss(dct): return np.dot(dct['x'], dct['x'])
    x0 = {'x': np.ones(2)}
    num_iters = 100
    step_size = 0.1
    mass = 0.
    self._CheckOptimizer(optimizers.momentum, loss, x0, num_iters, step_size, mass)

  def testRmspropVector(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.rmsprop, loss, x0, num_iters, step_size)

  @jtu.skip_on_devices('cpu')  # TODO(mattjj): investigate numerical failure
  def testAdamVector(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.adam, loss, x0, num_iters, step_size)

  def testSgdClosure(self):
    def loss(y, x): return y**2 * x**2
    x0 = 1.
    y = 1.
    num_iters = 20
    step_size = 0.1
    partial_loss = functools.partial(loss, y)
    self._CheckRun(optimizers.sgd, partial_loss, x0, num_iters, step_size)

  def testAdagrad(self):
    def loss(xs):
      x1, x2 = xs
      return np.sum(x1 ** 2) + np.sum(x2 ** 2)

    num_iters = 100
    step_size = 0.1
    x0 = (np.ones(2), np.ones((2, 2)))
    self._CheckOptimizer(optimizers.adagrad, loss, x0, num_iters, step_size)

  def testSM3(self):
    def loss(xs):
      x1, x2 = xs
      return np.sum(x1 ** 2) + np.sum(x2 ** 2)

    num_iters = 100
    step_size = 0.1
    x0 = (np.ones(2), np.ones((2, 2)))
    self._CheckOptimizer(optimizers.sm3, loss, x0, num_iters, step_size)

  def testSgdVectorExponentialDecaySchedule(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_schedule = optimizers.exponential_decay(0.1, 3, 2.)
    self._CheckFuns(optimizers.sgd, loss, x0, step_schedule)

  def testSgdVectorInverseTimeDecaySchedule(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_schedule = optimizers.inverse_time_decay(0.1, 3, 2.)
    self._CheckFuns(optimizers.sgd, loss, x0, step_schedule)

  def testAdamVectorInverseTimeDecaySchedule(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_schedule = optimizers.inverse_time_decay(0.1, 3, 2.)
    self._CheckFuns(optimizers.adam, loss, x0, step_schedule)

  def testMomentumVectorInverseTimeDecayStaircaseSchedule(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_sched = optimizers.inverse_time_decay(0.1, 3, 2., staircase=True)
    mass = 0.9
    self._CheckFuns(optimizers.momentum, loss, x0, step_sched, mass)

  def testRmspropVectorPiecewiseConstantSchedule(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_schedule = optimizers.piecewise_constant([25, 75], [1.0, 0.5, 0.1])
    self._CheckFuns(optimizers.rmsprop, loss, x0, step_schedule)

  def testTracedStepSize(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_size = 0.1

    init_fun, _, _ = optimizers.sgd(step_size)
    opt_state = init_fun(x0)

    @jit
    def update(opt_state, step_size):
      _, update_fun, get_params = optimizers.sgd(step_size)
      x = get_params(opt_state)
      g = grad(loss)(x)
      return update_fun(0, g, opt_state)

    update(opt_state, 0.9)  # doesn't crash

  # TODO(mattjj): re-enable
  # def testDeviceTupleState(self):
  #   init_fun, update_fun, _ = optimizers.sgd(0.1)
  #   opt_state = init_fun(np.zeros(3))
  #   self.assertIsInstance(opt_state, optimizers.OptimizerState)
  #   self.assertIsInstance(opt_state.packed_state, core.JaxTuple)
  #   opt_state = jit(update_fun)(0, np.zeros(3), opt_state)
  #   self.assertIsInstance(opt_state, optimizers.OptimizerState)
  #   self.assertIsInstance(opt_state.packed_state, xla.DeviceTuple)

  def testUpdateFunStructureMismatchErrorMessage(self):
    @optimizers.optimizer
    def opt_maker():
      def init_fun(x0):
        return {'x': x0}
      def update_fun(i, g, opt_state):
        x = opt_state['x']
        return {'x': x - 0.1 * g, 'v': g}  # bug!
      def get_params(opt_state):
        return opt_state['x']
      return init_fun, update_fun, get_params

    init_fun, update_fun, get_params = opt_maker()
    opt_state = init_fun(np.zeros(3))
    self.assertRaises(TypeError, lambda: update_fun(opt_state))

  def testUtilityNorm(self):
    x0 = (np.ones(2), (np.ones(3), np.ones(4)))
    norm = optimizers.l2_norm(x0)
    expected = onp.sqrt(onp.sum(onp.ones(2+3+4)**2))
    self.assertAllClose(norm, expected, check_dtypes=False)

  def testUtilityClipGrads(self):
    g = (np.ones(2), (np.ones(3), np.ones(4)))
    norm = optimizers.l2_norm(g)

    ans = optimizers.clip_grads(g, 1.1 * norm)
    expected = g
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = optimizers.l2_norm(optimizers.clip_grads(g, 0.9 * norm))
    expected = 0.9 * norm
    self.assertAllClose(ans, expected, check_dtypes=False)




if __name__ == '__main__':
  absltest.main()
