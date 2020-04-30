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

from functools import partial
from unittest import SkipTest

from absl.testing import absltest
import numpy as onp

from jax import lax
from jax import test_util as jtu
import jax.numpy as np

from examples import control

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS


def one_step_lqr(dim, T):
  Q = np.stack(T * (np.eye(dim),))
  q = np.zeros((T, dim))
  R = np.zeros((T, dim, dim))
  r = np.zeros((T, dim))
  M = np.zeros((T, dim, dim))
  A = np.stack(T * (np.eye(dim),))
  B = np.stack(T * (np.eye(dim),))
  return control.LqrSpec(Q, q, R, r, M, A, B)


def control_from_lqr(lqr):
  T, dim, _ = lqr.Q.shape
  dot = np.dot

  def cost(t, x, u):
    return (
        dot(dot(lqr.Q[t], x), x) + dot(lqr.q[t], x) +
        dot(dot(lqr.R[t], u), u) + dot(lqr.r[t], u) +
        dot(dot(lqr.M[t], u), x))

  def dynamics(t, x, u):
    return dot(lqr.A[t], x) + dot(lqr.B[t], u)

  return control.ControlSpec(cost, dynamics, T, dim, dim)


def one_step_control(dim, T):

  def cost(t, x, u):
    return np.dot(x, x)

  def dynamics(t, x, u):
    return x + u

  return control.ControlSpec(cost, dynamics, T, dim, dim)


class ControlExampleTest(jtu.JaxTestCase):

  def testTrajectoryCyclicIntegerCounter(self):
    num_states = 3

    def dynamics(t, x, u):
      return (x + u) % num_states

    T = 10

    U = np.ones((T, 1))
    X = control.trajectory(dynamics, U, np.zeros(1))
    expected = np.arange(T + 1) % num_states
    expected = np.reshape(expected, (T + 1, 1))
    self.assertAllClose(X, expected, check_dtypes=False)

    U = 2 * np.ones((T, 1))
    X = control.trajectory(dynamics, U, np.zeros(1))
    expected = np.cumsum(2 * np.ones(T)) % num_states
    expected = np.concatenate((np.zeros(1), expected))
    expected = np.reshape(expected, (T + 1, 1))
    self.assertAllClose(X, expected, check_dtypes=False)

  def testTrajectoryTimeVarying(self):
    T = 6

    def clip(x, lo, hi):
      return np.minimum(hi, np.maximum(lo, x))

    def dynamics(t, x, u):
      # computes `(x + u) if t > T else 0`
      return (x + u) * clip(t - T, 0, 1)

    U = np.ones((2 * T, 1))
    X = control.trajectory(dynamics, U, np.zeros(1))
    expected = np.concatenate((np.zeros(T + 1), np.arange(T)))
    expected = np.reshape(expected, (2 * T + 1, 1))
    self.assertAllClose(X, expected, check_dtypes=True)


  def testTrajectoryCyclicIndicator(self):
    num_states = 3

    def position(x):
      '''finds the index of a standard basis vector, e.g. [0, 1, 0] -> 1'''
      x = np.cumsum(x)
      x = 1 - x
      return np.sum(x, dtype=np.int32)

    def dynamics(t, x, u):
      '''moves  the next standard basis vector'''
      idx = (position(x) + u[0]) % num_states
      return lax.dynamic_slice_in_dim(np.eye(num_states), idx, 1)[0]

    T = 8

    U = np.ones((T, 1), dtype=np.int32)
    X = control.trajectory(dynamics, U, np.eye(num_states, dtype=np.int32)[0])
    expected = np.vstack((np.eye(num_states),) * 3)
    self.assertAllClose(X, expected, check_dtypes=True)


  def testLqrSolve(self):
    dim, T = 2, 10
    p = one_step_lqr(dim, T)
    K, k = control.lqr_solve(p)
    K_ = -np.stack(T * (np.eye(dim),))
    self.assertAllClose(K, K_, check_dtypes=True, atol=1e-6, rtol=1e-6)
    self.assertAllClose(k, np.zeros((T, dim)), check_dtypes=True)


  def testLqrPredict(self):
    randn = onp.random.RandomState(0).randn
    dim, T = 2, 10
    p = one_step_lqr(dim, T)
    x0 = randn(dim)
    X, U = control.lqr_predict(p, x0)
    self.assertAllClose(X[0], x0, check_dtypes=True)
    self.assertAllClose(U[0], -x0, check_dtypes=True,
                        atol=1e-6, rtol=1e-6)
    self.assertAllClose(X[1:], np.zeros((T, 2)), check_dtypes=True,
                        atol=1e-6, rtol=1e-6)
    self.assertAllClose(U[1:], np.zeros((T - 1, 2)), check_dtypes=True,
                        atol=1e-6, rtol=1e-6)


  def testIlqrWithLqrProblem(self):
    randn = onp.random.RandomState(0).randn
    dim, T, num_iters = 2, 10, 3
    lqr = one_step_lqr(dim, T)
    p = control_from_lqr(lqr)
    x0 = randn(dim)
    X, U = control.ilqr(num_iters, p, x0, np.zeros((T, dim)))
    self.assertAllClose(X[0], x0, check_dtypes=True)
    self.assertAllClose(U[0], -x0, check_dtypes=True)
    self.assertAllClose(X[1:], np.zeros((T, 2)), check_dtypes=True)
    self.assertAllClose(U[1:], np.zeros((T - 1, 2)), check_dtypes=True)


  def testIlqrWithLqrProblemSpecifiedGenerally(self):
    randn = onp.random.RandomState(0).randn
    dim, T, num_iters = 2, 10, 3
    p = one_step_control(dim, T)
    x0 = randn(dim)
    X, U = control.ilqr(num_iters, p, x0, np.zeros((T, dim)))
    self.assertAllClose(X[0], x0, check_dtypes=True)
    self.assertAllClose(U[0], -x0, check_dtypes=True)
    self.assertAllClose(X[1:], np.zeros((T, 2)), check_dtypes=True)
    self.assertAllClose(U[1:], np.zeros((T - 1, 2)), check_dtypes=True)


  def testIlqrWithNonlinearProblem(self):
    def cost(t, x, u):
      return (x[0] ** 2. + 1e-3 * u[0] ** 2.) / (t + 1.)

    def dynamics(t, x, u):
      return (x ** 2. - u ** 2.) / (t + 1.)

    T, num_iters, d = 10, 7, 1
    p = control.ControlSpec(cost, dynamics, T, d, d)

    x0 = np.array([0.2])
    X, U = control.ilqr(num_iters, p, x0, 1e-5 * np.ones((T, d)))
    assert_close = partial(self.assertAllClose, atol=1e-2, check_dtypes=True)
    assert_close(X[0], x0)
    assert_close(U[0] ** 2., x0 ** 2.)
    assert_close(X[1:], np.zeros((T, d)))
    assert_close(U[1:], np.zeros((T - 1, d)))


  def testMpcWithLqrProblem(self):
    randn = onp.random.RandomState(0).randn
    dim, T, num_iters = 2, 10, 3
    lqr = one_step_lqr(dim, T)
    p = control_from_lqr(lqr)
    x0 = randn(dim)
    solver = partial(control.ilqr, num_iters)
    X, U = control.mpc_predict(solver, p, x0, np.zeros((T, dim)))
    self.assertAllClose(X[0], x0, check_dtypes=True)
    self.assertAllClose(U[0], -x0, check_dtypes=True)
    self.assertAllClose(X[1:], np.zeros((T, 2)), check_dtypes=True)
    self.assertAllClose(U[1:], np.zeros((T - 1, 2)), check_dtypes=True)


  def testMpcWithLqrProblemSpecifiedGenerally(self):
    randn = onp.random.RandomState(0).randn
    dim, T, num_iters = 2, 10, 3
    p = one_step_control(dim, T)
    x0 = randn(dim)
    solver = partial(control.ilqr, num_iters)
    X, U = control.mpc_predict(solver, p, x0, np.zeros((T, dim)))
    self.assertAllClose(X[0], x0, check_dtypes=True)
    self.assertAllClose(U[0], -x0, check_dtypes=True)
    self.assertAllClose(X[1:], np.zeros((T, 2)), check_dtypes=True)
    self.assertAllClose(U[1:], np.zeros((T - 1, 2)), check_dtypes=True)


  def testMpcWithNonlinearProblem(self):
    def cost(t, x, u):
      return (x[0] ** 2. + 1e-3 * u[0] ** 2.) / (t + 1.)

    def dynamics(t, x, u):
      return (x ** 2. - u ** 2.) / (t + 1.)

    T, num_iters, d = 10, 7, 1
    p = control.ControlSpec(cost, dynamics, T, d, d)

    x0 = np.array([0.2])
    solver = partial(control.ilqr, num_iters)
    X, U = control.mpc_predict(solver, p, x0, 1e-5 * np.ones((T, d)))
    assert_close = partial(self.assertAllClose, atol=1e-2, check_dtypes=True)
    assert_close(X[0], x0)
    assert_close(U[0] ** 2., x0 ** 2.)
    assert_close(X[1:], np.zeros((T, d)))
    assert_close(U[1:], np.zeros((T - 1, d)))


if __name__ == '__main__':
  absltest.main()
