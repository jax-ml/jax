# Copyright 2020 Google LLC
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

from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import test_util as jtu
from jax import jit
from jax.config import config
from jax.scipy.optimize.bfgs_minimize import fmin_bfgs
from scipy.optimize import minimize as smin
import numpy as onp

config.parse_flags_with_absl()


def value_and_grad(f, fprime):
  def func(x):
    return f(x), fprime(x)

  return func


def rosenbrock(np):
  def func(x):
    return np.sum(100. * np.diff(x) ** 2 + (1. - x[:-1]) ** 2)

  return func


def himmelblau(np):
  def func(p):
    x, y = p
    return (x ** 2 + y - 11.) ** 2 + (x + y ** 2 - 7.) ** 2

  return func


def matyas(np):
  def func(p):
    x, y = p
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

  return func


def eggholder(np):
  def func(p):
    x, y = p
    return - (y + 47) * np.sin(np.sqrt(np.abs(x / 2. + y + 47.))) - x * np.sin(
      np.sqrt(np.abs(x - (y + 47.))))

  return func


class TestBFGS(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_func={}_maxiter={}".format(func_and_init[0].__name__, maxiter),
     "maxiter": maxiter, "func_and_init": func_and_init}
    for maxiter in [None]
    for func_and_init in [(rosenbrock, jnp.zeros(2)),
                          (himmelblau, jnp.zeros(2)),
                          (matyas, jnp.ones(2) * 6.),
                          (eggholder, jnp.ones(2) * 100.)]))
  def test_minimize(self, maxiter, func_and_init):
    # Note, cannot compare step for step with scipy BFGS because our line search is _slightly_ different.

    func, x0 = func_and_init

    def compare(func, x0):
      self._CompileAndCheck(func, lambda: [x0])

      @jit
      def min_op(x0):
        result = fmin_bfgs(func(jnp), x0,
                           options=dict(ls_maxiter=100, maxiter=maxiter, gtol=1e-6))
        return result

      jax_res = min_op(x0)

      scipy_res = smin(func(onp), x0, method='BFGS')

      self.assertAllClose(scipy_res.x, jax_res.x_k, atol=2e-5, check_dtypes=False)

    compare(func, x0)


if __name__ == "__main__":
  absltest.main()
