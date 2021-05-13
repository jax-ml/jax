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
import numpy as np
import scipy.optimize

from jax import numpy as jnp
from jax import test_util as jtu
from jax import jit
from jax.config import config
import jax.scipy.optimize

config.parse_flags_with_absl()


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


def zakharovFromIndices(x, ii):
  sum1 = (x**2).sum()
  sum2 = (0.5*ii*x).sum()
  answer = sum1+sum2**2+sum2**4
  return answer


class TestBFGS(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_func={}_maxiter={}".format(func_and_init[0].__name__, maxiter),
     "maxiter": maxiter, "func_and_init": func_and_init}
    for maxiter in [None]
    for func_and_init in [(rosenbrock, np.zeros(2)),
                          (himmelblau, np.zeros(2)),
                          (matyas, np.ones(2) * 6.),
                          (eggholder, np.ones(2) * 100.)]))
  def test_minimize(self, maxiter, func_and_init):
    # Note, cannot compare step for step with scipy BFGS because our line search is _slightly_ different.

    func, x0 = func_and_init

    @jit
    def min_op(x0):
      result = jax.scipy.optimize.minimize(
          func(jnp),
          x0,
          method='BFGS',
          options=dict(maxiter=maxiter, gtol=1e-6),
      )
      return result.x

    jax_res = min_op(x0)
    scipy_res = scipy.optimize.minimize(func(np), x0, method='BFGS').x
    self.assertAllClose(scipy_res, jax_res, atol=2e-5, check_dtypes=False)

  def test_fixes4594(self):
    n = 2
    A = jnp.eye(n) * 1e4
    def f(x):
      return jnp.mean((A @ x) ** 2)
    results = jax.scipy.optimize.minimize(f, jnp.ones(n), method='BFGS')
    self.assertAllClose(results.x, jnp.zeros(n), atol=1e-6, rtol=1e-6)

  @jtu.skip_on_flag('jax_enable_x64', False)
  def test_zakharov(self):
    def zakharov_fn(x):
      ii = jnp.arange(1, len(x) + 1, step=1)
      answer = zakharovFromIndices(x=x, ii=ii)
      return answer

    x0 = jnp.array([600.0, 700.0, 200.0, 100.0, 90.0, 1e4])
    eval_func = jax.jit(zakharov_fn)
    jax_res = jax.scipy.optimize.minimize(fun=eval_func, x0=x0, method='BFGS')
    self.assertLess(jax_res.fun, 1e-6)

  def test_minimize_bad_initial_values(self):
    # This test runs deliberately "bad" initial values to test that handling
    # of failed line search, etc. is the same across implementations
    initial_value = jnp.array([92, 0.001])
    opt_fn = himmelblau(jnp)
    jax_res = jax.scipy.optimize.minimize(
        fun=opt_fn,
        x0=initial_value,
        method='BFGS',
    ).x
    scipy_res = scipy.optimize.minimize(
        fun=opt_fn,
        jac=jax.grad(opt_fn),
        method='BFGS',
        x0=initial_value
    ).x
    self.assertAllClose(scipy_res, jax_res, atol=2e-5, check_dtypes=False)


  def test_args_must_be_tuple(self):
    A = jnp.eye(2) * 1e4
    def f(x):
      return jnp.mean((A @ x) ** 2)
    with self.assertRaisesRegex(TypeError, "args .* must be a tuple"):
      jax.scipy.optimize.minimize(f, jnp.ones(2), args=45, method='BFGS')


if __name__ == "__main__":
  absltest.main()
