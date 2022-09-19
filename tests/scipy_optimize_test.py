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
import scipy
import scipy.optimize

from jax import numpy as jnp
from jax._src import test_util as jtu
from jax import jit, value_and_grad
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


def newton_sqrt(x, a):
  return 0.5 * (x + a / x)


def zakharovFromIndices(x, ii):
  sum1 = (x**2).sum()
  sum2 = (0.5*ii*x).sum()
  answer = sum1+sum2**2+sum2**4
  return answer


class TestBFGS(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": f"_func={func_and_init[0].__name__}_maxiter={maxiter}",
     "maxiter": maxiter, "func_and_init": func_and_init}
    for maxiter in [None]
    for func_and_init in [(rosenbrock, np.zeros(2, dtype='float32')),
                          (himmelblau, np.ones(2, dtype='float32')),
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
      ii = jnp.arange(1, len(x) + 1, step=1, dtype=x.dtype)
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


class TestLBFGS(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": f"_func={func_and_init[0].__name__}_maxiter={maxiter}",
     "maxiter": maxiter, "func_and_init": func_and_init}
    for maxiter in [None]
    for func_and_init in [(rosenbrock, np.zeros(2)),
                          (himmelblau, np.zeros(2)),
                          (matyas, np.ones(2) * 6.),
                          (eggholder, np.ones(2) * 100.)]))
  def test_minimize(self, maxiter, func_and_init):

    func, x0 = func_and_init

    @jit
    def min_op(x0):
      result = jax.scipy.optimize.minimize(
          func(jnp),
          x0,
          method='l-bfgs-experimental-do-not-rely-on-this',
          options=dict(maxiter=maxiter, gtol=1e-7),
      )
      return result.x

    jax_res = min_op(x0)

    # Note that without bounds, L-BFGS-B is just L-BFGS
    with jtu.ignore_warning(category=DeprecationWarning,
                            message=".*tostring.*is deprecated.*"):
      scipy_res = scipy.optimize.minimize(func(np), x0, method='L-BFGS-B').x

    if func.__name__ == 'matyas':
      # scipy performs badly for Matyas, compare to true minimum instead
      self.assertAllClose(jax_res, jnp.zeros_like(jax_res), atol=1e-7)
      return

    if func.__name__ == 'eggholder':
      # L-BFGS performs poorly for the eggholder function.
      # Neither scipy nor jax find the true minimum, so we can only loosely (with high atol) compare the false results
      self.assertAllClose(jax_res, scipy_res, atol=1e-3)
      return

    self.assertAllClose(jax_res, scipy_res, atol=2e-5, check_dtypes=False)

  def test_minimize_complex_sphere(self):
    z0 = jnp.array([1., 2. - 3.j, 4., -5.j])

    def f(z):
      return jnp.real(jnp.dot(jnp.conj(z - z0), z - z0))

    @jit
    def min_op(x0):
      result = jax.scipy.optimize.minimize(
          f,
          x0,
          method='l-bfgs-experimental-do-not-rely-on-this',
          options=dict(gtol=1e-6),
      )
      return result.x

    jax_res = min_op(jnp.zeros_like(z0))

    self.assertAllClose(jax_res, z0)

  def test_complex_rosenbrock(self):
    complex_dim = 5

    f_re = rosenbrock(jnp)
    init_re = jnp.zeros((2 * complex_dim,), dtype=complex)
    expect_re = jnp.ones((2 * complex_dim,), dtype=complex)

    def f(z):
      x_re = jnp.concatenate([jnp.real(z), jnp.imag(z)])
      return f_re(x_re)

    init = init_re[:complex_dim] + 1.j * init_re[complex_dim:]
    expect = expect_re[:complex_dim] + 1.j * expect_re[complex_dim:]

    @jit
    def min_op(z0):
      result = jax.scipy.optimize.minimize(
          f,
          z0,
          method='l-bfgs-experimental-do-not-rely-on-this',
          options=dict(gtol=1e-6),
      )
      return result.x

    jax_res = min_op(init)
    self.assertAllClose(jax_res, expect, atol=2e-5)


class TestFixedPoint(jtu.JaxTestCase):
    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": f"_func={func_init_args_and_res[0].__name__}_maxiter={maxiter}_xtol={xtol}",
                "maxiter": maxiter,
                "xtol": xtol,
                "func_init_args_and_res": func_init_args_and_res,
            }
            for maxiter in [500]
            for xtol in [1e-8]
            for func_init_args_and_res in [(newton_sqrt, 2.0, (2.0,), np.sqrt(2.0))]
        )
    )
    def test_fixed_point(self, maxiter, xtol, func_init_args_and_res):
        func, x0, args, expected_res = func_init_args_and_res

        # Scipy supported methods
        for method in ("iteration", "del2"):
            jax_res = jax.scipy.optimize.fixed_point(
                func, x0, args, xtol, maxiter, method=method
            )
            scipy_res = scipy.optimize.fixed_point(
                func, x0, args, xtol, maxiter, method=method
            )

            self.assertAllClose(jax_res, scipy_res, atol=1e-10)
            self.assertAllClose(jax_res, expected_res, atol=1e-10)

        # Methods only supported by jax
        for method in ("newton",):
            jax_res = jax.scipy.optimize.fixed_point(
                func, x0, args, xtol, maxiter, method=method
            )

            self.assertAllClose(jax_res, expected_res, atol=1e-10)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": f"_maxiter={maxiter}_xtol={xtol}",
                "maxiter": maxiter,
                "xtol": xtol,
            }
            for maxiter in [500]
            for xtol in [1e-8]
        )
    )
    def test_gradient_sqrt(self, xtol, maxiter):
        sqrt_v_and_g = value_and_grad(jnp.sqrt)

        def f_iteration(x):
            return jax.scipy.optimize.fixed_point(
                newton_sqrt, x, (x,), xtol, maxiter, method="iteration"
            )

        def f_del2(x):
            return jax.scipy.optimize.fixed_point(
                newton_sqrt, x, (x,), xtol, maxiter, method="del2"
            )

        def f_newton(x):
            return jax.scipy.optimize.fixed_point(
                newton_sqrt, x, (x,), xtol, maxiter, method="newton"
            )

        fixed_point_iteration_v_and_g = value_and_grad(f_iteration)
        fixed_point_del2_v_and_g = value_and_grad(f_del2)
        fixed_point_newton_v_and_g = value_and_grad(f_newton)

        for x in jnp.linspace(1, 10, 10):
            sqrt_res, sqrt_g = sqrt_v_and_g(x)
            (
                fixed_point_iteration_res,
                fixed_point_iteration_g,
            ) = fixed_point_iteration_v_and_g(x)
            fixed_point_del2_res, fixed_point_del2_g = fixed_point_del2_v_and_g(x)
            fixed_point_newton_res, fixed_point_newton_g = fixed_point_newton_v_and_g(x)

            self.assertAllClose(sqrt_res, fixed_point_iteration_res, atol=1e-10)
            self.assertAllClose(sqrt_g, fixed_point_iteration_g, atol=1e-10)
            self.assertAllClose(sqrt_res, fixed_point_del2_res, atol=1e-10)
            self.assertAllClose(sqrt_g, fixed_point_del2_g, atol=1e-10)
            self.assertAllClose(sqrt_res, fixed_point_newton_res, atol=1e-10)
            self.assertAllClose(sqrt_g, fixed_point_newton_g, atol=1e-10)

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {
                "testcase_name": f"_maxiter={maxiter}_xtol={xtol}_init_args={init_args_and_res[0]}",
                "maxiter": maxiter,
                "xtol": xtol,
                "init_args_and_res": init_args_and_res,
            }
            for maxiter in [500]
            for xtol in [1e-8]
            for init_args_and_res in [(2.+0.j, jnp.sqrt(2.+0.j)), (2.+1j, jnp.sqrt(2.+1j))]
        )
    )
    def test_complex_sqrt(self, xtol, maxiter, init_args_and_res):
        x0, expected_res = init_args_and_res

        for method in ("iteration", "del2", "newton"):
            jax_res = jax.scipy.optimize.fixed_point(
                newton_sqrt, x0, (x0,), xtol, maxiter, method=method
            )
            self.assertAllClose(jax_res, expected_res, atol=1e-10)


if __name__ == "__main__":
  absltest.main()
