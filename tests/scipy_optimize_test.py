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

from absl.testing import absltest

from jax import numpy as jnp
from jax import test_util as jtu
from jax.config import config
from jax.scipy.optimize.bfgs_minimize import bfgs_minimize
from jax.scipy.optimize.line_search import line_search
import numpy as np

config.parse_flags_with_absl()


def value_and_grad(f, fprime):
  def func(x):
    return f(x), fprime(x)

  return func


class TestLineSearch(jtu.JaxTestCase):
  # -- scalar functions; must have dphi(0.) < 0

  def assert_wolfe(self, s, phi, derphi, c1=1e-4, c2=0.9, err_msg=""):
    """
    Check that strong Wolfe conditions apply
    """
    phi1 = phi(s)
    phi0 = phi(0)
    derphi0 = derphi(0)
    derphi1 = derphi(s)
    msg = "s = %s; phi(0) = %s; phi(s) = %s; phi'(0) = %s; phi'(s) = %s; %s" % (
      s, phi0, phi1, derphi0, derphi1, err_msg)

    self.assertTrue(phi1 <= phi0 + c1 * s * derphi0, "Wolfe 1 failed: " + msg)
    self.assertTrue(abs(derphi1) <= abs(c2 * derphi0), "Wolfe 2 failed: " + msg)

  def assert_line_wolfe(self, x, p, s, f, fprime, **kw):
    self.assert_wolfe(s, phi=lambda sp: f(x + p * sp),
                      derphi=lambda sp: jnp.dot(fprime(x + p * sp), p), **kw)

  def _scalar_func_1(self, s):
    self.fcount += 1
    p = -s - s ** 3 + s ** 4
    dp = -1 - 3 * s ** 2 + 4 * s ** 3
    return p, dp

  def _scalar_func_2(self, s):
    self.fcount += 1
    p = jnp.exp(-4 * s) + s ** 2
    dp = -4 * jnp.exp(-4 * s) + 2 * s
    return p, dp

  def _scalar_func_3(self, s):
    self.fcount += 1
    p = -jnp.sin(10 * s)
    dp = -10 * jnp.cos(10 * s)
    return p, dp

  # -- n-d functions

  def _line_func_1(self, x):
    self.fcount += 1
    f = jnp.dot(x, x)
    df = 2 * x
    return f, df

  def _line_func_2(self, x):
    self.fcount += 1
    f = jnp.dot(x, jnp.dot(self.A, x)) + 1
    df = jnp.dot(self.A + self.A.T, x)
    return f, df

  # --

  def setup_method(self):
    self.scalar_funcs = []
    self.line_funcs = []
    self.N = 20
    self.fcount = 0

    def bind_index(func, idx):
      # Remember Python's closure semantics!
      return lambda *a, **kw: func(*a, **kw)[idx]

    for name in sorted(dir(self)):
      if name.startswith('_scalar_func_'):
        value = getattr(self, name)
        self.scalar_funcs.append(
          (name, bind_index(value, 0), bind_index(value, 1)))
      elif name.startswith('_line_func_'):
        value = getattr(self, name)
        self.line_funcs.append(
          (name, bind_index(value, 0), bind_index(value, 1)))

    np.random.seed(1234)
    self.A = np.random.randn(self.N, self.N)

  def scalar_iter(self):
    for name, phi, derphi in self.scalar_funcs:
      for old_phi0 in np.random.randn(3):
        yield name, phi, derphi, old_phi0

  def line_iter(self):
    for name, f, fprime in self.line_funcs:
      k = 0
      while k < 9:
        x = np.random.randn(self.N)
        p = np.random.randn(self.N)
        if jnp.dot(p, fprime(x)) >= 0:
          # always pick a descent direction
          continue
        k += 1
        old_fv = float(np.random.randn())
        yield name, f, fprime, x, p, old_fv

  # -- Generic scalar searches

  def test_scalar_search_wolfe2(self):
    for name, phi, derphi, old_phi0 in self.scalar_iter():
      res = line_search(value_and_grad(phi, derphi), 0., 1.)
      s, phi1, derphi1 = res.a_k, res.f_k, res.g_k
      # s, phi1, phi0, derphi1 = ls.scalar_search_wolfe2(
      #     phi, derphi, phi(0), old_phi0, derphi(0))
      self.assertAllClose(phi1, phi(s), check_dtypes=False, atol=1e-6)
      if derphi1 is not None:
        self.assertAllClose(derphi1, derphi(s), check_dtypes=False, atol=1e-6)
      self.assert_wolfe(s, phi, derphi, err_msg="%s %g" % (name, old_phi0))

  # -- Generic line searches

  def test_line_search_wolfe2(self):
    c = 0
    smax = 512
    for name, f, fprime, x, p, old_f in self.line_iter():
      f0 = f(x)
      g0 = fprime(x)
      self.fcount = 0
      res = line_search(value_and_grad(f, fprime), x, p, f_0=f0, g_0=g0)
      s = res.a_k
      fc = res.nfev
      gc = res.ngev
      fv = res.f_k
      gv = res.g_k
      # s, fc, gc, fv, ofv, gv = ls.line_search_wolfe2(f, fprime, x, p,
      #                                                g0, f0, old_f,
      #                                                amax=smax)
      # assert_equal(self.fcount, fc+gc)
      self.assertAllClose(fv, f(x + s * p), check_dtypes=False, atol=1e-5)
      if gv is not None:
        self.assertAllClose(gv, fprime(x + s * p), check_dtypes=False, atol=1e-5)
      if s < smax:
        c += 1
    self.assertTrue(c > 3)  # check that the iterator really works...

  def test_line_search_wolfe2_bounds(self):
    # See gh-7475

    # For this f and p, starting at a point on axis 0, the strong Wolfe
    # condition 2 is met if and only if the step length s satisfies
    # |x + s| <= c2 * |x|
    f = lambda x: jnp.dot(x, x)
    fp = lambda x: 2 * x
    p = jnp.array([1, 0])

    # Smallest s satisfying strong Wolfe conditions for these arguments is 30
    x = -60 * p
    c2 = 0.5

    res = line_search(value_and_grad(f, fp), x, p, c2=c2)
    s = res.a_k
    # s, _, _, _, _, _ = ls.line_search_wolfe2(f, fp, x, p, amax=30, c2=c2)
    self.assert_line_wolfe(x, p, s, f, fp)
    self.assertTrue(s >= 30.)

    res = line_search(value_and_grad(f, fp), x, p, c2=c2, max_iterations=5)
    self.assertTrue(res.failed)
    # s=30 will only be tried on the 6th iteration, so this won't converge

  def test_line_search(self):
    import jax

    import jax.numpy as np

    def f(x):
      return np.cos(np.sum(np.exp(-x)) ** 2)

    # assert not line_search(jax.value_and_grad(f), np.ones(2), np.array([-0.5, -0.25])).failed
    xk = np.ones(2)
    pk = np.array([-0.5, -0.25])
    res = line_search(jax.value_and_grad(f), xk, pk, max_iterations=100)

    from scipy.optimize.linesearch import line_search_wolfe2

    scipy_res = line_search_wolfe2(f, jax.grad(f), xk, pk)

    # print(scipy_res[0], res.a_k)
    # print(scipy_res[3], res.f_k)

    self.assertAllClose(scipy_res[0], res.a_k, atol=1e-5, check_dtypes=False)
    self.assertAllClose(scipy_res[3], res.f_k, atol=1e-5, check_dtypes=False)

  # -- More specific tests


class TestBFGS(jtu.JaxTestCase):
  def test_minimize(self):
    from scipy.optimize import minimize as smin
    import numpy as onp

    def compare(func, x0):
      # @jax.jit
      def min_op(x0):
        result = bfgs_minimize(func(jnp), x0,
                               options=dict(ls_maxiter=10, maxiter=10, analytic_initial_hessian=False,
                                            g_tol=1e-6))
        return result

      jax_res = min_op(x0)

      scipy_res = smin(func(onp), x0, method='BFGS')
      # print(jax_res)
      # print(scipy_res)
      self.assertAllClose(scipy_res.x, jax_res.x_k, atol=1e-5, check_dtypes=False)

    def rosenbrock(np):
      def func(x):
        return np.sum(100. * np.diff(x) ** 2 + (1. - x[:-1]) ** 2)

      return func

    x0 = jnp.zeros(2)

    compare(rosenbrock, x0)

    def himmelblau(np):
      def func(p):
        x, y = p
        return (x ** 2 + y - 11.) ** 2 + (x + y ** 2 - 7.) ** 2

      return func

    x0 = jnp.zeros(2)

    compare(himmelblau, x0)

    def matyas(np):
      def func(p):
        x, y = p
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

      return func

    x0 = jnp.ones(2) * 6.

    compare(matyas, x0)

    def eggholder(np):
      def func(p):
        x, y = p
        return - (y + 47) * np.sin(np.sqrt(np.abs(x / 2. + y + 47.))) - x * np.sin(
          np.sqrt(np.abs(x - (y + 47.))))

      return func

    x0 = jnp.ones(2) * 100.

    compare(eggholder, x0)


if __name__ == "__main__":
  absltest.main()
