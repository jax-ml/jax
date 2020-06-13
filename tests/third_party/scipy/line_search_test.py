import numpy as np
from absl.testing import absltest, parameterized
from jax.config import config
from jax import test_util as jtu, numpy as jnp, value_and_grad, grad
from jax.scipy.optimize.line_search import line_search
from scipy.optimize.linesearch import line_search_wolfe2

config.parse_flags_with_absl()


def f_and_fprime(f, fprime):
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
    p = -s - s ** 3 + s ** 4
    dp = -1 - 3 * s ** 2 + 4 * s ** 3
    return p, dp

  def _scalar_func_2(self, s):
    p = jnp.exp(-4 * s) + s ** 2
    dp = -4 * jnp.exp(-4 * s) + 2 * s
    return p, dp

  def _scalar_func_3(self, s):
    p = -jnp.sin(10 * s)
    dp = -10 * jnp.cos(10 * s)
    return p, dp

  # -- n-d functions

  def _line_func_1(self, x):
    f = jnp.dot(x, x)
    df = 2 * x
    return f, df

  def _line_func_2(self, x):
    f = jnp.dot(x, jnp.dot(self.A, x)) + 1
    df = jnp.dot(self.A + self.A.T, x)
    return f, df

  # -- Generic scalar searches

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_name={}".format(name), "name": name}
    for name in ['_scalar_func_1',
                 '_scalar_func_2',
                 '_scalar_func_3']))
  def test_scalar_search_wolfe2(self, name):

    def bind_index(func, idx):
      # Remember Python's closure semantics!
      return lambda *a, **kw: func(*a, **kw)[idx]

    value = getattr(self, name)
    phi = bind_index(value, 0)
    derphi = bind_index(value, 1)
    for old_phi0 in np.random.randn(3):
      res = line_search(f_and_fprime(phi, derphi), 0., 1.)
      s, phi1, derphi1 = res.a_k, res.f_k, res.g_k
      self.assertAllClose(phi1, phi(s), check_dtypes=False, atol=1e-6)
      if derphi1 is not None:
        self.assertAllClose(derphi1, derphi(s), check_dtypes=False, atol=1e-6)
      self.assert_wolfe(s, phi, derphi, err_msg="%s %g" % (name, old_phi0))

  # -- Generic line searches

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_name={}".format(name), "name": name}
    for name in ['_line_func_1',
                 '_line_func_2']))
  def test_line_search_wolfe2(self, name):
    def bind_index(func, idx):
      # Remember Python's closure semantics!
      return lambda *a, **kw: func(*a, **kw)[idx]

    value = getattr(self, name)
    f = bind_index(value, 0)
    fprime = bind_index(value, 1)

    k = 0
    N = 20
    np.random.seed(1234)
    # sets A in one of the line funcs
    self.A = np.random.randn(N, N)
    while k < 9:
      x = np.random.randn(N)
      p = np.random.randn(N)
      if jnp.dot(p, fprime(x)) >= 0:
        # always pick a descent direction
        continue
      k += 1

      f0 = f(x)
      g0 = fprime(x)
      self.fcount = 0
      res = line_search(f_and_fprime(f, fprime), x, p, old_fval=f0, gfk=g0)
      s = res.a_k
      fv = res.f_k
      gv = res.g_k
      self.assertAllClose(fv, f(x + s * p), check_dtypes=False, atol=1e-5)
      if gv is not None:
        self.assertAllClose(gv, fprime(x + s * p), check_dtypes=False, atol=1e-5)

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

    res = line_search(f_and_fprime(f, fp), x, p, c2=c2)
    s = res.a_k
    # s, _, _, _, _, _ = ls.line_search_wolfe2(f, fp, x, p, amax=30, c2=c2)
    self.assert_line_wolfe(x, p, s, f, fp)
    self.assertTrue(s >= 30.)

    res = line_search(f_and_fprime(f, fp), x, p, c2=c2, maxiter=5)
    self.assertTrue(res.failed)
    # s=30 will only be tried on the 6th iteration, so this won't converge

  def test_line_search(self):

    def f(x):
      return jnp.cos(jnp.sum(jnp.exp(-x)) ** 2)

    # assert not line_search(jax.value_and_grad(f), np.ones(2), np.array([-0.5, -0.25])).failed
    xk = jnp.ones(2)
    pk = jnp.array([-0.5, -0.25])
    res = line_search(value_and_grad(f), xk, pk, maxiter=100)

    scipy_res = line_search_wolfe2(f, grad(f), xk, pk)

    self.assertAllClose(scipy_res[0], res.a_k, atol=1e-5, check_dtypes=False)
    self.assertAllClose(scipy_res[3], res.f_k, atol=1e-5, check_dtypes=False)

  # -- More specific tests


if __name__ == "__main__":
  absltest.main()
