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

from functools import partial

from absl.testing import parameterized
from absl.testing import absltest
import numpy as np
import scipy.sparse.linalg

from jax import jit
import jax.numpy as jnp
from jax import lax
from jax import test_util as jtu
from jax.tree_util import register_pytree_node_class
import jax.scipy.sparse.linalg

from jax.config import config
config.parse_flags_with_absl()
config.update("jax_enable_x64", True)

float_types = [np.float32, np.float64]
complex_types = [np.complex64, np.complex128]


def matmul_high_precision(a, b):
  return jnp.matmul(a, b, precision=lax.Precision.HIGHEST)


@jit
def posify(matrix):
  return matmul_high_precision(matrix, matrix.T.conj())


def lax_cg(A, b, M=None, atol=0.0, **kwargs):
  A = partial(matmul_high_precision, A)
  if M is not None:
    M = partial(matmul_high_precision, M)
  x, _ = jax.scipy.sparse.linalg.cg(A, b, atol=atol, M=M, **kwargs)
  return x


def lax_bicgstab(A, b, M=None, rtol=1e-5, atol=0.0, **kwargs):
  A = partial(matmul_high_precision, A)
  if M is not None:
    M = partial(matmul_high_precision, M)
  x, _ = jax.scipy.sparse.linalg.bicgstab(A, b, atol=atol, tol=rtol,
                                          M=M, **kwargs)
  return x


def scipy_cg(A, b, atol=0.0, **kwargs):
  x, info = scipy.sparse.linalg.cg(A, b, atol=atol, **kwargs)
  return x


def scipy_bicgstab(A, b, atol=0.0, **kwargs):
  x, info = scipy.sparse.linalg.bicgstab(A, b, atol=atol, **kwargs)
  return x


def rand_sym_pos_def(rng, shape, dtype):
  matrix = np.eye(N=shape[0], dtype=dtype) + rng(shape, dtype)
  return matrix @ matrix.T.conj()


def numpy_bicgstab(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=None):
  A = partial(matmul_high_precision, A)
  x0 = np.zeros_like(b) if x0 is None else x0
  # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.bicgstab
  bs = np.linalg.norm(b) ** 2
  atol2 = jnp.maximum(np.square(tol) * bs, np.square(atol))

  # https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method#Preconditioned_BiCGSTAB

  _identity = (lambda x: x)
  M = _identity if M is None else partial(matmul_high_precision, M)

  def cond_fun(value):
    x, r, *_, k = value
    rs = np.vdot(r, r).real
    return (rs > atol2) & (k < maxiter)

  def body_fun(value):
    x, r, rhat, alpha, omega, rho, p, q, k = value
    rho_ = np.vdot(rhat, r)
    beta = rho_ / rho * alpha / omega
    p_ = r + beta * (p - (omega * q))
    phat = M(p_)
    q_ = A(phat)
    alpha_ = rho_ / np.vdot(rhat, q_)
    s = r - alpha_ * q_
    if np.vdot(s, s).real < atol2:
      return x + alpha_ * phat, s, rhat, alpha_, omega, rho_, p_, q_, k + 1
    shat = M(s)
    t = A(shat)
    omega_ = np.vdot(t, s) / np.vdot(t, t)
    x_ = x + alpha_ * phat + omega_ * shat
    r_ = s - omega_ * t
    return x_, r_, rhat, alpha_, omega_, rho_, p_, q_, k + 1

  r0 = b - A(x0)
  rho0 = alpha0 = omega0 = np.vdot(b, b) / np.vdot(b, b)
  initial_value = (x0, r0, r0, alpha0, omega0, rho0, r0, r0, 0)

  val = initial_value
  while cond_fun(val):
    val = body_fun(val)
  x_final, *_ = val

  return x_final


def numpy_cg(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=None):
  A = partial(matmul_high_precision, A)
  x0 = np.zeros_like(b) if x0 is None else x0
  # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
  bs = np.linalg.norm(b) ** 2
  atol2 = np.maximum(np.square(tol) * bs, np.square(atol))

  # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

  _identity = (lambda x: x)
  M = _identity if M is None else partial(matmul_high_precision, M)

  def cond_fun(value):
    x, r, gamma, p, k = value
    rs = gamma if M is _identity else np.linalg.norm(r) ** 2
    return (rs > atol2) & (k < maxiter)

  def body_fun(value):
    x, r, gamma, p, k = value
    Ap = A(p)
    alpha = gamma / np.vdot(p, Ap).real
    x_ = x + alpha * p
    r_ = r - alpha * Ap
    z_ = M(r_)
    gamma_ = np.vdot(r_, z_).real
    beta_ = gamma_ / gamma
    p_ = z_ + beta_ * p
    return x_, r_, gamma_, p_, k + 1

  r0 = b - A(x0)
  p0 = z0 = M(r0)
  gamma0 = np.vdot(r0, z0).real
  initial_value = (x0, r0, gamma0, p0, 0)

  val = initial_value
  while cond_fun(val):
    val = body_fun(val)
  x_final, *_ = val

  return x_final


def poisson(shape, dtype):
  n = shape[0]
  data = np.ones((3, n), dtype=dtype)
  data[0, :] = 2
  data[1, :] = -1
  data[2, :] = -1
  a = scipy.sparse.spdiags(data, [0, -1, 1], n, n, format='csr')
  a.sort_indices()
  return a.A


class LaxBackedScipyTests(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_preconditioner={}_isolve={}".format(
            jtu.format_shape_dtype_string(shape, dtype),
            preconditioner, isolve[-1].__name__),
       "shape": shape, "dtype": dtype, "preconditioner": preconditioner,
        "isolve": isolve}
      for shape in [(4, 4), (7, 7), (32, 32)]
      for dtype in float_types + complex_types
      for preconditioner in [None, 'identity', 'exact', 'random']
      for isolve in [(scipy_bicgstab, lax_bicgstab), (scipy_cg, lax_cg)]))
  # TODO(#2951): reenable 'random' preconditioner.
  def test_isolve_against_scipy(self, shape, dtype, preconditioner, isolve):

    scipy_isolve, lax_isolve = isolve
    rng = jtu.rand_default(self.rng())
    if lax_isolve == lax_cg:
      A = rand_sym_pos_def(rng, shape, dtype)
    else:
      A = rng(shape, dtype)
    b = rng(shape[:1], dtype)

    if preconditioner == 'identity':
      M = np.eye(shape[0], dtype=dtype)
    elif preconditioner == 'random':
      M = np.linalg.inv(rand_sym_pos_def(rng, shape, dtype))
    elif preconditioner == 'exact':
      M = np.linalg.inv(A)
    else:
      M = None

    def args_maker():
      return A, b

    # sanity check with scipy first,
    # then check against numpy if agree
    # exposes some testing issues...
    expected = np.linalg.solve(A, b)
    scipy_result = scipy_isolve(A, b, M=M)
    if np.allclose(expected, scipy_result, rtol=2e-2):
      self._CheckAgainstNumpy(
        np.linalg.solve,
        partial(lax_isolve, M=M, atol=1e-6),
        args_maker,
        tol=2e-2)
    else:
      max_adiff = np.max(np.abs(expected - scipy_result))
      max_rdiff = np.max(np.abs(expected - scipy_result) / np.abs(expected))
      print(
        'scipy/numpy differ abs: {}, rel: {}'.format(max_adiff, max_rdiff))

    self._CheckAgainstNumpy(
        partial(scipy_isolve, M=M, maxiter=1),
        partial(lax_isolve, M=M, maxiter=1),
        args_maker,
        tol=1e-3)

    # TODO(shoyer,mattjj): I had to loosen the tolerance for complex64[7,7]
    #  with preconditioner=random
    self._CheckAgainstNumpy(
      partial(scipy_isolve, M=M, maxiter=3),
      partial(lax_isolve, M=M, maxiter=3),
      args_maker,
      tol=3e-3)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_isolve={}".format(jtu.format_shape_dtype_string(shape, dtype),
                                    isolve[1].__name__),
       "shape": shape, "dtype": dtype, "isolve": isolve}
      for shape in [(2, 2)]
      for dtype in float_types + complex_types
      for isolve in [(scipy_bicgstab, lax_bicgstab), (scipy_cg, lax_cg)]))
  def test_isolve_as_solve(self, shape, dtype, isolve):
    rng = jtu.rand_default(self.rng())
    a = rng(shape, dtype)
    a = posify(a) if isolve[1] == lax_cg else a
    b = rng(shape[:1], dtype)

    scipy_isolve, lax_isolve = isolve

    expected = np.linalg.solve(a, b)

    # test lax second to double check that it matches with scipy
    actual = lax_isolve(a, b)
    self.assertAllClose(expected, actual, rtol=1e-5)

    # check jit compilation
    actual = jit(lax_isolve)(a, b)
    self.assertAllClose(expected, actual, rtol=1e-5)

    jtu.check_grads(
        lambda x, y: lax_isolve(x, y),
        (a, b), order=2, rtol=3e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
           "isolve={}".format(isolve.__name__), "isolve": isolve}
      for isolve in [jax.scipy.sparse.linalg.bicgstab,
                     jax.scipy.sparse.linalg.cg]))
  def test_isolve_ndarray(self, isolve):
    A = lambda x: 2 * x
    b = jnp.arange(9.0).reshape((3, 3))
    expected = b / 2
    actual, _ = isolve(A, b)
    self.assertAllClose(expected, actual)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
           "isolve={}".format(isolve.__name__), "isolve": isolve}
      for isolve in [jax.scipy.sparse.linalg.bicgstab,
                     jax.scipy.sparse.linalg.cg]))
  def test_isolve_pytree(self, isolve):
    A = lambda x: {"a": x["a"] + 0.5 * x["b"], "b": 0.5 * x["a"] + x["b"]}
    b = {"a": 1.0, "b": -4.0}
    expected = {"a": 4.0, "b": -6.0}
    actual, _ = isolve(A, b)
    self.assertEqual(expected.keys(), actual.keys())
    self.assertAlmostEqual(expected["a"], actual["a"], places=6)
    self.assertAlmostEqual(expected["b"], actual["b"], places=6)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
           "isolve={}".format(isolve.__name__), "isolve": isolve}
      for isolve in [jax.scipy.sparse.linalg.bicgstab,
                     jax.scipy.sparse.linalg.cg]))
  def test_isolve_errors(self, isolve):
    A = lambda x: x
    b = jnp.zeros((2,))
    with self.assertRaisesRegex(
        ValueError, "x0 and b must have matching tree structure"):
      isolve(A, {'x': b}, {'y': b})
    with self.assertRaisesRegex(
        ValueError, "x0 and b must have matching shape"):
      isolve(A, b, b[:, np.newaxis])

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
           "isolve={}".format(isolve.__name__), "isolve": isolve}
      for isolve in [jax.scipy.sparse.linalg.bicgstab,
                     jax.scipy.sparse.linalg.cg]))
  def test_isolve_without_pytree_equality(self, isolve):

    @register_pytree_node_class
    class MinimalPytree:
      def __init__(self, value):
        self.value = value

      def tree_flatten(self):
        return [self.value], None

      @classmethod
      def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    A = lambda x: MinimalPytree(2 * x.value)
    b = MinimalPytree(jnp.arange(5.0))
    expected = b.value / 2
    actual, _ = isolve(A, b)
    self.assertAllClose(expected, actual.value)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
