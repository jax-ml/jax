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


import collections
import functools
from functools import partial
import itertools
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import scipy.special as osp_special

from jax._src import api
from jax import numpy as jnp
from jax import lax
from jax import scipy as jsp
from jax import test_util as jtu
from jax.scipy import special as lsp_special
import jax._src.scipy.eigh

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

all_shapes = [(), (4,), (3, 4), (3, 1), (1, 4), (2, 1, 4)]
compatible_shapes = [[(), ()],
                     [(4,), (3, 4)],
                     [(3, 1), (1, 4)],
                     [(2, 3, 4), (2, 1, 4)]]

float_dtypes = jtu.dtypes.floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.integer

# Params for the polar tests.
polar_shapes = [(16, 12), (12, 16), (128, 128)]
n_zero_svs = [0, 4]
degeneracies = [0, 4]
geometric_spectra = [False, True]
max_svs = [0.1, 10.]
nonzero_condition_numbers = [0.1, 100000]
sides = ["right", "left"]
methods = ["qdwh", "svd"]
seeds = [1, 10]

linear_sizes = [16, 128, 256]


def _initialize_polar_test(shape, n_zero_svs, degeneracy, geometric_spectrum,
                           max_sv, nonzero_condition_number, dtype):

  n_rows, n_cols = shape
  min_dim = min(shape)
  left_vecs = np.random.randn(n_rows, min_dim).astype(np.float64)
  left_vecs, _ = np.linalg.qr(left_vecs)
  right_vecs = np.random.randn(n_cols, min_dim).astype(np.float64)
  right_vecs, _ = np.linalg.qr(right_vecs)

  min_nonzero_sv = max_sv / nonzero_condition_number
  num_nonzero_svs = min_dim - n_zero_svs
  if geometric_spectrum:
    nonzero_svs = np.geomspace(min_nonzero_sv, max_sv, num=num_nonzero_svs,
                               dtype=np.float64)
  else:
    nonzero_svs = np.linspace(min_nonzero_sv, max_sv, num=num_nonzero_svs,
                              dtype=np.float64)
  half_point = n_zero_svs // 2
  for i in range(half_point, half_point + degeneracy):
    nonzero_svs[i] = nonzero_svs[half_point]
  svs = np.zeros(min(shape), dtype=np.float64)
  svs[n_zero_svs:] = nonzero_svs
  svs = svs[::-1]

  result = np.dot(left_vecs * svs, right_vecs.conj().T)
  result = jnp.array(result).astype(dtype)
  spectrum = jnp.array(svs).astype(dtype)
  return result, spectrum

OpRecord = collections.namedtuple(
    "OpRecord",
    ["name", "nargs", "dtypes", "rng_factory", "test_autodiff", "nondiff_argnums", "test_name"])


def op_record(name, nargs, dtypes, rng_factory, test_grad, nondiff_argnums=(), test_name=None):
  test_name = test_name or name
  nondiff_argnums = tuple(sorted(set(nondiff_argnums)))
  return OpRecord(name, nargs, dtypes, rng_factory, test_grad, nondiff_argnums, test_name)

# TODO(phawkins): we should probably separate out the function domains used for
# autodiff tests from the function domains used for equivalence testing. For
# example, logit should closely match its scipy equivalent everywhere, but we
# don't expect numerical gradient tests to pass for inputs very close to 0.

JAX_SPECIAL_FUNCTION_RECORDS = [
    op_record("betaln", 2, float_dtypes, jtu.rand_positive, False),
    op_record("betainc", 3, float_dtypes, jtu.rand_positive, False),
    op_record("digamma", 1, float_dtypes, jtu.rand_positive, True),
    op_record("gammainc", 2, float_dtypes, jtu.rand_positive, True),
    op_record("gammaincc", 2, float_dtypes, jtu.rand_positive, True),
    op_record("erf", 1, float_dtypes, jtu.rand_small_positive, True),
    op_record("erfc", 1, float_dtypes, jtu.rand_small_positive, True),
    op_record("erfinv", 1, float_dtypes, jtu.rand_small_positive, True),
    op_record("expit", 1, float_dtypes, jtu.rand_small_positive, True),
    # TODO: gammaln has slightly high error.
    op_record("gammaln", 1, float_dtypes, jtu.rand_positive, False),
    op_record("i0", 1, float_dtypes, jtu.rand_default, True),
    op_record("i0e", 1, float_dtypes, jtu.rand_default, True),
    op_record("i1", 1, float_dtypes, jtu.rand_default, True),
    op_record("i1e", 1, float_dtypes, jtu.rand_default, True),
    op_record("logit", 1, float_dtypes, partial(jtu.rand_uniform, low=0.05,
                                                high=0.95), True),
    op_record("log_ndtr", 1, float_dtypes, jtu.rand_default, True),
    op_record("ndtri", 1, float_dtypes, partial(jtu.rand_uniform, low=0.05,
                                                high=0.95),
              True),
    op_record("ndtr", 1, float_dtypes, jtu.rand_default, True),
    # TODO(phawkins): gradient of entr yields NaNs.
    op_record("entr", 1, float_dtypes, jtu.rand_default, False),
    op_record("polygamma", 2, (int_dtypes, float_dtypes), jtu.rand_positive, True, (0,)),
    op_record("xlogy", 2, float_dtypes, jtu.rand_positive, True),
    op_record("xlog1py", 2, float_dtypes, jtu.rand_default, True),
    # TODO: enable gradient test for zeta by restricting the domain of
    # of inputs to some reasonable intervals
    op_record("zeta", 2, float_dtypes, jtu.rand_positive, False),
]


class LaxBackedScipyTests(jtu.JaxTestCase):
  """Tests for LAX-backed Scipy implementation."""

  def _GetArgsMaker(self, rng, shapes, dtypes):
    return lambda: [rng(shape, dtype) for shape, dtype in zip(shapes, dtypes)]


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shapes={}_axis={}_keepdims={}_return_sign={}_use_b_{}".format(
          jtu.format_shape_dtype_string(shapes, dtype),
          axis, keepdims, return_sign, use_b),
       # TODO(b/133842870): re-enable when exp(nan) returns NaN on CPU.
       "shapes": shapes, "dtype": dtype,
       "axis": axis, "keepdims": keepdims,
       "return_sign": return_sign, "use_b": use_b}
      for shape_group in compatible_shapes for dtype in float_dtypes + complex_dtypes + int_dtypes
      for use_b in [False, True]
      for shapes in itertools.product(*(
        (shape_group, shape_group) if use_b else (shape_group,)))
      for axis in range(-max(len(shape) for shape in shapes),
                         max(len(shape) for shape in shapes))
      for keepdims in [False, True]
      for return_sign in [False, True]))
  @jtu.ignore_warning(category=RuntimeWarning,
                      message="invalid value encountered in .*")
  def testLogSumExp(self, shapes, dtype, axis,
                    keepdims, return_sign, use_b):
    if jtu.device_under_test() != "cpu":
      rng = jtu.rand_some_inf_and_nan(self.rng())
    else:
      rng = jtu.rand_default(self.rng())
    # TODO(mattjj): test autodiff
    if use_b:
      def scipy_fun(array_to_reduce, scale_array):
        return osp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                     return_sign=return_sign, b=scale_array)

      def lax_fun(array_to_reduce, scale_array):
        return lsp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                     return_sign=return_sign, b=scale_array)

      args_maker = lambda: [rng(shapes[0], dtype), rng(shapes[1], dtype)]
    else:
      def scipy_fun(array_to_reduce):
        return osp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                     return_sign=return_sign)

      def lax_fun(array_to_reduce):
        return lsp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                     return_sign=return_sign)

      args_maker = lambda: [rng(shapes[0], dtype)]
    tol = {np.float32: 1E-6, np.float64: 1E-14}
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker)
    self._CompileAndCheck(lax_fun, args_maker, rtol=tol, atol=tol)

  def testLogSumExpZeros(self):
    # Regression test for https://github.com/google/jax/issues/5370
    scipy_fun = lambda a, b: osp_special.logsumexp(a, b=b)
    lax_fun = lambda a, b: lsp_special.logsumexp(a, b=b)
    args_maker = lambda: [np.array([-1000, -2]), np.array([1, 0])]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker)
    self._CompileAndCheck(lax_fun, args_maker)

  @parameterized.named_parameters(itertools.chain.from_iterable(
    jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.test_name, shapes, dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes,
         "test_autodiff": rec.test_autodiff,
         "nondiff_argnums": rec.nondiff_argnums,
         "scipy_op": getattr(osp_special, rec.name),
         "lax_op": getattr(lsp_special, rec.name)}
        for shapes in itertools.combinations_with_replacement(all_shapes, rec.nargs)
        for dtypes in (itertools.combinations_with_replacement(rec.dtypes, rec.nargs)
          if isinstance(rec.dtypes, list) else itertools.product(*rec.dtypes)))
      for rec in JAX_SPECIAL_FUNCTION_RECORDS))
  def testScipySpecialFun(self, scipy_op, lax_op, rng_factory, shapes, dtypes,
                          test_autodiff, nondiff_argnums):
    if (jtu.device_under_test() == "cpu" and
        (lax_op is lsp_special.gammainc or lax_op is lsp_special.gammaincc)):
      # TODO(b/173608403): re-enable test when LLVM bug is fixed.
      raise unittest.SkipTest("Skipping test due to LLVM lowering bug")
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    args = args_maker()
    self.assertAllClose(scipy_op(*args), lax_op(*args), atol=1e-3, rtol=1e-3,
                        check_dtypes=False)
    self._CompileAndCheck(lax_op, args_maker, rtol=1e-4)

    if test_autodiff:
      def partial_lax_op(*vals):
        list_args = list(vals)
        for i in nondiff_argnums:
          list_args.insert(i, args[i])
        return lax_op(*list_args)

      assert list(nondiff_argnums) == sorted(set(nondiff_argnums))
      diff_args = [x for i, x in enumerate(args) if i not in nondiff_argnums]
      jtu.check_grads(partial_lax_op, diff_args, order=1,
                      atol=jtu.if_device_under_test("tpu", .1, 1e-3),
                      rtol=.1, eps=1e-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_d={}".format(
          jtu.format_shape_dtype_string(shape, dtype), d),
       "shape": shape, "dtype": dtype, "d": d}
      for shape in all_shapes
      for dtype in float_dtypes
      for d in [1, 2, 5]))
  def testMultigammaln(self, shape, dtype, d):
    def scipy_fun(a):
      return osp_special.multigammaln(a, d)

    def lax_fun(a):
      return lsp_special.multigammaln(a, d)

    rng = jtu.rand_positive(self.rng())
    args_maker = lambda: [rng(shape, dtype) + (d - 1) / 2.]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker,
                            tol={np.float32: 1e-3, np.float64: 1e-14})
    self._CompileAndCheck(lax_fun, args_maker)

  def testIssue980(self):
    x = np.full((4,), -1e20, dtype=np.float32)
    self.assertAllClose(np.zeros((4,), dtype=np.float32),
                        lsp_special.expit(x))

  def testIssue3758(self):
    x = np.array([1e5, 1e19, 1e10], dtype=np.float32)
    q = np.array([1., 40., 30.], dtype=np.float32)
    self.assertAllClose(np.array([1., 0., 0.], dtype=np.float32), lsp_special.zeta(x, q))

  def testXlogyShouldReturnZero(self):
    self.assertAllClose(lsp_special.xlogy(0., 0.), 0., check_dtypes=False)

  def testGradOfXlogyAtZero(self):
    partial_xlogy = functools.partial(lsp_special.xlogy, 0.)
    self.assertAllClose(api.grad(partial_xlogy)(0.), 0., check_dtypes=False)

  def testXlog1pyShouldReturnZero(self):
    self.assertAllClose(lsp_special.xlog1py(0., -1.), 0., check_dtypes=False)

  def testGradOfXlog1pyAtZero(self):
    partial_xlog1py = functools.partial(lsp_special.xlog1py, 0.)
    self.assertAllClose(api.grad(partial_xlog1py)(-1.), 0., check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_maxdegree={}_inputsize={}".format(l_max, num_z),
       "l_max": l_max,
       "num_z": num_z}
       for l_max, num_z in zip([1, 2, 3], [6, 7, 8])))
  def testLpmn(self, l_max, num_z):
    # Points on which the associated Legendre functions areevaluated.
    z = np.linspace(-0.2, 0.9, num_z)
    actual_p_vals, actual_p_derivatives = lsp_special.lpmn(m=l_max, n=l_max, z=z)

    # The expected results are obtained from scipy.
    expected_p_vals = np.zeros((l_max + 1, l_max + 1, num_z))
    expected_p_derivatives = np.zeros((l_max + 1, l_max + 1, num_z))

    for i in range(num_z):
      val, derivative = osp_special.lpmn(l_max, l_max, z[i])
      expected_p_vals[:, :, i] = val
      expected_p_derivatives[:, :, i] = derivative

    with self.subTest('Test values.'):
      self.assertAllClose(actual_p_vals, expected_p_vals, rtol=1e-6, atol=3.2e-6)

    with self.subTest('Test derivatives.'):
      self.assertAllClose(actual_p_derivatives,expected_p_derivatives,
              rtol=1e-6, atol=8.4e-4)

    with self.subTest('Test JIT compatibility'):
      args_maker = lambda: [z]
      lsp_special_fn = lambda z: lsp_special.lpmn(l_max, l_max, z)
      self._CompileAndCheck(lsp_special_fn, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_maxdegree={}_inputsize={}".format(l_max, num_z),
       "l_max": l_max,
       "num_z": num_z}
       for l_max, num_z in zip([3, 4, 6, 32], [2, 3, 4, 64])))
  def testNormalizedLpmnValues(self, l_max, num_z):
    # Points on which the associated Legendre functions areevaluated.
    z = np.linspace(-0.2, 0.9, num_z)
    is_normalized = True
    actual_p_vals = lsp_special.lpmn_values(l_max, l_max, z, is_normalized)

    # The expected results are obtained from scipy.
    expected_p_vals = np.zeros((l_max + 1, l_max + 1, num_z))
    for i in range(num_z):
      expected_p_vals[:, :, i] = osp_special.lpmn(l_max, l_max, z[i])[0]

    def apply_normalization(a):
      """Applies normalization to the associated Legendre functions."""
      num_m, num_l, _ = a.shape
      a_normalized = np.zeros_like(a)
      for m in range(num_m):
        for l in range(num_l):
          c0 = (2.0 * l + 1.0) * osp_special.factorial(l - m)
          c1 = (4.0 * np.pi) * osp_special.factorial(l + m)
          c2 = np.sqrt(c0 / c1)
          a_normalized[m, l] = c2 * a[m, l]
      return a_normalized

    # The results from scipy are not normalized and the comparison requires
    # normalizing the results.
    expected_p_vals_normalized = apply_normalization(expected_p_vals)

    with self.subTest('Test accuracy.'):
      self.assertAllClose(actual_p_vals, expected_p_vals_normalized, rtol=1e-6, atol=3.2e-6)

    with self.subTest('Test JIT compatibility'):
      args_maker = lambda: [z]
      lsp_special_fn = lambda z: lsp_special.lpmn_values(l_max, l_max, z, is_normalized)
      self._CompileAndCheck(lsp_special_fn, args_maker)

  def testSphHarmAccuracy(self):
    m = jnp.arange(-3, 3)[:, None]
    n = jnp.arange(3, 6)
    n_max = 5
    theta = 0.0
    phi = jnp.pi

    expected = lsp_special.sph_harm(m, n, theta, phi, n_max)

    actual = osp_special.sph_harm(m, n, theta, phi)

    self.assertAllClose(actual, expected, rtol=1e-8, atol=9e-5)

  def testSphHarmOrderZeroDegreeZero(self):
    """Tests the spherical harmonics of order zero and degree zero."""
    theta = jnp.array([0.3])
    phi = jnp.array([2.3])
    n_max = 0

    expected = jnp.array([1.0 / jnp.sqrt(4.0 * np.pi)])
    actual = jnp.real(
        lsp_special.sph_harm(jnp.array([0]), jnp.array([0]), theta, phi, n_max))

    self.assertAllClose(actual, expected, rtol=1.1e-7, atol=3e-8)

  def testSphHarmOrderZeroDegreeOne(self):
    """Tests the spherical harmonics of order one and degree zero."""
    theta = jnp.array([2.0])
    phi = jnp.array([3.1])
    n_max = 1

    expected = jnp.sqrt(3.0 / (4.0 * np.pi)) * jnp.cos(phi)
    actual = jnp.real(
        lsp_special.sph_harm(jnp.array([0]), jnp.array([1]), theta, phi, n_max))

    self.assertAllClose(actual, expected, rtol=7e-8, atol=1.5e-8)

  def testSphHarmOrderOneDegreeOne(self):
    """Tests the spherical harmonics of order one and degree one."""
    theta = jnp.array([2.0])
    phi = jnp.array([2.5])
    n_max = 1

    expected = (-1.0 / 2.0 * jnp.sqrt(3.0 / (2.0 * np.pi)) *
                jnp.sin(phi) * jnp.exp(1j * theta))
    actual = lsp_special.sph_harm(
        jnp.array([1]), jnp.array([1]), theta, phi, n_max)

    self.assertAllClose(actual, expected, rtol=1e-8, atol=6e-8)

  @parameterized.named_parameters(jtu.cases_from_list(
      {'testcase_name': '_maxdegree={}_inputsize={}_dtype={}'.format(
        l_max, num_z, dtype),
       'l_max': l_max, 'num_z': num_z, 'dtype': dtype}
      for l_max, num_z in zip([1, 3, 8, 10], [2, 6, 7, 8])
      for dtype in jtu.dtypes.all_integer))
  def testSphHarmForJitAndAgainstNumpy(self, l_max, num_z, dtype):
    """Tests against JIT compatibility and Numpy."""
    n_max = l_max
    shape = (num_z,)
    rng = jtu.rand_int(self.rng(), -l_max, l_max + 1)

    lsp_special_fn = partial(lsp_special.sph_harm, n_max=n_max)

    def args_maker():
      m = rng(shape, dtype)
      n = abs(m)
      theta = jnp.linspace(-4.0, 5.0, num_z)
      phi = jnp.linspace(-2.0, 1.0, num_z)
      return m, n, theta, phi

    with self.subTest('Test JIT compatibility'):
      self._CompileAndCheck(lsp_special_fn, args_maker)

    with self.subTest('Test against numpy.'):
      self._CheckAgainstNumpy(osp_special.sph_harm, lsp_special_fn, args_maker)

  def testSphHarmCornerCaseWithWrongNmax(self):
    """Tests the corner case where `n_max` is not the maximum value of `n`."""
    m = jnp.array([2])
    n = jnp.array([10])
    n_clipped = jnp.array([6])
    n_max = 6
    theta = jnp.array([0.9])
    phi = jnp.array([0.2])

    expected = lsp_special.sph_harm(m, n, theta, phi, n_max)

    actual = lsp_special.sph_harm(m, n_clipped, theta, phi, n_max)

    self.assertAllClose(actual, expected, rtol=1e-8, atol=9e-5)

  @parameterized.named_parameters(jtu.cases_from_list(
      {'testcase_name':
        '_shape={}'
        '_n_zero_sv={}_degeneracy={}_geometric_spectrum={}'
        '_max_sv={}_method={}_side={}'
        '_nonzero_condition_number={}_seed={}'.format(
          jtu.format_shape_dtype_string(
            shape, jnp.dtype(dtype).name).replace(" ", ""),
          n_zero_sv, degeneracy, geometric_spectrum, max_sv,
          method, side, nonzero_condition_number, seed
        ),
        'n_zero_sv': n_zero_sv, 'degeneracy': degeneracy,
        'geometric_spectrum': geometric_spectrum,
        'max_sv': max_sv, 'shape': shape, 'method': method,
        'side': side, 'nonzero_condition_number': nonzero_condition_number,
        'dtype': dtype, 'seed': seed}
      for n_zero_sv in n_zero_svs
      for degeneracy in degeneracies
      for geometric_spectrum in geometric_spectra
      for max_sv in max_svs
      for shape in polar_shapes
      for method in methods
      for side in sides
      for nonzero_condition_number in nonzero_condition_numbers
      for dtype in jtu.dtypes.floating
      for seed in seeds))
  def testPolar(
    self, n_zero_sv, degeneracy, geometric_spectrum, max_sv, shape, method,
      side, nonzero_condition_number, dtype, seed):
    """ Tests jax.scipy.linalg.polar."""
    if jtu.device_under_test() != "cpu":
      if jnp.dtype(dtype).name in ("bfloat16", "float16"):
        raise unittest.SkipTest("Skip half precision off CPU.")
      if method == "svd":
        raise unittest.SkipTest("Can't use SVD mode on TPU/GPU.")

    np.random.seed(seed)
    matrix, _ = _initialize_polar_test(
      shape, n_zero_sv, degeneracy, geometric_spectrum, max_sv,
      nonzero_condition_number, dtype)
    if jnp.dtype(dtype).name in ("bfloat16", "float16"):
      self.assertRaises(
        NotImplementedError, jsp.linalg.polar, matrix, method=method,
        side=side)
      return

    unitary, posdef, info = jsp.linalg.polar(matrix, method=method, side=side)
    if shape[0] >= shape[1]:
      should_be_eye = np.matmul(unitary.conj().T, unitary)
    else:
      should_be_eye = np.matmul(unitary, unitary.conj().T)
    tol = 10 * jnp.finfo(matrix.dtype).eps
    eye_mat = np.eye(should_be_eye.shape[0], dtype=should_be_eye.dtype)
    with self.subTest('Test unitarity.'):
      self.assertAllClose(
        eye_mat, should_be_eye, atol=tol * min(shape))

    with self.subTest('Test Hermiticity.'):
      self.assertAllClose(
        posdef, posdef.conj().T, atol=tol * jnp.linalg.norm(posdef))

    ev, _ = np.linalg.eigh(posdef)
    ev = ev[np.abs(ev) > tol * np.linalg.norm(posdef)]
    negative_ev = jnp.sum(ev < 0.)
    with self.subTest('Test positive definiteness.'):
      assert negative_ev == 0.

    if side == "right":
      recon = jnp.matmul(unitary, posdef, precision=lax.Precision.HIGHEST)
    elif side == "left":
      recon = jnp.matmul(posdef, unitary, precision=lax.Precision.HIGHEST)
    with self.subTest('Test reconstruction.'):
      self.assertAllClose(
        matrix, recon, atol=tol * jnp.linalg.norm(matrix))

  @parameterized.named_parameters(jtu.cases_from_list(
    {'testcase_name':
      '_linear_size_={}_seed={}_dtype={}'.format(
        linear_size, seed, jnp.dtype(dtype).name
      ),
      'linear_size': linear_size, 'seed': seed, 'dtype': dtype}
    for linear_size in linear_sizes
    for seed in seeds
    for dtype in jtu.dtypes.floating))
  def test_spectral_dac_eigh(self, linear_size, seed, dtype):
    if jtu.device_under_test != "cpu":
      raise unittest.SkipTest("Skip eigh off CPU for now.")
    if jnp.dtype(dtype).name in ("bfloat16", "float16"):
      if jtu.device_under_test() != "cpu":
        raise unittest.SkipTest("Skip half precision off CPU.")

    np.random.seed(seed)
    H = np.random.randn(linear_size, linear_size)
    H = jnp.array(0.5 * (H + H.conj().T)).astype(dtype)
    if jnp.dtype(dtype).name in ("bfloat16", "float16"):
      self.assertRaises(
        NotImplementedError, jax._src.scipy.eigh.eigh, H)
      return
    evs, V = jax._src.scipy.eigh.eigh(H)
    ev_exp, eV_exp = jnp.linalg.eigh(H)
    HV = jnp.dot(H, V, precision=lax.Precision.HIGHEST)
    vV = evs * V
    eps = jnp.finfo(H.dtype).eps
    atol = jnp.linalg.norm(H) * eps
    self.assertAllClose(ev_exp, jnp.sort(evs), atol=20 * atol)
    self.assertAllClose(HV, vV, atol=30 * atol)

  @parameterized.named_parameters(jtu.cases_from_list(
    {'testcase_name':
      '_linear_size_={}_seed={}_dtype={}'.format(
        linear_size, seed, jnp.dtype(dtype).name
      ),
      'linear_size': linear_size, 'seed': seed, 'dtype': dtype}
    for linear_size in linear_sizes
    for seed in seeds
    for dtype in jtu.dtypes.floating))
  def test_spectral_dac_svd(self, linear_size, seed, dtype):
    if jnp.dtype(dtype).name in ("bfloat16", "float16"):
      if jtu.device_under_test() != "cpu":
        raise unittest.SkipTest("Skip half precision off CPU.")

    np.random.seed(seed)
    A = np.random.randn(linear_size, linear_size).astype(dtype)
    if jnp.dtype(dtype).name in ("bfloat16", "float16"):
      self.assertRaises(
        NotImplementedError, jax._src.scipy.eigh.svd, A)
      return
    S_expected = np.linalg.svd(A, compute_uv=False)
    U, S, V = jax._src.scipy.eigh.svd(A)
    recon = jnp.dot((U * S), V, precision=lax.Precision.HIGHEST)
    eps = jnp.finfo(dtype).eps
    eps = eps * jnp.linalg.norm(A) * 10
    self.assertAllClose(np.sort(S), np.sort(S_expected), atol=eps)
    self.assertAllClose(A, recon, atol=eps)

    # U is unitary.
    u_unitary_delta = jnp.dot(U.conj().T, U, precision=lax.Precision.HIGHEST)
    u_eye = jnp.eye(u_unitary_delta.shape[0], dtype=dtype)
    self.assertAllClose(u_unitary_delta, u_eye, atol=eps)

    # V is unitary.
    v_unitary_delta = jnp.dot(V.conj().T, V, precision=lax.Precision.HIGHEST)
    v_eye = jnp.eye(v_unitary_delta.shape[0], dtype=dtype)
    self.assertAllClose(v_unitary_delta, v_eye, atol=eps)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
