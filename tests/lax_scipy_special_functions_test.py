# Copyright 2018 The JAX Authors.
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
import itertools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import scipy
import scipy.special as osp_special

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax._src import test_util as jtu
from jax.scipy import special as lsp_special

jax.config.parse_flags_with_absl()


all_shapes = [(), (4,), (3, 4), (3, 1), (1, 4), (2, 1, 4)]

OpRecord = collections.namedtuple(
    "OpRecord",
    ["name", "nargs", "dtypes", "rng_factory", "test_autodiff", "nondiff_argnums", "test_name"])


def op_record(name, nargs, dtypes, rng_factory, test_grad, nondiff_argnums=(), test_name=None):
  test_name = test_name or name
  nondiff_argnums = tuple(sorted(set(nondiff_argnums)))
  return OpRecord(name, nargs, dtypes, rng_factory, test_grad, nondiff_argnums, test_name)


float_dtypes = jtu.dtypes.floating
int_dtypes = jtu.dtypes.integer

# TODO(phawkins): we should probably separate out the function domains used for
# autodiff tests from the function domains used for equivalence testing. For
# example, logit should closely match its scipy equivalent everywhere, but we
# don't expect numerical gradient tests to pass for inputs very close to 0.

JAX_SPECIAL_FUNCTION_RECORDS = [
    op_record(
        "beta", 2, float_dtypes, jtu.rand_default, False
    ),
    op_record(
        "betaln", 2, float_dtypes, jtu.rand_default, False
    ),
    op_record(
        "betainc", 3, float_dtypes, jtu.rand_positive, False
    ),
    op_record(
        "gamma", 1, float_dtypes, jtu.rand_default, True
    ),
    op_record(
        "digamma", 1, float_dtypes, jtu.rand_positive, True
    ),
    op_record(
        "gammainc", 2, float_dtypes, jtu.rand_positive, True
    ),
    op_record(
        "gammaincc", 2, float_dtypes, jtu.rand_positive, True
    ),
    op_record(
        "gammasgn", 1, float_dtypes, jtu.rand_default, True
    ),
    op_record(
        "erf", 1, float_dtypes, jtu.rand_small_positive, True
    ),
    op_record(
        "erfc", 1, float_dtypes, jtu.rand_small_positive, True
    ),
    op_record(
        "erfinv", 1, float_dtypes, jtu.rand_small_positive, True
    ),
    op_record(
        "expit", 1, float_dtypes, jtu.rand_small_positive, True
    ),
    op_record(
        "sici", 1, float_dtypes, jtu.rand_default, True
    ),
    # TODO: gammaln has slightly high error.
    op_record(
        "gammaln", 1, float_dtypes, jtu.rand_positive, False
    ),
    op_record(
        "factorial", 1, float_dtypes, jtu.rand_default, True
    ),
    op_record(
        "fresnel", 1, float_dtypes,
        functools.partial(jtu.rand_default, scale=30), True
    ),
    op_record(
        "i0", 1, float_dtypes, jtu.rand_default, True
    ),
    op_record(
        # Note: values near zero can fail numeric gradient tests.
        "i0e", 1, float_dtypes,
        functools.partial(jtu.rand_not_small, offset=0.1), True
    ),
    op_record(
        "i1", 1, float_dtypes, jtu.rand_default, True
    ),
    op_record(
        "i1e", 1, float_dtypes, jtu.rand_default, True
    ),
    op_record(
        "logit", 1, float_dtypes,
        functools.partial(jtu.rand_uniform, low=0.05, high=0.95), True),
    op_record(
        "log_ndtr", 1, float_dtypes, jtu.rand_default, True
    ),
    op_record(
        "ndtri", 1, float_dtypes,
        functools.partial(jtu.rand_uniform, low=0.0, high=1.0), True,
    ),
    op_record(
        "ndtr", 1, float_dtypes, jtu.rand_default, True
    ),
    # TODO(phawkins): gradient of entr yields NaNs.
    op_record(
        "entr", 1, float_dtypes, jtu.rand_default, False
    ),
    op_record(
        "polygamma", 2, (int_dtypes, float_dtypes),
        jtu.rand_positive, True, (0,)),
    op_record(
        "xlogy", 2, float_dtypes, jtu.rand_positive, True
    ),
    op_record(
        "xlog1py", 2, float_dtypes, jtu.rand_default, True
    ),
    op_record("zeta", 2, float_dtypes, jtu.rand_positive, True),
    # TODO: float64 produces aborts on gpu, potentially related to use of jnp.piecewise
    op_record(
        "expi", 1, [np.float32],
        functools.partial(jtu.rand_not_small, offset=0.1), True),
    op_record("exp1", 1, [np.float32], jtu.rand_positive, True),
    op_record(
        "expn", 2, (int_dtypes, [np.float32]), jtu.rand_positive, True, (0,)),
    op_record("kl_div", 2, float_dtypes, jtu.rand_positive, True),
    op_record(
        "rel_entr", 2, float_dtypes, jtu.rand_positive, True,
    ),
    op_record("poch", 2, float_dtypes, jtu.rand_positive, True),
    op_record(
        "hyp1f1", 3, float_dtypes,
        functools.partial(jtu.rand_uniform, low=0.5, high=30), True
    ),
    op_record(
        "hyp2f1", 4, float_dtypes,
        functools.partial(jtu.rand_uniform, low=0.1, high=0.9), True
    ),
    op_record("log_softmax", 1, float_dtypes, jtu.rand_default, True),
    op_record("softmax", 1, float_dtypes, jtu.rand_default, True),
]


def _pretty_special_fun_name(case):
  shapes_str = "_".join("x".join(map(str, shape)) if shape else "s"
                        for shape in case["shapes"])
  dtypes_str = "_".join(np.dtype(d).name for d in case["dtypes"])
  name = f"_{case['op']}_{shapes_str}_{dtypes_str}"
  return dict(**case, testcase_name=name)


class LaxScipySpecialFunctionsTest(jtu.JaxTestCase):

  def _GetArgsMaker(self, rng, shapes, dtypes):
    return lambda: [rng(shape, dtype) for shape, dtype in zip(shapes, dtypes)]

  @parameterized.named_parameters(itertools.chain.from_iterable(
    map(_pretty_special_fun_name, jtu.sample_product_testcases(
      [dict(op=rec.name, rng_factory=rec.rng_factory,
            test_autodiff=rec.test_autodiff,
            nondiff_argnums=rec.nondiff_argnums)],
      shapes=itertools.combinations_with_replacement(all_shapes, rec.nargs),
      dtypes=(itertools.combinations_with_replacement(rec.dtypes, rec.nargs)
        if isinstance(rec.dtypes, list) else itertools.product(*rec.dtypes)),
    ))
    for rec in JAX_SPECIAL_FUNCTION_RECORDS
  ))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises dtype promotion
  def testScipySpecialFun(self, op, rng_factory, shapes, dtypes,
                          test_autodiff, nondiff_argnums):
    scipy_op = getattr(osp_special, op)
    lax_op = getattr(lsp_special, op)
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
                      atol=.1 if jtu.test_device_matches(["tpu"]) else 1e-3,
                      rtol=.1, eps=1e-3)

  @jtu.sample_product(
      n=[0, 1, 2, 3, 10, 50]
  )
  def testScipySpecialFunBernoulli(self, n):
    dtype = jnp.zeros(0).dtype  # default float dtype.
    scipy_op = lambda: osp_special.bernoulli(n).astype(dtype)
    lax_op = functools.partial(lsp_special.bernoulli, n)
    args_maker = lambda: []
    self._CheckAgainstNumpy(scipy_op, lax_op, args_maker, atol=0, rtol=1E-5)
    self._CompileAndCheck(lax_op, args_maker, atol=0, rtol=1E-5)

  def testGammaSign(self):
    dtype = jnp.zeros(0).dtype  # default float dtype.
    typ = dtype.type
    testcases = [
      (np.arange(-10, 0).astype(dtype), np.array([np.nan] * 10, dtype=dtype)),
      (np.nextafter(np.arange(-5, 0).astype(dtype), typ(-np.inf)),
       np.array([1, -1, 1, -1, 1], dtype=dtype)),
      (np.nextafter(np.arange(-5, 0).astype(dtype), typ(np.inf)),
       np.array([-1, 1, -1, 1, -1], dtype=dtype)),
      (np.arange(0, 10).astype(dtype), np.ones((10,), dtype)),
      (np.nextafter(np.arange(0, 10).astype(dtype), typ(np.inf)),
       np.ones((10,), dtype)),
      (np.nextafter(np.arange(1, 10).astype(dtype), typ(-np.inf)),
       np.ones((9,), dtype)),
      (np.array([-np.inf, -0.0, 0.0, np.inf, np.nan]),
       np.array([np.nan, -1.0, 1.0, 1.0, np.nan]))
    ]
    for inp, out in testcases:
      self.assertArraysEqual(out, lsp_special.gammasgn(inp))
      self.assertArraysEqual(out, jnp.sign(lsp_special.gamma(inp)))
      if jtu.parse_version(scipy.__version__) >= (1, 15):
        self.assertArraysEqual(out, osp_special.gammasgn(inp))
        self.assertAllClose(osp_special.gammasgn(inp),
                            lsp_special.gammasgn(inp))

  def testNdtriExtremeValues(self):
    # Testing at the extreme values (bounds (0. and 1.) and outside the bounds).
    dtype = jnp.zeros(0).dtype  # default float dtype.
    args_maker = lambda: [np.arange(-10, 10).astype(dtype)]
    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    self._CheckAgainstNumpy(osp_special.ndtri, lsp_special.ndtri, args_maker, rtol=rtol)
    self._CompileAndCheck(lsp_special.ndtri, args_maker, rtol=rtol)

  @parameterized.parameters([True, False])
  def testNdtriDebugInfs(self, with_jit):
    # ref: https://github.com/jax-ml/jax/issues/29328
    f = jax.jit(lsp_special.ndtri) if with_jit else lsp_special.ndtri
    with jax.debug_infs(True):
      f(0.5)  # Doesn't crash
      with self.assertRaisesRegex(FloatingPointError, "invalid value \\(inf\\)"):
        f(1.0)
      with self.assertRaisesRegex(FloatingPointError, "invalid value \\(inf\\)"):
        f(0.0)

  def testRelEntrExtremeValues(self):
    # Testing at the extreme values (bounds (0. and 1.) and outside the bounds).
    dtype = jnp.zeros(0).dtype  # default float dtype.
    args_maker = lambda: [np.array([-2, -2, -2, -1, -1, -1, 0, 0, 0]).astype(dtype),
                          np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1]).astype(dtype)]
    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    self._CheckAgainstNumpy(osp_special.rel_entr, lsp_special.rel_entr, args_maker, rtol=rtol)
    self._CompileAndCheck(lsp_special.rel_entr, args_maker, rtol=rtol)

  def testBetaParameterDeprecation(self):
    with self.assertNoWarnings():
      lsp_special.beta(1, 1)
      lsp_special.beta(1, b=1)
      lsp_special.beta(a=1, b=1)
    with self.assertRaises(TypeError):
      lsp_special.beta(x=1, y=1)

  def testExpnTracerLeaks(self):
    # Regression test for https://github.com/jax-ml/jax/issues/26972
    with jax.checking_leaks():
      lsp_special.expi(jnp.ones(()))

  def testExpiDisableJit(self):
    # Regression test for https://github.com/jax-ml/jax/issues/27019
    x = jnp.array([-0.5])
    with jax.disable_jit(True):
      result_nojit = lsp_special.expi(x)
    with jax.disable_jit(False):
      result_jit = lsp_special.expi(x)
    self.assertAllClose(result_jit, result_nojit)

  def testGammaIncBoundaryValues(self):
    dtype = dtypes.default_float_dtype()
    nan = float('nan')
    inf = float('inf')
    if jtu.parse_version(scipy.__version__) >= (1, 16):
      a_samples = [0, 0, 0, 1, nan,   1, nan,   0,   1, 1, nan]
      x_samples = [0, 1, 2, 0,   1, nan, nan, inf, inf, -1, inf]
    else:
      # disable samples that contradict with scipy/scipy#22441
      a_samples = [0, 0, 0, 1, nan,   1, nan,   0,   1, 1]
      x_samples = [0, 1, 2, 0,   1, nan, nan, inf, inf, -1]

    args_maker = lambda: (np.array(a_samples, dtype=dtype), np.array(x_samples, dtype=dtype))

    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    self._CheckAgainstNumpy(lsp_special.gammainc, osp_special.gammainc, args_maker, rtol=rtol)
    self._CompileAndCheck(lsp_special.gammainc, args_maker, rtol=rtol)

  def testGammaIncCBoundaryValues(self):
    dtype = dtypes.default_float_dtype()
    nan = float('nan')
    inf = float('inf')
    if jtu.parse_version(scipy.__version__) >= (1, 16):
      a_samples = [0, 0, 0, 1, nan,   1, nan,   0,   1, 1, nan]
      x_samples = [0, 1, 2, 0,   1, nan, nan, inf, inf, -1, inf]
    else:
      # disable samples that contradict with scipy/scipy#22441
      a_samples = [0, 0, 0, 1, nan,   1, nan,   0,   1, 1]
      x_samples = [0, 1, 2, 0,   1, nan, nan, inf, inf, -1]

    args_maker = lambda: (np.array(a_samples, dtype=dtype), np.array(x_samples, dtype=dtype))

    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    self._CheckAgainstNumpy(lsp_special.gammaincc, osp_special.gammaincc, args_maker, rtol=rtol)
    self._CompileAndCheck(lsp_special.gammaincc, args_maker, rtol=rtol)

  def testBetaIncBoundaryValues(self):
    dtype = dtypes.default_float_dtype()
    fi = jax.numpy.finfo(dtype)
    nan = float('nan')
    inf = float('inf')
    tiny = fi.tiny
    eps = fi.eps
    if jtu.parse_version(scipy.__version__) >= (1, 16):
      # TODO(pearu): enable tiny samples when a fix to scipy/scipy#22682
      # will be available
      a_samples = [nan, -0.5, inf, 0, eps, 1, tiny][:-1]
      b_samples = [nan, -0.5, inf, 0, eps, 1, tiny][:-1]
    else:
      # disabled samples that contradict with scipy/scipy#22425
      a_samples = [nan, -0.5, 0.5]
      b_samples = [nan, -0.5, 0.5]
    x_samples = [nan, -0.5, 0, 0.5, 1, 1.5]

    a_samples = np.array(a_samples, dtype=dtype)
    b_samples = np.array(b_samples, dtype=dtype)
    x_samples = np.array(x_samples, dtype=dtype)

    args_maker = lambda: np.meshgrid(a_samples, b_samples, x_samples)

    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 5e-5
    self._CheckAgainstNumpy(osp_special.betainc, lsp_special.betainc, args_maker, rtol=rtol)
    self._CompileAndCheck(lsp_special.betainc, args_maker, rtol=rtol)

  def testHyp2f1SpecialCases(self):
    dtype = dtypes.default_float_dtype()

    a_samples = np.array([0, 1, 1, 1, 1, 5, 5, 0.245, 0.45, 0.45, 2, 0.4, 0.32, 4, 4], dtype=dtype)
    b_samples = np.array([1, 0, 1, 1, 1, 1, 1, 3, 0.7, 0.7, 1, 0.7, 0.76, 2, 3], dtype=dtype)
    c_samples = np.array([1, 1, 0, 1, -1, 3, 3, 3, 0.45, 0.45, 5, 0.3, 0.11, 7, 7], dtype=dtype)
    x_samples = np.array([1, 1, 1, 0, 1, 0.5, 1, 0.35, 0.35, 1.5, 1, 0.4, 0.95, 0.95, 0.95], dtype=dtype)

    args_maker = lambda: (a_samples, b_samples, c_samples, x_samples)
    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 5e-5
    self._CheckAgainstNumpy(osp_special.hyp2f1, lsp_special.hyp2f1, args_maker, rtol=rtol)
    self._CompileAndCheck(lsp_special.hyp2f1, args_maker, rtol=rtol)

  def testSiciEdgeCases(self):
    dtype = jnp.zeros(0).dtype
    x_samples = np.array([0.0, np.inf, -np.inf], dtype=dtype)
    scipy_op = lambda x: osp_special.sici(x)
    lax_op = lambda x: lsp_special.sici(x)
    si_scipy, ci_scipy = scipy_op(x_samples)
    si_jax, ci_jax = lax_op(x_samples)

    expected_si = np.array([0.0, np.pi/2, -np.pi/2], dtype=dtype)
    expected_ci = np.array([-np.inf, 0.0, np.nan], dtype=dtype)
    self.assertAllClose(si_jax, si_scipy, atol=1e-6, rtol=1e-6)
    self.assertAllClose(ci_jax, ci_scipy, atol=1e-6, rtol=1e-6)
    self.assertAllClose(si_jax, expected_si, atol=1e-6, rtol=1e-6)
    self.assertAllClose(ci_jax, expected_ci, atol=1e-6, rtol=1e-6)

  @jtu.sample_product(
    scale=[1, 10, 1e9],
    shape=[(5,), (10,)]
  )
  def testSiciValueRanges(self, scale, shape):
    rng = jtu.rand_default(self.rng(), scale=scale)
    args_maker = lambda: [rng(shape, jnp.float32)]
    rtol = 5e-3 if jtu.test_device_matches(["tpu"]) else 1e-6
    self._CheckAgainstNumpy(
        osp_special.sici, lsp_special.sici, args_maker, rtol=rtol)

  def testSiciRaiseOnComplexInput(self):
    samples = jnp.arange(5, dtype=complex)
    with self.assertRaisesRegex(ValueError, "Argument `x` to sici must be real-valued."):
      lsp_special.sici(samples)




class LambertwTest(jtu.JaxTestCase):
  """Tests for jax.scipy.special.lambertw."""

  @jtu.sample_product(dtype=[np.float32, np.float64])
  def testLambertwK0RealAgainstScipy(self, dtype):
    """SciPy parity for k=0 real inputs across dtypes."""
    with jax.enable_x64(dtype == np.float64):
      z = np.array([-0.3, -0.1, 0.0, 0.1, 1.0, 5.0, 100.0], dtype=dtype)
      expected = np.real(osp_special.lambertw(z.astype(np.float64), k=0)).astype(dtype)
      atol, rtol = (2e-6, 2e-6) if dtype == np.float32 else (1e-12, 1e-12)
      self.assertAllClose(lsp_special.lambertw(z), expected, atol=atol, rtol=rtol)

  def testLambertwK0ComplexAgainstScipy(self):
    """SciPy parity for k=0 complex inputs including broad random grid."""
    with jax.enable_x64():
      rng = np.random.RandomState(42)
      z = np.array(rng.uniform(-5, 5, size=200) + 1j * rng.uniform(-5, 5, size=200),
                    dtype=np.complex128)
      self.assertAllClose(lsp_special.lambertw(z),
                          osp_special.lambertw(z, k=0), atol=1e-6, rtol=1e-6)

  def testLambertwK0ComplexBranchCut(self):
    """Test k=0 on the negative real axis below -1/e where W is complex."""
    with jax.enable_x64():
      z = np.array([-1+0j, -0.5+0j, -0.4+0j, -1+1e-12j, -1-1e-12j,
                     -0.8094117647058823+1e-12j],
                    dtype=np.complex128)
      self.assertAllClose(lsp_special.lambertw(z),
                          osp_special.lambertw(z, k=0), atol=1e-8, rtol=1e-8)

  def testLambertwK0BranchPointAndSpecialValues(self):
    """Test branch point, poles, and special values for k=0."""
    with jax.enable_x64():
      # Branch point: W(-1/e) = -1, below returns NaN
      bp = -1.0 / np.e
      self.assertAllClose(lsp_special.lambertw(bp), -1.0, atol=1e-12)
      self.assertTrue(np.isnan(np.asarray(lsp_special.lambertw(bp - 1e-15))))
      # Special values
      self.assertAllClose(lsp_special.lambertw(0.0), 0.0, atol=1e-15)
      self.assertTrue(np.isposinf(np.asarray(lsp_special.lambertw(jnp.inf))))
      self.assertTrue(np.isnan(np.asarray(lsp_special.lambertw(jnp.nan))))

  def testLambertwComplexBranchPoint(self):
    """W(-1/e + 0j) should match SciPy (nan+nanj) for both k=0 and k=-1."""
    with jax.enable_x64():
      bp = np.complex128(-1.0 / np.e)
      for k in [0, -1]:
        scipy_result = np.asarray(osp_special.lambertw(bp, k=k))
        result = np.asarray(lsp_special.lambertw(bp, k=k))
        # SciPy returns nan+nanj at the branch point
        self.assertTrue(np.isnan(scipy_result.real) and np.isnan(result.real),
                        f"Real part mismatch at branch point for k={k}")
        self.assertTrue(np.isnan(scipy_result.imag) and np.isnan(result.imag),
                        f"Imag part mismatch at branch point for k={k}")

  def testLambertwComplexInfinity(self):
    """Test complex infinity handling for both branches."""
    with jax.enable_x64():
      z = np.array([np.inf+0j, -np.inf+0j, np.inf+1j, -np.inf+1j],
                    dtype=np.complex128)
      for k_val in [0, -1]:
        expected = osp_special.lambertw(z, k=k_val)
        actual = np.asarray(lsp_special.lambertw(z, k=k_val))
        self.assertAllClose(actual, expected, atol=1e-6, rtol=1e-6)

  def testLambertwKm1AgainstScipy(self):
    """SciPy parity for k=-1 on real and complex inputs."""
    with jax.enable_x64():
      # Real k=-1
      z_real = np.array([-0.3, -0.2, -0.1, -0.01, -1e-6], dtype=np.float64)
      self.assertAllClose(lsp_special.lambertw(z_real, k=-1),
                          np.real(osp_special.lambertw(z_real, k=-1)),
                          atol=1e-8, rtol=1e-8)
      # Complex k=-1
      z_cplx = np.array([-0.2+0.1j, 1+1j, 1+0j, 10+0j], dtype=np.complex128)
      self.assertAllClose(lsp_special.lambertw(z_cplx, k=-1),
                          osp_special.lambertw(z_cplx, k=-1), atol=1e-6, rtol=1e-6)
      # Branch point and zero
      self.assertAllClose(lsp_special.lambertw(-1.0/np.e, k=-1), -1.0, atol=1e-12)
      self.assertTrue(np.isneginf(np.asarray(lsp_special.lambertw(0.0, k=-1))))

  def testLambertwJitAndVmap(self):
    """Test JIT and vmap compatibility."""
    with jax.enable_x64():
      z = jnp.array([0.1, 1.0, 5.0], dtype=jnp.float64)
      eager = lsp_special.lambertw(z)
      self.assertAllClose(jax.jit(lsp_special.lambertw)(z), eager, atol=1e-12)
      self.assertAllClose(jax.vmap(lsp_special.lambertw)(z), eager, atol=1e-12)

  def testLambertwDefiningEquation(self):
    """Verify W(z) * exp(W(z)) == z for both branches."""
    with jax.enable_x64():
      z_k0 = jnp.array([-0.35, 0.1, 1.0, 10.0], dtype=jnp.float64)
      w_k0 = lsp_special.lambertw(z_k0)
      self.assertAllClose(w_k0 * jnp.exp(w_k0), z_k0, atol=1e-10, rtol=1e-10)
      z_km1 = jnp.array([-0.3, -0.1, -0.01], dtype=jnp.float64)
      w_km1 = lsp_special.lambertw(z_km1, k=-1)
      self.assertAllClose(w_km1 * jnp.exp(w_km1), z_km1, atol=1e-8, rtol=1e-8)

  def testLambertwGrad(self):
    """Test autodiff on safe inputs."""
    with jax.enable_x64():
      jtu.check_grads(lsp_special.lambertw,
                      (jnp.array(0.3, dtype=jnp.float64),),
                      order=1, atol=1e-6, rtol=1e-5, eps=1e-5)
      # dW/dz at z=0 is 1
      self.assertAllClose(jax.grad(lsp_special.lambertw)(0.0), 1.0, atol=1e-6)

  def testLambertwUnsupportedBranchRaises(self):
    with self.assertRaisesRegex(NotImplementedError, "k=0 and k=-1"):
      lsp_special.lambertw(1.0, k=1)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
