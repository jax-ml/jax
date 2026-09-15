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
import math

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
        "boxcox", 2, float_dtypes, jtu.rand_positive, True
    ),
    op_record(
        "boxcox1p", 2, float_dtypes,
        functools.partial(jtu.rand_uniform, low=-0.5, high=5.0), True
    ),
    op_record(
        "gamma", 1, float_dtypes, jtu.rand_default, True
    ),
    op_record(
        "gamma", 1, jtu.dtypes.complex, jtu.rand_default, False,
        test_name="gamma_complex"
    ),
    op_record(
        "loggamma", 1, float_dtypes, jtu.rand_positive, True
    ),
    op_record(
        "loggamma", 1, jtu.dtypes.complex, jtu.rand_default, False,
        test_name="loggamma_complex"
    ),
    op_record(
        "dawsn", 1, float_dtypes, jtu.rand_default, True
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
        "erfcx", 1, float_dtypes + jtu.dtypes.complex, jtu.rand_default, True
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
        "comb", 2, float_dtypes, jtu.rand_positive, False
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
    op_record("owens_t", 2, float_dtypes, jtu.rand_default, True),
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
    op_record("wofz", 1, jtu.dtypes.complex, jtu.rand_default, False),
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

  def testErfcxLargeX(self):
    # Verify no overflow and agreement with scipy in the asymptotic regime
    # (float32: x > ~9.4, float64: x > ~26.6 — where exp(x^2) would overflow naively)
    x = np.array([10., 20., 50., 100.], dtype=np.float32)
    jax_val = lsp_special.erfcx(x)
    scipy_val = osp_special.erfcx(x)
    self.assertAllClose(jax_val, scipy_val, rtol=1e-5)
    if jax.config.x64_enabled:
      x = np.array([27., 50., 100., 500.], dtype=np.float64)
      jax_val = lsp_special.erfcx(x)
      scipy_val = osp_special.erfcx(x)
      self.assertAllClose(jax_val, scipy_val, rtol=1e-12)

  def testWofzAccuracy(self):
    # Verify wofz agrees with scipy over the full complex plane (float32).
    rng = jtu.rand_default(np.random.RandomState(0))
    z = rng((50,), np.complex64)
    jax_val = np.array(lsp_special.wofz(z))
    scipy_val = osp_special.wofz(z.astype(np.complex128)).astype(np.complex64)
    self.assertAllClose(jax_val, scipy_val, rtol=1e-5)

  def testWofzLowerHalfPlane(self):
    # The reflection formula w(-z) = 2*exp(-z^2) - w(z) must hold.
    rng = jtu.rand_default(np.random.RandomState(1))
    z = rng((20,), np.complex64)
    z_lower = z.real - 1j * np.abs(z.imag) - 1j * 0.1  # ensure Im < 0
    jax_w = np.array(lsp_special.wofz(z_lower))
    scipy_w = osp_special.wofz(z_lower.astype(np.complex128)).astype(np.complex64)
    self.assertAllClose(jax_w, scipy_w, rtol=1e-5)

  def testWofzJvp(self):
    # d/dz w(z) = -2z*w(z) + 2i/sqrt(pi) — test against numerical diff.
    rng = jtu.rand_default(np.random.RandomState(2))
    z = rng((10,), np.complex64) + 0.5j  # stay in upper half-plane
    import jax
    primals, tangents = jax.jvp(lsp_special.wofz, (z,), (np.ones_like(z),))
    expected_tangents = -2 * z * primals + jnp.array(2j / np.sqrt(np.pi), dtype=z.dtype)
    self.assertAllClose(tangents, expected_tangents, rtol=1e-4)

  def testDawsnLargeX(self):
    # Verify correctness in the large-x rational regime (region 2: [3.25, 6.25)
    # and region 3: [6.25, inf)) and at the odd-symmetry boundaries.
    x = np.array([-10., -6.25, -3.25, 0., 3.25, 6.25, 10., 100.], dtype=float)
    self.assertAllClose(lsp_special.dawsn(x), osp_special.dawsn(x))

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

  @parameterized.parameters(
      ([-1, -1, 5, 5], [0, 2, -1, 7], False),
      ([0, 0], [0, 4], True),
  )
  def testCombBoundaryValues(self, N_samples, k_samples, repetition):
    if repetition and jtu.parse_version(scipy.__version__) < (1, 17):
      self.skipTest("comb with repetition=True boundary values require scipy 1.17 or newer")
    dtype = dtypes.default_float_dtype()
    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    args_maker = lambda: (np.array(N_samples, dtype=dtype), np.array(k_samples, dtype=dtype))
    scipy_op = functools.partial(osp_special.comb, repetition=repetition)
    lax_op = functools.partial(lsp_special.comb, repetition=repetition)
    self._CheckAgainstNumpy(scipy_op, lax_op, args_maker, rtol=rtol)
    self._CompileAndCheck(lax_op, args_maker, rtol=rtol)

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
      a_samples = [0, 0, 0, 1, nan, 1, nan, 0, 1, 1, nan, inf, inf, inf, inf, inf]
      x_samples = [0, 1, 2, 0, 1, nan, nan, inf, inf, -1, inf, 0, 1, inf, nan, -1]
    else:
      # disable samples that contradict with scipy/scipy#22441
      a_samples = [0, 0, 0, 1, nan, 1, nan, 0, 1, 1, inf, inf, inf, inf]
      x_samples = [0, 1, 2, 0, 1, nan, nan, inf, inf, -1, 0, 1, inf, -1]

    args_maker = lambda: (np.array(a_samples, dtype=dtype), np.array(x_samples, dtype=dtype))

    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    self._CheckAgainstNumpy(
        osp_special.gammainc, lsp_special.gammainc, args_maker, rtol=rtol
    )
    self._CompileAndCheck(lsp_special.gammainc, args_maker, rtol=rtol)

  def testGammaIncCBoundaryValues(self):
    dtype = dtypes.default_float_dtype()
    nan = float('nan')
    inf = float('inf')
    if jtu.parse_version(scipy.__version__) >= (1, 16):
      a_samples = [0, 0, 0, 1, nan, 1, nan, 0, 1, 1, nan, inf, inf, inf, inf, inf]
      x_samples = [0, 1, 2, 0, 1, nan, nan, inf, inf, -1, inf, 0, 1, inf, nan, -1]
    else:
      # disable samples that contradict with scipy/scipy#22441
      a_samples = [0, 0, 0, 1, nan, 1, nan, 0, 1, 1, inf, inf, inf, inf]
      x_samples = [0, 1, 2, 0, 1, nan, nan, inf, inf, -1, 0, 1, inf, -1]

    args_maker = lambda: (np.array(a_samples, dtype=dtype), np.array(x_samples, dtype=dtype))

    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    self._CheckAgainstNumpy(
        osp_special.gammaincc, lsp_special.gammaincc, args_maker, rtol=rtol
    )
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

  def testComplexGammaPoles(self):
    """Test that gamma returns nan+nanj at non-positive integer poles."""
    poles = jnp.array([0+0j, -1+0j, -2+0j, -5+0j])
    result = np.array(lsp_special.gamma(poles))
    # Both real and imaginary parts should be NaN
    self.assertTrue(np.all(np.isnan(result.real)))
    self.assertTrue(np.all(np.isnan(result.imag)))

  def testComplexGammaBranchCut(self):
    """Test gamma near the negative real axis and at the reflection boundary."""
    # Points near poles (approached from above/below) should match SciPy
    z = np.array([-0.5+0j, -1.5+0j, 0.5+1j, 0.5-1j, -2.5+1e-12j, -2.5-1e-12j])
    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    self.assertAllClose(lsp_special.gamma(z), osp_special.gamma(z),
                        atol=1e-5, rtol=rtol)

  @parameterized.parameters(range(1, 5))
  def test_i0_gradients(self, order):
    # Regression test for https://github.com/jax-ml/jax/issues/40627 & 40628
    def d_i0(x, n=order):
      """Closed form of the nth derivative of i0(x)"""
      return sum(math.comb(n, k) * osp_special.iv(abs(2 * k - n), x) for k in range(n + 1)) / (2 ** n)

    eps = dtypes.finfo(dtypes.default_float_dtype()).eps
    x = np.array([0.0, eps * 0.01, eps * 0.1, eps, eps * 10, eps * 100, 1.0])
    args_maker = lambda: [x]
    f = jax.scipy.special.i0
    for _ in range(order):
      f = jax.grad(f)
    self._CheckAgainstNumpy(jax.vmap(f), d_i0, args_maker, rtol=1e-5)
    self._CompileAndCheck(jax.vmap(f), args_maker, rtol=1e-5)

  @parameterized.parameters(range(1, 5))
  def test_i1_gradients(self, order):
    def d_i1(x, n=order):
      """Closed form of the nth derivative of i1(x)"""
      return sum(math.comb(n + 1, k) * osp_special.iv(abs(2 * k - n - 1), x) for k in range(n + 2)) / (2 ** (n + 1))

    eps = dtypes.finfo(dtypes.default_float_dtype()).eps
    x = np.array([0.0, eps * 0.01, eps * 0.1, eps, eps * 10, eps * 100, 1.0])
    args_maker = lambda: [x]
    f = jax.scipy.special.i1
    for _ in range(order):
      f = jax.grad(f)
    self._CheckAgainstNumpy(jax.vmap(f), d_i1, args_maker, rtol=1e-5)
    self._CompileAndCheck(jax.vmap(f), args_maker, rtol=1e-5)

  def test_gammaln_ulp_error(self):
    # Exact zeros at x = 1.0 and x = 2.0
    self.assertEqual(float(lsp_special.gammaln(np.float32(1.0))), 0.0)
    self.assertEqual(float(lsp_special.gammaln(np.float32(2.0))), 0.0)

    # Special values: poles, overflow threshold, infinities, NaN
    special_inf = np.array([
        0.0, -0.0, -1.0, -2.0, -3.0, -10.0, -1000.0, -1e7,
        4.0850034e36, np.inf, -np.inf
    ], dtype=np.float32)
    res_inf = np.asarray(lsp_special.gammaln(special_inf))
    self.assertTrue(np.all(np.isposinf(res_inf)), f"Expected +inf, got {res_inf}")
    self.assertTrue(np.isnan(float(lsp_special.gammaln(np.float32(np.nan)))))

    # Derivative check (JVP / grad)
    x_grad = np.array([0.25, 0.75, 1.5, 3.5, 12.0, -0.5, -1.5, -2.5], dtype=np.float32)
    self.assertAllClose(
        jax.vmap(jax.grad(lsp_special.gammaln))(x_grad),
        lsp_special.digamma(x_grad),
        rtol=1e-5, atol=1e-5)

    def max_ulp_vs_ref(xs_f32, ref_f64):
      ys_f32 = np.asarray(lsp_special.gammaln(xs_f32), dtype=np.float64)
      ref_as_f32 = ref_f64.astype(np.float32)
      ulp = np.abs(np.spacing(ref_as_f32).astype(np.float64))
      errs = np.abs(ys_f32 - ref_f64) / ulp
      idx = int(np.argmax(errs))
      return float(errs[idx]), float(xs_f32[idx])

    # 1. 60-digit mpmath certified points within +/- 2 ULP of all 15 negative roots
    CERTIFIED_ROOT_PTS = [(-2.457024335861206, 6.098183754018183e-07), (-2.457024574279785, 2.484696988890678e-07), (-2.4570248126983643, -1.1287842529645816e-07), (-2.4570250511169434, -4.742259971548743e-07), (-2.4570252895355225, -8.355730166862955e-07), (-3.143580436706543, 3.5146424031690423e-06), (-3.143580675125122, 1.659292425530781e-06), (-3.143580913543701, -1.9605461541614247e-07), (-3.1435811519622803, -2.051398719680857e-06), (-3.1435813903808594, -3.906739887272491e-06), (-2.7476820945739746, -1.0570121172887281e-06), (-2.7476823329925537, -6.005974428686868e-07), (-2.747682571411133, -1.4418167987778365e-07), (-2.747682809829712, 3.1223517168561536e-07), (-2.747683048248291, 7.686531118231443e-07), (-4.039361000061035, 2.249564626449368e-05), (-4.039361476898193, 9.720763558572296e-06), (-4.039361953735352, -3.053971690394376e-06), (-4.03936243057251, -1.5828559485961922e-05), (-4.039362907409668, -2.8602999831685795e-05), (-3.955293893814087, -8.104382793261057e-06), (-3.955294132232666, -3.1631758619440246e-06), (-3.955294370651245, 1.7780596858250852e-06), (-3.955294609069824, 6.7193238503496265e-06), (-3.9552948474884033, 1.1660616631932957e-05), (-5.0082173347473145, 0.00010283681295424625), (-5.008217811584473, 4.400893627975495e-05), (-5.008218288421631, -1.4815572806334704e-05), (-5.008218765258789, -7.363671469471415e-05), (-5.008219242095947, -0.00013245448977600686), (-4.991543769836426, -0.00010146515471314859), (-4.991544246673584, -4.590078166675967e-05), (-4.991544723510742, 9.666772148365217e-06), (-4.9915452003479, 6.523750709091674e-05), (-4.991545677185059, 0.00012081142351964627), (-6.001384258270264, 0.0007502033977111201), (-6.001384735107422, 0.0004049005350435363), (-6.00138521194458, 5.9716251821000246e-05), (-6.001385688781738, -0.00028534953358012286), (-6.0013861656188965, -0.0006302969026992184), (-5.9986066818237305, -0.000571583788528323), (-5.998607158660889, -0.0002301889621487998), (-5.998607635498047, 0.00011132306745354033), (-5.998608112335205, 0.0004529523805681395), (-5.998608589172363, 0.0007946990575669714), (-7.000197410583496, 0.0046656094182512976), (-7.000197887420654, 0.002252102177623758), (-7.0001983642578125, -0.0001555986874670934), (-7.000198841094971, -0.0025575210588133666), (-7.000199317932129, -0.004953692617858581), (-6.999800682067871, -0.004150185635306569), (-6.999801158905029, -0.00175593634176963), (-6.9998016357421875, 0.0006440637640683667), (-6.999802112579346, 0.0030498423636509397), (-6.999802589416504, 0.0054614273387702), (-8.000022888183594, 0.08023788354150395), (-8.00002384185791, 0.03941384761713908), (-8.000024795532227, 0.00019109306263327084), (-8.000025749206543, -0.037551276318553026), (-8.00002670288086, -0.0739209618848819), (-7.999974250793457, -0.03744103667945246), (-7.999974727630615, -0.01874992444328908), (-7.999975204467773, 0.0002972497521375302), (-7.999975681304932, 0.019714314834692555), (-7.99997615814209, 0.03951592135704702), (-9.000000953674316, 1.0611139836802737), (-9.000001907348633, 0.367964655686062), (-9.00000286102295, -0.037502599853472283), (-9.000003814697266, -0.32518681973372665), (-9.000004768371582, -0.5483325184735134), (-8.999995231628418, -0.5483110440874025), (-8.999996185302734, -0.3251696402248379), (-8.99999713897705, -0.03748971522180572), (-8.999998092651367, 0.3679732454405064), (-8.999999046325684, 1.061118278557496)]
    cert_xs = np.array([p[0] for p in CERTIFIED_ROOT_PTS], dtype=np.float32)
    cert_ref = np.array([p[1] for p in CERTIFIED_ROOT_PTS], dtype=np.float64)
    max_root_ulp, worst_root_x = max_ulp_vs_ref(cert_xs, cert_ref)
    self.assertLessEqual(
        max_root_ulp, 1.0,
        f"Max ULP {max_root_ulp} at negative root x={worst_root_x} exceeds 1.0")

    # 2. Sweep across positive and negative float32 domains vs float64 scipy
    sweep_xs = np.concatenate([
        np.logspace(-37, -1, 500, dtype=np.float32),
        np.linspace(0.1, 0.4999, 500, dtype=np.float32),
        np.linspace(0.5, 0.9999, 500, dtype=np.float32),
        np.linspace(1.0001, 1.9999, 1000, dtype=np.float32),
        np.linspace(2.0001, 4.0, 500, dtype=np.float32),
        np.linspace(4.001, 16.0, 500, dtype=np.float32),
        np.logspace(1.21, 35.0, 500, dtype=np.float32),
        np.array([
            np.nextafter(np.float32(1.0), np.float32(0.0)),
            np.nextafter(np.float32(1.0), np.float32(2.0)),
            np.nextafter(np.float32(2.0), np.float32(1.0)),
            np.nextafter(np.float32(2.0), np.float32(3.0)),
            np.float32(1.4616321),
            np.float32(10.92287),
            np.float32(4.085003e36),
            # Global worst-case inputs across all 3.32 billion float32 numbers:
            np.float32(4.012665),    # global positive max ULP (0.525179 ULP)
            np.float32(-3.1757019),  # global negative reflection max ULP (0.691768 ULP)
            np.float32(-2.5576105),  # global overall float32 max ULP (0.697693 ULP)
            np.float32(-2.5056348),  # (-2.6, -2.5) root window boundary test
        ], dtype=np.float32),
        np.linspace(-1.999, -1.001, 400, dtype=np.float32),
        np.linspace(-0.999, -0.001, 400, dtype=np.float32),
        np.linspace(-99.9, -10.1, 400, dtype=np.float32),
    ])
    sweep_xs = sweep_xs[sweep_xs != np.floor(sweep_xs)]
    sweep_ref = osp_special.gammaln(sweep_xs.astype(np.float64))
    max_sweep_ulp, worst_sweep_x = max_ulp_vs_ref(sweep_xs, sweep_ref)
    self.assertLessEqual(
        max_sweep_ulp, 1.0,
        f"Max ULP {max_sweep_ulp} at x={worst_sweep_x} exceeds 1.0")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
