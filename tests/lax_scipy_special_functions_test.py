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
from jax._src.lax import lax as lax_internal
from jax._src.lib import jaxlib_extension_version
from jax._src.scipy import special as special_internal
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
        "gammaincinv", 2, float_dtypes,
        functools.partial(jtu.rand_uniform, low=0.05, high=0.95), True
    ),
    op_record(
        "gammainccinv", 2, float_dtypes,
        functools.partial(jtu.rand_uniform, low=0.05, high=0.95), True
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
        "erfcinv", 1, float_dtypes,
        functools.partial(jtu.rand_uniform, low=0.05, high=1.95), True
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
    op_record("exp1", 1, float_dtypes, jtu.rand_positive, True),
    op_record(
        "expn", 2, (int_dtypes, float_dtypes), jtu.rand_positive, True, (0,)),
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

  @jtu.sample_product(dtype=float_dtypes)
  def testExp1BranchesTerminate(self, dtype):
    # jnp.piecewise evaluates expn's continued-fraction branch on x <= 1
    # inputs, where it does not converge; this must not hang (#13543).
    x = np.array([1e-8, 1e-6, 0.5, 1.0, np.nextafter(1.0, 2.0),
                  np.nextafter(1.0, 0.0), 2.0, 100.0], dtype=dtype)
    rtol = {np.float32: 1e-4, np.float64: 1e-5}[dtype]
    atol = {np.float32: 1e-5, np.float64: 1e-6}[dtype]
    self.assertAllClose(
        lsp_special.exp1(x),
        osp_special.exp1(x.astype(np.float64)).astype(dtype),
        rtol=rtol, atol=atol)

  @jtu.sample_product(dtype=float_dtypes, n=[0, 1, 2, 10])
  def testExpnBranchesTerminate(self, dtype, n):
    x = np.array([1e-8, 1e-6, 0.5, 1.0, np.nextafter(1.0, 2.0),
                  np.nextafter(1.0, 0.0), 2.0, 100.0], dtype=dtype)
    rtol = {np.float32: 1e-4, np.float64: 1e-5}[dtype]
    atol = {np.float32: 1e-5, np.float64: 1e-6}[dtype]
    self.assertAllClose(
        lsp_special.expn(n, x),
        osp_special.expn(n, x.astype(np.float64)).astype(dtype),
        rtol=rtol, atol=atol)

  @jtu.sample_product(dtype=float_dtypes, n=[-1, 0, 1, 2, 3, 4, 5, 10, 50, 5000])
  def testExpnAtZeroBoundary(self, dtype, n):
    # Regression test for https://github.com/jax-ml/jax/issues/41205
    x = np.array(0.0, dtype=dtype)
    tol = {np.float32: 1e-6, np.float64: 1e-14}
    expected = osp_special.expn(n, x.astype(np.float64)).astype(dtype)
    self.assertAllClose(lsp_special.expn(n, x), expected, rtol=tol, atol=tol)

  @jtu.sample_product(dtype=float_dtypes)
  def testExpnOutOfDomainAndInf(self, dtype):
    n = np.array([-2, -1, -1, 0, 1, 5000, 0, 1, 2, 5000, 1, 2], dtype=dtype)
    x = np.array([0.0, 0.5, 2.0, -1.0, -1.0, -1.0, np.inf, np.inf, np.inf,
                  np.inf, 1e20, np.nan], dtype=dtype)
    expected = osp_special.expn(n.astype(np.float64), x.astype(np.float64)).astype(dtype)
    self.assertAllClose(lsp_special.expn(n, x), expected)
    self.assertAllClose(lsp_special.exp1(np.array(np.inf, dtype=dtype)),
                        np.array(0.0, dtype=dtype))

  @jtu.sample_product(dtype=float_dtypes)
  def testExpnFloatNTruncation(self, dtype):
    # Verify non-integer float n is truncated to integer like scipy.special.expn
    n_float = np.array([0.8, 1.8, 2.5, 3.9], dtype=dtype)
    x_float = np.array([0.3, 0.3, 0.0, 2.0], dtype=dtype)
    n_int = np.trunc(n_float)
    self.assertAllClose(lsp_special.expn(n_float, x_float),
                        lsp_special.expn(n_int, x_float))

  @jtu.sample_product(dtype=float_dtypes, n=[40, 100, 200, 5000, 5001])
  def testExpnLargeN(self, dtype, n):
    x = np.array([0.0, 0.5, 1.0, 2.0, 10.0], dtype=dtype)
    rtol = {np.float32: 1e-4, np.float64: 1e-5}
    atol = {np.float32: 1e-5, np.float64: 1e-6}
    expected = osp_special.expn(n, x.astype(np.float64)).astype(dtype)
    self.assertAllClose(lsp_special.expn(n, x), expected, rtol=rtol, atol=atol)
    self.assertAllClose(
        lsp_special.expn(dtype(1e20), dtype(1.0)),
        dtype(np.exp(-1.0) / 1e20), rtol=rtol, atol=atol)

  @jtu.sample_product(dtype=float_dtypes, n=[0, 1, 2, 3])
  def testExpnGrads(self, dtype, n):
    x = np.array([0.5, 1.5, 3.0], dtype=dtype)
    tol = 1e-2 if jtu.test_device_matches(["tpu"]) else 1e-3
    jtu.check_grads(lambda x_: lsp_special.expn(n, x_), (x,), order=2,
                    atol=tol, rtol=tol, eps=1e-3)
    if n == 1:
      jtu.check_grads(lsp_special.exp1, (x,), order=2,
                      atol=tol, rtol=tol, eps=1e-3)

  @jtu.sample_product(dtype=float_dtypes)
  @jax.numpy_dtype_promotion('standard')  # polygamma mixes int n with float x.
  def testZetaLargeQ(self, dtype):
    if jaxlib_extension_version < 510:
      self.skipTest("Requires jaxlib_extension_version >= 510")
    # Regression test for https://github.com/jax-ml/jax/issues/38637,
    # https://github.com/jax-ml/jax/issues/38608 and
    # https://github.com/jax-ml/jax/issues/41366.
    # For q_big, q**-4 underflows to zero but zeta(4, q) ~ q**-3 / 3 does not.
    q_big = {np.float32: 1e12, np.float64: 1e100}[dtype]
    x = np.array([2.0, 4.0, 4.0], dtype=dtype)
    q = np.array([np.inf, np.inf, q_big], dtype=dtype)
    expected = np.array([0.0, 0.0, osp_special.zeta(4.0, q_big)], dtype=dtype)
    rtol = {np.float32: 1e-5, np.float64: 1e-12}
    self.assertAllClose(lsp_special.zeta(x, q), expected, rtol=rtol)
    self.assertAllClose(lsp_special.polygamma(np.array([1, 2]), q[:2]),
                        np.zeros(2, dtype))

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

  @parameterized.parameters([True, False])
  def testGammaDebugNans(self, with_jit):
    finfo = jnp.finfo(dtypes.default_float_dtype())
    # Cover all branches of _gamma_lanczos_impl: |x| < sqrt(eps) (including
    # +/-0.0), negative non-integers (reflection), positive integers <= max_int
    # (factorial table) and > max_int, x > overflow (including where both p and
    # e overflow), x < underflow, and +inf.
    x = jnp.asarray(
        [-1000.5, -100.5, -50.5, -5.5, -0.5, -float(finfo.tiny), -0.0, 0.0,
         float(finfo.tiny), 1e-5, 0.5, 1.0, 2.0, 10.0, 15.5, 35.0, 50.0, 100.0,
         220.0, 1000.0, float(finfo.max), np.inf],
        dtype=finfo.dtype)
    x_grad = jnp.asarray([-50.5, -5.5, -0.5, 0.5, 1.0, 2.5, 10.0, 15.5],
                         dtype=finfo.dtype)
    f = lsp_special.gamma
    df = jax.grad(lambda x: f(x).sum())
    if with_jit:
      f, df = jax.jit(f), jax.jit(df)
    with jax.debug_nans(True), jax.disable_jit(not with_jit):
      f(x)  # Doesn't crash
      df(x_grad)
      for nan_input in (-1.0, -2.0, -100.0, -float(finfo.max), -np.inf, np.nan):
        with self.assertRaisesRegex(FloatingPointError,
                                    "invalid value \\(nan\\)"):
          f(jnp.asarray(nan_input, dtype=finfo.dtype))

  @jtu.sample_product(dtype=float_dtypes)
  def test_log_ndtr_accuracy(self, dtype):
    # Regression test for https://github.com/jax-ml/jax/issues/41676:
    # log_ndtr(x) for x > 0 must avoid precision loss from evaluating
    # log(ndtr(x)) when ndtr(x) is close to 1.
    upper = 8.0 if dtype == np.float64 else 5.0
    x = np.linspace(0.1, upper, 200, dtype=dtype)
    expected = osp_special.log_ndtr(x.astype(np.float64)).astype(dtype)
    rtol = {
        np.float32: 1e-3 if jtu.test_device_matches(["tpu"]) else 1e-5,
        np.float64: 1e-13,
    }
    self.assertAllClose(lsp_special.log_ndtr(x), expected, atol=0, rtol=rtol)

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

  def testErfcinvExtremeValues(self):
    dtype = jnp.zeros(0).dtype
    args_maker = lambda: [np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.nan], dtype=dtype)]
    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    self._CheckAgainstNumpy(osp_special.erfcinv, lsp_special.erfcinv, args_maker, rtol=rtol)
    self._CompileAndCheck(lsp_special.erfcinv, args_maker, rtol=rtol)

  @parameterized.parameters([True, False])
  def testErfcinvDebugInfs(self, with_jit):
    f = jax.jit(lsp_special.erfcinv) if with_jit else lsp_special.erfcinv
    with jax.debug_infs(True):
      f(1.0)  # Doesn't crash
      with self.assertRaisesRegex(FloatingPointError, "invalid value \\(inf\\)"):
        f(0.0)
      with self.assertRaisesRegex(FloatingPointError, "invalid value \\(inf\\)"):
        f(2.0)

  @parameterized.parameters("rel_entr", "kl_div")
  def testRelEntrExtremeValues(self, op):
    # Testing at the extreme values, including inf and nan inputs (see
    # https://github.com/jax-ml/jax/issues/38262 and
    # https://github.com/jax-ml/jax/issues/38239).
    dtype = jnp.zeros(0).dtype  # default float dtype.
    vals = np.array([np.nan, -np.inf, -2, -1, 0, 1, np.inf], dtype=dtype)
    args_maker = lambda: [vals[:, None], vals[None, :]]
    scipy_op, lax_op = getattr(osp_special, op), getattr(lsp_special, op)
    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    self._CheckAgainstNumpy(scipy_op, lax_op, args_maker, rtol=rtol)
    self._CompileAndCheck(lax_op, args_maker, rtol=rtol)

  @parameterized.parameters("rel_entr", "kl_div")
  def testRelEntrPrecision(self, op):
    # Regression test for https://github.com/jax-ml/jax/issues/41200: p close
    # to q must not lose precision or produce negative kl_div, and large p must
    # not overflow to nan.
    dtype = jnp.zeros(0).dtype  # default float dtype.
    finfo = jnp.finfo(dtype)
    big = finfo.max / 4  # p * log(p) overflows for p this large.
    args_maker = lambda: [
        np.array([2 ** 20 * (1 + finfo.eps), big, big], dtype=dtype),
        np.array([2 ** 20, big, big / 2], dtype=dtype),
    ]
    if op == "rel_entr":
      expected_fn = osp_special.rel_entr
    else:
      expected_fn = lambda p, q: np.array(
          [0.5 * 2 ** 20 * finfo.eps ** 2, 0.0, big * (np.log(2.0) - 0.5)],
          dtype=dtype,
      )
    lax_op = getattr(lsp_special, op)
    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    self._CheckAgainstNumpy(expected_fn, lax_op, args_maker, atol=0, rtol=rtol)
    self._CompileAndCheck(lax_op, args_maker, atol=0, rtol=rtol)

    p32, q32 = np.float32(7.742924), np.float32(7.7429233)
    delta32 = p32 - q32
    expected32 = (
        delta32 if op == "rel_entr" else np.float32(0.5) * delta32 * (delta32 / q32)
    )
    self.assertAllClose(lax_op(p32, q32), expected32, atol=0, rtol=1e-6)

    if dtype == np.float64:
      p64 = np.float64(9.999999999999978e29)
      q64 = np.float64(9.999998737240631e29)
      expected64 = (
          osp_special.rel_entr(p64, q64)
          if op == "rel_entr"
          else np.float64(7.972806512934917e15)
      )
      self.assertAllClose(lax_op(p64, q64), expected64, atol=0, rtol=1e-15)

  @jtu.sample_product(
      op=["rel_entr", "kl_div"],
      dtype=float_dtypes,
  )
  def testRelEntrSmallQGrad(self, op, dtype):
    # Differentiating at 0 < q < finfo(dtype).max**(-0.5) must not overflow
    # 1 / q^2 to inf or nan.
    lax_op = getattr(lsp_special, op)
    small = {
        np.float16: 2e-3,
        dtypes.bfloat16: 1e-20,
        np.float32: 1e-20,
        np.float64: 1e-160,
    }[dtype]
    p = jnp.array([small, 2 * small], dtype=dtype)
    q = jnp.array([small, small], dtype=dtype)
    dp, dq = jax.grad(lambda p, q: lax_op(p, q).sum(), argnums=(0, 1))(p, q)
    if op == "rel_entr":
      expected_dp = np.array([1.0, np.log(2.0) + 1.0], dtype=dtype)
      expected_dq = np.array([-1.0, -2.0], dtype=dtype)
    else:
      expected_dp = np.array([0.0, np.log(2.0)], dtype=dtype)
      expected_dq = np.array([0.0, -1.0], dtype=dtype)
    rtol = 1e-2 if dtypes.finfo(dtype).bits < 32 else 1e-5
    self.assertAllClose(dp, expected_dp, atol=0, rtol=rtol)
    self.assertAllClose(dq, expected_dq, atol=0, rtol=rtol)

  @parameterized.product(op=["rel_entr", "kl_div"], with_jit=[True, False])
  def testRelEntrDebugNansAndInfs(self, op, with_jit):
    lax_op = getattr(lsp_special, op)
    f = jax.jit(lax_op) if with_jit else lax_op
    with jax.debug_nans(True), jax.debug_infs(True):
      f(1.0, 2.0)  # Doesn't crash
      jax.grad(f, argnums=(0, 1))(1.0, 2.0)
      with self.assertRaisesRegex(FloatingPointError, "invalid value \\(inf\\)"):
        f(1.0, 0.0)
      with self.assertRaisesRegex(FloatingPointError, "invalid value \\(nan\\)"):
        f(np.nan, 1.0)
    if op == "rel_entr":
      with jax.debug_nans(True):
        self.assertEqual(f(1.0, np.inf), -np.inf)

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

  def testExpiZero(self):
    self.assertEqual(lsp_special.expi(0.0), -np.inf)

  @jtu.sample_product(dtype=float_dtypes)
  def testExpiInf(self, dtype):
    # Regression test for https://github.com/jax-ml/jax/issues/38519
    args_maker = lambda: [np.array([-np.inf, 0.0, np.inf, np.nan], dtype=dtype)]
    # Note: scipy < 1.16 returned NaN for expi(+inf).
    scipy_op = lambda x: np.where(
        np.isposinf(x), np.inf, osp_special.expi(x.astype(np.float64))
    ).astype(x.dtype)
    self._CheckAgainstNumpy(scipy_op, lsp_special.expi, args_maker)
    self._CompileAndCheck(lsp_special.expi, args_maker)
    with jax.debug_nans(True):
      x = jnp.array([-np.inf, -1.0, 0.0, 1.0, np.inf], dtype=dtype)
      self.assertAllClose(lsp_special.expi(x), scipy_op(np.asarray(x)))

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

  @jtu.sample_product(
      op=["gammaincinv", "gammainccinv"],
      dtype=jtu.dtypes.floating,
  )
  def testGammaIncInvAccuracy(self, op, dtype):
    # Comprehensive accuracy tests are in tests/numerics/gamma_test.py; this
    # checks a grid that spans the tails of the distribution.
    finfo = jnp.finfo(dtype)
    a = np.logspace(-3, 4, 29)
    y = np.concatenate([
        np.logspace(-30, -1, 30),
        np.linspace(0.1, 0.9, 17)[1:-1],
        1 - np.logspace(-1, np.log10(finfo.eps), 20),
    ])
    a, y = (v.astype(dtype).ravel() for v in np.meshgrid(a, y))
    scipy_op = getattr(osp_special, op)
    expected = scipy_op(a.astype(np.float64), y.astype(np.float64))
    # Skip points whose result underflows or overflows in `dtype`.
    keep = ((expected > 1e3 * float(finfo.tiny))
            & (expected < float(finfo.max) / 10))
    a, y = a[keep], y[keep]
    args_maker = lambda: [a, y]
    if jtu.test_device_matches(["cpu"]):
      rtol = {dtypes.bfloat16: 1e-2, np.float16: 1e-3, np.float32: 3e-4,
              np.float64: 1e-10}
    else:
      rtol = {dtypes.bfloat16: 1e-2, np.float16: 1e-3, np.float32: 1e-2,
              np.float64: 1e-9}
    self._CheckAgainstNumpy(
        lambda a, y: scipy_op(a.astype(np.float64), y.astype(np.float64)),
        getattr(lsp_special, op), args_maker, check_dtypes=False, atol=0,
        rtol=rtol)
    self._CompileAndCheck(getattr(lsp_special, op), args_maker, rtol=rtol)

  @jtu.sample_product(
      op=["gammaincinv", "gammainccinv"],
      dtype=jtu.dtypes.floating,
  )
  def testGammaIncInvGrads(self, op, dtype):
    a = np.array([0.05, 0.3, 0.9, 1.0, 1.7, 5.0, 40.0, 300.0], dtype=dtype)
    y = np.array([0.6, 0.2, 0.5, 0.8, 0.99, 0.05, 0.3, 0.7], dtype=dtype)
    lax_op = getattr(lsp_special, op)
    tol = {dtypes.bfloat16: 2e-1, np.float16: 1e-1, np.float32: 5e-2,
           np.float64: 1e-5}
    eps = {dtypes.bfloat16: 1e-1, np.float16: 1e-2, np.float32: 1e-3,
           np.float64: 1e-6}[dtype]
    jtu.check_grads(lax_op, (a, y), order=1, modes=["fwd", "rev"],
                    atol=tol, rtol=tol, eps=eps)
    if dtype == np.float64:
      # Finite differences of lower precision JVPs are too noisy for order 2.
      jtu.check_grads(lambda y: lax_op(a, y), (y,), order=2,
                      modes=["fwd", "rev"], atol=tol, rtol=tol, eps=eps)

  @parameterized.product(op=["gammaincinv", "gammainccinv"],
                         with_jit=[True, False])
  def testGammaIncInvDebugNans(self, op, with_jit):
    # Every branch of the initial approximation and of the refinement is
    # evaluated for every element, so the branches that are not selected must
    # not produce NaNs either. Without jit, debug_nans checks the result of
    # every primitive, including those of the unselected branches.
    finfo = jnp.finfo(dtypes.default_float_dtype())
    # a < 1 and a >= 1, the thresholds of the initial approximation (0.3, 1,
    # 500, 1e5), of the refinement (200 in float32, 1e6 in float64) and of the
    # gradients (1e4 in float32, 1e6 in float64), and huge a.
    a = [float(finfo.tiny), 1e-6, 0.05, 0.3, 0.5, 0.999, 1.0, 1.5, 5.0, 30.0,
         150.0, 300.0, 600.0, 1e4, 2e5, 1e7, 1e30, float(finfo.max)]
    # Both tails, the switch between them at 0.5, and the endpoints.
    y = [0.0, float(finfo.tiny), 1e-30, 1e-5, 0.01, 0.15, 0.3, 0.5, 0.6, 0.9,
         0.99, 1 - 1e-5, 1 - float(finfo.eps), 1.0]
    a, y = (jnp.asarray(v.ravel(), dtype=finfo.dtype)
            for v in np.meshgrid(a, y))
    f = getattr(lsp_special, op)
    # With jit disabled, only the gradient with respect to y is checked, since
    # the one with respect to a uses lax.random_gamma_grad, which creates a NaN
    # constant.
    argnums = (0, 1) if with_jit else 1
    df = jax.grad(lambda a, y: f(a, y).sum(), argnums=argnums)
    if with_jit:
      f, df = jax.jit(f), jax.jit(df)
    with jax.debug_nans(True), jax.disable_jit(not with_jit):
      f(a, y)  # Doesn't crash
      df(a, y)
      with self.assertRaisesRegex(FloatingPointError, "invalid value \\(nan\\)"):
        f(2.0, 1.5)

  @parameterized.parameters("gammaincinv", "gammainccinv")
  def testGammaIncInvGradsAtEndpoints(self, op):
    # The result is constant (0 or inf) at y in {0, 1}, so the gradient with
    # respect to a is zero there.
    lax_op = getattr(lsp_special, op)
    a = jnp.array([0.5, 1.0, 2.0, 0.5, 1.0, 2.0])
    y = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    da, dy = jax.grad(lambda a, y: lax_op(a, y).sum(), argnums=(0, 1))(a, y)
    self.assertArraysEqual(da, jnp.zeros_like(a))
    if op == "gammaincinv":
      expected_dy = jnp.array([0.0, 1.0, np.inf, np.inf, np.inf, np.inf])
    else:
      expected_dy = jnp.array([-np.inf] * 3 + [0.0, -1.0, -np.inf])
    self.assertAllClose(dy, expected_dy.astype(dy.dtype))

  @jtu.sample_product(dtype=float_dtypes)
  def testBetaIncResultDoesNotDependOnBatch(self, dtype):
    # Each element of the continued fraction should stop iterating once it has
    # converged, so the result for an element must not depend on the other
    # elements in the batch (#28547).
    n = 20
    a = jtu.rand_uniform(self.rng(), low=0.5, high=10.)((n,), dtype)
    b = jtu.rand_uniform(self.rng(), low=0.5, high=10.)((n,), dtype)
    x = jtu.rand_uniform(self.rng())((n,), dtype)
    betainc = jax.jit(lsp_special.betainc)
    expected = np.concatenate(
        [betainc(a[i:i + 1], b[i:i + 1], x[i:i + 1]) for i in range(n)])
    self.assertAllClose(betainc(a, b, x), expected,
                        rtol={np.float32: 1e-6, np.float64: 1e-14})

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

  @jtu.sample_product(dtype=float_dtypes)
  def testHyp2f1SpecialCases(self, dtype):
    eps = np.finfo(dtype).eps

    a_samples = np.array([
        0, 1, 1, 1, 1, 5, 5, 0.245, 0.45, 0.45, 2, 0.4, 0.32, 4, 4,
        0.3, 1.5, 1.2, 2.5, 0.5, 2.1, 1e-4, 0.46,
    ], dtype=dtype)
    b_samples = np.array([
        1, 0, 1, 1, 1, 1, 1, 3, 0.7, 0.7, 1, 0.7, 0.76, 2, 3,
        0.7, 2.5, 1.8, 3.5, 22.5, 3.2, 0.9999, 0.54,
    ], dtype=dtype)
    c_samples = np.array([
        1, 1, 0, 1, -1, 3, 3, 3, 0.45, 0.45, 5, 0.3, 0.11, 7, 7,
        3.0, 7.0, 1.0, 3.0, 2.0, 2.1 + 3.2 + 3.0 + 10 * eps, 1.0, 1.0,
    ], dtype=dtype)
    x_samples = np.array([
        1, 1, 1, 0, 1, 0.5, 1, 0.35, 0.35, 1.5, 1, 0.4, 0.95, 0.95, 0.95,
        0.97, 0.95, 0.93, 0.95, 0.91, 0.95, 0.95, 0.95,
    ], dtype=dtype)

    args_maker = lambda: (a_samples, b_samples, c_samples, x_samples)
    rtol = 1E-3 if jtu.test_device_matches(["tpu"]) else 5e-5
    self._CheckAgainstNumpy(osp_special.hyp2f1, lsp_special.hyp2f1, args_maker, rtol=rtol)
    self._CompileAndCheck(lsp_special.hyp2f1, args_maker, rtol=rtol)

    with jax.debug_nans(True), jax.debug_infs(True):
      a_dbg = jnp.array([4.0, 2.0, 1e-4], dtype=dtype)
      b_dbg = jnp.array([2.0, 1.0, 0.9999], dtype=dtype)
      c_dbg = jnp.array([7.0, 5.0, 1.0], dtype=dtype)
      x_dbg = jnp.array([0.95, 1.0, 0.95], dtype=dtype)
      out = lsp_special.hyp2f1(a_dbg, b_dbg, c_dbg, x_dbg)
      self.assertAllClose(
          out,
          osp_special.hyp2f1(
              np.array([4.0, 2.0, 1e-4], dtype=dtype),
              np.array([2.0, 1.0, 0.9999], dtype=dtype),
              np.array([7.0, 5.0, 1.0], dtype=dtype),
              np.array([0.95, 1.0, 0.95], dtype=dtype),
          ),
          rtol=rtol,
      )
      # Run _hyp2f1_digamma_transform eagerly to check loop carries for NaN/Inf.
      log_gamma_ratio = (
          jax.lax.lgamma(c_dbg) - jax.lax.lgamma(a_dbg) - jax.lax.lgamma(b_dbg)
      )
      sgn_gamma_ratio = jnp.ones_like(c_dbg)
      jax.vmap(special_internal._hyp2f1_digamma_transform)(
          a_dbg, b_dbg, c_dbg, x_dbg, log_gamma_ratio, sgn_gamma_ratio
      )

  @jtu.sample_product(dtype=float_dtypes)
  def testHyp2f1ParameterDerivatives(self, dtype):
    a = np.array([0.15, 5.0], dtype=dtype)
    b = np.array([0.25, 3.0], dtype=dtype)
    c = np.array([0.8, 8.0], dtype=dtype)
    x = np.array([0.4, 0.25], dtype=dtype)
    tol = 1e-2 if jtu.test_device_matches(["tpu"]) else 1e-3
    jtu.check_grads(lsp_special.hyp2f1, (a, b, c, x), order=1,
                    atol=tol, rtol=tol, eps=1e-3)

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

  def test_i1_vmap_jacfwd_maclaurin(self):
    def d_i1_ref(x, n):
      x64 = np.asarray(x, dtype=np.float64)
      return sum(
          math.comb(n + 1, k) * osp_special.iv(abs(2 * k - n - 1), x64)
          for k in range(n + 2)
      ) / (2 ** (n + 1))

    # Nested vmap(jacfwd) through i1's Maclaurin series branch (|x| <= sqrt(eps)).
    xs = jnp.array([0.0, 1e-4], dtype=jnp.float32)
    g = jax.scipy.special.i1
    derivs_0 = []
    derivs_small = []
    for n in range(1, 5):
      g = (jax.vmap(jax.jacfwd(g)) if n == 1
           else jax.vmap(jax.jacfwd(lambda z, g=g: g(z[None])[0])))
      vals = g(xs)
      derivs_0.append(vals[0])
      derivs_small.append(vals[1])

    self.assertAllClose(
        jnp.stack(derivs_0),
        jnp.array([0.5, 0.0, 0.375, 0.0], dtype=jnp.float32),
        atol=1e-6)
    expected_small = jnp.array(
        [d_i1_ref(1e-4, n) for n in range(1, 5)], dtype=jnp.float32)
    self.assertAllClose(
        jnp.stack(derivs_small), expected_small, atol=1e-6, rtol=1e-5)

  def testBetalnHessianAtZero(self):
    # Regression test for https://github.com/jax-ml/jax/issues/34353
    def f(log_a, log_b):
      a = jnp.exp(log_a)
      b = jnp.exp(log_b)
      return lsp_special.betaln(a, b) - lsp_special.betaln(a + 1, b)
    hess = jax.hessian(f, argnums=(0, 1))(0., 0.)
    expected = jnp.array([[0.25, -0.25], [-0.25, 0.25]])
    tol = {
        np.float32: 1e-4 if jtu.test_device_matches(["tpu"]) else 1e-5,
        np.float64: 1e-13,
    }
    self.assertAllClose(jnp.array(hess), expected, atol=tol, rtol=tol)

  def test_gammaln_exact_zeros_and_special_values(self):
    # Exact zeros at x = 1.0 and x = 2.0
    self.assertEqual(float(lsp_special.gammaln(np.float32(1.0))), 0.0)
    self.assertEqual(float(lsp_special.gammaln(np.float32(2.0))), 0.0)

    # Special values: poles, overflow threshold, infinities, NaN
    max_finite_arg = np.float32(4.085003e36)
    first_overflow_arg = np.nextafter(max_finite_arg, np.float32(np.inf))
    self.assertTrue(np.isfinite(float(lsp_special.gammaln(max_finite_arg))))
    special_inf = np.array([
        0.0, -0.0, -1.0, -2.0, -3.0, -10.0, -1000.0, -1e7,
        first_overflow_arg, np.inf, -np.inf
    ], dtype=np.float32)
    res_inf = np.asarray(lsp_special.gammaln(special_inf))
    self.assertTrue(np.all(np.isposinf(res_inf)), f"Expected +inf, got {res_inf}")
    self.assertTrue(np.isnan(float(lsp_special.gammaln(np.float32(np.nan)))))

  @jtu.sample_product(dtype=float_dtypes)
  def test_digamma_special_values(self, dtype):
    large_neg_int = -1e16 if dtype == np.float64 else -1e7
    special_in = np.array(
        [0.0, -0.0, np.inf, -np.inf, -1.0, -2.0, -10.0, large_neg_int, np.nan],
        dtype=dtype,
    )
    expected = np.array(
        [-np.inf, np.inf, np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        dtype=dtype,
    )
    actual = np.asarray(lsp_special.digamma(special_in))
    np.testing.assert_equal(actual, expected)

  def test_gammaln_grad(self):
    x_grad = np.array([0.25, 0.75, 1.5, 3.5, 12.0, -0.5, -1.5, -2.5], dtype=np.float32)
    self.assertAllClose(
        jax.vmap(jax.grad(lsp_special.gammaln))(x_grad),
        lsp_special.digamma(x_grad),
        rtol=1e-5, atol=1e-5)

  def test_lgamma_weak_type_scalar(self):
    x = lax_internal._convert_element_type(
        2.0, np.dtype(np.float32), weak_type=True
    )
    y = jax.lax.lgamma(x)
    self.assertTrue(dtypes.is_weakly_typed(y))
    self.assertEqual(float(y), 0.0)

  def test_gammaln_negative_roots_ulp(self):
    # 75 certified points (5 float32 neighbors around each of the 15 negative
    # roots of lgamma on (-10, -2)), with reference values computed via 60-digit
    # mpmath: `float(mpmath.log(abs(mpmath.gamma(mpmath.mpf(float(x))))))`.
    certified_root_pts = [
        (-2.457024335861206, 6.098183754018183e-07),
        (-2.457024574279785, 2.484696988890678e-07),
        (-2.4570248126983643, -1.1287842529645816e-07),
        (-2.4570250511169434, -4.742259971548743e-07),
        (-2.4570252895355225, -8.355730166862955e-07),
        (-3.143580436706543, 3.5146424031690423e-06),
        (-3.143580675125122, 1.659292425530781e-06),
        (-3.143580913543701, -1.9605461541614247e-07),
        (-3.1435811519622803, -2.051398719680857e-06),
        (-3.1435813903808594, -3.906739887272491e-06),
        (-2.7476820945739746, -1.0570121172887281e-06),
        (-2.7476823329925537, -6.005974428686868e-07),
        (-2.747682571411133, -1.4418167987778365e-07),
        (-2.747682809829712, 3.1223517168561536e-07),
        (-2.747683048248291, 7.686531118231443e-07),
        (-4.039361000061035, 2.249564626449368e-05),
        (-4.039361476898193, 9.720763558572296e-06),
        (-4.039361953735352, -3.053971690394376e-06),
        (-4.03936243057251, -1.5828559485961922e-05),
        (-4.039362907409668, -2.8602999831685795e-05),
        (-3.955293893814087, -8.104382793261057e-06),
        (-3.955294132232666, -3.1631758619440246e-06),
        (-3.955294370651245, 1.7780596858250852e-06),
        (-3.955294609069824, 6.7193238503496265e-06),
        (-3.9552948474884033, 1.1660616631932957e-05),
        (-5.0082173347473145, 0.00010283681295424625),
        (-5.008217811584473, 4.400893627975495e-05),
        (-5.008218288421631, -1.4815572806334704e-05),
        (-5.008218765258789, -7.363671469471415e-05),
        (-5.008219242095947, -0.00013245448977600686),
        (-4.991543769836426, -0.00010146515471314859),
        (-4.991544246673584, -4.590078166675967e-05),
        (-4.991544723510742, 9.666772148365217e-06),
        (-4.9915452003479, 6.523750709091674e-05),
        (-4.991545677185059, 0.00012081142351964627),
        (-6.001384258270264, 0.0007502033977111201),
        (-6.001384735107422, 0.0004049005350435363),
        (-6.00138521194458, 5.9716251821000246e-05),
        (-6.001385688781738, -0.00028534953358012286),
        (-6.0013861656188965, -0.0006302969026992184),
        (-5.9986066818237305, -0.000571583788528323),
        (-5.998607158660889, -0.0002301889621487998),
        (-5.998607635498047, 0.00011132306745354033),
        (-5.998608112335205, 0.0004529523805681395),
        (-5.998608589172363, 0.0007946990575669714),
        (-7.000197410583496, 0.0046656094182512976),
        (-7.000197887420654, 0.002252102177623758),
        (-7.0001983642578125, -0.0001555986874670934),
        (-7.000198841094971, -0.0025575210588133666),
        (-7.000199317932129, -0.004953692617858581),
        (-6.999800682067871, -0.004150185635306569),
        (-6.999801158905029, -0.00175593634176963),
        (-6.9998016357421875, 0.0006440637640683667),
        (-6.999802112579346, 0.0030498423636509397),
        (-6.999802589416504, 0.0054614273387702),
        (-8.000022888183594, 0.08023788354150395),
        (-8.00002384185791, 0.03941384761713908),
        (-8.000024795532227, 0.00019109306263327084),
        (-8.000025749206543, -0.037551276318553026),
        (-8.00002670288086, -0.0739209618848819),
        (-7.999974250793457, -0.03744103667945246),
        (-7.999974727630615, -0.01874992444328908),
        (-7.999975204467773, 0.0002972497521375302),
        (-7.999975681304932, 0.019714314834692555),
        (-7.99997615814209, 0.03951592135704702),
        (-9.000000953674316, 1.0611139836802737),
        (-9.000001907348633, 0.367964655686062),
        (-9.00000286102295, -0.037502599853472283),
        (-9.000003814697266, -0.32518681973372665),
        (-9.000004768371582, -0.5483325184735134),
        (-8.999995231628418, -0.5483110440874025),
        (-8.999996185302734, -0.3251696402248379),
        (-8.99999713897705, -0.03748971522180572),
        (-8.999998092651367, 0.3679732454405064),
        (-8.999999046325684, 1.061118278557496),
    ]
    cert_xs = np.array([p[0] for p in certified_root_pts], dtype=np.float32)
    cert_ref = np.array([p[1] for p in certified_root_pts], dtype=np.float64)
    ys_f32 = np.asarray(lsp_special.gammaln(cert_xs), dtype=np.float64)
    ulp = np.abs(np.spacing(cert_ref.astype(np.float32)).astype(np.float64))
    errs = np.abs(ys_f32 - cert_ref) / ulp
    idx = int(np.argmax(errs))
    max_root_ulp, worst_root_x = float(errs[idx]), float(cert_xs[idx])
    self.assertLessEqual(
        max_root_ulp, 2.0,
        f"Max ULP {max_root_ulp} at negative root x={worst_root_x} exceeds 2.0")

  def test_gammaln_sweep_ulp(self):
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
            # Worst-case inputs across positive and negative float32 domains:
            np.float32(0.70681477),
            np.float32(2.0086858),
            np.float32(4.012665),
            np.float32(-3.1757019),
            np.float32(-2.5576105),
            np.float32(-2.5056348),
        ], dtype=np.float32),
        np.linspace(-1.999, -1.001, 400, dtype=np.float32),
        np.linspace(-0.999, -0.001, 400, dtype=np.float32),
        np.linspace(-99.9, -10.1, 400, dtype=np.float32),
    ])
    sweep_xs = sweep_xs[sweep_xs != np.floor(sweep_xs)]
    sweep_ref = osp_special.gammaln(sweep_xs.astype(np.float64))
    ys_f32 = np.asarray(lsp_special.gammaln(sweep_xs), dtype=np.float64)
    ulp = np.abs(np.spacing(sweep_ref.astype(np.float32)).astype(np.float64))
    errs = np.abs(ys_f32 - sweep_ref) / ulp
    idx = int(np.argmax(errs))
    max_sweep_ulp, worst_sweep_x = float(errs[idx]), float(sweep_xs[idx])
    self.assertLessEqual(
        max_sweep_ulp, 3.6,
        f"Max ULP {max_sweep_ulp} at x={worst_sweep_x} exceeds 3.6")

  @jtu.run_on_devices("cpu", "gpu")
  def test_gammaln_f64_negative_roots_ulp(self):
    # Certified 100-digit mpmath reference values (ref_hi, ref_lo) for the
    # 3 nearest float64 values around all 29 roots of lgamma on (-17, -2).
    certified_root_pts = [
        (-2.457024738220801, -6.168712140884665e-16, -1.1724810400044292e-32),
        (-2.4570247382208006, 5.619192358950097e-17, -2.1099251131628624e-33),
        (-2.4570247382208, 7.292550612674704e-16, -2.3725667710553419e-32),
        (-2.747682646727413, 1.0234931857070284e-15, -5.815475533689439e-32),
        (-2.7476826467274127, 1.733509244024501e-16, -8.202386541675038e-33),
        (-2.7476826467274122, -6.767913369021244e-16, 2.2103194612943698e-32),
        (-3.1435808883499803, -1.7579854183617677e-15, -8.017668454538754e-32),
        (-3.14358088834998, 1.6978655906121085e-15, -2.650245429739361e-32),
        (-3.1435808883499794, 5.153716599585995e-15, -2.3656873247166562e-31),
        (-4.039361839740538, -2.945930505051872e-14, 1.9926236539899832e-30),
        (-4.039361839740537, -5.664578074060335e-15, -2.310104163553895e-31),
        (-4.039361839740536, 1.8130148902398562e-14, -4.740941085166155e-31),
        (-3.9552942848585984, 8.789393144349778e-15, 7.393596062340719e-31),
        (-3.955294284858598, -4.14382750757705e-16, -1.484838516795255e-32),
        (-3.9552942848585975, -9.618158645865089e-15, -9.273334773992607e-32),
        (-5.008218168322594, -1.0414877920967533e-13, 3.890947542272188e-30),
        (-5.0082181683225935, 5.4188509265538106e-15, -3.087697519452421e-31),
        (-5.008218168322593, 1.1498648106279463e-13, -4.883302900304056e-30),
        (-4.991544640560049, 1.2118818570529118e-13, 1.0087702419250904e-29),
        (-4.991544640560048, 1.7683619350849613e-14, 6.802050189961466e-31),
        (-4.991544640560047, -8.582094700358093e-14, 2.68903147487666e-30),
        (-6.001385294453156, -6.892410931887324e-13, -7.79031942135438e-30),
        (-6.001385294453155, -4.6433836942838004e-14, -2.8240401579926895e-30),
        (-6.001385294453154, 5.963734193034675e-13, -3.5213040342279675e-30),
        (-5.998607480080876, 3.9894309784344193e-13, 1.635251310128767e-31),
        (-5.998607480080875, -2.3721063667118474e-13, 9.315898412228238e-30),
        (-5.998607480080874, -8.733643711854045e-13, -3.897714828817849e-29),
        (-7.000198333407326, -5.743202285494227e-12, -2.926203019341081e-28),
        (-7.000198333407325, -1.263203743493977e-12, -1.668206597127877e-29),
        (-7.000198333407324, 3.2167947985263277e-12, -7.074971754319108e-29),
        (-6.999801507890639, 4.525967654362372e-12, -2.982850683385144e-28),
        (-6.999801507890638, 5.313011065735902e-14, 1.2534571068034888e-30),
        (-6.999801507890637, -4.419707433027631e-12, 2.4952644109448514e-29),
        (-8.000024800270683, -5.407075634040359e-11, 1.8802317830174375e-27),
        (-8.000024800270682, 1.75595561986039e-11, -5.938674806026605e-28),
        (-8.00002480027068, 8.918986874274175e-11, -1.2081920453859645e-27),
        (-7.9999751970958215, 3.36862440532578e-11, 2.0705321558877385e-27),
        (-7.999975197095821, -2.1213071311827735e-12, -1.5609212852669594e-28),
        (-7.99997519709582, -3.792885831434103e-11, -2.8476204873520135e-27),
        (-9.000002755714824, -6.101697362085198e-10, 2.7288562265919278e-27),
        (-9.000002755714823, 3.444263328391509e-11, 1.401161623436041e-27),
        (-9.00000275571482, 6.790550031918699e-10, -1.3049928114642722e-26),
        (-8.999997244250979, 5.640901896019856e-10, -3.4399603112744113e-26),
        (-8.999997244250977, -8.050618056741812e-11, -4.5729198798458426e-27),
        (-8.999997244250975, -7.251025503213122e-10, 3.8338878910221144e-26),
        (-10.000000275573015, -5.179246902549113e-09, 3.162321103039447e-25),
        (-10.000000275573013, 1.2668051387565237e-09, -7.873844332328773e-26),
        (-10.000000275573012, 7.712857221613694e-09, -7.101547105269576e-26),
        (-9.999999724426631, 8.217989771788738e-09, 5.381094078934247e-25),
        (-9.99999972442663, 1.7719543958825935e-09, -2.1976305076693928e-26),
        (-9.999999724426628, -4.674080938472126e-09, -3.18635354543402e-25),
        (-11.000000025052108, -4.356008489074073e-08, -2.1579103722023138e-24),
        (-11.000000025052106, 2.734640389948086e-08, 2.3951152444899338e-25),
        (-11.000000025052104, 9.825289771743235e-08, -1.8531020431268426e-24),
        (-10.999999974947892, 7.88275632625459e-08, 3.8984260769149134e-24),
        (-10.99999997494789, 7.92108817813105e-09, -7.32695326610357e-25),
        (-10.999999974947889, -6.298538187855533e-08, -3.1713846083955095e-24),
        (-12.000000002087678, -9.094235653270309e-07, -3.54857283347055e-23),
        (-12.000000002087676, -5.854619992113373e-08, 1.3602697451136098e-24),
        (-12.000000002087674, 7.923318894776628e-07, 2.3303065926051062e-23),
        (-11.999999997912326, 8.028787978744734e-07, -2.9788462795805954e-23),
        (-11.999999997912324, -4.799928255018621e-08, -2.0823367332414347e-24),
        (-11.999999997912322, -8.988766389819464e-07, -2.5143023704070188e-24),
        (-13.000000000160592, -6.86068936352033e-06, -5.384108237655489e-23),
        (-13.00000000016059, 4.200706921268597e-06, -4.641977741130562e-23),
        (-13.000000000160588, 1.5262225561898617e-05, 8.52145242035022e-22),
        (-12.999999999839412, 1.5263061556653325e-05, -5.968560127827181e-22),
        (-12.99999999983941, 4.201542925270726e-06, -6.73236125478115e-24),
        (-12.999999999839408, -6.8598533502707775e-06, -2.1953088975436652e-22),
        (-14.000000000011472, -8.428665026119921e-05, 4.79949023045375e-21),
        (-14.00000000001147, 7.057204149708364e-05, -5.007000228507924e-21),
        (-14.000000000011468, 0.00022545471818410072, 9.750588614122606e-21),
        (-13.999999999988532, 0.00022545477952377025, -1.3186084232934577e-20),
        (-13.99999999998853, 7.057210284625434e-05, -1.770024227465518e-21),
        (-13.999999999988528, -8.428658890252731e-05, -6.4474663886106704e-21),
        (-15.000000000000766, -0.001167626509417739, 9.270547832399377e-21),
        (-15.000000000000764, 0.0011552549067268103, -3.923735615601284e-20),
        (-15.000000000000762, 0.0034835446663228154, -1.5679262683532767e-19),
        (-14.999999999999238, 0.0034835446705004324, 7.247168913395353e-20),
        (-14.999999999999236, 0.0011552549109141656, -7.834098153052744e-20),
        (-14.999999999999234, -0.0011676265052206457, -8.136058438881012e-20),
        (-16.00000000000005, -0.03985276881869601, 3.1350903422853657e-18),
        (-16.000000000000046, 0.03425520333503583, 8.035076519752615e-19),
        (-16.000000000000043, 0.11429791100858222, -3.2274129709007887e-18),
        (-15.999999999999954, 0.03425520333529479, 3.36692029490901e-18),
        (-15.999999999999952, -0.0034851246475472554, 1.6830914752443702e-20),
        (-15.99999999999995, -0.03985276881841712, -2.644488539364109e-18),
    ]
    cert_xs = np.array([p[0] for p in certified_root_pts], dtype=np.float64)
    cert_hi = np.array([p[1] for p in certified_root_pts], dtype=np.float64)
    cert_lo = np.array([p[2] for p in certified_root_pts], dtype=np.float64)
    with jax.enable_x64():
      ys_f64 = np.asarray(lsp_special.gammaln(cert_xs), dtype=np.float64)
    ulp = np.abs(np.spacing(cert_hi))
    errs = np.abs((ys_f64 - cert_hi) - cert_lo) / ulp
    idx = int(np.argmax(errs))
    max_root_ulp, worst_root_x = float(errs[idx]), float(cert_xs[idx])
    self.assertLessEqual(
        max_root_ulp, 2.0,
        f"Max ULP {max_root_ulp} at negative root x={worst_root_x} exceeds 2.0")

  @jtu.sample_product(
      dtype=[np.float32, np.float64],
  )
  def testLoggammaZero(self, dtype):
    # Regression test for https://github.com/jax-ml/jax/issues/38240
    x = np.array([0.0, -0.0], dtype=dtype)
    self._CheckAgainstNumpy(
        osp_special.loggamma,
        jax.scipy.special.loggamma,
        args_maker=lambda: (x,),
        check_dtypes=False,
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
