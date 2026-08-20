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
import mpmath

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import scipy
import scipy.special as osp_special

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.scipy import special as _src_special
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
        "boxcox", 2, float_dtypes, jtu.rand_positive, False
    ),
    op_record(
        "boxcox1p", 2, float_dtypes,
        functools.partial(jtu.rand_uniform, low=-0.5, high=5.0), False
    ),
    op_record(
        "inv_boxcox", 2, float_dtypes, jtu.rand_positive, False
    ),
    op_record(
        "inv_boxcox1p", 2, float_dtypes, jtu.rand_positive, False
    ),
    op_record(
        "exprel", 1, float_dtypes, jtu.rand_default, True
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


def _dense_around(dt, center=0.0, high=None, num=9,
                   include_tiny=False):
  """Symmetric log-spaced test points around `center`, from ±eps to ±high."""
  eps = float(np.finfo(dt).eps)
  tiny = [float(np.finfo(dt).tiny)] if include_tiny else []
  core = [-eps] + [-t for t in tiny] + [0.0] + tiny + [eps]
  if high is None:
    return center + np.array(core)
  ladder = np.geomspace(eps, high, num)
  return np.unique(np.concatenate(
      [center - ladder[::-1], center + np.array(core),
       center + ladder]
  ))


def _signed_sweep(dt, high=5.0, num=5, spread=0.5):
  """Signed points across [-high, high], dense near zero."""
  pos = np.geomspace(0.01, high, num)
  near = _dense_around(dt, high=spread, include_tiny=True)
  return jnp.asarray(
      np.union1d(np.concatenate([-pos, pos]), near), dt)


def _boxcox_inputs(dt, is_1p=False):
  """Positive x in [0.01, 100], dense near 1."""
  pts = np.union1d(
      np.geomspace(0.01, 100.0, 9),
      _dense_around(dt, center=1.0, high=0.5))
  return jnp.asarray(pts - 1.0 if is_1p else pts, dt)


def _boxcox_lambdas(dt):
  """Lambdas in [-2, 2], dense near 0, ±1 (branch points)."""
  return np.unique(np.concatenate([
      [-2.0, -0.5, 0.5, 2.0],
      _dense_around(dt, center=-1.0),
      _dense_around(dt, high=0.5, num=7, include_tiny=True),
      _dense_around(dt, center=1.0),
  ]))


def _boxcox_grid(dt, **kwargs):
  """(x, lambda) pairs for boxcox / boxcox1p tests."""
  xg, lg = np.meshgrid(
      _boxcox_inputs(dt, **kwargs), _boxcox_lambdas(dt))
  return jnp.asarray(xg.ravel(), dt), jnp.asarray(lg.ravel(), dt)


def _inv_boxcox_grid(dt):
  """(y, lambda) pairs for inv_boxcox, filtered to valid domain."""
  yg, lg = np.meshgrid(_signed_sweep(dt), _boxcox_lambdas(dt))
  yf, lf = yg.ravel(), lg.ravel()
  valid = (1.0 + lf * yf) > 0.0
  return jnp.asarray(yf[valid], dt), jnp.asarray(lf[valid], dt)


def _mp_eval(fn, x):
  """Evaluate scalar fn via mpmath at 50-digit precision."""
  x_np = np.asarray(x, dtype=np.float64)
  with mpmath.workdps(50):
    res = [float(fn(mpmath.mpf(xi))) for xi in x_np.flat]
  return np.asarray(res, dtype=np.float64).reshape(x.shape)


def _np_exprel(x):
  """mpmath float64 reference for exprel(x)."""
  return _mp_eval(lambda xi: mpmath.expm1(xi) / xi if xi != 0 else 1.0, x)


def _np_dexprel_dx(x):
  """mpmath float64 reference for d/dx exprel(x)."""
  def fn(xi):
    if abs(xi) < 1e-10:
      return 0.5 + float(xi) / 3.0
    return float((mpmath.exp(xi) * (xi - 1) + 1) / xi**2)
  return _mp_eval(fn, x)


def _np_logrel(x):
  """mpmath float64 reference for logrel(x) = log1p(x) / x."""
  return _mp_eval(lambda xi: mpmath.log1p(xi) / xi if xi != 0 else 1.0, x)


def _np_dlogrel_dx(x):
  """mpmath float64 reference for d/dx logrel(x)."""
  def fn(xi):
    if abs(xi) < 1e-10:
      return -0.5 + 2.0 * float(xi) / 3.0
    return float((xi / (1 + xi) - mpmath.log1p(xi)) / xi**2)
  return _mp_eval(fn, x)


def _np_boxcox(x, lmbda, is_1p=False):
  """NumPy float64 reference for boxcox(x, lmbda) or boxcox1p."""
  x64 = np.asarray(x, dtype=np.float64)
  l64 = np.asarray(lmbda, dtype=np.float64)
  lx = np.log1p(x64) if is_1p else np.log(x64)
  return lx * _np_exprel(l64 * lx)


def _np_boxcox_grad(x, lmbda, is_1p=False):
  """NumPy float64 reference for gradients (d/dx, d/dlmbda) of boxcox."""
  x64 = np.asarray(x, dtype=np.float64)
  l64 = np.asarray(lmbda, dtype=np.float64)
  lx = np.log1p(x64) if is_1p else np.log(x64)
  dx = np.exp((l64 - 1.0) * lx)
  dlmbda = lx * (lx * _np_dexprel_dx(l64 * lx))
  return dx, dlmbda


def _np_boxcox_log_input_grad(lx, lmbda):
  """NumPy float64 reference for gradients of _boxcox_log_input."""
  lx64 = np.asarray(lx, dtype=np.float64)
  l64 = np.asarray(lmbda, dtype=np.float64)
  dlx = np.exp(l64 * lx64)
  dlmbda = lx64 * (lx64 * _np_dexprel_dx(l64 * lx64))
  return dlx, dlmbda


def _np_inv_boxcox_log_output(y, lmbda):
  """NumPy float64 reference for _inv_boxcox_log_output(y, lmbda)."""
  y64 = np.asarray(y, dtype=np.float64)
  l64 = np.asarray(lmbda, dtype=np.float64)
  return y64 * _np_logrel(l64 * y64)


def _np_inv_boxcox_log_output_grad(y, lmbda):
  """NumPy float64 reference for gradients of _inv_boxcox_log_output."""
  y64 = np.asarray(y, dtype=np.float64)
  l64 = np.asarray(lmbda, dtype=np.float64)
  lx = _np_inv_boxcox_log_output(y64, l64)
  dy = np.where(1.0 + l64 * y64 <= 0.0, np.nan, np.exp(-l64 * lx))
  dlmbda = y64 * (y64 * _np_dlogrel_dx(l64 * y64))
  return dy, dlmbda


def _np_inv_boxcox(y, lmbda, is_1p=False):
  """NumPy float64 reference for inv_boxcox(y, lmbda) or inv_boxcox1p."""
  log_out = _np_inv_boxcox_log_output(y, lmbda)
  return np.expm1(log_out) if is_1p else np.exp(log_out)


def _np_inv_boxcox_grad(y, lmbda):  # No is_1p: gradients are identical.
  """NumPy float64 reference for gradients of inv_boxcox or inv_boxcox1p."""
  y64 = np.asarray(y, dtype=np.float64)
  l64 = np.asarray(lmbda, dtype=np.float64)
  lx = _np_inv_boxcox_log_output(y64, l64)
  fwd = np.exp(lx)
  dy = np.where(1.0 + l64 * y64 <= 0.0, np.nan, np.exp((1.0 - l64) * lx))
  dlmbda = fwd * (y64 * (y64 * _np_dlogrel_dx(l64 * y64)))
  return dy, dlmbda


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

  def _boxcox_tol(self, dt):
    if jtu.test_device_matches(["tpu"]):
      return 1e-3
    return 1e-14 if dt == jnp.float64 else None

  def _check_boxcox_vals(self, dt, out, ref):
    tol = self._boxcox_tol(dt)
    self.assertAllClose(
        out, jnp.asarray(ref, dtype=dt), atol=tol, rtol=tol)

  def _check_boxcox_grads(self, dt, a, b, out_fn, ref_fn):
    tol = self._boxcox_tol(dt)
    d_out = jax.vmap(jax.grad(out_fn, argnums=(0, 1)))(a, b)
    d_ref = ref_fn(a, b)
    for o, r in zip(d_out, d_ref):
      self.assertAllClose(
          o, jnp.asarray(r, dtype=dt), atol=tol, rtol=tol)

  @parameterized.parameters(jtu.dtypes.floating)
  def testExprel(self, dt):
    x = _signed_sweep(dt)
    out = lsp_special.exprel(x)
    ref = _np_exprel(x)
    self._check_boxcox_vals(dt, out, ref)

  @parameterized.parameters(jtu.dtypes.floating)
  def testExprelGrad(self, dt):
    x = _signed_sweep(dt)
    out = jax.vmap(jax.grad(lsp_special.exprel))(x)
    ref = _np_dexprel_dx(x)
    self._check_boxcox_vals(dt, out, ref)

  @parameterized.parameters(jtu.dtypes.floating)
  def testExprelGradCrossover(self, dt):
    # Tests crossover of d/dx exprel at +/- 0.4 across [-0.8, 0.8].
    x = jnp.linspace(dt(-0.8), dt(0.8), 201, dtype=dt)
    out = jax.vmap(jax.grad(lsp_special.exprel))(x)
    ref = _np_dexprel_dx(x)
    self._check_boxcox_vals(dt, out, ref)

  @parameterized.parameters(jtu.dtypes.floating)
  def testBoxcoxLogInput(self, dt):
    x, lmbda = _boxcox_grid(dt)
    lx = jnp.log(x)
    out = _src_special._boxcox_log_input(lx, lmbda)
    ref = _np_boxcox(x, lmbda)
    self._check_boxcox_vals(dt, out, ref)

  @parameterized.parameters(jtu.dtypes.floating)
  def testBoxcoxLogInputGrad(self, dt):
    x, lmbda = _boxcox_grid(dt)
    lx = jnp.log(x)
    out_fn = _src_special._boxcox_log_input
    ref_fn = _np_boxcox_log_input_grad
    self._check_boxcox_grads(dt, lx, lmbda, out_fn, ref_fn)

  @parameterized.parameters(jtu.dtypes.floating)
  def testBoxcox(self, dt):
    x, lmbda = _boxcox_grid(dt)
    out = lsp_special.boxcox(x, lmbda)
    ref = _np_boxcox(x, lmbda)
    self._check_boxcox_vals(dt, out, ref)

  @parameterized.parameters(jtu.dtypes.floating)
  def testBoxcox1p(self, dt):
    x, lmbda = _boxcox_grid(dt, is_1p=True)
    out = lsp_special.boxcox1p(x, lmbda)
    ref = _np_boxcox(x, lmbda, is_1p=True)
    self._check_boxcox_vals(dt, out, ref)

  @parameterized.parameters(jtu.dtypes.floating)
  def testBoxcoxGrad(self, dt):
    x, lmbda = _boxcox_grid(dt)
    out_fn = lsp_special.boxcox
    ref_fn = _np_boxcox_grad
    self._check_boxcox_grads(dt, x, lmbda, out_fn, ref_fn)

  @parameterized.parameters(jtu.dtypes.floating)
  def testBoxcox1pGrad(self, dt):
    x, lmbda = _boxcox_grid(dt, is_1p=True)
    out_fn = lsp_special.boxcox1p
    ref_fn = functools.partial(_np_boxcox_grad, is_1p=True)
    self._check_boxcox_grads(dt, x, lmbda, out_fn, ref_fn)

  @parameterized.named_parameters(("", False), ("1p", True))
  def testBoxcoxGradXCancellation(self, is_1p):
    # Verifies that d/dx of boxcox / boxcox1p does not suffer catastrophic
    # cancellation when lambda * log(x) << -1 (large x, negative lambda).
    dt = jnp.float32
    x = jnp.array([100.0, 1e7, 1e8], dtype=dt)
    lmbda = jnp.array([-10.0, -2.0, -2.0], dtype=dt)
    out_fn = lsp_special.boxcox1p if is_1p else lsp_special.boxcox
    dx_out = jax.vmap(jax.grad(out_fn, argnums=0))(x, lmbda)
    dx_ref, _ = _np_boxcox_grad(x, lmbda, is_1p=is_1p)
    dx_ref = jnp.asarray(dx_ref, dtype=dt)
    self.assertAllClose(dx_out, dx_ref)

  @parameterized.named_parameters(("", False), ("1p", True))
  def testBoxcoxGradXUnderflow(self, is_1p):
    # Verifies that d/dx of boxcox / boxcox1p does not prematurely underflow
    # when x << 1 and lambda > 1 (where exp(lambda * log(x)) underflows to 0).
    dt = jnp.float32
    if is_1p:
      # 1+x in [5e-5, 8e-5]: exp(9.5*log1p(x)) underflows to 0,
      # but exp(8.5*log1p(x)) is normal in float32.
      x = jnp.linspace(5e-5, 8e-5, 4, dtype=dt) - 1
      lmbda = jnp.full_like(x, 9.5)
    else:
      x = jnp.array([1e-25, 1e-20, 1e-15], dtype=dt)
      lmbda = jnp.array([2.0, 2.0, 2.0], dtype=dt)
    out_fn = lsp_special.boxcox1p if is_1p else lsp_special.boxcox
    dx_out = jax.vmap(jax.grad(out_fn, argnums=0))(x, lmbda)
    dx_ref, _ = _np_boxcox_grad(x, lmbda, is_1p=is_1p)
    dx_ref = jnp.asarray(dx_ref, dtype=dt)
    self.assertAllClose(dx_out, dx_ref)

  def testBoxcoxAnchorPoints(self):
    # Tests anchor points (d/dx=1, d/dlmbda=0) for (inv_)boxcox(1p).
    dt = jnp.float32
    lmbdas = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=dt)
    x0 = jnp.zeros_like(lmbdas)
    x1 = jnp.ones_like(lmbdas)
    for fn, x, ref in [
        (lsp_special.boxcox, x1, x0),
        (lsp_special.boxcox1p, x0, x0),
        (lsp_special.inv_boxcox, x0, x1),
        (lsp_special.inv_boxcox1p, x0, x0),
    ]:
      self.assertAllClose(jax.vmap(fn)(x, lmbdas), ref)
      dx, dl = jax.vmap(jax.grad(fn, argnums=(0, 1)))(x, lmbdas)
      self.assertAllClose(dx, x1)
      self.assertAllClose(dl, x0)

  def testBoxcoxGradNearZero(self):
    # Verifies d/dlmbda boxcox approaches 0.5 * log(x)^2 as lambda -> 0.
    dt = jnp.float32
    l0 = jnp.array([0.0, 1e-20, -1e-20, 1e-10, -1e-10], dtype=dt)
    x = jnp.array(2.0, dtype=dt)
    out = jax.vmap(lambda l: jax.grad(lsp_special.boxcox, argnums=1)(x, l))(l0)
    ref = jnp.full_like(l0, 0.5 * (np.log(2.0) ** 2))
    self.assertAllClose(out, ref)

  def testBoxcoxHessian(self):
    # Probes 2nd-order autodiff (Hessian symmetry and finiteness).
    rtol = 1e-4 if jtu.test_device_matches(["tpu"]) else None
    for fn in [
        lsp_special.boxcox,
        lsp_special.boxcox1p,
        lsp_special.inv_boxcox,
        lsp_special.inv_boxcox1p,
    ]:
      h = jax.hessian(fn, argnums=(0, 1))(jnp.float32(2.0), jnp.float32(0.5))
      self.assertTrue(jnp.all(jnp.isfinite(h[0][0])))
      self.assertTrue(jnp.all(jnp.isfinite(h[1][1])))
      self.assertAllClose(h[0][1], h[1][0], rtol=rtol)

  @parameterized.parameters(jtu.dtypes.floating)
  def testLogrel(self, dt):
    x = _boxcox_inputs(dt, is_1p=True)
    out = _src_special._logrel(x)
    ref = _np_logrel(x)
    self._check_boxcox_vals(dt, out, ref)

  @parameterized.parameters(jtu.dtypes.floating)
  def testLogrelGrad(self, dt):
    x = _boxcox_inputs(dt, is_1p=True)
    out = jax.vmap(jax.grad(_src_special._logrel))(x)
    ref = _np_dlogrel_dx(x)
    self._check_boxcox_vals(dt, out, ref)

  @parameterized.parameters(jtu.dtypes.floating)
  def testLogrelGradCrossover(self, dt):
    # Tests crossover of d/dx logrel at +/- 0.2 across [-0.4, 0.4].
    x = jnp.linspace(dt(-0.4), dt(0.4), 201, dtype=dt)
    out = jax.vmap(jax.grad(_src_special._logrel))(x)
    ref = _np_dlogrel_dx(x)
    self._check_boxcox_vals(dt, out, ref)

  def testExprelSecondDerivativeAtOrigin(self):
    self.assertAllClose(
        jax.grad(jax.grad(lsp_special.exprel))(0.0), 1.0 / 3.0)

  def testLogrelSecondDerivativeAtOrigin(self):
    self.assertAllClose(
        jax.grad(jax.grad(_src_special._logrel))(0.0), 2.0 / 3.0)

  @parameterized.parameters(jtu.dtypes.floating)
  def testInvBoxcoxLogOutput(self, dt):
    y, lmbda = _inv_boxcox_grid(dt)
    out = _src_special._inv_boxcox_log_output(y, lmbda)
    ref = _np_inv_boxcox_log_output(y, lmbda)
    self._check_boxcox_vals(dt, out, ref)

  @parameterized.parameters(jtu.dtypes.floating)
  def testInvBoxcoxLogOutputGrad(self, dt):
    y, lmbda = _inv_boxcox_grid(dt)
    out_fn = _src_special._inv_boxcox_log_output
    ref_fn = _np_inv_boxcox_log_output_grad
    self._check_boxcox_grads(dt, y, lmbda, out_fn, ref_fn)

  @parameterized.parameters(jtu.dtypes.floating)
  def testInvBoxcox(self, dt):
    y, lmbda = _inv_boxcox_grid(dt)
    out = lsp_special.inv_boxcox(y, lmbda)
    ref = _np_inv_boxcox(y, lmbda)
    self._check_boxcox_vals(dt, out, ref)

  @parameterized.parameters(jtu.dtypes.floating)
  def testInvBoxcox1p(self, dt):
    y, lmbda = _inv_boxcox_grid(dt)
    out = lsp_special.inv_boxcox1p(y, lmbda)
    ref = _np_inv_boxcox(y, lmbda, is_1p=True)
    self._check_boxcox_vals(dt, out, ref)

  @parameterized.parameters(jtu.dtypes.floating)
  def testInvBoxcoxGrad(self, dt):
    y, lmbda = _inv_boxcox_grid(dt)
    out_fn = lsp_special.inv_boxcox
    ref_fn = _np_inv_boxcox_grad
    self._check_boxcox_grads(dt, y, lmbda, out_fn, ref_fn)

  @parameterized.parameters(jtu.dtypes.floating)
  def testInvBoxcox1pGrad(self, dt):
    y, lmbda = _inv_boxcox_grid(dt)
    out_fn = lsp_special.inv_boxcox1p
    ref_fn = _np_inv_boxcox_grad
    self._check_boxcox_grads(dt, y, lmbda, out_fn, ref_fn)

  def testBoxcoxLogInputAdditionLaw(self):
    # Assert Lie exponential homomorphism for φ = _boxcox_log_input:
    # φ(u1 + u2) = φ(u1) + φ(u2) + λ · φ(u1) · φ(u2).
    tol = 1e-4 if jtu.test_device_matches(["tpu"]) else None
    u1 = jnp.array([-1.5, -0.3, 0.0, 0.4, 1.2])[:, None]
    u2 = jnp.array([0.8, -0.2, 0.5, 0.0, -0.9])[:, None]
    lmbda = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])[None, :]
    u_stacked = jnp.stack([u1 + u2, u1, u2])
    y_sum, y1, y2 = _src_special._boxcox_log_input(u_stacked, lmbda[None])
    self.assertAllClose(y_sum, y1 + y2 + lmbda * y1 * y2, atol=tol, rtol=tol)

  def testInvBoxcoxLogOutputAdditionLaw(self):
    # Assert Lie logarithm homomorphism for φ⁻¹ = _inv_boxcox_log_output:
    # φ⁻¹(y1 + y2 + λ · y1 · y2) = φ⁻¹(y1) + φ⁻¹(y2).
    tol = 1e-4 if jtu.test_device_matches(["tpu"]) else None
    y1 = jnp.array([0.1, 0.3, 0.0, 0.5, 0.2])[:, None]
    y2 = jnp.array([0.2, 0.1, 0.4, 0.0, 0.3])[:, None]
    lmbda = jnp.array([-0.5, 0.0, 0.5, 1.5])[None, :]
    group_sum = y1 + y2 + lmbda * y1 * y2
    y_stacked = jnp.stack(jnp.broadcast_arrays(group_sum, y1, y2))
    u_sum, u1, u2 = _src_special._inv_boxcox_log_output(y_stacked, lmbda[None])
    self.assertAllClose(u_sum, u1 + u2, atol=tol, rtol=tol)

  @parameterized.parameters(jtu.dtypes.floating)
  def testBoxcoxRoundtrip(self, dt):
    x = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=dt)
    lmbda = jnp.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=dt)
    xg, lg = jnp.meshgrid(x, lmbda)
    xf, lf = xg.ravel(), lg.ravel()
    self._check_boxcox_vals(
        dt, lsp_special.inv_boxcox(
            lsp_special.boxcox(xf, lf), lf), xf)

  @parameterized.parameters(jtu.dtypes.floating)
  def testBoxcox1pRoundtrip(self, dt):
    x = jnp.array([-0.5, -0.1, 0.0, 0.1, 0.5, 2.0, 5.0], dtype=dt)
    lmbda = jnp.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=dt)
    xg, lg = jnp.meshgrid(x, lmbda)
    xf, lf = xg.ravel(), lg.ravel()
    self._check_boxcox_vals(
        dt, lsp_special.inv_boxcox1p(
            lsp_special.boxcox1p(xf, lf), lf), xf)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
