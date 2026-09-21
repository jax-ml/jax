# Copyright 2026 The JAX Authors.
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

"""Unit tests for numerics precision testing utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp

# Under pytest, tests run against an installed wheel that does not
# include `jax.tests`, so skip before importing `jax.tests.numerics`.
if jtu.is_running_under_pytest():
  import pytest

  pytest.skip("Only runs under Bazel", allow_module_level=True)

from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64


@jtu.skip_under_pytest("Only runs under Bazel")
class UlpDiffTest(jtu.JaxTestCase):

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_ulp_diff_ftz_and_sub_ulp(self, dtype):
    dt = np.dtype(dtype).type
    tiny = float(np.finfo(dtype).tiny)
    min_subnormal = float(np.nextafter(dt(0.0), dt(1.0)))
    max_subnormal = float(np.nextafter(dt(tiny), dt(0.0)))
    one = dt(1.0)
    next_one = np.nextafter(one, dt(2.0))
    ulp_one = mpmath.mpf(float(next_one)) - mpmath.mpf(1.0)
    mant_bits = np.finfo(dtype).nmant

    def ref_val(v):
      return v if dtype == f64 else float(v)

    base_cases = [
        # (x_dtype, ref, expected_ulp_ftz_true, expected_ulp_ftz_false)
        (dt(0.0), ref_val(min_subnormal), 0.0, 1.0),
        (dt(0.0), ref_val(max_subnormal), 0.0, float((1 << mant_bits) - 1)),
        (dt(min_subnormal), ref_val(max_subnormal), 0.0,
         float((1 << mant_bits) - 2)),
        (dt(0.0), ref_val(-0.0), 0.0, 0.0),
        (dt(tiny), ref_val(0.0), 1.0, float(1 << mant_bits)),
        (dt(tiny), ref_val(min_subnormal), 1.0, float((1 << mant_bits) - 1)),
        (dt(tiny), ref_val(max_subnormal), 1.0, 1.0),
        (dt(tiny), ref_val(-tiny), 2.0, float(1 << (mant_bits + 1))),
        (one, ref_val(float(next_one)), 1.0, 1.0),
        # Fractional / real ULP cases in [1.0, 2.0)
        (one, ref_val(mpmath.mpf(1.0) + 0.25 * ulp_one), 0.25, 0.25),
        (one, ref_val(mpmath.mpf(1.0) + 0.5 * ulp_one), 0.5, 0.5),
        (next_one, ref_val(mpmath.mpf(1.0) + 0.25 * ulp_one), 0.75, 0.75),
    ]
    cases = base_cases + [
        (dt(-x), -y, exp_ftz, exp_no_ftz)
        for x, y, exp_ftz, exp_no_ftz in base_cases
    ]

    x_arr = np.array([x for x, _, _, _ in cases], dtype=dtype)
    y_arr = np.array(
        [y for _, y, _, _ in cases],
        dtype=object if dtype == f64 else np.float64,
    )
    expected_ftz = np.array([e for _, _, e, _ in cases], dtype=np.float64)
    expected_no_ftz = np.array([e for _, _, _, e in cases], dtype=np.float64)

    np.testing.assert_allclose(
        util.ulp_diff(x_arr, y_arr, dtype, ftz=True),
        expected_ftz,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        util.ulp_diff(x_arr, y_arr, dtype, ftz=False),
        expected_no_ftz,
        rtol=1e-12,
    )

    for ftz, expected in [(True, expected_ftz), (False, expected_no_ftz)]:
      _, top_k = util.eval_ulp_stats(
          x_arr,
          x_arr,
          y_arr,
          dtype,
          ftz=ftz,
          k=len(cases),
      )
      self.assertAlmostEqual(top_k[0][0], float(np.max(expected)))

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_signed_histogram_counts(self, dtype):
    dt = np.dtype(dtype).type
    center = dt(1.5)
    plus_one_ulp = np.nextafter(center, dt(2.0))
    minus_one_ulp = np.nextafter(center, dt(0.0))

    # 20 elements at +1 ULP, 15 elements at -1 ULP, 29 elements at 0 ULP.
    comp = np.array(
        [plus_one_ulp] * 20 + [minus_one_ulp] * 15 + [center] * 29, dtype=dtype
    )
    ref = np.full(64, 1.5, dtype=np.float64)

    counts_dict, top_k = util.eval_ulp_stats(
        comp, comp, ref, dtype, ftz=False, k=5
    )
    self.assertEqual(
        counts_dict,
        {"[-1, -0.5) ULP": 15, "0 ULP": 29, "(+0.5, +1] ULP": 20},
    )
    self.assertEqual(top_k[0][0], 1.0)

    hist_str = util.render_histogram_from_counts(counts_dict, 64)
    # Intermediate zero-count bins [-0.5, 0) and (0, +0.5] between [-1, -0.5)
    # and (+0.5, +1] must be included rather than skipped:
    expected_labels = [
        "[-1, -0.5) ULP:",
        "[-0.5, 0) ULP:",
        "0 ULP:",
        "(0, +0.5] ULP:",
        "(+0.5, +1] ULP:",
    ]
    hist_lines = hist_str.splitlines()
    self.assertLen(hist_lines, len(expected_labels))
    for line, expected_label in zip(hist_lines, expected_labels):
      self.assertIn(expected_label, line)

  def test_eval_ulp_stats_large_k_fallback(self):
    n = 16384 * 20
    x = np.ones(n, dtype=np.float32)
    ref = np.ones(n, dtype=np.float64)
    counts_dict, top_k = util.eval_ulp_stats(
        x, x, ref, jnp.float32, ftz=True, k=20000
    )
    self.assertEqual(counts_dict, {"0 ULP": n})
    self.assertLen(top_k, 20000)

  def test_signed_zero_check(self):
    # A function returning +0.0 where -0.0 is expected:
    def bad_neg_zero(x):
      return jnp.where(
          jnp.signbit(x) & (x == 0.0), jnp.asarray(0.0, dtype=x.dtype), x
      )

    with self.assertRaises(AssertionError) as ctx:
      util.check_unary_precision(
          self,
          bad_neg_zero,
          lambda x: x,
          lambda x: x,
          jnp.float16,
      )
    self.assertIn("Signed zero mismatch", str(ctx.exception))

    # Suppressing signed zero checks passes:
    util.check_unary_precision(
        self,
        bad_neg_zero,
        lambda x: x,
        lambda x: x,
        jnp.float16,
        check_signed_zeros=False,
    )
    util.check_unary_precision(
        self,
        bad_neg_zero,
        lambda x: x,
        lambda x: x,
        jnp.float16,
        check_signed_zeros=[(jtu.device_under_test(), {f16: False})],
    )

    # Under input_ftz=True on float64, flushing a negative subnormal input to
    # +0.0 instead of -0.0 must also be caught by the signed-zero check.
    tiny_f64 = np.finfo(np.float64).tiny

    def bad_f64_subnormal_ftz(x):
      return jnp.where(jnp.abs(x) < tiny_f64, jnp.float64(0.0), x)

    with self.assertRaises(AssertionError) as ctx_f64:
      util.check_unary_precision(
          self,
          bad_f64_subnormal_ftz,
          lambda x: x,
          lambda x: x,
          jnp.float64,
          input_ftz=True,
      )
    self.assertIn("Signed zero mismatch", str(ctx_f64.exception))

  @parameterized.parameters(True, False)
  def test_ulp_diff_jax_vs_mpmath_cross_check(self, ftz: bool):
    rng = np.random.RandomState(42)
    dt = np.float32
    finfo = np.finfo(dt)
    tiny = float(finfo.tiny)
    min_sub = float(np.nextafter(dt(0.0), dt(1.0)))
    max_sub = float(np.nextafter(dt(tiny), dt(0.0)))
    emax = finfo.maxexp - 1
    p = finfo.nmant + 1
    overflow_thresh = float(
        np.ldexp(1.0, emax + 1) - np.ldexp(0.5, emax - (p - 1))
    )

    special_vals = [
        0.0,
        -0.0,
        min_sub,
        -min_sub,
        max_sub,
        -max_sub,
        tiny,
        -tiny,
        1.0,
        -1.0,
        float(finfo.max),
        -float(finfo.max),
        overflow_thresh,
        -overflow_thresh,
        overflow_thresh * (1.0 - 1e-7),
        overflow_thresh * (1.0 + 1e-7),
        float("inf"),
        float("-inf"),
        float("nan"),
    ]

    computed_list = []
    reference_list = []

    # All pairs of special values
    for c in special_vals:
      for r in special_vals:
        computed_list.append(c)
        reference_list.append(r)

    # Random full-range inputs and perturbations
    n_rand = 500
    rand_c = jtu.rand_fullrange(rng)((n_rand,), dt)
    for c in rand_c:
      c_val = float(c)
      computed_list.extend([c_val, c_val, c_val, c_val])
      reference_list.append(c_val)
      reference_list.append(float(np.nextafter(c, dt(np.inf))))
      reference_list.append(-c_val)
      reference_list.append(float(jtu.rand_fullrange(rng)((), dt)))

    c_arr = np.array(computed_list, dtype=np.float64)
    r_arr = np.array(reference_list, dtype=np.float64)

    cpu_dev = jax.devices("cpu")[0]
    with jax.enable_x64(True), jax.default_device(cpu_dev):
      ulp_jax = np.asarray(
          util._ulp_diff_jax(
              jnp.asarray(c_arr), jnp.asarray(r_arr), dt, ftz=ftz
          )
      )

    ulp_mp = util._ulp_diff_mpmath(c_arr, r_arr, dt, ftz=ftz)

    np.testing.assert_allclose(ulp_jax, ulp_mp, rtol=1e-12, atol=1e-12)

  def test_format_worst_cases_overflow(self):
    # When y_ref in float64 exceeds float32 max (e.g. lgamma(large_f32)),
    # _format_worst_cases must not raise RuntimeWarning: overflow in cast.
    top_k = [(0.0, float("3.4e38"), float("inf"), 1e40)]
    out = util._format_worst_cases(
        top_k, np.dtype("u4"), lambda x: mpmath.mpf("1e40"), jnp.float32
    )
    self.assertIn("inf (0x7f800000)", out)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
