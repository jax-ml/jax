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

import itertools

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
config.update("jax_enable_x64", True)

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64


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
    max_float = dt(np.finfo(dtype).max)
    ulp_max = mpmath.ldexp(1, np.finfo(dtype).maxexp - 1 - mant_bits)

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
        # Continuous ULP distance across the max_float / infinity boundary
        (max_float, ref_val(mpmath.mpf(float(max_float)) + 0.5 * ulp_max),
         0.5, 0.5),
        (max_float, ref_val(mpmath.mpf(float(max_float)) + 0.75 * ulp_max),
         0.75, 0.75),
        (dt(np.inf), ref_val(mpmath.mpf(float(max_float)) + 0.5 * ulp_max),
         0.0, 0.0),
        (dt(np.inf), ref_val(mpmath.mpf(float(max_float)) + 0.25 * ulp_max),
         0.75, 0.75),
        (dt(np.inf), ref_val(float(max_float)), 1.0, 1.0),
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

  def test_one_to_two_ulp_histogram_bins(self):
    signed_ulps = jnp.array(
        [-2.0, -1.75, -1.5, -1.25, 1.25, 1.5, 1.75, 2.0], dtype=jnp.float32
    )
    bins = np.asarray(util._map_to_bins(signed_ulps))
    labels = [util._BIN_LABELS[b] for b in bins]
    expected = [
        "[-2, -1.5) ULP",
        "[-2, -1.5) ULP",
        "[-1.5, -1) ULP",
        "[-1.5, -1) ULP",
        "(+1, +1.5] ULP",
        "(+1, +1.5] ULP",
        "(+1.5, +2] ULP",
        "(+1.5, +2] ULP",
    ]
    self.assertEqual(labels, expected)

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

    ulp_mp = util.ulp_diff_mpmath(c_arr, r_arr, dt, ftz=ftz)

    np.testing.assert_allclose(ulp_jax, ulp_mp, rtol=1e-12, atol=1e-12)

  def test_format_worst_cases_overflow(self):
    # When y_ref in float64 exceeds float32 max (e.g. lgamma(large_f32)),
    # _format_worst_cases must not raise RuntimeWarning: overflow in cast.
    top_k = [(0.0, float("3.4e38"), float("inf"), 1e40)]
    out = util._format_worst_cases(
        top_k, np.dtype("u4"), lambda x: mpmath.mpf("1e40"), jnp.float32
    )
    self.assertIn("inf (0x7f800000)", out)

  def test_class_sharded_test_loader(self):
    loader = util.ClassShardedTestLoader()

    # Shard 0 perspective
    iter_0 = itertools.cycle(range(2))
    self.assertEqual(
        loader.shardTestCaseNames(iter_0, ["test_a1", "test_a2"], 0),
        ["test_a1", "test_a2"],
    )
    self.assertEqual(
        loader.shardTestCaseNames(iter_0, ["test_b1"], 0),
        [],
    )
    self.assertEqual(
        loader.shardTestCaseNames(iter_0, [], 0),
        [],
    )
    self.assertEqual(
        loader.shardTestCaseNames(iter_0, ["test_c1"], 0),
        ["test_c1"],
    )

    # Shard 1 perspective
    iter_1 = itertools.cycle(range(2))
    self.assertEqual(
        loader.shardTestCaseNames(iter_1, ["test_a1", "test_a2"], 1),
        [],
    )
    self.assertEqual(
        loader.shardTestCaseNames(iter_1, ["test_b1"], 1),
        ["test_b1"],
    )
    self.assertEqual(
        loader.shardTestCaseNames(iter_1, [], 1),
        [],
    )
    self.assertEqual(
        loader.shardTestCaseNames(iter_1, ["test_c1"], 1),
        [],
    )

  def test_eval_mpmath_multi_arg(self):
    # Test multi-argument eval_mpmath evaluation, subnormal flushing, and
    # preservation of IEEE-754 signbit for -0.0 and signed NaN.
    self.assertEqual(
        util.eval_mpmath(lambda x, y: x + y, 1.0, 2.0), mpmath.mpf(3.0)
    )
    self.assertEqual(
        util.eval_mpmath(lambda a, b, c: a * b + c, 2.0, 3.0, 4.0),
        mpmath.mpf(10.0),
    )
    self.assertTrue(
        mpmath.isnan(util.eval_mpmath(lambda x, y: x + y, float("nan"), 1.0))
    )
    self.assertTrue(
        util.eval_mpmath(lambda x: bool(np.signbit(float(x))), -0.0)
    )
    pos_nan = np.uint64(0x7FF8000000000000).view(np.float64)
    neg_nan = np.uint64(0xFFF8000000000000).view(np.float64)
    self.assertFalse(
        util.eval_mpmath(lambda x: bool(np.signbit(float(x))), pos_nan)
    )
    self.assertTrue(
        util.eval_mpmath(lambda x: bool(np.signbit(float(x))), neg_nan)
    )

  def test_check_nary_precision_binary(self):
    # Verify check_nary_precision correctly evaluates an exact binary op.
    util.check_nary_precision(
        self,
        jnp.add,
        np.add,
        lambda x, y: x + y,
        jnp.float32,
        nargs=2,
        bounds=0.5,
        max_samples=1000,
    )

  def test_make_exhaustive_chunk_nary(self):
    # Unary (nargs=1): bit patterns [0, 1, 2, 3]
    (x1,) = util._make_exhaustive_chunk(0, 4, 1, jnp.float16)
    np.testing.assert_array_equal(
        x1.view(np.uint16), np.array([0, 1, 2, 3], dtype=np.uint16)
    )

    # Binary (nargs=2) on float16 across the 65536 rollover boundary:
    # flat indices [65534, 65535, 65536, 65537] map to:
    #   arg0 = [65534, 65535, 0, 1], arg1 = [0, 0, 1, 1]
    a0, a1 = util._make_exhaustive_chunk(65534, 4, 2, jnp.float16)
    np.testing.assert_array_equal(
        a0.view(np.uint16), np.array([65534, 65535, 0, 1], dtype=np.uint16)
    )
    np.testing.assert_array_equal(
        a1.view(np.uint16), np.array([0, 0, 1, 1], dtype=np.uint16)
    )

    # Aligned binary (nargs=2) fast-path on float16:
    c0, c1 = util._make_exhaustive_chunk(65536, 2 * 65536, 2, jnp.float16)
    self.assertEqual(c0.shape, (2 * 65536,))
    self.assertEqual(c1.shape, (2 * 65536,))
    np.testing.assert_array_equal(
        c0.view(np.uint16)[:4], np.array([0, 1, 2, 3], dtype=np.uint16)
    )
    np.testing.assert_array_equal(
        c1.view(np.uint16)[[0, 65535, 65536, 131071]],
        np.array([1, 1, 2, 2], dtype=np.uint16),
    )

    # Ternary (nargs=3) on bfloat16 at flat index (1) + (2 << 16) + (3 << 32):
    flat_idx = 1 + (2 << 16) + (3 << 32)
    b0, b1, b2 = util._make_exhaustive_chunk(flat_idx, 1, 3, jnp.bfloat16)
    self.assertEqual(int(b0.view(np.uint16)[0]), 1)
    self.assertEqual(int(b1.view(np.uint16)[0]), 2)
    self.assertEqual(int(b2.view(np.uint16)[0]), 3)

  def test_check_nary_precision_signed_zero_mismatch(self):
    # Verify that a signed-zero mismatch in a binary function is detected.
    def bad_binary_neg_zero(x, y):
      return jnp.zeros_like(x)

    with self.assertRaises(AssertionError) as ctx:
      util.check_nary_precision(
          self,
          bad_binary_neg_zero,
          lambda x, y: np.full_like(x, -0.0),
          lambda x, y: mpmath.mpf(0.0),
          jnp.float16,
          nargs=2,
          max_samples=1000,
      )
    self.assertIn("Signed zero mismatch", str(ctx.exception))
    self.assertIn("inputs =", str(ctx.exception))

  def test_format_worst_cases_nary(self):
    top_k = [(1.5, (1.0, 2.0), 3.0, 3.0)]
    out = util._format_worst_cases(
        top_k,
        np.dtype("u4"),
        lambda x, y: x + y,
        jnp.float32,
    )
    self.assertIn("Inputs", out)
    self.assertIn("(1, 2)", out)

  def test_resolve_ignore_inputs_callable(self):
    pred = lambda x, y: x == y
    rule = [("cpu", {jnp.float32: pred})]
    resolved = util.resolve_ignore_inputs(rule, "cpu", jnp.float32)
    self.assertIs(resolved, pred)
    self.assertIsNone(util.resolve_ignore_inputs(None, "cpu", jnp.float32))

  def test_resolve_ignore_inputs_sequence(self):
    rule = [("cpu", {jnp.float32: [0x1234]})]
    pred = util.resolve_ignore_inputs(rule, "cpu", jnp.float32)
    self.assertTrue(callable(pred))
    self.assertTrue(pred(np.array([np.uint32(0x1234).view(np.float32)])))
    self.assertFalse(pred(np.array([np.uint32(0x5678).view(np.float32)])))

  def test_resolve_ignore_inputs_non_callable_error(self):
    rule = [("cpu", {jnp.float32: 1234})]
    with self.assertRaises(TypeError):
      util.resolve_ignore_inputs(rule, "cpu", jnp.float32)

  def test_check_nary_precision_with_ignore_inputs(self):
    # Test that an intentionally inaccurate input region is ignored via callable mask.
    def faulty_add(x, y):
      return jnp.where(x > 0.0, x + y + 10.0, x + y)

    # Without ignore_inputs, it fails.
    with self.assertRaises(AssertionError):
      util.check_nary_precision(
          self,
          faulty_add,
          np.add,
          lambda x, y: x + y,
          jnp.float32,
          nargs=2,
          bounds=0.5,
          max_samples=1000,
      )

    # With ignore_inputs callable predicate, it passes.
    util.check_nary_precision(
        self,
        faulty_add,
        np.add,
        lambda x, y: x + y,
        jnp.float32,
        nargs=2,
        bounds=0.5,
        ignore_inputs=[
            (["cpu", "gpu", "tpu"], {jnp.float32: lambda x, y: x > 0.0})
        ],
        max_samples=1000,
    )

  def test_check_precision_omitted_mpmath_fn(self):

    # Verify that check_unary_precision and check_nary_precision succeed when
    # mpmath_fn is omitted (falling back to ref_fn across all dtypes).
    util.check_unary_precision(
        self,
        jnp.negative,
        np.negative,
        dtype=jnp.float64,
        bounds=0.5,
        max_samples=100,
    )
    util.check_unary_precision(
        self,
        jnp.negative,
        np.negative,
        jnp.float64,
        bounds=0.5,
        max_samples=100,
    )
    util.check_nary_precision(
        self,
        jnp.add,
        np.add,
        dtype=jnp.float64,
        nargs=2,
        bounds=0.5,
        max_samples=100,
    )
    util.check_nary_precision(
        self,
        jnp.add,
        np.add,
        jnp.float64,
        nargs=2,
        bounds=0.5,
        max_samples=100,
    )

  def test_register_benchmark_nary(self):
    util.register_benchmark(jnp.add, nargs=2, name="test_add_nary")

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_with_ulp_neighbors(self, dtype):
    dt = np.dtype(dtype).type
    uint_dtype = np.dtype(f"u{np.dtype(dtype).itemsize}")
    finfo = np.finfo(dtype)
    min_sub = np.nextafter(dt(0.0), dt(1.0))
    max_float = dt(finfo.max)

    # Expanding around +0.0 and -0.0 with radius 2 must retain both signed
    # zeros and step into positive and negative subnormals.
    zeros_nb = util._with_ulp_neighbors([0.0, -0.0], dtype, radius=2)
    zeros_bits = set(zeros_nb.view(uint_dtype).tolist())
    self.assertIn(int(dt(0.0).view(uint_dtype)), zeros_bits)
    self.assertIn(int(dt(-0.0).view(uint_dtype)), zeros_bits)
    self.assertIn(int(min_sub.view(uint_dtype)), zeros_bits)
    self.assertIn(int((-min_sub).view(uint_dtype)), zeros_bits)

    # Expanding around max_float and inf must step across the max_float / inf
    # boundary without wrapping into NaN or negative values.
    edge_nb = util._with_ulp_neighbors(
        [max_float, float("inf"), float("-inf"), float("nan")], dtype, radius=2
    )
    self.assertTrue(np.any(np.isnan(edge_nb)))
    finite_or_inf = edge_nb[~np.isnan(edge_nb)]
    self.assertIn(float(max_float), finite_or_inf.astype(np.float64))
    self.assertIn(
        float(np.nextafter(max_float, dt(0.0))),
        finite_or_inf.astype(np.float64),
    )
    self.assertTrue(np.any(np.isposinf(finite_or_inf)))
    self.assertTrue(np.any(np.isneginf(finite_or_inf)))

    # Bit patterns (excluding NaN) must be unique.
    non_nan_bits = finite_or_inf.view(uint_dtype).tolist()
    self.assertLen(non_nan_bits, len(set(non_nan_bits)))

    # Expanding around +inf alone must step down into max_float without wrapping
    # into negative finite values.
    pos_inf_nb = util._with_ulp_neighbors([float("inf")], dtype, radius=2)
    finite_from_pos_inf = pos_inf_nb[np.isfinite(pos_inf_nb)]
    self.assertTrue(np.all(finite_from_pos_inf >= 0.0))
    self.assertIn(float(max_float), finite_from_pos_inf.astype(np.float64))

    # Under a small max_points budget, the highest-priority center must retain
    # its immediate +-1 ULP neighbors even when many centers follow.
    lead = dt(3.25)
    small_nb = util._with_ulp_neighbors(
        [float(lead)] + [float(i) for i in range(10, 100)],
        dtype,
        max_points=12,
        radius=2,
    )
    small_bits = set(small_nb.view(uint_dtype).tolist())
    self.assertIn(int(lead.view(uint_dtype)), small_bits)
    self.assertIn(
        int(np.nextafter(lead, dt(np.inf)).view(uint_dtype)), small_bits
    )
    self.assertIn(
        int(np.nextafter(lead, dt(-np.inf)).view(uint_dtype)), small_bits
    )

  def test_interesting_points_needle_detection_unary(self):
    # A defect at a single float64 ULP neighbor of pi (common interesting point)
    # and at a custom interesting point is caught even with a small sample budget.
    pi_next = np.nextafter(np.float64(np.pi), np.float64(np.inf))

    def buggy_at_pi_neighbor(x):
      return jnp.where(x == pi_next, x + 1e-12, x)

    with self.assertRaises(AssertionError):
      util.check_unary_precision(
          self,
          buggy_at_pi_neighbor,
          lambda x: x,
          lambda x: x,
          jnp.float64,
          bounds=0.5,
          max_samples=4096,
      )

    custom_pt = 12.3456789
    custom_next = np.nextafter(np.float64(custom_pt), np.float64(-np.inf))

    def buggy_at_custom_point(x):
      return jnp.where(x == custom_next, x + 1e-12, x)

    with self.assertRaises(AssertionError):
      util.check_unary_precision(
          self,
          buggy_at_custom_point,
          lambda x: x,
          lambda x: x,
          jnp.float64,
          bounds=0.5,
          max_samples=4096,
          interesting_points=[custom_pt],
      )

  def test_interesting_points_needle_detection_binary(self):
    # A defect at a binary pair adjacent to a custom 2-tuple is
    # caught during non-exhaustive binary testing.
    x_bad = np.nextafter(np.float32(7.25), np.float32(np.inf))
    y_bad = np.nextafter(np.float32(-3.5), np.float32(-np.inf))

    def buggy_binary(x, y):
      return jnp.where((x == x_bad) & (y == y_bad), x + y + 1.0, x + y)

    with self.assertRaises(AssertionError):
      util.check_nary_precision(
          self,
          buggy_binary,
          np.add,
          lambda x, y: x + y,
          jnp.float32,
          nargs=2,
          bounds=0.5,
          max_samples=4096,
          interesting_points=[(7.25, -3.5)],
      )

  def test_f16_default_non_ftz(self):
    # float16 defaults to non-FTZ mode across all platforms, while other
    # floating-point dtypes default to FTZ.
    self.assertFalse(util._default_ftz(jnp.float16))
    self.assertFalse(util._default_ftz(np.float16))
    self.assertTrue(util._default_ftz(jnp.bfloat16))
    self.assertTrue(util._default_ftz(jnp.float32))
    self.assertTrue(util._default_ftz(jnp.float64))

    # _resolve_override falls back to _default_ftz for float16 when omitted.
    self.assertFalse(
        util._resolve_override(
            None, "tpu", jnp.float16, util._default_ftz(jnp.float16)
        )
    )
    self.assertFalse(
        util._resolve_override(
            [("gpu", {jnp.bfloat16: False})],
            "gpu",
            jnp.float16,
            util._default_ftz(jnp.float16),
        )
    )
    # Explicit scalar override is respected.
    self.assertTrue(
        util._resolve_override(
            True, "tpu", jnp.float16, util._default_ftz(jnp.float16)
        )
    )


if __name__ == "__main__":
  absltest.main(testLoader=util.ClassShardedTestLoader())
