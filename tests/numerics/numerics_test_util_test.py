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
import numpy as np

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64


@jtu.skip_under_pytest("Only runs under Bazel")
class UlpDiffTest(jtu.JaxTestCase):

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_ulp_diff_ftz_and_signed_zeros(self, dtype):
    dt = np.dtype(dtype).type
    tiny = np.finfo(dtype).tiny
    min_subnormal = np.nextafter(dt(0.0), dt(1.0))
    max_subnormal = np.nextafter(tiny, dt(0.0))
    one = dt(1.0)
    next_one = np.nextafter(one, dt(2.0))
    mant_bits = np.finfo(dtype).nmant

    base_cases = [
        # (x, y, expected_ulp_ftz_true, expected_ulp_ftz_false)
        (dt(0.0), min_subnormal, 0, 1),
        (dt(0.0), max_subnormal, 0, (1 << mant_bits) - 1),
        (min_subnormal, max_subnormal, 0, (1 << mant_bits) - 2),
        (dt(0.0), dt(-0.0), 1, 1),
        (min_subnormal, -min_subnormal, 1, 3),
        (min_subnormal, dt(-0.0), 1, 2),
        (tiny, dt(0.0), 1, 1 << mant_bits),
        (tiny, min_subnormal, 1, (1 << mant_bits) - 1),
        (tiny, max_subnormal, 1, 1),
        (tiny, dt(-0.0), 2, (1 << mant_bits) + 1),
        (tiny, -min_subnormal, 2, (1 << mant_bits) + 2),
        (tiny, -tiny, 3, (1 << (mant_bits + 1)) + 1),
        (one, next_one, 1, 1),
    ]
    cases = base_cases + [
        (-x, -y, exp_ftz, exp_no_ftz)
        for x, y, exp_ftz, exp_no_ftz in base_cases
    ]

    x_arr = np.array([x for x, _, _, _ in cases], dtype=dtype)
    y_arr = np.array([y for _, y, _, _ in cases], dtype=dtype)
    expected_ftz = np.array([e for _, _, e, _ in cases], dtype=np.uint64)
    expected_no_ftz = np.array([e for _, _, _, e in cases], dtype=np.uint64)

    np.testing.assert_array_equal(
        util.ulp_diff(x_arr, y_arr, dtype, ftz=True), expected_ftz
    )
    np.testing.assert_array_equal(
        util.ulp_diff(x_arr, y_arr, dtype, ftz=False), expected_no_ftz
    )

    for ftz, expected in [(True, expected_ftz), (False, expected_no_ftz)]:
      _, top_k = util.eval_ulp_stats(
          x_arr, x_arr, y_arr, dtype, ftz=ftz, k=len(cases)
      )
      self.assertEqual(top_k[0][0], int(np.max(expected)))

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_signed_histogram_counts(self, dtype):
    dt = np.dtype(dtype).type
    one = dt(1.0)
    plus_one_ulp = np.nextafter(one, dt(2.0))
    minus_one_ulp = np.nextafter(one, dt(0.0))

    # 20 elements at +1 ULP, 15 elements at -1 ULP, 29 elements at 0 ULP.
    comp = np.array(
        [plus_one_ulp] * 20 + [minus_one_ulp] * 15 + [one] * 29, dtype=dtype
    )
    ref = np.full(64, one, dtype=dtype)

    counts_dict, _ = util.eval_ulp_stats(
        ref, comp, ref, dtype, ftz=False, max_bincount=100, k=5
    )
    self.assertEqual(counts_dict, {-1: 15, 0: 29, 1: 20})

    hist_str = util.render_histogram_from_counts(
        {-1: 15, 0: 29, 1: 20, 42: 2, 100000: 1}, 67
    )
    self.assertIn("-1 ULP:", hist_str)
    self.assertIn("0 ULP:", hist_str)
    self.assertIn("+1 ULP:", hist_str)
    self.assertIn("[+10, +100) ULP:", hist_str)
    self.assertIn(">=+100000 ULP:", hist_str)

  def test_map_to_31_bins_matches_histogram_bin(self):
    test_ulps = [
        0, 1, 9, 10, 11, 99, 100, 101, 999, 1000, 1001,
        9999, 10000, 10001, 99999, 100000, 100001, 250000,
    ]
    cpu_dev = jax.devices("cpu")[0]
    with jax.enable_x64(True), jax.default_device(cpu_dev):
      ulp_arr = jnp.array(test_ulps, dtype=jnp.uint64)
      for sign in [True, False]:
        pos_arr = jnp.full_like(ulp_arr, sign, dtype=jnp.bool_)
        bins = np.asarray(util._map_to_31_bins(ulp_arr, pos_arr))
        for u_val, bin_idx in zip(test_ulps, bins):
          signed_val = u_val if sign else -u_val
          rep_val = util._BIN_REPRESENTATIVES[int(bin_idx)]
          self.assertEqual(
              util._histogram_bin(signed_val),
              util._histogram_bin(rep_val),
              f"Mismatch at signed_val={signed_val} (bin={bin_idx}, rep={rep_val})",
          )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
