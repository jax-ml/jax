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

"""Tests for evaluate_sampler_bias."""

from absl.testing import absltest
from jax.tools import evaluate_sampler_bias


class EvaluateSamplerBiasTest(absltest.TestCase):

  def test_parse_value_string(self):
    self.assertTrue(evaluate_sampler_bias.parse_value_string("true"))
    self.assertFalse(evaluate_sampler_bias.parse_value_string("false"))
    self.assertEqual(evaluate_sampler_bias.parse_value_string("42"), 42)
    self.assertEqual(evaluate_sampler_bias.parse_value_string("3.14"), 3.14)
    self.assertEqual(evaluate_sampler_bias.parse_value_string("exact"), "exact")

  def test_parse_extra_kwargs(self):
    args = ["--method=exact", "--a", "2.0", "--flag"]
    kwargs = evaluate_sampler_bias.parse_extra_kwargs(args)
    self.assertEqual(kwargs["method"], "exact")
    self.assertEqual(kwargs["a"], 2.0)
    self.assertTrue(kwargs["flag"])

  def test_get_scipy_distribution(self):
    dist_gamma = evaluate_sampler_bias.get_scipy_distribution("gamma", {"a": 2.0})
    self.assertIsNotNone(dist_gamma)
    self.assertAlmostEqual(dist_gamma.ppf(0.5), 1.67834699, places=4)

    dist_norm = evaluate_sampler_bias.get_scipy_distribution("normal", {})
    self.assertAlmostEqual(dist_norm.ppf(0.5), 0.0)

    dist_poisson = evaluate_sampler_bias.get_scipy_distribution("poisson", {"lam": 5.0})
    self.assertIsNotNone(dist_poisson)
    self.assertEqual(dist_poisson.ppf(0.5), 5.0)

  def test_evaluate_bias_gamma(self):
    res = evaluate_sampler_bias.evaluate_bias(
        sampler_name="gamma",
        kwargs={"a": 2.0, "method": "exact"},
        num_samples=100000,
        seed=0,
        dtype_str="float32",
        quantiles=[0.25, 0.5, 0.75],
    )
    self.assertEqual(res.sampler_name, "gamma")
    self.assertIsNotNone(res.results)
    self.assertLen(res.results, 3)
    self.assertLess(res.max_abs_error, 0.05)

  def test_evaluate_bias_normal(self):
    res = evaluate_sampler_bias.evaluate_bias(
        sampler_name="normal",
        kwargs={},
        num_samples=100000,
        seed=42,
        dtype_str="float32",
        quantiles=[0.1, 0.5, 0.9],
    )
    self.assertEqual(res.sampler_name, "normal")
    self.assertLess(res.max_abs_error, 0.05)

  def test_evaluate_bias_poisson(self):
    res = evaluate_sampler_bias.evaluate_bias(
        sampler_name="poisson",
        kwargs={"lam": 5.0},
        num_samples=100000,
        seed=42,
        dtype_str="int32",
        quantiles=[0.1, 0.5, 0.9],
    )
    self.assertEqual(res.sampler_name, "poisson")
    self.assertIsNotNone(res.results)
    self.assertLen(res.results, 3)
    self.assertLess(res.max_abs_error, 1.0)
    self.assertLess(res.max_z_score, 4.0)

  def test_format_table_output(self):
    res = evaluate_sampler_bias.evaluate_bias(
        sampler_name="uniform",
        kwargs={"minval": 0.0, "maxval": 1.0},
        num_samples=1000,
        seed=0,
        dtype_str="float32",
        quantiles=[0.25, 0.75],
    )
    tbl = evaluate_sampler_bias.format_table_output(res)
    self.assertIn("Sampler Bias Evaluation: jax.random.uniform", tbl)
    self.assertIn("Quantile", tbl)


if __name__ == "__main__":
  absltest.main()
