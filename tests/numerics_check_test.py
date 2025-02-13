# Copyright 2024 The JAX Authors.
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
from __future__ import annotations

from absl.testing import absltest

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src.lax.lax import _reduce_sum
from jax.experimental import numerics_check

jax.config.parse_flags_with_absl()

def _f32_to_bf16(arr: jax.Array) -> jax.Array:
  arr = lax.reduce_precision(arr, exponent_bits=8, mantissa_bits=7)
  arr = arr.astype(jnp.bfloat16)
  return arr


def _mse(arr1: jax.Array, arr2: jax.Array) -> jax.Array:
  return jnp.mean(jnp.square(arr1 - arr2.astype(arr1.dtype)))


def _g_dot_delta(g: jax.Array, arr1: jax.Array, arr2: jax.Array) -> jax.Array:
  delta = arr1 - arr2.astype(arr1.dtype)
  return jnp.sum(g * delta)

class NumericsCheckTest(jtu.JaxTestCase):
  def test_numerics_check(self):
    def matmul(x, y):
      return jnp.sum(x @ y)

    key = jax.random.key(42)
    x = jax.random.uniform(key, (128, 128), dtype=jnp.float32) * 1e7
    y = (
      jax.random.uniform(jax.random.split(key)[1], (128, 128), dtype=jnp.float32) * 1e-7
    )
    args = x, y
    check, source_metrics = numerics_check.numerics_check(matmul)
    metric_keys = source_metrics(*args)
    metrics = numerics_check.metric_keys_to_metrics(metric_keys)
    result = check(metrics, *args)
    expected = matmul(*args)
    self.assertArraysEqual(result, expected)

    metrics: numerics_check.Metrics = jax.grad(check)(metrics, *args)
    print("\n\nTrace Order:")
    numerics_check.print_metrics(metric_keys, metrics)

    pjit_1_metrics, dot_metrics, pjit_2_metrics, sum_metrics = metrics
    zero = jnp.zeros((), dtype=jnp.float32)
    ones = jnp.ones((128, 128), dtype=jnp.float32)
    one = jnp.ones((), dtype=jnp.float32)

    self.assertEqual(pjit_1_metrics, ((zero, zero), zero, ()))
    self.assertEqual(pjit_2_metrics, ((zero,), zero, ()))

    dot_in_x = _mse(ones @ y, _f32_to_bf16(ones) @ _f32_to_bf16(y))
    dot_in_y = _mse(ones @ x, _f32_to_bf16(ones) @ _f32_to_bf16(x))
    self.assertAllClose(dot_metrics[0], (dot_in_x, dot_in_y))
    dot_out = _g_dot_delta(ones, x @ y, _f32_to_bf16(x) @ _f32_to_bf16(y))
    self.assertAllClose(dot_metrics[1], dot_out)

    self.assertEqual(sum_metrics[0], (zero,))
    sum_out = _g_dot_delta(
      one,
      _reduce_sum(x @ y, axes=(0, 1)),
      _reduce_sum(_f32_to_bf16(x @ y), axes=(0, 1)),
    )
    self.assertAllClose(sum_metrics[1], sum_out)

    print("\n\nSorted by In Metrics:")
    metric_keys, metrics = numerics_check.sort_metrics_by_in_metrics(
      metric_keys, metrics
    )
    numerics_check.print_metrics(metric_keys, metrics)
    print("\n\nSorted by Out Metric:")
    metric_keys, metrics = numerics_check.sort_metrics_by_out_metric(
      metric_keys, metrics
    )
    numerics_check.print_metrics(metric_keys, metrics)

  def test_jit_numerics_check(self):
    @jax.jit
    def matmul(x, y):
      return jnp.sum(x @ y)

    key = jax.random.key(42)
    x = jax.random.uniform(key, (128, 128), dtype=jnp.float32) * 1e7
    y = (
      jax.random.uniform(jax.random.split(key)[1], (128, 128), dtype=jnp.float32) * 1e-7
    )
    args = x, y
    check, source_metrics = numerics_check.numerics_check(matmul)
    metric_keys = source_metrics(*args)
    metrics = numerics_check.metric_keys_to_metrics(metric_keys)
    result = jax.jit(check)(metrics, *args)
    expected = matmul(*args)
    self.assertArraysEqual(result, expected)

    metrics: numerics_check.Metrics = jax.jit(jax.grad(check))(metrics, *args)
    print("\n\nTrace Order:")
    numerics_check.print_metrics(metric_keys, metrics)

    pjit_1_metrics, dot_metrics, sum_metrics = metrics
    zero = jnp.zeros((), dtype=jnp.float32)
    ones = jnp.ones((128, 128), dtype=jnp.float32)
    one = jnp.ones((), dtype=jnp.float32)

    self.assertEqual(pjit_1_metrics, ((zero, zero), zero, ()))

    dot_in_x = _mse(ones @ y, _f32_to_bf16(ones) @ _f32_to_bf16(y))
    dot_in_y = _mse(ones @ x, _f32_to_bf16(ones) @ _f32_to_bf16(x))
    self.assertAllClose(dot_metrics[0], (dot_in_x, dot_in_y))
    dot_out = _g_dot_delta(ones, x @ y, _f32_to_bf16(x) @ _f32_to_bf16(y))
    self.assertAllClose(dot_metrics[1], dot_out)

    self.assertEqual(sum_metrics[0], (zero,))
    sum_out = _g_dot_delta(
      one,
      _reduce_sum(x @ y, axes=(0, 1)),
      _reduce_sum(_f32_to_bf16(x @ y), axes=(0, 1)),
    )
    self.assertAllClose(sum_metrics[1], sum_out)

    print("\n\nSorted by In Metrics:")
    metric_keys, metrics = numerics_check.sort_metrics_by_in_metrics(
      metric_keys, metrics
    )
    numerics_check.print_metrics(metric_keys, metrics)
    print("\n\nSorted by Out Metric:")
    metric_keys, metrics = numerics_check.sort_metrics_by_out_metric(
      metric_keys, metrics
    )
    numerics_check.print_metrics(metric_keys, metrics)

  def test_demo(self):
    @jax.jit
    def matmul_with_residual(x, ys):
      def layer1(x, y):
        return x + jax.nn.gelu(x @ y)

      def layer2(x, y):
        return x + jax.nn.swish(x @ y)

      def layer3(x, y):
        return x + jax.nn.sigmoid(x @ y)

      def layer4(x, y):
        return x + jax.nn.tanh(x @ y)

      for l, y in zip([layer1, layer2, layer3, layer4], ys):
        x = l(x, y)
      return jnp.sum(x)

    key = jax.random.key(42)
    x = jax.random.uniform(key, (128, 1024), dtype=jnp.float32)
    y = jax.random.uniform(jax.random.split(key)[1], (1024, 1024), dtype=jnp.float32)
    args = x, (y, y, y, y)
    check, source_metrics = numerics_check.numerics_check(matmul_with_residual)
    metric_keys = source_metrics(*args)
    metrics = numerics_check.metric_keys_to_metrics(metric_keys)
    metrics: numerics_check.Metrics = jax.jit(jax.grad(check))(metrics, *args)
    metric_keys, metrics = numerics_check.sort_metrics_by_dupe_metrics(
      metric_keys, metrics
    )
    numerics_check.print_metrics(metric_keys, metrics, normalize_out_metric=False)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
