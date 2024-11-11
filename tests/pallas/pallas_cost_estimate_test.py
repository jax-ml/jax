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
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas import cost_estimate


config.parse_flags_with_absl()


class PallasCostEstimateTest(jtu.JaxTestCase):

  def test_exp_add(self):
    def exp_add(x, y):
      return jnp.exp(x + y)
    cost = cost_estimate.cost_estimate(exp_add,
                                       jnp.ones(10, dtype=jnp.float32),
                                       jnp.ones(10, dtype=jnp.float32))
    self.assertEqual(cost.flops, 10)
    self.assertEqual(cost.transcendentals, 10)
    self.assertEqual(cost.bytes_accessed, 4 * 30)

  def test_very_large_matmul(self):
    def matmul(a, b):
      return a @ b
    m, k, n = 400_000, 800_000, 900_000
    cost = cost_estimate.cost_estimate(
        matmul,
        jax.ShapeDtypeStruct((m, k), jnp.bfloat16),
        jax.ShapeDtypeStruct((k, n), jnp.bfloat16))
    self.assertEqual(cost.flops, 2*m*k*n)
    self.assertEqual(cost.transcendentals, 0)
    self.assertEqual(cost.bytes_accessed, 2*(m*k + n*k + m*n))

  def test_batched_matmul(self):
    def matmul(a, b):
      return jnp.matmul(a, b)
    b, m, k, n = 7, 37, 91, 23
    cost = cost_estimate.cost_estimate(
        matmul,
        jax.ShapeDtypeStruct((b, m, k), jnp.float32),
        jax.ShapeDtypeStruct((b, k, n), jnp.float32))
    self.assertEqual(cost.flops, 2*b*m*k*n)
    self.assertEqual(cost.transcendentals, 0)
    self.assertEqual(cost.bytes_accessed, 4*(b*m*k + b*n*k + b*m*n))

  def test_attention(self):
    qk_dim = 16
    v_dim = 4
    kv_len = 128
    q_len = 64
    def attention(q, k, v):
      return jax.nn.softmax(q @ k.T, axis=-1) @ v
    cost = cost_estimate.cost_estimate(
        attention,
        jnp.zeros((q_len, qk_dim), dtype=jnp.float32),
        jnp.zeros((kv_len, qk_dim), dtype=jnp.float32),
        jnp.zeros((kv_len, v_dim), dtype=jnp.float32))
    qk_cost = 2 * q_len * kv_len * qk_dim
    v_cost = 2 * q_len * kv_len * v_dim
    softmax_flops = kv_len * q_len
    self.assertEqual(cost.flops, qk_cost + v_cost + 2 * softmax_flops + q_len)
    self.assertEqual(cost.transcendentals, softmax_flops)
    input_bytes = q_len * qk_dim + kv_len * qk_dim + kv_len * v_dim
    output_bytes = q_len * v_dim
    self.assertEqual(cost.bytes_accessed, 4 * (input_bytes + output_bytes))

  @parameterized.parameters(
      (1, 0), (7, 5), (8, 4), (9, 5)
  )
  def test_integer_pow(self, power, expected_flops_per_element):
    cost = cost_estimate.cost_estimate(lambda x: lax.integer_pow(x, power),
                                       jnp.ones(10, dtype=jnp.float32))
    self.assertEqual(cost.flops, 10 * expected_flops_per_element)
    self.assertEqual(cost.transcendentals, 0)
    self.assertEqual(cost.bytes_accessed, 80)

if __name__ == "__main__":
  absltest.main()
