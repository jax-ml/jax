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
"""Tests for cost_estimate.py."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl


class CostEstimateTest(absltest.TestCase):
    """Tests for Pallas cost estimation."""

    def test_dot_general_single_contracting_dim(self):
        """Test FLOP counting with single contracting dimension."""

        def matmul(x, y):
            return jnp.einsum("mk,kn->mn", x, y)

        x = jax.random.normal(jax.random.key(0), (64, 128))
        y = jax.random.normal(jax.random.key(1), (128, 256))

        xla_flops = jax.jit(matmul).lower(x, y).compile().cost_analysis()["flops"]
        pl_flops = pl.estimate_cost(matmul, x, y).flops

        self.assertEqual(xla_flops, pl_flops)

    def test_dot_general_multiple_contracting_dims(self):
        """Test FLOP counting with multiple contracting dimensions.

        This is a regression test for https://github.com/jax-ml/jax/issues/33388
        where FLOPs were incorrectly doubled for each contracting dimension.
        """

        def test(x, y):
            return jnp.einsum("...mk,...kn->mn", x, y)

        x = jax.random.normal(jax.random.key(0), (2, 64, 128))
        y = jax.random.normal(jax.random.key(1), (2, 128, 256))

        xla_flops = jax.jit(test).lower(x, y).compile().cost_analysis()["flops"]
        pl_flops = pl.estimate_cost(test, x, y).flops

        # Expected: 64 * 256 * (2 * 128) * 2 = 8,388,608
        expected = 64 * 256 * 2 * 128 * 2

        self.assertEqual(xla_flops, expected)
        self.assertEqual(pl_flops, expected)


if __name__ == "__main__":
    absltest.main()
