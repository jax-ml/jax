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

"""Tests for jax.experimental.fused_linear_cross_entropy."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental.fused_linear_cross_entropy import (
    fused_linear_cross_entropy_loss,
)
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


# ---------------------------------------------------------------------------
# Reference (naive) implementation
# ---------------------------------------------------------------------------

def _naive_loss(x, w, labels):
  """Cross-entropy via full [N, V] logits — reference only."""
  logits = x.astype(jnp.float32) @ w.astype(jnp.float32).T  # [N, V]
  log_z = logsumexp(logits, axis=-1)
  N = x.shape[0]
  target_logit = logits[jnp.arange(N), labels]
  return jnp.mean(-target_logit + log_z)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class FusedLinearCrossEntropyTest(jtu.JaxTestCase):

  # ------------------------------------------------------------------
  # Forward accuracy
  # ------------------------------------------------------------------

  @parameterized.named_parameters(
      dict(testcase_name="tiny", N=4, D=8, V=16, chunk_size=4),
      dict(testcase_name="small_exact", N=8, D=16, V=32, chunk_size=8),
      dict(testcase_name="small_remainder", N=8, D=16, V=30, chunk_size=8),
      dict(testcase_name="V_lt_chunk", N=4, D=8, V=5, chunk_size=16),
      dict(testcase_name="V_eq_chunk", N=4, D=8, V=16, chunk_size=16),
      dict(testcase_name="medium", N=32, D=64, V=256, chunk_size=64),
      dict(testcase_name="one_chunk", N=8, D=16, V=4, chunk_size=128),
      dict(testcase_name="N1", N=1, D=8, V=16, chunk_size=4),
  )
  def test_forward_float32(self, N, D, V, chunk_size):
    key = jax.random.key(0)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    fused = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size)
    naive = _naive_loss(x, w, labels)

    np.testing.assert_allclose(fused, naive, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      dict(testcase_name="small", N=8, D=16, V=32, chunk_size=8),
      dict(testcase_name="remainder", N=8, D=16, V=30, chunk_size=8),
      dict(testcase_name="medium", N=32, D=64, V=256, chunk_size=64),
  )
  def test_forward_bfloat16(self, N, D, V, chunk_size):
    key = jax.random.key(42)
    x = jax.random.normal(key, (N, D), dtype=jnp.bfloat16)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.bfloat16)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    fused = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size)
    naive = _naive_loss(x, w, labels)

    # bfloat16 has ~2 decimal digits of precision; use generous tolerances.
    np.testing.assert_allclose(
        float(fused), float(naive), rtol=1e-2, atol=1e-2
    )

  # ------------------------------------------------------------------
  # Gradient accuracy
  # ------------------------------------------------------------------

  @parameterized.named_parameters(
      dict(testcase_name="tiny", N=4, D=8, V=16, chunk_size=4),
      dict(testcase_name="exact", N=8, D=16, V=32, chunk_size=8),
      dict(testcase_name="remainder", N=8, D=16, V=30, chunk_size=8),
      dict(testcase_name="V_lt_chunk", N=4, D=8, V=5, chunk_size=16),
      dict(testcase_name="medium", N=16, D=32, V=128, chunk_size=32),
  )
  def test_gradients_x_float32(self, N, D, V, chunk_size):
    key = jax.random.key(7)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    dx_fused = jax.grad(
        lambda x_: fused_linear_cross_entropy_loss(x_, w, labels, chunk_size=chunk_size)
    )(x)
    dx_naive = jax.grad(lambda x_: _naive_loss(x_, w, labels))(x)

    np.testing.assert_allclose(dx_fused, dx_naive, rtol=1e-4, atol=1e-5)

  @parameterized.named_parameters(
      dict(testcase_name="tiny", N=4, D=8, V=16, chunk_size=4),
      dict(testcase_name="exact", N=8, D=16, V=32, chunk_size=8),
      dict(testcase_name="remainder", N=8, D=16, V=30, chunk_size=8),
      dict(testcase_name="medium", N=16, D=32, V=128, chunk_size=32),
  )
  def test_gradients_w_float32(self, N, D, V, chunk_size):
    key = jax.random.key(13)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    dw_fused = jax.grad(
        lambda w_: fused_linear_cross_entropy_loss(x, w_, labels, chunk_size=chunk_size),
    )(w)
    dw_naive = jax.grad(lambda w_: _naive_loss(x, w_, labels))(w)

    np.testing.assert_allclose(dw_fused, dw_naive, rtol=1e-4, atol=1e-5)

  def test_gradients_both_float32(self):
    """value_and_grad returns correct loss and gradients simultaneously."""
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(99)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    def fused_fn(x_, w_):
      return fused_linear_cross_entropy_loss(x_, w_, labels, chunk_size=chunk_size)

    def naive_fn(x_, w_):
      return _naive_loss(x_, w_, labels)

    (loss_f, (dx_f, dw_f)) = jax.value_and_grad(fused_fn, argnums=(0, 1))(x, w)
    (loss_n, (dx_n, dw_n)) = jax.value_and_grad(naive_fn, argnums=(0, 1))(x, w)

    np.testing.assert_allclose(loss_f, loss_n, rtol=1e-5)
    np.testing.assert_allclose(dx_f, dx_n, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dw_f, dw_n, rtol=1e-4, atol=1e-5)

  def test_gradients_bfloat16(self):
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(3)
    x = jax.random.normal(key, (N, D), dtype=jnp.bfloat16)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.bfloat16)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    dx_fused = jax.grad(
        lambda x_: fused_linear_cross_entropy_loss(x_, w, labels, chunk_size=chunk_size)
    )(x)
    dx_naive = jax.grad(lambda x_: _naive_loss(x_, w, labels))(x)

    # Cast both to float32 for comparison (bfloat16 has low precision)
    np.testing.assert_allclose(
        dx_fused.astype(jnp.float32),
        dx_naive.astype(jnp.float32),
        rtol=5e-2,
        atol=5e-2,
    )

  # ------------------------------------------------------------------
  # JIT compatibility
  # ------------------------------------------------------------------

  def test_jit_forward(self):
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(5)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    jit_fn = jax.jit(
        lambda x_, w_: fused_linear_cross_entropy_loss(
            x_, w_, labels, chunk_size=chunk_size
        )
    )
    fused = jit_fn(x, w)
    naive = _naive_loss(x, w, labels)
    np.testing.assert_allclose(fused, naive, rtol=1e-5, atol=1e-5)

  def test_jit_grad(self):
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(6)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    jit_grad = jax.jit(
        jax.grad(
            lambda x_: fused_linear_cross_entropy_loss(
                x_, w, labels, chunk_size=chunk_size
            )
        )
    )
    dx_fused = jit_grad(x)
    dx_naive = jax.grad(lambda x_: _naive_loss(x_, w, labels))(x)
    np.testing.assert_allclose(dx_fused, dx_naive, rtol=1e-4, atol=1e-5)

  # ------------------------------------------------------------------
  # Edge cases
  # ------------------------------------------------------------------

  def test_V_equals_one(self):
    """Single-class vocabulary: loss should be near zero."""
    N, D, V, chunk_size = 4, 8, 1, 4
    key = jax.random.key(10)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jnp.zeros((N,), dtype=jnp.int32)  # only class 0

    fused = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size)
    naive = _naive_loss(x, w, labels)
    np.testing.assert_allclose(fused, naive, rtol=1e-5, atol=1e-5)
    # With V=1, softmax is trivially 1 at label 0, so loss is 0.
    np.testing.assert_allclose(fused, 0.0, atol=1e-5)

  def test_chunk_size_larger_than_V(self):
    """chunk_size > V: single chunk with partial masking."""
    N, D, V, chunk_size = 4, 8, 7, 64
    key = jax.random.key(11)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    fused = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size)
    naive = _naive_loss(x, w, labels)
    np.testing.assert_allclose(fused, naive, rtol=1e-5, atol=1e-5)

  def test_single_sample(self):
    """N=1 edge case."""
    N, D, V, chunk_size = 1, 8, 16, 4
    key = jax.random.key(12)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    fused = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size)
    naive = _naive_loss(x, w, labels)
    np.testing.assert_allclose(fused, naive, rtol=1e-5, atol=1e-5)

  def test_chunk_size_1(self):
    """Degenerate chunk_size=1: one vocab entry per scan step."""
    N, D, V, chunk_size = 4, 8, 8, 1
    key = jax.random.key(14)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    fused = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size)
    naive = _naive_loss(x, w, labels)
    np.testing.assert_allclose(fused, naive, rtol=1e-5, atol=1e-5)

  def test_gradient_x_chunk_size_1(self):
    N, D, V, chunk_size = 4, 8, 8, 1
    key = jax.random.key(15)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    dx_f = jax.grad(
        lambda x_: fused_linear_cross_entropy_loss(x_, w, labels, chunk_size=chunk_size)
    )(x)
    dx_n = jax.grad(lambda x_: _naive_loss(x_, w, labels))(x)
    np.testing.assert_allclose(dx_f, dx_n, rtol=1e-4, atol=1e-5)

  def test_large_vocab_forward(self):
    """Larger V to exercise multi-chunk behaviour."""
    N, D, V, chunk_size = 16, 64, 1024, 128
    key = jax.random.key(20)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    fused = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size)
    naive = _naive_loss(x, w, labels)
    np.testing.assert_allclose(fused, naive, rtol=1e-4, atol=1e-4)

  def test_large_vocab_grad(self):
    N, D, V, chunk_size = 8, 32, 512, 64
    key = jax.random.key(21)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    dx_f = jax.grad(
        lambda x_: fused_linear_cross_entropy_loss(x_, w, labels, chunk_size=chunk_size)
    )(x)
    dx_n = jax.grad(lambda x_: _naive_loss(x_, w, labels))(x)
    np.testing.assert_allclose(dx_f, dx_n, rtol=1e-4, atol=1e-4)

  def test_default_chunk_size(self):
    """Calling without chunk_size uses the default (4096)."""
    N, D, V = 4, 8, 16
    key = jax.random.key(30)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    fused = fused_linear_cross_entropy_loss(x, w, labels)
    naive = _naive_loss(x, w, labels)
    np.testing.assert_allclose(fused, naive, rtol=1e-5, atol=1e-5)

  def test_invalid_chunk_size_raises(self):
    x = jnp.ones((4, 8))
    w = jnp.ones((16, 8))
    labels = jnp.zeros((4,), dtype=jnp.int32)
    with self.assertRaises(ValueError):
      fused_linear_cross_entropy_loss(x, w, labels, chunk_size=0)


if __name__ == "__main__":
  absltest.main()
