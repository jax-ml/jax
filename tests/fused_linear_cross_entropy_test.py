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
from jax.nn import logsumexp
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

  # ==================================================================
  # Feature 1: shift parameter
  # ==================================================================

  def test_shift_1_flat_input(self):
    """shift=1 on a flat [T, D] input matches manual slice."""
    T, D, V, chunk_size = 16, 8, 32, 8
    key = jax.random.key(100)
    x = jax.random.normal(key, (T, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (T,), 0, V)

    # fused with shift=1
    fused = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size, shift=1)
    # naive: manually slice
    naive = _naive_loss(x[:-1], w, labels[1:])

    np.testing.assert_allclose(fused, naive, rtol=1e-5, atol=1e-5)

  def test_shift_1_batched_input(self):
    """shift=1 on a batched [B, T, D] input matches manual slice + reshape."""
    B, T, D, V, chunk_size = 3, 10, 8, 32, 8
    key = jax.random.key(101)
    x = jax.random.normal(key, (B, T, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (B, T), 0, V)

    fused = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size, shift=1)
    # naive: slice then flatten
    x_s = x[:, :-1, :].reshape(-1, D)
    labels_s = labels[:, 1:].reshape(-1)
    naive = _naive_loss(x_s, w, labels_s)

    np.testing.assert_allclose(fused, naive, rtol=1e-5, atol=1e-5)

  def test_shift_greater_than_1(self):
    """shift=2 slices off two positions."""
    T, D, V, chunk_size = 12, 8, 16, 4
    key = jax.random.key(102)
    x = jax.random.normal(key, (T, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (T,), 0, V)

    fused = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size, shift=2)
    naive = _naive_loss(x[:-2], w, labels[2:])
    np.testing.assert_allclose(fused, naive, rtol=1e-5, atol=1e-5)

  def test_shift_0_unchanged(self):
    """shift=0 is a no-op."""
    N, D, V, chunk_size = 8, 8, 16, 4
    key = jax.random.key(103)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    fused_shift0 = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size, shift=0)
    fused_no_shift = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size)
    np.testing.assert_allclose(fused_shift0, fused_no_shift, rtol=0, atol=0)

  def test_shift_gradient_x(self):
    """Gradients w.r.t. x with shift=1 are correct."""
    T, D, V, chunk_size = 12, 8, 16, 4
    key = jax.random.key(104)
    x = jax.random.normal(key, (T, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (T,), 0, V)

    dx_fused = jax.grad(
        lambda x_: fused_linear_cross_entropy_loss(x_, w, labels, chunk_size=chunk_size, shift=1)
    )(x)
    # Naive: gradient of loss w.r.t. x, accounting for the slice x[:-1].
    # jax.grad will correctly give zero gradient for x[-1] since it's sliced off.
    dx_naive = jax.grad(
        lambda x_: _naive_loss(x_[:-1], w, labels[1:])
    )(x)

    np.testing.assert_allclose(dx_fused, dx_naive, rtol=1e-4, atol=1e-5)

  def test_shift_gradient_w(self):
    """Gradients w.r.t. w with shift=1 are correct."""
    T, D, V, chunk_size = 12, 8, 16, 4
    key = jax.random.key(105)
    x = jax.random.normal(key, (T, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (T,), 0, V)

    dw_fused = jax.grad(
        lambda w_: fused_linear_cross_entropy_loss(x, w_, labels, chunk_size=chunk_size, shift=1)
    )(w)
    dw_naive = jax.grad(lambda w_: _naive_loss(x[:-1], w_, labels[1:]))(w)

    np.testing.assert_allclose(dw_fused, dw_naive, rtol=1e-4, atol=1e-5)

  def test_shift_invalid_raises(self):
    x = jnp.ones((8, 8))
    w = jnp.ones((16, 8))
    labels = jnp.zeros((8,), dtype=jnp.int32)
    with self.assertRaises(ValueError):
      fused_linear_cross_entropy_loss(x, w, labels, shift=-1)

  # ==================================================================
  # Feature 2: vocab_sort_indices
  # ==================================================================

  def test_vocab_sort_identity_permutation(self):
    """Identity permutation is a no-op."""
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(200)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    identity = jnp.arange(V)
    fused_sorted = fused_linear_cross_entropy_loss(
        x, w, labels, chunk_size=chunk_size, vocab_sort_indices=identity
    )
    fused_plain = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size)
    np.testing.assert_allclose(fused_sorted, fused_plain, rtol=1e-5, atol=1e-5)

  def test_vocab_sort_loss_unchanged(self):
    """Arbitrary permutation does not change the loss value."""
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(201)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    # Random permutation of vocabulary
    perm = jax.random.permutation(jax.random.fold_in(key, 3), V)

    fused_sorted = fused_linear_cross_entropy_loss(
        x, w, labels, chunk_size=chunk_size, vocab_sort_indices=perm
    )
    naive = _naive_loss(x, w, labels)
    np.testing.assert_allclose(fused_sorted, naive, rtol=1e-5, atol=1e-5)

  def test_vocab_sort_gradient_x_unchanged(self):
    """dx is unaffected by vocab sorting (labels still in original space)."""
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(202)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    perm = jax.random.permutation(jax.random.fold_in(key, 3), V)

    dx_sorted = jax.grad(
        lambda x_: fused_linear_cross_entropy_loss(
            x_, w, labels, chunk_size=chunk_size, vocab_sort_indices=perm
        )
    )(x)
    dx_naive = jax.grad(lambda x_: _naive_loss(x_, w, labels))(x)

    np.testing.assert_allclose(dx_sorted, dx_naive, rtol=1e-4, atol=1e-5)

  def test_vocab_sort_gradient_w_correctly_unsorded(self):
    """dw is correctly scattered back to original weight row order."""
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(203)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    perm = jax.random.permutation(jax.random.fold_in(key, 3), V)

    dw_sorted_call = jax.grad(
        lambda w_: fused_linear_cross_entropy_loss(
            x, w_, labels, chunk_size=chunk_size, vocab_sort_indices=perm
        )
    )(w)
    dw_naive = jax.grad(lambda w_: _naive_loss(x, w_, labels))(w)

    # Both dw arrays should be in original vocabulary order.
    np.testing.assert_allclose(dw_sorted_call, dw_naive, rtol=1e-4, atol=1e-5)

  def test_vocab_sort_combined_with_shift(self):
    """Vocab sorting and shift can be combined."""
    T, D, V, chunk_size = 12, 8, 16, 4
    key = jax.random.key(204)
    x = jax.random.normal(key, (T, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (T,), 0, V)

    perm = jax.random.permutation(jax.random.fold_in(key, 3), V)

    fused = fused_linear_cross_entropy_loss(
        x, w, labels, chunk_size=chunk_size, shift=1, vocab_sort_indices=perm
    )
    naive = _naive_loss(x[:-1], w, labels[1:])
    np.testing.assert_allclose(fused, naive, rtol=1e-5, atol=1e-5)

  # ==================================================================
  # Feature 3: filter_eps gradient filtering
  # ==================================================================

  def test_filter_eps_no_effect_on_forward_loss(self):
    """filter_eps never changes the forward loss value."""
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(300)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    loss_no_filter = fused_linear_cross_entropy_loss(x, w, labels, chunk_size=chunk_size)
    loss_small_eps = fused_linear_cross_entropy_loss(
        x, w, labels, chunk_size=chunk_size, filter_eps=0.0
    )
    loss_large_eps = fused_linear_cross_entropy_loss(
        x, w, labels, chunk_size=chunk_size, filter_eps=100.0
    )
    loss_inf_eps = fused_linear_cross_entropy_loss(
        x, w, labels, chunk_size=chunk_size, filter_eps=float("inf")
    )

    # Forward is always exact regardless of filter_eps.
    np.testing.assert_allclose(loss_small_eps, loss_no_filter, rtol=0, atol=0)
    np.testing.assert_allclose(loss_large_eps, loss_no_filter, rtol=0, atol=0)
    np.testing.assert_allclose(loss_inf_eps, loss_no_filter, rtol=0, atol=0)

  def test_filter_eps_inf_exact_gradients(self):
    """filter_eps=inf keeps all chunks active → exact gradients."""
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(301)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    dx_no_filter = jax.grad(
        lambda x_: fused_linear_cross_entropy_loss(x_, w, labels, chunk_size=chunk_size)
    )(x)
    dx_inf_eps = jax.grad(
        lambda x_: fused_linear_cross_entropy_loss(
            x_, w, labels, chunk_size=chunk_size, filter_eps=float("inf")
        )
    )(x)

    # filter_eps=inf → all chunks active, must be numerically identical.
    np.testing.assert_allclose(dx_inf_eps, dx_no_filter, rtol=1e-6, atol=1e-6)

  def test_filter_eps_inf_exact_gradients_w(self):
    """filter_eps=inf gives exact dw."""
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(302)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    dw_no_filter = jax.grad(
        lambda w_: fused_linear_cross_entropy_loss(x, w_, labels, chunk_size=chunk_size)
    )(w)
    dw_inf_eps = jax.grad(
        lambda w_: fused_linear_cross_entropy_loss(
            x, w_, labels, chunk_size=chunk_size, filter_eps=float("inf")
        )
    )(w)
    np.testing.assert_allclose(dw_inf_eps, dw_no_filter, rtol=1e-6, atol=1e-6)

  def test_filter_eps_approximate_peaked_distribution(self):
    """Moderate filter_eps gives close gradients on a peaked distribution.

    We construct a case where the first vocab chunk has logits ~100× larger
    than all others.  With filter_eps=50, every other chunk is filtered in
    the backward.  Because their softmax contributions are exp(-100)≈0, the
    gradient error is negligible.
    """
    N, D, V, chunk_size = 8, 16, 64, 8
    key = jax.random.key(303)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    # Small weights for all tokens, then massively boost the first chunk.
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32) * 0.01
    w = w.at[:chunk_size].set(
        jax.random.normal(jax.random.fold_in(key, 2), (chunk_size, D)) * 10.0
    )
    # Labels in the high-logit chunk so that the label correction is not filtered.
    labels = jax.random.randint(jax.random.fold_in(key, 3), (N,), 0, chunk_size)

    dx_exact = jax.grad(
        lambda x_: fused_linear_cross_entropy_loss(x_, w, labels, chunk_size=chunk_size)
    )(x)
    dx_filtered = jax.grad(
        lambda x_: fused_linear_cross_entropy_loss(
            x_, w, labels, chunk_size=chunk_size, filter_eps=50.0
        )
    )(x)

    # The first chunk dominates; other chunks contribute < exp(-50) ≈ 0.
    np.testing.assert_allclose(dx_filtered, dx_exact, rtol=1e-4, atol=1e-4)

  def test_filter_eps_combined_with_vocab_sort(self):
    """filter_eps and vocab_sort_indices can be used together."""
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(304)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)
    perm = jax.random.permutation(jax.random.fold_in(key, 3), V)

    # With filter_eps=inf both features are active; loss must be exact.
    loss = fused_linear_cross_entropy_loss(
        x, w, labels,
        chunk_size=chunk_size,
        vocab_sort_indices=perm,
        filter_eps=float("inf"),
    )
    naive = _naive_loss(x, w, labels)
    np.testing.assert_allclose(loss, naive, rtol=1e-5, atol=1e-5)

  def test_filter_eps_jit(self):
    """filter_eps works under jax.jit."""
    N, D, V, chunk_size = 8, 16, 32, 8
    key = jax.random.key(305)
    x = jax.random.normal(key, (N, D), dtype=jnp.float32)
    w = jax.random.normal(jax.random.fold_in(key, 1), (V, D), dtype=jnp.float32)
    labels = jax.random.randint(jax.random.fold_in(key, 2), (N,), 0, V)

    jit_grad = jax.jit(
        jax.grad(
            lambda x_: fused_linear_cross_entropy_loss(
                x_, w, labels, chunk_size=chunk_size, filter_eps=float("inf")
            )
        )
    )
    dx_jit = jit_grad(x)
    dx_ref = jax.grad(
        lambda x_: fused_linear_cross_entropy_loss(x_, w, labels, chunk_size=chunk_size)
    )(x)
    np.testing.assert_allclose(dx_jit, dx_ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
