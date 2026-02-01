"""
Test for CUDA 13 segfault fix (GitHub issue #34696)

Tests the segmentation fault that occurs when combining jax.grad, jax.vmap, 
and matrix multiplication with CUDA 13.

This test should be integrated into the JAX test suite to prevent regression.
"""

import unittest
import jax
import jax.numpy as jnp
import jax._src.test_util as jtu


class Cuda13SegfaultTest(jtu.JaxTestCase):
  """Tests for CUDA 13 cuBLASLt segfault issue #34696"""

  def test_grad_vmap_matmul_original_issue(self):
    """Test the original failing case from issue #34696.
    
    This combines jax.grad, jax.vmap, and matrix multiplication which
    triggered a segmentation fault in cuBLASLt with CUDA 13.
    """
    key = jax.random.PRNGKey(0)
    batch_size, in_size, out_size = 32, 2, 3
    wkey, bkey = jax.random.split(key)
    weight = jax.random.normal(wkey, (out_size, in_size))

    def f(params, x):
      (weight, ) = params
      def linear(x):
        return weight @ x
      pred_y = jax.vmap(linear)(x)
      return jnp.sum((pred_y) ** 2)

    x = jnp.zeros((batch_size, in_size))

    # This should not raise a segmentation fault
    grad_fn = jax.grad(f)
    result = grad_fn((weight, ), x)
    
    # Verify the result has the right shape
    self.assertEqual(result[0].shape, weight.shape)

  def test_grad_vmap_matmul_with_jit(self):
    """Test grad + vmap + matmul with JIT compilation."""
    key = jax.random.PRNGKey(0)
    batch_size, in_size, out_size = 32, 2, 3
    wkey, bkey = jax.random.split(key)
    weight = jax.random.normal(wkey, (out_size, in_size))

    def f(params, x):
      (weight, ) = params
      def linear(x):
        return weight @ x
      pred_y = jax.vmap(linear)(x)
      return jnp.sum((pred_y) ** 2)

    x = jnp.zeros((batch_size, in_size))

    # JIT-compiled version
    grad_fn = jax.jit(jax.grad(f))
    result = grad_fn((weight, ), x)
    
    self.assertEqual(result[0].shape, weight.shape)

  def test_vmap_grad_matmul_reversed_order(self):
    """Test vmap(grad(matmul)) - reversed composition order."""
    key = jax.random.PRNGKey(0)
    batch_size, in_size, out_size = 32, 2, 3
    wkey = jax.random.PRNGKey(1)
    weight = jax.random.normal(wkey, (out_size, in_size))

    def f(weight, x):
      return jnp.sum((weight @ x) ** 2)

    def grad_f(weight, x):
      return jax.grad(f)(weight, x)

    x = jax.random.normal(jax.random.PRNGKey(2), (batch_size, in_size))

    # This tests the reverse composition: vmap(grad(f))
    vmapped_grad = jax.vmap(grad_f, in_axes=(None, 0))
    result = vmapped_grad(weight, x)
    
    self.assertEqual(result.shape, (batch_size,  out_size, in_size))

  def test_grad_vmap_multiple_matmuls(self):
    """Test grad with multiple matmuls inside vmap."""
    key = jax.random.PRNGKey(0)
    batch_size, hidden_size = 32, 64
    
    w1_key, w2_key, x_key = jax.random.split(key, 3)
    w1 = jax.random.normal(w1_key, (hidden_size, hidden_size))
    w2 = jax.random.normal(w2_key, (hidden_size, hidden_size))

    def f(params, x):
      w1, w2 = params
      def layer(x):
        x = w1 @ x
        x = jax.nn.relu(x)
        x = w2 @ x
        return x
      
      y = jax.vmap(layer)(x)
      return jnp.sum(y ** 2)

    x = jax.random.normal(x_key, (batch_size, hidden_size))

    grad_fn = jax.grad(f)
    result = grad_fn((w1, w2), x)
    
    self.assertEqual(result[0].shape, w1.shape)
    self.assertEqual(result[1].shape, w2.shape)

  def test_grad_vmap_batched_matmul(self):
    """Test grad with batched matrix multiplication in vmap."""
    key = jax.random.PRNGKey(0)
    batch_size, m, n, k = 32, 16, 32, 8
    
    A_key, B_key, x_key = jax.random.split(key, 3)
    A = jax.random.normal(A_key, (m, n))
    B = jax.random.normal(B_key, (n, k))

    def f(A, B, x):
      def compute(x):
        return jnp.sum((A @ B @ x) ** 2)
      return jnp.sum(jax.vmap(compute)(x))

    x = jax.random.normal(x_key, (batch_size, k))

    grad_fn = jax.grad(f, argnums=(0, 1))
    grad_A, grad_B = grad_fn(A, B, x)
    
    self.assertEqual(grad_A.shape, A.shape)
    self.assertEqual(grad_B.shape, B.shape)


if __name__ == '__main__':
  unittest.main()
