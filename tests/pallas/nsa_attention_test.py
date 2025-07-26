import unittest
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.gpu.nsa_attention import nsa_attention, nsa_attention_reference

class TestNSAAttention(unittest.TestCase):

    def setUp(self):
        self.key = jax.random.PRNGKey(0)
        self.batch_size = 2
        self.seq_len = 16
        self.num_heads = 4
        self.head_dim = 8
        self.q = jax.random.normal(self.key, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        self.k = jax.random.normal(self.key, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        self.v = jax.random.normal(self.key, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        self.segment_ids = jax.random.randint(self.key, (self.batch_size, self.seq_len), 0, 2)

    def test_forward_pass(self):
        out = nsa_attention(self.q, self.k, self.v, self.segment_ids)
        out_ref = nsa_attention_reference(self.q, self.k, self.v, self.segment_ids)
        self.assertTrue(jnp.allclose(out, out_ref, atol=1e-5))

    def test_backward_pass(self):
        def loss_fn(q, k, v):
            out = nsa_attention(q, k, v, self.segment_ids)
            return jnp.sum(out)

        grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
        grads = grad_fn(self.q, self.k, self.v)
        self.assertEqual(len(grads), 3)
        for grad in grads:
            self.assertEqual(grad.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))

    def test_edge_cases(self):
        # Test with empty input
        q_empty = jnp.zeros((self.batch_size, 0, self.num_heads, self.head_dim))
        k_empty = jnp.zeros((self.batch_size, 0, self.num_heads, self.head_dim))
        v_empty = jnp.zeros((self.batch_size, 0, self.num_heads, self.head_dim))
        out_empty = nsa_attention(q_empty, k_empty, v_empty, self.segment_ids)
        self.assertEqual(out_empty.shape, (self.batch_size, 0, self.num_heads, self.head_dim))

        # Test with different sequence lengths
        q_diff_len = jax.random.normal(self.key, (self.batch_size, self.seq_len + 1, self.num_heads, self.head_dim))
        k_diff_len = jax.random.normal(self.key, (self.batch_size, self.seq_len + 2, self.num_heads, self.head_dim))
        v_diff_len = jax.random.normal(self.key, (self.batch_size, self.seq_len + 2, self.num_heads, self.head_dim))
        out_diff_len = nsa_attention(q_diff_len, k_diff_len, v_diff_len, self.segment_ids)
        self.assertEqual(out_diff_len.shape, (self.batch_size, self.seq_len + 1, self.num_heads, self.head_dim))

if __name__ == '__main__':
    unittest.main()
