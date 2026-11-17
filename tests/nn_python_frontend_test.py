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

"""Host tests for the optional cuDNN Python frontend dispatch."""
import os
import types
from unittest import mock

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.nn import functions


class PythonFrontendTest(parameterized.TestCase):

  @parameterized.parameters(None, '0', '1')
  def test_dispatch(self, setting):
    q = jnp.ones((1, 4, 2, 8), jnp.float16)
    def fake(q, k, v, *args, **kwargs):
      return q * 2
    ops = types.ModuleType('cudnn.ops')
    ops.sdpa_forward_jax = mock.Mock(side_effect=lambda q, *a, **kw: (
        q * 2, jnp.zeros(q.shape[:-1], jnp.float32)))
    ops.sdpa_backward_jax = mock.Mock()
    env = {} if setting is None else {
        'JAX_CUDNN_SDPA_USE_PYTHON_FRONTEND': setting}
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.dict('sys.modules', {
            'cudnn': types.SimpleNamespace(ops=ops), 'cudnn.ops': ops}),
        mock.patch.object(functions, 'cudnn_dot_product_attention',
                          side_effect=fake) as legacy,
    ):
      result = jax.jit(lambda q: jax.nn.dot_product_attention(
          q, q, q, implementation='cudnn'))(q)
      np.testing.assert_array_equal(result, q * 2)
      self.assertEqual(ops.sdpa_forward_jax.call_count, int(setting == '1'))
      ops.sdpa_backward_jax.assert_not_called()
      self.assertEqual(legacy.call_count, int(setting != '1'))

  @parameterized.parameters(False, True)
  def test_explicit_backward_and_saved_stats(self, unbatched):
    shape = (4, 2, 8) if unbatched else (1, 4, 2, 8)
    q = jnp.ones(shape, jnp.float16)
    bias = jnp.zeros((1, 2, 4, 4), jnp.float16)
    mask = jnp.ones((1, 1, 4, 4), bool)
    lengths = jnp.array([3], jnp.int32)
    ops = types.ModuleType('cudnn.ops')

    def forward(q, k, v, bias, mask, **kwargs):
      self.assertEqual(kwargs['scale'], 0.25)
      self.assertTrue(kwargs['is_causal'])
      self.assertEqual(kwargs['local_window_size'], (2, 0))
      self.assertEqual(kwargs['query_seq_lengths'].dtype, jnp.int32)
      self.assertEqual(kwargs['key_value_seq_lengths'].dtype, jnp.int32)
      return q * 2, jnp.full(q.shape[:-1], 3, jnp.float32)

    def backward(q, k, v, output, grad_output, stats, bias, mask, **kwargs):
      self.assertEqual(stats.dtype, jnp.float32)
      self.assertEqual(stats.shape, q.shape[:-1])
      self.assertEqual(output.shape, q.shape)
      self.assertEqual(mask.dtype, jnp.bool_)
      return (grad_output * 2, jnp.zeros_like(k), jnp.zeros_like(v),
              jnp.ones_like(bias))

    ops.sdpa_forward_jax = mock.Mock(side_effect=forward)
    ops.sdpa_backward_jax = mock.Mock(side_effect=backward)

    def loss(q, k, v, bias, mask, lengths):
      output, residual = jax.nn.dot_product_attention(
          q, k, v, bias, mask, scale=0.25, is_causal=True,
          query_seq_lengths=lengths, key_value_seq_lengths=lengths,
          local_window_size=(2, 0), return_residual=True,
          implementation='cudnn')
      return output.sum() + residual.sum(), residual

    with (
        mock.patch.dict(os.environ,
                        {'JAX_CUDNN_SDPA_USE_PYTHON_FRONTEND': '1'}),
        mock.patch.dict('sys.modules', {
            'cudnn': types.SimpleNamespace(ops=ops), 'cudnn.ops': ops}),
    ):
      (_, residual), grads = jax.jit(jax.value_and_grad(
          loss, argnums=(0, 1, 2, 3), has_aux=True))(
              q, q, q, bias, mask, lengths)
    self.assertEqual(residual.dtype, q.dtype)
    self.assertEqual(residual.shape, q.shape[:-1])
    np.testing.assert_array_equal(residual, jnp.full(q.shape[:-1], 3, q.dtype))
    for got, expected in zip(grads, (q * 2, q * 0, q * 0, bias + 1)):
      np.testing.assert_array_equal(got, expected)
    ops.sdpa_forward_jax.assert_called_once()
    ops.sdpa_backward_jax.assert_called_once()

  def test_invalid_setting(self):
    with mock.patch.dict(os.environ, {'JAX_CUDNN_SDPA_USE_PYTHON_FRONTEND': '2'}):
      q = jnp.ones((1, 4, 2, 8), jnp.float16)
      with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
        jax.nn.dot_product_attention(q, q, q, implementation='cudnn')
      jax.nn.dot_product_attention(q, q, q, implementation='xla')


if __name__ == '__main__':
  absltest.main()
