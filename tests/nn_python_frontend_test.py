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
    env = {} if setting is None else {
        'JAX_CUDNN_SDPA_USE_PYTHON_FRONTEND': setting}
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.dict('sys.modules', {
            'cudnn': types.SimpleNamespace(sdpa_jax=mock.Mock(side_effect=fake))}),
        mock.patch.object(functions, 'cudnn_dot_product_attention',
                          side_effect=fake) as legacy,
    ):
      import cudnn
      result = jax.jit(lambda q: jax.nn.dot_product_attention(
          q, q, q, implementation='cudnn'))(q)
      np.testing.assert_array_equal(result, q * 2)
      self.assertEqual(cudnn.sdpa_jax.call_count, int(setting == '1'))
      self.assertEqual(legacy.call_count, int(setting != '1'))

  def test_invalid_setting(self):
    with mock.patch.dict(os.environ, {'JAX_CUDNN_SDPA_USE_PYTHON_FRONTEND': '2'}):
      q = jnp.ones((1, 4, 2, 8), jnp.float16)
      with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
        jax.nn.dot_product_attention(q, q, q, implementation='cudnn')
      jax.nn.dot_product_attention(q, q, q, implementation='xla')


if __name__ == '__main__':
  absltest.main()
