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

from functools import partial
from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src.cudnn import fused_attention_stablehlo

jax.config.parse_flags_with_absl()

class CudnnAttentionBatcherTest(jtu.JaxTestCase):

  def test_dot_product_attention_jacobian_vmap_compatibility(self):
    """Fixes #38495: Verifies that the cuDNN fused attention batcher handles vmap/jacobian axes correctly."""
    
    # 1. Simulate the exact 5D environment triggered during a jacobian/vmap transformation.
    # The leading dimension (4) represents the mapped tracking axis.
    batched_query = jnp.zeros((4, 2, 4, 16, 32), dtype=jnp.float16)
    batched_key   = jnp.zeros((4, 2, 4, 16, 32), dtype=jnp.float16)
    batched_value = jnp.zeros((4, 2, 4, 16, 32), dtype=jnp.float16)
    grad_output   = jnp.zeros((4, 2, 4, 16, 32), dtype=jnp.float16)

    batched_args = (
        batched_query, batched_key, batched_value, 
        None, None, None, None, None, None, None, 
        jnp.zeros((4, 2, 4, 16)), jnp.zeros((4, 2, 4, 16, 32)), grad_output
    )

    # Inform the batcher that the vmap axis is at index 0.
    batch_dims = (0, 0, 0, None, None, None, None, None, None, None, 0, 0, 0)

    # 2. Mock the lower-level primitive binder to safely decouple the test 
    # from the host machine's physical hardware/plugin configurations.
    class MockPrimitiveWrapper:
      def bind(self, *args, **kwargs):
        # Return the expected isolated 4D shape inside the loop execution context
        return [args[0], args[1], args[2]]

    # Temporarily intercept the backend binder for pure mathematical shape verification
    original_wrapper = fused_attention_stablehlo._dot_product_attention_bwd_p_wrapper
    fused_attention_stablehlo._dot_product_attention_bwd_p_wrapper = MockPrimitiveWrapper()

    try:
      # 3. Trigger the modified batcher function under evaluation
      grads, out_bdims = fused_attention_stablehlo._dot_product_attention_bwd_batcher(
          batched_args, batch_dims,
          scale=1.0, seed=0, dropout_rate=0.0, variadic_args=(False, False),
          mask_type=0, layout=0, sliding_window_length=None
      )
      
      # 4. Assert structural integrity and proper axis restoration
      self.assertEqual(grads[0].shape, (4, 2, 4, 16, 32))
      self.assertEqual(out_bdims[0], 0)
      
    finally:
      # Restore original wrapper to preserve global clean state
      fused_attention_stablehlo._dot_product_attention_bwd_p_wrapper = original_wrapper

if __name__ == "__main__":
  absltest.main()