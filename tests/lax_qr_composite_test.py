# Copyright 2025 The JAX Authors.
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

"""Tests for QR decomposition composite implementation."""

import unittest
import jax
import jax.numpy as jnp
from jax._src.lax import linalg
from jax.interpreters import mlir
import numpy as np

class QRCompositeTest(unittest.TestCase):
  """Tests for QR decomposition composite implementation."""

  def test_qr_custom_backend_lowering(self):
    """Test that QR operations are lowered to composite ops for custom backends."""
    # Only proceed with the test if we have a functioning jaxlib
    try:
      import jaxlib
    except ImportError:
      self.skipTest("jaxlib not available")

    # Create a simple matrix for testing QR
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)
    
    # Check that we get correct results for the QR decomposition
    q, r = jax.lax.linalg.qr(x, full_matrices=False)
    
    # The numerical results should match numpy's implementation
    q_np, r_np = np.linalg.qr(x, mode='reduced')
    np.testing.assert_allclose(q, q_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r, r_np, rtol=1e-5, atol=1e-5)
    
    # Compile and check the MLIR for the presence of composite ops
    # We need to fake being a different platform to trigger the composite lowering
    original_platforms = getattr(jax.interpreters.mlir.ModuleContext, 'platforms', None)
    
    # Mock a custom platform to trigger our composite implementation
    class FakeModule:
      platforms = ['fake_platform']
    
    def get_hlo(f, *args, **kwargs):
      old_module_call = mlir.ModuleContext.__call__
      hlo_output = None
      
      def module_context_mock(self, *args, **kw):
        self.platforms = ['fake_platform']
        return old_module_call(self, *args, **kw)
      
      try:
        mlir.ModuleContext.__call__ = module_context_mock
        hlo_output = jax.jit(f).lower(*args, **kwargs).as_text()
      finally:
        mlir.ModuleContext.__call__ = old_module_call
      
      return hlo_output
    
    # Get the HLO for the QR operation
    hlo_output = get_hlo(lambda x: jax.lax.linalg.qr(x, full_matrices=False), x)
    
    # Verify that the composite operations are in the generated HLO
    self.assertIn('stablehlo.composite = "qr.geqrf"', hlo_output)
    self.assertIn('stablehlo.composite = "qr.householder_product"', hlo_output)
    
    # Now verify that for the default backend (CPU/GPU) we still use custom_calls
    hlo_output_default = jax.jit(lambda x: jax.lax.linalg.qr(x, full_matrices=False)).lower(x).as_text()
    self.assertIn('stablehlo.custom_call @Qr', hlo_output_default)
    self.assertIn('stablehlo.custom_call @ProductOfElementaryHouseholderReflectors', hlo_output_default)


if __name__ == "__main__":
  unittest.main()
