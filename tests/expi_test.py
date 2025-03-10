import unittest
import jax
import jax.numpy as jnp
from jax.scipy.special import expi
import numpy as np

class ExpiTestCase(unittest.TestCase):
  """Tests for fixing the expi function issue with JIT disabled."""
  
  def setUp(self):
    # Save the original JIT state to restore after the test
    self.original_jit_state = jax.config.jax_disable_jit
  
  def tearDown(self):
    # Restore the original JIT state
    jax.config.update("jax_disable_jit", self.original_jit_state)
  
  def test_expi_negative_array_jit_disabled(self):
    """Test that expi works correctly for negative array inputs with JIT disabled."""
    # Test values in the range -1 <= x < 0
    test_values = jnp.array([-0.9, -0.75, -0.5, -0.25, -0.1, -0.01])
    
    # Get the expected results with JIT enabled
    jax.config.update("jax_disable_jit", False)
    expected_results = expi(test_values)
    expected_single = expi(jnp.array([-0.5]))
    
    # Test with JIT disabled
    jax.config.update("jax_disable_jit", True)
    
    # Test individual values in an array
    for i, val in enumerate(test_values):
      test_array = jnp.array([val])
      result = expi(test_array)
      self.assertAllClose(result, jnp.array([expected_results[i]]))
    
    # Test the specific case from the bug report
    result_single = expi(jnp.array([-0.5]))
    self.assertAllClose(result_single, expected_single)
  
  def test_expi_scalar_jit_disabled(self):
    """Test that expi works correctly for scalar inputs with JIT disabled."""
    # Test scalar inputs
    test_values = [-0.9, -0.75, -0.5, -0.25, -0.1, -0.01]
    
    # Get the expected results with JIT enabled
    jax.config.update("jax_disable_jit", False)
    expected_results = [expi(val) for val in test_values]
    
    # Test with JIT disabled
    jax.config.update("jax_disable_jit", True)
    
    # Test scalar values
    for i, val in enumerate(test_values):
      result = expi(val)
      self.assertAllClose(result, expected_results[i])
  
  def assertAllClose(self, x, y, rtol=1e-5, atol=1e-5):
    """Helper method to test that values are close."""
    np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)

if __name__ == "__main__":
  unittest.main() 