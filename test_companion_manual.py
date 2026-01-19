#!/usr/bin/env python3
"""Manual test for companion matrix implementation."""

import sys
sys.path.insert(0, '.')

import numpy as np
import jax
import jax.numpy as jnp
from jax._src.scipy import linalg as jsp_linalg

def test_basic():
    """Test basic companion matrix creation."""
    print("Test 1: Basic companion matrix [1, 2, 3]")
    a = np.array([1, 2, 3])
    result = jsp_linalg.companion(a)
    expected = np.array([[-2.0, -3.0], [1.0, 0.0]])
    print(f"Result:\n{result}")
    print(f"Expected:\n{expected}")
    assert np.allclose(result, expected), "Test 1 failed!"
    print("✓ Test 1 passed\n")

def test_float():
    """Test with float values."""
    print("Test 2: Float coefficients [2.0, 5.0, -10.0]")
    a = np.array([2.0, 5.0, -10.0])
    result = jsp_linalg.companion(a)
    expected = np.array([[-2.5, 5.0], [1.0, 0.0]])
    print(f"Result:\n{result}")
    print(f"Expected:\n{expected}")
    assert np.allclose(result, expected), "Test 2 failed!"
    print("✓ Test 2 passed\n")

def test_larger():
    """Test larger matrix."""
    print("Test 3: Larger matrix [1, -10, 31, -30]")
    a = np.array([1, -10, 31, -30])
    result = jsp_linalg.companion(a)
    expected = np.array([[10., -31., 30.], [1., 0., 0.], [0., 1., 0.]])
    print(f"Result:\n{result}")
    print(f"Expected:\n{expected}")
    assert np.allclose(result, expected), "Test 3 failed!"
    print("✓ Test 3 passed\n")

def test_complex():
    """Test complex coefficients."""
    print("Test 4: Complex coefficients [1+2j, 3-1j, 2+4j]")
    a = np.array([1.0+2.0j, 3.0-1.0j, 2.0+4.0j])
    result = jsp_linalg.companion(a)
    print(f"Result:\n{result}")
    print(f"Result dtype: {result.dtype}")
    assert result.dtype == jnp.complex64 or result.dtype == jnp.complex128, "Complex dtype check failed!"
    print("✓ Test 4 passed\n")

def test_error_cases():
    """Test error handling."""
    print("Test 5: Error cases")
    
    # Test n < 2
    try:
        jsp_linalg.companion(np.array([1]))
        assert False, "Should have raised ValueError for n < 2"
    except ValueError:
        print("✓ Correctly raised ValueError for n < 2")
    
    # Test zero leading coefficient
    try:
        jsp_linalg.companion(np.array([0, 1, 2]))
        assert False, "Should have raised ValueError for a[0] == 0"
    except ValueError:
        print("✓ Correctly raised ValueError for a[0] == 0")
    
    print("✓ Test 5 passed\n")

def test_jit_compilation():
    """Test that function can be JIT compiled."""
    print("Test 6: JIT compilation")
    jitted_companion = jax.jit(jsp_linalg.companion)
    a = np.array([1.0, 2.0, 3.0])
    result = jitted_companion(a)
    expected = np.array([[-2.0, -3.0], [1.0, 0.0]])
    print(f"JIT compiled result:\n{result}")
    assert np.allclose(result, expected), "JIT compilation test failed!"
    print("✓ Test 6 passed\n")

if __name__ == "__main__":
    print("="*60)
    print("Testing JAX companion matrix implementation")
    print("="*60 + "\n")
    
    try:
        test_basic()
        test_float()
        test_larger()
        test_complex()
        test_error_cases()
        test_jit_compilation()
        
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
