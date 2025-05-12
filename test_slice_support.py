"""Test script for slice support in JAX *_argnum parameters."""

import jax
import jax.numpy as jnp
import numpy as np

def test_grad_with_slice():
    """Test grad with slice argnums."""
    print("\n=== Testing grad with slice ===")
    
    def f(w, x, y, z):
        return w * x * y * z
    
    # Test with integer argnum (original behavior)
    g0 = jax.grad(f, argnums=0)
    result = g0(2.0, 3.0, 4.0, 5.0)
    print(f"grad(f)(2.0, 3.0, 4.0, 5.0) w.r.t. w = {result}")
    assert np.isclose(result, 60.0)  # 3*4*5 = 60
    
    # Test with tuple of argnums (original behavior)
    g01 = jax.grad(f, argnums=(0, 1))
    result = g01(2.0, 3.0, 4.0, 5.0)
    print(f"grad(f)(2.0, 3.0, 4.0, 5.0) w.r.t. (w, x) = {result}")
    assert np.isclose(result[0], 60.0)  # 3*4*5 = 60
    assert np.isclose(result[1], 40.0)  # 2*4*5 = 40
    
    # Test with basic slice (new behavior)
    g01_slice = jax.grad(f, argnums=slice(0, 2))
    result = g01_slice(2.0, 3.0, 4.0, 5.0)
    print(f"grad(f)(2.0, 3.0, 4.0, 5.0) w.r.t. slice(0, 2) = {result}")
    assert np.isclose(result[0], 60.0)  # 3*4*5 = 60
    assert np.isclose(result[1], 40.0)  # 2*4*5 = 40
    
    # Test with slice with step
    g02_slice = jax.grad(f, argnums=slice(0, 4, 2))
    result = g02_slice(2.0, 3.0, 4.0, 5.0)
    print(f"grad(f)(2.0, 3.0, 4.0, 5.0) w.r.t. slice(0, 4, 2) = {result}")
    assert np.isclose(result[0], 60.0)  # 3*4*5 = 60
    assert np.isclose(result[1], 30.0)  # 2*3*5 = 30
    
    # Test with negative indices in slice
    g_neg_slice = jax.grad(f, argnums=slice(-4, -2))
    result = g_neg_slice(2.0, 3.0, 4.0, 5.0)
    print(f"grad(f)(2.0, 3.0, 4.0, 5.0) w.r.t. slice(-4, -2) = {result}")
    assert np.isclose(result[0], 60.0)  # 3*4*5 = 60
    assert np.isclose(result[1], 40.0)  # 2*4*5 = 40
    
    # Test with None start value (default to 0)
    g_none_start = jax.grad(f, argnums=slice(None, 2))
    result = g_none_start(2.0, 3.0, 4.0, 5.0)
    print(f"grad(f)(2.0, 3.0, 4.0, 5.0) w.r.t. slice(None, 2) = {result}")
    assert np.isclose(result[0], 60.0)  # 3*4*5 = 60
    assert np.isclose(result[1], 40.0)  # 2*4*5 = 40
    
    # Test with slice covering all arguments
    g_all = jax.grad(f, argnums=slice(0, 4))
    result = g_all(2.0, 3.0, 4.0, 5.0)
    print(f"grad(f)(2.0, 3.0, 4.0, 5.0) w.r.t. slice(0, 4) = {result}")
    assert np.isclose(result[0], 60.0)  # 3*4*5 = 60
    assert np.isclose(result[1], 40.0)  # 2*4*5 = 40
    assert np.isclose(result[2], 30.0)  # 2*3*5 = 30
    assert np.isclose(result[3], 24.0)  # 2*3*4 = 24
    
    print("All grad tests passed!")

def test_value_and_grad_with_slice():
    """Test value_and_grad with slice argnums."""
    print("\n=== Testing value_and_grad with slice ===")
    
    def f(w, x, y, z):
        return w * x * y * z
    
    # Test with slice
    vg = jax.value_and_grad(f, argnums=slice(1, 3))
    val, grads = vg(2.0, 3.0, 4.0, 5.0)
    print(f"value_and_grad(f)(2.0, 3.0, 4.0, 5.0) value = {val}")
    print(f"value_and_grad(f)(2.0, 3.0, 4.0, 5.0) grads for slice(1, 3) = {grads}")
    assert np.isclose(val, 120.0)
    assert np.isclose(grads[0], 40.0)  # ∂f/∂x = w * y * z = 2 * 4 * 5 = 40
    assert np.isclose(grads[1], 30.0)  # ∂f/∂y = w * x * z = 2 * 3 * 5 = 30
    
    print("All value_and_grad tests passed!")

def test_error_handling():
    """Test error handling for invalid slice inputs."""
    print("\n=== Testing error handling for slices ===")
    
    def f(w, x, y, z):
        return w * x * y * z
    
    # Test error for slice with None stop
    try:
        jax.grad(f, argnums=slice(0, None))
        assert False, "Expected ValueError for slice with None stop"
    except ValueError as e:
        print(f"Correctly raised ValueError for slice(0, None): {e}")
        assert "stop must be specified" in str(e)
    
    # Test error for slice(None, None)
    try:
        jax.grad(f, argnums=slice(None, None))
        assert False, "Expected ValueError for slice(None, None)"
    except ValueError as e:
        print(f"Correctly raised ValueError for slice(None, None): {e}")
        assert "both start and stop" in str(e)
    
    # Test error for empty slice
    try:
        jax.grad(f, argnums=slice(2, 1))
        assert False, "Expected ValueError for empty slice range"
    except ValueError as e:
        print(f"Correctly raised ValueError for slice(2, 1): {e}")
        assert "empty sequence of indices" in str(e)
    
    # Test error for invalid input type
    try:
        jax.grad(f, argnums="invalid")
        assert False, "Expected TypeError for non-integer, non-slice input"
    except TypeError as e:
        print(f"Correctly raised TypeError for non-integer input: {e}")
    
    print("All error handling tests passed!")

def test_jit_with_slice():
    """Test jit with slice for static_argnums."""
    print("\n=== Testing jit with slice for static_argnums ===")
    
    def f(a, b, c, d):
        # a and b will be static when using slice(0, 2)
        return a * b + c * d
    
    # Use slice for static_argnums
    jit_f = jax.jit(f, static_argnums=slice(0, 2))
    result = jit_f(2, 3, jnp.array(4.0), jnp.array(5.0))
    print(f"jit(f)(2, 3, 4.0, 5.0) with static_argnums=slice(0, 2) = {result}")
    assert np.isclose(result, 26.0)
    
    print("All jit tests passed!")

def test_jacfwd_with_slice():
    """Test that jacfwd works with slice objects."""
    print("\n=== Testing jacfwd with slice ===")
    
    # Define a function with multiple arguments that returns multiple outputs
    def f(w, x, y, z):
        return jnp.array([w * x, y * z, w * z, x * y])

    # Test with slice syntax
    jac_f = jax.jacfwd(f, argnums=slice(1, 3))
    jac_x, jac_y = jac_f(2.0, 3.0, 4.0, 5.0)
    
    print(f"jacfwd(f)(2.0, 3.0, 4.0, 5.0) jacobians for slice(1, 3):")
    print(f"∂f/∂x = {jac_x}")
    print(f"∂f/∂y = {jac_y}")
    
    # The Jacobians are returned as arrays with shape (output_dim, 1)
    # For our function, output_dim is 4
    assert jac_x.shape[0] == 4  # 4 outputs
    assert jac_y.shape[0] == 4  # 4 outputs
    
    # Check specific values
    assert jnp.isclose(jac_x[0], 2.0)  # ∂(w*x)/∂x = w = 2.0
    assert jnp.isclose(jac_x[1], 0.0)  # ∂(y*z)/∂x = 0
    assert jnp.isclose(jac_x[2], 0.0)  # ∂(w*z)/∂x = 0
    assert jnp.isclose(jac_x[3], 4.0)  # ∂(x*y)/∂x = y = 4.0
    
    assert jnp.isclose(jac_y[0], 0.0)  # ∂(w*x)/∂y = 0
    assert jnp.isclose(jac_y[1], 5.0)  # ∂(y*z)/∂y = z = 5.0
    assert jnp.isclose(jac_y[2], 0.0)  # ∂(w*z)/∂y = 0
    assert jnp.isclose(jac_y[3], 3.0)  # ∂(x*y)/∂y = x = 3.0
    
    print("All jacfwd tests passed!")


if __name__ == "__main__":
    print("Testing slice support for JAX *_argnum parameters")
    test_grad_with_slice()
    test_value_and_grad_with_slice()
    test_jit_with_slice()
    test_jacfwd_with_slice()
    test_error_handling()
    print("\nAll tests passed successfully!")
