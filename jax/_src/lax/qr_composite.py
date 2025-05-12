"""
Implementation of QR decomposition using stablehlo.composite for custom backends.

This implementation provides two operational modes:
1. For CPU/GPU/ROCm: Uses native custom_call operations for optimal performance
2. For other backends: Uses stablehlo.composite to enable explicit implementations

This can be tested directly with:
$ python qr_composite_fixed.py
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax._src.interpreters import mlir
from jax._src.lax import linalg
from jax._src.lib.mlir.dialects import hlo

# Flag to force composite operations for all platforms (testing only)
FORCE_COMPOSITE = False

def add_composite_attribute(op, name):
    """Add stablehlo.composite attribute to an operation.
    
    Args:
        op: The operation to add the attribute to.
        name: The name of the composite operation (e.g., "qr.geqrf").
        
    Returns:
        The modified operation with the composite attribute added.
        
    Raises:
        AttributeError: If the operation doesn't have an attributes dictionary.
    """
    try:
        op.operation.attributes["stablehlo.composite"] = mlir.ir.StringAttr.get(name)
        return op
    except Exception as e:
        raise AttributeError(f"Failed to add composite attribute '{name}': {str(e)}") from e

def register_qr_composite_lowering():
    """Register composite lowering rules for QR decomposition.
    
    This replaces the standard lowering rules for QR-related operations
    with versions that use stablehlo.composite for custom backends.
    
    Returns:
        The original custom_call function that was replaced.
    """
    try:
        # Store original custom_call implementation
        original_custom_call = mlir.custom_call
        
        # Save it for restoration later
        mlir._original_custom_call = original_custom_call
        
        def patched_custom_call(call_target_name, **kwargs):
            """Patched version of custom_call that adds composite attributes for QR operations.
            
            Args:
                call_target_name: Name of the target operation.
                **kwargs: Arguments passed to the original custom_call function.
                
            Returns:
                Modified operation with composite attributes if applicable.
            """
            op = original_custom_call(call_target_name, **kwargs)
            
            # For QR-related operations, add a composite attribute
            # Only when not on CPU/GPU/ROCm or when forced
            platform = os.environ.get("JAX_PLATFORM_NAME", "").lower()
            if FORCE_COMPOSITE or platform not in ["cpu", "cuda", "rocm"]:
                if "qr" in call_target_name.lower() or "geqrf" in call_target_name.lower():
                    return add_composite_attribute(op, "qr.geqrf")
                elif "householder" in call_target_name.lower() or "orgqr" in call_target_name.lower():
                    return add_composite_attribute(op, "qr.householder_product")
            
            return op
        
        # Replace the custom_call function
        mlir.custom_call = patched_custom_call
        
        print("✓ Registered QR composite lowering")
        return original_custom_call
    except Exception as e:
        print(f"× Failed to register composite lowerings: {str(e)}")
        return None

def restore_original_lowering():
    """Restore the original custom_call implementation.
    
    This undoes the patching done by register_qr_composite_lowering.
    The original custom_call implementation is restored from the saved reference.
    
    Returns:
        bool: True if restoration was successful, False otherwise.
    """
    try:
        if hasattr(mlir, "_original_custom_call"):
            mlir.custom_call = mlir._original_custom_call
            delattr(mlir, "_original_custom_call")
            print("✓ Restored original QR lowering")
            return True
        else:
            print("× No original custom_call implementation to restore")
            return False
    except Exception as e:
        print(f"× Failed to restore original lowering: {str(e)}")
        return False

def enable_force_composite():
    """Enable forcing composite operations for all platforms.
    
    This forces the use of composite operations regardless of the platform.
    Useful for testing the composite implementation on standard platforms.
    
    Returns:
        bool: Previous value of FORCE_COMPOSITE flag.
    """
    global FORCE_COMPOSITE
    old_value = FORCE_COMPOSITE
    FORCE_COMPOSITE = True
    print("✓ Force composite mode enabled")
    return old_value

def make_test_matrix(shape=(4, 3), dtype=np.float32, seed=42):
    """Create a well-conditioned test matrix for QR decomposition.
    
    Args:
        shape: Tuple specifying the shape of the matrix (m, n).
        dtype: Data type of the matrix elements.
        seed: Random seed for reproducibility.
        
    Returns:
        A randomly generated matrix of the specified shape and dtype.
    """
    # Using a random matrix with good conditioning
    key = jax.random.key(seed)
    return jax.random.normal(key, shape, dtype=dtype)

def test_qr_decomposition(platform_name="default", force_composite=False, 
                          rtol=1e-5, atol=1e-5, test_matrix=None):
    """Test QR decomposition on a given platform.
    
    Args:
        platform_name: String identifier for the platform being tested.
        force_composite: If True, forces the use of composite operations.
        rtol: Relative tolerance for numerical verification.
        atol: Absolute tolerance for numerical verification.
        test_matrix: Optional pre-generated test matrix. If None, one will be created.
        
    Returns:
        A tuple containing:
            - implementation_type: String indicating the type of implementation detected
              ("composite", "custom_call", or "mixed").
            - success: Boolean indicating if the numerical verification passed.
    
    Raises:
        RuntimeError: If JIT compilation or execution fails.
    """
    global FORCE_COMPOSITE
    print(f"\n===== Testing QR decomposition on {platform_name} =====")
    
    # Create or use provided test matrix
    x = test_matrix if test_matrix is not None else make_test_matrix()
    
    # Save the original FORCE_COMPOSITE flag
    old_force = FORCE_COMPOSITE
    if force_composite:
        enable_force_composite()
    
    try:
        # Define a JIT-compiled QR function
        @jax.jit
        def qr_fn(x):
            return linalg.qr(x)
        
        try:
            # Get the HLO representation to analyze the implementation
            hlo_text = qr_fn.lower(x).as_text()
            
            # Look for composite operations
            composite_found = False
            for line in hlo_text.split('\n'):
                if 'stablehlo.composite' in line:
                    print(f"✓ Found composite operation: {line.strip()}")
                    composite_found = True
                    break
            
            # Look for custom_call operations without composite attribute
            pure_custom_call = False
            for line in hlo_text.split('\n'):
                if 'custom_call' in line and 'stablehlo.composite' not in line:
                    if any(s in line for s in ['Qr', 'geqrf', 'orgqr']):
                        print(f"✓ Found custom_call: {line.strip()}")
                        pure_custom_call = True
                        break
            
            # Determine the implementation type
            if composite_found and not pure_custom_call:
                impl_type = "composite"
            elif pure_custom_call and not composite_found:
                impl_type = "custom_call"
            else:
                impl_type = "mixed"
            
            print(f"Implementation type: {impl_type}")
        except Exception as e:
            print(f"× Error analyzing HLO: {str(e)}")
            impl_type = "unknown"
        
        try:
            # Run QR computation
            q, r = qr_fn(x)
            
            # Verify numerical correctness
            # Check Q orthogonality (Q.T @ Q ≈ I)
            q_ortho = jnp.allclose(q.T @ q, jnp.eye(q.shape[1]), rtol=rtol, atol=atol)
            # Check Q*R = X reconstruction
            reconstruct = jnp.allclose(q @ r, x, rtol=rtol, atol=atol)
            
            print(f"Q shape: {q.shape}, R shape: {r.shape}")
            print(f"Q orthogonal: {q_ortho}")
            print(f"Q*R = X: {reconstruct}")
            
            return impl_type, q_ortho and reconstruct
        except Exception as e:
            print(f"× Error in QR computation: {str(e)}")
            return impl_type, False
    finally:
        # Restore the original FORCE_COMPOSITE flag
        FORCE_COMPOSITE = old_force

def main():
    """Run a comprehensive QR composite implementation test suite.
    
    This function tests the QR implementation with both default settings 
    and with forced composite operations. It verifies that:
    1. Composite attributes are properly added to QR operations
    2. Numerical results are correct
    3. The implementation works across different platforms
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        print("QR Composite Implementation Tester")
        print("==================================")
        print(f"JAX Platform: {os.environ.get('JAX_PLATFORM_NAME', 'default')}")
        print(f"JAX_QR_USE_COMPOSITE: {os.environ.get('JAX_QR_USE_COMPOSITE', '0')}")
        
        # Create a test matrix to use for all tests (for consistency)
        test_matrix = make_test_matrix(shape=(4, 3), dtype=np.float32, seed=42)
        
        # Register our composite implementation
        original_custom_call = register_qr_composite_lowering()
        if original_custom_call is None:
            print("\n× Failed to register composite lowerings. Aborting tests.")
            return 1
        
        try:
            # Test with default settings
            impl1, success1 = test_qr_decomposition(
                "default", test_matrix=test_matrix, rtol=1e-5, atol=1e-5)
            
            # Test with forced composite
            impl2, success2 = test_qr_decomposition(
                "forced_composite", force_composite=True, 
                test_matrix=test_matrix, rtol=1e-5, atol=1e-5)
            
            # Summary
            print("\n===== Test Summary =====")
            print(f"Default implementation: {impl1} - {'✓ Success' if success1 else '× Failed'}")
            print(f"Forced composite: {impl2} - {'✓ Success' if success2 else '× Failed'}")
            
            # Our goal is to have composite attributes for QR operations 
            # across platforms and have numerical verification pass
            if success1 and success2 and 'composite' in impl1 and 'composite' in impl2:
                print("\n✓ Overall test PASSED: QR composite implementation working correctly!")
                print("\nThe stablehlo.composite attribute is successfully added to QR operations")
                print("allowing custom backends to implement these operations explicitly.")
                return 0
            else:
                print("\n× Overall test FAILED: Implementation issues detected")
                if not success1:
                    print("  - Default implementation failed numerical verification")
                if not success2:
                    print("  - Forced composite implementation failed numerical verification")
                if 'composite' not in impl1:
                    print("  - Default implementation doesn't use composite operations")
                if 'composite' not in impl2:
                    print("  - Forced composite mode didn't use composite operations")
                return 1
        finally:
            # Always restore original implementation
            restore_original_lowering()
    except Exception as e:
        print(f"\n× Unexpected error in test suite: {str(e)}")
        # Try to restore original implementation in case of errors
        try:
            restore_original_lowering()
        except:
            pass
        return 1

if __name__ == "__main__":
    exit_code = main()
    import sys
    sys.exit(exit_code)
