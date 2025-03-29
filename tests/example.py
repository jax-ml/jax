import sys
sys.path.insert(0, "..")
import jax
import jax.numpy as jnp
from safe_lax_map import consistent_lax_map

if __name__ == "__main__":
    # Create a sample input. For example, a 3D array similar to the issue example.
    sino = jnp.zeros((512, 1000, 1536))
    sino = sino.at[:, :, 400:-400].set(1)
    
    # Define a simple function: simulate a convolution by adding a constant.
    def apply_convolution_to_view(view):
        # For demonstration, just add 1 to every element.
        return view + 1

    # Use our wrapper with fixed batch size 128.
    filtered_sino_fixed = consistent_lax_map(apply_convolution_to_view, sino, fixed_batch_size=128)

    # Print the maximum value to verify the operation.
    print('Max with fixed chunk size is {}'.format(jnp.amax(filtered_sino_fixed)))
    
    def test_consistent_results():
        # Create a simple array.
        x = jnp.arange(1000).reshape(1000, 1)
        def f(a):
            return a + 1
        result1 = consistent_lax_map(f, x, fixed_batch_size=128)
        result2 = consistent_lax_map(f, x, fixed_batch_size=64)
        assert jnp.allclose(result1, result2), "Results differ for different fixed batch sizes!"
        print("Consistency test passed.")

    test_consistent_results()