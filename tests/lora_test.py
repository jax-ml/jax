import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.experimental.lora import lora_dot
import pytest

def test_lora_dot_gradients():
    key = jax.random.PRNGKey(0)
    in_features, out_features, rank = 16, 8, 4
    
    # Initialize parameters
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x = jax.random.normal(k1, (1, in_features))
    w_frozen = jax.random.normal(k2, (out_features, in_features))
    lora_a = jax.random.normal(k3, (rank, in_features))
    lora_b = jax.random.normal(k4, (out_features, rank))
    scale = 2.0

    # Define a simple MSE loss function
    def loss_fn(x, w, a, b, s):
        out = lora_dot(x, w, a, b, s)
        return jnp.sum(out**2)

    # Calculate gradients
    # We ask for grads for (x, w_frozen, lora_a, lora_b)
    grads = grad(loss_fn, argnums=(0, 1, 2, 3))(x, w_frozen, lora_a, lora_b, scale)
    
    dx, dw, da, db = grads

    # Assertions
    # 1. dw must be None or a zero-array because of our custom VJP
    if dw is not None:
        assert jnp.all(dw == 0), "Frozen weight gradient should be zero!"
    
    # 2. da and db must NOT be zero (they are learning)
    assert jnp.any(da != 0), "LoRA A gradient should not be zero"
    assert jnp.any(db != 0), "LoRA B gradient should not be zero"
    
    print("✓ Gradients verified: W is frozen, A and B are learning.")

@jit
def test_lora_jit_compatibility():
    # Ensure XLA can compile our custom VJP without errors
    x = jnp.ones((1, 4))
    w = jnp.ones((2, 4))
    a = jnp.ones((2, 4))
    b = jnp.ones((2, 2))
    res = lora_dot(x, w, a, b, 1.0)
    assert res.shape == (1, 2)
    print("✓ JIT compilation successful.")

if __name__ == "__main__":
    test_lora_dot_gradients()
    test_lora_jit_compatibility()