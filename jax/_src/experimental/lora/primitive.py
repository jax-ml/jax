import jax
import jax.numpy as jnp
from jax import custom_vjp

@custom_vjp
def lora_dot(x, w_frozen, lora_a, lora_b, scale):
    """
    Standard Forward Pass: y = x @ (w_frozen + (lora_b @ lora_a))
    But calculated as: y = (x @ w_frozen) + (x @ lora_a.T @ lora_b.T) * scale
    """
    # We use the distributive property to keep memory usage low
    base_out = jnp.dot(x, w_frozen.T)
    # Note: Using transposed A and B to match standard LoRA convention (r, in) and (out, r)
    lora_out = jnp.dot(jnp.dot(x, lora_a.T), lora_b.T) * scale
    return base_out + lora_out

def lora_dot_fwd(x, w_frozen, lora_a, lora_b, scale):
    """The forward pass for Autodiff: returns result and 'residue' for backward pass."""
    res = (x, w_frozen, lora_a, lora_b, scale)
    return lora_dot(x, w_frozen, lora_a, lora_b, scale), res

def lora_dot_bwd(res, g):
    """The backward pass (VJP) with corrected matrix dimensions."""
    x, w_frozen, lora_a, lora_b, scale = res
    
    # g shape: (batch, out_features) -> (1, 8)
    
    # 1. Grad w.r.t x: (g @ w_frozen) + scale * (g @ lora_b @ lora_a)
    # (1, 8) @ (8, 16) + (1, 8) @ (8, 4) @ (4, 16) = (1, 16)
    dx = jnp.dot(g, w_frozen) + scale * jnp.dot(jnp.dot(g, lora_b), lora_a)
    
    # 2. Grad w.r.t lora_a: scale * (g @ lora_b).T @ x
    # (1, 8) @ (8, 4) = (1, 4) 
    # (1, 4).T @ (1, 16) = (4, 16) -> Matches lora_a shape!
    da = scale * jnp.dot(jnp.dot(g, lora_b).T, x)
    
    # 3. Grad w.r.t lora_b: scale * g.T @ (x @ lora_a.T)
    # (8, 1) @ ((1, 16) @ (16, 4)) = (8, 1) @ (1, 4) = (8, 4) -> Matches lora_b shape!
    db = scale * jnp.dot(g.T, jnp.dot(x, lora_a.T))
    
    return dx, None, da, db, None

lora_dot.defvjp(lora_dot_fwd, lora_dot_bwd)