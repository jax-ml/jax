"""
Steps to run this test with local JAX version (with CUDA/cuDNN support):

1. Copy an existing venv that has CUDA-enabled JAX/jaxlib installed:
   cp -r /home/sheil/personal_projects/intro_to_jax/venv /home/sheil/personal_projects/jax/venv

2. Install the local JAX version in editable mode in the copied venv:
   /home/sheil/personal_projects/jax/venv/bin/pip install -e /home/sheil/personal_projects/jax

3. Run the test using the venv's Python interpreter:
   /home/sheil/personal_projects/jax/venv/bin/python tests/issue_32242.py

Note: This test requires cuDNN because it uses implementation='cudnn' for dot_product_attention.
If cuDNN is not available, the test will fail with "RuntimeError: cuDNN is not detected."
"""

import jax
import jax.numpy as jnp

def f(b, x):
  out = jax.nn.dot_product_attention(x, x, x, bias=b, implementation='cudnn')
  return jnp.sum(out)

f_grad = jax.jit(jax.grad(f))

seq_len = 128
batch = 8
heads = 4

x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, heads, 32), dtype=jnp.bfloat16)
bias = jax.random.normal(jax.random.PRNGKey(0), (1, heads, seq_len, seq_len), dtype=jnp.float32)

grad = f_grad(bias, x)
jnp.isfinite(grad).all()
print(grad)