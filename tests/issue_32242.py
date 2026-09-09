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

HOW TO DEBUG THIS FILE END-TO-END IN VSCODE:

This file is set up with debugpy for remote debugging. Here's how to use it:

1. Make sure .vscode/launch.json has an attach configuration:
   {
       "name": "Attach to issue_32242",
       "type": "debugpy",
       "request": "attach",
       "connect": {
           "host": "localhost",
           "port": 5678
       },
       "justMyCode": false
   }

2. Set breakpoints in this file where you want to pause execution
   (click to the left of the line numbers in VSCode)

3. Start the script in the background:
   /home/sheil/personal_projects/jax/venv/bin/python tests/issue_32242.py &

4. In VSCode, open the Run and Debug panel (Ctrl+Shift+D)

5. Select "Attach to issue_32242" from the dropdown at the top

6. Click the green play button (or press F5) to attach the debugger

7. The script will resume execution and hit any breakpoints you set.
   You can then:
   - Step through code (F10 = step over, F11 = step into, F5 = continue)
   - Inspect variables in the Variables panel
   - Evaluate expressions in the Debug Console

The debugpy.listen(5678) and debugpy.wait_for_client() lines at the top of this
file make it wait for a debugger to attach before continuing execution.
"""

import jax
import jax.numpy as jnp

import debugpy
debugpy.listen(5678)
print("Waiting for debugger attach...")
debugpy.wait_for_client()


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