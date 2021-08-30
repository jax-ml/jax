---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Transformations

```{currentmodule} jax
```

At its core, JAX is an extensible system for transforming numerical functions.
This section will discuss four transformations that are of primary interest:
{func}`grad`, {func}`jit`, {func}`vmap`, and {func}`pmap`.

## Automatic differentiation with `grad`

JAX has roughly the same API as [Autograd](https://github.com/hips/autograd).
The most popular function is {func}`jax.grad` for {term}`reverse-mode<VJP>` gradients:

```{code-cell}
:tags: [remove-cell]
def _setup():
  # Set up runtime to mimic an 8-core machine for pmap example below:
  import os
  flags = os.environ.get('XLA_FLAGS', '')
  os.environ['XLA_FLAGS'] = flags + " --xla_force_host_platform_device_count=8"

  # consume the CPU warning
  import jax
  _ = jax.numpy.arange(10)
_setup()
del _setup
```

```{code-cell}
from jax import grad
import jax.numpy as jnp

def tanh(x):  # Define a function
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)  # Obtain its gradient function
print(grad_tanh(1.0))   # Evaluate it at x = 1.0
```

You can differentiate to any order with {func}`grad`.

```{code-cell}
print(grad(grad(grad(tanh)))(1.0))
```

For more advanced autodiff, you can use {func}`jax.vjp` for
{term}`reverse-mode vector-Jacobian products<VJP>` and {func}`jax.jvp` for
{term}`forward-mode Jacobian-vector products<JVP>`. The two can be composed arbitrarily with
one another, and with other JAX transformations. Here's one way to compose those
to make a function that efficiently computes [full Hessian
matrices](https://jax.readthedocs.io/en/latest/jax.html#jax.hessian):

```{code-cell}
from jax import jit, jacfwd, jacrev

def hessian(fun):
  return jit(jacfwd(jacrev(fun)))
```

As with [Autograd](https://github.com/hips/autograd), you're free to use
differentiation with Python control structures:

```{code-cell}
def abs_val(x):
  if x > 0:
    return x
  else:
    return -x

abs_val_grad = grad(abs_val)
print(abs_val_grad(1.0))
```
```{code-cell}
print(abs_val_grad(-1.0))
```

See the [reference docs on automatic
differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
and the [JAX Autodiff
Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
for more.

## Compilation with `jit`

You can use XLA to compile your functions end-to-end with {func}`jax.jit`
used either as an `@jit` decorator or as a higher-order function.

```{code-cell}
import jax.numpy as jnp
from jax import jit

def slow_f(x):
  # Element-wise ops see a large benefit from fusion
  return x * x + x * 2.0

x = jnp.ones((5000, 5000))
%timeit slow_f(x).block_until_ready()
```
```{code-cell}
fast_f = jit(slow_f)

# Results are the same
assert jnp.allclose(slow_f(x), fast_f(x))

%timeit fast_f(x).block_until_ready()
```

You can mix {func}`jit` and {func}`grad` and any other JAX transformation however you like.

Using {func}`jit` puts constraints on the kind of Python control flow the function can use; see
[🔪 JAX - The Sharp Bits 🔪](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-+-JIT) for more.

## Auto-vectorization with `vmap`

{func}`jax.vmap` is the vectorizing map.
It has the familiar semantics of mapping a function along array axes, but
instead of keeping the loop on the outside, it pushes the loop down into a
function’s primitive operations for better performance.

Using {func}`vmap` can save you from having to carry around batch dimensions in your
code. For example, consider this simple *unbatched* neural network prediction
function:

```{code-cell}
def predict(params, input_vec):
  assert input_vec.ndim == 1
  activations = input_vec
  for W, b in params:
    outputs = jnp.dot(W, activations) + b  # `input_vec` on the right-hand side!
    activations = jnp.tanh(outputs)
  return outputs
```

We often instead write `jnp.dot(inputs, W)` to allow for a batch dimension on the
left side of `inputs`, but we’ve written this particular prediction function to
apply only to single input vectors. If we wanted to apply this function to a
batch of inputs at once, semantically we could just write

```{code-cell}
:tags: [hide-cell]
# Create some sample inputs & parameters
import numpy as np
k, N = 10, 5
input_batch = np.random.rand(k, N)
params = [
  (np.random.rand(N, N), np.random.rand(N)),
  (np.random.rand(N, N), np.random.rand(N)),
]
```

```{code-cell}
from functools import partial
predictions = jnp.stack(list(map(partial(predict, params), input_batch)))
```

But pushing one example through the network at a time would be slow! It’s better
to vectorize the computation, so that at every layer we’re doing matrix-matrix
multiplication rather than matrix-vector multiplication.

The {func}`vmap` function does that transformation for us. That is, if we write:

```{code-cell}
from jax import vmap
predictions = vmap(partial(predict, params))(input_batch)
# or, alternatively
predictions = vmap(predict, in_axes=(None, 0))(params, input_batch)
```

then the {func}`vmap` function will push the outer loop inside the function, and our
machine will end up executing matrix-matrix multiplications exactly as if we’d
done the batching by hand.

It’s easy enough to manually batch a simple neural network without {func}`vmap`, but
in other cases manual vectorization can be impractical or impossible. Take the
problem of efficiently computing per-example gradients: that is, for a fixed set
of parameters, we want to compute the gradient of our loss function evaluated
separately at each example in a batch. With {func}`vmap`, it’s easy:

```{code-cell}
:tags: [hide-cell]
# create a sample loss function & inputs
def loss(params, x, y0):
  y = predict(params, x)
  return jnp.sum((y - y0) ** 2)

inputs = np.random.rand(k, N)
targets = np.random.rand(k, N)
```

```{code-cell}
per_example_gradients = vmap(partial(grad(loss), params))(inputs, targets)
```

Of course, {func}`vmap` can be arbitrarily composed with {func}`jit`, {func}`grad`,
and any other JAX transformation! We use {func}`vmap` with both forward- and reverse-mode automatic
differentiation for fast Jacobian and Hessian matrix calculations in
{func}`jax.jacfwd`, {func}`jax.jacrev`, and {func}`jax.hessian`.

## SPMD programming with `pmap`

For parallel programming of multiple accelerators, like multiple GPUs, use
{func}`jax.pmap`.
With {func}`pmap` you write single-program multiple-data (SPMD) programs, including
fast parallel collective communication operations. Applying {func}`pmap` will mean
that the function you write is compiled by XLA (similarly to {func}`jit`), then
replicated and executed in parallel across devices.

Here's an example on an 8-core machine:

```{code-cell}
from jax import random, pmap
import jax.numpy as jnp

# Create 8 random 5000 x 6000 matrices, one per core
keys = random.split(random.PRNGKey(0), 8)
mats = pmap(lambda key: random.normal(key, (5000, 6000)))(keys)

# Run a local matmul on each device in parallel (no data transfer)
result = pmap(lambda x: jnp.dot(x, x.T))(mats)  # result.shape is (8, 5000, 5000)

# Compute the mean on each device in parallel and print the result
print(pmap(jnp.mean)(result))
```

In addition to expressing pure maps, you can use fast {ref}`jax-parallel-operators` between devices:

```{code-cell}
from functools import partial
from jax import lax

@partial(pmap, axis_name='i')
def normalize(x):
  return x / lax.psum(x, 'i')

print(normalize(jnp.arange(4.)))
```

You can even [nest `pmap` functions](https://colab.research.google.com/github/google/jax/blob/main/cloud_tpu_colabs/Pmap_Cookbook.ipynb#scrollTo=MdRscR5MONuN) for more sophisticated communication patterns.

It all composes, so you're free to differentiate through parallel computations:

```{code-cell}
from jax import grad

@pmap
def f(x):
  y = jnp.sin(x)
  @pmap
  def g(z):
    return jnp.cos(z) * jnp.tan(y.sum()) * jnp.tanh(x).sum()
  return grad(lambda w: jnp.sum(g(w)))(x)

x = jnp.arange(8.0).reshape(2, 4)
print(f(x))
```
```{code-cell}
print(grad(lambda x: f(x).sum())(x))
```

When reverse-mode differentiating a {func}`pmap` function (e.g. with {func}`grad`), the
backward pass of the computation is parallelized just like the forward pass.

See the [SPMD Cookbook](https://colab.research.google.com/github/google/jax/blob/main/cloud_tpu_colabs/Pmap_Cookbook.ipynb)
and the [SPMD MNIST classifier from scratch example](https://github.com/google/jax/blob/main/examples/spmd_mnist_classifier_fromscratch.py)
for more.
