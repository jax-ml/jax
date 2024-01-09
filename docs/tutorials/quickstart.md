---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Quickstart

**JAX a library for array-oriented numerical computation (*a la* [NumPy](https://numpy.org/)), with automatic differentiation and JIT compilation to enable high-performance machine learning research**.

This document provides a quick overview of essential JAX features, so you can get started with JAX quickly:

* JAX provides a unified NumPy-like interface to computations that run on CPU, GPU, or TPU, in local or distributed settings.
* JAX features built-in Just-In-Time (JIT) compilation via [Open XLA](https://github.com/openxla), an open-source machine learning compiler ecosystem.
* JAX functions support efficient evaluation of gradients via its automatic differentiation transformations.
* JAX functions can be automatically vectorized to efficiently map them over arrays representing batches of inputs.

## Installation

JAX can be installed for CPU on Linux, Windows, and macOS directly from the [Python Package Index](https://pypi.org/project/jax/):
```
pip install "jax[cpu]"
```
For more detailed installation information, including installation with GPU support, check out {ref}`installation`.

## JAX as NumPy

Most JAX usage is through the familiar {mod}`jax.numpy` API, which is typically imported under the `jnp` alias:

```{code-cell}
import jax.numpy as jnp
```

With this import, you can immediately use JAX in a similar manner to typical NumPy programs,
including using NumPy-style array creation functions, Python functions and operators, and
array attributes and methods:

```{code-cell}
def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(5.0)
print(selu(x))
```

You'll find a few differences between JAX arrays and NumPy arrays once you begin digging-in;
these are explored in  [🔪 JAX - The Sharp Bits 🔪](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Using {func}`~jax.jit` to speed up functions
JAX runs transparently on the GPU or TPU (falling back to CPU if you don't have one). However, in the above example, JAX is dispatching kernels to the chip one operation at a time. If we have a sequence of operations, we can use the `@jit` decorator to compile multiple operations together using XLA.

We can use IPython's `%timeit` to quickly benchmark our `selu` function, using `block_until_ready()` to
account for JAX's dynamic dispatch (See {ref}`async-dispatch`):

```{code-cell}
from jax import random

key = random.key(1701)
x = random.normal(key, (1_000_000,))
%timeit selu(x).block_until_ready()
```

(notice we've used {mod}`jax.random` to generate some random numbers; for details on
how to generate random numbers in JAX, check out {ref}`pseudorandom-numbers`).

We can speed the execution of this function with `@jit`, which will jit-compile the
first time `selu` is called and will be cached thereafter.

```{code-cell}
from jax import jit

selu_jit = jit(selu)
_ = selu_jit(x)  # compiles on first call
%timeit selu_jit(x).block_until_ready()
```

The above timing represent execution on CPU, but the same code can be run on GPU or TPU for
an even greater speedup.

For more on JIT compilation in JAX, check out {ref}`jit-compilation`.

## Taking derivatives with {func}`~jax.grad`

In addition to evaluating numerical functions, we can also to transform them.
One transformation is [automatic differentiation (autodiff)](https://en.wikipedia.org/wiki/Automatic_differentiation).
In JAX, you can compute gradients with the {func}`~jax.grad` function.

```{code-cell}
from jax import grad

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))
```

Let's verify with finite differences that our result is correct.

```{code-cell}
def first_finite_differences(f, x, eps=1E-3):
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])

print(first_finite_differences(sum_logistic, x_small))
```

The {func}`~jax.grad` and {func}`~jax.jit` transformations compose and can be mixed arbitrarily.
In the above example we jitted `sum_logistic` and then took its derivative. We can go further:

```{code-cell}
print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))
```

For more advanced autodiff, you can use {func}`jax.vjp` for reverse-mode vector-Jacobian products and {func}`jax.jvp` for forward-mode Jacobian-vector products.
The two can be composed arbitrarily with one another, and with other JAX transformations.
Here's one way to compose them to make a function that efficiently computes full Hessian matrices:

```{code-cell}
from jax import jacfwd, jacrev
def hessian(fun):
  return jit(jacfwd(jacrev(fun)))
print(hessian(sum_logistic)(x_small))
```

This kind of composition produces efficient code in practice; this is more-or-less how JAX's built-in {func}`jax.hessian` function is implemented.

For more on automatic differentiation in JAX, check out {ref}`automatic-differentiation`.

## Auto-vectorization with {func}`~jax.vmap`

Another useful transformation is {func}`~jax.vmap`, the vectorizing map.
It has the familiar semantics of mapping a function along array axes, but instead of keeping the loop on the outside, it pushes the loop down into a function’s primitive operations for better performance.
When composed with {func}`~jax.jit`, it can be just as fast as adding the batch dimensions manually.

We're going to work with a simple example, and promote matrix-vector products into matrix-matrix products using {func}`~jax.vmap`.
Although this is easy to do by hand in this specific case, the same technique can apply to more complicated functions.

```{code-cell}
mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
  return jnp.dot(mat, v)
```

Given a function such as `apply_matrix`, we can loop over a batch dimension in Python, but usually the performance of doing so is poor.

```{code-cell}
def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')
%timeit naively_batched_apply_matrix(batched_x).block_until_ready()
```

We know how to batch this operation manually.
In this case, `jnp.dot` handles extra batch dimensions transparently.

```{code-cell}
@jit
def batched_apply_matrix(v_batched):
  return jnp.dot(v_batched, mat.T)

print('Manually batched')
%timeit batched_apply_matrix(batched_x).block_until_ready()
```

However, suppose we had a more complicated function without batching support. We can use {func}`~jax.vmap` to add batching support automatically.

```{code-cell}
from jax import vmap

@jit
def vmap_batched_apply_matrix(v_batched):
  return vmap(apply_matrix)(v_batched)

print('Auto-vectorized with vmap')
%timeit vmap_batched_apply_matrix(batched_x).block_until_ready()
```

Of course, {func}`~jax.vmap` can be arbitrarily composed with {func}`~jax.jit`, {func}`~jax.grad`, and any other JAX transformation.

For more on automatic vectorization in JAX, check out {ref}`automatic-vectorization`.

This is just a taste of what JAX can do. We're really excited to see what you do with it!
