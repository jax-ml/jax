---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.0
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "zMIrmiaZxiJC"}

# Automatic vectorization in JAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/master/docs/jax-101/03-vectorization.ipynb)

*Authors: Matteo Hessel*

In the previous section we discussed JIT compilation via the `jax.jit` function. This notebook discusses another of JAX's transforms: vectorization via `jax.vmap`.

+++ {"id": "Kw-_imBrx4nN"}

## Manual Vectorization

Consider the following simple code that computes the convolution of two one-dimensional vectors:

```{code-cell} ipython3
:id: 5Obro91lwE_s
:outputId: 061983c6-2faa-4a54-83a5-d2a823f61087

import jax
import jax.numpy as jnp

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)

convolve(x, w)
```

+++ {"id": "z_nPhEhLRysk"}

Suppose we would like to apply this function to a batch of weights `w` to a batch of vectors `x`.

```{code-cell} ipython3
:id: rHQJnnrVUbxE

xs = jnp.stack([x, x])
ws = jnp.stack([w, w])
```

+++ {"id": "ghaJQW1aUfPi"}

The most naive option would be to simply loop over the batch in Python:

```{code-cell} ipython3
:id: yM-IycdlzGyJ
:outputId: 07ed6ffc-0265-45ef-d585-4b5fa7d221f1

def manually_batched_convolve(xs, ws):
  output = []
  for i in range(xs.shape[0]):
    output.append(convolve(xs[i], ws[i]))
  return jnp.stack(output)

manually_batched_convolve(xs, ws)
```

+++ {"id": "VTh0l_1SUlh4"}

This produces the correct result, however it is not very efficient.

In order to batch the computation efficiently, you would normally have to rewrite the function manually to ensure it is done in vectorized form. This is not particularly difficult to implement, but does involve changing how the function treats indices, axes, and other parts of the input.

For example, we could manually rewrite `convolve()` to support vectorized computation across the batch dimension as follows:

```{code-cell} ipython3
:id: I4Wd9nrcTRRL
:outputId: 0b037b43-7b41-4625-f9e0-a6e0dbc4c65a

def manually_vectorized_convolve(xs, ws):
  output = []
  for i in range(1, xs.shape[-1] -1):
    output.append(jnp.sum(xs[:, i-1:i+2] * ws, axis=1))
  return jnp.stack(output, axis=1)

manually_vectorized_convolve(xs, ws)
```

+++ {"id": "DW-RJ2Zs2QVu"}

Such re-implementation is messy and error-prone; fortunately JAX provides another way.

+++ {"id": "2oVLanQmUAo_"}

## Automatic Vectorization

In JAX, the `jax.vmap` transformation is designed to generate such a vectorized implementation of a function automatically:

```{code-cell} ipython3
:id: Brl-BoTqSQDw
:outputId: af608dbb-27f2-4fbc-f225-79f3101b13ff

auto_batch_convolve = jax.vmap(convolve)

auto_batch_convolve(xs, ws)
```

+++ {"id": "7aVAy7332lFj"}

It does this by tracing the function similarly to `jax.jit`, and automatically adding batch axes at the beginning of each input.

If the batch dimension is not the first, you may use the `in_axes` and `out_axes` arguments to specify the location of the batch dimension in inputs and outputs. These may be an integer if the batch axis is the same for all inputs and outputs, or lists, otherwise.

```{code-cell} ipython3
:id: _VEEm1CGT2n0
:outputId: 751e0fbf-bdfb-41df-9436-4da5de23123f

auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)
wst = jnp.transpose(ws)

auto_batch_convolve_v2(xst, wst)
```

+++ {"id": "-gNiLuxzSX32"}

`jax.vmap` also supports the case where only one of the arguments is batched: for example, if you would like to convolve to a single set of weights `w` with a batch of vectors `x`; in this case the `in_axes` argument can be set to `None`:

```{code-cell} ipython3
:id: 2s2YDsamSxki
:outputId: 5c70879b-5cce-4549-e38a-f45dbe663ab2

batch_convolve_v3 = jax.vmap(convolve, in_axes=[0, None])

batch_convolve_v3(xs, w)
```

+++ {"id": "bsxT4hA6RTCG"}

## Combining transformations

As with all JAX transformations, `jax.jit` and `jax.vmap` are designed to be composable, which means you can wrap a vmapped function with `jit`, or a JITted function with `vmap`, and everything will work correctly:

```{code-cell} ipython3
:id: gsC-Myg0RVdj
:outputId: cbdd384e-6633-4cea-b1a0-a01ad934a768

jitted_batch_convolve = jax.jit(auto_batch_convolve)

jitted_batch_convolve(xs, ws)
```
