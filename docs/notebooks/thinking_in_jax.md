---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "LQHmwePqryRU"}

# How to think in JAX

<!--* freshness: { reviewed: '2024-04-08' } *-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax/blob/main/docs/notebooks/thinking_in_jax.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/jax-ml/jax/blob/main/docs/notebooks/thinking_in_jax.ipynb)

JAX provides a simple and powerful API for writing accelerated numerical code, but working effectively in JAX sometimes requires extra consideration. This document is meant to help build a ground-up understanding of how JAX operates, so that you can use it more effectively.

+++ {"id": "nayIExVUtsVD"}

## JAX vs. NumPy

**Key concepts:**

- JAX provides a NumPy-inspired interface for convenience.
- Through duck-typing, JAX arrays can often be used as drop-in replacements of NumPy arrays.
- Unlike NumPy arrays, JAX arrays are always immutable.

The arrays themselves are implemented as different Python types:

```{code-cell} ipython3
:id: kZaOXL7-uvUP
:outputId: 7fd4dd8e-4194-4983-ac6b-28059f8feb90

import numpy as np
import jax.numpy as jnp

x_np = np.linspace(0, 10, 1000)
x_jnp = jnp.linspace(0, 10, 1000)
```

```{code-cell} ipython3
:id: PjFFunI7xNe8
:outputId: d3b0007e-7997-45c0-d4b8-9f5699cedcbc

type(x_np)
```

```{code-cell} ipython3
:id: kpv5K7QYxQnX
:outputId: ba68a1de-f938-477d-9942-83a839aeca09

type(x_jnp)
```

+++ {"id": "Mx94Ri7euEZm"}

Python's [duck-typing](https://en.wikipedia.org/wiki/Duck_typing) allows JAX arrays and NumPy arrays to be used interchangeably in many places.

However, there is one important difference between JAX and NumPy arrays: JAX arrays are immutable, meaning that once created their contents cannot be changed.

Here is an example of mutating an array in NumPy:

```{code-cell} ipython3
:id: fzp-y1ZVyGD4
:outputId: 6eb76bf8-0edd-43a5-b2be-85a79fb23190

# NumPy: mutable arrays
x = np.arange(10)
x[0] = 10
print(x)
```

+++ {"id": "nQ-De0xcJ1lT"}

The equivalent in JAX results in an error, as JAX arrays are immutable:

```{code-cell} ipython3
:id: l2AP0QERb0P7
:outputId: 528a8e5f-538f-4739-fe95-1c3605ba8c8a

%xmode minimal
```

```{code-cell} ipython3
:id: pCPX0JR-yM4i
:outputId: c7bf4afd-8b7f-4dac-d065-8189679861d6
:tags: [raises-exception]

# JAX: immutable arrays
x = jnp.arange(10)
x[0] = 10
```

+++ {"id": "yRYF0YgO3F4H"}

For updating individual elements, JAX provides an [indexed update syntax](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax-numpy-ndarray-at) that returns an updated copy:

```{code-cell} ipython3
:id: 8zqPEAeP3UK5
:outputId: 20a40c26-3419-4e60-bd2c-83ad30bd7650

y = x.at[0].set(10)
print(x)
print(y)
```

## JAX arrays (`jax.Array`)

**Key concepts:**
- Create arrays using JAX API functions.
- JAX array objects have a `devices` attribute that indicates where the array is stored.
- JAX arrays can be *sharded* across multiple devices for parallel computation.

The default array implementation in JAX is [`jax.Array`](https://docs.jax.dev/en/latest/_autosummary/jax.Array.html#jax.Array). In many ways it is similar to
the [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) type that you may be familiar with from the NumPy package, but it
has some important differences.

### Array creation

We typically don't call the `jax.Array` constructor directly, but rather create arrays via JAX API functions.
For example, [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html#module-jax.numpy) provides familiar NumPy-style array construction functionality
such as `jax.numpy.zeros`, `jax.numpy.linspace`, `jax.numpy.arange`, etc.

```{code-cell} ipython3
import jax
import jax.numpy as jnp

x = jnp.arange(5)
isinstance(x, jax.Array)
```

If you use Python type annotations in your code, `jax.Array` is the appropriate
annotation for jax array objects (see [`jax.typing`](https://docs.jax.dev/en/latest/jax.typing.html#module-jax.typing) for more discussion).

+++

### Array devices and sharding

JAX Array objects have a `devices` method that lets you inspect where the contents of the array are stored. In the simplest cases, this will be a single CPU device:

```{code-cell} ipython3
x.devices()
```

In general, an array may be *sharded* across multiple devices, in a manner that can be inspected via the `sharding` attribute:

```{code-cell} ipython3
x.sharding
```

Here the array is on a single device, but in general a JAX array can be
sharded across multiple devices, or even multiple hosts.
To read more about sharded arrays and parallel computation, refer to {ref}`sharded-computation`.

+++ {"id": "886BGDPeyXCu"}

## NumPy, lax & XLA: JAX API layering

**Key concepts:**

- `jax.numpy` is a high-level wrapper that provides a familiar interface.
- `jax.lax` is a lower-level API that is stricter and often more powerful.
- All JAX operations are implemented in terms of operations in [XLA](https://www.tensorflow.org/xla/) – the Accelerated Linear Algebra compiler.

+++ {"id": "BjE4m2sZy4hh"}

If you look at the source of `jax.numpy`, you'll see that all the operations are eventually expressed in terms of functions defined in `jax.lax`. You can think of `jax.lax` as a stricter, but often more powerful, API for working with multi-dimensional arrays.

For example, while `jax.numpy` will implicitly promote arguments to allow operations between mixed data types, `jax.lax` will not:

```{code-cell} ipython3
:id: c6EFPcj12mw0
:outputId: 827d09eb-c8aa-43bc-b471-0a6c9c4f6601

import jax.numpy as jnp
jnp.add(1, 1.0)  # jax.numpy API implicitly promotes mixed types.
```

```{code-cell} ipython3
:id: 0VkqlcXL2qSp
:outputId: 7e1e9233-2fe1-46a8-8eb1-1d1dbc54b58c
:tags: [raises-exception]

from jax import lax
lax.add(1, 1.0)  # jax.lax API requires explicit type promotion.
```

+++ {"id": "aC9TkXaTEu7A"}

If using `jax.lax` directly, you'll have to do type promotion explicitly in such cases:

```{code-cell} ipython3
:id: 3PNQlieT81mi
:outputId: 4bd2b6f3-d2d1-44cb-f8ee-18976ae40239

lax.add(jnp.float32(1), 1.0)
```

+++ {"id": "M3HDuM4x2eTL"}

Along with this strictness, `jax.lax` also provides efficient APIs for some more general operations than are supported by NumPy.

For example, consider a 1D convolution, which can be expressed in NumPy this way:

```{code-cell} ipython3
:id: Bv-7XexyzVCN
:outputId: d570f64a-ca61-456f-8cab-6cd643cb8ea1

x = jnp.array([1, 2, 1])
y = jnp.ones(10)
jnp.convolve(x, y)
```

+++ {"id": "0GPqgT7S0q8r"}

Under the hood, this NumPy operation is translated to a much more general convolution implemented by [`lax.conv_general_dilated`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv_general_dilated.html):

```{code-cell} ipython3
:id: pi4f6ikjzc3l
:outputId: 0bb56ae2-7837-4c04-ff8b-6cbc0565b7d7

from jax import lax
result = lax.conv_general_dilated(
    x.reshape(1, 1, 3).astype(float),  # note: explicit promotion
    y.reshape(1, 1, 10),
    window_strides=(1,),
    padding=[(len(y) - 1, len(y) - 1)])  # equivalent of padding='full' in NumPy
result[0, 0]
```

+++ {"id": "7mdo6ycczlbd"}

This is a batched convolution operation designed to be efficient for the types of convolutions often used in deep neural nets. It requires much more boilerplate, but is far more flexible and scalable than the convolution provided by NumPy (See [Convolutions in JAX](https://docs.jax.dev/en/latest/notebooks/convolutions.html) for more detail on JAX convolutions).

At their heart, all `jax.lax` operations are Python wrappers for operations in XLA; here, for example, the convolution implementation is provided by [XLA:ConvWithGeneralPadding](https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution).
Every JAX operation is eventually expressed in terms of these fundamental XLA operations, which is what enables just-in-time (JIT) compilation.

+++ {"id": "NJfWa2PktD5_"}

## To JIT or not to JIT

**Key concepts:**

- By default JAX executes operations one at a time, in sequence.
- Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized together and run at once.
- Not all JAX code can be JIT compiled, as it requires array shapes to be static & known at compile time.

The fact that all JAX operations are expressed in terms of XLA allows JAX to use the XLA compiler to execute blocks of code very efficiently.

For example, consider this function that normalizes the rows of a 2D matrix, expressed in terms of `jax.numpy` operations:

```{code-cell} ipython3
:id: SQj_UKGc-7kQ

import jax.numpy as jnp

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)
```

+++ {"id": "0yVo_OKSAolW"}

A just-in-time compiled version of the function can be created using the `jax.jit` transform:

```{code-cell} ipython3
:id: oHLwGmhZAnCY

from jax import jit
norm_compiled = jit(norm)
```

+++ {"id": "Q3H9ig5GA2Ms"}

This function returns the same results as the original, up to standard floating-point accuracy:

```{code-cell} ipython3
:id: oz7zzyS3AwMc
:outputId: ed1c796c-59f8-4238-f6e2-f54330edadf0

np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
np.allclose(norm(X), norm_compiled(X), atol=1E-6)
```

+++ {"id": "3GvisB-CA9M8"}

But due to the compilation (which includes fusing of operations, avoidance of allocating temporary arrays, and a host of other tricks), execution times can be orders of magnitude faster in the JIT-compiled case (note the use of `block_until_ready()` to account for JAX's [asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)):

```{code-cell} ipython3
:id: 6mUB6VdDAEIY
:outputId: 1050a69c-e713-44c1-b3eb-1ef875691978

%timeit norm(X).block_until_ready()
%timeit norm_compiled(X).block_until_ready()
```

+++ {"id": "B1eGBGn0tMba"}

That said, `jax.jit` does have limitations: in particular, it requires all arrays to have static shapes. That means that some JAX operations are incompatible with JIT compilation.

For example, this operation can be executed in op-by-op mode:

```{code-cell} ipython3
:id: YfZd9mW7CSKM
:outputId: 6fdbfde4-7cde-447f-badf-26e1f8db288d

def get_negatives(x):
  return x[x < 0]

x = jnp.array(np.random.randn(10))
get_negatives(x)
```

+++ {"id": "g6niKxoQC2mZ"}

But it returns an error if you attempt to execute it in jit mode:

```{code-cell} ipython3
:id: yYWvE4rxCjPK
:outputId: 9cf7f2d4-8f28-4265-d701-d52086cfd437
:tags: [raises-exception]

jit(get_negatives)(x)
```

+++ {"id": "vFL6DNpECfVz"}

This is because the function generates an array whose shape is not known at compile time: the size of the output depends on the values of the input array, and so it is not compatible with JIT.

+++ {"id": "BzBnKbXwXjLV"}

## JIT mechanics: tracing and static variables

**Key concepts:**

- JIT and other JAX transforms work by *tracing* a function to determine its effect on inputs of a specific shape and type.

- Variables that you don't want to be traced can be marked as *static*

To use `jax.jit` effectively, it is useful to understand how it works. Let's put a few `print()` statements within a JIT-compiled function and then call the function:

```{code-cell} ipython3
:id: TfjVIVuD4gnc
:outputId: 9f4ddcaa-8ab7-4984-afb6-47fede5314ea

@jit
def f(x, y):
  print("Running f():")
  print(f"  x = {x}")
  print(f"  y = {y}")
  result = jnp.dot(x + 1, y + 1)
  print(f"  result = {result}")
  return result

x = np.random.randn(3, 4)
y = np.random.randn(4)
f(x, y)
```

+++ {"id": "Ts1fP45A40QV"}

Notice that the print statements execute, but rather than printing the data we passed to the function, though, it prints *tracer* objects that stand-in for them.

These tracer objects are what `jax.jit` uses to extract the sequence of operations specified by the function. Basic tracers are stand-ins that encode the **shape** and **dtype** of the arrays, but are agnostic to the values. This recorded sequence of computations can then be efficiently applied within XLA to new inputs with the same shape and dtype, without having to re-execute the Python code.

When we call the compiled function again on matching inputs, no re-compilation is required and nothing is printed because the result is computed in compiled XLA rather than in Python:

```{code-cell} ipython3
:id: xGntvzNH7skE
:outputId: 43aaeee6-3853-4b00-fb2b-646df695204a

x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
f(x2, y2)
```

+++ {"id": "9EB9WkRX7fm0"}

The extracted sequence of operations is encoded in a JAX expression, or *jaxpr* for short. You can view the jaxpr using the `jax.make_jaxpr` transformation:

```{code-cell} ipython3
:id: 89TMp_Op5-JZ
:outputId: 48212815-059a-4af1-de82-cd39ecac264a

from jax import make_jaxpr

def f(x, y):
  return jnp.dot(x + 1, y + 1)

make_jaxpr(f)(x, y)
```

+++ {"id": "0Oq9S4MZ90TL"}

Note one consequence of this: because JIT compilation is done *without* information on the content of the array, control flow statements in the function cannot depend on traced values. For example, this fails:

```{code-cell} ipython3
:id: A0rFdM95-Ix_
:outputId: e37bf04e-6a6a-4536-e423-f082f52d5f11
:tags: [raises-exception]

@jit
def f(x, neg):
  return -x if neg else x

f(1, True)
```

+++ {"id": "DkTO9m8j-TYI"}

If there are variables that you would not like to be traced, they can be marked as static for the purposes of JIT compilation:

```{code-cell} ipython3
:id: K1C7ZnVv-lbv
:outputId: e9d6cce3-b036-43da-ad99-887af9625ab0

from functools import partial

@partial(jit, static_argnums=(1,))
def f(x, neg):
  return -x if neg else x

f(1, True)
```

+++ {"id": "dD7p4LRsGzhx"}

Note that calling a JIT-compiled function with a different static argument results in re-compilation, so the function still works as expected:

```{code-cell} ipython3
:id: sXqczBOrG7-w
:outputId: 5fb7c278-b87e-4a6b-ef50-5e4e9c765b52

f(1, False)
```

+++ {"id": "ZESlrDngGVb1"}

Understanding which values and operations will be static and which will be traced is a key part of using `jax.jit` effectively.
