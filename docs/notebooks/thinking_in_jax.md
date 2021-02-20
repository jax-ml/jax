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

```{code-cell}
:id: aPUwOm-eCSFD
:tags: [remove-cell]

# Configure ipython to hide long tracebacks.
import sys
ipython = get_ipython()

def minimal_traceback(*args, **kwargs):
  etype, value, tb = sys.exc_info()
  value.__cause__ = None  # suppress chained exceptions
  stb = ipython.InteractiveTB.structured_traceback(etype, value, tb)
  del stb[3:-1]
  return ipython._showtraceback(etype, value, stb)

ipython.showtraceback = minimal_traceback
```

+++ {"id": "LQHmwePqryRU"}

# How to Think in JAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google/jax/blob/master/docs/notebooks/thinking_in_jax.ipynb)

JAX provides a simple and powerful API for writing accelerated numerical code, but working effectively in JAX sometimes requires extra consideration. This document is meant to help build a ground-up understanding of how JAX operates, so that you can use it more effectively.

+++ {"id": "nayIExVUtsVD"}

## JAX vs. NumPy

**Key Concepts:**

- JAX provides a NumPy-inspired interface for convenience.
- Through duck-typing, JAX arrays can often be used as drop-in replacements of NumPy arrays.
- Unlike NumPy arrays, JAX arrays are always immutable.

NumPy provides a well-known, powerful API for working with numerical data. For convenience, JAX provides `jax.numpy` which closely mirrors the numpy API and provides easy entry into JAX. Almost anything that can be done with `numpy` can be done with `jax.numpy`:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 265
id: kZaOXL7-uvUP
outputId: 17a9ee0a-8719-44bb-a9fe-4c9f24649fef
---
import matplotlib.pyplot as plt
import numpy as np

x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
plt.plot(x_np, y_np);
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 282
id: 18XbGpRLuZlr
outputId: 9e98d928-1925-45b1-d886-37956ca95e7c
---
import jax.numpy as jnp

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp);
```

+++ {"id": "kTZcsCJiuPG8"}

The code blocks are identical aside from replacing `np` with `jnp`, and the results are the same. As we can see, JAX arrays can often be used directly in place of NumPy arrays for things like plotting.

The arrays themselves are implemented as different Python types:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: PjFFunI7xNe8
outputId: e1706c61-2821-437a-efcd-d8082f913c1f
---
type(x_np)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: kpv5K7QYxQnX
outputId: 8a3f1cb6-c6d6-494c-8efe-24a8217a9d55
---
type(x_jnp)
```

+++ {"id": "Mx94Ri7euEZm"}

Python's [duck-typing](https://en.wikipedia.org/wiki/Duck_typing) allows JAX arrays and NumPy arrays to be used interchangeably in many places.

However, there is one important difference between JAX and NumPy arrays: JAX arrays are immutable, meaning that once created their contents cannot be changed.

Here is an example of mutating an array in NumPy:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: fzp-y1ZVyGD4
outputId: 300a44cc-1ccd-4fb2-f0ee-2179763f7690
---
# NumPy: mutable arrays
x = np.arange(10)
x[0] = 10
print(x)
```

+++ {"id": "nQ-De0xcJ1lT"}

The equivalent in JAX results in an error, as JAX arrays are immutable:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 215
id: pCPX0JR-yM4i
outputId: 02a442bc-8f23-4dce-9500-81cd28c0b21f
tags: [raises-exception]
---
# JAX: immutable arrays
x = jnp.arange(10)
x[0] = 10
```

+++ {"id": "yRYF0YgO3F4H"}

For updating individual elements, JAX provides an [indexed update syntax](https://jax.readthedocs.io/en/latest/jax.ops.html#syntactic-sugar-for-indexed-update-operators) that returns an updated copy:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 8zqPEAeP3UK5
outputId: 7e6c996d-d0b0-4d52-e722-410ba78eb3b1
---
y = x.at[0].set(10)
print(x)
print(y)
```

+++ {"id": "886BGDPeyXCu"}

## NumPy, lax & XLA: JAX API layering

**Key Concepts:**

- `jax.numpy` is a high-level wrapper that provides a familiar interface.
- `jax.lax` is a lower-level API that is stricter and often more powerful.
- All JAX operations are implemented in terms of operations in [XLA](https://www.tensorflow.org/xla/) – the Accelerated Linear Algebra compiler.

+++ {"id": "BjE4m2sZy4hh"}

If you look at the source of `jax.numpy`, you'll see that all the operations are eventually expressed in terms of functions defined in `jax.lax`. You can think of `jax.lax` as a stricter, but often more powerful, API for working with multi-dimensional arrays.

For example, while `jax.numpy` will implicitly promote arguments to allow operations between mixed data types, `jax.lax` will not:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: c6EFPcj12mw0
outputId: 730e2ca4-30a5-45bc-923c-c3a5143496e2
---
import jax.numpy as jnp
jnp.add(1, 1.0)  # jax.numpy API implicitly promotes mixed types.
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 181
id: 0VkqlcXL2qSp
outputId: 601b0562-3e6a-402d-f83b-3afdd1e7e7c4
tags: [raises-exception]
---
from jax import lax
lax.add(1, 1.0)  # jax.lax API requires explicit type promotion.
```

+++ {"id": "aC9TkXaTEu7A"}

If using `jax.lax` directly, you'll have to do type promotion explicitly in such cases:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 3PNQlieT81mi
outputId: cb3ed074-f410-456f-c086-23107eae2634
---
lax.add(jnp.float32(1), 1.0)
```

+++ {"id": "M3HDuM4x2eTL"}

Along with this strictness, `jax.lax` also provides efficient APIs for some more general operations than are supported by NumPy.

For example, consider a 1D convolution, which can be expressed in NumPy this way:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Bv-7XexyzVCN
outputId: f5d38cd8-e7fc-49e2-bff3-a0eee306cb54
---
x = jnp.array([1, 2, 1])
y = jnp.ones(10)
jnp.convolve(x, y)
```

+++ {"id": "0GPqgT7S0q8r"}

Under the hood, this NumPy operation is translated to a much more general convolution implemented by [`lax.conv_general_dilated`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html):

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: pi4f6ikjzc3l
outputId: b9b37edc-b911-4010-aaf8-ee8f500111d7
---
from jax import lax
result = lax.conv_general_dilated(
    x.reshape(1, 1, 3).astype(float),  # note: explicit promotion
    y.reshape(1, 1, 10),
    window_strides=(1,),
    padding=[(len(y) - 1, len(y) - 1)])  # equivalent of padding='full' in NumPy
result[0, 0]
```

+++ {"id": "7mdo6ycczlbd"}

This is a batched convolution operation designed to be efficient for the types of convolutions often used in deep neural nets. It requires much more boilerplate, but is far more flexible and scalable than the convolution provided by NumPy (See [JAX Sharp Bits: Convolutions](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Convolutions) for more detail on JAX convolutions).

At their heart, all `jax.lax` operations are Python wrappers for operations in XLA; here, for example, the convolution implementation is provided by [XLA:ConvWithGeneralPadding](https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution).
Every JAX operation is eventually expressed in terms of these fundamental XLA operations, which is what enables just-in-time (JIT) compilation.

+++ {"id": "NJfWa2PktD5_"}

## To JIT or not to JIT

**Key Concepts:**

- By default JAX executes operations one at a time, in sequence.
- Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized together and run at once.
- Not all JAX code can be JIT compiled, as it requires array shapes to be static & known at compile time.

The fact that all JAX operations are expressed in terms of XLA allows JAX to use the XLA compiler to execute blocks of code very efficiently.

For example, consider this function that normalizes the rows of a 2D matrix, expressed in terms of `jax.numpy` operations:

```{code-cell}
:id: SQj_UKGc-7kQ

import jax.numpy as jnp

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)
```

+++ {"id": "0yVo_OKSAolW"}

A just-in-time compiled version of the function can be created using the `jax.jit` transform:

```{code-cell}
:id: oHLwGmhZAnCY

from jax import jit
norm_compiled = jit(norm)
```

+++ {"id": "Q3H9ig5GA2Ms"}

This function returns the same results as the original, up to standard floating-point accuracy:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: oz7zzyS3AwMc
outputId: 914f9242-82c4-4365-abb2-77843a704e03
---
np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
np.allclose(norm(X), norm_compiled(X), atol=1E-6)
```

+++ {"id": "3GvisB-CA9M8"}

But due to the compilation (which includes fusing of operations, avoidance of allocating temporary arrays, and a host of other tricks), execution times can be orders of magnitude faster in the JIT-compiled case (note the use of `block_until_ready()` to account for JAX's [asynchronous dispatch](https://jax.readthedocs.io/en/latest/async_dispatch.html)):

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 6mUB6VdDAEIY
outputId: 5d7e1bbd-4064-4fe3-f3d9-5435b5283199
---
%timeit norm(X).block_until_ready()
%timeit norm_compiled(X).block_until_ready()
```

+++ {"id": "B1eGBGn0tMba"}

That said, `jax.jit` does have limitations: in particular, it requires all arrays to have static shapes. That means that some JAX operations are incompatible with JIT compilation.

For example, this operation can be executed in op-by-op mode:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: YfZd9mW7CSKM
outputId: 899fedcc-0857-4381-8f57-bb653e0aa2f1
---
def get_negatives(x):
  return x[x < 0]

x = jnp.array(np.random.randn(10))
get_negatives(x)
```

+++ {"id": "g6niKxoQC2mZ"}

But it returns an error if you attempt to execute it in jit mode:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 164
id: yYWvE4rxCjPK
outputId: 765b46d3-49cd-41b7-9815-e8bb7cd80175
tags: [raises-exception]
---
jit(get_negatives)(x)
```

+++ {"id": "vFL6DNpECfVz"}

This is because the function generates an array whose shape is not known at compile time: the size of the output depends on the values of the input array, and so it is not compatible with JIT.

+++ {"id": "BzBnKbXwXjLV"}

## JIT mechanics: tracing and static variables

**Key Concepts:**

- JIT and other JAX transforms work by *tracing* a function to determine its effect on inputs of a specific shape and type.

- Variables that you don't want to be traced can be marked as *static*

To use `jax.jit` effectively, it is useful to understand how it works. Let's put a few `print()` statements within a JIT-compiled function and then call the function:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: TfjVIVuD4gnc
outputId: df6ad898-b047-4ad1-eb18-2fbcb3fd2ab3
---
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

When we call the compiled fuction again on matching inputs, no re-compilation is required and nothing is printed because the result is computed in compiled XLA rather than in Python:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: xGntvzNH7skE
outputId: 66694b8b-181f-4635-a8e2-1fc7f244d94b
---
x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
f(x2, y2)
```

+++ {"id": "9EB9WkRX7fm0"}

The extracted sequence of operations is encoded in a JAX expression, or *jaxpr* for short. You can view the jaxpr using the `jax.make_jaxpr` transformation:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 89TMp_Op5-JZ
outputId: 151210e2-af6f-4950-ac1e-9fdb81d4aae1
---
from jax import make_jaxpr

def f(x, y):
  return jnp.dot(x + 1, y + 1)

make_jaxpr(f)(x, y)
```

+++ {"id": "0Oq9S4MZ90TL"}

Note one consequence of this: because JIT compilation is done *without* information on the content of the array, control flow statements in the function cannot depend on traced values. For example, this fails:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 419
id: A0rFdM95-Ix_
outputId: d7ffa367-b241-488e-df96-ad0576536605
tags: [raises-exception]
---
@jit
def f(x, neg):
  return -x if neg else x

f(1, True)
```

+++ {"id": "DkTO9m8j-TYI"}

If there are variables that you would not like to be traced, they can be marked as static for the purposes of JIT compilation:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: K1C7ZnVv-lbv
outputId: cdbdf152-30fd-4ecb-c9ec-1d1124f337f7
---
from functools import partial

@partial(jit, static_argnums=(1,))
def f(x, neg):
  return -x if neg else x

f(1, True)
```

+++ {"id": "dD7p4LRsGzhx"}

Note that calling a JIT-compiled function with a different static argument results in re-compilation, so the function still works as expected:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: sXqczBOrG7-w
outputId: 3a3f50e6-d1fc-42bb-d6df-eb3d206e4b67
---
f(1, False)
```

+++ {"id": "ZESlrDngGVb1"}

Understanding which values and operations will be static and which will be traced is a key part of using `jax.jit` effectively.

+++ {"id": "r-RCl_wD5lI7"}

## Static vs Traced Operations

**Key Concepts:**

- Just as values can be either static or traced, operations can be static or traced.

- Static operations are evaluated at compile-time in Python; traced operations are compiled & evaluated at run-time in XLA.

- Use `numpy` for operations that you want to be static; use `jax.numpy` for operations that you want to be traced.

This distinction between static and traced values makes it important to think about how to keep a static value static. Consider this function:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 385
id: XJCQ7slcD4iU
outputId: a89a5614-7359-4dc7-c165-03e7d0fc6610
tags: [raises-exception]
---
import jax.numpy as jnp
from jax import jit

@jit
def f(x):
  return x.reshape(jnp.array(x.shape).prod())

x = jnp.ones((2, 3))
f(x)
```

+++ {"id": "ZO3GMGrHBZDS"}

This fails with an error specifying that a tracer was found in `jax.numpy.reshape`. Let's add some print statements to the function to understand why this is happening:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Cb4mbeVZEi_q
outputId: f72c1ce3-950c-400f-bfea-10c0d0118911
---
@jit
def f(x):
  print(f"x = {x}")
  print(f"x.shape = {x.shape}")
  print(f"jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")
  # comment this out to avoid the error:
  # return x.reshape(jnp.array(x.shape).prod())

f(x)
```

+++ {"id": "viSQPc3jEwJr"}

Notice that although `x` is traced, `x.shape` is a static value. However, when we use `jnp.array` and `jnp.prod` on this static value, it becomes a traced value, at which point it cannot be used in a function like `reshape()` that requires a static input (recall: array shapes must be static).

A useful pattern is to use `numpy` for operations that should be static (i.e. done at compile-time), and use `jax.numpy` for operations that should be traced (i.e. compiled and executed at run-time). For this function, it might look like this:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: GiovOOPcGJhg
outputId: 399ee059-1807-4866-9beb-1c5131e38e15
---
from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def f(x):
  return x.reshape((np.prod(x.shape),))

f(x)
```

+++ {"id": "C-QZ5d1DG-dv"}

For this reason, a standard convention in JAX programs is to `import numpy as np` and `import jax.numpy as jnp` so that both interfaces are available for finer control over whether operations are performed in a static matter (with `numpy`, once at compile-time) or a traced manner (with `jax.numpy`, optimized at run-time).
