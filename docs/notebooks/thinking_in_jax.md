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
:nbsphinx: hidden

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

# How to Think in JAX

JAX provides a simple and powerful API for writing accelerated numerical code, but working effectively in JAX sometimes requires extra consideration. This document is meant to help build a ground-up understanding of how JAX operates, so that you can use it more effectively.

## JAX vs. NumPy

**Key Concepts:**

- JAX provides a NumPy-inspired interface for convenience.
- Through duck-typing, JAX arrays can often be used as drop-in replacements of NumPy arrays.
- Unlike NumPy arrays, JAX arrays are always immutable.

NumPy provides a well-known, powerful API for working with numerical data. For convenience, JAX provides `jax.numpy` which closely mirrors the numpy API and provides easy entry into JAX. Almost anything that can be done with `numpy` can be done with `jax.numpy`:

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np

x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
plt.plot(x_np, y_np);
```

```{code-cell}
import jax.numpy as jnp

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp);
```

The code blocks are identical aside from replacing `np` with `jnp`, and the results are the same. As we can see, JAX arrays can often be used directly in place of NumPy arrays for things like plotting.

The arrays themselves are implemented as different Python types:

```{code-cell}
type(x_np)
```

```{code-cell}
type(x_jnp)
```

Python's [duck-typing](https://en.wikipedia.org/wiki/Duck_typing) allows JAX arrays and NumPy arrays to be used interchangeably in many places.

However, there is one important difference between JAX and NumPy arrays: JAX arrays are immutable, meaning that once created their contents cannot be changed.

Here is an example of mutating an array in NumPy:

```{code-cell}
# NumPy: mutable arrays
x = np.arange(10)
x[0] = 10
print(x)
```

The equivalent in JAX results in an error, as JAX arrays are immutable:

```{code-cell}
# JAX: immutable arrays
x = jnp.arange(10)
x[0] = 10
```

For updating individual elements, JAX provides an [indexed update syntax](https://jax.readthedocs.io/en/latest/jax.ops.html#syntactic-sugar-for-indexed-update-operators) that returns an updated copy:

```{code-cell}
y = x.at[0].set(10)
print(x)
print(y)
```

## NumPy, lax & XLA: JAX API layering

**Key Concepts:**

- `jax.numpy` is a high-level wrapper that provides a familiar interface.
- `jax.lax` is a lower-level API that is stricter and often more powerful.
- All JAX operations are implemented in terms of operations in [XLA](https://www.tensorflow.org/xla/) – the Accelerated Linear Algebra compiler.

If you look at the source of `jax.numpy`, you'll see that all the operations are eventually expressed in terms of functions defined in `jax.lax`. You can think of `jax.lax` as a stricter, but often more powerful, API for working with multi-dimensional arrays.

For example, while `jax.numpy` will implicitly promote arguments to allow operations between mixed data types, `jax.lax` will not:

```{code-cell}
import jax.numpy as jnp
jnp.add(1, 1.0)  # jax.numpy API implicitly promotes mixed types.
```

```{code-cell}
from jax import lax
lax.add(1, 1.0)  # jax.lax API requires explicit type promotion.
```

If using `jax.lax` directly, you'll have to do type promotion explicitly in such cases:

```{code-cell}
lax.add(jnp.float32(1), 1.0)
```

Along with this strictness, `jax.lax` also provides efficient APIs for some more general operations than are supported by NumPy.

For example, consider a 1D convolution, which can be expressed in NumPy this way:

```{code-cell}
x = jnp.array([1, 2, 1])
y = jnp.ones(10)
jnp.convolve(x, y)
```

Under the hood, this NumPy operation is translated to a much more general convolution implemented by [`lax.conv_general_dilated`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html):

```{code-cell}
from jax import lax
result = lax.conv_general_dilated(
    x.reshape(1, 1, 3).astype(float),  # note: explicit promotion
    y.reshape(1, 1, 10),
    window_strides=(1,),
    padding=[(len(y) - 1, len(y) - 1)])  # equivalent of padding='full' in NumPy
result[0, 0]
```

This is a batched convolution operation designed to be efficient for the types of convolutions often used in deep neural nets. It requires much more boilerplate, but is far more flexible and scalable than the convolution provided by NumPy (See [JAX Sharp Bits: Convolutions](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Convolutions) for more detail on JAX convolutions).

At their heart, all `jax.lax` operations are Python wrappers for operations in XLA; here, for example, the convolution implementation is provided by [XLA:ConvWithGeneralPadding](https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution).
Every JAX operation is eventually expressed in terms of these fundamental XLA operations, which is what enables just-in-time (JIT) compilation.

## To JIT or not to JIT

**Key Concepts:**

- By default JAX executes operations one at a time, in sequence.
- Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized together and run at once.
- Not all JAX code can be JIT compiled, as it requires array shapes to be static & known at compile time.

The fact that all JAX operations are expressed in terms of XLA allows JAX to use the XLA compiler to execute blocks of code very efficiently.

For example, consider this function that normalizes the rows of a 2D matrix, expressed in terms of `jax.numpy` operations:

```{code-cell}
import jax.numpy as jnp

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)
```

A just-in-time compiled version of the function can be created using the `jax.jit` transform:

```{code-cell}
from jax import jit
norm_compiled = jit(norm)
```

This function returns the same results as the original, up to standard floating-point accuracy:

```{code-cell}
np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
np.allclose(norm(X), norm_compiled(X), atol=1E-6)
```

But due to the compilation (which includes fusing of operations, avoidance of allocating temporary arrays, and a host of other tricks), execution times can be orders of magnitude faster in the JIT-compiled case (note the use of `block_until_ready()` to account for JAX's [asynchronous dispatch](https://jax.readthedocs.io/en/latest/async_dispatch.html)):

```{code-cell}
%timeit norm(X).block_until_ready()
%timeit norm_compiled(X).block_until_ready()
```

That said, `jax.jit` does have limitations: in particular, it requires all arrays to have static shapes. That means that some JAX operations are incompatible with JIT compilation.

For example, this operation can be executed in op-by-op mode:

```{code-cell}
def get_negatives(x):
  return x[x < 0]

x = jnp.array(np.random.randn(10))
get_negatives(x)
```

But it returns an error if you attempt to execute it in jit mode:

```{code-cell}
jit(get_negatives)(x)
```

This is because the function generates an array whose shape is not known at compile time: the size of the output depends on the values of the input array, and so it is not compatible with JIT.

## JIT mechanics: tracing and static variables

**Key Concepts:**

- JIT and other JAX transforms work by *tracing* a function to determine its effect on inputs of a specific shape and type.

- Variables that you don't want to be traced can be marked as *static*

To use `jax.jit` effectively, it is useful to understand how it works. Let's put a few `print()` statements within a JIT-compiled function and then call the function:

```{code-cell}
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

Notice that the print statements execute, but rather than printing the data we passed to the function, though, it prints *tracer* objects that stand-in for them.

These tracer objects are what `jax.jit` uses to extract the sequence of operations specified by the function. Basic tracers are stand-ins that encode the **shape** and **dtype** of the arrays, but are agnostic to the values. This recorded sequence of computations can then be efficiently applied within XLA to new inputs with the same shape and dtype, without having to re-execute the Python code.

When we call the compiled fuction again on matching inputs, no re-compilation is required and nothing is printed because the result is computed in compiled XLA rather than in Python:

```{code-cell}
x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
f(x2, y2)
```

The extracted sequence of operations is encoded in a JAX expression, or *jaxpr* for short. You can view the jaxpr using the `jax.make_jaxpr` transformation:

```{code-cell}
from jax import make_jaxpr

def f(x, y):
  return jnp.dot(x + 1, y + 1)

make_jaxpr(f)(x, y)
```

Note one consequence of this: because JIT compilation is done *without* information on the content of the array, control flow statements in the function cannot depend on traced values. For example, this fails:

```{code-cell}
@jit
def f(x, neg):
  return -x if neg else x

f(1, True)
```

If there are variables that you would not like to be traced, they can be marked as static for the purposes of JIT compilation:

```{code-cell}
from functools import partial

@partial(jit, static_argnums=(1,))
def f(x, neg):
  return -x if neg else x

f(1, True)
```

Note that calling a JIT-compiled function with a different static argument results in re-compilation, so the function still works as expected:

```{code-cell}
f(1, False)
```

Understanding which values and operations will be static and which will be traced is a key part of using `jax.jit` effectively.

## Static vs Traced Operations

**Key Concepts:**

- Just as values can be either static or traced, operations can be static or traced.

- Static operations are evaluated at compile-time in Python; traced operations are compiled & evaluated at run-time in XLA.

- Use `numpy` for operations that you want to be static; use `jax.numpy` for operations that you want to be traced.

This distinction between static and traced values makes it important to think about how to keep a static value static. Consider this function:

```{code-cell}
import jax.numpy as jnp
from jax import jit

@jit
def f(x):
  return x.reshape(jnp.array(x.shape).prod())

x = jnp.ones((2, 3))
f(x)
```

This fails with an error specifying that a tracer was found in `jax.numpy.reshape`. Let's add some print statements to the function to understand why this is happening:

```{code-cell}
@jit
def f(x):
  print(f"x = {x}")
  print(f"x.shape = {x.shape}")
  print(f"jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")
  # comment this out to avoid the error:
  # return x.reshape(jnp.array(x.shape).prot())

f(x)
```

Notice that although `x` is traced, `x.shape` is a static value. However, when we use `jnp.array` and `jnp.prod` on this static value, it becomes a traced value, at which point it cannot be used in a function like `reshape()` that requires a static input (recall: array shapes must be static).

A useful pattern is to use `numpy` for operations that should be static (i.e. done at compile-time), and use `jax.numpy` for operations that should be traced (i.e. compiled and executed at run-time). For this function, it might look like this:

```{code-cell}
from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def f(x):
  return x.reshape((np.prod(x.shape),))

f(x)
```

For this reason, a standard convention in JAX programs is to `import numpy as np` and `import jax.numpy as jnp` so that both interfaces are available for finer control over whether operations are performed in a static matter (with `numpy`, once at compile-time) or a traced manner (with `jax.numpy`, optimized at run-time).
