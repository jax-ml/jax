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

```{code-cell}
:tags: [remove-cell]

# This ensures that code cell tracebacks appearing below will be concise.
%xmode minimal
```

(thinking-in-jax)=
# How to think in JAX

In this tutorial you will learn about how JAX operates, so that you can use it more effectively. JAX provides a simple and powerful API for writing accelerated numerical code, and working effectively in JAX sometimes requires extra consideration. This document will help you build a ground-up understanding of the JAX API.


## JAX versus NumPy

**Key concepts:**

- JAX provides a NumPy-inspired interface for convenience.
- Through [duck typing](https://en.wikipedia.org/wiki/Duck_typing), JAX arrays (`jax.Array`s) can often be used as drop-in replacements of NumPy arrays ([`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)s).
- Unlike NumPy arrays, JAX arrays are always immutable.

NumPy provides a well-known, powerful API for working with numerical data. For convenience, JAX provides `jax.numpy` which closely mirrors the numpy API and provides easy entry into JAX. Almost anything that can be done with `numpy` can be done with `jax.numpy`:

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np # Ordinary NumPy

x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
plt.plot(x_np, y_np);
```

```{code-cell}
import jax.numpy as jnp # JAX NumPy

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp);
```

The code blocks are identical aside from replacing NumPy (`np`) with JAX NumPy (`jnp`), and the results are the same. JAX arrays can often be used directly in place of NumPy arrays for things like plotting.

The arrays themselves are implemented as different Python types:

```{code-cell}
type(x_np)
```

```{code-cell}
type(x_jnp)
```

Python's [duck typing](https://en.wikipedia.org/wiki/Duck_typing) allows JAX arrays `jax.Array`s and NumPy arrays `numpy.ndarray`s to be used interchangeably in many places.

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
:tags: [raises-exception]

# JAX: immutable arrays
x = jnp.arange(10)
x[0] = 10
```

For updating individual elements, JAX provides an [indexed update syntax](https://jax.readthedocs.io/en/latest/jax.ops.html#indexed-update-operators) that returns an updated copy:

```{code-cell}
y = x.at[0].set(10)
print(x)
print(y)
```

(thinking-in-jax-jax-arrays)=
## JAX arrays (`jax.Array`s)

**Key concepts:**

- `jax.Array` is the default array implementation in JAX.
- JAX arrays may be stored on a single device, or sharded across many devices.

When you create an array in JAX, the type is `jax.Array`:

```{code-cell}
import jax
import jax.numpy as jnp

x = jnp.arange(10)
isinstance(x, jax.Array)
```

`jax.Array` is also the appropriate type annotation for functions with array inputs or outputs:

```{code-cell}
def f(x: jax.Array) -> jax.Array:
  return jnp.sin(x) ** 2 + jnp.cos(x) ** 2
```

JAX Array objects have a `devices` method that lets you inspect where the contents of the array are stored. In the simplest cases, this will be a single CPU device:

```{code-cell}
x.devices()
```

In general, an array may be *sharded* across multiple devices, in a manner that can be inspected via the `sharding` attribute:

```{code-cell}
x.sharding
```

In this case the sharding is on a single device, but in general a JAX array can be
sharded across multiple devices, or even multiple hosts.
To read more about sharded arrays and parallel computation, refer to {ref}`single-host-sharding`

(thinking-in-jax-pytrees)=
## Pytrees

**Key concepts:**

- JAX supports tuples, dicts, lists, and more general containers of arrays through the
  *pytree* abstraction.

Often it is convenient for applications to work with collections of arrays: for example,
a neural network might organize its parameters in a dictionary of arrays with meaningful
keys. Rather than handle such structures on a case-by-case basis, JAX relies on a *pytree*
abstraction to treat such collections in a uniform matter.
In JAX any pytree is safe to pass to transformed functions, which makes them much more flexible
than if they only accepted single arrays as arguments.

Here are some examples of objects that can be treated as pytrees:

```{code-cell}
# (nested) list of parameters
params = [1, 2, (jnp.arange(3), jnp.ones(2))]

print(jax.tree.structure(params))
print(jax.tree.leaves(params))
```

```{code-cell}
# Dictionary of parameters
params = {'n': 5, 'W': jnp.ones((2, 2)), 'b': jnp.zeros(2)}

print(jax.tree.structure(params))
print(jax.tree.leaves(params))
```

```{code-cell}
# Named tuple of parameters
from typing import NamedTuple

class Params(NamedTuple):
  a: int
  b: float

params = Params(1, 5.0)
print(jax.tree.structure(params))
print(jax.tree.leaves(params))
```

JAX has a number of general-purpose utilities for working with PyTrees; for example
the functions {func}`jax.tree.map` can be used to map a function to every leaf in a
tree, and {func}`jax.tree.reduce` can be used to apply a reduction across the leaves
in a tree.

You can learn more in the {ref}`working-with-pytrees` tutorial.

(thinking-in-jax-jax-api-layering)=
## NumPy, `jax.lax` and XLA: JAX API layering

**Key concepts:**

- {mod}`jax.numpy` is a high-level wrapper that provides a familiar interface.
- {mod}`jax.lax` is a lower-level API that is stricter and often more powerful.
- All JAX operations are implemented in terms of operations in [XLA](https://www.tensorflow.org/xla/) — the Accelerated Linear Algebra compiler.

If you look at the source of {mod}`jax.numpy`, you'll see that all the operations are eventually expressed in terms of functions defined in {mod}`jax.lax`. You can think of {mod}`jax.lax` as a stricter, but often more powerful, API for working with multi-dimensional arrays.

For example, while {mod}`jax.numpy` will implicitly promote arguments to allow operations between mixed data types, {mod}`jax.lax` will not:

```{code-cell}
import jax.numpy as jnp
jnp.add(1, 1.0)  # `jax.numpy` API implicitly promotes mixed types.
```

```{code-cell}
:tags: [raises-exception]

from jax import lax
lax.add(1, 1.0)  # `jax.lax` API requires explicit type promotion.
```

If using {mod}`jax.lax` directly, you'll have to do type promotion explicitly in such cases:

```{code-cell}
lax.add(jnp.float32(1), 1.0)
```

Along with this strictness, {mod}`jax.lax` also provides efficient APIs for some more general operations than are supported by NumPy.

For example, consider a 1D convolution, which can be expressed in NumPy this way:

```{code-cell}
x = jnp.array([1, 2, 1])
y = jnp.ones(10)
jnp.convolve(x, y)
```

Under the hood, this NumPy operation is translated to a much more general convolution implemented by {func}`jax.lax.conv_general_dilated`:

```{code-cell}
from jax import lax
result = lax.conv_general_dilated(
    x.reshape(1, 1, 3).astype(float),  # note: explicit promotion
    y.reshape(1, 1, 10),
    window_strides=(1,),
    padding=[(len(y) - 1, len(y) - 1)])  # equivalent of `padding='full'`` in NumPy
result[0, 0]
```

This is a batched convolution operation designed to be efficient for the types of convolutions often used in deep neural nets. It requires much more boilerplate, but is far more flexible and scalable than the convolution provided by NumPy (Refer to [Convolutions in JAX](https://jax.readthedocs.io/en/latest/notebooks/convolutions.html) for more details on JAX convolutions).

At their heart, all {mod}`jax.lax` operations are Python wrappers for operations in XLA. Here, for example, the convolution implementation is provided by [XLA:ConvWithGeneralPadding](https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution).
Every JAX operation is eventually expressed in terms of these fundamental XLA operations, which is what enables just-in-time (JIT) compilation.


(thinking-in-jax-to-jit-or-not-to-jit)=
## To JIT or not to JIT (`jax.jit`)

**Key concepts:**

- By default JAX executes operations one at a time, in sequence.
- Using a just-in-time (JIT) compilation decorator — {func}`jax.jit` — sequences of operations can be optimized together and run at once.
- Not all JAX code can be JIT compiled, as it requires array shapes to be static & known at compile time.

The fact that all JAX operations are expressed in terms of XLA allows JAX to use the XLA compiler to execute blocks of code very efficiently.

For example, consider this function that normalizes the rows of a 2D matrix, expressed in terms of {mod}`jax.numpy` operations:

```{code-cell}
import jax.numpy as jnp

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)
```

A JIT-compiled version of the function can be created using the {func}`jax.jit` transform:

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

But due to the compilation (which includes fusing of operations, avoidance of allocating temporary arrays, and a host of other tricks), execution times can be orders of magnitude faster in the JIT-compiled case (note the use of {func}`jax.block_until_ready` to account for JAX's [asynchronous dispatch](https://jax.readthedocs.io/en/latest/async_dispatch.html)):

```{code-cell}
%timeit norm(X).block_until_ready()
%timeit norm_compiled(X).block_until_ready()
```

That said, {func}`jax.jit` does have limitations: in particular, it requires all arrays to have static shapes. This means some JAX operations are incompatible with JIT compilation.

For example, this operation can be executed in op-by-op mode:

```{code-cell}
def get_negatives(x):
  return x[x < 0]

x = jnp.array(np.random.randn(10))
get_negatives(x)
```

But it returns an error if you attempt to execute it in {func}`jax.jit` mode:

```{code-cell}
:tags: [raises-exception]

jit(get_negatives)(x)
```

This is because the function generates an array whose shape is not known at compile time: the size of the output depends on the values of the input array, and so it is not compatible with JIT.

(thinking-in-jax-jit-mechanics)=
## JIT mechanics: tracing and static variables

**Key concepts:**

- JIT and other JAX transforms work by *tracing* a function to determine its effect on inputs of a specific shape and type.

- Variables that you don't want to be traced can be marked as *static*

To use {func}`jax.jit` effectively, it is useful to understand how it works. Let's put a few `print()` statements within a JIT-compiled function and then call the function:

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

Notice that the print statements execute, but rather than printing the data you passed to the function, though, it prints *tracer* objects that stand-in for them.

These tracer objects are what {func}`jax.jit` uses to extract the sequence of operations specified by the function. Basic tracers are stand-ins that encode the **shape** and **dtype** of the arrays, but are agnostic to the values. This recorded sequence of computations can then be efficiently applied within XLA to new inputs with the same shape and dtype, without having to re-execute the Python code.

When you call the compiled function again on matching inputs, no recompilation is required and nothing is printed because the result is computed in compiled XLA rather than in Python:

```{code-cell}
x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
f(x2, y2)
```

The extracted sequence of operations is encoded in a JAX expression, or *jaxpr* for short. You can view the jaxpr using the {func}`jax.make_jaxpr` transformation:

```{code-cell}
from jax import make_jaxpr

def f(x, y):
  return jnp.dot(x + 1, y + 1)

make_jaxpr(f)(x, y)
```

Note one consequence of this: because JIT compilation is done *without* information on the content of the array, control flow statements in the function cannot depend on traced values. For example, this fails:

```{code-cell}
:tags: [raises-exception]

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

Note that calling a JIT-compiled function with a different static argument results in recompilation, so the function still works as expected:

```{code-cell}
f(1, False)
```

Understanding which values and operations will be static and which will be traced is a key part of using {func}`jax.jit` effectively.


(thinking-in-jax-static-versus-traced-operations)=
## Static versus traced operations

**Key concepts:**

- Just as values can be either static or traced, operations can be static or traced.
- Static operations are evaluated at compile-time in Python. Traced operations are compiled & evaluated at run-time in XLA.
- Use NumPy (`numpy`) for operations that you want to be static. Use JAX NumPy {mod}`jax.numpy` for operations that you want to be traced.

This distinction between static and traced values makes it important to think about how to keep a static value static. Consider this function:

```{code-cell}
:tags: [raises-exception]

import jax.numpy as jnp
from jax import jit

@jit
def f(x):
  return x.reshape(jnp.array(x.shape).prod())

x = jnp.ones((2, 3))
f(x)
```

This fails with an error specifying that a tracer was found instead of a 1D sequence of concrete values of integer type. Let's add some print statements to the function to understand why this is happening:

```{code-cell}
@jit
def f(x):
  print(f"x = {x}")
  print(f"x.shape = {x.shape}")
  print(f"jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")
  # comment this out to avoid the error:
  # return x.reshape(jnp.array(x.shape).prod())

f(x)
```

Notice that although `x` is traced, `x.shape` is a static value. However, when you use {func}`jnp.array` and {func}`jnp.prod` on this static value, it becomes a traced value, at which point it cannot be used in a function like `reshape()` that requires a static input (recall: array shapes must be static).

A useful pattern is to:

- Use NumPy (`numpy`) for operations that should be static (i.e., done at compile-time); and
- Use JAX NumPy (`jax.numpy`) for operations that should be traced (i.e. compiled and executed at run-time).

For this function, it might look like this:

```{code-cell}
from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def f(x):
  return x.reshape((np.prod(x.shape),))

f(x)
```

For this reason, a standard convention in JAX programs is to `import numpy as np` and `import jax.numpy as jnp` so that both interfaces are available for finer control over whether operations are performed in a static matter (with `numpy`, once at compile-time) or a traced manner (with {mod}`jax.numpy`, optimized at run-time).
