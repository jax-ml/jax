---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(key-concepts)=
# Key concepts

<!--* freshness: { reviewed: '2024-05-03' } *-->

This section briefly introduces some key concepts of the JAX package.

(key-concepts-transformations)=
## Transformations
Along with functions to operate on arrays, JAX includes a number of
{term}`transformations <transformation>` which operate on JAX functions. These include

- {func}`jax.jit`: Just-in-time (JIT) compilation; see {ref}`jit-compilation`
- {func}`jax.vmap`: Vectorizing transform; see {ref}`automatic-vectorization`
- {func}`jax.grad`: Gradient transform; see {ref}`automatic-differentiation`

as well as several others. Transformations accept a function as an argument, and return a
new transformed function. For example, here's how you might JIT-compile a simple SELU function:

```{code-cell}
import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

selu_jit = jax.jit(selu)
print(selu_jit(1.0))
```

Often you'll see transformations applied using Python's decorator syntax for convenience:

```{code-cell}
@jax.jit
def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
```

(key-concepts-tracing)=
## Tracing

The magic behind transformations is the notion of a {term}`Tracer`.
Tracers are abstract stand-ins for array objects, and are passed to JAX functions in order
to extract the sequence of operations that the function encodes.

You can see this by printing any array value within transformed JAX code; for example:

```{code-cell}
@jax.jit
def f(x):
  print(x)
  return x + 1

x = jnp.arange(5)
result = f(x)
```

The value printed is not the array `x`, but a {class}`~jax.core.Tracer` instance that
represents essential attributes of `x`, such as its `shape` and `dtype`. By executing
the function with traced values, JAX can determine the sequence of operations encoded
by the function before those operations are actually executed: transformations like
{func}`~jax.jit`, {func}`~jax.vmap`, and {func}`~jax.grad` can then map this sequence
of input operations to a transformed sequence of operations.

### Static vs traced operations

Just as values can be either static or traced, operations can be static or traced.
Static operations are evaluated at compile-time in Python; traced operations are
compiled & evaluated at run-time in XLA.

This distinction between static and traced values makes it important to think about
how to keep a static value static. Consider this function:

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

This fails with an error specifying that a tracer was found instead of a 1D sequence
of concrete values of integer type. Let's add some print statements to the function
to understand why this is happening:

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

Notice that although `x` is traced, `x.shape` is a static value. However, when we
use `jnp.array` and `jnp.prod` on this static value, it becomes a traced value, at
which point it cannot be used in a function like `reshape()` that requires a static
input (recall: array shapes must be static).

A useful pattern is to use `numpy` for operations that should be static (i.e. done
at compile-time), and use `jax.numpy` for operations that should be traced (i.e.
compiled and executed at run-time). For this function, it might look like this:

```{code-cell}
from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def f(x):
  return x.reshape((np.prod(x.shape),))

f(x)
```

For this reason, a standard convention in JAX programs is to `import numpy as np`
and `import jax.numpy as jnp` so that both interfaces are available for finer
control over whether operations are performed in a static manner (with `numpy`,
once at compile-time) or a traced manner (with `jax.numpy`, optimized at run-time).

(key-concepts-jaxprs)=
## Jaxprs

JAX has its own intermediate representation for sequences of operations, known as a {term}`jaxpr`.
A jaxpr (short for *JAX exPRession*) is a simple representation of a functional program, comprising a sequence of {term}`primitive` operations.

For example, consider the `selu` function we defined above:

```{code-cell}
def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
```

We can use the {func}`jax.make_jaxpr` utility to convert this function into a jaxpr
given a particular input:

```{code-cell}
x = jnp.arange(5.0)
jax.make_jaxpr(selu)(x)
```

Comparing this to the Python function definition, we see that it encodes the precise
sequence of operations that the function represents. We'll go into more depth about
jaxprs later in {ref}`jax-internals-jaxpr`.

(key-concepts-pytrees)=
## Pytrees

JAX functions and transformations fundamentally operate on arrays, but in practice it is
convenient to write code that works with collection of arrays: for example, a neural
network might organize its parameters in a dictionary of arrays with meaningful keys.
Rather than handle such structures on a case-by-case basis, JAX relies on the {term}`pytree`
abstraction to treat such collections in a uniform manner.

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

## JAX API layering: NumPy, lax & XLA

All JAX operations are implemented in terms of operations in [XLA](https://www.openxla.org/xla/) – the Accelerated Linear Algebra compiler. If you look at the source of `jax.numpy`, you'll see that all the operations are eventually expressed in terms of functions defined in {mod}`jax.lax`. While `jax.numpy` is a high-level wrapper that provides a familiar interface, you can think of `jax.lax` as a stricter, but often more powerful, lower-level API for working with multi-dimensional arrays.

For example, while `jax.numpy` will implicitly promote arguments to allow operations between mixed data types, `jax.lax` will not:

```{code-cell}
import jax.numpy as jnp
jnp.add(1, 1.0)  # jax.numpy API implicitly promotes mixed types.
```

```{code-cell}
:tags: [raises-exception]

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

Under the hood, this NumPy operation is translated to a much more general convolution implemented by [`lax.conv_general_dilated`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv_general_dilated.html):

```{code-cell}
from jax import lax
result = lax.conv_general_dilated(
    x.reshape(1, 1, 3).astype(float),  # note: explicit promotion
    y.reshape(1, 1, 10),
    window_strides=(1,),
    padding=[(len(y) - 1, len(y) - 1)])  # equivalent of padding='full' in NumPy
result[0, 0]
```

This is a batched convolution operation designed to be efficient for the types of convolutions often used in deep neural nets. It requires much more boilerplate, but is far more flexible and scalable than the convolution provided by NumPy (See [Convolutions in JAX](https://docs.jax.dev/en/latest/notebooks/convolutions.html) for more detail on JAX convolutions).

At their heart, all `jax.lax` operations are Python wrappers for operations in XLA; here, for example, the convolution implementation is provided by [XLA:ConvWithGeneralPadding](https://www.openxla.org/xla/operation_semantics#convwithgeneralpadding_convolution).
Every JAX operation is eventually expressed in terms of these fundamental XLA operations, which is what enables just-in-time (JIT) compilation.
