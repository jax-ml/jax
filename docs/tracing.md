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

(tracing-tutorial)=
# Tracing

`jax.jit` and other JAX transforms work by *tracing* a function to determine its effect on inputs of a specific shape and type. For a window into tracing, let's put a few `print()` statements within a JIT-compiled function and then call the function:

```{code-cell}
from jax import jit
import jax.numpy as jnp
import numpy as np

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

Notice that the print statements execute, but rather than printing the data we
passed to the function, though, it prints *tracer* objects that stand-in for
them (something like `Traced<ShapedArray(float32[])>`).

These tracer objects are what `jax.jit` uses to extract the sequence of
operations specified by the function. Basic tracers are stand-ins that encode
the **shape** and **dtype** of the arrays, but are agnostic to the values. This
recorded sequence of computations can then be efficiently applied within XLA to
new inputs with the same shape and dtype, without having to re-execute the
Python code.

When we call the compiled function again on matching inputs, no re-compilation
is required and nothing is printed because the result is computed in compiled
XLA rather than in Python:

```{code-cell}
x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
f(x2, y2)
```

The extracted sequence of operations is encoded in a JAX expression, or
[*jaxpr*](key-concepts-jaxprs) for short. You can view the jaxpr using the
`jax.make_jaxpr` transformation:

```{code-cell}
from jax import make_jaxpr

def f(x, y):
  return jnp.dot(x + 1, y + 1)

make_jaxpr(f)(x, y)
```

Note one consequence of this: because JIT compilation is done *without*
information on the content of the array, control flow statements in the function
cannot depend on traced values (see {ref}`control-flow`). For example, this fails:

```{code-cell}
:tags: [raises-exception]

@jit
def f(x, neg):
  return -x if neg else x

f(1, True)
```

If there are variables that you would not like to be traced, they can be marked
as *static* for the purposes of JIT compilation:

```{code-cell}
from functools import partial

@partial(jit, static_argnums=(1,))
def f(x, neg):
  return -x if neg else x

f(1, True)
```

Note that calling a JIT-compiled function with a different static argument
results in re-compilation, so the function still works as expected:

```{code-cell}
f(1, False)
```

### Static vs traced operations

Just as values can be either static or traced, operations can be static or
traced. Static operations are evaluated at compile-time in Python; traced
operations are compiled & evaluated at run-time in XLA.

This distinction between static and traced values makes it important to think
about how to keep a static value static. Consider this function:

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

This fails with an error specifying that a tracer was found instead of a 1D
sequence of concrete values of integer type. Let's add some print statements to
the function to understand why this is happening:

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

Notice that although `x` is traced, `x.shape` is a static value. However, when
we use `jnp.array` and `jnp.prod` on this static value, it becomes a traced
value, at which point it cannot be used in a function like `reshape()` that
requires a static input (recall: array shapes must be static).

A useful pattern is to use `numpy` for operations that should be static (i.e.
done at compile-time), and use `jax.numpy` for operations that should be traced
(i.e. compiled and executed at run-time). For this function, it might look like
this:

```{code-cell}
from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def f(x):
  return x.reshape((np.prod(x.shape),))

f(x)
```

For this reason, a standard convention in JAX programs is to
`import numpy as np` and `import jax.numpy as jnp` so that both interfaces are
available for finer control over whether operations are performed in a static
manner (with `numpy`, once at compile-time) or a traced manner (with
`jax.numpy`, optimized at run-time).

Understanding which values and operations will be static and which will be
traced is a key part of using `jax.jit` effectively.

(faq-different-kinds-of-jax-values)=
## Different kinds of JAX values

A tracer value carries an **abstract** value, e.g., `ShapedArray` with
information about the shape and dtype of an array. We will refer here to such
tracers as **abstract tracers**. Some tracers, e.g., those that are introduced
for arguments of autodiff transformations, carry `ConcreteArray` abstract values
that actually include the regular array data, and are used, e.g., for resolving
conditionals. We will refer here to such tracers as **concrete tracers**. Tracer
values computed from these concrete tracers, perhaps in combination with regular
values, result in concrete tracers. A **concrete value** is either a regular
value or a concrete tracer.

Typically, computations that involve at least a tracer value will produce a
tracer value. There are very few exceptions, when a computation can be
entirely done using the abstract value carried by a tracer, in which case the
result can be a **regular** Python value. For example, getting the shape of a
tracer with `ShapedArray` abstract value. Another example is when explicitly
casting a concrete tracer value to a regular type, e.g., `int(x)` or
`x.astype(float)`. Another such situation is for `bool(x)`, which produces a
Python bool when concreteness makes it possible. That case is especially salient
because of how often it arises in control flow.

Here is how the transformations introduce abstract or concrete tracers:

* {func}`jax.jit`: introduces **abstract tracers** for all positional arguments
  except those denoted by `static_argnums`, which remain regular
  values.
* {func}`jax.pmap`: introduces **abstract tracers** for all positional arguments
  except those denoted by `static_broadcasted_argnums`.
* {func}`jax.vmap`, {func}`jax.make_jaxpr`, {func}`xla_computation`:
  introduce **abstract tracers** for all positional arguments.
* {func}`jax.jvp` and {func}`jax.grad` introduce **concrete tracers**
  for all positional arguments. An exception is when these transformations
  are within an outer transformation and the actual arguments are
  themselves abstract tracers; in that case, the tracers introduced
  by the autodiff transformations are also abstract tracers.
* All higher-order control-flow primitives ({func}`lax.cond`,
  {func}`lax.while_loop`, {func}`lax.fori_loop`, {func}`lax.scan`) when they
  process the functionals introduce **abstract tracers**, whether or not there
  is a JAX transformation in progress.

All of this is relevant when you have code that can operate
only on regular Python values, such as code that has conditional
control-flow based on data:

```{code-cell}
def divide(x, y):
  return x / y if y >= 1. else 0.
```

If we want to apply {func}`jax.jit`, we must ensure to specify `static_argnums=1`
to ensure `y` stays a regular value. This is due to the boolean expression
`y >= 1.`, which requires concrete values (regular or tracers). The
same would happen if we write explicitly `bool(y >= 1.)`, or `int(y)`,
or `float(y)`.

Interestingly, `jax.grad(divide)(3., 2.)`, works because {func}`jax.grad`
uses concrete tracers, and resolves the conditional using the concrete
value of `y`.
