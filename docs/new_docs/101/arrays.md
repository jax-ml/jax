---
jupytext:
  formats: md:myst
  notebook_metadata_filter: nosearch
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
nosearch: true
---

```{code-cell}
:tags: [remove-cell]

# This ensures that code cell tracebacks appearing below will be concise.
%xmode minimal
```

(jax-101-arrays)=
# Arrays and `jax.numpy`

<!--* freshness: { reviewed: '2026-07-09' } *-->

JAX's basic data type is the array, {class}`jax.Array`, and its basic API for
working with arrays is {mod}`jax.numpy`, which closely mirrors NumPy. Almost
anything that can be done with `numpy` can be done with `jax.numpy`. This page
covers what carries over from NumPy unchanged, and the handful of differences
that matter from day one: immutability, default dtypes, and a few indexing
behaviors.

## From NumPy to JAX

The conventional way to import `jax.numpy` is under the alias `jnp`, alongside
regular NumPy as `np`:

```{code-cell}
import jax
import jax.numpy as jnp
import numpy as np
```

With this import, array creation, arithmetic, indexing, reductions, and so on
all look just like NumPy:

```{code-cell}
x = jnp.arange(9.0).reshape(3, 3)
y = jnp.ones(3)

print(x @ y)
print(x.sum(axis=0))
print(x[1, :2])
```

The arrays themselves are instances of {class}`jax.Array`:

```{code-cell}
isinstance(x, jax.Array)
```

If you use type annotations in your code, `jax.Array` is the appropriate
annotation for JAX array values (see {mod}`jax.typing` for more discussion).

Every JAX value also has a *JAX type*, which you can inspect with
{func}`jax.typeof`. For an array, the JAX type roughly means its shape and
dtype:

```{code-cell}
print(jax.typeof(x))
```

This compact notation appears all over JAX, and we'll meet it again when we
look at how JAX records whole programs. JAX types can also carry more than
shape and dtype — notably *sharding*, describing how an array is laid out
across devices, covered in the performance and scaling docs — but shape and
dtype are the heart of it.

JAX arrays and NumPy arrays can often be used interchangeably: `jax.numpy`
functions accept NumPy arrays, and most NumPy functions accept JAX arrays
(thanks to Python's duck typing). It's common to use `np` for data loading and
lightweight host-side manipulation and `jnp` for the computation you care
about. As we'll see in {ref}`jax-101-transformations`, keeping both imports
around is also a useful convention when working with JAX's transformations.

## JAX arrays are immutable

The most important difference from NumPy: once created, a JAX array's contents
cannot be changed. NumPy allows in-place mutation:

```{code-cell}
x_np = np.arange(10)
x_np[0] = 10
print(x_np)
```

The equivalent raises an error in JAX:

```{code-cell}
:tags: [raises-exception]

x = jnp.arange(10)
x[0] = 10
```

Instead, JAX provides *functional updates* via the {attr}`~jax.Array.at`
property, which return a new array with the update applied:

```{code-cell}
y = x.at[0].set(10)
print(x)  # unchanged
print(y)
```

Alongside `set`, there's a family of update operations, including `add`,
`multiply`, `min`, and `max`:

```{code-cell}
print(x.at[3].add(100))
print(x.at[:3].max(5))
```

Writing updates functionally may feel wasteful — it looks like every update
copies the whole array. Outside of compiled code that's accurate, but inside
{func}`jax.jit`-compiled functions the compiler can usually perform these
updates in place. And when you genuinely want mutable arrays — for data
plumbing, or for explicit control over memory — JAX has a first-class answer
in refs, covered in {ref}`jax-101-refs`.

## Default dtypes and precision

NumPy defaults to 64-bit floating point. JAX defaults to 32-bit, which is
usually what you want on accelerators like GPUs and TPUs:

```{code-cell}
print(np.array([1.0, 2.0]).dtype)
print(jnp.array([1.0, 2.0]).dtype)
```

In fact, by default JAX disables 64-bit dtypes altogether: requesting
`float64` produces a `float32` array (with a warning). If you need 64-bit
precision — say, for numerical work where the extra bits matter — enable it at
startup:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

For the full story, including how to control default dtypes more finely, see
{doc}`/default_dtypes`.

When operations mix dtypes, JAX applies type promotion rules that are similar
to NumPy's but not identical — in particular they're designed to avoid
accidentally promoting everything to 64-bit:

```{code-cell}
(jnp.arange(3) + 1.5).dtype
```

See {doc}`/type_promotion` for the precise promotion semantics.

## Indexing differences

JAX supports NumPy-style indexing, including slices and advanced integer
array indexing. But some edge-case behaviors differ, because JAX is designed
so that every operation can be compiled for accelerators, where raising an
exception from inside a computation isn't an option.

Most notably, out-of-bounds indexing doesn't raise an error. When *reading*,
the index is clamped to the bounds of the array:

```{code-cell}
x = jnp.arange(10)
x[11]  # clamped to x[9]
```

When *writing* with `.at`, out-of-bounds updates are dropped:

```{code-cell}
x.at[11].set(99)  # update is dropped
```

Both behaviors can be adjusted with the `mode` argument to indexing
operations; see {attr}`jax.numpy.ndarray.at` for the options.

Another small difference: `jax.numpy` functions require arrays (or values of
Python's built-in numeric types) as inputs, rather than silently converting
Python lists:

```{code-cell}
:tags: [raises-exception]

jnp.sum([1, 2, 3])
```

This is deliberate. Silently converting lists to arrays is a common source of
hidden performance problems, so JAX asks you to do it explicitly:

```{code-cell}
jnp.sum(jnp.array([1, 2, 3]))
```

## Arrays live on devices

A JAX array's data lives on one or more *devices* — CPU, GPU, or TPU. The same
JAX code runs on all of them; JAX places arrays on the default device
(typically your accelerator, if you have one) and computations follow their
data. An array can even be *sharded* across many devices, so that JAX programs
can be written once and run on one chip or thousands. Every array carries a
`sharding` attribute describing exactly how its data is placed:

```{code-cell}
x.sharding
```

On a single-device machine this isn't very interesting, but it's the
foundation of JAX's approach to parallelism and scaling. That story is told in
the performance and scaling docs; see {ref}`jax-201-sharding`.

## Under the hood: `jax.numpy`, `jax.lax`, and XLA

`jax.numpy` is a convenience layer. Its functions are implemented in terms of
{mod}`jax.lax`, a stricter, lower-level API. Most `jax.numpy` functions are
thin wrappers that handle NumPy-style conveniences and then defer to their
`jax.lax` counterpart. Here's `jnp.sin`, simplified from the real
implementation:

```python
def sin(x):
  x = jnp.asarray(x)                            # accept built-in Python numbers
  if not jnp.issubdtype(x.dtype, jnp.inexact):
    x = x.astype(float)                         # promote integers to floating point
  return lax.sin(x)
```

The `jax.numpy` layer's job is NumPy-style argument handling — accepting
built-in Python numbers, promoting dtypes — while the computation itself
belongs to `jax.lax`. The `jax.lax` operations in turn
correspond closely to [XLA HLO operations](https://openxla.org/xla/operation_semantics),
the vocabulary of [XLA](https://www.openxla.org/xla/), the compiler that
ultimately runs JAX computations: `lax.sin` maps essentially one-to-one onto
XLA's `Sin`.

Being closer to the compiler, `jax.lax` skips the conveniences. For example,
where `jax.numpy` implicitly promotes mixed types:

```{code-cell}
jnp.add(1, 1.0)
```

`jax.lax` requires explicit promotion:

```{code-cell}
:tags: [raises-exception]

from jax import lax
lax.add(1, 1.0)
```

In exchange for the strictness, `jax.lax` exposes operations that are more
general than NumPy's, like arbitrarily strided and dilated convolutions, and
structured control flow. You can write a lot of JAX without touching
`jax.lax`, but it's useful to know it's there: when you can't find a
`jax.numpy` function for something, check {mod}`jax.lax`.

## Next steps

Arrays and `jax.numpy` are the vocabulary of JAX programs. The verbs — the
things JAX can *do* with functions on arrays that NumPy can't — are the
subject of the next page, {ref}`jax-101-transformations`.
