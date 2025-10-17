---
jupytext:
  formats: ipynb,md:myst
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

# Device-local array layout control

The `jax.experimental.layout` package provides ways to control
how JAX arrays are laid out in device-local memory.

## Terminology

Array layout is tightly coupled with array
[sharding](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>).
Together, a layout and a sharding fully describes how an array's
values are laid out across (distributed) memories. Along these lines,
we use the following terminology:

* **Layout**: how an array's values are laid out within each memory in
    which they reside (e.g., in the memory of a single device
    memory). A typical layout specification is a minor-to-major order
    listing of array dimensions.
* **Sharding**: how an array's values are distributed *across*
    different memory spaces, such as multiple device memories
    (e.g. described by sharding some dimensions and replicating
    others).
* **Format**: the pairing of **layout** and **sharding**,
    providing a complete picture of an array's memory placement.

## Types

There are two Python types that come up when controlling array
layouts: `Layout` and `Format`.

* The `Layout` class is used to define the in-memory
  layout of an array. It has the following key attributes:

  * `major_to_minor`: A tuple of integers specifying the dimension
    ordering in memory. For example, for a 2-dimensional array, `(0, 1)`
    indicates row-major layout and `(1, 0)` indicates column-major.

  * `_tiling`: An intentionally hidden, highly experimental, optional
    attribute to specify a tiled layout.

  * `AUTO`: A special, static sentinel object that can be used with
    `jax.jit` to request that the compiler automatically determine
    a good layout for a compiled function's input or output arrays.

* The `Format` class carries both a `Layout` and a `Sharding`, with
  either one taking on a default value when it is not specified.
  When the layout is explicitly specified, the sharding must be
  as well.

JAX API functions, such as `jax.jit` and `jax.device_put`, accept
`Sharding`s for sharding control or `Format`s for additional layout
control. They typically do not accept `Layout` instances directly.

## Specifying and reading layouts

By passing `Format` objects to `jax.jit` in place of shardings (in the
`in_shardings` and `out_shardings` arguments), you can guide the
compiler's layout decisions. Similarly you can pass `Format`s instead
of `Sharding`s to `jax.device_put` to control the layout of the
resulting array.

Let's see an example that uses both explicit and automatic layouts (as
in `Layout.AUTO`). Imagine we have two compiled functions, `init_fn`
and `apply_fn`. Say we expect `init_fn` to be called roughly once, but
`apply_fn` to be called on the output of `init_fn` many times, so that
we care much more about the performance of `apply_fn`. We may want to
have the compiler choose a good layout for `apply_fn` and constrain
`init_fn` to produce arrays of such layout. We can do this as follows:

```{code-cell}
import jax, jax.numpy as jnp
from jax.experimental.layout import Layout, Format
from jax.sharding import SingleDeviceSharding
import numpy as np

def init_fn(x, y):
  return x * 2, y * 3

def apply_fn(x, y):
  return x[0, :], y[:, 0]
```

Since `apply_fn` reads a contiguous column of its second argument `y`,
it makes sense to lay it out in column-major order (where columns are
stored contiguously). Using `Layout.AUTO`, we can ask the compiler to
infer good input layouts and see that it indeed chooses to request the
second argument in column-major layout.

```{code-cell}
shape = (4 * 128, 8 * 128)
duck = jax.ShapeDtypeStruct(shape, jnp.float32)

# Compile the `apply` function with layouts inferred automatically
apply_exe = jax.jit(
    apply_fn,
    in_shardings=Format(Layout.AUTO),
    out_shardings=Format(Layout.AUTO),
).trace(duck, duck).lower().compile()

# Read back the inferred input layout
arg_formats, kwarg_formats = apply_exe.input_formats
assert len(kwarg_formats) == 0
assert arg_formats[0].layout.major_to_minor == (0, 1)
assert arg_formats[1].layout.major_to_minor == (1, 0)
```

We can then compile `init_fn` to explicitly match this layout in its
outputs.

```{code-cell}
init_exe = jax.jit(init_fn, out_shardings=arg_formats).trace(
    duck, duck).lower().compile()

assert init_exe.output_formats == arg_formats
```

Finally we can see how the compiled `apply_fn` behaves when called
with differently laid out input arrays. The behavior varies with
whether inputs are
[committed](https://docs.jax.dev/en/latest/faq.html#controlling-data-and-computation-placement-on-devices). As
the following test demonstrates, if the argument arrays are committed,
then the pre-compiled `apply_fn` requires they match the layout
determined by the compiler above. Meanwhile it accepts uncommitted
arrays of any layout (including, of course, the inferred layout). In
this case, the arrays may be relaid out prior to invoking the compiled
computation.

```{code-cell}
def test(x, y, msg):
  print(f'-- {msg}:')
  print('x major_to_minor =', x.format.layout.major_to_minor)
  print('y major_to_minor =', y.format.layout.major_to_minor)
  try:
    apply_exe(x, y)
    print('-> `apply` called successfully')
  except ValueError as e:
    assert 'does not match' in str(e)
    print('-> error: mismatched input layouts')
  print()

dev = jax.devices()[0]

x1 = y1 = jnp.ones(shape)
test(x1, y1, 'uncommitted with mismatched layout')

x2, y2 = init_exe(x1, y1)
test(x2, y2, 'uncommitted with matching layout')

x3 = jnp.ones(shape)
y3 = jax.device_put(np.ones(shape), Format(Layout(major_to_minor=(1, 0)),
                                           SingleDeviceSharding(dev)))
test(x3, y3, 'committed with matching layout')

x4 = jnp.ones(shape)
y4 = jax.device_put(np.ones(shape), Format(Layout(major_to_minor=(0, 1)),
                                           SingleDeviceSharding(dev)))
test(x4, y4, 'committed with mismatched layout')
```

## Constraining intermediate layouts

We can also enforce a specific layout on an intermediate value within
a JIT-compiled function using `with_layout_constraint`:

```{code-cell}
from jax.experimental.layout import with_layout_constraint

@jax.jit
def f(x):
  y = x.T
  # Enforce a specific layout on `y`
  y = with_layout_constraint(y, Layout(major_to_minor=(0, 1)))
  return y * 2
```

This is analogous to
[`jax.lax.with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html),
for constraining layouts rather than shardings.
