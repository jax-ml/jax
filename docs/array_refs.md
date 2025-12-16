---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst,py
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
---

```{raw-cell}

---
Copyright 2025 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---
```

# `Ref`: mutable arrays for data plumbing and memory control

JAX `Array`s are immutable, representing mathematical values. Immutability can
make code easier to reason about, and is useful for optimized compilation,
parallelization, rematerialization, and transformations like autodiff.

But immutability is constraining too:
* **expressiveness** --- plumbing out intermediate data or maintaining state,
  e.g. for normalization statistics or metrics, can feel heavyweight;
* **performance** --- it's more difficult to reason about performance, like
  memory lifetimes and in-place updates.

`Ref`s can help! They represent mutable arrays that can be read and written
in-place. These array references are compatible with JAX transformations, like
`jax.jit` and `jax.grad`:

```{code-cell}
import jax
import jax.numpy as jnp

x_ref = jax.new_ref(jnp.zeros(3))  # new array ref, with initial value [0., 0., 0.]

@jax.jit
def f():
  x_ref[1] += 1.  # indexed add-update

print(x_ref)  # Ref([0., 0., 0.])
f()
f()
print(x_ref)  # Ref([0., 2., 0.])
```

The indexing syntax follows NumPy's. For a `Ref` called `x_ref`, we can
read its entire value into an `Array` by writing `x_ref[...]`, and write its
entire value using `x_ref[...] = A` for some `Array`-valued expression `A`:

```{code-cell}
def g(x):
  x_ref = jax.new_ref(0.)
  x_ref[...] = jnp.sin(x)
  return x_ref[...]

print(jax.grad(g)(1.0))  # 0.54
```

`Ref` is a distinct type from `Array`, and it comes with some important
constraints and limitations. In particular, indexed reading and writing is just
about the *only* thing you can do with an `Ref`. References can't be passed
where `Array`s are expected:

```{code-cell}
x_ref = jax.new_ref(1.0)
try:
  jnp.sin(x_ref)  # error! can't do math on refs
except Exception as e:
  print(e)
```

To do math, you need to read the ref's value first, like `jnp.sin(x_ref[...])`.

So what _can_ you do with `Ref`? Read on for the details, and some useful
recipes.

### API

If you've ever used
[Pallas](https://docs.jax.dev/en/latest/pallas/quickstart.html), then `Ref`
should look familiar. A big difference is that you can create new `Ref`s
yourself directly using `jax.new_ref`:

```{code-cell}
from jax import Array, Ref

def array_ref(init_val: Array) -> Ref:
  """Introduce a new reference with given initial value."""
```

`jax.freeze` is its antithesis, invalidating the given ref (so that accessing it
afterwards is an error) and producing its final value:

```{code-cell}
def freeze(ref: Ref) -> Array:
  """Invalidate given reference and produce its final value."""
```

In between creating and destroying them, you can perform indexed reads and
writes on refs. You can read and write using the functions `jax.ref.get` and
`jax.ref.swap`, but usually you'd just use NumPy-style array indexing syntax:

```{code-cell}
import types
Index = int | slice | Array | types.EllipsisType
Indexer = Index | tuple[Index, ...]

def get(ref: Ref, idx: Indexer) -> Array:
  """Returns `ref[idx]` for NumPy-style indexer `idx`."""

def swap(ref: Ref, idx: Indexer, val: Array) -> Array:
  """Performs `newval, ref[idx] = ref[idx], val` and returns `newval`."""
```

Here, `Indexer` can be any NumPy indexing expression:

```{code-cell}
x_ref = jax.new_ref(jnp.arange(12.).reshape(3, 4))

# int indexing
row = x_ref[0]
x_ref[1] = row

# tuple indexing
val = x_ref[1, 2]
x_ref[2, 3] = val

# slice indexing
col = x_ref[:, 1]
x_ref[0, :3] = col

# advanced int array indexing
vals = x_ref[jnp.array([0, 0, 1]), jnp.array([1, 2, 3])]
x_ref[jnp.array([1, 2, 1]), jnp.array([0, 0, 1])] = vals
```

As with `Array`s, indexing mostly follows NumPy behavior, except for
out-of-bounds indexing which [behaves in the usual way for JAX
`Array`s](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing).

### Pure and impure functions

A function that takes a ref as an argument (either explicitly or by lexical
closure) is considered _impure_. For example:

```{code-cell}
# takes ref as an argument => impure
@jax.jit
def impure1(x_ref, y_ref):
  x_ref[...] = y_ref[...]

# closes over ref => impure
y_ref = jax.new_ref(0)

@jax.jit
def impure2(x):
  y_ref[...] = x
```

If a function only uses refs internally, it is still considered _pure_. Purity
is in the eye of the caller. For example:

```{code-cell}
# internal refs => still pure
@jax.jit
def pure1(x):
  ref = jax.new_ref(x)
  ref[...] = ref[...] + ref[...]
  return ref[...]
```

Pure functions, even those that use refs internally, are familiar: for example,
they work with transformations like `jax.grad`, `jax.vmap`, `jax.shard_map`, and
others in the usual way.

Impure functions are sequenced in Python program order.

### Restrictions

`Ref`s are second-class, in the sense that there are restrictions on their
use:

* **Can't return refs** from `jit`\-decorated functions or the bodies of
  higher-order primitives like `jax.lax.scan`, `jax.lax.while_loop`, or
  `jax.lax.cond`
* **Can't pass a ref as an argument more than once** to `jit`\-decorated
  functions or higher-order primitives
* **Can only `freeze` in creation scope**
* **No higher-order refs** (refs-to-refs)

For example, these are errors:

```{code-cell}
x_ref = jax.new_ref(0.)

# can't return refs
@jax.jit
def err1(x_ref):
  x_ref[...] = 5.
  return x_ref  # error!
try:
  err1(x_ref)
except Exception as e:
  print(e)

# can't pass a ref as an argument more than once
@jax.jit
def err2(x_ref, y_ref):
  ...
try:
  err2(x_ref, x_ref)  # error!
except Exception as e:
  print(e)

# can't pass and close over the same ref
@jax.jit
def err3(y_ref):
  y_ref[...] = x_ref[...]
try:
  err3(x_ref)  # error!
except Exception as e:
  print(e)

# can only freeze in creation scope
@jax.jit
def err4(x_ref):
  jax.freeze(x_ref)
try:
  err4(x_ref)  # error!
except Exception as e:
  print(e)
```

These restrictions exist to rule out aliasing, where two refs might refer to the
same mutable memory, making programs harder to reason about and transform.
Weaker restrictions would also suffice, so some of these restrictions may be
lifted as we improve JAX's ability to verify that no aliasing is present.

There are also restrictions stemming from undefined semantics, e.g. in the
presence of parallelism or rematerialization:

* **Can't `vmap` or `shard_map` a function that closes over refs**
* **Can't apply `jax.remat`/`jax.checkpoint` to an impure function**

For example, here are ways you can and can't use `vmap` with impure functions:

```{code-cell}
# vmap over ref args is okay
def dist(x, y, out_ref):
  assert x.ndim == y.ndim == 1
  assert out_ref.ndim == 0
  out_ref[...] = jnp.sum((x - y) ** 2)

vecs = jnp.arange(12.).reshape(3, 4)
out_ref = jax.new_ref(jnp.zeros((3, 3)))
jax.vmap(jax.vmap(dist, (0, None, 0)), (None, 0, 0))(vecs, vecs, out_ref)  # ok!
print(out_ref)
```

```{code-cell}
# vmap with a closed-over ref is not
x_ref = jax.new_ref(0.)

def err5(x):
  x_ref[...] = x

try:
  jax.vmap(err5)(jnp.arange(3.))  # error!
except Exception as e:
  print(e)
```

The latter is an error because it's not clear which value `x_ref` should be
after we run `jax.vmap(err5)`.

### `Ref`s and automatic differentiation

Autodiff can be applied to pure functions as before, even if they use array refs
internally. For example:

```{code-cell}
@jax.jit
def pure2(x):
  ref = jax.new_ref(x)
  ref[...] = ref[...] + ref[...]
  return ref[...]

print(jax.grad(pure1)(3.0))  # 2.0
```

Autodiff can also be applied to functions that take array refs as arguments, if
those arguments are only used for plumbing and not involved in differentiation:

```{code-cell}
# error
def err6(x, some_plumbing_ref):
  y = x + x
  some_plumbing_ref[...] += y
  return y

# fine
def foo(x, some_plumbing_ref):
  y = x + x
  some_plumbing_ref[...] += jax.lax.stop_gradient(y)
  return y
```

You can combine plumbing refs with `custom_vjp` to plumb data out of the
backward pass of a differentiated function:

```{code-cell}
# First, define the helper `stash_grads`:

@jax.custom_vjp
def stash_grads(grads_ref, x):
  return x

def stash_grads_fwd(grads_ref, x):
  return x, grads_ref

def stash_grads_bwd(grads_ref, g):
  grads_ref[...] = g
  return None, g

stash_grads.defvjp(stash_grads_fwd, stash_grads_bwd)
```

```{code-cell}
# Now, use `stash_grads` to stash intermediate gradients:

def f(x, grads_ref):
  x = jnp.sin(x)
  x = stash_grads(grads_ref, x)
  return x

grads_ref = jax.new_ref(0.)
f(1., grads_ref)
print(grads_ref)
```

Notice `stash_grads_fwd` is returning a `Ref` here. That's a special
allowance for `custom_vjp` fwd rules: it's really syntax for indicating which
ref arguments should be shared by both the fwd and bwd rules. So any refs
returned by a fwd rule must be arguments to that fwd rule.

### `Ref`s and performance

At the top level, when calling `jit`\-decorated functions, `Ref`s obviate
the need for donation, since they are effectively always donated:

```{code-cell}
@jax.jit
def sin_inplace(x_ref):
  x_ref[...] = jnp.sin(x_ref[...])

x_ref = jax.new_ref(jnp.arange(3.))
print(x_ref.unsafe_buffer_pointer(), x_ref)
sin_inplace(x_ref)
print(x_ref.unsafe_buffer_pointer(), x_ref)
```

Here `sin_inplace` operates in-place, updating the buffer backing `x_ref` so
that its address stays the same.

Under a `jit`, you should expect array references to point to fixed buffer
addresses, and for indexed updates to be performed in-place.

**Temporary caveat:** dispatch from Python to impure `jit`\-compiled functions
that take `Ref` inputs is currently slower than dispatch to pure
`jit`\-compiled functions, since it takes a less optimized path.

### `foreach`, a new way to write `scan`

As you may know, `jax.lax.scan` is a loop construct with a built-in fixed access
pattern for scanned-over inputs and outputs. The access pattern is built in for
autodiff reasons: if we were instead to slice into immutable inputs directly,
reverse-mode autodiff would end up creating one-hot gradients and summing them
up, which can be asymptotically inefficient. See [Sec 5.3.3 of the Dex
paper](https://arxiv.org/pdf/2104.05372).

But reading slices of `Ref`s doesn't have this efficiency problem: when we
apply reverse-mode autodiff, we always generate in-place accumulation
operations. As a result, we no longer need to be constrained by `scan`'s fixed
access pattern. We can write more flexible loops, e.g. with non-sequential
access.

Moreover, having mutation available allows for some syntax tricks, like in this
recipe for a `foreach` decorator:

```{code-cell}
import jax
import jax.numpy as jnp
from jax.lax import scan

def foreach(*args):
  def decorator(body):
    return scan(lambda _, elts: (None, body(*elts)), None, args)[1]
  return decorator
```

```{code-cell}
r = jax.new_ref(0)
xs = jnp.arange(10)

@foreach(xs)
def ys(x):
  r[...] += x
  return x * 2

print(r)   # Ref(45, dtype=int32)
print(ys)  # [ 0  2  4  6  8 10 12 14 16 18]
```

Here, the loop runs immediately, updating `r` in-place and binding `ys` to be
the mapped result.
