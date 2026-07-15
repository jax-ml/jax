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

(jax-301-vjp-objects)=
# First-class VJPs

<!--* freshness: { reviewed: '2026-07-16' } *-->

The callable returned by `jax.vjp` is more than a closure to call once and
throw away: it's a *VJP object*, a first-class value. It's a pytree whose
leaves are the residual values saved by the forward pass, so it can be
passed into and out of compiled functions, serialized, or offloaded like any
other data — and its saved state can be inspected and edited. This page
covers what that enables: splitting the forward and backward passes into
separately compiled functions run on your own schedule, and excluding
argument values (like weights) from the saved state with `saveable_args`.

```{code-cell}
import jax
import jax.numpy as jnp
from jax import grad, jit
```

(jax-301-fwd-bwd-split)=
## Splitting the forward and backward passes

`jax.grad` and `jax.vjp` package the forward and backward passes together:
under a `jax.jit`, they compile into a single program. Usually that's what
you want. But sometimes it's useful to run the forward and backward passes
as *separate* compiled functions — say, to interleave the forward and
backward passes of different microbatches or pipeline stages on a schedule
of your own, with each function compiled once and reused many times.

You can build this out of `jax.vjp` directly:

```{code-cell}
def fwd_and_bwd(f):
  def fwd(*args):
    return jax.vjp(f, *args)
  def bwd(f_vjp, y_bar):
    return f_vjp(y_bar)
  return jit(fwd), jit(bwd)
```

The trick is that the callable returned by `jax.vjp` is itself a pytree: its
leaves are the residual values saved by the forward pass, and its tree
structure records the backward-pass computation. So it can be returned out
of one jit-compiled function and passed into another, like any other data.
Notice that there's nothing specific to `f` in `bwd` — it's just "apply".

Each of `fwd` and `bwd` compiles once, and we can call them however many
times and in whatever order we like:

```{code-cell}
def layer(W, x):
  return jnp.tanh(x @ W)

fwd, bwd = fwd_and_bwd(layer)

W1, W2 = jnp.ones((3, 3)), 2. * jnp.ones((3, 3))
x0 = jnp.ones((2, 3))

# forward through two layers, then backward, on our own schedule:
x1, res1 = fwd(W1, x0)
x2, res2 = fwd(W2, x1)
dW2, dx1 = bwd(res2, jnp.ones_like(x2))
dW1, dx0 = bwd(res1, dx1)
```

That computes the same gradients that an end-to-end `jax.grad` would:

```{code-cell}
def two_layers(W1, W2, x):
  return jnp.sum(layer(W2, layer(W1, x)))

dW1_ref, dW2_ref = grad(two_layers, argnums=(0, 1))(W1, W2, x0)
print(jnp.allclose(dW1, dW1_ref), jnp.allclose(dW2, dW2_ref))
```

JAX also provides this pattern prepackaged as `jax.fwd_and_bwd`, with an
`argnums` argument selecting which inputs to produce cotangents for:

```{code-cell}
fwd, bwd = jax.fwd_and_bwd(layer, argnums=(0, 1))

y, residuals = fwd(W1, x0)
dW1, dx0 = bwd(residuals, jnp.ones_like(y))
print(dW1.shape, dx0.shape)
```

## What the VJP object saves

The VJP object exposes its saved state in two attributes: `args_res` holds
argument values that the backward pass needs *verbatim*, arranged to mirror
the arguments, and `opaque_residuals` holds values computed during the
forward pass. For `layer`, the backward pass of `x @ W` needs both `x` and
`W` as they were:

```{code-cell}
y, f_vjp = jax.vjp(layer, W1, x0)
print(f_vjp.args_res)
```

(An argument the backward pass doesn't need would appear as a `NotNeeded()`
sentinel instead.)

So every VJP object in the pipeline example above carries a full copy of its
layer's weights. If we're running many microbatches through the same layer
before applying their backward passes — or serializing or offloading each
VJP object — every one of them duplicates the weights, which are typically
the biggest thing in the saved state and which we already have. Only the
activations vary per microbatch.

## Marking arguments not saveable: `saveable_args`

The `saveable_args` argument to `jax.vjp` is a tuple-tree of bools (nested
tuples with bool leaves), one entry per argument, defaulting to the single
bool `True` — everything saveable, the usual behavior. Where a `False`
applies, argument values that would have been saved verbatim are instead
replaced with `NotSaveable()` sentinels:

```{code-cell}
y, f_vjp = jax.vjp(layer, W1, x0, saveable_args=(False, True))
print(f_vjp.args_res)
```

`NotSaveable` is an empty pytree node, so flattening the VJP object — to
serialize or offload it — includes no leaves for those arguments:

```{code-cell}
print(len(jax.tree.leaves(f_vjp)))  # 3, not 4: W1 isn't part of the saved state
```

## Restoring before the backward pass

Before the VJP function can be applied, the missing values must be restored.
Forgetting is an error that names the arguments still needing restoration:

```{code-cell}
try:
  f_vjp(jnp.ones_like(y))
except ValueError as e:
  print(e)
```

Restore by assigning into `args_res`, or more functionally with `replace`:

```{code-cell}
f_vjp.args_res[0] = W1
# or: f_vjp = f_vjp.replace(args_res=[W1, f_vjp.args_res[1]])
dW1, dx0 = f_vjp(jnp.ones_like(y))
print(dW1.shape, dx0.shape)
```

Here's the pipeline example again, with the weights left out of the saved
state and instead passed to the backward function directly:

```{code-cell}
def fwd_light(W, x):
  return jax.vjp(layer, W, x, saveable_args=(False, True))

def bwd_light(f_vjp, W, y_bar):
  f_vjp.args_res[0] = W
  return f_vjp(y_bar)

fwd_light, bwd_light = jit(fwd_light), jit(bwd_light)

x1, res1 = fwd_light(W1, x0)
x2, res2 = fwd_light(W2, x1)
dW2, dx1 = bwd_light(res2, W2, jnp.ones_like(x2))
dW1, dx0 = bwd_light(res1, W1, dx1)
print(jnp.allclose(dW1, dW1_ref), jnp.allclose(dW2, dW2_ref))
```

## The fine print

`saveable_args` must form a tree prefix of the arguments, in a loose sense:
containers are matched only by their number of children, so a tuple entry
can line up with a dict argument, and a single bool broadcasts over a whole
argument subtree (that's how the default `True` covers everything). When
restoring, you can assign values with the original pytree structure:

```{code-cell}
def g(d):
  return d['bye'] @ d['hi']

d = {'hi': jnp.ones((4, 5)), 'bye': jnp.ones((3, 4))}
_, g_vjp = jax.vjp(g, d, saveable_args=((True, False),))
g_vjp.args_res = [d]  # restore with the original dict
d_grad, = g_vjp(jnp.ones((3, 5)))
print(jax.tree.map(jnp.shape, d_grad))
```

Two more details worth knowing:

- Only argument values saved *verbatim* are affected. Residuals *computed*
  from arguments are saved in `opaque_residuals` as usual — `saveable_args`
  never causes recomputation. For the save-versus-recompute tradeoff, see
  {doc}`remat`.
- Arguments the backward pass doesn't need stay `NotNeeded()` even when
  marked `False`, so `args_res` shows exactly which values must be restored.
