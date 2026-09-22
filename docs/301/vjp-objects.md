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

(jax-301-vjp-objects)=
# First-class VJPs

<!--* freshness: { reviewed: '2026-07-16' } *-->

The callable returned by `jax.vjp` is a *VJP object*, a first-class value.
It's a pytree whose leaves are the residual values saved by the forward
pass, so it can be passed into and out of compiled functions, serialized,
or offloaded like any other data, and its saved state can be inspected and
edited. This page covers what that enables: splitting the forward and
backward passes into separately compiled functions run on your own schedule,
and excluding argument values (like weights) from the saved state with
`saveable_args`, for example to re-gather sharded weights on the backward
pass instead of saving them.

```{code-cell}
import jax
import jax.numpy as jnp
from jax import grad, jit

jax.config.update('jax_num_cpu_devices', 4)  # for the sharded example below
```

(jax-301-fwd-bwd-split)=
## Splitting the forward and backward passes

`jax.grad` and `jax.vjp` package the forward and backward passes together:
under a `jax.jit`, they compile into a single program. Usually that's what
you want, but sometimes it's useful to run the forward and backward passes
as *separate* compiled functions, say to interleave the forward and
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

This works because the callable returned by `jax.vjp` is itself a pytree:
its leaves are the residual values saved by the forward pass, and its tree
structure records the backward-pass computation. So it can be returned out
of one jitted function and passed into another, like any other data.
Notice that there's nothing specific to `f` in `bwd`: it only applies its
argument.

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

The VJP object exposes its saved state in three attributes: `args_res` holds
argument values that the backward pass needs *verbatim*, arranged to mirror
the arguments; `opaque_residuals` holds values computed during the forward
pass; and `structured_residuals` is a third channel, described below. For
`layer`, the backward pass of `x @ W` needs both `x` and `W` as they were:

```{code-cell}
y, f_vjp = jax.vjp(layer, W1, x0)
print(f_vjp.args_res)
```

(An argument the backward pass doesn't need would appear as a `NotNeeded()`
sentinel instead.)

So every VJP object in the pipeline example above carries a full copy of its
layer's weights. If we're running many microbatches through the same layer
before applying their backward passes, or serializing or offloading each
VJP object, every one of them duplicates the weights, which are typically
the biggest thing in the saved state and which we already have. Only the
activations vary per microbatch.

The third attribute, `structured_residuals`, holds residuals that keep a
user-meaningful structure instead of being flattened into the opaque list.
Custom derivative rules can save named pytrees of residuals there, and
transformations carry them through with structure intact: `scan` stacks
entries across iterations, `cond` records a tagged sum marking which branch
ran, and `shard_map` stacks per-shard entries along a leading mesh axis. JAX
can also deduplicate what's saved, storing a value that appears under several
names just once (an optimization, not a guarantee). For typical programs
`structured_residuals` is empty; populating it is up to hijax primitives'
rules, covered in {ref}`jax-301-structured-residuals`.

(jax-301-saveable-args)=
## Marking arguments not saveable: `saveable_args`

The `saveable_args` argument to `jax.vjp` is a tuple-tree of bools (nested
tuples with bool leaves), one entry per argument, defaulting to the single
bool `True`, meaning everything is saveable. Where a `False` applies,
argument values that would have been saved verbatim are instead replaced
with `NotSaveable()` sentinels:

```{code-cell}
y, f_vjp = jax.vjp(layer, W1, x0, saveable_args=(False, True))
print(f_vjp.args_res)
```

`NotSaveable` is an empty pytree node, so flattening the VJP object (to
serialize or offload it) includes no leaves for those arguments:

```{code-cell}
print(len(jax.tree.leaves(f_vjp)))  # 3, not 4: W1 isn't part of the saved state
```

## Restoring before the backward pass

Before the VJP function can be applied, the missing values must be restored.
Calling without restoring them raises an error naming the arguments still
needing restoration:

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

(jax-301-fsdp-vjp)=
## Example: re-gathering sharded weights

In fully sharded data parallelism (FSDP), each device stores only a shard of
each weight matrix, and all-gathers the full matrix just before using it
({ref}`jax-201-shard-map` builds this with `jax.remat`). Autodiff would
ordinarily save the gathered weights for the backward pass, so every device
would end up holding every full weight matrix after all. With
`saveable_args`, a layer can instead save only its shard and re-gather on the
backward pass.

Here's a decorator that does that for any layer function `layer(w, x)`,
where `w` is an array or pytree of weights sharded along the leading axis
over the mesh axis `'devices'`. It takes over the backward pass with
`jax.custom_vjp` (see {doc}`custom-jvp-vjp`). The forward rule calls
`jax.vjp` with the gathered weights marked not saveable. The backward rule
re-gathers them, restores them into the VJP object, and reduce-scatters the
weight gradient back into shards:

```{code-cell}
jax.set_mesh(jax.make_mesh((4,), ('devices',)))

def gather(w_frag):
  return jax.tree.map(lambda a: jax.lax.all_gather(a, 'devices', tiled=True), w_frag)

def scatter(w_bar):
  return jax.tree.map(lambda a: jax.lax.psum_scatter(
      a, 'devices', scatter_dimension=0, tiled=True), w_bar)

def fsdp(layer, prevent_cse=True):
  @jax.custom_vjp
  def f(w_frag, x):
    return layer(gather(w_frag), x)

  def f_fwd(w_frag, x):
    y, layer_vjp = jax.vjp(layer, gather(w_frag), x, saveable_args=(False, True))
    return y, (layer_vjp, w_frag)  # save the shard, not the gathered weights

  def f_bwd(res, y_bar):
    layer_vjp, w_frag = res
    if prevent_cse:
      w_frag = jax.lax.optimization_barrier(w_frag)
    layer_vjp.args_res[0] = gather(w_frag)  # re-gather
    w_bar, x_bar = layer_vjp(y_bar)
    return scatter(w_bar), x_bar

  f.defvjp(f_fwd, f_bwd)
  return f
```

The {func}`~jax.lax.optimization_barrier` matters. The forward and backward
passes here compile into one program, and without the barrier, XLA's
common-subexpression elimination would notice that the backward pass's
`gather(w_frag)` matches the forward pass's, and reuse the forward result
instead. That keeps the gathered weights alive until the backward pass, the
very thing we set out to avoid. (`jax.checkpoint` guards against the same
problem with its `prevent_cse` option, hence the name here.)

Let's try it on a small three-layer network, with the weights and the batch
both sharded over four devices, and check the gradients against an unsharded
reference:

```{code-cell}
def dense(W, x):
  return jnp.tanh(x @ W)

def make_loss(layer):
  @jax.shard_map(in_specs=(jax.P('devices'), jax.P('devices')), out_specs=jax.P())
  def loss(Ws, xs):
    for W in Ws:
      xs = layer(W, xs)
    return jax.lax.pmean(jnp.mean(xs ** 2), 'devices')
  return loss

def loss_ref(Ws, xs):
  for W in Ws:
    xs = dense(W, xs)
  return jnp.mean(xs ** 2)

Ws = [jax.random.normal(jax.random.key(i), (16, 16)) / 4 for i in range(3)]
xs = jax.random.normal(jax.random.key(3), (8, 16))
Ws_sharded, xs_sharded = jax.device_put((Ws, xs), jax.P('devices'))

loss_fsdp = make_loss(fsdp(dense))
grads = jax.jit(jax.grad(loss_fsdp))(Ws_sharded, xs_sharded)
grads_ref = jax.jit(jax.grad(loss_ref))(Ws, xs)
print(all(jnp.allclose(g, g_ref, atol=1e-5) for g, g_ref in zip(grads, grads_ref)))
print(jax.typeof(grads[0]))  # sharded like the weights
```

Counting all-gathers in the compiled gradient computation shows the barrier
doing its job:

```{code-cell}
def count_all_gathers(loss, *args):
  hlo = jax.jit(jax.grad(loss)).lower(*args).compile().as_text()
  return hlo.count('all-gather(')

print(count_all_gathers(loss_fsdp, Ws_sharded, xs_sharded))
print(count_all_gathers(make_loss(fsdp(dense, prevent_cse=False)),
                        Ws_sharded, xs_sharded))
```

With the barrier there are 5: one per layer on the forward pass, plus 2
re-gathers on the backward pass. (The first layer's weights would be needed
only for a gradient with respect to the input data, which isn't computed.)
Without it, XLA merges the re-gathers into the forward all-gathers, leaving
3, just as if the gathered weights had been saved.

## Details

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

Three more details:

- Only argument values saved *verbatim* are affected. Residuals *computed*
  from arguments are saved in `opaque_residuals` as usual, and
  `saveable_args` never causes recomputation. For the save-versus-recompute
  tradeoff, see {doc}`remat`.
- Arguments the backward pass doesn't need stay `NotNeeded()` even when
  marked `False`, so `args_res` shows exactly which values must be restored.
- If you restore a value by *recomputing* it in the same compiled program as
  the forward pass, as in the FSDP example above, apply
  {func}`jax.lax.optimization_barrier` to the recomputation's inputs.
  Otherwise XLA's common-subexpression elimination may reuse the forward
  pass's copy instead, silently undoing the memory savings. Separately
  compiled forward and backward functions, like `fwd_light` and `bwd_light`,
  don't need this, since CSE doesn't act across programs.
