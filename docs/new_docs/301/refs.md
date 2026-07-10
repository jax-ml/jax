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

(jax-301-refs)=
# Autodiff with refs

<!--* freshness: { reviewed: '2026-07-10' } *-->

Refs — JAX's mutable arrays, introduced in {ref}`jax-101-refs` — interact
with automatic differentiation in ways that go beyond what immutable values
can express: plumbing data out of backward passes, accumulating gradients
in place across many contributions, and differentiating with respect to
mutable state itself. This page covers the interaction in detail, ending
with a recipe that uses refs to relax `jax.lax.scan`'s fixed access pattern.

```{code-cell}
import jax
import jax.numpy as jnp
```

## `Ref`s and automatic differentiation

Autodiff can be applied to pure functions as before, even if they use array refs
internally. For example:

```{code-cell}
@jax.jit
def pure2(x):
  ref = jax.new_ref(x)
  ref[...] = ref[...] + ref[...]
  return ref[...]

print(jax.grad(pure2)(3.0))  # 2.0
```

Autodiff can also be applied to functions that take array refs as arguments.
The simplest case is when those ref arguments are only used for plumbing, and
aren't involved in differentiation. For example, `jax.grad` differentiates
with respect to its function's first argument by default, so ref arguments in
other positions are just along for the ride. Only non-differentiated values
can be written into such plumbing refs:

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

Differentiating _with respect to_ a ref-typed argument is another matter:
pointing `jax.grad` at one (e.g. via `argnums`) is an error, since the
gradient for a ref must itself live in a ref. Instead, use `jax.vjp` and
`with_refs`, described below.

You can combine plumbing refs with {func}`~jax.custom_vjp` to plumb data out of the
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
  x = stash_grads(grads_ref, x)
  x = jnp.sin(x)
  return x

grads_ref = jax.new_ref(0.)
jax.grad(f)(1., grads_ref)
print(grads_ref)  # Ref(0.54), the gradient at the stash point: cos(1.)
```

Notice `stash_grads_fwd` is returning a `Ref` here. That's a special
allowance for `custom_vjp` fwd rules: it's really syntax for indicating which
ref arguments should be shared by both the fwd and bwd rules. So any refs
returned by a fwd rule must be arguments to that fwd rule.

## Differentiating with respect to `Ref` arguments

The plumbing refs above are just passengers: they carry data out of the
computation, but no gradients flow through them. We can also differentiate
_with respect to_ a ref argument. Since the gradient for a `Ref`-typed input
is itself `Ref`-typed, `jax.grad` doesn't apply here. Instead we use
`jax.vjp`, and bind a gradient ref to the VJP function using its `with_refs`
method:

```{code-cell}
def f(x_ref):
  return x_ref[...] ** 2

x_ref = jax.new_ref(2.)
y, f_vjp = jax.vjp(f, x_ref)

x_grad_ref = jax.new_ref(0.)
f_vjp.with_refs(x_grad_ref)(1.0)  # bind the gradient ref, then apply the VJP
print(x_grad_ref)  # Ref(4.)
```

Here `with_refs` takes one entry per argument of the differentiated function
and returns a new VJP function with those gradient refs bound. When
differentiating with respect to a ref argument, using `with_refs` is
mandatory; the gradient needs a ref to be accumulated into, so calling the
VJP function without binding one is an error:

```{code-cell}
_, f_vjp = jax.vjp(f, jax.new_ref(2.))
try:
  f_vjp(1.0)  # error! no ref bound for the ref-typed argument's gradient
except Exception as e:
  print(e)  # ... gradient must be accumulated into a ref ... `with_refs` ...
```

The gradient is accumulated into the bound ref via `+=`, added to whatever
the ref already contains rather than overwriting it:

```{code-cell}
x_grad_ref = jax.new_ref(100.)
_, f_vjp = jax.vjp(f, jax.new_ref(2.))
f_vjp.with_refs(x_grad_ref)(1.0)
print(x_grad_ref)  # Ref(104.), i.e. 100. + 4.: accumulated, not set
```

Accumulating rather than setting might seem like an odd choice, but it means
one gradient ref can collect contributions from several backward passes with
no extra memory traffic, as in the examples below.

In-place updates to the ref argument inside the differentiated function are
differentiated too. The result is the gradient with respect to the ref's
initial value:

```{code-cell}
def g(x_ref):
  x_ref[...] = jnp.sin(x_ref[...])
  return x_ref[...] ** 2

x_ref = jax.new_ref(2.)
_, g_vjp = jax.vjp(g, x_ref)  # runs g, so x_ref is updated in-place here
g_grad_ref = jax.new_ref(0.)
g_vjp.with_refs(g_grad_ref)(1.0)
print(g_grad_ref)  # Ref(-0.757), i.e. 2*sin(2)*cos(2)
```

## Gradient refs for value arguments

When differentiating with respect to an ordinary `Array` argument,
`with_refs` is optional: we can call the VJP function directly and get the
gradient back as a value in the usual way, or we can bind a ref and have the
gradient accumulated into it in-place:

```{code-cell}
_, sin_vjp = jax.vjp(jnp.sin, 1.0)
x_bar, = sin_vjp(1.0)  # the usual way: gradient returned as a value
print(x_bar)  # 0.54

grad_ref = jax.new_ref(0.)
_, sin_vjp = jax.vjp(jnp.sin, 1.0)
sin_vjp.with_refs(grad_ref)(1.0)  # gradient accumulated into grad_ref
print(grad_ref)  # Ref(0.54)
```

We can mix and match. Each entry of `with_refs` can be:

* a `Ref`, meaning accumulate this argument's gradient into the ref in-place
  (the VJP call then returns a `jax.ad.GradRef()` placeholder in that
  position);
* `jax.ad.GradValue()`, meaning return this argument's gradient as a value in
  the usual way (the default); or
* `jax.ad.DontWant()`, meaning don't compute this argument's gradient at all
  (the VJP call returns a `jax.ad.DidntWant()` placeholder in that position —
  more on this below).

One reason to use a gradient ref here is to exploit sparsity. Consider
differentiating a function that slices its input:

```{code-cell}
@jax.jit
def take(x, i):
  return x[i]

x = jnp.arange(10.)

_, take_vjp = jax.vjp(take, x, 3)
x_bar, _ = take_vjp(1.0)
print(x_bar)  # [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
```

The gradient with respect to `x` is one-hot: the backward pass materializes a
dense array of zeros and sets a single element of it. If we compute many such
gradients and sum them, say over a loop of sparse accesses, we pay for a
dense array's worth of memory traffic on each one, even though each
contribution only touches one element.

If we instead bind a gradient ref with `with_refs`, each backward pass
performs a sparse in-place add-update, writing only where it needs to:

```{code-cell}
grad_ref = jax.new_ref(jnp.zeros(10))

for i in [3, 5, 3]:
  _, take_vjp = jax.vjp(take, x, i)
  take_vjp.with_refs(grad_ref, jax.ad.GradValue())(1.0)  # no gradient ref for i

print(grad_ref)  # Ref([0., 0., 0., 2., 0., 1., 0., 0., 0., 0.])
```

We can check that the update is sparse by inspecting the jaxpr of a single
VJP application:

```{code-cell}
@jax.make_jaxpr
def take_vjp_jaxpr():
  _, take_vjp = jax.vjp(take, x, 3)
  take_vjp.with_refs(grad_ref, jax.ad.GradValue())(1.0)

print(take_vjp_jaxpr())
```

The backward pass boils down to `b[c:c+1] += j`, an in-place add-update of
one element of the gradient ref, with no dense one-hot array in sight.

## `DontWant`: skipping unneeded gradients

The VJP function returned by `jax.vjp` computes gradients for all the
arguments of the differentiated function. But sometimes we don't need all of
them, like when differentiating with respect to parameters but not data.
Passing `jax.ad.DontWant()` for an argument tells the backward pass not to
compute its gradient at all, playing the same role for `jax.vjp` that
`argnums` plays for `jax.grad`:

```{code-cell}
def predict(W, x):
  return W @ x

W = jnp.ones((4, 4))
x = jnp.ones(4)

_, f_vjp = jax.vjp(predict, W, x)
W_bar, x_bar = f_vjp.with_refs(jax.ad.GradValue(), jax.ad.DontWant())(jnp.ones(4))
print(W_bar[0])  # [1., 1., 1., 1.]
print(x_bar)  # DidntWant()
```

This is more than a convenience: it can save real work in the backward pass,
like an eager form of dead code elimination. Transpose rules can check for
`DontWant` and skip computing the corresponding cotangents. For example, the
transpose of matrix multiplication usually computes two dot products, one for
each operand's gradient, but with `DontWant` it computes only one. We can see
that by counting the `dot_general` operations in the jaxpr of each VJP
application:

```{code-cell}
_, f_vjp = jax.vjp(predict, W, x)
both = jax.make_jaxpr(lambda: f_vjp(jnp.ones(4)))()
only_W = jax.make_jaxpr(
    lambda: f_vjp.with_refs(jax.ad.GradValue(), jax.ad.DontWant())(jnp.ones(4)))()

print(str(both).count('dot_general'))    # 2, one dot for each gradient
print(str(only_W).count('dot_general'))  # 1, the dot for x_bar is skipped
```

## Example: gradient accumulation over microbatches

Here's a more realistic recipe that puts these pieces together. In pipelined
training, we often split a batch into microbatches, run a forward and
backward pass one microbatch at a time, and accumulate weight gradients as we
go. Using `with_refs`, each microbatch's backward pass accumulates directly
into a single fixed gradient buffer:

```{code-cell}
NUM_LAYERS = 3
NUM_MUBATCHES = 5
MUBATCH_SIZE = 7

def mubatch_loss(Ws, xs):
  # inner loop over layers
  act, _ = jax.lax.scan(lambda x, W: (jnp.dot(x, W), None), xs, Ws)
  return jnp.mean(act)

def process_batch(Ws, xs_batch):
  grad_acc = jax.new_ref(jnp.zeros_like(Ws))

  def process_mubatch(_, xs):
    loss, f_vjp = jax.vjp(lambda Ws: mubatch_loss(Ws, xs), Ws)
    f_vjp.with_refs(grad_acc)(jnp.ones_like(loss))  # accumulate in-place
    return (), loss

  xs_mubatches = xs_batch.reshape(NUM_MUBATCHES, MUBATCH_SIZE, -1)
  # outer loop over microbatches
  (), losses = jax.lax.scan(process_mubatch, (), xs_mubatches)
  return jax.freeze(grad_acc), losses

Ws = jnp.ones((NUM_LAYERS, 4, 4))
xs_batch = jnp.ones((NUM_MUBATCHES * MUBATCH_SIZE, 4))
grads, losses = process_batch(Ws, xs_batch)
```

Each iteration of the outer scan runs a forward and backward pass for one
microbatch, and `with_refs(grad_acc)` makes the backward pass add that
microbatch's gradient contribution directly into `grad_acc`. Note that the
scan body closes over `grad_acc`, which is fine for `scan` (though it
wouldn't be for `vmap` or `shard_map`, as discussed above). Once all the
microbatches are processed, we `freeze` the accumulator to get the total
batch gradient as an immutable `Array`.

The result matches what we'd get by differentiating the summed loss directly:

```{code-cell}
xs_mubatches = xs_batch.reshape(NUM_MUBATCHES, MUBATCH_SIZE, -1)
grads_expected = jax.grad(
    lambda Ws: sum(mubatch_loss(Ws, xs) for xs in xs_mubatches))(Ws)
print(jnp.allclose(grads, grads_expected, atol=1e-3, rtol=1e-3))  # True
```

But unlike that version, the ref-based version never materializes
per-microbatch gradients as separate arrays: there's one gradient buffer,
allocated once, no matter how many microbatches we process.

## `foreach`, a new way to write `scan`

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
