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

(jax-301-remat)=
# Gradient checkpointing with `jax.checkpoint` (`jax.remat`)

<!--* freshness: { reviewed: '2026-07-10' } *-->

When you differentiate a function in reverse mode, JAX's autodiff saves
intermediate values from the forward pass — called *residuals* — to use in
the backward pass. For big models, residual memory is often *the* memory
problem. The {func}`jax.checkpoint` decorator (aliased as {func}`jax.remat`)
gives you control over which intermediates are saved and which are
*rematerialized* — recomputed on the backward pass — trading memory for
FLOPs.

This page describes JAX's new rematerialization implementation, enabled with
the `jax_remat3` flag (on its way to becoming the default):

```{code-cell}
import jax

jax.config.update('jax_remat3', True)

import jax.numpy as jnp
```

## Seeing what's saved

Consider a little three-layer network:

```{code-cell}
def g(W, x):
  y = jnp.dot(W, x)
  return jnp.sin(y)

def f(W1, W2, W3, x):
  x = g(W1, x)
  x = g(W2, x)
  x = g(W3, x)
  return x

W1 = jnp.ones((5, 4))
W2 = jnp.ones((6, 5))
W3 = jnp.ones((7, 6))
x = jnp.ones(4)

# Inspect the 'residual' values to be saved on the forward pass
# if you were to evaluate `jax.grad(f)(W1, W2, W3, x)`
from jax.ad_checkpoint import print_saved_residuals
print_saved_residuals(f, W1, W2, W3, x)
```

By applying {func}`jax.checkpoint` to sub-functions, as a decorator or at
specific application sites, you force JAX not to save any of that
sub-function's residuals. Instead, only the inputs of a
{func}`jax.checkpoint`-decorated function might be saved, and any residuals
consumed on the backward pass are recomputed from those inputs as needed:

```{code-cell}
def f2(W1, W2, W3, x):
  x = jax.checkpoint(g)(W1, x)
  x = jax.checkpoint(g)(W2, x)
  x = jax.checkpoint(g)(W3, x)
  return x

print_saved_residuals(f2, W1, W2, W3, x)
```

Here, the values of two `sin` applications are saved because they are
arguments in subsequent applications of the
{func}`jax.checkpoint`-decorated `g` function, and inputs to a
{func}`jax.checkpoint`-decorated function may be saved. But no values of
`cos` applications are saved.

To control what's saved without editing the function to be differentiated,
name the values you care about with
{func}`jax.ad_checkpoint.checkpoint_name`, and select among the names with a
rematerialization *policy*:

```{code-cell}
from jax.ad_checkpoint import checkpoint_name

def f4(W1, W2, W3, x):
  x = checkpoint_name(g(W1, x), name='a')
  x = checkpoint_name(g(W2, x), name='b')
  x = checkpoint_name(g(W3, x), name='c')
  return x

f4 = jax.checkpoint(f4, policy=jax.checkpoint_policies.save_only_these_names('a'))
print_saved_residuals(f4, W1, W2, W3, x)
```

When playing around with these toy examples, you can get a closer look at
what's going on using a custom `print_fwd_bwd` utility defined here:

```{code-cell}
from jax.tree_util import tree_flatten, tree_unflatten

from rich.console import Console
from rich.table import Table
import rich.text

def print_fwd_bwd(f, *args, **kwargs) -> None:
  args, in_tree = tree_flatten((args, kwargs))

  def f_(*args):
    args, kwargs = tree_unflatten(in_tree, args)
    return f(*args, **kwargs)

  fwd = jax.jit(lambda *args: jax.vjp(f_, *args)).trace(*args).jaxpr.jaxpr

  y, f_vjp = jax.vjp(f_, *args)
  res, in_tree = tree_flatten(f_vjp)

  def g_(*args):
    *res, y = args
    f_vjp = tree_unflatten(in_tree, res)
    return f_vjp(y)

  bwd = jax.jit(g_).trace(*res, y).jaxpr.jaxpr

  table = Table(show_header=False, show_lines=True, padding=(1, 2, 0, 2), box=None)
  table.add_row("[bold green]forward computation:",
                "[bold green]backward computation:")
  table.add_row(rich.text.Text.from_ansi(str(fwd)),
                rich.text.Text.from_ansi(str(bwd)))
  console = Console(width=240, force_jupyter=True)
  console.print(table)

def _renderable_repr(self):
  return self.html
rich.jupyter.JupyterRenderable._repr_html_ = _renderable_repr
```

```{code-cell}
# Without using `jax.checkpoint`:
print_fwd_bwd(f, W1, W2, W3, x)
```

```{code-cell}
# Using `jax.checkpoint` with a save-only-these-names policy:
print_fwd_bwd(f4, W1, W2, W3, x)
```

## Let's think step by step

### `jax.checkpoint` fundamentals

In both {func}`jax.linearize` and {func}`jax.vjp`, there is flexibility in
how and when some values are computed. Different choices can trade off memory
use against FLOPs. JAX provides control over these choices with
{func}`jax.checkpoint`.

One such choice is whether to perform Jacobian coefficient computations on
the forward pass, as soon as the inputs are available, or on the backward
pass, just before the coefficients are needed. Consider the example of
`sin_vjp`:

```{code-cell}
def sin_vjp(x):
  y = jnp.sin(x)
  cos_x = jnp.cos(x)
  return y, lambda y_bar: cos_x * y_bar
```

Another valid implementation would compute the value of `jnp.cos(x)` on the
backward pass rather than on the forward pass:

```{code-cell}
def sin_vjp2(x):
  y = jnp.sin(x)
  return y, lambda y_bar: jnp.cos(x) * y_bar
```

For this particular function, the amount of memory used by the two versions
is the same, though we've reduced the FLOPs for the primal computation (the
forward pass) and increased the FLOPs for the cotangent computation (the
backward pass).

There's another choice when it comes to function composition. Recall the VJP
rule for a composition of two functions:

```{code-cell}
def f(x):
  y = g(x)
  z = h(y)
  return z

def f_vjp(x):
  y, g_vjp = jax.vjp(g, x)
  z, h_vjp = jax.vjp(h, y)
  def f_bwd(z_bar):
    y_bar, = h_vjp(z_bar)
    x_bar, = g_vjp(y_bar)
    return x_bar
  return z, f_bwd
```

An alternative is:

```{code-cell}
def f_vjp_checkpoint(x):
  y = g(x)
  z, h_vjp = jax.vjp(h, y)
  def f_bwd2(z_bar):
    y_bar, = h_vjp(z_bar)
    _, g_vjp = jax.vjp(g, x)
    x_bar, = g_vjp(y_bar)
    return x_bar
  return z, f_bwd2
```

Using words, this alternative implementation doesn't compute `g_vjp`, or the
residual values in its closure, on the forward pass. Instead, it only
computes them in the backward pass `f_bwd2`. That means `f_vjp_checkpoint`
requires less memory: if `g` and `h` each required similar amounts of memory
for their residuals, each much larger than `x`, then the function produced by
`f_vjp_checkpoint(x)` requires half the memory as that of `f_vjp(x)`!

The cost you pay is redundant work: in `f_bwd2` you must re-evaluate `g(x)`
as part of `jax.vjp(g, x)` just to discard its value (in the underscore
variable on the line `_, g_vjp = jax.vjp(g, x)`).

You can get this VJP behavior in autodiff — without having to write VJP
functions directly — by instead using {func}`jax.checkpoint` in an
alternative definition of the original function `f`:

```{code-cell}
def f_checkpoint(x):
  y = jax.checkpoint(g)(x)
  z = h(y)
  return z
```

In other words, you apply {func}`jax.checkpoint` to `g` — the first stage of
`f` — rather than to `f` itself. This way, when you evaluate
`jax.grad(f_checkpoint)(x)`, you'd get a computation like:

1. Run the forward pass of `g`, discarding residual values.
2. Run the forward pass of `h`, saving residuals.
3. Run the backward pass of `h`, consuming residuals from step 2.
4. Re-run the forward pass of `g`, saving residuals.
5. Run the backward pass of `g`, consuming residuals from step 4.

That is, by evaluating `jax.grad(f_checkpoint)(x)` we'd get the same
computation as:

```{code-cell}
def f_checkpoint_grad(x):
  y = g(x)                  # step 1
  _, h_vjp = jax.vjp(h)(y)  # step 2
  y_bar, = h_vjp(1.0)       # step 3
  _, g_vjp = jax.vjp(g, x)  # step 4
  x_bar, = g_vjp(y_bar)     # step 5
  return x_bar
```

In general, `jax.checkpoint(foo)` is a new function which has the same
input-output behavior as `foo`, but behaves differently under autodiff,
particularly under {func}`jax.linearize` and {func}`jax.vjp` (and their
wrappers, like {func}`jax.grad`) but not {func}`jax.jvp`. When
differentiated, only the input to a {func}`jax.checkpoint`-differentiated
function is stored on the forward pass. On the backward pass, the residuals
(intermediates from `foo` and its Jacobian coefficient values needed for the
backward pass) are recomputed.

Notice that if `f = lambda x: h(g(x))` is the function you want to
differentiate (in other words, if you want to apply `jax.grad(f)`) you don't
get any memory savings by applying {func}`jax.checkpoint` to `f` itself.
That's because evaluating `jax.grad(jax.checkpoint(f))(x)` would lead to a
computation such as:

1. Run the forward pass, discarding all residuals.
2. Immediately re-run the forward pass, saving residuals.
3. Run the backward pass, consuming residuals from step 2.

That is, in code, something like:

```{code-cell}
def f_grad_bad(x):
  _ = f(x)                  # step 1
  _, f_vjp = jax.vjp(f, x)  # step 2
  x_bar, = f_vjp(1.0)       # step 3
  return x_bar
```

You also wouldn't get any memory savings by applying {func}`jax.checkpoint`
to `h`, the second stage of `f`. That's because evaluating
`jax.grad(lambda x: jax.checkpoint(h)(g(x)))` would lead to a computation
such as:

1. Run the forward pass of `g`, saving residuals.
2. Run the forward pass of `h`, discarding residuals.
3. Immediately re-run the forward pass of `h`, saving residuals.
4. Run the backward pass of `h`, consuming residuals from step 3.
5. Run the backward pass of `g`, consuming residuals from step 1.

In code, something like:

```{code-cell}
def f_grad_bad2(x):
  y, g_vjp = jax.vjp(g, x)  # step 1
  z = h(y)                  # step 2
  _, h_vjp = jax.vjp(h, y)  # step 3
  y_bar, = h_vjp(1.0)       # step 4
  x_bar, = g_vjp(y_bar)     # step 5
  return x_bar
```

Slightly more generally, if you had a chain composition of functions, such as
`f = lambda x: f3(f2(f1(x)))`, and were interested in evaluating
`jax.grad(f)`, you could say that you:

* Shouldn't apply {func}`jax.checkpoint` to the whole function `f`, since
  that wouldn't save any memory (and will perform wasteful recomputation).
* Shouldn't apply {func}`jax.checkpoint` to the last sub-function `f3`, since
  that wouldn't save any memory (and will perform wasteful recomputation).
* Could apply {func}`jax.checkpoint` to `f1`, `f2`, or their composition
  `lambda x: f2(f1(x))`, since any of those might save memory and would
  express different memory/recompute tradeoffs.

### Policies: naming what's saveable

As shown so far, using {func}`jax.checkpoint` switches from one extreme to
another:

* Without {func}`jax.checkpoint`, JAX's autodiff tends to compute everything
  possible on the forward pass and store it for the backward pass.
* With a {func}`jax.checkpoint` decorator, you instead compute as little as
  possible on the forward pass and recompute values as needed on the
  backward pass.

To operate between these two extremes, saving some things and not others,
you use the `policy` argument to {func}`jax.checkpoint`. A policy is *data*,
not code: it names which values are allowed to be saved as residuals, and
everything else is recomputed. The workflow has two halves:

1. **Name values** in the function being differentiated with
   {func}`jax.ad_checkpoint.checkpoint_name`. By itself, `checkpoint_name`
   is just an identity function — it only attaches a label.
2. **Select names** in the policy, with one of the constructors on
   {obj}`jax.checkpoint_policies`:

   * `save_only_these_names(*names)` — values with these names are saveable;
     everything else is recomputed;
   * `save_any_names_but_these(*names)` — every *named* value is saveable
     except these;
   * `save_and_offload_only_these_names(...)` — like `save_only_these_names`,
     but some names are offloaded to another memory space instead of kept
     (more below);
   * `everything_saveable` and `nothing_saveable` — the two extremes, as
     escape hatches (the former is the default behavior as if no policy were
     given; the latter is like no policy but full rematerialization).

For example, consider this function to be differentiated, with named layer
outputs:

```{code-cell}
def loss(params, x, y):
  return jnp.sum((predict(params, x) - y)**2)

def predict(params, x):
  *Ws, Wlast = params
  for i, W in enumerate(Ws):
    x = layer(W, x)
    x = checkpoint_name(x, name=f'layer{i}_output')
  x = jnp.dot(Wlast, x)
  return x

def layer(W, x):
  return jnp.sin(jnp.dot(W, x))

W1 = W2 = W3 = jnp.ones((4, 4))
params = [W1, W2, W3]
x = jnp.ones(4)
y = jnp.ones(4)
```

```{code-cell}
print_saved_residuals(loss, params, x, y)
```

```{code-cell}
loss_checkpoint = jax.checkpoint(
    loss, policy=jax.checkpoint_policies.save_any_names_but_these('layer1_output'))
print_saved_residuals(loss_checkpoint, params, x, y)
```

Notice that by providing a policy, you didn't need to change how `loss`,
`predict`, or `layer` are *called*: you can experiment with policies in
calling code (such as a training script) while the names live with the model
code. And policies only indicate what is *saveable*: a value is saved only if
it's actually needed by the backward pass.

```{note}
In older versions of JAX, a policy could also be an arbitrary *callable*
that inspected each primitive application and returned whether its outputs
were saveable (e.g. `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`).
Under the new implementation, policies are defunctionalized — they're the
name-based data described above — which keeps them simple, serializable, and
predictable. For cases where a function's remat behavior should depend on
more than a name, the function author can use `custom_remat`, described
below.
```

### Offloading instead of recomputing

Recomputation isn't the only alternative to keeping a residual in
accelerator memory: a residual can also be *offloaded* to another memory
space (typically host memory) on the forward pass and brought back when the
backward pass needs it, trading transfer bandwidth instead of FLOPs.

The policy `jax.checkpoint_policies.save_and_offload_only_these_names` takes
four arguments: `names_which_can_be_saved` (kept on device),
`names_which_can_be_offloaded` (moved to the destination memory space), and
the offloading source and destination. Values with other names, and unnamed
values, are recomputed.

```{code-cell}
from functools import partial

policy = jax.checkpoint_policies.save_and_offload_only_these_names(
    names_which_can_be_saved=['y'], names_which_can_be_offloaded=['z'],
    offload_src='device', offload_dst='pinned_host')

@partial(jax.checkpoint, policy=policy)
def f_offload(x):
  y = checkpoint_name(jnp.sin(x), 'y')   # saved on device
  z = checkpoint_name(jnp.sin(y), 'z')   # offloaded to pinned host memory
  w = checkpoint_name(jnp.sin(z), 'w')   # recomputed on the backward pass
  return jnp.sum(w)

print(jax.grad(f_offload)(jnp.arange(4.)))
print_saved_residuals(f_offload, jnp.arange(4.))
```

The `f32<host>[4]` residual is the offloaded value: it's kept, but in host
memory rather than device memory.

## Custom remat behavior with `custom_remat`

Name-based policies choose among values a function has named. Sometimes a
function *author* knows something better: a specific quantity that's worth
saving because it makes the backward pass cheap, or a way to restructure the
recomputation entirely. `custom_remat` lets a function carry its own
rematerialization behavior, including behavior that depends on the ambient
policy.

{func}`jax.custom_remat` — called as `custom_remat(f, f_fwd, f_rem, f_bwd)` — takes four functions:

* `f` is the primal function, used everywhere outside of rematerialized
  differentiation;
* `f_fwd(policy, *args) -> (out, res)` runs on the forward pass *inside a
  rematerialized region*: it receives the ambient checkpoint policy, and
  decides what residuals (if any) to keep;
* `f_rem(res, *args) -> (out, res2)` runs on the backward pass to
  rematerialize: given whatever `f_fwd` kept, plus the original arguments,
  it (re)computes the output and the residuals the backward rule needs;
* `f_bwd(res2, g) -> arg_cotangents` is the backward rule.

For example: the derivative of `sin` is `cos`, so a memory/FLOPs sweet spot
for `sin` under rematerialization can be to save the cosine — a value the
standard rules would recompute. Here's a `sin` that always saves its cosine
when rematerialized:

```{code-cell}
sin = jax.custom_remat(jnp.sin,
                   lambda _, x: (jnp.sin(x), jnp.cos(x)),   # keep cos(x)
                   lambda cos_x, x: (jnp.sin(x), cos_x),    # rematerialize
                   lambda cos_x, g: (cos_x * g,))           # backward rule

f = jax.remat(lambda x: sin(sin(x)))
print(jax.grad(f)(1.0))
print(jax.grad(jnp.sin)(jnp.sin(1.0)) * jnp.cos(1.0))  # chain rule, for reference
```

Because `f_fwd` receives the policy, the behavior can also *respond* to it.
Here's a `sin` that saves its cosine only when the ambient policy declares
the name `'cos'` saveable, and otherwise defers to full recomputation:

```{code-cell}
SaveOnlyTheseNames = jax.checkpoint_policies.SaveOnlyTheseNames

def sin_fwd(policy, x):
  if isinstance(policy, SaveOnlyTheseNames) and 'cos' in policy.saveable_names:
    return jnp.sin(x), jnp.cos(x)
  else:
    return jnp.sin(x), None

def sin_rem(cos_x, x):
  if cos_x is None:
    cos_x = jnp.cos(x)
  return jnp.sin(x), cos_x

def sin_bwd(cos_x, g):
  return cos_x * g,

sin = jax.custom_remat(jnp.sin, sin_fwd, sin_rem, sin_bwd)

save_cos = jax.checkpoint_policies.save_only_these_names('cos')
f = jax.checkpoint(lambda x: sin(sin(x)), policy=save_cos)
print(jax.grad(f)(3.0))

f = jax.checkpoint(lambda x: sin(sin(x)),
                   policy=jax.checkpoint_policies.nothing_saveable)
print(jax.grad(f)(3.0))
```

`custom_remat` currently supports reverse-mode differentiation of the
rematerialized function (which is where rematerialization matters).

## Advanced: recursive `jax.checkpoint`

By applying {func}`jax.checkpoint` in the right way, there are many tradeoffs
between memory usage and (re)computation that can be expressed. One
surprising example is _recursive_ checkpointing, where you apply
{func}`jax.checkpoint` to a function which itself calls
{func}`jax.checkpoint`-decorated functions in a way so that memory usage from
the chain composition of $D$ functions scales like $\mathcal{O}(\log_2 D)$
rather than $\mathcal{O}(D)$.

As a toy example, consider the chain composition of multiple
{func}`jax.numpy.sin` functions:

```{code-cell}
def chain_compose(funs):
  def f(x):
    for fun in funs:
      x = fun(x)
    return x
  return f

f = chain_compose([jnp.sin] * 8)
print_saved_residuals(f, 3.)
```

In general, the number of stored residuals scales linearly with the length of
the chain:

```{code-cell}
f = chain_compose([jnp.sin] * 16)
print_saved_residuals(f, 3.)
```

But you can apply {func}`jax.checkpoint` recursively to improve the scaling:

```{code-cell}
def recursive_checkpoint(funs):
  if len(funs) == 1:
    return funs[0]
  elif len(funs) == 2:
    f1, f2 = funs
    return lambda x: f1(f2(x))
  else:
    f1 = recursive_checkpoint(funs[:len(funs)//2])
    f2 = recursive_checkpoint(funs[len(funs)//2:])
    return lambda x: f1(jax.checkpoint(f2)(x))
```

```{code-cell}
f = recursive_checkpoint([jnp.sin] * 8)
print_saved_residuals(f, 3.)
```

```{code-cell}
f = recursive_checkpoint([jnp.sin] * 16)
print_saved_residuals(f, 3.)
```

The cost here, as usual, is recomputation: in particular, you end up
performing $\mathcal{O}(\log_2 D)$ times as many FLOPs:

```{code-cell}
f = chain_compose([jnp.sin] * 8)
print_fwd_bwd(f, 3.)
```

```{code-cell}
f = recursive_checkpoint([jnp.sin] * 8)
print_fwd_bwd(f, 3.)
```

## Practical notes

When differentiated functions are staged out to XLA for compilation — for
example by applying {func}`jax.jit` to a function which contains a
{func}`jax.grad` call — XLA will automatically optimize the computation,
including decisions about when to compute or rematerialize values. As a
result, **{func}`jax.checkpoint` often isn't needed for differentiated
functions under a {func}`jax.jit`**. XLA will optimize things for you.

One exception is when using staged-out control flow, like
{func}`jax.lax.scan`. Automatic compiler optimizations across multiple
control flow primitives (for example, across a forward-pass `scan` and the
corresponding backward-pass `scan`), typically aren't as thorough. As a
result, it's often a good idea to use {func}`jax.checkpoint` on the body
function passed to {func}`jax.lax.scan`.

For example, one common pattern in large
[Transformer models](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))
is to express the architecture as a {func}`jax.lax.scan` over layers so as to
reduce compilation times. That is, using a simple fully-connected network as
an analogy, instead of writing something like this:

```{code-cell}
LayerParam = tuple[jnp.ndarray, jnp.ndarray]  # Weights-bias pair for a layer.
ParamsList = list[LayerParam]

def net(params: ParamsList, x: jnp.ndarray):
  for W, b in params:
    x = jnp.maximum(jnp.dot(x, W) + b, 0.)
  return x
```

Instead, iterate over the layer application with {func}`jax.lax.scan`:

```{code-cell}
params = [(jnp.array([[0.5, 0.5], [1., 1.]]), jnp.array([0.5, 0.5])),
          (jnp.array([[0.5, 0.5], [1., 1.]]), jnp.array([0.5, 0.5]))]

all_weights = jnp.stack([W for W, _ in params])
all_biases = jnp.stack([b for _, b in params])

def layer(x, W_b_pair):
  W, b = W_b_pair
  out = jnp.maximum(jnp.dot(x, W) + b, 0.)
  return out, None

def net(all_weights, all_biases, x):
  x, _ = jax.lax.scan(layer, x, (all_weights, all_biases))
  return x
```

This scan-over-layers version reduces compile times, but by foiling some
compiler optimizations it can lead to inefficient computation of gradients.
To mitigate the issue, you can use {func}`jax.checkpoint` on the scanned
function — either plain, or with a names-based policy to keep the residuals
you know are worth their memory:

```{code-cell}
@partial(jax.checkpoint,
         policy=jax.checkpoint_policies.save_only_these_names('preactivation'))
def layer(x, W_b_pair):
  W, b = W_b_pair
  pre = checkpoint_name(jnp.dot(x, W) + b, 'preactivation')
  out = jnp.maximum(pre, 0.)
  return out, None
```

By using {func}`jax.checkpoint` this way, you're manually controlling which
values JAX's autodiff saves between the forward and backward passes, and
therefore not relying on XLA optimizations to choose for you.
