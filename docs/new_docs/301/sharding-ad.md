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

(jax-301-sharding-ad)=
# Autodiff and explicit sharding

<!--* freshness: { reviewed: '2026-07-10' } *-->

In explicit sharding mode ({ref}`jax-201-sharding`), shardings are part of
JAX types: `jax.typeof(x)` might print `float32[8@X,4]`, meaning the leading
axis is sharded along mesh axis `X`. This page is about what happens when you
differentiate such a program. The main idea:

**You control how your backward pass is sharded, through simple, local
reasoning about your forward pass.**

The rule that delivers this is that *cotangent types are a function of primal
types*: the type of each gradient value in the backward pass — shape, dtype,
and sharding — is determined by the type of the corresponding value in the
forward pass. Once you know your forward-pass types, you know your
backward-pass types, and hence where backward-pass communication happens.

There's a second theme running through this page. If every axis of every
array were plain *sharded*, autodiff would need nothing new: sharded
cotangents for sharded primals, no communication anywhere. It's
*replication* — wanting a full copy of an array on several devices — that
makes things interesting. Replication's dual under transposition is
*reduction*, i.e. a cross-device sum, and someone has to decide where that
sum happens. Demanding efficient autodiff plus local reasoning in the
presence of replication is what leads to the two new sharding states
introduced below, `unreduced` and `reduced`.

We'll use two CPU devices and a one-axis mesh throughout:

```{code-cell}
import jax
import jax.numpy as jnp
jax.config.update('jax_num_cpu_devices', 2)

jax.set_mesh(jax.make_mesh((2,), ('X',)))  # explicit mode by default
```

## Cotangent shardings are a function of primal shardings

Here's a data-parallel loss: the batch `x` is sharded along `X`, and the
weights `w` are replicated (a full copy on every device):

```{code-cell}
x = jax.device_put(jnp.arange(8 * 4.).reshape(8, 4), jax.P('X', None))
w = jax.device_put(jnp.arange(4 * 2.).reshape(4, 2) / 10., jax.P(None, None))

def loss(w, x):
  return jnp.sum((x @ w) ** 2)

dw, dx = jax.grad(loss, argnums=(0, 1))(w, x)
print(jax.typeof(w), '->', jax.typeof(dw))
print(jax.typeof(x), '->', jax.typeof(dx))
```

The gradient with respect to a sharded input is sharded the same way, and the
gradient with respect to a replicated input is replicated. This isn't just
true of the top-level inputs and outputs of `jax.grad`; it holds for every
intermediate in the backward pass. We can see it in the jaxpr, where each
cotangent value's sharding matches its primal's:

```{code-cell}
print(jax.jit(jax.grad(loss)).trace(w, x).jaxpr)
```

The final two equations compute `dw`: a `dot_general` of two values sharded
along `@X` producing a replicated (unsharded) result, then a transpose. Hold
that thought — we'll come back to what that dot costs.

There are two reasons JAX insists that cotangent shardings are determined by
primal shardings:

1. **User control.** The goal of explicit mode is that user-written code
   determines all the shardings in the computation, in an easy-to-predict,
   local way. The backward pass is part of the computation — but you don't
   write it, autodiff does. Making cotangent shardings a function of primal
   shardings means your forward-pass sharding decisions *are* your
   backward-pass sharding decisions. (Compiler-based automatic sharding mode
   has no analogous guarantee: there, backward-pass shardings can be chosen
   by the compiler, unrelated to the corresponding primal shardings.)

2. **Ruling out ambiguities.** If a variable is used more than once in the
   forward pass (fan-out), autodiff generates an addition of cotangents in
   the backward pass. If cotangent shardings could be unrelated to primal
   shardings, the two summands might have *different* shardings, and the
   addition would require communication that nothing in your code specifies.
   (Similarly, the zero cotangents autodiff generates would have no
   determined sharding.) When cotangent shardings are a function of primal
   shardings, both summands automatically agree: they're cotangents for the
   same primal variable, so they have the same type.

This is a special case of a general principle in JAX's autodiff: cotangent
types are always a function of the corresponding primal types — shapes,
dtypes, and now shardings. That gives us three things: each op's backward
rule has a clear type and knows exactly what kind of cotangent it will
receive; a well-typed forward program always yields a well-typed backward
program; and you can predict backward-pass types by looking only at your
forward-pass code, without knowing what the rest of the program looks like.

## Where backward-pass communication comes from

In the example above, look at the types in the backward pass of `x @ w`. The
cotangent `dw` must be replicated, because `w` is. But the data it's built
from is spread across devices: `dw` is (the transpose of) a contraction of
two arrays sharded along `@X`, the batch axis. Producing a replicated result
from a contraction over a sharded axis requires a cross-device sum — an
AllReduce. The partitioner inserts it inside that final `dot_general`.

This is the familiar gradient synchronization of data-parallel training, and
notice that you can predict it purely locally: `w` is replicated, each device
touches only part of the batch, so somewhere in the backward pass the
per-device gradient contributions must be summed. The types tell you the
communication exists — but in the jaxpr above it's *implicit*, hidden inside
an op whose operands have a sharded contracting dimension. The rest of this
page is about making that communication explicit: something you can see in
the types, move around, and batch up.

## Unreduced: a reduction waiting to happen

Consider a matmul whose *contracting* dimension is sharded:

```{code-cell}
a = jax.device_put(jnp.arange(4.).reshape(2, 2), jax.P(None, 'X'))
b = jax.device_put(jnp.arange(4., 8.).reshape(2, 2), jax.P('X', None))
print(jax.typeof(a))
print(jax.typeof(b))
```

```{code-cell}
:tags: [raises-exception]

a @ b
```

JAX makes us say what we want here, because there's a genuine choice. Look at
the data each device holds:

```text
     a: P(None, 'X')       b: P('X', None)
  (device k has column k)  (device k has row k)

        [ 0 | 1 ]             [ 4  5 ]
                      @       ---------
        [ 2 | 3 ]             [ 6  7 ]
```

Device 0 can multiply its column of `a` by its row of `b` without any
communication, and likewise device 1. Each local matmul produces a
full-shape *partial sum*, and the true answer is the elementwise sum of the
two. One option is to finish the job with an AllReduce, by asking for an
ordinary output sharding:

```{code-cell}
c = jnp.einsum('ij,jk->ik', a, b, out_sharding=jax.P(None, None))
print(jax.typeof(c))
print(c)
```

The other option is to *stop before the reduction*:

```{code-cell}
c = jnp.einsum('ij,jk->ik', a, b,
               out_sharding=jax.P(None, None, unreduced={'X'}))
print(jax.typeof(c))
```

The type `float32[2,2]{U:X}` reads: a 2×2 array, *unreduced* along mesh axis
`X`. Each device along `X` holds a full-shape partial sum, and the array's
true value is the sum of those pieces:

```{code-cell}
for shard in c.addressable_shards:
  print(f'device {shard.device.id}:\n{shard.data}')
```

An unreduced array is a reduction waiting to happen. To cash it in, reshard
to an ordinary sharding — this is where the deferred AllReduce runs:

```{code-cell}
print(jax.reshard(c, jax.P(None, None)))
```

Why defer? Because you might want to do more work first, and pay for fewer,
bigger collectives. The catch is that only *linear* operations make sense on
unreduced arrays. Adding two arrays unreduced along the same axes is fine —
sums of partial sums are partial sums of sums:

```{code-cell}
c2 = jnp.einsum('ij,jk->ik', 2. * a, b,
                out_sharding=jax.P(None, None, unreduced={'X'}))
print(jax.typeof(c + c2))
```

That lets you compute a sum of sharded matmuls with a single AllReduce at the
end — think of a LoRA-style `x @ W + x @ A @ B` — rather than one per
matmul. Nonlinear operations, on the other hand, *can't* have a rule for
unreduced inputs: the cosine of a sum is not the sum of the cosines, so
applying `cos` to each device's partial sum would just compute the wrong
answer. Such ops raise an error instead:

```{code-cell}
:tags: [raises-exception]

jnp.cos(c)
```

One honest caveat: for straight-line code like `c + c2` above, XLA can often
merge adjacent AllReduces on its own, so the compiled program may be equally
good either way. The type-level guarantee matters when the compiler can't
find the merge — above all, across the iterations of a loop. We'll see
exactly that in the capstone example below.

## Reduced: choosing unreduced gradients

Now back to autodiff. We said cotangent shardings are a function of primal
shardings, and we've seen two entries of that function: sharded primals get
sharded cotangents, replicated primals get replicated cotangents.

The replicated entry is exactly where backward-pass communication comes
from: replication's transpose is a cross-device sum. With
`CT(Replicated) = Replicated`, autodiff performs that sum *eagerly* — each
backward-pass op that produces a cotangent for a replicated primal does its
AllReduce on the spot, implicitly, like the `dw` dot we saw earlier. The
natural, before-any-communication state of such a cotangent is a bunch of
per-device partial sums — an *unreduced* array! So what if we want autodiff
to leave it that way, and let us decide when to reduce?

We need a way to ask for that in the forward pass, and it has to be a new
*type* — remember, cotangent types are a function of primal types. That's
what `reduced` is for:

```{code-cell}
w_ = jax.reshard(w, jax.P(None, None, reduced={'X'}))
print(jax.typeof(w_))
```

An array that's *reduced* along `X`, written `{R:X}`, is physically identical
to a replicated array: a full copy on every device along `X`. (The name is
the past tense of "unreduced": it's the state an array is in after its
reduction has happened.) In the forward pass it behaves exactly like a
replicated array, and the reshard above is communication-free — no data
moves. The *only* difference is how autodiff treats it. The complete
cotangent map is:

| primal type       | cotangent type    |
|-------------------|-------------------|
| sharded `@X`      | sharded `@X`      |
| replicated        | replicated        |
| reduced `{R:X}`   | unreduced `{U:X}` |
| unreduced `{U:X}` | reduced `{R:X}`   |

Use `Replicated` and you get replicated gradients; use `Reduced` and you get
unreduced gradients. That's the only difference between them.

Here's the payoff. The reshard-to-reduced above is a communication-free cast
in the forward pass, and under transposition it becomes a
reshard-from-unreduced in the backward pass — which is precisely the
AllReduce. **A free cast in your forward code pins down where the collective
runs in your backward code.** Backward-pass communication becomes something
you can see, place, and reason about locally, while writing the forward
pass. Compare the backward pass of our data-parallel loss with and without
the cast:

```{code-cell}
def loss2(w, x):
  w = jax.reshard(w, jax.P(None, None, reduced={'X'}))
  return jnp.sum((x @ w) ** 2)

print(jax.jit(jax.grad(loss2)).trace(w, x).jaxpr)
```

Where the original jaxpr had a `dot_general` with an implicit AllReduce
buried inside, this one shows the dot producing an explicit
`f32[2,4]{U:X}` value, and a final `reshard` — the transposed cast — turning
it into the replicated `dw`. Same math, same total communication, but now the
reduction is a visible, movable object in the program.

And once it's visible, you can move it. Suppose the weights are used twice —
two heads, two microbatches, a LoRA branch:

```{code-cell}
def loss_fanout(w, x1, x2):
  w = jax.reshard(w, jax.P(None, None, reduced={'X'}))
  return jnp.sum(x1 @ w) + jnp.sum(x2 @ w)

x1 = jax.device_put(jnp.ones((8, 4)), jax.P('X', None))
x2 = jax.device_put(jnp.ones((8, 4)), jax.P('X', None))
print(jax.jit(jax.grad(loss_fanout)).trace(w, x1, x2).jaxpr)
```

Both backward dots produce `{U:X}` contributions, the fan-out addition
happens *unreduced* (addition is linear!), and a single `reshard` performs
one AllReduce for the whole gradient. Without the cast, each contribution
would be reduced separately inside its own dot. Here the fusion is guaranteed
by the types, not left to compiler pattern-matching — which is about to
matter.

## Capstone: microbatch gradient accumulation

The classic place this bites is gradient accumulation. A realistic training
step has *two* loops that the compiler can't see through: a scan over layers
inside the model, and a scan over microbatches accumulating gradients, with
one update at the end. The gradient AllReduce should happen once per step —
but if the weights are replicated, every microbatch's backward pass
synchronizes its own gradient contribution, inside both loops, and XLA
cannot hoist collectives out of a loop for you. With `reduced` weights, the
gradients come out unreduced, the accumulator stays unreduced across the
whole scan, and you reduce once:

```{code-cell}
def predict(stacked_ws, xs):  # stacked_ws: [layer, features, features]
  def apply_layer(xs, w):
    return jnp.tanh(xs @ w), None
  final_xs, _ = jax.lax.scan(apply_layer, xs, stacked_ws)
  return final_xs

def loss3(stacked_ws, batch):
  return jnp.sum(predict(stacked_ws, batch) ** 2)

@jax.jit
def step(stacked_ws, xs):  # xs: [microbatch, batch@X, features]
  def microbatch_step(grad_acc, xs_mb):
    grads = jax.grad(loss3)(stacked_ws, xs_mb)
    # ws are reduced, so grads are unreduced -- and we can check it!
    assert jax.typeof(grads).sharding.spec.unreduced == {'X'}
    return grad_acc + grads, None

  grad_acc = jax.reshard(jnp.zeros_like(stacked_ws), jax.P(unreduced={'X'}))
  grad_acc, _ = jax.lax.scan(microbatch_step, grad_acc, xs)
  grads = jax.reshard(grad_acc, jax.P())        # the one AllReduce
  ws = jax.reshard(stacked_ws, jax.P())         # free: full copies already
  return ws - 0.01 * grads

stacked_ws = jax.device_put(jnp.stack([jnp.eye(4) / 2] * 3),
                            jax.P(reduced={'X'}))
xs = jax.device_put(jnp.ones((5, 2, 4)), jax.P(None, 'X', None))

new_ws = step(stacked_ws, xs)
print(jax.typeof(new_ws))
```

Everything type-checks locally: the weights are `{R:X}`, so each microbatch's
gradients come out `{U:X}` — even though they're computed by a scan over
layers; unreduced arrays support addition, so the microbatch scan carry
accumulates them; and one reshard to replicated is the step's single
AllReduce. Notice the `assert` in the scan body: because shardings are part
of JAX types, "the gradients are unreduced" isn't a hope or a comment — it's
a property of a value you can check with `jax.typeof`, inside traced code,
at trace time. The updated weights come back replicated; casting them back to
`{R:X}` for the next step is free. And the compiled program keeps the
promise — exactly one AllReduce, outside both loops:

```{code-cell}
import re

def print_all_reduces(jitted, *args):
  hlo = jitted.lower(*args).compile().as_text()
  for line in hlo.splitlines():
    if 'all-reduce(' in line or 'all-reduce-start(' in line:
      print(re.search(r'op_name="([^"]*)"', line).group(1))

print_all_reduces(step, stacked_ws, xs)
```

Compare the same step written with plain replicated weights:

```{code-cell}
@jax.jit
def step_replicated(stacked_ws, xs):
  def microbatch_step(grad_acc, xs_mb):
    grads = jax.grad(loss3)(stacked_ws, xs_mb)  # replicated: AllReduce inside!
    return grad_acc + grads, None
  grad_acc = jnp.zeros_like(stacked_ws)
  grad_acc, _ = jax.lax.scan(microbatch_step, grad_acc, xs)
  return stacked_ws - 0.01 * grad_acc

ws_replicated = jax.reshard(stacked_ws, jax.P())
print_all_reduces(step_replicated, ws_replicated, xs)
```

The op name says it all: `while/body/.../while/body` — this AllReduce sits
inside the transposed layer scan, inside the microbatch scan: it runs once
per layer per microbatch. Hoisting the gradient reduction out of loops this
way — from once per layer per microbatch down to once per step — has produced
large wins in production LLM training — in one case cutting per-step time
spent in gradient reduction by several times — and it's a transformation
the compiler cannot make on its own, because it can't pattern-match
collectives through a loop.

## Why this design?

A few frequently-asked design questions, for the curious.

**Why not make the cotangent of Replicated be Unreduced?** It would be
coherent — but it takes options away. Lots of code returns replicated values
(most obviously, loss values!), and making their cotangents unreduced would
introduce surprising communication requirements into existing programs. By
keeping `Replicated ↔ Replicated` and adding `Reduced ↔ Unreduced` as a
separate pair, you get to choose per-array whether gradients arrive
replicated (reduction done for you, eagerly) or unreduced (reduction is
yours to place), and the choice is made by a communication-free cast in the
forward pass.

**Is there a theoretical justification for Unreduced, or is it just a
practical trick?** Here's an argument that something like it is forced.
Suppose we want the following, none of which is individually exotic:

1. autodiff should preserve communication cost, op by op: a forward op that
   requires no communication should transpose to a backward op that requires
   no communication (or at least, we need *some* way to write such a
   backward pass);
2. replication should be expressible — we don't want *only* sharded axes;
3. multiplying a replicated scalar by a sharded vector, a communication-free
   forward op, should be expressible;
4. cotangents should have the same shape as their primals.

Transposing (3) with respect to the scalar means mapping a sharded cotangent
vector to a cotangent for the replicated scalar. Doing that with no
communication (1) forces a state that is scalar-shaped (4) but holds only a
per-device *piece* of the answer, pending a sum: that's Unreduced. (Without
premise 4 you could simulate it by tacking on an extra device-indexed axis;
Unreduced is in effect that axis, tracked in the sharding instead of the
shape.)

This page covered explicit sharding mode. In manual mode
({ref}`jax-201-shard-map`), where you write per-device code with explicit
collectives like `psum`, the same questions get answered differently — how
forward-pass collectives transpose has its own story, planned for a future
page.
