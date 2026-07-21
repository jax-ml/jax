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

(jax-101-state)=
# Stateful computations

<!--* freshness: { reviewed: '2026-07-09' } *-->

Real programs have state: model parameters that change across training steps,
optimizer momentum, running statistics, counters. But
{ref}`jax-101-transformations` established that JAX transformations expect
*pure* functions — everything in through arguments, everything out through
return values, no hidden updates on the side.

This page covers the two ways to express stateful computation in JAX:

1. **Threading state through pure functions** — state goes in as an argument
   and comes out as a return value. This is the classic pattern, and the way
   state is handled across nearly the entire JAX ecosystem.
2. **Refs** — explicit mutable arrays, for when you really do want to write to
   memory in place.

## The problem with hidden state

Let's start with a simple stateful program: a counter.

```{code-cell}
import jax
import jax.numpy as jnp

class Counter:
  """A simple counter."""

  def __init__(self):
    self.n = 0

  def count(self) -> int:
    """Increments the counter and returns the new value."""
    self.n += 1
    return self.n

counter = Counter()

for _ in range(3):
  print(counter.count())
```

The `n` attribute maintains the counter's *state* between calls, updated as a
side effect. As plain Python this works fine — but `count` is impure, and it
falls apart under transformation. We can see exactly how by looking at what
tracing records, using the jaxpr-inspection idiom from {ref}`jax-101-tracing`:

```{code-cell}
counter = Counter()
jax.jit(counter.count).trace().jaxpr
```

The recorded program takes no input, performs no operations, and always
returns 1. The side effect `self.n += 1` ran once, at trace time; the *value*
`self.n` produced was captured as a constant; and the state update itself is
nowhere in the jaxpr. Any transformation working from this recording is
working with a function that returns 1, forever. (Under `jax.jit`, which
caches traces, that's precisely what you'd observe: the compiled counter
returns 1 on every call.)

## Threading state through pure functions

The fix is to make the state part of the function's interface: take the
current state as an argument, and return the updated state alongside the
output.

```{code-cell}
CounterState = int

class CounterV2:

  def count(self, n: CounterState) -> tuple[int, CounterState]:
    # You could just return n+1 once, but here we separate its role as
    # the output and as the counter state for didactic purposes.
    return n + 1, n + 1

  def reset(self) -> CounterState:
    return 0

counter = CounterV2()
state = counter.reset()

for _ in range(3):
  value, state = counter.count(state)
  print(value)
```

The caller now keeps track of the state explicitly. In exchange, `count` is
pure — and tracing it tells a completely different story:

```{code-cell}
jax.jit(counter.count).trace(0).jaxpr
```

The state flows visibly through the recorded program: in as the argument, out
as a result. Nothing is baked in, nothing is hidden, and every transformation
handles this function correctly, because there's nothing left to mishandle.

### The general recipe

What we did to the counter works for any stateful computation. Take a class of
the form

```python
class StatefulClass:

  state: State

  def stateful_method(*args, **kwargs) -> Output:
```

and turn it into functions of the form

```python
def stateless_method(state: State, *args, **kwargs) -> tuple[Output, State]:
```

This is a common [functional programming](https://en.wikipedia.org/wiki/Functional_programming)
pattern, and it's how state is expressed in nearly all JAX programs. Training
loops are its signature application: the parameters, and any optimizer state,
thread through a pure `update` function step after step:

```{code-cell}
def update(params, opt_state, batch):
  """One training step: consume state, produce new state."""
  grads = jax.grad(loss_fn)(params, batch)
  new_params, new_opt_state = optimizer_step(params, grads, opt_state)
  return new_params, new_opt_state

def loss_fn(params, batch):
  x, y = batch
  return jnp.mean((params['w'] * x + params['b'] - y) ** 2)

def optimizer_step(params, grads, opt_state, lr=0.1, decay=0.9):
  # gradient descent with momentum: the momentum is optimizer state
  new_opt_state = jax.tree.map(
      lambda m, g: decay * m + g, opt_state, grads)
  new_params = jax.tree.map(
      lambda p, m: p - lr * m, params, new_opt_state)
  return new_params, new_opt_state

params = {'w': jnp.float32(1.0), 'b': jnp.float32(0.0)}
opt_state = jax.tree.map(jnp.zeros_like, params)
batch = (jnp.array([1.0, 2.0, 3.0]), jnp.array([3.0, 5.0, 7.0]))

for step in range(100):
  params, opt_state = update(params, opt_state, batch)

print(jax.tree.map(lambda x: round(float(x), 2), params))  # fits y = 2x + 1
```

Notice that the state here is a *pytree* — a dict of parameters and a matching
dict of momenta — so the pattern scales from a single counter to an entire
model without changing shape (see {ref}`jax-101-pytrees`). This is also the
convention you'll meet everywhere in the JAX ecosystem: optimizer libraries
like [Optax](https://optax.readthedocs.io/) are built around
`update(grads, opt_state, ...) -> (updates, new_opt_state)`, and neural
network libraries handle parameters the same way.

Threading state as values has a deeper payoff, too: because each state is an
immutable snapshot, transformations apply cleanly to the whole loop — you can
differentiate through an update step, or `vmap` it to run many independent
training runs at once, without worrying about aliased mutations.

(jax-101-refs)=
## Refs: mutable arrays

Threading is the workhorse, but it can be heavyweight. If a function deep in
your call stack wants to record a metric, every function along the way must
plumb that state in and out of its signature.

For cases like this, JAX has **refs**: references to mutable memory, created with
{func}`jax.new_ref <jax.ref.new_ref>` — a distinct type from `Array`, with its
own rules for how it interacts with transformations. Refs can be read and
written in place with NumPy-style indexing. Here's our counter one more time,
with its state in a ref, mutated for real:

```{code-cell}
counter_ref = jax.new_ref(0)

def count():
  counter_ref[...] += 1
  return counter_ref[...]

print(count(), count(), count())
```

For a ref `x_ref`, `x_ref[...]` reads the whole current value as an `Array`,
`x_ref[...] = v` writes one, and any NumPy-style index works in place of
`...`:

```{code-cell}
x_ref = jax.new_ref(jnp.arange(6.0).reshape(2, 3))

x_ref[0, 0] = 100.0     # indexed write
x_ref[1] += 10.0        # indexed add-update
print(x_ref[...])       # read the full value
print(x_ref[:, 1])      # read a column
```

Indexed reads and writes are essentially the *only* things you can do with a
ref. In particular, you can't do math directly on one — read it first:

```{code-cell}
:tags: [raises-exception]

jnp.sin(x_ref)  # error! read the value out first: jnp.sin(x_ref[...])
```

When you're done mutating, {func}`jax.freeze <jax.ref.freeze>` invalidates the
ref and returns its final value as an ordinary immutable array:

```{code-cell}
final = jax.freeze(x_ref)
final
```

```{note}
Refs are a relatively new JAX feature. The core API shown here is expected to
be stable, but some corners (especially around autodiff) are still evolving.
```

## Refs and purity

How do refs square with the "pure functions only" rule? By a simple
accounting: a function is **impure** if refs cross its boundary — taken as
arguments, or captured from an enclosing scope, like `count` above. A function
that creates and uses refs purely *internally* is still pure — purity is in
the eye of the caller:

```{code-cell}
def normalize(x):        # pure: refs used internally only
  acc = jax.new_ref(0.0)
  acc[...] = jnp.sum(x)  # (a real program would do something less trivial)
  return x / acc[...]

jax.grad(lambda x: normalize(x).sum())(jnp.arange(1.0, 4.0))
```

Pure functions that use refs internally work with all transformations in the
usual way, as the `jax.grad` call above shows. Impure functions — those that
take refs as arguments — are more constrained, but still work with many
transformations. For example, you can `vmap` an impure function over a batch
of ref entries:

```{code-cell}
def scale_into(x, out_ref):   # impure: takes a ref argument
  out_ref[...] = 2.0 * x

xs = jnp.arange(3.0)
out_ref = jax.new_ref(jnp.zeros(3))
jax.vmap(scale_into)(xs, out_ref)   # each instance writes its own entry
print(out_ref[...])
```

What you *can't* do is `vmap` a function that closes over a ref — with every
batch member writing to the same shared location, the final value would be
ambiguous:

```{code-cell}
:tags: [raises-exception]

r = jax.new_ref(0.0)

def write_shared(x):
  r[...] = x    # every batch member writes the same ref!

jax.vmap(write_shared)(jnp.arange(3.0))
```

## Restrictions

Refs come with rules designed to rule out *aliasing* — two refs unknowingly
pointing at the same memory — and other situations where the meaning of a
program would become unclear:

- You can't return a ref from a `jit`-compiled function or from the body of a
  control-flow operation like `jax.lax.scan`.
- You can't pass the same ref twice to a `jit`-compiled function, nor pass a
  ref that the function also captures from an enclosing scope.
- You can only `freeze` a ref in the scope where it was created.
- No refs-to-refs.
- As above: no `vmap` (or `shard_map`) over functions that close over refs.

If you hit one of these, the error message will say so directly. Some of these
restrictions may be lifted over time.

## Performance, and what's next for refs

Refs aren't just for expressiveness — they give you control over memory. Under
`jax.jit`, writing through a ref updates its buffer in place rather than
allocating a fresh array, and passing refs into a jitted function avoids
copies without any donation ceremony. The performance story, like the rest of
the `jit` story, is covered in the performance and scaling docs.

Refs also interact with automatic differentiation: you can plumb values out of
backward passes, accumulate gradients in place across microbatches, and
differentiate with respect to ref arguments. That material lives with the
advanced autodiff docs; see {doc}`/array_refs` for the full treatment in the
meantime.

## Where you've arrived

This completes the expressiveness tour: arrays and `jax.numpy` as the
vocabulary, `grad` and `vmap` as the verbs, pytrees for structure, keys for
randomness, and — for state — threaded values or refs.

What you can't do yet is make it *fast* — that's a matter of `jax.jit`,
sharding, and profiling, and it's exactly where the performance and scaling
docs pick up: {ref}`jax-201-jit`.
