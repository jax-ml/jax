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
optimizer momentum, running statistics, counters. This page covers the two ways
to express stateful computation in JAX:

1. **Threading state through pure functions** — state goes in as an argument
   and comes out as a return value. This is the classic pattern, and the way
   state is handled across nearly the entire JAX ecosystem.
2. **Refs** — explicit mutable arrays, for when plumbing is a pain, or you
   really do want to write to memory in place.

## The problem with untraceable state

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
side effect. As plain Python this works fine, but it falls apart under JAX
transformations. We can see exactly how by looking at what tracing records,
using the jaxpr-inspection idiom from {ref}`jax-101-tracing`:

```{code-cell}
counter = Counter()
jax.jit(counter.count).trace().jaxpr
```

The recorded program takes no input, performs no operations, and always
returns 1. The side effect `self.n += 1` ran once, at trace time; the *value*
`self.n` produced was captured as a constant; and the state update itself is
nowhere in the jaxpr. Any transformation working from this traced view is
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
pure and hence easy to trace, and tracing it tells a completely different story:

```{code-cell}
jax.jit(counter.count).trace(0).jaxpr
```

The state flows visibly through the recorded program: in as the argument, out
as a result. Nothing is baked in, nothing is hidden, and every transformation
handles this function correctly, because there's nothing left to mishandle.

What we did to the counter works for any stateful computation. Take a class of
the form

```python
class StatefulClass:

  state: State

  def stateful_method(*args, **kwargs) -> Output:
    ...
```

and turn it into functions of the form

```python
def stateless_method(state: State, *args, **kwargs) -> tuple[Output, State]:
  ...
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

Notice that the state here is a *pytree*, so the pattern scales from a single
counter to an entire model without changing shape (see {ref}`jax-101-pytrees`).
This is also the convention you'll meet everywhere in the JAX ecosystem:
optimizer libraries like [Optax](https://optax.readthedocs.io/) are built
around `update(grads, opt_state, ...) -> (updates, new_opt_state)`, and neural
network libraries handle parameters the same way.

Threading state as values has a deeper payoff, too: because each state is an
immutable snapshot, transformations apply cleanly to the whole loop. You can
differentiate through an update step, or `vmap` it to run many independent
training runs at once, without worrying about aliased mutations.

(jax-101-refs)=
## Refs: mutable arrays

Threading values is the workhorse approach, but it can be awkward. If a
function deep in your call stack wants to update normalization statistics or
record a metric, every function along the way must plumb that state in and out
of its signature.

For cases like this, JAX has **refs**: mutable arrays that can be read and
written in place. Create one with {func}`jax.new_ref <jax.ref.new_ref>`:

```{code-cell}
x_ref = jax.new_ref(jnp.zeros(3))  # new array ref, with initial value [0., 0., 0.]

@jax.jit
def f():
  x_ref[1] += 1.  # indexed add-update

print(x_ref)  # Ref([0., 0., 0.])
f()
f()
print(x_ref)  # Ref([0., 2., 0.])
```

For a ref called `x_ref`, we can read its entire value into an `Array` by
writing `x_ref[...]`, and write its entire value using `x_ref[...] = A` for
some `Array`-valued expression `A`:

```{code-cell}
def g(x):
  x_ref = jax.new_ref(0.)
  x_ref[...] = jnp.sin(x)
  return x_ref[...]

print(jax.grad(g)(1.0))  # 0.54
```

Refs are a distinct type from `Array`, and come with some important
constraints and limitations. In particular, indexed reading and writing is
just about the *only* thing you can do with a ref. References can't be passed
where `Array`s are expected:

```{code-cell}
:tags: [raises-exception]

x_ref = jax.new_ref(1.0)
jnp.sin(x_ref)  # error! can't do math on refs
```

To do math, you need to read the ref's value first, like `jnp.sin(x_ref[...])`.

Reads and writes accept any NumPy indexing expression:

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

Indexing mostly follows NumPy behavior, except that an out-of-bounds index
raises an error when its value is known in advance (unlike with `Array`s,
where reads clamp and writes drop; see {ref}`jax-101-arrays`).

When you're done mutating, {func}`jax.freeze <jax.ref.freeze>` invalidates the
ref (so that accessing it afterwards is an error) and produces its final value
as an ordinary immutable `Array`:

```{code-cell}
final = jax.freeze(x_ref)
final
```

### Refs and purity

How do refs square with the purity condition from
{ref}`jax-101-transformations`? Recall that functional purity provides three
main benefits: it makes code and transformations easier for the user to reason
about; it makes code easier for the compiler to optimize, parallelize, and
scale; and it makes code easier for JAX to trace.

Because operations on refs are intercepted, tracing isn't a problem.
Their use does somewhat constrain the compiler's ability to transform code, but
only as much as explicit state threading would.
The main new thing to learn is how refs interact with transformations.

A function still counts as pure if it only uses refs internally. That
is, a function is impure if and only if it takes a ref as an input (either an
explicit argument or via closure). Purity is in the eye of the caller. So
functions that use refs internally transform the same way any pure function
would:

```{code-cell}
def normalize(x):        # pure: refs used internally only
  acc = jax.new_ref(0.0)
  acc[...] = jnp.sum(x)  # (a real program would do something less trivial)
  return x / acc[...]

jax.grad(lambda x: normalize(x).sum())(jnp.arange(1.0, 4.0))
```

Impure functions, meaning those that take refs as inputs, are more
constrained, but still work with many transformations. For example, you can
`vmap` an impure function over a batch of ref entries:

```{code-cell}
def scale_into(x, out_ref):   # impure: takes a ref argument
  out_ref[...] = 2.0 * x

xs = jnp.arange(3.0)
out_ref = jax.new_ref(jnp.zeros(3))
jax.vmap(scale_into)(xs, out_ref)   # each instance writes its own entry
print(out_ref[...])
```

What you *can't* do is `vmap` a function that closes over a ref, because with
every batch member writing to the same shared location, the final value would
be ambiguous:

```{code-cell}
:tags: [raises-exception]

r = jax.new_ref(0.0)

def write_shared(x):
  r[...] = x    # every batch member writes the same ref!

jax.vmap(write_shared)(jnp.arange(3.0))
```

### Ref restrictions

Refs come with rules designed to rule out *aliasing*, meaning two refs
pointing at the same memory, and other situations where the meaning of a
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

### Ref performance and further reading

Refs aren't just for expressiveness: they're also a tool for performance.
Writing to a ref updates its buffer in place rather than allocating a fresh
array. The full performance story is covered in {ref}`jax-201-jit`.

Refs also interact with automatic differentiation: you can plumb values out of
backward passes, accumulate gradients in place across microbatches, and
differentiate with respect to ref arguments. That material lives with
{ref}`jax-301-refs`.

## Where you've arrived

This completes the expressiveness tour: arrays and `jax.numpy` as the
vocabulary, `grad` and `vmap` as the verbs, pytrees for structure, keys for
randomness, and threaded values or refs for state.

What you can't do yet is make it *fast*. That's a matter of `jax.jit`,
sharding, and profiling, and it's exactly where the performance and scaling
docs pick up: {ref}`jax-201-jit`.
