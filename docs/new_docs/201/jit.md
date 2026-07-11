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

(jax-201-jit)=
# Just-in-time compilation

<!--* freshness: { reviewed: '2026-07-09' } *-->

{func}`jax.jit` is the transformation that makes JAX code fast: it compiles a
Python function into a single optimized computation, fused and specialized for
your hardware. The 101 docs introduced the machinery underneath —
{ref}`tracing <jax-101-tracing>`, jaxprs, and the purity rules — and promised
that `jax.jit(f)` computes exactly what `f` computes, just faster. This page
is about the "faster" part: what compilation buys, when it happens (and
re-happens), and the levers you control — static arguments, caching,
in-place updates with refs and buffer donation, and asynchronous dispatch.

## What compilation buys

Without `jit`, JAX executes operations one at a time, dispatching each to the
device as it's encountered. That's flexible, but it limits performance in two
ways: each dispatch carries Python and runtime overhead, and the compiler
never gets to see the whole computation at once, so it can't optimize across
operations.

`jax.jit` fixes both. It traces the function to a jaxpr (exactly as described
in {ref}`jax-101-tracing`) and hands the whole jaxpr to
[XLA](https://www.openxla.org/xla/), which compiles it into a single
optimized executable — fusing operations, eliminating temporary arrays, and
specializing for your CPU, GPU, or TPU. Consider a *scaled exponential linear
unit* ([SELU](https://proceedings.neurips.cc/paper/6698-self-normalizing-neural-networks.pdf)),
an operation commonly used in deep learning:

```{code-cell}
import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)
%timeit selu(x).block_until_ready()
```

```{code-cell}
selu_jit = jax.jit(selu)

# Warm up: the first call traces and compiles.
selu_jit(x).block_until_ready()

%timeit selu_jit(x).block_until_ready()
```

Here's what just happened:

1. We defined `selu_jit` as the compiled version of `selu`.
2. We called `selu_jit` once on `x`. This is where the tracing and XLA
   compilation happen. (We do this warm-up call outside the timing loop so
   that we measure execution, not compilation.)
3. We timed the compiled version. Subsequent calls never run the Python
   function `selu` at all — they go straight to the compiled executable.

(Note the use of {func}`~jax.block_until_ready`: because of JAX's
asynchronous dispatch, covered [below](#asynchronous-dispatch), timing
without it would measure only how long it takes to *launch* the work.)

## What the compiled function captures

A jaxpr records the function *as executed on the JAX types it was traced
with*. Anything Python-level is resolved during tracing and frozen into the
compiled result. That includes side effects — they run (once) at trace time
and don't exist in the compiled code — and it includes Python control flow.
If a function branches on a static property like rank, the jaxpr knows only
the branch that was taken:

```{code-cell}
def log2_if_rank_2(x):
  if x.ndim == 2:
    return jnp.log(x) / jnp.log(2.0)
  else:
    return x

jax.jit(log2_if_rank_2).trace(jnp.array([1, 2, 3])).jaxpr
```

For a rank-1 input, the compiled function is just the identity — the other
branch is gone. That's fine, because the cached executable is only reused for
inputs of the same JAX type; a rank-2 input triggers a fresh trace that takes
the other branch.

Impure functions are dangerous under `jit` for exactly this reason: side
effects happen once, at trace time, and are then absent from the cached
executable, so they might appear to work on the first call and silently
vanish afterwards. JAX often can't detect the impurity. (For debug printing
that survives compilation, use {func}`jax.debug.print` — see {doc}`debugging`;
for general side effects at a performance cost, see
{func}`jax.experimental.io_callback` ({doc}`callbacks`); to check for leaked tracers, use
{func}`jax.check_tracer_leaks`.)

## Why can't we just jit everything?

Given the speedups, you might wonder why we don't just apply `jax.jit` to
every function. Recall from {ref}`jax-101-transformations` that traced code
can't specialize on data — and `jit`'s tracers carry no values at all, only
JAX types. Value-dependent Python control flow therefore fails:

```{code-cell}
:tags: [raises-exception]

# Condition on the value of x.

def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

jax.jit(f)(10)  # Raises an error
```

```{code-cell}
:tags: [raises-exception]

# While loop conditioned on x and n.

def g(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

jax.jit(g)(10, 20)  # Raises an error
```

The problem in both cases is that we tried to condition the program's
trace-time flow on runtime values. Traced values inside `jit`, like `x` and
`n` here, can only affect control flow through their JAX types — shape,
dtype — not their values. For much more on this, including the structured
control-flow operations `jax.lax.cond` and `jax.lax.scan` that express
data-dependent control flow *inside* compiled code, see {doc}`control-flow`.

One pragmatic option is to jit only part of a function. If the expensive
work is inside the loop body, compile just the body and let Python drive the
loop:

```{code-cell}
@jax.jit
def loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

g_inner_jitted(10, 20)
```

This gives up cross-operation optimization for the loop itself, but keeps
each iteration compiled. (If you do this, read [JIT and caching](#jit-and-caching)
below to avoid accidentally recompiling every iteration.)

(jax-201-jit-static-arguments)=
## Marking arguments as static

If we really do need to branch on the value of an argument, we can tell JAX
to treat that argument as *static*: a regular Python value fixed at trace
time, rather than a traced array. The `static_argnums` and `static_argnames`
parameters to `jax.jit` do this:

```{code-cell}
f_jit_correct = jax.jit(f, static_argnums=0)
print(f_jit_correct(10))
```

```{code-cell}
g_jit_correct = jax.jit(g, static_argnames=['n'])
print(g_jit_correct(10, 20))
```

To specify static arguments when using `jit` as a decorator, use the
decorator-factory pattern:

```{code-cell}
@jax.jit(static_argnames=['n'])
def g_jit_decorated(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

print(g_jit_decorated(10, 20))
```

The cost: the compiled executable is specialized to the *value* of each
static argument, so JAX recompiles whenever a static argument takes a value
it hasn't seen. Marking an argument static is only a good strategy when the
function will see a limited set of values for it. (Static argument values
must also be hashable, since they become part of the cache key.)

## JIT and caching

The first call to a jitted function pays for tracing and compilation;
everything after that is supposed to be cheap. Understanding how the cache
works is key to actually getting that deal.

Suppose we define `f = jax.jit(g)`. When `f` is first invoked, it gets
compiled and the resulting executable is cached. The cache key includes:

- the JAX types of the arguments — new shapes or dtypes trigger a fresh
  trace and compile (this is what made the rank-1/rank-2 example above safe);
- the values of any static arguments; and
- the identity of the function `g` itself.

That last point has a practical consequence: avoid calling `jax.jit` on
temporary functions defined inside loops or other inner scopes. The cache
relies on the hash of the function, so freshly-created lambdas and `partial`
objects — even ones wrapping the same underlying code — look like new
functions every time, and recompile every time:

```{code-cell}
from functools import partial

def unjitted_loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted_partial(x, n):
  i = 0
  while i < n:
    # Don't do this! A fresh `partial` has a different hash each time,
    # so this recompiles on every iteration.
    i = jax.jit(partial(unjitted_loop_body))(i)
  return x + i

def g_inner_jitted_lambda(x, n):
  i = 0
  while i < n:
    # Don't do this either! Same problem: a fresh lambda every iteration.
    i = jax.jit(lambda x: unjitted_loop_body(x))(i)
  return x + i

def g_inner_jitted_normal(x, n):
  i = 0
  while i < n:
    # This is OK: `jax.jit` sees the same function object each time,
    # and finds the cached executable.
    i = jax.jit(unjitted_loop_body)(i)
  return x + i

print("jit called in a loop with partials:")
%timeit g_inner_jitted_partial(10, 20).block_until_ready()

print("jit called in a loop with lambdas:")
%timeit g_inner_jitted_lambda(10, 20).block_until_ready()

print("jit called in a loop with caching:")
%timeit g_inner_jitted_normal(10, 20).block_until_ready()
```

When a program recompiles more than you expect, you don't have to guess why.
Setting the `jax_explain_cache_misses` config option makes JAX print an
explanation for every tracing cache miss, including which previously-seen
call the new one most closely resembles and what differs:

```{code-cell}
jax.config.update("jax_explain_cache_misses", True)

@jax.jit
def double(x):
  return 2 * x

_ = double(jnp.arange(3))    # cache miss: never seen this function before
_ = double(jnp.arange(3))    # cache hit: silent
_ = double(jnp.arange(3.0))  # cache miss: x's JAX type changed
```

```{code-cell}
jax.config.update("jax_explain_cache_misses", False)
```

The first call misses because the function has never been traced; the second
is a silent cache hit; and the third misses because `x`'s JAX type changed
from `i32[3]` to `f32[3]` — reported in exactly the `jax.typeof` notation.
Each miss cites the call site, so unintended recompilation is easy to trace
to its source: an argument that keeps changing shape, a static argument with
too many values, or the fresh-function pitfall above, whose telltale is a
`never seen function` miss on every call.

The other classic source of endless recompilation is data whose shape
genuinely varies from call to call: ragged sequence lengths, a partial final
batch at the end of an epoch. Since the cache is keyed on JAX types, every
new shape is a fresh compile. The standard fix is to pad variable-shaped
data up to a small, fixed set of *bucket* shapes, trading a little wasted
compute (and a masking `jnp.where` or two) for a bounded number of
compilations.

(Should you ever need to drop every cached trace and executable — say, to
measure compilation from a clean slate — {func}`jax.clear_caches` does
exactly that.)

For a lighter-weight signal, `jax_log_compiles` logs one line per
compilation, without the diagnosis. When tracing or compilation time itself
becomes the problem, {doc}`slow-compilation` is a field guide to diagnosing
it; and the persistent compilation cache
({doc}`/persistent_compilation_cache`) keeps compiled executables across
process restarts.

## Using `jit` with methods

Most `jax.jit` examples decorate stand-alone functions, but sooner or later
you'll want to jit a *method* — which raises a question the examples don't
answer: what should JAX do with `self`?

```{code-cell}
:tags: [raises-exception]

class CustomClass:
  def __init__(self, x: jax.Array, mul: bool):
    self.x = x
    self.mul = mul

  @jax.jit  # <---- How to do this correctly?
  def calc(self, y):
    if self.mul:
      return self.x * y
    return y

c = CustomClass(2, True)
c.calc(3)
```

The problem is that the first argument to `calc` is `self`, of type
`CustomClass`, and JAX doesn't know how to handle that type. There are three
basic strategies, in increasing order of power.

### Strategy 1: a jitted helper function

The most straightforward approach: keep `jit` on a stand-alone helper that
takes only JAX-compatible arguments, and have the method call it:

```{code-cell}
class CustomClass:
  def __init__(self, x: jax.Array, mul: bool):
    self.x = x
    self.mul = mul

  def calc(self, y):
    return _calc(self.mul, self.x, y)

@jax.jit(static_argnums=0)
def _calc(mul, x, y):
  if mul:
    return x * y
  return y

c = CustomClass(2, True)
print(c.calc(3))
```

This is simple and explicit, and JAX never has to learn about `CustomClass`
at all. The cost is that the method's logic lives outside the class.

### Strategy 2: marking `self` as static

Another common pattern is to mark the `self` argument as static. Done
naively, it's a trap:

```{code-cell}
class CustomClass:
  def __init__(self, x: jax.Array, mul: bool):
    self.x = x
    self.mul = mul

  # WARNING: broken, as we'll see below. Don't copy & paste!
  @jax.jit(static_argnums=0)
  def calc(self, y):
    if self.mul:
      return self.x * y
    return y

c = CustomClass(2, True)
print(c.calc(3))
```

No more error — but there's a catch. Static arguments become cache keys, so
JAX relies on their hash and equality. The default `__hash__` for a custom
object is its object ID, which doesn't change when the object mutates — so
mutating the object silently serves stale compiled code:

```{code-cell}
c.mul = False
print(c.calc(3))  # Should print 3... but the cache doesn't know mul changed
```

You can partially address this by defining `__hash__` and `__eq__` in terms
of the object's contents, so that differing objects actually miss the cache:

```{code-cell}
class CustomClass:
  def __init__(self, x: jax.Array, mul: bool):
    self.x = x
    self.mul = mul

  @jax.jit(static_argnums=0)
  def calc(self, y):
    if self.mul:
      return self.x * y
    return y

  def __hash__(self):
    return hash((self.x, self.mul))

  def __eq__(self, other):
    return (isinstance(other, CustomClass) and
            (self.x, self.mul) == (other.x, other.mul))
```

This works with `jit` and other transformations **so long as you never
mutate the object**: mutating a value that's in use as a hash key leads to
subtle problems — which is exactly why Python's mutable containers (`dict`,
`list`) don't define `__hash__`. If your class mutates its attributes, it
isn't really static, and there's a better option.

### Strategy 3: registering the class as a pytree

The most flexible approach is to register the type as a custom pytree node
({ref}`jax-101-custom-pytrees`), saying explicitly which parts are dynamic
data and which are static metadata:

```{code-cell}
from jax import tree_util

class CustomClass:
  def __init__(self, x: jax.Array, mul: bool):
    self.x = x
    self.mul = mul

  @jax.jit
  def calc(self, y):
    if self.mul:
      return self.x * y
    return y

  def _tree_flatten(self):
    children = (self.x,)          # arrays / dynamic values
    aux_data = {'mul': self.mul}  # static values
    return (children, aux_data)

  @classmethod
  def _tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)

tree_util.register_pytree_node(CustomClass,
                               CustomClass._tree_flatten,
                               CustomClass._tree_unflatten)
```

Now `self` flows through `jit` like any other pytree argument: the arrays in
`children` are traced, the `aux_data` is static, plain `@jax.jit` works with
no `static_argnums`, and the problems above solve themselves:

```{code-cell}
c = CustomClass(2, True)
print(c.calc(3))

c.mul = False                        # mutation is detected
print(c.calc(3))

c = CustomClass(jnp.array(2), True)  # non-hashable x is supported
print(c.calc(3))
```

(If your class is a dataclass, {func}`jax.tree_util.register_dataclass` gets
you the same result with much less ceremony — see
{ref}`jax-101-custom-pytrees`.)

Beyond *when* compilation happens, you can also control what the compiler
*does* — XLA flags set per function or process-wide, and metadata attached to
individual operations. That's the subject of {doc}`controlling-xla`.

(jax-201-jit-refs)=
## In-place updates with refs

A compiled function's inputs and outputs are immutable arrays, so at the
`jit` boundary every output gets a fresh buffer — even when you're
conceptually just *updating* an input, as in a training step that maps
parameters to new parameters. Inside the compiled computation XLA reuses
memory aggressively, but it can't reuse an input's buffer for an output
unless you tell it something extra. JAX has two ways to say it: pass the data
as a **ref**, or **donate** the input buffer. Refs express the intent
directly, so we cover them first; donation, next, is the longer-standing
mechanism you'll meet throughout existing JAX code.

A ref ({ref}`jax-101-refs`) passed into a jitted function is mutable memory:
writes through it happen in place, and no fresh buffer is allocated. We can
watch the buffer address stay fixed across the call:

```{code-cell}
@jax.jit
def sin_inplace(x_ref):
  x_ref[...] = jnp.sin(x_ref[...])

x_ref = jax.new_ref(jnp.arange(3.0))
print(x_ref.unsafe_buffer_pointer(), x_ref)
sin_inplace(x_ref)
print(x_ref.unsafe_buffer_pointer(), x_ref)
```

`sin_inplace` updates the buffer backing `x_ref`, so its address stays the
same. In general, under a `jit` you should expect refs to point to fixed
buffer addresses, and indexed updates to be performed in place. Compared to
donation, there's no promise for you to keep and nothing to accidentally
misuse afterwards: the ref *is* the buffer, before and after the call.

```{note}
**Temporary caveat:** dispatch from Python to impure `jit`-compiled functions
that take ref inputs is currently slower than dispatch to pure
`jit`-compiled functions, since it takes a less optimized path.
```

Working with refs does mean writing the function differently — mutation
through indexed writes, rather than functional updates. For code already
written functionally (which is most existing JAX code, and the convention
across the ecosystem), buffer donation achieves the in-place effect without
changing the function.

(jax-201-buffer-donation)=
## Buffer donation

When JAX executes a computation, it uses device buffers for all inputs and
outputs. If you know an input won't be needed after the computation, and it
matches the shape and element type of an output, you can *donate* the input
buffer to hold that output, reducing peak memory by the size of the donated
buffer. The canonical case is a functional update, where the new state
replaces the old:

```python
params, state = jax.jit(update_fn, donate_argnums=(0, 1))(params, state)
```

Think of donation as a memory-efficient functional update on immutable
arrays. *Within* a compiled computation, XLA reuses buffers automatically;
at the `jit` boundary, though, JAX must assume you might still hold a
reference to the input — unless you promise otherwise with `donate_argnums`
(or `donate_argnames`):

```python
def add(x, y):
  return x + y

x = jax.device_put(np.ones((2, 3)))
y = jax.device_put(np.ones((2, 3)))
# Execute `add` with donation of the buffer for `y`. The result has
# the same shape and type as `y`, so it will share its buffer.
z = jax.jit(add, donate_argnums=(1,))(x, y)
```

Donation comes with rules and sharp edges:

- **Donated means gone.** After the call, the donated input is invalid, and
  using it is an error:

  ```python
  z = jax.jit(add, donate_argnums=(1,))(x, y)
  w = y + 1  # Reuses `y`, whose buffer was donated above
  # >> RuntimeError: Invalid argument: CopyToHostAsync() called on invalid buffer
  ```

- **Keyword arguments aren't donated** by `donate_argnums`. This code donates
  nothing:

  ```python
  params, state = jax.jit(update_fn, donate_argnums=(0, 1))(params=params, state=state)
  ```

- **Pytree arguments donate all their buffers.** Donating an argument that's
  a pytree donates every array in it:

  ```python
  def add_ones(xs: list[jax.Array]):
    return [x + 1 for x in xs]

  xs = [jax.device_put(np.ones((2, 3))), jax.device_put(np.ones((3, 4)))]
  # Donates the buffers of both arrays in `xs`.
  z = jax.jit(add_ones, donate_argnums=0)(xs)
  ```

- **Unusable donations are dropped, with a warning.** If there are more
  donated buffers than outputs to hold them, or no output matches a donated
  buffer's shape and element type, you'll see
  `UserWarning: Some donated buffers were not usable`.

As the sharp edges above suggest, donation is a promise layered on top of
immutable semantics, and it's possible to get subtly wrong — the promise is
checked only partially, and only at runtime. When you have the freedom to
restructure, refs ({ref}`jax-201-jit-refs` above) express the same intent
with less to misuse. Donation remains the workhorse for functionally-written
code, which is to say most JAX code today.

(jax-201-async-dispatch)=
## Asynchronous dispatch

One more piece of the performance model, relevant with and without `jit`:
JAX does not wait for a computation to finish before returning control to
Python. Consider:

```{code-cell}
import numpy as np
from jax import random

x = random.uniform(random.key(0), (1000, 1000))
jnp.dot(x, x) + 3.0
```

When `jnp.dot(x, x)` executes, JAX returns immediately with a
{class}`jax.Array` that is really a *future*: a value that will be produced
on the device but isn't necessarily available yet. We can inspect its JAX
type, and pass it into further JAX computations, without waiting — only when
we actually *look at* the value from the host (printing it, converting it to
a NumPy array) does Python block until the computation is done. (The cell
above shows values only because rendering the result forced the wait.)

This is why asynchronous dispatch is valuable: Python "runs ahead" of the
device, enqueueing work and staying off the critical path. As long as the
host can keep the device busy and doesn't inspect results, an arbitrary
amount of work can be in flight.

The surprising consequence is for measurement. Time an operation naively and
you measure only the dispatch:

```{code-cell}
%time jnp.dot(x, x)
```

A fraction of a millisecond would be a suspiciously good time for a
1000×1000 matrix multiplication! To measure the actual computation, force
completion with {meth}`~jax.Array.block_until_ready`:

```{code-cell}
%time jnp.dot(x, x).block_until_ready()
```

Blocking without transferring the result to the host (as
`block_until_ready` does) is usually faster than forcing a transfer with
`np.asarray(...)`, and is the right tool for microbenchmarks. Benchmarking
JAX code has a few more pitfalls of this flavor — the next page,
{doc}`profiling`, starts there.

## Where to next

- {doc}`aot` — running `jit`'s stages (trace, lower, compile) ahead of time,
  for inspection or control.
- {doc}`profiling` — measure before you optimize: benchmarking pitfalls and
  the JAX profiler.
- {doc}`control-flow` — expressing conditionals and loops that live *inside*
  compiled code.
- {doc}`debugging` — printing and inspecting values under `jit`.
- {doc}`slow-compilation` — when tracing or compilation itself is the
  bottleneck.
- {doc}`precision` — controlling the speed/accuracy tradeoff inside matmuls.
- {doc}`controlling-xla` — XLA compiler flags and per-operation metadata.
