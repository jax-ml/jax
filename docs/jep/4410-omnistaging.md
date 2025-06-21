# Omnistaging
_mattjj@_
_Sept 25 2020_

This is more of an upgrade guide than a design doc.

## Contents

* [tl;dr](#tldr)
* [What is "omnistaging" and why is it useful?](#what-is-omnistaging-and-why-is-it-useful)
* [What issues can arise when omnistaging is switched on?](#what-issues-can-arise-when-omnistaging-is-switched-on)
  * [Using `jax.numpy` for shape computations](#using-jaxnumpy-for-shape-computations)
  * [Side-effects](#side-effects)
  * [Small numerical differences based on XLA optimizations](#small-numerical-differences-based-on-xla-optimizations)
  * [Dependence on JAX internal APIs that changed](#dependence-on-jax-internal-apis-that-changed)
  * [Triggering XLA compile time bugs](#triggering-xla-compile-time-bugs)

## tl;dr

### What's going on?

A change to JAX's tracing infrastructure called “omnistaging”
([jax-ml/jax#3370](https://github.com/jax-ml/jax/pull/3370)) was switched on in
jax==0.2.0. This change improves memory performance, trace execution time, and
simplifies jax internals, but may cause some existing code to break. Breakage is
usually a result of buggy code, so long-term it’s best to fix the bugs, but
omnistaging can also be disabled as a temporary workaround. And we're happy to
help you with fixes!

### How do I know if omnistaging broke my code?

The easiest way to tell if omnistaging is responsible is to disable omnistaging
and see if the issues go away. See the [What issues can arise when omnistaging
is switched on?](#what-issues-can-arise-when-omnistaging-is-switched-on) section
below.

### How can I disable omnistaging for now?

*Note: this applies to JAX versions 0.2.0 through 0.2.11; omnistaging cannot be
disabled in JAX versions 0.2.12 and higher*

It is temporarily possible to disable omnistaging by
1. setting the shell environment variable `JAX_OMNISTAGING` to something falsey;
2. setting the boolean flag `jax_omnistaging` to something falsey if your code
   parses flags with absl;
3. using this statement near the top of your main file:
```python
jax.config.disable_omnistaging()
```

### How do I fix bugs exposed by omnistaging?

By far the most common issue with omnistaging is using `jax.numpy` to compute
shape values or other trace-time constants. See the code block below for a quick
example, and for full details along with other issues see the section [What
issues can arise when omnistaging is switched
on?](#what-issues-can-arise-when-omnistaging-is-switched-on).

Instead of this:

```python
@jit
def f(x):
  input_size = jnp.prod(x.shape)
  if input_size > 100:
    ...
```

do this:

```python
import numpy as np

@jit
def f(x):
  input_size = np.prod(x.shape)
  if input_size > 100:
    ...
```

Instead of thinking of `jax.numpy` as a drop-in replacement for `numpy`, it's
now better to think of using `jax.numpy` operations only when you want to perform a
computation on an accelerator (like your GPU).


## What is "omnistaging" and why is it useful?

Omnistaging is the name for a JAX core upgrade aimed at staging out more
computation from op-by-op Python to XLA, and avoiding any "trace-time constant
folding" in `jit`, `pmap`, and control flow primitives. As a result, omnistaging
improves JAX's memory performance (sometimes dramatically) both by reducing
fragmentation during tracing and by producing fewer large compile-time constants
for XLA. It can also improve tracing performance by eliminating op-by-op
execution at tracing time. Further, omnistaging simplifies JAX core internals,
fixing many outstanding bugs and setting the stage for important upcoming
features.

The name "omnistaging" means staging out everything possible.

### Toy example

JAX transformations like `jit` and `pmap` stage out computations to XLA. That
is, we apply them to functions comprising multiple primitive operations so that
rather being executed one at a time from Python the operations are all part of
one end-to-end optimized XLA computation.

But exactly which operations get staged out? Until omnistaging, JAX staged out
computation based on data dependence only. Here's an example function, followed
by the XLA HLO program it stages out _before_ the omnistaging change:

```python
from jax import jit
import jax.numpy as jnp

@jit
def f(x):
  y = jnp.add(1, 1)
  return x * y

f(3)
```

```
ENTRY jit_f.6 {
  constant.2 = pred[] constant(false)
  parameter.1 = s32[] parameter(0)
  constant.3 = s32[] constant(2)
  multiply.4 = s32[] multiply(parameter.1, constant.3)
  ROOT tuple.5 = (s32[]) tuple(multiply.4)
}
```

Notice that the `add` operation is not staged out. Instead, we only see a
multiply.

Here's the HLO generated from this function _after_ the omnistaging change:

```
ENTRY jit_f.8 {
  constant.2 = pred[] constant(false)
  parameter.1 = s32[] parameter(0)
  constant.3 = s32[] constant(1)
  constant.4 = s32[] constant(1)
  add.5 = s32[] add(constant.3, constant.4)
  multiply.6 = s32[] multiply(parameter.1, add.5)
  ROOT tuple.7 = (s32[]) tuple(multiply.6)
}
```

### Slightly less toy example

Here's a less toy example which can arise in practice when we want to create
boolean masks:

```python
import jax.numpy as jnp
from jax import lax

@jit
def select_tril(x):
  mask = jnp.arange(x.shape[0])[:, None] > jnp.arange(x.shape[1])
  return lax.select(mask, x, jnp.zeros_like(x))  # lax.select is like jnp.where

x = np.arange(12).reshape((3, 4))
select_tril(x)
```

_Before_ omnistaging:

```
ENTRY jit_select_tril.8 {
  constant.3 = pred[] constant(false)
  constant.1 = pred[3,4]{1,0} constant({...})
  parameter.2 = s32[3,4]{1,0} parameter(0)
  constant.4 = s32[] constant(0)
  broadcast.5 = s32[3,4]{1,0} broadcast(constant.4), dimensions={}
  select.6 = s32[3,4]{1,0} select(constant.1, parameter.2, broadcast.5)
  ROOT tuple.7 = (s32[3,4]{1,0}) tuple(select.6)
}
```

The `select` operation is staged out, but the operations for constructing the
constant `mask` are not. Rather than being staged out, the operations that
construct `mask` are executed op-by-op at Python tracing time, and XLA only sees
a compile time constant `constant.1` representing the value of `mask`. That’s
unfortunate, because if we had staged out the operations for constructing
`mask`, XLA could have fused them into the `select` and avoided materializing
the result at all. As a result we end up wasting memory with a potentially-large
constant, wasting time dispatching multiple un-fused op-by-op XLA computations,
and potentially even fragmenting memory.

(The `broadcast` that corresponds to the construction of the zeros array for
`jnp.zeros_like(x)` is staged out because JAX is lazy about very simple
expressions from [jax-ml/jax#1668](https://github.com/jax-ml/jax/pull/1668). After
omnistaging, we can remove that lazy sublanguage and simplify JAX internals.)

The reason the creation of `mask` is not staged out is that, before omnistaging,
`jit` operates based on data dependence. That is, `jit` stages out only those
operations in a function that have a data dependence on an argument. Control
flow primitives and `pmap` behave similarly. In the case of `select_tril`, the
operations to construct the constant `mask` do not have a data dependence on the
argument x, so they are not staged out; only the `lax.select` call has a data
dependence.

With omnistaging all `jax.numpy` calls in the dynamic context of a
`jit`-transformed function are staged out to XLA. That is, after omnistaging the
computation XLA sees for `select_tril` is

```
ENTRY jit_select_tril.16 {
  constant.4 = pred[] constant(false)
  iota.1 = s32[3]{0} iota(), iota_dimension=0
  broadcast.5 = s32[3,1]{1,0} broadcast(iota.1), dimensions={0}
  reshape.7 = s32[3]{0} reshape(broadcast.5)
  broadcast.8 = s32[3,4]{1,0} broadcast(reshape.7), dimensions={0}
  iota.2 = s32[4]{0} iota(), iota_dimension=0
  broadcast.6 = s32[1,4]{1,0} broadcast(iota.2), dimensions={1}
  reshape.9 = s32[4]{0} reshape(broadcast.6)
  broadcast.10 = s32[3,4]{1,0} broadcast(reshape.9), dimensions={1}
  compare.11 = pred[3,4]{1,0} compare(broadcast.8, broadcast.10), direction=GT
  parameter.3 = s32[3,4]{1,0} parameter(0)
  constant.12 = s32[] constant(0)
  broadcast.13 = s32[3,4]{1,0} broadcast(constant.12), dimensions={}
  select.14 = s32[3,4]{1,0} select(compare.11, parameter.3, broadcast.13)
  ROOT tuple.15 = (s32[3,4]{1,0}) tuple(select.14)
}
```

## What issues can arise when omnistaging is switched on?

As a consequence of staging out all `jax.numpy` operations from Python to XLA
when in the dynamic context of a `jit` or `pmap`, some code that worked
previously can start raising loud errors. As explained below, these behaviors
were already buggy before omnistaging, but omnistaging makes them into hard
errors.

### Using `jax.numpy` for shape computations

#### Example
```
from jax import jit
import jax.numpy as jnp

@jit
def ex1(x):
  size = jnp.prod(jnp.array(x.shape))
  return x.reshape((size,))

ex1(jnp.ones((3, 4)))
```

#### Error message

```
[... full traceback ...]
  File "/home/mattjj/packages/jax/jax/core.py", line 862, in raise_concretization_error
    raise ConcretizationTypeError(msg)
jax.core.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected.

The error arose in jax.numpy.reshape.

While tracing the function ex1 at ex1.py:4, this value became a tracer due to JAX operations on these lines:

  operation c:int32[] = reduce_prod[ axes=(0,) ] b:int32[2]
    from line ex1.py:6 (ex1)

You can use transformation parameters such as `static_argnums` for `jit` to avoid tracing particular arguments of transformed functions.

See https://docs.jax.dev/en/latest/faq.html#abstract-tracer-value-encountered-where-concrete-value-is-expected-error for more information.

Encountered tracer value: Traced<ShapedArray(int32[])>with<DynamicJaxprTrace(level=0/1)>
```

#### Explanation

With omnistaging, we can't use `jax.numpy` for shape computations as in the use
of `jnp.prod` above because in the dynamic context of a jit function those
operations will be staged out of Python as values to be computed at execution
time, yet we need them to be compile-time (and hence trace-time) constants.

Before omnistaging, this code wouldn't have raised an error, but it was a common
performance bug: the `jnp.prod` computation would have been executed on the
device at tracing time, meaning extra compilation, transfers, synchronization,
allocations, and potentially memory fragmentation.


#### Solution

The solution is simply to use the original `numpy` for shape calculations like
these. Not only do we avoid the error, but also we keep the computations on the
host (and with lower overheads).

This issue was common enough in code that we tried to make the error
message especially good. In addition to the stack trace showing where an
abstract tracer value caused a problem (the `jnp.reshape` line in the full stack
trace, on omni.py:10), we also explain why this value became a tracer in the
first place by pointing to the upstream primitive operation that caused it to
become an abstract tracer (the `reduce_prod` from `jnp.prod` on omni.py:9) and to
which `jit`-decorated function the tracer belongs (`ex1` on omni.py:6).


### Side-effects

#### Example

```python
from jax import jit
from jax import random

key = random.PRNGKey(0)

def init():
  global key
  key, subkey = random.split(key)
  return random.normal(subkey, ())

print(init())  # -1.2515389
print(init())  # -0.58665067

init = jit(init)
print(init())  # 0.48648298
print(init())  # 0.48648298  !!
```

That last call has repeated randomness but no hard error, because we aren't
re-executing the Python. But if we look at `key`, we see an escaped tracer _when
omnistaging is on_:
```python
print(key) # Traced<ShapedArray(uint32[2])>with<DynamicJaxprTrace(level=0/1)>
```
Before omnistaging, the `random.split` call would not be staged out and so we
wouldn't get an escaped tracer. The code would still be buggy in that the jitted
function wouldn't be reproducing the semantics of the original function (because
of the repeated use of the same PRNG key), ultimately due to the side effect.

With omnistaging on, if we touch `key` again, we'll get an escaped tracer error:

```python
random.normal(key, ())
```

#### Error message

```
[... full stack trace …]
  File "/home/mattjj/packages/jax/jax/interpreters/partial_eval.py", line 836, in _assert_live
    raise core.escaped_tracer_error(msg)
jax.core.UnexpectedTracerError: Encountered an unexpected tracer. Perhaps this tracer escaped through global state from a previously traced function.
The functions being transformed should not save traced values to global state. Detail: tracer created on line example.py:8 (init).
```

#### Explanation

The second largest category of omnistaging issues we found had to do with
side-effecting code. This code already voided the JAX warranty by transforming
effectful functions, but due to pre-omnistaging "trace-time constant folding"
behavior, some side effecting functions could nevertheless behave correctly.
Omnistaging catches more of these errors.

#### Solution

The solution is to identify JAX-transformed functions that rely on side effects,
and to rewrite them not to be effectful.


### Small numerical differences based on XLA optimizations

Because with omnistaging more computations are being staged out to XLA, rather
than some being executed at trace time, that can have the effect of reordering
floating point operations. As a result, we've seen numerical behaviors change in
a way that causes tests with overly tight tolerances to fail when omnistaging is
switched on.


### Dependence on JAX internal APIs that changed

Omnistaging involved some big revisions to JAX's core code, including removing
or changing internal functions. Any code that relies on such internal
JAX APIs can break when omnistaging is switched on, either with build errors
(from pytype) or runtime errors.

### Triggering XLA compile time bugs

Because omnistaging involves staging out more code to XLA, we've seen it trigger
pre-existing XLA compile-time bugs on some backends. The best thing to do with
these is to report them so we can work with the XLA teams on fixes.
