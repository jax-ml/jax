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

```{code-cell}
:tags: [remove-cell]

# This ensures that code cell tracebacks appearing below will be concise.
%xmode minimal
```

(jax-201-control-flow)=
# Control flow and logical operators with `jit`

<!--* freshness: { reviewed: '2026-07-09' } *-->

When executing eagerly (outside of `jit`), JAX code works with Python control
flow and logical operators, like `and` or `or`, just like NumPy code. Using
control flow and logical operators with `jit` is more complicated.

Python control flow and logical operators are evaluated at `jax.jit` trace
time, such that the compiled function represents a single control path.
Logical operators affect the path via short-circuiting. If the path depends
on the values of the inputs, the function (by default) cannot be traced with
`jax.jit`.

```{code-cell}
from jax import jit
import jax.numpy as jnp
```

So this doesn't work:

```{code-cell}
:tags: [raises-exception]

@jit
def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

# This will fail!
f(2)
```

Neither does this:

```{code-cell}
:tags: [raises-exception]

@jit
def g(x):
  return (x < 3) and (x > 0)

# This will fail!
g(2)
```

__What gives!?__

Recall the tracing story from {ref}`jax-101-tracing` and {doc}`jit`: so that
the compiled code can be cached and reused for many argument values, `jit`
traces your function with tracers that carry only the JAX type, not any
concrete value.  That generality is exactly what fails above: on a line like
`if x < 3` (or a short-circuiting `and`), Python demands a concrete value to
choose a path, but we have no concrete value for `x < 3`.

There's a dial here: trace more abstractly and the compiled result is more
reusable, but your Python code is more constrained; trace more concretely and
the Python code is freer, but you recompile more often. The `static_argnames`
(or `static_argnums`) argument to `jit` ({ref}`jax-201-jit-static-arguments`)
turns that dial per argument, tracing on the concrete values of the arguments
you mark. Here's that example function again:

```{code-cell}
def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

f = jit(f, static_argnames='x')

print(f(2.))
```

Here's another example, this time involving a loop:

```{code-cell}
def f(x, n):
  y = 0.
  for i in range(n):
    y = y + x[i]
  return y

f = jit(f, static_argnames='n')

f(jnp.array([2., 3., 4.]), 2)
```

In effect, the loop gets statically unrolled.

## Shapes that depend on argument values

These control-flow issues also come up in a more subtle way: functions we
want to `jit` can't specialize the shapes of internal arrays on argument
*values* (specializing on argument *shapes* is fine). As a trivial example,
here's a function whose output shape depends on the input value `length`:

```{code-cell}
def example_fun(length, val):
  return jnp.ones((length,)) * val
# un-jit'd works fine
print(example_fun(5, 4))
```

```{code-cell}
:tags: [raises-exception]

bad_example_jit = jit(example_fun)
# this will fail:
bad_example_jit(10, 4)
```

```{code-cell}
# static_argnames tells JAX to recompile on changes at these argument positions:
good_example_jit = jit(example_fun, static_argnames='length')
# first compile
print(good_example_jit(10, 4))
# recompiles
print(good_example_jit(5, 4))
```

`static_argnames` works well if `length` in our example rarely changes, but
it means constant recompilation if it changes often. (For shapes that genuinely vary
call to call, see the padding-to-buckets advice in {doc}`jit`.)

## Structured control flow primitives

There are more options for control flow in JAX. Say you want to avoid
re-compilations but still want to use control flow that's traceable, and that
avoids unrolling large loops. Then you can use these four structured
control-flow primitives:

 - `lax.cond`  _differentiable_
 - `lax.while_loop` __fwd-mode-differentiable__
 - `lax.fori_loop` __fwd-mode-differentiable__ in general; __fwd and
   rev-mode differentiable__ if endpoints are static.
 - `lax.scan` _differentiable_

### `cond`

Python equivalent:

```python
def cond(pred, true_fun, false_fun, operand):
  if pred:
    return true_fun(operand)
  else:
    return false_fun(operand)
```

```{code-cell}
from jax import lax

operand = jnp.array([0.])
print(lax.cond(True, lambda x: x+1, lambda x: x-1, operand))
# --> array([1.], dtype=float32)
print(lax.cond(False, lambda x: x+1, lambda x: x-1, operand))
# --> array([-1.], dtype=float32)
```

Unlike a Python `if`, the predicate here can be a traced value. The choice of
branch happens on the device, at run time, inside the compiled program.

`jax.lax` provides two other functions that allow branching on dynamic
predicates:

- {func}`lax.select <jax.lax.select>` is like a batched version of
  `lax.cond`, with the choices expressed as pre-computed arrays rather than
  as functions.
- {func}`lax.switch <jax.lax.switch>` is like `lax.cond`, but allows
  switching between any number of callable choices.

In addition, `jax.numpy` provides several numpy-style interfaces to these
functions:

- {func}`jnp.where <jax.numpy.where>` with three arguments is the numpy-style
  wrapper of `lax.select`.
- {func}`jnp.piecewise <jax.numpy.piecewise>` is a numpy-style wrapper of
  `lax.switch`, but switches on a list of boolean conditions rather than a
  single scalar index.
- {func}`jnp.select <jax.numpy.select>` has an API similar to
  `jnp.piecewise`, but the choices are given as pre-computed arrays rather
  than as functions. It is implemented in terms of multiple calls to
  `lax.select`.

### `while_loop`

Python equivalent:

```python
def while_loop(cond_fun, body_fun, init_val):
  val = init_val
  while cond_fun(val):
    val = body_fun(val)
  return val
```

```{code-cell}
init_val = 0
cond_fun = lambda x: x < 10
body_fun = lambda x: x + 1
lax.while_loop(cond_fun, body_fun, init_val)
# --> array(10, dtype=int32)
```

Note the differentiability annotation above: `while_loop` is only
forward-mode differentiable. Reverse-mode autodiff needs to run the loop
backwards, saving each iteration's intermediates on the way forward, which
requires a bound on the number of iterations, and a `while_loop`'s trip count
is dynamic and unbounded. For reverse-mode differentiation through a loop,
use `scan` (fixed length), or `fori_loop` with static bounds (which lowers to
`scan`).

### `fori_loop`

Python equivalent:

```python
def fori_loop(start, stop, body_fun, init_val):
  val = init_val
  for i in range(start, stop):
    val = body_fun(i, val)
  return val
```

```{code-cell}
init_val = 0
start = 0
stop = 10
body_fun = lambda i, x: x + i
lax.fori_loop(start, stop, body_fun, init_val)
# --> array(45, dtype=int32)
```

### `scan`

The most commonly used of the four is {func}`jax.lax.scan`: a loop with a
fixed number of iterations that carries state from step to step, optionally
consuming a per-step slice of an input array and stacking per-step outputs.

Python equivalent:

```python
def scan(f, init, xs):
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)
```

```{code-cell}
def cumsum_step(carry, x):
  new_carry = carry + x
  return new_carry, new_carry   # (next state, this step's output)

final, cumulative = lax.scan(cumsum_step, 0.0, jnp.arange(1., 5.))
print(final)
print(cumulative)
```

Compared to unrolling a Python loop, `scan` compiles the body once no matter
how many iterations run, so long training loops and sequence models compile
in constant time instead of time proportional to the loop length. And unlike
`while_loop`, `scan` supports both forward- and reverse-mode
differentiation, which is why it's the standard way to express a training
loop's steps or an RNN's time axis inside `jit`.

For fine-tuning that compile-time/run-time trade, `scan` takes an `unroll`
parameter: `unroll=k` makes each iteration of the compiled loop perform `k`
steps of the scan, and `unroll=True` unrolls the loop entirely. Larger unroll
amounts give XLA more opportunity to fuse and parallelize across steps, at
the cost of compile time and program size, which is often worthwhile when the
body is small relative to per-iteration overhead. (`lax.fori_loop` accepts the
same parameter.)

## Logical operators

`jax.numpy` provides `logical_and`, `logical_or`, and `logical_not`, which
operate element-wise on arrays and can be evaluated under `jit` without
recompiling. Like their NumPy counterparts, the binary operators do not
short-circuit. Bitwise operators (`&`, `|`, `~`) can also be used with `jit`.

For example, consider a function that checks if its input is a positive even
integer. The pure Python and JAX versions give the same answer when the input
is scalar.

```{code-cell}
def python_check_positive_even(x):
  is_even = x % 2 == 0
  # `and` short-circuits, so when `is_even` is `False`, `x > 0` is not evaluated.
  return is_even and (x > 0)

@jit
def jax_check_positive_even(x):
  is_even = x % 2 == 0
  # `logical_and` does not short-circuit, so `x > 0` is always evaluated.
  return jnp.logical_and(is_even, x > 0)

print(python_check_positive_even(24))
print(jax_check_positive_even(24))
```

When the JAX version with `logical_and` is applied to an array, it returns
elementwise values.

```{code-cell}
x = jnp.array([-1, 2, 5])
print(jax_check_positive_even(x))
```

Python logical operators error when applied to JAX arrays of more than one
element, even without `jit`. This replicates NumPy's behavior.

```{code-cell}
:tags: [raises-exception]

print(python_check_positive_even(x))
```

## Python control flow without `jit`

The constraints on this page are about `jit` (and other transformations that
trace with abstract values, like `vmap`). Under plain `jax.grad`, ordinary
Python control flow works, with no `lax.cond` required, because `grad` traces
with concrete values. See
{ref}`jax-101-grad-control-flow` in the 101 docs.

## Next steps

This page completes the compilation thread of these docs: what `jit` buys
({doc}`jit`), its stages ({doc}`aot`), and control flow inside compiled
functions. Next, {doc}`placement` covers where arrays live, with the *mesh*
as JAX's unit of placement.
