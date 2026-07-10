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
flow and logical operators just like NumPy code. Using control flow and
logical operators with `jit` is more complicated.

In a nutshell, Python control flow and logical operators are evaluated at JIT
compile time, such that the compiled function represents a single path
through the [control flow graph](https://en.wikipedia.org/wiki/Control-flow_graph)
(logical operators affect the path via short-circuiting). If the path depends
on the values of the inputs, the function (by default) cannot be JIT
compiled. The path may depend on the shape or dtype of the inputs, and the
function is re-compiled every time it is called on an input with a new shape
or dtype.

```{code-cell}
from jax import grad, jit
import jax.numpy as jnp
```

For example, this works:

```{code-cell}
@jit
def f(x):
  for i in range(3):
    x = 2 * x
  return x

print(f(3))
```

So does this:

```{code-cell}
@jit
def g(x):
  y = 0.
  for i in range(x.shape[0]):
    y = y + x[i]
  return y

print(g(jnp.array([1., 2., 3.])))
```

But this doesn't, at least by default:

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
  return (x > 0) and (x < 3)

# This will fail!
g(2)
```

__What gives!?__

When we `jit`-compile a function, we usually want to compile a version of the
function that works for many different argument values, so that we can cache
and reuse the compiled code. That way we don't have to re-compile on each
function evaluation.

For example, if we evaluate an `@jit` function on the array
`jnp.array([1., 2., 3.], jnp.float32)`, we might want to compile code that we
can reuse to evaluate the function on `jnp.array([4., 5., 6.], jnp.float32)`
to save on compile time.

To get a view of your Python code that is valid for many different argument
values, JAX traces it with tracers that carry only the JAX type — for
example, `float32[3]`, standing for *every* array of that shape and dtype.
(Internally, this abstract value is a `ShapedArray`, which is what
{func}`jax.typeof` returns.) Tracing on the JAX type gives a view of the
function that can be reused for any concrete value with that type: that's how
we save on compile time.

But there's a tradeoff here: if we trace a Python function on a `float32[]`
input that isn't committed to a specific concrete value, when we hit a line
like `if x < 3`, the expression `x < 3` evaluates to an abstract `bool[]`
that represents the set `{True, False}`. When Python attempts to coerce that
to a concrete `True` or `False`, we get an error: we don't know which branch
to take, and can't continue tracing! The tradeoff is that with higher levels
of abstraction we gain a more general view of the Python code (and thus save
on re-compilations), but we require more constraints on the Python code to
complete the trace.

The good news is that you can control this tradeoff yourself. By having `jit`
trace on more refined abstract values, you can relax the traceability
constraints. For example, using the `static_argnames` (or `static_argnums`)
argument to `jit` (see {ref}`jax-201-jit-static-arguments`), we can specify
to trace on concrete values of some arguments. Here's that example function
again:

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

In effect, the loop gets statically unrolled. JAX can also trace at _higher_
levels of abstraction, like `Unshaped`, but that's not currently the default
for any transformation.

️⚠️ **functions with argument-__value__ dependent shapes**

These control-flow issues also come up in a more subtle way: numerical
functions we want to __jit__ can't specialize the shapes of internal arrays
on argument _values_ (specializing on argument __shapes__ is ok). As a
trivial example, let's make a function whose output happens to depend on the
input variable `length`.

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

`static_argnames` can be handy if `length` in our example rarely changes, but
it would be disastrous if it changed a lot!

Lastly, if your function has global side-effects, JAX's tracer can cause
weird things to happen. A common gotcha is trying to print arrays inside
__jit__'d functions:

```{code-cell}
@jit
def f(x):
  print(x)
  y = 2 * x
  print(y)
  return y
f(2)
```

(The prints show tracers, not values, and they run at trace time only. For
printing that shows runtime values on every call, use {func}`jax.debug.print`
— see {doc}`debugging`.)

## Structured control flow primitives

There are more options for control flow in JAX. Say you want to avoid
re-compilations but still want to use control flow that's traceable, and that
avoids un-rolling large loops. Then you can use these 4 structured control
flow primitives:

 - `lax.cond`  _differentiable_
 - `lax.while_loop` __fwd-mode-differentiable__
 - `lax.fori_loop` __fwd-mode-differentiable__ in general; __fwd and
   rev-mode differentiable__ if endpoints are static.
 - `lax.scan` _differentiable_

### `cond`

python equivalent:

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

Unlike a Python `if`, the predicate here can be a traced value — the choice
of branch happens on the device, at run time, inside the compiled program.

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

python equivalent:

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

### `fori_loop`

python equivalent:

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

The workhorse of the four is {func}`jax.lax.scan`: a loop with a fixed number
of iterations that carries state from step to step, optionally consuming a
per-step slice of an input array and stacking per-step outputs.

python equivalent:

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
how many iterations run — long training loops and sequence models compile in
constant time instead of time proportional to the loop length. And unlike
`while_loop`, `scan` supports both forward- and reverse-mode
differentiation, which is why it's the standard way to express a training
loop's steps or an RNN's time axis inside `jit`.

For fine-tuning that compile-time/run-time trade, `scan` takes an `unroll`
parameter: `unroll=k` makes each iteration of the compiled loop perform `k`
steps of the scan, and `unroll=True` unrolls the loop entirely. Larger unroll
amounts give XLA more opportunity to fuse and parallelize across steps, at
the cost of compile time and program size — often worthwhile when the body
is small relative to per-iteration overhead. (`lax.fori_loop` accepts the
same parameter.)

## Logical operators

`jax.numpy` provides `logical_and`, `logical_or`, and `logical_not`, which
operate element-wise on arrays and can be evaluated under `jit` without
recompiling. Like their NumPy counterparts, the binary operators do not short
circuit. Bitwise operators (`&`, `|`, `~`) can also be used with `jit`.

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
  # `logical_and` does not short circuit, so `x > 0` is always evaluated.
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
Python control flow just works — no `lax.cond` required — because `grad`
traces with concrete values. See
{ref}`grad works with Python control flow <jax-101-transformations>` in the
101 docs.
