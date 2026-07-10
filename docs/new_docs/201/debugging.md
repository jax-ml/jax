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

(jax-201-debugging)=
# Debugging runtime values

<!--* freshness: { reviewed: '2026-07-09' } *-->

Do you have exploding gradients? Are NaNs making you gnash your teeth? Just want
to poke around the intermediate values in your computation? This section
introduces you to a set of built-in JAX debugging methods that you can use with
various JAX transformations.

**Summary:**

- Use {func}`jax.debug.print` to print values to stdout in `jax.jit`-, `jax.vmap`-, and
  other transformation-decorated functions.
- JAX offers config flags and context managers that enable catching errors more easily. For example, enable the
  `jax_debug_nans` flag to automatically detect when NaNs are produced in `jax.jit`-compiled code and enable the
  `jax_disable_jit` flag to disable JIT-compilation.

## `jax.debug.print` for simple inspection

Here is a rule of thumb:

- Use {func}`jax.debug.print` for traced (dynamic) array values with {func}`jax.jit`, {func}`jax.vmap` and others.
- Use Python {func}`print` for static values, such as dtypes and array shapes.

Recall from {doc}`jit` (and {ref}`jax-101-tracing`) that when transforming a function
with {func}`jax.jit`, the Python code is executed with abstract tracers in place of
your arrays. Because of this,
the Python {func}`print` function will only print this tracer value:

```{code-cell}
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
  print("print(x) ->", x)
  y = jnp.sin(x)
  print("print(y) ->", y)
  return y

result = f(2.)
```

Python's `print` executes at trace-time, before the runtime values exist.
If you want to print the actual runtime values, you can use {func}`jax.debug.print`:

```{code-cell}
@jax.jit
def f(x):
  jax.debug.print("jax.debug.print(x) -> {x}", x=x)
  y = jnp.sin(x)
  jax.debug.print("jax.debug.print(y) -> {y}", y=y)
  return y

result = f(2.)
```

Similarly, within {func}`jax.vmap`, using Python's `print` will only print the tracer;
to print the values being mapped over, use {func}`jax.debug.print`:

```{code-cell}
def f(x):
  jax.debug.print("jax.debug.print(x) -> {}", x)
  y = jnp.sin(x)
  jax.debug.print("jax.debug.print(y) -> {}", y)
  return y

xs = jnp.arange(3.)

result = jax.vmap(f)(xs)
```

Here's the result with {func}`jax.lax.map`, which is a sequential map rather than a
vectorization:

```{code-cell}
result = jax.lax.map(f, xs)
```

Notice the order is different, as {func}`jax.vmap` and {func}`jax.lax.map` compute the same results in different ways. When debugging, the evaluation order details are exactly what you may need to inspect.

Below is an example with {func}`jax.grad`, where {func}`jax.debug.print` only prints the forward pass. In this case, the behavior is similar to Python's {func}`print`, but it's consistent if you apply {func}`jax.jit` during the call.

```{code-cell}
def f(x):
  jax.debug.print("jax.debug.print(x) -> {}", x)
  return x ** 2

result = jax.grad(f)(1.)
```

Sometimes, when the arguments don't depend on one another, calls to {func}`jax.debug.print` may print them in a different order when staged out with a JAX transformation. If you need the original order, such as `x: ...` first and then `y: ...` second, add the `ordered=True` parameter.

For example:

```{code-cell}
@jax.jit
def f(x, y):
  jax.debug.print("jax.debug.print(x) -> {}", x, ordered=True)
  jax.debug.print("jax.debug.print(y) -> {}", y, ordered=True)
  return x + y

f(1, 2)
```

(The reordering happens because the compiler receives a *functional*
representation of the staged-out computation, in which the imperative order
of your Python statements is gone and only data dependence remains — invisible
for pure code, visible once printing enters the picture.)

### Sharp bits of `jax.debug.print`

A few cautions worth knowing before you sprinkle `jax.debug.print`
everywhere:

**Format strings are deferred.** The format string can't be an f-string:
f-strings format immediately, while `jax.debug.print` needs to delay
formatting until the runtime value exists. Pass values as arguments, as in
the examples above.

**Printing on the backward pass takes an extra step.** As shown above,
`jax.debug.print` fires on the forward pass only. To see gradients, wrap a
print in a {func}`jax.custom_vjp`:

```{code-cell}
@jax.custom_vjp
def print_grad(x):
  return x

def print_grad_fwd(x):
  return x, None

def print_grad_bwd(_, x_grad):
  jax.debug.print("x_grad: {}", x_grad)
  return (x_grad,)

print_grad.defvjp(print_grad_fwd, print_grad_bwd)

def f(x):
  x = print_grad(x)
  return x * 2.

jax.grad(f)(1.)
```

**Debug prints perturb the computation.** Adding `jax.debug.print` changes
the program XLA compiles: a value that would have lived inside a fused kernel
must be materialized so it can be printed, which can change performance,
memory usage, and even numerics (different fusions can round differently).
Keep this in mind when debugging numerical mysteries — the act of looking can
disturb the thing you're looking at. Printing sharded values likewise forces
synchronization to gather the value.

**Prints are asynchronous.** Like the computations they're embedded in
({ref}`jax-201-async-dispatch`), debug prints can arrive after the enclosing
function has returned — even after `block_until_ready()`, which waits for
values, not side effects. To wait for outstanding prints, use
{func}`jax.effects_barrier`:

```{code-cell}
@jax.jit
def f(x):
  jax.debug.print("x: {}", x)
  return x

f(2.).block_until_ready()
jax.effects_barrier()
```

## `jax.debug.callback` for more control during debugging

{func}`jax.debug.print` is implemented using the more flexible
{func}`jax.debug.callback`, which gives greater control over the host-side
logic executed via a Python callback.
It is compatible with {func}`jax.jit`, {func}`jax.vmap`, {func}`jax.grad` and other
transformations (refer to the {ref}`jax-201-callbacks-flavors` table in
{doc}`callbacks` for more information).

For example:

```{code-cell}
import logging

def log_value(x):
  logging.warning(f'Logged value: {x}')

@jax.jit
def f(x):
  jax.debug.callback(log_value, x)
  return x

f(1.0);
```

This callback is compatible with other transformations, including {func}`jax.vmap` and {func}`jax.grad`:

```{code-cell}
x = jnp.arange(5.0)
jax.vmap(f)(x);
```

```{code-cell}
jax.grad(f)(1.0);
```

This can make {func}`jax.debug.callback` useful for general-purpose debugging.

You can learn more about {func}`jax.debug.callback` and other kinds of JAX callbacks in {doc}`callbacks`.


## Throwing Python errors with JAX's debug flags

JAX offers flags and context managers that enable catching errors more
easily: `jax_debug_nans` to automatically detect when NaNs are produced in
`jax.jit`-compiled code, and `jax_disable_jit` to disable JIT-compilation,
enabling the use of traditional Python debugging tools like `print` and
`pdb`.

### `jax_debug_nans`

`jax_debug_nans` is a JAX flag that when enabled, will cause computations to
error-out immediately on production of a NaN. Switching this option on adds a
NaN check to every floating point type value produced by XLA. That means
values are pulled back to the host and checked as ndarrays for every
primitive operation not under an `@jax.jit`.

For code under an `@jax.jit`, the output of every `@jax.jit` function is
checked and if a NaN is present it will re-run the function in de-optimized
op-by-op mode, effectively removing one level of `@jax.jit` at a time.

There could be tricky situations that arise, like NaNs that only occur under
a `@jax.jit` but don't get produced in de-optimized mode. In that case you'll
see a warning message print out but your code will continue to execute.

If the NaNs are being produced in the backward pass of a gradient evaluation,
when an exception is raised several frames up in the stack trace you will be
in the `backward_pass` function, which is essentially a simple jaxpr
interpreter that walks the sequence of primitive operations in reverse.

To turn on the NaN-checker, do one of:

* run your code inside the `jax.debug_nans` context manager, using
  `with jax.debug_nans(True):`;
* set the `JAX_DEBUG_NANS=True` environment variable;
* add `jax.config.update("jax_debug_nans", True)` near the top of your main
  file;
* add `jax.config.parse_flags_with_absl()` to your main file, then set the
  option using a command-line flag like `--jax_debug_nans=True`.

For example:

```{code-cell}
import traceback
jax.config.update("jax_debug_nans", True)

def f(x):
  w = 3 * jnp.square(x)
  return jnp.log(-w)

# The stack trace is very long, so only print a couple of lines.
try:
  f(5.)
except FloatingPointError as e:
  print(traceback.format_exc(limit=2))
```

The NaN generated was caught, with an ordinary Python exception — so running
`%debug` in IPython gives a post-mortem debugger at the fault. This also
works with functions under `@jax.jit`:

```{code-cell}
:tags: [raises-exception]

jax.jit(f)(5.)
```

When a NaN appears in the output of an `@jax.jit` function, JAX re-runs the
de-optimized code, so we still get a clear stack trace pointing at the
producing operation. The `jax.debug_nans` context manager can scope the
checking; since we activated it globally above, let's deactivate it:

```{code-cell}
with jax.debug_nans(False):
  print(jax.jit(f)(5.))
```

```{code-cell}
jax.config.update("jax_debug_nans", False)  # back off for the rest of this page
```

**Strengths:** easy to apply; precisely detects where NaNs were produced;
throws a standard Python exception and is compatible with PDB postmortem.

**Limitations:** re-running functions eagerly can be slow, and the constant
device-to-host checks cost real performance — don't leave the NaN-checker on
when you're not debugging. It also errors on false positives, e.g.
intentionally created NaNs.

### `jax_debug_infs`

`jax_debug_infs` works similarly to `jax_debug_nans`. It often needs to be
combined with `jax_disable_jit`, since Infs might not cascade to the output
the way NaNs do.

### `jax_disable_jit`

`jax_disable_jit` is a JAX flag that when enabled, disables JIT-compilation
throughout JAX (including in control flow functions like `jax.lax.cond` and
`jax.lax.scan`). With compilation out of the picture, your function is plain
Python running eagerly, so all the ordinary tools work: `print`, `pdb`,
Python's built-in `breakpoint()`.

You can disable JIT-compilation by:

* running your code inside the `jax.disable_jit` context manager, using
  `with jax.disable_jit():`;
* setting the `JAX_DISABLE_JIT=True` environment variable;
* adding `jax.config.update("jax_disable_jit", True)` near the top of your
  main file;
* adding `jax.config.parse_flags_with_absl()` to your main file, then setting
  the option using a command-line flag like `--jax_disable_jit=True`.

For example:

```python
import jax
jax.config.update("jax_disable_jit", True)

def f(x):
  y = jnp.log(x)
  if jnp.isnan(y):
    breakpoint()
  return y

jax.jit(f)(-2.)  # ==> Enters PDB breakpoint!
```

**Strengths:** easy to apply; enables Python's built-in `breakpoint` and
`print`; throws standard Python exceptions and is compatible with PDB
postmortem.

**Limitations:** running functions without JIT-compilation can be slow.

```{warning}
These flags are best suited to single-process development, and don't work
well in multi-controller (multi-process) JAX ({doc}`/multi_process`).
Raising a Python error on one process but not the others — say, when only
one process's shard produces a NaN under `jax_debug_nans` — breaks the
assumption that every process runs the same program in lockstep, and the
usual symptom is the remaining processes hanging in a collective.
```

## Next steps

Check out {doc}`slow-compilation` if the thing that needs debugging is
tracing or compile time itself. For the mechanism underlying
`jax.debug.print` and `jax.debug.callback`, see {doc}`callbacks`.
