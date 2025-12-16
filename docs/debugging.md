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

(debugging)=
# Introduction to debugging

<!--* freshness: { reviewed: '2024-05-10' } *-->

Do you have exploding gradients? Are NaNs making you gnash your teeth? Just want
to poke around the intermediate values in your computation? This section
introduces you to a set of built-in JAX debugging methods that you can use with
various JAX transformations.

**Summary:**

- Use {func}`jax.debug.print` to print values to stdout in `jax.jit`-,`jax.pmap`-, and `pjit`-decorated functions,
  and {func}`jax.debug.breakpoint` to pause execution of your compiled function to inspect values in the call stack.
- {mod}`jax.experimental.checkify` lets you add `jit`-able runtime error checking (e.g. out of bounds indexing) to your JAX
  code.
- JAX offers config flags and context managers that enable catching errors more easily. For example, enable the
  `jax_debug_nans` flag to automatically detect when NaNs are produced in `jax.jit`-compiled code and enable the
  `jax_disable_jit` flag to disable JIT-compilation.

## `jax.debug.print` for simple inspection

Here is a rule of thumb:

- Use {func}`jax.debug.print` for traced (dynamic) array values with {func}`jax.jit`, {func}`jax.vmap` and others.
- Use Python {func}`print` for static values, such as dtypes and array shapes.

Recall from {ref}`jit-compilation` that when transforming a function with {func}`jax.jit`,
the Python code is executed with abstract tracers in place of your arrays. Because of this,
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

To learn more about {func}`jax.debug.print` and its Sharp Bits, refer to {ref}`advanced-debugging`.

## `jax.debug.breakpoint` for `pdb`-like debugging

**Summary:** Use {func}`jax.debug.breakpoint` to pause the execution of your JAX program to inspect values.

To pause your compiled JAX program during certain points during debugging, you can use {func}`jax.debug.breakpoint`. The prompt is similar to Python `pdb`, and it allows you to inspect the values in the call stack. In fact, {func}`jax.debug.breakpoint` is an application of {func}`jax.debug.callback` that captures information about the call stack.

To print all available commands during a `breakpoint` debugging session, use the `help` command. (Full debugger commands, the Sharp Bits, its strengths and limitations are covered in {ref}`advanced-debugging`.)

Here is an example of what a debugger session might look like:

```{code-cell}
:tags: [skip-execution]

@jax.jit
def f(x):
  y, z = jnp.sin(x), jnp.cos(x)
  jax.debug.breakpoint()
  return y * z

f(2.) # ==> Pauses during execution
```

![JAX debugger](_static/debugger.gif)

For value-dependent breakpointing, you can use runtime conditionals like {func}`jax.lax.cond`:

```{code-cell}
def breakpoint_if_nonfinite(x):
  is_finite = jnp.isfinite(x).all()
  def true_fn(x):
    pass
  def false_fn(x):
    jax.debug.breakpoint()
  jax.lax.cond(is_finite, true_fn, false_fn, x)

@jax.jit
def f(x, y):
  z = x / y
  breakpoint_if_nonfinite(z)
  return z

f(2., 1.) # ==> No breakpoint
```

```{code-cell}
:tags: [skip-execution]

f(2., 0.) # ==> Pauses during execution
```

## `jax.debug.callback` for more control during debugging

Both {func}`jax.debug.print` and {func}`jax.debug.breakpoint` are implemented using
the more flexible {func}`jax.debug.callback`, which gives greater control over the
host-side logic executed via a Python callback.
It is compatible with {func}`jax.jit`, {func}`jax.vmap`, {func}`jax.grad` and other
transformations (refer to the {ref}`external-callbacks-flavors-of-callback` table in
{ref}`external-callbacks` for more information).

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

You can learn more about {func}`jax.debug.callback` and other kinds of JAX callbacks in {ref}`external-callbacks`.

Read more in [](debugging/print_breakpoint).

## Functional error checks with `jax.experimental.checkify`

**Summary:** Checkify lets you add `jit`-able runtime error checking (e.g. out of bounds indexing) to your JAX code. Use the `checkify.checkify` transformation together with the assert-like `checkify.check` function to add runtime checks to JAX code:

```python
from jax.experimental import checkify
import jax
import jax.numpy as jnp

def f(x, i):
  checkify.check(i >= 0, "index needs to be non-negative!")
  y = x[i]
  z = jnp.sin(y)
  return z

jittable_f = checkify.checkify(f)

err, z = jax.jit(jittable_f)(jnp.ones((5,)), -1)
print(err.get())
# >> index needs to be non-negative! (check failed at <...>:6 (f))
```

You can also use checkify to automatically add common checks:

```python
errors = checkify.user_checks | checkify.index_checks | checkify.float_checks
checked_f = checkify.checkify(f, errors=errors)

err, z = checked_f(jnp.ones((5,)), 100)
err.throw()
# ValueError: out-of-bounds indexing at <..>:7 (f)

err, z = checked_f(jnp.ones((5,)), -1)
err.throw()
# ValueError: index needs to be non-negative! (check failed at <…>:6 (f))

err, z = checked_f(jnp.array([jnp.inf, 1]), 0)
err.throw()
# ValueError: nan generated by primitive sin at <...>:8 (f)
```

Read more in [](debugging/checkify_guide).

### Throwing Python errors with JAX's debug flags

**Summary:** Enable the `jax_debug_nans` flag to automatically detect when NaNs are produced in `jax.jit`-compiled code (but not in `jax.pmap` or `jax.pjit`-compiled code) and enable the `jax_disable_jit` flag to disable JIT-compilation, enabling use of traditional Python debugging tools like `print` and `pdb`.

```python
import jax
jax.config.update("jax_debug_nans", True)

def f(x, y):
  return x / y

jax.jit(f)(0., 0.)  # ==> raises FloatingPointError exception!
```

Read more in [](debugging/flags).

## Next steps

Check out the {ref}`advanced-debugging` to learn more about debugging in JAX.

```{toctree}
:hidden:

debugging/print_breakpoint
debugging/checkify_guide
debugging/flags
```
