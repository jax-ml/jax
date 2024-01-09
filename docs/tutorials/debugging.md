---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(debugging)=
# Debugging 101

This tutorial introduces you to a set of built-in JAX debugging methods — {func}`jax.debug.print`, {func}`jax.debug.breakpoint`, and {func}`jax.debug.callback` — that you can use with various JAX transformations.

Let's begin with {func}`jax.debug.print`.

## JAX `debug.print` for high-level debugging

**TL;DR** Here is a rule of thumb:

- Use {func}`jax.debug.print` for traced (dynamic) array values with {func}`jax.jit`, {func}`jax.vmap` and others.
- Use Python `print` for static values, such as dtypes and array shapes.

With some JAX transformations, such as {func}`jax.grad` and {func}`jax.vmap`, you can use Python’s built-in `print` function to print out numerical values. However, with {func}`jax.jit` for example, you need to use {func}`jax.debug.print`, because those transformations delay numerical evaluation.

Below is a basic example with {func}`jax.jit`:

```{code-cell}
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    jax.debug.print("This is `jax.debug.print` of x {x}", x=x)
    y = jnp.sin(x)
    jax.debug.print("This is `jax.debug.print` of y {y} 🤯", y=y)
    return y

f(2.)
```

{func}`jax.debug.print` can reveal the information about how computations are evaluated.

Here's an example with {func}`jax.vmap`:

```{code-cell}
def f(x):
    jax.debug.print("This is `jax.debug.print` of x: {}", x)
    y = jnp.sin(x)
    jax.debug.print("This is `jax.debug.print` of y: {}", y)
    return y

xs = jnp.arange(3.)

jax.vmap(f)(xs)
```

Here's an example with {func}`jax.lax.map`:

```{code-cell}
jax.lax.map(f, xs)
```

Notice the order is different, as {func}`jax.vmap` and {func}`jax.lax.map` compute the same results in different ways. When debugging, the evaluation order details are exactly what you may need to inspect.

Below is an example with {func}`jax.grad`, where {func}`jax.debug.print` only prints the forward pass. In this case, the behavior is similar to Python's `print`, but it's consistent if you apply {func}`jax.jit` during the call.

```{code-cell}
def f(x):
    jax.debug.print("This is `jax.debug.print` of x: {}", x)
    return x ** 2

jax.grad(f)(1.)
```

Sometimes, when the arguments don't depend on one another, calls to {func}`jax.debug.print` may print them in a different order when staged out with a JAX transformation. If you need the original order, such as `x: ...` first and then `y: ...` second, add the `ordered=True` parameter.

For example:

```{code-cell}
@jax.jit
def f(x, y):
    jax.debug.print("This is `jax.debug.print of x: {}", x, ordered=True)
    jax.debug.print("This is `jax.debug.print of y: {}", y, ordered=True)
    return x + y
```

To learn more about {func}`jax.debug.print` and its Sharp Bits, refer to {ref}`advanced-debugging`.


## JAX `debug.breakpoint` for `pdb`-like debugging

**TL;DR** Use {func}`jax.debug.breakpoint` to pause the execution of your JAX program to inspect values.

To pause your compiled JAX program during certain points during debugging, you can use {func}`jax.debug.breakpoint`. The prompt is similar to Python `pdb`, and it allows you to inspect the values in the call stack. In fact, {func}`jax.debug.breakpoint` is an application of {func}`jax.debug.callback` that captures information about the call stack.

To print all available commands during a `breakpoint` debugging session, use the `help` command. (Full debugger commands, the Sharp Bits, its strengths and limitations are covered in {ref}`advanced-debugging`.)

Example:

```{code-cell}
:tags: [raises-exception]

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
f(2., 0.) # ==> Pauses during execution
```

![JAX debugger](../_static/debugger.gif)

## JAX `debug.callback` for more control during debugging

As mentioned in the beginning, {func}`jax.debug.print` is a small wrapper around {func}`jax.debug.callback`. The {func}`jax.debug.callback` method allows you to have greater control over string formatting and the debugging output, like printing or plotting. It is compatible with {func}`jax.jit`, {func}`jax.vmap`, {func}`jax.grad` and other transformations (refer to the {ref}`external-callbacks-flavors-of-callback` table in {ref]`external-callbacks` for more information).

For example:

```{code-cell}
def log_value(x):
  print("log:", x)

@jax.jit
def f(x):
  jax.debug.callback(log_value, x)
  return x

f(1.0);
```

This callback is compatible with {func}`jax.vmap` and {func}`jax.grad`:

```{code-cell}
x = jnp.arange(5.0)
jax.vmap(f)(x);
```

```{code-cell}
jax.grad(f)(1.0);
```

This can make {func}`jax.debug.callback` useful for general-purpose debugging.

You can learn more about different flavors of JAX callbacks in {ref}`external-callbacks-flavors-of-callback` and {ref}`external-callbacks-exploring-debug-callback`.

## Next steps

Check out the {ref}`advanced-debugging` to learn more about debugging in JAX.
