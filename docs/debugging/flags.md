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

(debugging-flags)=
# JAX debugging flags

<!--* freshness: { reviewed: '2025-10-28' } *-->

JAX offers flags and context managers that enable catching errors more easily.

## `jax_debug_nans` configuration option and context manager

**Summary:** Enable the `jax_debug_nans` flag to automatically detect when NaNs are produced in `jax.jit`-compiled code.

`jax_debug_nans` is a JAX flag that when enabled, will cause computations to error-out immediately on production of a NaN. Switching this option on adds a NaN check to every floating point type value produced by XLA. That means values are pulled back to the host and checked as ndarrays for every primitive operation not under an `@jax.jit`.

For code under an `@jax.jit`, the output of every `@jax.jit` function is checked and if a NaN is present it will re-run the function in de-optimized op-by-op mode, effectively removing one level of `@jax.jit` at a time.

There could be tricky situations that arise, like NaNs that only occur under a `@jax.jit` but don't get produced in de-optimized mode. In that case you'll see a warning message print out but your code will continue to execute.

If the NaNs are being produced in the backward pass of a gradient evaluation, when an exception is raised several frames up in the stack trace you will be in the backward_pass function, which is essentially a simple jaxpr interpreter that walks the sequence of primitive operations in reverse.

### Usage

If you want to trace where NaNs are occurring in your functions or gradients, you can turn on the NaN-checker by doing one of:
* running your code inside the `jax.debug_nans` context manager, using `with jax.debug_nans(True):`;
* setting the `JAX_DEBUG_NANS=True` environment variable;
* adding `jax.config.update("jax_debug_nans", True)` near the top of your main file;
* adding `jax.config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--jax_debug_nans=True`;

### Example(s)

```{code-cell}
import jax
import jax.numpy as jnp
import traceback
jax.config.update("jax_debug_nans", True)

def f(x):
  w = 3 * jnp.square(x)
  return jnp.log(-w)

# The stack trace is very long so only print a couple lines.
try:
  f(5.)
except FloatingPointError as e:
  print(traceback.format_exc(limit=2))
```

The NaN generated was caught. By running `%debug`, we can get a post-mortem debugger. This also works with functions under `@jax.jit`, as the example below shows.

```{code-cell}
:tags: [raises-exception]

jax.jit(f)(5.)
```

When this code sees a NaN in the output of an `@jax.jit` function, it calls into the de-optimized code, so we still get a clear stack trace. And we can run a post-mortem debugger with `%debug` to inspect all the values to figure out the error.

The `jax.debug_nans` context manager can be used to activate/deactivate NaN debugging. Since we activated it above with `jax.config.update`, let's deactivate it:

```{code-cell}
with jax.debug_nans(False):
  print(jax.jit(f)(5.))
```

#### Strengths and limitations of `jax_debug_nans`
##### Strengths
* Easy to apply
* Precisely detects where NaNs were produced
* Throws a standard Python exception and is compatible with PDB postmortem

##### Limitations
* Re-running functions eagerly can be slow. You shouldn't have the NaN-checker on if you're not debugging, as it can introduce lots of device-host round-trips and performance regressions.
* Errors on false positives (e.g. intentionally created NaNs)

## `jax_debug_infs` configuration option and context manager

`jax_debug_infs` works similarly to `jax_debug_nans`. `jax_debug_infs` often needs to be combined with `jax_disable_jit`, since Infs might not cascade to the output like NaNs. Alternatively, `jax.experimental.checkify` may be used to find Infs in intermediates.

Full documentation of `jax_debug_infs` is forthcoming.
<!-- https://github.com/jax-ml/jax/issues/17722 -->

## `jax_disable_jit` configuration option and context manager

**Summary:** Enable the `jax_disable_jit` flag to disable JIT-compilation, enabling use of traditional Python debugging tools like `print` and `pdb`

`jax_disable_jit` is a JAX flag that when enabled, disables JIT-compilation throughout JAX (including in control flow functions like `jax.lax.cond` and `jax.lax.scan`).

### Usage

You can disable JIT-compilation by:
* setting the `JAX_DISABLE_JIT=True` environment variable;
* adding `jax.config.update("jax_disable_jit", True)` near the top of your main file;
* adding `jax.config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--jax_disable_jit=True`;

### Examples

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

#### Strengths and limitations of `jax_disable_jit`

##### Strengths
* Easy to apply
* Enables use of Python's built-in `breakpoint` and `print`
* Throws standard Python exceptions and is compatible with PDB postmortem

##### Limitations
* Running functions without JIT-compilation can be slow

## `jax_debug_infs` configuration option and context manager

**Summary:** Enable the `jax_debug_infs` flag to automatically detect when infinities (infs) are produced in `jax.jit`-compiled code.

`jax_debug_infs` is a JAX flag that, when enabled, raises an error when an infinite value is detected in computations. Similar to `jax_debug_nans`, it has special handling for JIT-compiled functions—when an infinite output is detected, the function is re-run eagerly (without compilation) to pinpoint the specific primitive that caused the issue.

### Usage

To detect infinite values in your functions, enable the flag using one of the following methods:
* Set the `JAX_DEBUG_INFS=True` environment variable;
* Add `jax.config.update("jax_debug_infs", True)` near the top of your main file;
* Add `jax.config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--jax_debug_infs=True`;

### Example

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_debug_infs", True)

def f(x):
  return 1.0 / x

jax.jit(f)(0.)  # ==> raises FloatingPointError exception!
```

#### Strengths and limitations of `jax_debug_infs`

##### Strengths
* Easy to apply
* Precisely detects where infinities were produced
* Throws a standard Python exception and is compatible with PDB postmortem

##### Limitations
* Not compatible with `jax.pmap` or `jax.pjit`
* Re-running functions eagerly can be slow
* Errors on false positives (e.g., intentionally created infinities)

### Additional Note:
`jax_debug_infs` is often used in combination with `jax_disable_jit` since infinities might not always propagate to the output like NaNs do. Alternatively, `checkify` can be used to detect infinities in intermediate computations.
