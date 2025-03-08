# JAX debugging flags  

<!--* freshness: { reviewed: '2024-04-11' } *-->

JAX offers flags and context managers that enable catching errors more easily.
  
## `jax_debug_nans` configuration option and context manager

**Summary:** Enable the `jax_debug_nans` flag to automatically detect when NaNs are produced in `jax.jit`-compiled code (but not in `jax.pmap` or `jax.pjit`-compiled code).

`jax_debug_nans` is a JAX flag that when enabled, automatically raises an error when a NaN is detected. It has special handling for JIT-compiled -- when a NaN output is detected from a JIT-ted function, the function is re-run eagerly (i.e. without compilation) and will throw an error at the specific primitive that produced the NaN.

### Usage

If you want to trace where NaNs are occurring in your functions or gradients, you can turn on the NaN-checker by:
* setting the `JAX_DEBUG_NANS=True` environment variable;
* adding `jax.config.update("jax_debug_nans", True)` near the top of your main file;
* adding `jax.config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--jax_debug_nans=True`;

### Example(s)

```python
import jax
jax.config.update("jax_debug_nans", True)

def f(x, y):
  return x / y
jax.jit(f)(0., 0.)  # ==> raises FloatingPointError exception!
```

#### Strengths and limitations of `jax_debug_nans`
##### Strengths
* Easy to apply
* Precisely detects where NaNs were produced
* Throws a standard Python exception and is compatible with PDB postmortem

##### Limitations
* Not compatible with `jax.pmap` or `jax.pjit`
* Re-running functions eagerly can be slow
* Errors on false positives (e.g. intentionally created NaNs)

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
* Not compatible with `jax.pmap` or `jax.pjit`
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
