# JAX debugging flags  

<!--* freshness: { reviewed: '2024-04-11' } *-->

JAX offers flags and context managers that enable catching errors more easily.
  
### jax_debug_nans Configuration Option and Context Manager

**Summary**: Enable the `jax_debug_nans` flag to automatically detect when NaNs are produced in jax.jit-compiled code (but not in `jax.pmap` or `jax.pjit-compiled` code). However, this flag does not detect `inf` (infinity) values.

`jax_debug_nans` is a JAX flag that, when enabled, automatically raises an error when a NaN is detected. It has special handling for JIT-compiled functions—when a NaN output is detected in a JIT-compiled function, the function is re-run eagerly (i.e., without compilation) and throws an error at the specific primitive that produced the NaN.


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
#### Important Note: `jax_debug_nans` Does Not Detect `inf`
 While `jax_debug_nans` helps catch `NaN` values, it does not detect `inf` values (e.g., from division by zero or large exponentials). To handle `inf` values, consider:

  Using `jax_disable_jit` to inspect computations interactively.
  Using `jax.checkify` to validate both `NaN` and `inf` values at runtime.

  ### Example using `jax.checkify`
 
```python
import jax
import jax.numpy as jnp
from jax.experimental import checkify

checkify_fn = checkify.checkify(lambda x: jnp.exp(x))  # Wrap function
f_checkified = checkify_fn()
(issues, result) = f_checkified(1000.0)

print(issues)  # Warns about 'inf' values
print(result)  # Output: inf

```


#### Strengths and limitations of `jax_debug_nans`
##### Strengths
* Easy to apply
* Precisely detects where NaNs were produced
* Throws a standard Python exception and is compatible with PDB postmortem

##### Limitations
* Not compatible with `jax.pmap` or `jax.pjit`
* Does not detect `inf` values (use `jax.checkify`instead)
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
