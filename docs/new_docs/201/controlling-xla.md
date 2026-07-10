---
nosearch: true
---

(jax-201-controlling-xla)=
# Controlling XLA from JAX

<!--* freshness: { reviewed: '2026-07-10' } *-->

JAX programs are compiled by [XLA](https://openxla.org/xla), and mostly you
can leave the compiler alone. When you can't, JAX offers two levels of
control: **compiler flags**, which steer how XLA compiles (globally or per
function), and **XLA metadata**, which annotates individual operations in the
compiled program — increasingly load-bearing for compiler-level tooling like
debuggers and scheduling hints.

(jax-201-compiler-flags)=
## XLA compiler flags

The [XLA](https://openxla.org/xla) compiler's behavior is controlled by
*flags*. There are two ways to set them: per compiled function, with
`jax.jit`'s `compiler_options` parameter — usually what you want — or
process-wide, with the `XLA_FLAGS` environment variable.

### Per function: `jit`'s `compiler_options`

{func}`jax.jit` accepts a `compiler_options` dictionary that applies only to
the compilation of that one function, leaving the rest of your program
untouched. Keys are XLA flag names without the `--` prefix, and values are
ordinary Python booleans, numbers, or strings:

```python
@jax.jit(compiler_options={
    "xla_embed_ir_in_executable": True,
    "xla_gpu_auto_spmd_partitioning_memory_budget_ratio": 0.5,
})
def f(x):
  return jnp.sqrt(x ** 2) + 1.

f(1.0)
```

Alongside XLA's debug-option flags, `compiler_options` also accepts XLA's
compilation-effort knobs, like `optimization_level`,
`memory_fitting_level`, and `exec_time_optimization_effort`:

```python
g = jax.jit(f, compiler_options={"exec_time_optimization_effort": 0.5})
```

Unrecognized keys or invalid values raise an error at compilation time, so
typos announce themselves:

```python
jax.jit(f, compiler_options={"not_a_real_flag": 1})(1.0)
# ==> JaxRuntimeError: INVALID_ARGUMENT: No such compile option: 'not_a_real_flag'
```

The same dictionary can be passed at the compile step when using the
ahead-of-time APIs ({doc}`aot`):

```python
compiled = jax.jit(f).lower(1.0).compile(
    compiler_options={"xla_embed_ir_in_executable": True})
```

### Process-wide: the `XLA_FLAGS` environment variable

To configure XLA for the whole process — including compilations you don't
control directly — set the `XLA_FLAGS` environment variable, with flags
separated by spaces:

```bash
XLA_FLAGS='--flag1=value1 --flag2=value2' python3 source.py
```

`XLA_FLAGS` is read when JAX initializes its backends, so it must be set
before then — in practice, before importing JAX. Changing it afterwards has
no effect.

For the flags themselves: XLA's flags are defined with their default values
in [xla/debug_options_flags.cc](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc),
with a complete list in [xla.proto](https://github.com/openxla/xla/blob/main/xla/xla.proto),
and XLA publishes a [guide to the key flags](https://openxla.org/xla/flags_guidance).

(jax-201-xla-metadata)=
## Attaching metadata to operations: `set_xla_metadata`

**Summary:** `set_xla_metadata` allows you to attach metadata to operations in
your JAX code. This metadata is passed down to the XLA compiler as
`frontend_attributes` and can be used to enable compiler-level tooling, such
as the XLA-TPU debugger.

You can use it in three ways:

1. Tag an individual operation by wrapping its output value
2. Tag a block of operations using a context manager
3. Tag all operations in a function using a decorator

**Warning:** `set_xla_metadata` is an experimental feature and its API is
subject to change.

### What is XLA Metadata?
When JAX transforms and compiles your code, it ultimately generates an XLA (Accelerated Linear Algebra) computation graph. Each operation in this graph can have associated metadata, specifically `frontend_attributes`. This metadata doesn't change the numerical result of the operation, but it can be used to signal special behavior to the compiler or runtime.

`set_xla_metadata` provides a way to attach this metadata directly from your JAX code. This is a powerful feature for low-level debugging and profiling.

### Usage
#### Tagging Individual Operations
Tagging an individual operation gives you precise control over which parts of your computation you want to inspect. To do this, you wrap the output (value) of an operation with `set_xla_metadata`. When wrapping a function with multiple operations within, only the final operation of said function will be tagged.

```python
import jax
import jax.numpy as jnp
from jax.experimental.xla_metadata import set_xla_metadata

### Tagging an individual operation
def value_tagging(x):
  y = jnp.sin(x)
  z = jnp.cos(x)
  return set_xla_metadata(y * z, breakpoint=True)

print(jax.jit(value_tagging).lower(1.0).as_text("hlo"))
```
Results in:
```
ENTRY main.5 {
  x.1 = f32[] parameter(0)
  sin.2 = f32[] sine(x.1)
  cos.3 = f32[] cosine(x.1)
  ROOT mul.4 = f32[] multiply(sin.2, cos.3), frontend_attributes={breakpoint="true"}
}
```
#### Tagging a Block of Code with a Context Manager or Decorator
If you want to apply the same metadata to a larger section of code, you can use `set_xla_metadata` as a context manager. All JAX operations within the `with` block will have the specified metadata attached.

```python
import jax
import jax.numpy as jnp
from jax.experimental.xla_metadata import set_xla_metadata

### Tagging a block of code
def context_tagging(x):
  with set_xla_metadata(_xla_log=True):
    y = jnp.sin(x)
    z = jnp.cos(y)
    return y * z

print(jax.jit(context_tagging).lower(1.0).as_text("hlo"))
```
Results in:
```
ENTRY main.5 {
  x.1 = f32[] parameter(0)
  sin.2 = f32[] sine(x.1), frontend_attributes={_xla_log="true"}
  cos.3 = f32[] cosine(sin.2), frontend_attributes={_xla_log="true"}
  ROOT mul.4 = f32[] multiply(sin.2, cos.3), frontend_attributes={_xla_log="true"}
}
```

If you want to tag all operations in a function, you can also use `set_xla_metadata` as a decorator:

```python
import jax
import jax.numpy as jnp
from jax.experimental.xla_metadata import set_xla_metadata

### Tagging with a decorator
@set_xla_metadata(_xla_log=True)
@jax.jit
def decorator_tagging(x):
  y = jnp.sin(x)
  z = jnp.cos(y)
  return y * z

print(decorator_tagging.lower(1.0).as_text("hlo"))
```
This will result in the same HLO as above.

### Interaction with JAX Transformations
`set_xla_metadata` utilizes either a `XlaMetadataContextManager` or JAX `primitive` depending on use-case and is compatible with JAX's transformations like `jit`, `vmap`, and `grad`.
*   **`vmap`**: When you `vmap` a function containing `set_xla_metadata`, the metadata will be applied to all of the relevant batched operations.
*   **`grad`**:
    1. When tagging a block of operations with the **context manager** `with set_xla_metadata(...):`, the metadata is applied to both the forward pass and backward pass of the operations within it.
    2. Tagging **individual ops** with `set_xla_metadata()` currently only applies to the forward pass of a function. To tag individual operations generated by the backward pass (i.e., the gradient computation), a simple `custom_vjp` can be used:
        ```python
        import jax
        import jax.numpy as jnp
        from jax.experimental.xla_metadata import set_xla_metadata

        def fn(x):
            y = jnp.sin(x)
            z = jnp.cos(x)
            return y * z

        metadata = {"example": "grad_tagging"}

        # --- Define Custom VJP to tag gradients ---
        @jax.custom_vjp
        def wrapped_fn(x):
            return fn(x)

        def fwd(*args):
            primal_out, vjp_fn = jax.vjp(fn, *args)
            return primal_out, vjp_fn

        def bwd(vjp_fn, cts_in):
            cts_out = vjp_fn(cts_in)
            cts_out = set_xla_metadata(cts_out, **metadata)
            return cts_out

        wrapped_fn.defvjp(fwd, bwd)
        # ------

        print(jax.jit(jax.grad(wrapped_fn)).lower(jnp.array(3.0)).as_text("hlo"))
        ```
        Results in:
        ```
        ENTRY main.10 {
          x.1 = f32[] parameter(0)
          sin.2 = f32[] sine(x.1)
          neg.6 = f32[] negate(sin.2)
          sin.5 = f32[] sine(x.1)
          mul.7 = f32[] multiply(neg.6, sin.5)
          cos.4 = f32[] cosine(x.1)
          cos.3 = f32[] cosine(x.1)
          mul.8 = f32[] multiply(cos.4, cos.3)
          ROOT add_any.9 = f32[] add(mul.7, mul.8), frontend_attributes={example="grad_tagging"}
        }
        ```
##### Strengths and Limitations of `set_xla_metadata`

###### Strengths
*   **Variable Control:** Allows you to target individual operations or blocks of operations.
*   **Non-Intrusive:** Does not change the numerical output or fusion behavior of your program.
*   **Enables Powerful Tooling:** Unlocks the potential for sophisticated debugging and analysis at the compiler level.

###### Limitations
*   **Attributes may be lost:** While it's intended for XLA metadata to be maintained throughout transformations and HLO optimizations, certain edge-cases may result in the metadata being lost.
*   **Forward-pass only:** Metadata is not currently automatically propagated to gradients <u>when tagging individual operations</u> in the backward pass. A `custom_vjp` must be used in order to tag gradients in this case. See above for an example.
*   **Liable to change**: `set_xla_metadata` is an experimental feature and its API is subject to change.

