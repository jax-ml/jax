---
nosearch: true
---

(jax-201-controlling-xla)=
# Controlling XLA from JAX

<!--* freshness: { reviewed: '2026-07-15' } *-->

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
## Attaching metadata to operations: `xla_metadata_call`

Each operation in a compiled XLA program can carry *metadata*: string-valued
`frontend_attributes` that don't change what the operation computes, but that
compiler-level tooling — the XLA-TPU debugger, fusion control, scheduling
hints — can read. JAX's interface for attaching metadata is
`jax.experimental.xla_metadata`, and the recommended entry point is
`xla_metadata_call`.

These APIs are experimental, and subject to change.

### Tagging a function: `xla_metadata_call`

`xla_metadata_call` wraps a function so that the operations it performs carry
the given metadata, passed as keyword arguments:

```python
import jax
import jax.numpy as jnp
from jax.experimental.xla_metadata import xla_metadata_call

@xla_metadata_call(tag="my_block")
def block(x):
  y = jnp.sin(x)
  return y * jnp.cos(x)

@jax.jit
def f(x):
  return block(x) + 1.

print(f.lower(1.0).as_text("hlo"))
```

The wrapped function is staged out as its own subcomputation, and the call
into it carries the metadata:

```
xla_metadata_call.1 {
  Arg_0.1 = f32[] parameter(0)
  sin.1 = f32[] sine(Arg_0.1)
  cos.1 = f32[] cosine(Arg_0.1)
  ROOT mul.1 = f32[] multiply(sin.1, cos.1)
}

ENTRY main.2 {
  x.1 = f32[] parameter(0)
  xla_metadata_call.1 = f32[] call(x.1), to_apply=xla_metadata_call.1, frontend_attributes={tag="my_block"}
  constant.1 = f32[] constant(1)
  ROOT add.1 = f32[] add(xla_metadata_call.1, constant.1)
}
```

When XLA inlines the call during optimization, it propagates the attributes
onto the inlined operations, so the metadata lands on each operation from the
wrapped function.

Metadata values may be strings, bools, ints, or floats; all are attached as
strings, with bools rendered as `"true"`/`"false"`.

### Metadata survives transformations

Because the metadata is attached to a function rather than to ambient tracing
state, JAX transformations preserve it. Under `jax.vmap` the batched
operations carry it, and under `jax.grad` every computation derived from the
function — forward pass and backward pass — carries it too:

```python
@xla_metadata_call(tag="my_block")
def block(x):
  return jnp.sin(x)

print(jax.jit(jax.grad(block)).lower(1.0).as_text("hlo"))
```

```
xla_metadata_call.1 {
  Arg_0.1 = f32[] parameter(0)
  ROOT cos.1 = f32[] cosine(Arg_0.1)
}

xla_metadata_call_0.2 {
  Arg_1.1 = f32[] parameter(1)
  Arg_0.3 = f32[] parameter(0)
  ROOT mul.1 = f32[] multiply(Arg_1.1, Arg_0.3)
}

ENTRY main.3 {
  x.1 = f32[] parameter(0)
  xla_metadata_call.2 = f32[] call(x.1), to_apply=xla_metadata_call.1, frontend_attributes={tag="my_block"}
  constant.1 = f32[] constant(1)
  ROOT xla_metadata_call.3 = f32[] call(xla_metadata_call.2, constant.1), to_apply=xla_metadata_call_0.2, frontend_attributes={tag="my_block"}
}
```

The forward-pass computation (here just the `cosine` residual saved for the
backward pass) and the backward-pass computation are each staged out and
tagged.

To leave the backward pass untagged, or to tag it differently — say, with its
own scheduling group — use `xla_metadata_call2`, which takes the metadata as a
dict plus an `ad_metadata` option:

```python
from jax.experimental.xla_metadata import xla_metadata_call2

f = xla_metadata_call2(jnp.sin, {"tag": "x"})                             # tag fwd + bwd (default)
f = xla_metadata_call2(jnp.sin, {"tag": "x"}, ad_metadata='drop')         # tag fwd only
f = xla_metadata_call2(jnp.sin, {"tag": "x"}, ad_metadata={"tag": "y"})   # retag bwd
```

`ad_metadata` governs the computations autodiff derives beyond the forward
pass: the linearized (tangent) and transposed (backward-pass) computations.
One caveat: under forward-mode `jax.jvp`, the primal and tangent operations
are staged out as one fused computation, so they keep the forward metadata
regardless of `ad_metadata`.

One application built on this: `must_fuse_call` in
`jax.experimental.xla_metadata` wraps a function so that XLA must place all of
its operations in a single fusion.

### Tagging one operation in place: `set_xla_metadata`

There is also `set_xla_metadata`, with two modes. Wrapping a *value* tags just
the operation that produced it:

```python
from jax.experimental.xla_metadata import set_xla_metadata

def g(x):
  y = jnp.sin(x)
  z = jnp.cos(x)
  return set_xla_metadata(y * z, breakpoint=True)

print(jax.jit(g).lower(1.0).as_text("hlo"))
```

```
ENTRY main.1 {
  x.1 = f32[] parameter(0)
  sin.1 = f32[] sine(x.1)
  cos.1 = f32[] cosine(x.1)
  ROOT mul.1 = f32[] multiply(sin.1, cos.1), frontend_attributes={breakpoint="true"}
}
```

Called with no value, `set_xla_metadata(k="v")` instead acts as a context
manager (or decorator) that tags every operation traced under it.

Both modes come with caveats that `xla_metadata_call` avoids:

- The context manager works by setting ambient tracing state, which is part
  of `jit`'s cache key. Any jit-compiled function called under it — including
  library code that jits internally — is re-traced and re-compiled for each
  distinct metadata context, rather than reusing its existing cache entries.
- Value-tagging doesn't propagate through autodiff: differentiate `g` above
  and the backward-pass operations come out untagged.

Prefer `xla_metadata_call` unless you need to tag exactly one operation in
place.

Finally, a caveat that applies to all of these APIs: XLA intends to preserve
`frontend_attributes` through its optimization passes, but edge cases can
drop them, so inspect the optimized HLO if a tool depends on the metadata
surviving.

