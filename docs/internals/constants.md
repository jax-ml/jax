(constants-note)=

# Handling of closed-over constants

"Closed-over constants" are non-scalar arrays that are encountered during JAX
tracing of a function and do not have dependencies on any of the function's
arguments.
JAX operations such as `jax.numpy` and `lax` are staged out and do not create
closed-over constants.
In the following example, the arrays
`a_jax_array` and `np.full` are closed-over constants, but `jnp.full`
is not. We refer below to closed-over constants simply as constants.

```python
import numpy as np
from jax import jit
from jax import numpy as jnp

a_jax_array = jnp.ones((16,), dtype=np.float32)

@jit
def f(x):
  return x + a_jax_array + np.full((16,), 42.) + jnp.full((16,), 142.)
```

It is easy to close over constants inadvertently. You can set the configuration
environment variable `JAX_CAPTURED_CONSTANTS_WARN_BYTES` to any non-negative
value to log the closed-over constants of at least that size during lowering
of a function.

We describe below the **future** internal implementation details for
constants. As of October 2026, this is not yet the default implementation;
it is enabled by the environment variable `JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=True`.
See further [below](#previous-implementation-and-migration-hints)
for the details of the previous implementation, including its drawbacks.

## Tracing

When JAX tracing encounters a closed-over non-scalar constant (e.g. `np.ndarray`
or `jax.Array`), it is captured into `Jaxpr.consts` and paired with a
corresponding `core.Var` in `Jaxpr.constvars`.
The function `core.is_literalable` only turns 0D scalar constants
into `core.Literal`.
There are no changes to how tracing works due to `JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=True`.

## Lowering

When lowering the code to HLO we could just emit a `stablehlo.constant`
operation for each constant in `Jaxpr.consts`, but this would have several disadvantages:

 * if the constant is a `jax.Array` (e.g., the `a_jax_array` above), then it is
 pulled from the device to the host during lowering, and it will later be
 re-materialized on the device when the lowered module executes.
 This can increase the host memory usage, sometimes dramatically.
 Furthermore, if the constant is sharded on multiple devices this
 sharding is lost.
 * large constants increase the size of the HLO, especially if
 the same constant is used multiple times. Also, the XLA compiler will attempt
 to constant-fold them, resulting in warnings and slow compilation. Furthermore,
 we have observed that XLA constant-folding sometimes produces slightly
 different
 numerics compared to compiled code.
 See also [Large closed-over constants are inlined in the HLO code #29684](https://github.com/jax-ml/jax/issues/29684).

Instead, during lowering we use the function `core.jaxpr_const_args` to collect
all hoistable non-scalar constants from `Jaxpr.consts` and recursively from any sub-`Jaxpr`s in equation parameters (e.g. `cond` branches,
`while_loop` bodies, `scan`). These constants are uniquified by their `id`.
The `core.jaxpr_const_args` function is memoized for each `Jaxpr` on which it is called.

The lowered HLO entry function will take one additional argument
for each unique constant appearing in the `Jaxpr` (or sub-`Jaxpr`s) to which it corresponds.
These arguments, referred to as `const_args`,
come after the dimension variable arguments, after the
token arguments, and just before the actual array arguments.
During lowering we maintain a mapping `const_lowering: dict[int, mlir.IrValues]`
from the `id(c)` of the constants to the HLO SSA values for the
corresponding const args.
This mapping is stored in the `mlir.LoweringRuleContext`.
When lowering `Jaxpr.constvars` (or sub-`Jaxpr.constvars` via `mlir.ir_consts`),
`_ir_constant` looks up `id(c)` in `const_lowering`:
when a constant is encountered, we reuse the entry block argument
from `const_lowering` instead of emitting a `stablehlo.constant`.

An exception is made for small constants, with size at most
`config.embedded_constants_max_bytes`.
These are not hoisted as additional arguments and are instead embedded in the
generated HLO and the executable. This helps avoid the overhead of passing
many small constants as arguments.

When we lower an HLO inner function (i.e., not the `main` function),
we call again `core.jaxpr_const_args`
to get the actual constants in the corresponding `Jaxpr`. These are
expected to be among the constants for which we have a `const_lowering`.
The inner function will get its own smaller set of `const_args` and
its own `const_lowering` mapping to be used when lowering the body.
E.g., the function `mlir.lower_jaxpr_as_fun` is one place where some
of this happens.

The function `mlir.jaxpr_subcomp` does not create a new HLO function,
but instead creates a block within the current function. It uses
the enclosing function's `const_lowering`.

Note also that there will still be `stablehlo.constant` in the lowered
code:
  * when the constant is a scalar; we want these constants to be
  available to XLA for constant folding.
  * when the constant is small, with size at most
  `config.embedded_constants_max_bytes`, as described above.
  * when the constant did not appear in the traced program, and is
  hence not in the `Jaxpr`. This can happen for constants that
  arise during lowering, e.g., the lowering of some PRNG functions
  include constants.
  * when we are exporting: at the moment, we do not hoist constant args
  when we export because the export serialization does not currently support
  serialization of arrays.
  We use the `mlir.LoweringParameters.hoist_constants_as_args` parameter
  to control this.

One additional complication is that some of the internal lowering functions
need to take the argument avals and sometimes also the shardings and
layouts for the arguments. Furthermore, the avals, shardings, and layout for
all arguments, including the const args,
are used also after lowering also. Therefore, it is convenient
to compute these fairly high in the call stack, e.g., in
`pxla.lower_sharding_computations`, and pass them down.

For example, the functions `mlir.lower_jaxpr_to_module`,
`pjit._pjit_cached_lower_jaxpr_to_fun`, and, `mlir.lower_jaxpr_to_fun`
take `in_avals`, `in_shardings`, and `in_layouts` that
that include both the avals for const_args and for the regular args
(the ones corresponding to the `Jaxpr.invars`).
They also take a `num_const_args` argument.

## Compilation and execution

The lowered MLIR module contains arguments for the const args, so
the compiled executable will need to be passed the const args.
It is important to choose the right place where we prepend the
const args. For example, in the following code, the second invocation
of the jitted function `f` is expected to hit the C++ jit cache without
any Python code executing.

```python
const = jnp.array([42.])
f = jax.jit(lambda: const)

f()
f()
```

(TODO: yashk2810 plans to write a description of how the jit caches work.)
This means that the `const` will have to be passed to the executable in C++
(and thus stored in `pxla.MeshExecutableFastpathData`),
and therefore the C++ cache
miss functions (e.g., `pjit._cpp_pjit.cache_miss`,
or `aot_cache_miss` in `pxla.MeshExecutable.create_cpp_call`)
will not take the const args as arguments. Instead these cache
miss functions will have to prepend the const args.

The C++ fast path has support for const args starting with jaxlib 0.7.1.
In prior versions, the fast path is disabled when there are const args.

To implement this scheme, we keep the `const_args` in
`stages.Lowering`, `stages.Lowered`, `stages.CompiledCallParams`,
and `pxla.MeshExecutable`.
In `stages.Compiled` the fields, e.g., `in_avals`, do not include
`const_args`.

Interestingly, when we serialize an executable, e.g., for the compilation
cache, we do not need to serialize the closed over constants. The executable
itself does not contain them, and needs to take them as const args.
Whoever is going to deserialize the cached executable will have to pass
the const args.

In AOT mode, the lowering and execution may
use different values of the `jax_enable_x64` configuration value.
If the constants are 64-bit `ndarray` we must use the same value
of `jax_enable_x64` for lowering and execution.

## Previous implementation and migration hints

This describes the current way we handle closed-over constants, as
of October 2026 (as long as `JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=False`).

Previously, JAX did not hoist closed-over constants as separate arguments.
Instead, they were inlined directly in the HLO code as `stablehlo.constant`
operations. The disadvantages are listed [above](#lowering).

Turning on `JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=True` can have visible effects
on the behavior of your code. If your code uses internal functions for
the lowering and execution paths, you will have to update your code as
described above in [lowering](#lowering) and
[compilation and execution](#compilation-and-execution).

If your code closes over constants you may notice that your executables are
smaller now but they take additional arguments. If your code closed over
`jax.Array` constants you will notice that your code does fewer copies to
bring those constants to the host during lowering.

If your code closes over `jax.Array` constants that are committed to devices,
you may get an error "Received incompatible devices for jitted computation".
This can happen because now those constants become arguments and JAX checks
that all arguments have a consistent device assignment. The solution is to
either not close over these constants or explicitly bring them to the host
or reshard them with a compatible device assignment before closing over them.
