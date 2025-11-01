# Generating reproducers for JAX errors

<!--* freshness: { reviewed: '2025-10-15' } *-->

Have you encountered hard-to-debug JAX error in a large user program?
Do you believe that there is a small pure JAX program, without additional libraries,
that reproduces the same error?

WARNING: this code is experimental, expect changes or deletion.

## Usage

If you get an uncaught exception from under a JAX API call,
you can set `JAX_REPRO_DIR` to a directory where JAX should attempt to save a Python source
file that contains the JAX API calls that ought to reproduce the error.
This mechanism can be enabled simply by setting the `JAX_REPRO_DIR` variable
(e.g., to "sponge" if using this in Google tests).
JAX will track the sequence of nested JAX API calls, capturing all user-functions,
their calls to JAX APIs, and then recursively the user functions that are
called by JAX during tracing.
If an uncaught exception arises, then we save repro that should result in
the same call tree, and hopefully can reproduce the error.
One can get the path and source code of the saved repro by calling `repro.last_error_repro()`.

Alternatively, you can get repros by explicitly calling an API even in absence of errors:

```
   from jax._src import repro
   col = repro.collector(fun)  # fun should be a nullary Callable
   try:
      result = col()  # Executes `fun` and returns its result
   finally:
     repro_source = col.to_source()
     repro_path = repro.save()
```

`repro.collector` will error if `JAX_REPRO_DIR` is not set.
In the usage above, all tracked JAX API invocations are saved in the repro.
In the implicit use (save on uncaught exception),
successful JAX calls at the top-level are not retained.

One can think of this mechanism as a way to stage out (slice) a pure JAX program from a large JAX program.
This is somewhat similar to staging out a Jaxpr with `jax.jit(f).trace(*args).jaxpr`
(or the old `jax.make_jaxpr`), except that:

  * it produces Python source code rather that a dump of a Jaxpr.
    This means that the result can be more readable, more editable, and can be executed directly
    e.g., in a debugger.
  * it works even if there are errors in the user program or in JAX.
    Then the produced output may reproduce the error.
  * the repro is higher-level than the Jaxpr: the higher-order Jaxpr primitives
    are replaced by calls to high-level JAX API.
    E.g., instead of seeing the `lax.scan_p` primitive with its low-level details,
    you will see a call to `lax.scan`. The higher-order primitives in JAX often have
    complicated parameters, and sometimes even references to Python callables.
    However, most first-order primitives are going to be represented in a similar way as in a Jaxpr.
  * the repro code will contain the sequence of JAX API calls as they appear
    in the user code, e.g., `jax.jvp(jax.vmap(f))`, even when the Jaxpr
    would reflect the code after these transformations.

## Configuration options

This section is very likely to change.

There are two configuration options:

 * `JAX_REPRO_DIR` denotes the directory where reproducers are saved. A non-empty
   value also triggers the tracking of the call tree, so that a reproducer is saved
   on error.
 * `JAX_REPRO_FLAGS` contains a comma-separated flags that configure how repro generation works.
    You can specify a flag without a value, in which case it takes a default value, e.g., `True`,
    or you can specify a value using `=value`. For example, `enable_checks,log_traceback_frames=10`.
    * `enable_checks` (default `True`) performs some invariant checks during the tracking and emitting.
      The main reason to turn this off is to gain a tiny bit of performance.
      You can see these errors in the log labelled "Repro error".
      To see the logs when running pytest, use `-o log_cli=1 -o log_cli_level=INFO`.
    * `enable_checks_with_tracebacks` (default `True`, implies `enable_checks`) when a check fails, print
      a traceback. Even when the check fails during repro emitting (e.g., that a value used in an
      operation has been defined), we try to print the traceback for when the value was created (instead
      of the traceback during emitting).
    * `enable_checks_as_errors` (default `False`, implies `enable_checks`) when a check fails, print
      a traceback and throw a `ReproError ` exception.
    * `log_traceback_frames` (default 40) how many frames from the traceback to show.
    * `fake_array_threshold` (default 128) arrays with `.size()` larger than this value are replaced
      with `np.ones` with the right shape and dtype. Smaller arrays are emitted as `np` array literals.

## Limitations

So many ...

Not all errors will be reproduced by this mechanism:

  * errors from numpy won't be captured,
  * errors from the jax.numpy layer. E.g., the rank checking happens in the jax.numpy layer,
    so it won't be reproduced by this version of repros, but the shape checking
    happens after binding the JAX primitives, so it will be reproducible,
  * we currently do not try to preserve the exact data arrays, and for
    arrays larger than a threshold we will use np.ones.
  * `jax.named_call` is not handled currently (treated as a noop)

During call tracking we attempt to alter as little as possible the execution.
There are a few known differences though:

  * JAX cache tracing is foiled, so you are going to see functions tracer multiple times.
    E.g., JAX will avoid tracing a function repeatedly with similar arguments,
    but when repros are enabled this cache is disabled.
   This will result in slower tracing, and for functions with side-effects it may even
   alter the execution of the program.

  * when using `jax.custom_vjp`, when we do higher-order differentiation the custom
   fwd and bwd functions are being called more than once, and we assume that the
   subsequent calls will produce the same Jaxpr as earlier ones.

  * we emit jax.Array as np.ndarray (e.g., loosing sharding)

  * if we don't recognize the jax.checkpoint policy param, we print a warning
    and we use `dots_saveable` as a replacement.
