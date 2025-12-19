# Generating reproducers for JAX errors

<!--* freshness: { reviewed: '2025-10-15' } *-->

WARNING: this code is experimental, expect changes or deletion.

Have you encountered a hard-to-debug JAX error in a large user program,
perhaps using several other libraries on top of JAX?
Do you believe that there is a small and pure JAX program, without additional
layers of libraries, that reproduces the same error?


## Usage

If you get an uncaught exception from under a JAX API call,
you can set `JAX_REPRO_DIR` to a directory where JAX should attempt to save a Python source
file that contains the JAX API calls that ought to reproduce the error.
This mechanism can be enabled simply by setting the `JAX_REPRO_DIR` variable.

JAX will track the sequence of nested JAX API calls, capturing all user-functions,
their calls to JAX APIs, and then recursively the user functions that are
called by JAX during tracing.
If an uncaught exception arises, then we save a repro that should result in
the same call tree, and hopefully can reproduce the error.
One can get the path and source code of the saved repro by
calling `repro.last_saved_repro()`.

The above use case is the "implicit" repro generation. You can also
generate repros "explicitly", even in absence of errors:

```
   from jax._src import repro  # TODO: find final location
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

One can think of this mechanism as a way to stage out a pure JAX program
from a large JAX program.
This is somewhat similar to staging out a Jaxpr with `jax.jit(f).trace(*args).jaxpr`
(or the old `jax.make_jaxpr`), except that:

  * it produces Python source code rather that a dump of a Jaxpr, which
    should be more readable, more editable, and can be executed directly,
    e.g., in a debugger.
  * it works even if there are errors in the user program or in JAX before
    a Jaxprs is produced; the repro source may reproduce the error.
  * the repro is higher-level than the Jaxpr. E.g., instead of seeing the
    `lax.scan_p` primitive with its low-level details,
    you will see a call to `lax.scan`. The higher-order primitives in JAX often have
    complicated parameters, and sometimes even references to Python callables.
    Furthermore, some JAX transformations, e.g., `vmap` or `jvp`,
    do not stage a Jaxpr, and the first Jaxpr produced will reflect the
    result of the transformations. In contrast, the repro source will
    contains calls to `jax.vmap` and `jax.jvp`.


## Configuration options

This section is very likely to change.

There are two configuration options:

 * `JAX_REPRO_DIR` denotes the directory where reproducers are saved. A non-empty
   value also triggers the tracking of the call tree, so that a reproducer is saved
   on error. It can be `sponge` for use in internal Google tests.
 * `JAX_REPRO_FLAGS` contains comma-separated flags that configure how repro generation works.
    You can specify a flag without a value, in which case it takes a default value, e.g., `True`,
    or you can specify a value using `=value`. For example, `log_calls,log_traceback_frames=10`.
    * `log_calls` (default 0). An integer value that controls the repro tracking logging (for debugging
      the repro module). The recognized values are: 0 (no logging, default), 1 (log all calls except the
      JAX primitive.bind), 2 (log all calls).
    * `log_call_details` (default ""). A sequence of call ids for which to log more details (
      for debugging the repro module). E.g., `log_call_details=3+5+6`.
    * `error_mode` (default "defer"). Configures the handling of repro collection and generation
      errors. The possible values are:
       * "ignore"
       * "log" -- the errors are logged as `logging.error`. Each error message contains
         `log_traceback_frames` stack frames.
       * "defer" -- the errors are logged and at the end of the explicit
         repro collection a `repro.ReproError` will be generated.
       * "raise" -- a `repro.ReproError` is raised when the first error appears.

    * `log_traceback_frames` (default 40) how many frames from the traceback to show.
    * `fake_array_threshold` (default 128) arrays with `.size()` larger than this value are replaced
      with `np.ones` with the right shape and dtype. Smaller arrays are emitted as `np` array literals.

## Limitations

So many ...

Not all errors will be reproduced by this mechanism:

  * Errors from numpy won't be captured,
  * Errors from the jax.numpy layer. E.g., the rank checking happens in the jax.numpy layer,
    so it won't be reproduced by this version of repros, but the shape checking
    happens after binding the JAX primitives, so it will be reproducible,
  * The repro contains some argument preprocessing, e.g., the `static_argnums`
    are removed. Errors involving the handling of `static_argnums` won't be
    reproduced.
  * we currently do not try to preserve the exact data arrays, and for
    arrays larger than a threshold we will use np.ones.
  * `jax.named_call` is not handled currently (treated as a noop)

During call tracking we attempt to alter as little as possible the execution.
There are a few known differences though:

  * JAX cache tracing is foiled, so you are going to see functions traced multiple times.
    E.g., JAX will normally avoid tracing a function repeatedly with similar arguments,
    but when repros are enabled this cache is disabled.
   This will result in slower tracing, and for functions with side-effects it may even
   alter the execution of the program.

  * when using `jax.custom_vjp`, when we do higher-order differentiation the custom
   fwd and bwd functions are being called more than once, and we assume that the
   subsequent calls will produce the same Jaxpr as earlier ones.

  * we emit jax.Array as np.ndarray (e.g., loosing sharding)

  * if we don't recognize the jax.checkpoint policy param, we print a warning
    and we use `dots_saveable` as a replacement.


## Design

There are two phases: repro collection, when we follow certain function calls
and construct a call tree, and repro generation when we produce Python
source code that should reproduce the same call tree.

### Repro collection

### Tracking JAX and USER functions

We label top-level higher-order JAX APIs with
`traceback_util.api_boundary(repro_api_name="jax.jit")`. These
APIs take user functions as arguments, which we must
also wrap and track. In an earlier design, I tried
to identify these user functions by looking for callables
among the positional arguments. This is not enough, because
some custom PyTrees, e.g., `flax.module`, are callable
yet we sometimes must treat them as containers with arrays.

Instead, for each of these calls we also pass a `wrap_user_func_args`
argument which takes in the args and kwargs and returns
the args with the user-functions wrapped. By default,
`wrap_user_func` wraps the first positional argument, if
it is a callable.

We keep a call stack of USER and JAX functions as the
program executes. The bottom of the stack (first call)
is always a JAX API function. Some of the JAX functions
will call into USER functions that were passed as arguments,
in order to trace them. Then the user function may
call again into JAX functions.

Additionally, we intercept the `core.Primitive.bind` and
we consider those as first-order JAX calls.

Note that we ignore all calls to JAX functions (primitives
or JAX APIs) if they are made from a JAX function. This
can happen because we use, e.g., `jax.jit` or `lax.scan`
internally in JAX, and we don't need to track those.
The same mechanism will ignore the calls to the
higher-order primitives, e.g., `lax.scan_p`, because
those are bound always from a JAX API call (since we
annotate all higher-order JAX APIs).

Finally, when we are inside a USER call, we collect
in a list all the JAX calls made. The end result is
a top-evel call node of a JAX API function with
some USER functions passed as arguments. Each USER
function contains a list of JAX calls, each with
its own USER function among arguments. This is the
date structure that results from repro collection.

Note that a USER function cannot call another USER
function, because USER functions are called during
tracing and are passed only arrays.

We should see at most one call to each USER function
object, because JAX should not trace a function
twice (an exception happens for `fuser.fusible`,
see below). It is possible that we don't see a
call for some USER functions, if JAX do not trace
them. E.g., in `jax.custom_vjp` we pass a user function
for the forward pass and one for the backwards, but
the latter will never be traced if we do not differentiate
the function.

#### Handling caches

One of the trickiest issues was to
collect repros in presence of JAX caches. JAX will try
to memoize the tracing of user function, e.g., based
on the shapes and types of the arguments. In reality,
the cache keys are quite a bit more complicated.

If we just wrap the user functions passed to the JAX
APIs and track their calls, we may see a function called
multiple times, and some calls we won't even see if
they hit the cache. E.g., for the program:

```
def fun(x): ...
j_fun = jax.jit(fun)
y0 = j_fun(0)
y1 = j_fun(0.)
y2 = j_fun(1)
```

we will see calls to `fun` coming from `j_fun(0)` and `j_fun(0.)`,
but we won't see corresponding to `j_fun(1)` because it will hit
the cache for the first call.

The solution that I ended up using, is to set up a set of
predefined trampolines for the JAX API calls, indexed by
the `api_name` (e.g., "jax.jit"). These trampolines will
behave as if the code above had been:

```
def func(x):
y0 = jax_jit_call(fun, 0)
y1 = jax_jit_call(fun, 0.)
y2 = jax_jit_call(fun, 1)
```

Furthermore, because `jax_jit_call` will wrap the firts argument
as a fresh object on each call, the undelying JAX caches will miss.
These new JAX APIs will appear in the generated repro. They
are defined in `repro_api.py`.

With this system of trampolines (defined in `tracker.py`) we turn the
JAX program into one that uses modified JAX APIs that take user functions
as arguments but do not return functions. The trampolines end up being
quite useful to turn all the various forms of higher-order JAX APIs,
e.g., `jax.custom_vjp` with multiple user-defined functions,
into a uniform system of APIs that take all user-functions as
positional arguments, along with other non-callable arguments.

In some very rare cases, we had to retain functions that
return other function. E.g.,

```
def fun(x): ...
y, f_vjp = jax.vjp(f, x)
x_ct = f_vjp(y_ct)
```

We must mark the returned `f_vjp` function as a JAX function (because
it calls back into JAX internals). We do this in the trampoline
for `jax.vjp`.

### Miscellaneous collection issues

For each call, we store the arguments and results. These would
be tracers for the USER functions, but may be arrays or other
objects. If we store the arguments literally, we would run into
issues when they are mutable; user functions may mutate them.
We would also leak tracers and run into internal JAX error checks.
There is also no point in storing the custom PyTrees, because
they can never be part of the generated repro (to keep it pure JAX).

So, we first "normalize" the arguments before storing them into
the call nodes. Normalization turns custom PyTrees into tuples,
except for some hardcoded custom PyTrees for which we know
how to emit source code (see below in [repro generation](#repro-generation)).


### Repro generation


