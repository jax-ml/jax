# Generating reproducers for JAX errors

<!--* freshness: { reviewed: '2026-05-01' } *-->

WARNING: this code is experimental, expect changes or deletion.

Have you encountered a hard-to-debug **JAX error** in a large user program,
perhaps using several other libraries on top of JAX?
Do you believe that there is a small and pure JAX program, without additional
layers of libraries, that reproduces the same error? The reproducer tool
aims to help you produce such a program.

Summary:
  * [Usage](#usage)
  * [Configuration options](#configuration-options)
  * [Design](#design)
  * [Limitations](#limitations)
  * [Development](#development)

## Usage

If the `JAX_REPRO_DIR` is set to a directory, and JAX encounters an
uncaught exception under a JAX API call, e.g., `jax.jit`, it will attempt
to save in that directory a Python source file that when executed should
reproduce the error. That source file is standalone except for
importing `jax`.

JAX will track the sequence of nested JAX API calls, capturing all
user-functions that JAX traces, their calls to JAX APIs, and so on.
In large JAX programs we have seen the call depth grow to 30 or so.
If an uncaught exception arises, then we save a repro that should result in
the same call tree, and hopefully can reproduce the error.
One can get the path and source code of the saved repro by
calling `repro.last_saved_repro()`.

The above use case is the "implicit" repro generation, when all you need
to do is to set `JAX_REPRO_DIR`, and encounter an error.
You can also generate repros "explicitly", even in absence of errors,
by using the `repro.collect` function transformation:

```
fn = repro.collect(fn, repro_name="name")
# Once we trace `fn` we save `$JAX_REPRO_DIR/name_<counter>.py`
```

In the explicit usage mode, all tracked JAX API invocations are saved in the
repro.
In the implicit use (save on uncaught exception),
successful JAX calls at the top-level are not retained.

One can think of this mechanism as a way to stage out a pure JAX program
from a large JAX program.
This is somewhat similar to staging out a Jaxpr with
`jax.jit(f).trace(*args).jaxpr` (or the old `jax.make_jaxpr`), except that:

  * it produces Python source code rather that a dump of a Jaxpr, which
    should be more readable, more editable, and can be executed directly,
    e.g., in a debugger.
  * it works even if there are errors encountered before a Jaxprs is produced;
    the repro source may reproduce the error.
  * the repro is higher-level than the Jaxpr. E.g., instead of seeing the
    `lax.scan_p` primitive with its low-level details,
    you will see a call to `lax.scan`. The higher-order primitives in JAX
    often have complicated parameters, and sometimes even references to Python
    callables. Furthermore, some JAX transformations, e.g., `vmap` or `jvp`,
    do not stage a Jaxpr, and the first Jaxpr produced will reflect the
    result of the transformations. In contrast, the repro source will
    contains calls to `jax.vmap` and `jax.jvp`.
  * In Jaxpr the arguments are passed in a flat list, while in repros we
    retain standard PyTrees, with the same dictionary keys as in the user
    program.


## Configuration options

This section is very likely to change.

There are two main configuration flags:

 * `JAX_REPRO_DIR` denotes the directory where reproducers are saved. A
   non-empty value also triggers the tracking of the call tree, so that a
   reproducer is saved on error. It can be `sponge` for use in internal
   Google tests.
 * `JAX_REPRO_FLAGS` contains comma-separated flags that configure details of
   repro generation. You can specify a flag without a value, in which case it
   takes a default value, e.g., `True`, or you can specify a value using
   `=value`. For example, `log_calls,log_traceback_frames=10`.
    * `log_calls` (default 0). An integer value that controls the repro
      tracking logging (for debugging the repro module). The recognized
      values are: 0 (no logging, default), 1 (log all calls except the JAX
      primitive.bind), 2 (log all calls).
    * `log_call_details` (default ""). A sequence of call ids for which to log
      more details (for debugging the repro module). E.g.,
      `log_call_details=3+5+6`.
    * `error_mode` (default "defer"). Configures the handling of repro
      collection and generation errors. The possible values are:
       * "ignore"
       * "log" -- the errors are logged as `logging.error`. Each error message
         contains `log_traceback_frames` stack frames.
       * "defer" -- the errors are logged and at the end of the explicit
         repro collection a `repro.ReproError` will be generated.
       * "raise" -- a `repro.ReproError` is raised when the first error appears.

    * `log_traceback_frames` (default 40) how many frames from the traceback to
      show.
    * `fake_array_threshold` (default 128) arrays with `.size()` larger than
      this value are replaced with `np.ones` with the right shape and dtype.
      Smaller arrays are emitted as `np` array literals.

## Limitations

So many ...

Not all errors will be reproduced by this mechanism:

  * Errors from numpy won't be captured,
  * Errors from the jax.numpy layer are not captured.
    E.g., the rank checking happens in the jax.numpy layer,
    so it won't be reproduced by this version of repros, but the shape checking
    happens after binding the JAX primitives, so it will be reproducible.
  * The repro contains some argument preprocessing, e.g., the `static_argnums`
    are removed. Errors involving the handling of `static_argnums` won't be
    reproduced.
  * We currently do not try to preserve the exact data arrays, and for
    arrays larger than a threshold we will use np.ones.
  * `jax.named_call` is not handled currently (treated as a noop)
  * We do not capture some of the context managers that control the lowered
    code, e.g., `set_xla_metadata`. We do capture `set_mesh` though.

During call tracking we attempt to alter the execution as little as possible.
There are a few known differences though:

  * JAX cache tracing is foiled, so you are going to see functions traced
    multiple times. E.g., JAX will normally avoid tracing a function repeatedly
    with similar arguments, but when repros are enabled this cache is disabled.
    This will result in slower tracing, and for functions with side-effects it
    may even alter the result or side-effects of the program.

  * when using `jax.custom_vjp`, when we do higher-order differentiation the
    custom fwd and bwd functions are being called more than once, and we assume
    that the subsequent calls will produce the same Jaxpr as earlier ones.

  * we emit jax.Array as np.ndarray (e.g., loosing sharding)

  * if we don't recognize the jax.checkpoint policy param, we print a warning
    and we use `dots_saveable` as a replacement.


## Design

### JAX higher-order APIs

There are a few different categories of JAX APIs. Here are some examples:

- higher-order functions that return only arrays,
  e.g. `jax.lax.cond: Arr x (Arr -> Arr) x (Arr -> Arr) -> Arr`.
- higher-order functions that return only a single function
  e.g. `jax.vmap: (Arr -> Arr) -> (Arr -> Arr)`.
- higher-order function that return arrays **and** functions
  e.g. `jax.vjp: (Arr -> Arr) x Arr -> Arr x (Arr -> Arr)`.

#### APIs that return only arrays

The simplest higher-order JAX APIs, such as `jax.lax.cond` and
`jax.lax.scan`, return only arrays, not functions. For example, the type of
`jax.lax.cond` is:
```
jax.lax.cond: Arr x (Arr -> Arr) x (Arr -> Arr) -> Arr
```

Once we intercept a call to `jax.lax.cond` we assume that its callable arguments
are user functions for which we need to synthesize Python source code. We wrap
those functions so that we can intercept when they are called, and then we call
the actual `jax.lax.cond`. We will notice when JAX traces the wrapped user
functions and we record the arguments. We also record all the calls to
the first-order JAX primitives made from user functions.
It is possible that the user functions call
`jax.lax.cond` themselves. Therefore, we record a call tree with the root being
the top-level call to `jax.lax.cond`. On each path through the tree, JAX API
calls alternate with calls to user functions. The leaves of the call tree are
calls to first-order JAX primitives. We call this process that happens during
tracing "tracking".

If we need to emit a repro (on an uncaught exception or if explicitly requested)
we can turn the call tree into source code. For more details,
see [Repro emitting](#repro-emitting).

### APIs that return only a single function

A more complicated form of higher-order APIs are function transformations,
which return a single first-order function. For example, `jax.vmap`,
`jax.jit`, `jax.grad`, `jax.shard_map`, and many others.

The type of `jax.vmap` is:
```
jax.vmap: (Arr -> Arr) -> (Arr -> Arr)
```

These are more complicated to handle because the result of `jax.vmap(fun)`
may be invoked multiple times, and each invocation may trigger
re-tracing of the user function `fun`. In the most general case, there
could be conditionals inside `fun` that result in different executions.
We must be ready to generate different function definitions for `fun` for
each invocation of `jax.vmap(fun)(...)`.

We reduce these function transformations to the previous case by uncurrying
them. For example, we rewrite `jax.vmap` to a two-argument function `vmap_call`:
```
vmap_call: (Arr -> Arr) x Arr -> Arr
```

Thus a program:
```
def fun(...): ...
vf = jax.vmap(fun)
vf(a1)
vf(a2)
```

would be rewritten as
```
vmap_call(fun, a1)
vmap_call(fun, a2)
```

We handle this program in a similar way to `jax.lax.cond` (or other APIs that
return arrays). This allows us to generate a separate source for `fun` for
each of the calls.

We assume that for a JAX API that returns to the user program a single
function, the only thing that the user program can do with it is to call it.
This is not tecnically true in JAX, e.g., the return from `jax.jit` has some
extra fields for manipulating caches, or for invoking the AOT APIs, `.lower()`,
`.trace()`, etc. We have a few bespoke workarounds to handle those cases.

### APIs that return a tuple of arrays and functions

A small number of JAX APIs return tuples of functions and arrays. For example,
`jax.vjp`:
```
jax.vjp: (Arr -> Arr) x Arr -> Arr x (Arr -> Arr)
```

`jax.vjp` returns the primal output for the function passed as the first
argument at the point specified by the second argument, and it also returns
the function to compute the cotangent of the primal input given the
cotangent of the output.

#### The general case

For the general case of a JAX higher-order API `jaxapi` of the form:
```
*out_arrays, *out_funcs = jaxapi(*in_arrays, *in_funcs)
```
where `out_arrays` is non-empty or `out_funcs` is not a singleton.
We make the following assumptions (see later for a discussion of some
exceptions):
  * `jaxapi` is annotated with `api_boundary` with an optional
    `map_user_func_args` to identify the `in_funcs` function arguments.

      * if one of these annotations is missing, we are going to see user code
        calling directly into the tracing machinery. The repro tracking
        machinery will give an error.
  * the `in_funcs` are considered references to user functions for which we
    must generate reproducers.
  * `jaxapi` invokes each user function at most once, for tracing (with
    tracer arguments).
  * `jaxapi` does not leak the user functions to the user code, i.e., the
    wrapped user functions it took as arguments can only be invoked internally
    by `jaxapi`.

      * in an earlier implementation of the handling of `fuser.push_block_spec`
      I forgot to traverse the returned `pallas.BlockSpec` to wrap the `index_map` functions
      and I ended up with spurious calls to wrapped user functions directly
      from user code.

  * if `out_funcs` is not empty, then they are first-order functions. They are
    considered JAX functions, and as such are not allowed to further invoke
    the user functions passed to `jaxapi`.
  *

There are a few other more complicated cases, e.g., `custom_vjp`, or the
`fuser` APIs, but they can be reduced to the cases above with a bit of argument
and result rewriting.

### API Boundary Tags

We label the **higher-order JAX APIs** with
`traceback_util.api_boundary(repro_api_name="jax.jit")`. There are about
50 such APIs in JAX.
These APIs take user functions as arguments, for which we must synthesize
Python source code.

Most JAX higher-order APIs will take a single user function as the first
argument. For more complex cases, one can use the
`wrap_user_func_args` argument to `api_boundary` to describe
how to identify among the user functions among the arguments.
In an earlier design, I tried to identify these user functions automatically
without relying on `wrap_user_func_args`. First, I tried to detect user
functions by looking for callables among the positional arguments. This is
not enough, because some custom PyTrees, e.g., `flax.module`, are callable yet
we sometimes must treat them as containers with arrays. I also tried to detect
user functions by their source location but that turned out to be too brittle
(there are functions without accurate source location, and there are some
helper functions in the JAX code that that should be considered user functions).

We could wrap even first-order JAX APIs with `api_boundary` if we want
to see them in the generated repro source. By default, we don't, which means
that you will not see `jax.numpy.einsum` in the generated source and instead
you will see a call to `jax.jit` with a body that contains a few `dot`
primitives, which is how `jax.numpy.einsum` is implemented.

See the [Trampolines](#trampolines) section below for how we choose the
`repro_api_name` parameter for `api_boundary`.

### Trampolines

For many higher-order JAX APIs we use a system of trampolines to rewrite their
calls to helper functions that help the reproducers. These trampolines are
defined in `trampolines.api_trampolines` (indexed by `repro_api_name`) and
are processed by `api_boundary`. The trampolines typically redirect the call to
helper functions defined in `repro_api.py`. The reproducer code will contain
references to the functions in `repro_api.py`, not to the original JAX APIs.

For example, for `jax.lax.cond` we use a trampoline `repro_api.cond` for the
sole purpose of being able to specify `wrap_user_func_args` without putting it
in the main JAX sources. So, in the main sources we annotate `jax.lax.cond`
with `api_boundary(repro_api_name="jax.lax.cond")`. Then we define a trampoline
that redirects the call to `repro_api.cond`, on which we put
a `api_boundary(repro_api_name="cond", wrap_user_func_args=...)`.
The reproducer source contains calls to `repro_api.cond`, which allows us
to synthesize reproducers for reproducers (this is handy, see below).

Another class of trampolines are the uncurrying trampolines, which we use
for APIs that return a single function, such as `jax.vmap`. The trampoline
for `jax.vmap` will rewrite the call to `repro_api.vmap_call`.

We explain here the naming convention for the `repro_api_name` parameter
passed to `api_boundary` and used by the trampolines:
  * `repro_api_name` is the name by which we expect user code to
    reference the function. It starts with `jax.` (or `pallas.` or `fuser.`).
    In any case, there is generally a `.` in the name.
  * Some of the APIs do not have a trampoline, e.g., `lax.scatter_add`. The
    reproducer code for this uses the exact `repro_api_name`. The generated
    repro sources import the file `repro_runtime.py` to provide the right names.
  * The name of the `repro_api` trampoline function is formed by taking the
    `repro_api_name`, removing the `jax.` prefix if present, and perhaps append
    `_call` if it is an uncurried trampoline. E.g., `jax.vmap` becomes
    `repro_api.vmap_call`.
  * The functions in `repro_api.py` also have their own `api_boundary`
    annotations, using `repro_api_name` as the name of the function in
    `repro_api.py`, e.g., `vmap`.

### Handling caches

JAX has a complex system of caches, and this creates problems for repro
generation. We rely on the fact that we observe the calls to all wrapped
user functions, but with caches we may miss some.
Consider the following example:
```
def fun(x): ...
j_fun = jax.jit(fun)
y0 = j_fun(0)
y1 = j_fun(0.)
y2 = j_fun(1)  # hits the `fun` tracing cache for `j_fun(0)`
```

which is rewritted using trampolines to:
```
def fun(x): ...
y0 = jit_call(fun, 0)
y1 = jit_call(fun, 0.)
y2 = jit_call(fun, 1)
```

On the 3rd call to `jit_call` we will may miss the tracing call to `fun`, and
we don't know which of several previous invocations of `fun` to use here.

Since the `jit_call` is annotated with `api_boundary`, it will wrap the `fun`
argument, and each invocation produces a fresh wrapper. This foils the
tracing caches, and we are guaranteed to see the invocation all all wrapped
`fun` in each call. But this slows down the tracing significantly, and also
results in generating more versions of `fun` than necessary.

We can recover some of the caching benefits, by monkey patching the internal
`pe.trace_to_jaxpr` function that invokes the tracing for a function. We
maintain out own weakref_lru_cache to skip tracing while detecting which of
the previous `Func` objects for the same user function hit the cache.

There may be other caches that we are missing, but the `trace_to_jaxpr` one
makes a big-enough difference for now.

### Handling PyTrees

For each call, we store the arguments and results. These would be tracers for
the user functions, but may be arrays or other objects. We cannot store these
directly in the call trees:
  * If these are user PyTrees we cannot generate code to construct them in the
    repro, because the constructors could depend on user code that we cannot
    trace. To remedy this, we normalize arguments and results by traversing them
    and keeping the standard containers (tuples, lists, dictionaries, None)
    and flattening one level of the user PyTrees as tuples.
  * However, we keep some JAX PyTrees for which we have
  special emitter rules, e.g., lax.GatherDimensionNumbers.
  * if these are mutable values, they may be mutated by the user program.
  Normalization ensures that we make copies of lists and dictionaries.

We considered completely flattening the arguments and results, but that
turned out to be unnecessary and removes one level of readability of the
repros, e.g., by turning nested dictionaries with user-readable key names
into very large tuples. Furthermore, this would require some processing of
parameters such as `nondiff_argnums` to refer to flat tuple elements.

### Handling statics

A few JAX APIs allow "static" values that are not captured by JAX tracing.
E.g., `jax.jit` can declare that some arguments are static, which means that
they are passed to the user function but are not captured in the Jaxpr.
For reproducers we must also not capture statics because some of them may
involve complex user data structures that we don't want to have in a
standalone reproducer.

For example, in the following program:
```
class MyData: ...
def f(x: jax.Array, extra: MyData) -> jax.Array: ...
jax.jit(f, static_argnums=(1,))(x)
```

we would like to see in the reproducer something like:
```
def f(x: jax.Array) -> jax.Array: ...
jit_call(f, x)
```

TODO: explain how we deal with statics.

### Data structure

Tracking collects the following data structure (defined in `tracker.py`):

  * a `FunctionDef` represents the definition of a user function that
    was passed to one of the higher-order JAX APIs, e.g., to `jax.jit`, or even
    to `repro.collect`. It contains a `body` field with the list of `Statement`s
    that were encountered during the JAX tracing of that function.

  * a `Statement` represents a call to a bind of a JAX primitive (e.g.,
    `add_p.bind(...)`), or a call to a JAX API (e.g. `jax.sin(...)`). There is
    a top-level statement, which is either the call to `repro.collect` when we
    are doing explicit collection, or the top-level JAX API call being traced
    (for implicit collection).

Both `FunctionDef` and `Statement` are subclasses of `Call` which provides
common fields:
  * `parent` - for a `Statement` this is the `FunctionDef` that contains it,
    for a `FunctionDef` this is the `Statement` that called the higher-order
    JAX function that had this user-function as one of its arguments.
  * `id` - a unique index counting `Call` constructors, for debugging.
  * `level` - 1 more than the level of the `parent`. We log `Call`s as
    "[<level>.<id>]". You can enable this logging with `log_calls`.
  * `args`, `kwargs`, `result` these are normalized objects that were used
    at the call site. These are typically PyTress of tracers.

We also use object of type `Func` to represent wrapped functions:
  * either a JAX API, in which case `api_name` is the name of the API and
    `is_user==False`.
  * or a user function that was passed to a higher-order API, in which case
    `is_user==True` and `function_def` points to the `FunctionDef` (once
    JAX calls the function).
  * of a JAX function that was returned from a higher-order API that is not
    curried, e.g., `jax.vjp`.

We currently maintain the following invariants and assumptions:
  * The `Statement` and `FunctionDef` form a call tree with
    a `Statement` at the root, and
    alternating `Statement` and `FunctionDef` on any path. The leaves of the
    tree are `Statement`s representing calls to first-order primitives.
  * All higher-order JAX APIs are annotated with `traceback_util.api_boundary`
    with a `repro_api_name` field. We currently support about 50 such API
    functions. If we forget to annotate a higher-order API, and user-level code
    calls it, you may get several kinds of errors:
      * "USER function calls directly into tracing" - if a user-level function
        calls this unannotated API and JAX starts tracing some function.
      * "Binding primitive scan_p containing Jaxprs or functions, for f" - This
        would happen, e.g., if `jax.scan` were not annotated.
  * We expect that when a higher-order JAX API is called with a user function,
    that user function is invoked exactly once, to be traced (with
    `jax.core.Tracer` arguments). If the user function is called more than once
    you will see a warning "Ignoring additional invocation to USER function".
  * It is possible that we don't see a call for some USER functions, if JAX
    does not trace them. E.g., in `jax.custom_vjp` we pass a user function for the
    forward pass and one for the backwards, but the latter will never be traced if
    we do not differentiate the function.
  * If the USER function passed to a higher-order API leaks the USER function
    back to the user-level code, you may see the additional invocation
    warning or a "USER call made from USER parent" error message.
      * This is true for the JAX APIs, but in one earlier attempt we have
        tried to wrap the `BlockSpec.index_map` as a USER function in the
        `BlockSpec` constructor. This was a problem because the
        `BlockSpec` is accessible to the user code and there are programs that call
        the index map themselves. A further problem was that this could be done
        before the repro collection starts with `repro.collector`, which resets
        the tracking state and we end up reusing the ids for the
        `BlockSpec.index_map` objects in later calls. This confused the repro
        generation code.
  * User functions can call `jax.numpy` and `lax` operations, which will
    result in binding one or more JAX primitives. We intercept the calls to
    `core.Primitive.bind` to collect such calls.
  * User functions cannot bind directly higher-order primitives, e.g.,
    `scan_p`. We expect that they call instead one of the JAX higher-order
    APIs, e.g., `jax.scan`.
  * We ignore all calls to JAX functions (primitives or JAX APIs) if they
    are made from a `Statement`. This can happen when we are inside JAX and
    we call `jax.jit` or `lax.scan` internally. The same
    mechanism will ignore binding of higher-order primitives, e.g.,
    `lax.scan_p`, because those are bound always from a JAX API call (since we
    annotate all higher-order JAX APIs).


## Repro emitting

We generate code for a `FunctionDef` by emitting a sequence of calls either
to bind a first-order primitive, or to functions like `repro_api.vmap_call`.
We introduce function arguments and local variables for all tracers that appear
as `args` for the `FunctionDef` and as the `result` of the `Statement`s in the
body. These are then passed as the `args` of `Statement` and the `result` of
the enclosing `FunctionDef`.

JAX primitives can take a variety of parameters of internal types, such as
`jax.NamedSharding`. Most of the `emitter.py` module defines functions that
can emit code to construct these internal data structures.

Emitting starts from the top-level `Statement`, which typically is a call
to a JAX API such as `repro_api.jit_call`. As we emit the code for the
arguments we encounter USER `Func` objects, for which we must generate code
for their `FunctionDef`.

The main difficulty when emitting is to handle nested function definitions.
In the following emitted repro the names have a numeric suffix that denotes
the order in which they are generated:
```
def sin3(x4):
  v5 = repro_api.bind("sin", x4)
  return v5

def f1(x2):
  y6 = jit_call(sin3, x2)
  def g7(y8):
    v9 = y8 + x2  # x2 is external
    return v9
  z10 = jit_call(g7, y6)
  return z10

v11 = jit_call(f1, 5)
```

As we emit the `FunctionDef` of `f1`, we encounter first a call to `jit_call(sin)`,
and we generate the body for `sin`. Since this body does not have external
references we can emit it at top-level.
For the call `jit_call(g7)` we emit the body of `g7` and we observe an
external reference to `x2`. We cannot lift the body of `g7` to top level,
it must be as high as possible while its references are in scope.

## Development

Until I can merge this somewhere upstream I am evolving it in a personal
branch. I keep it as a stack of commits, with the base one adding the
infrastructure and a bunch of basic APIs (jit, grad, vmap, etc.) and subsequent
ones adding more exotic APIs: various Pallas APIs, fuser, hijax. Those are
still in flux. It is also useful to see what it takes to add support for a new
API.

See [repro_skill_adding_api.md](repro_skill_adding_api.md) for a step-by-step guide.