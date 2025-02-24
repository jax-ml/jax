(ahead-of-time-lowering)=

# Ahead-of-time lowering and compilation

<!--* freshness: { reviewed: '2024-06-12' } *-->

JAX's `jax.jit` transformation returns a function that, when called,
compiles a computation and runs it on accelerators (or the CPU). As
the JIT acronym indicates, all compilation happens _just-in-time_ for
execution.

Some situations call for _ahead-of-time_ (AOT) compilation instead. When you
want to fully compile prior to execution time, or you want control over when
different parts of the compilation process take place, JAX has some options for
you.

First, let's review the stages of compilation. Suppose that `f` is a
function/callable output by {func}`jax.jit`, say `f = jax.jit(F)` for some input
callable `F`. When it is invoked with arguments, say `f(x, y)` where `x` and `y`
are arrays, JAX does the following in order:

1. **Stage out** a specialized version of the original Python callable
   `F` to an internal representation. The specialization reflects a
   restriction of `F` to input types inferred from properties of the
   arguments `x` and `y` (usually their shape and element type). JAX
   carries out this specialization by a process that we call
   _tracing_. During tracing, JAX stages the specialization of `F` to
   a jaxpr, which is a function in the [Jaxpr intermediate
   language](https://jax.readthedocs.io/en/latest/jaxpr.html).

2. **Lower** this specialized, staged-out computation to the XLA compiler's
   input language, StableHLO.

3. **Compile** the lowered HLO program to produce an optimized executable for
   the target device (CPU, GPU, or TPU).

4. **Execute** the compiled executable with the arrays `x` and `y` as arguments.

JAX's AOT API gives you direct control over each of these steps, plus
some other features along the way. An example:

```python
>>> import jax

>>> def f(x, y): return 2 * x + y
>>> x, y = 3, 4

>>> traced = jax.jit(f).trace(x, y)

>>> # Print the specialized, staged-out representation (as Jaxpr IR)
>>> print(traced.jaxpr)
{ lambda ; a:i32[] b:i32[]. let c:i32[] = mul 2 a; d:i32[] = add c b in (d,) }

>>> lowered = traced.lower()

>>> # Print lowered HLO
>>> print(lowered.as_text())
module @jit_f attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<2> : tensor<i32>
    %0 = stablehlo.multiply %c, %arg0 : tensor<i32>
    %1 = stablehlo.add %0, %arg1 : tensor<i32>
    return %1 : tensor<i32>
  }
}

>>> compiled = lowered.compile()

>>> # Query for cost analysis, print FLOP estimate
>>> compiled.cost_analysis()['flops']
2.0

>>> # Execute the compiled function!
>>> compiled(x, y)
Array(10, dtype=int32, weak_type=True)

```

Note that the lowered objects can be used only in the same process
in which they were lowered. For exporting use cases, see the {ref}`export` APIs.

See the {mod}`jax.stages` documentation for more details on what functionality
the lowering and compiled functions provide.

All optional arguments to `jit`---such as `static_argnums`---are respected in
the corresponding tracing, lowering, compilation, and execution.

In the example above, we can replace the arguments to `trace` with any objects
that have `shape` and `dtype` attributes:

```python
>>> i32_scalar = jax.ShapeDtypeStruct((), jnp.dtype('int32'))
>>> jax.jit(f).trace(i32_scalar, i32_scalar).lower().compile()(x, y)
Array(10, dtype=int32)

```

More generally, `trace` only needs its arguments to structurally supply what JAX
must know for specialization and lowering. For typical array arguments like the
ones above, this means `shape` and `dtype` fields. For static arguments, by
contrast, JAX needs actual array values (more on this
[below](#tracing-with-static-arguments)).

Invoking an AOT-compiled function with arguments that are incompatible with its
tracing raises an error:

```python
>>> x_1d = y_1d = jnp.arange(3)
>>> jax.jit(f).trace(i32_scalar, i32_scalar).lower().compile()(x_1d, y_1d)  # doctest: +IGNORE_EXCEPTION_DETAIL
...
Traceback (most recent call last):
TypeError: Argument types differ from the types for which this computation was compiled. The mismatches are:
Argument 'x' compiled with int32[] and called with int32[3]
Argument 'y' compiled with int32[] and called with int32[3]

>>> x_f = y_f = jnp.float32(72.)
>>> jax.jit(f).trace(i32_scalar, i32_scalar).lower().compile()(x_f, y_f)  # doctest: +IGNORE_EXCEPTION_DETAIL
...
Traceback (most recent call last):
TypeError: Argument types differ from the types for which this computation was compiled. The mismatches are:
Argument 'x' compiled with int32[] and called with float32[]
Argument 'y' compiled with int32[] and called with float32[]

```

Relatedly, AOT-compiled functions [cannot be transformed by JAX's just-in-time
transformations](#aot-compiled-functions-cannot-be-transformed) such as
`jax.jit`, {func}`jax.grad`, and {func}`jax.vmap`.


## Tracing with static arguments

Tracing with static arguments underscores the interaction between options
passed to `jax.jit`, the arguments passed to `trace`, and the arguments needed
to invoke the resulting compiled function. Continuing with our example above:

```python
>>> lowered_with_x = jax.jit(f, static_argnums=0).trace(7, 8).lower()

>>> # Lowered HLO, specialized to the *value* of the first argument (7)
>>> print(lowered_with_x.as_text())
module @jit_f attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i32>) -> (tensor<i32> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<14> : tensor<i32>
    %0 = stablehlo.add %c, %arg0 : tensor<i32>
    return %0 : tensor<i32>
  }
}

>>> lowered_with_x.compile()(5)
Array(19, dtype=int32, weak_type=True)

```

Note that `trace` here takes two arguments as usual, but the subsequent compiled
function accepts only the remaining non-static second argument. The static first
argument (value 7) is taken as a constant at lowering time and built into the
lowered computation, where it is possibly folded in with other constants. In
this case, its multiplication by 2 is simplified, resulting in the constant 14.

Although the second argument to `trace` above can be replaced by a hollow
shape/dtype structure, it is necessary that the static first argument be a
concrete value. Otherwise, tracing errs:

```python
>>> jax.jit(f, static_argnums=0).trace(i32_scalar, i32_scalar)  # doctest: +SKIP
Traceback (most recent call last):
TypeError: unsupported operand type(s) for *: 'int' and 'ShapeDtypeStruct'

>>> jax.jit(f, static_argnums=0).trace(10, i32_scalar).lower().compile()(5)
Array(25, dtype=int32)

```

The results of `trace` and of `lower` are not safe to serialize directly for use
in a different process. See {ref}`export` for additional APIs for this purpose.

## AOT-compiled functions cannot be transformed

Compiled functions are specialized to a particular set of argument "types," such
as arrays with a specific shape and element type in our running example. From
JAX's internal point of view, transformations such as {func}`jax.vmap` alter the
type signature of functions in a way that invalidates the compiled-for type
signature. As a policy, JAX simply disallows compiled functions to be involved
in transformations. Example:

```python
>>> def g(x):
...   assert x.shape == (3, 2)
...   return x @ jnp.ones(2)

>>> def make_z(*shape):
...   return jnp.arange(np.prod(shape)).reshape(shape)

>>> z, zs = make_z(3, 2), make_z(4, 3, 2)

>>> g_jit = jax.jit(g)
>>> g_aot = jax.jit(g).trace(z).lower().compile()

>>> jax.vmap(g_jit)(zs)
Array([[ 1.,  5.,  9.],
       [13., 17., 21.],
       [25., 29., 33.],
       [37., 41., 45.]], dtype=float32)

>>> jax.vmap(g_aot)(zs)  # doctest: +SKIP
Traceback (most recent call last):
TypeError: Cannot apply JAX transformations to a function lowered and compiled for a particular signature. Detected argument of Tracer type <class 'jax._src.interpreters.batching.BatchTracer'>

```

A similar error is raised when `g_aot` is involved in autodiff
(e.g. {func}`jax.grad`). For consistency, transformation by `jax.jit` is
disallowed as well, even though `jit` does not meaningfully modify its
argument's type signature.


## Debug information and analyses, when available

In addition to the primary AOT functionality (separate and explicit lowering,
compilation, and execution), JAX's various AOT stages also offer some additional
features to help with debugging and gathering compiler feedback.

For instance, as the initial example above shows, lowered functions often offer
a text representation. Compiled functions do the same, and also offer cost and
memory analyses from the compiler. All of these are provided via methods on the
{class}`jax.stages.Lowered` and {class}`jax.stages.Compiled` objects (e.g.,
`lowered.as_text()` and `compiled.cost_analysis()` above).
You can obtain more debugging information, e.g., source location,
by using the `debug_info` parameter to `lowered.as_text()`.

These methods are meant as an aid for manual inspection and debugging, not as a
reliably programmable API. Their availability and output vary by compiler,
platform, and runtime. This makes for two important caveats:

1. If some functionality is unavailable on JAX's current backend, then the
   method for it returns something trivial (and `False`-like). For example, if
   the compiler underlying JAX does not provide a cost analysis, then
   `compiled.cost_analysis()` will be `None`.

2. If some functionality is available, there are still very limited guarantees
   on what the corresponding method provides. The return value is not required
   to be consistent---in type, structure, or value---across JAX configurations,
   backends/platforms, versions, or even invocations of the method. JAX cannot
   guarantee that the output of `compiled.cost_analysis()` on one day will
   remain the same on the following day.

When in doubt, see the package API documentation for {mod}`jax.stages`.
