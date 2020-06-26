# Heap Profiling

The JAX Heap Profiler allows us to explore how and why JAX programs are using
GPU or TPU memory. For example, it can be used to:

* Figure out which arrays and executables are in GPU memory at a given time, or
* Track down memory leaks.

## Installation

The JAX heap profiler emits output that can be interpreted using the
[`pprof`](https://github.com/google/pprof) tool. Start by installing `pprof`,
by following its
[installation instructions](https://github.com/google/pprof#building-pprof).
At the time of writing, installing `pprof` requires first installing
[Go](https://golang.org/) and [Graphviz](http://www.graphviz.org/), and then
running

```shell
go get -u github.com/google/pprof
```

which installs `pprof` as `$GOPATH/bin/pprof`, where `GOPATH` defaults to
`~/go`.

## Understanding how a JAX program is using GPU or TPU memory

A common use of the heap profiler is to figure out why a JAX program is using
a large amount of GPU or TPU memory, for example if trying to debug an
out-of-memory problem.

To capture a heap profile to disk, use
{func}`jax.profiler.save_heap_profile`. For example, consider the following
Python program:

```python
import jax
import jax.numpy as jnp
import jax.profiler

def func1(x):
  return jnp.tile(x, 10) * 0.5

def func2(x):
  y = func1(x)
  return y, jnp.tile(x, 10) + 1

x = jax.random.normal(jax.random.PRNGKey(42), (1000, 1000))
y, z = func2(x)

z.block_until_ready()

jax.profiler.save_heap_profile("heap.prof")
```

If we first run the program above and then execute

```shell
pprof --web heap.prof
```

`pprof` opens a web browser containing the following visualization of the heap
profile in callgraph format:

![Heap profiling example](_static/heap_profile.svg)

The pprof documentation explains
[how to interpret callgraph visualizations](https://github.com/google/pprof/blob/master/doc/README.md#interpreting-the-callgraph).

The callgraph is a visualization of
the Python stack at the point the allocation of each live buffer was made.
For example, in this specific case, the visualization shows that
`func2` and its callees were responsible for allocating 76.30MB, of which
38.15MB was allocated inside the call from `func1` to `func2`.

Functions compiled with {func}`jax.jit` are opaque to the heap profiler.
That is, any memory allocated inside a `jit`-compiled function will be
attributed to the function as whole.

In the example, the call to `block_until_ready()` is to ensure that `func2`
completes before the heap profile is collected. See {doc}`async_dispatch` for
more details.

## Debugging memory leaks

We can also use the JAX heap profiler to track down memory leaks by using
`pprof` to visualize the change in memory usage between two heap profiles
taken at different times. For example consider the following program which
accumulates JAX arrays into a constantly-growing Python list.

```python
import jax
import jax.numpy as jnp
import jax.profiler

def afunction():
  return jax.random.normal(jax.random.PRNGKey(77), (1000000,))

z = afunction()

def anotherfunc():
  arrays = []
  for i in range(1, 10):
    x = jax.random.normal(jax.random.PRNGKey(42), (i, 10000))
    arrays.append(x)
    x.block_until_ready()
    jax.profiler.save_heap_profile(f"heap{i}.prof")

anotherfunc()
```

If we simply visualize the heap profile at the end of execution (`heap9.prof`),
it may not be obvious that each iteration of the loop in `anotherfunc`
accumulates more heap allocations:

```shell
pprof --web heap9.prof
```

![Heap profile at end of execution](_static/heap_profile_leak1.svg)

The large but fixed allocation inside `afunction` dominates the profile but does
not grow over time.

By using `pprof`'s
[`--diff_base` feature](https://github.com/google/pprof/blob/master/doc/README.md#comparing-profiles) to visualize the change in memory usage
across loop iterations, we can identify why the memory usage of the
program increases over time:

```shell
pprof --web --diff_base heap1.prof heap9.prof
```

![Heap profile at end of execution](_static/heap_profile_leak2.svg)
