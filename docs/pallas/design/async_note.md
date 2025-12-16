(pallas_async)=
# Pallas Async Operations

## Background \+ Motivation

We’d like to expose APIs in Pallas to explicitly overlap computation and communication *across multiple kernels*.

### XLA Async Decomposition

As motivation, consider the following JAX pseudocode:

```py
def f(x):
  y = ppermute(x)
  z = x + 1
  return y, z
```

In this function, we could perform the `ppermute` at the same time as the `x + 1`. This is an optimization XLA does automatically by:

1. decomposing `ppermute` into a `ppermute_start` and `ppermute_done` op, which are connected via a future.  
2. scheduling the `x + 1` between the `ppermute_start` and `ppermute_done`,

resulting in the following program:

```py
def f(x):
  fut = ppermute_start(x)
  z = x + 1  # happens at the same time as ppermute
  y = ppermute_done(fut)
  return y, z
```

### Async ops inside kernels

Now imagine we aren’t using XLA’s `ppermute` but have our own custom Pallas `ppermute`.

```py
def ppermute_kernel(x_ref, y_ref, send_sem, recv_sem):
  right_neighbor = ...
  descriptor = pltpu.make_async_remote_copy(x_ref, y_ref, send_sem, recv_sem, device_id=right_neighbor)
  descriptor.start()
  descriptor.wait_send()
  descriptor.wait_recv()

def ppermute(x):
  return pl.pallas_call(ppermute_kernel, out_shape=x, ...)(x)
```

Currently, we cannot decompose `ppermute` into a `start/done` pair as XLA does, so instead we explicitly **fuse** the `x + 1` into the kernel.

```py
def add_one(x_ref, z_ref):
  z_ref[...] = x_ref[...] + 1

def ppermute_add_one_kernel(x_ref, y_ref, z_ref, send_sem, recv_sem):
  right_neighbor = ...
  descriptor = pltpu.make_async_remote_copy(x_ref, y_ref, send_sem, recv_sem, device_id=right_neighbor)
  descriptor.start()

  # Explicitly schedule inner kernel between start/wait
  pltpu.emit_pipeline(add_one)(x_ref, z_ref)

  descriptor.wait_send()
  descriptor.wait_recv()

def ppermute_and_add_one(x):
  return pl.pallas_call(ppermute_add_one_kernel, out_shape=(x, x), ...)(x)

```

The goal is to enable writing separate kernels for starting the `ppermute` and waiting on it to complete, so that we can use a regular old `x + 1` in between (or whatever compute we want). This makes the code more readable, maintainable, and less bug-prone.

## How do we implement decomposed Pallas async operations (on TPU)?

The main thing to figure out when implementing decomposed async operations in Pallas is what the `future` that is passed between them contains. Specifically, it must contain some important state about the operation happening in the background.

If we look at the Pallas code, we can see that we need a “descriptor” to both start and wait on a remote copy. Can we plumb this descriptor out of the Pallas kernel, and then pass it into another one? Well kinda. The underlying TPU hardware tracks async op progress via a pair of semaphores: `send_sem` enables us to wait on when a device is done sending data to its neighbor and `recv_sem` tracks the data transfer sent to a device from their neighbor. If we imagine writing a start kernel and a done kernel, all we’d need to pass from the start to the done would be the semaphores and some information about how much to wait on those semaphores.

We can do this via extending Pallas to support returning semaphores from kernels.

```py
def ppermute_start_kernel(
    in_ref, send_sem, recv_sem, out_ref, *, axis_name,
):
  axis_size = jax.lax.psum(1, axis_name)
  left_neighbor = jax.lax.rem(
      jax.lax.axis_index(axis_name) - 1 + axis_size, axis_size
  )
  right_neighbor = jax.lax.rem(jax.lax.axis_index(axis_name) + 1, axis_size)
  barrier_sem = pltpu.get_barrier_semaphore()
  pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor)
  pltpu.semaphore_wait(barrier_sem, 1)
  pltpu.make_async_remote_copy(
      in_ref, out_ref, send_sem, recv_sem, device_id=right_neighbor
  ).start()

def ppermute_start(x, *, axis_name) -> tuple[Semaphore, Semaphore, Array]:
  send_sem, recv_sem, out = pl.pallas_call(
      functools.partial(ppermute_start_kernel, axis_name=axis_name),
      out_shape=(
          pltpu.SemaphoreType.DMA(()),
          pltpu.SemaphoreType.DMA(()),
          jax.ShapeDtypeStruct(
              x.shape,
              dtype=x.dtype,
          ),
      ),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.ANY),
      ],
      out_specs=(
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.ANY),
      ),
  )(x)
  return send_sem, recv_sem, out
```

Note that something subtle is happening here. Pallas is telling XLA that it would like some outputs to be semaphores (a.k.a. sync flags) and XLA will treat them as “reserved” (e.g. while they are alive in the XLA program, those sync flags cannot be allocated by other kernels). They behave similarly to barrier semaphores, which are reserved semaphores managed by XLA.

Another thing to notice is that we return the output buffer `out` from the start kernel *while it’s being actively copied into*.

Now we write the `done` kernel that performs the blocking operation. We pass `out` into the kernel to compute the shape needed to block on the semaphore.

```py
def ppermute_done_kernel(ref, send_sem, recv_sem, _):
  pltpu.make_async_copy(ref, ref, send_sem).wait()
  pltpu.make_async_copy(ref, ref, recv_sem).wait()

def ppermute_done(send_sem, recv_sem, out) ->Array:
  out = pl.pallas_call(
      ppermute_done_kernel,
      out_shape=(
          jax.ShapeDtypeStruct(
              out.shape,
              dtype=out.dtype,
          ),
      ),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.ANY),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
      input_output_aliases={0:0}
  )(out, send_sem, recv_sem)
  return out
```

Note: we i/o alias the output buffer here to guarantee that the consumers are downstream of the `ppermute_done`.

We now can implement the decomposed collective permute.

```py
def f(x):
  fut = ppermute_start(x)
  z = x + 1  # happens at the same time as ppermute
  y = ppermute_done(fut)
  return y, z
```

***OR CAN WE?***

## Why *doesn’t* this work?

There are three remaining issues with this, each of which exists outside of Pallas to some degree. Here they are at a high level.

1. Scheduling \- just because we write `ppermute_start`, then `x + 1`, then `ppermute_done` doesn’t guarantee that they will happen in that order. XLA is responsible for scheduling, so when we write JAX programs, we are setting up data dependencies that XLA will respect but XLA will not respect the specific order of operations written in JAX.  
2. Lifetimes \- XLA assumes that once a value is out of scope in the dependency graph, its memory can be freed for use by other values. If we have an op that asynchronously copies x \-\> y, we need to ensure that x is alive until the copy is complete, otherwise we will be copying from garbage memory.  
3. Defensive copies \- XLA reserves the right to create copies of values. We need to make sure we don’t introduce unnecessary copies to a) avoid unnecessary runtime overhead and b) ensure correctness. 

We will go over these issues one by one and suggest fixes.

### Scheduling

How do we explicitly force ops to happen in a particular order in JAX? Note that this is not a Pallas specific problem, and if we had async ops implemented using an alternative method, we’d still run into this.

One way is to introduce an optimization barrier into the XLA program. The optimization barrier will prevent XLA moving ops around it.

Here’s our original code:

```py
def f(x):
  fut = ppermute_start(x)
  z = x + 1
  y = ppermute_done(fut)
  return y, z
```

XLA could choose to execute `x + 1` in any of three places:

```py
def f(x):
  z = x + 1
  fut = ppermute_start(x)
  y = ppermute_done(fut)
  return y, z

# OR

def f(x):
  fut = ppermute_start(x)
  z = x + 1
  y = ppermute_done(fut)
  return y, z

# OR

def f(x):
  fut = ppermute_start(x)
  y = ppermute_done(fut)
  z = x + 1
  return y, z
```

To force the `x + 1` to happen between the `ppermute` ops, we can use `optimization_barrier`, which is semantically the identity function (i.e. `lambda x: x`) but introduces an explicit data dependency between values. Specifically, if we make the `x` that is used in `x + 1` dependent on the `fut` returned by `ppermute_start`, it must happen after `ppermute_start`.

We also introduce a dependency that forces the output value `y` to depend on `z`.

```py
def f(x):
  fut = ppermute_start(x)
  x, fut = optimization_barrier((x, fut))  # x now depends on fut
  z = x + 1
  z, fut = optimization_barrier((z, fut)) # fut now depends on z
  y = ppermute_done(fut)
  return y, z
```

`optimization_barrier` is a good enough hammer for us to explicitly write out schedules.

### Lifetimes

Let’s look at our original code again and assume the ops are happening in the correct order.

```py
def f(x):
  fut = ppermute_start(x)
  z = x + 1
  y = ppermute_done(fut)
  return y, z
```

Let’s look at which point in the program XLA believes it is okay to free the buffer for `x`. It would be the point after which `x` is no longer used, specifically after `z = x + 1`.

```py
def f(x):
  fut = ppermute_start(x)
  z = x + 1
  # XLA can free x here!
  y = ppermute_done(fut)
  return y, z
```

If XLA frees `x` after `z = x + 1` has completed, we run into a very bad problem. The `ppermute` could still be actively copying `x` to the neighbor after `z = x + 1` which means if `x` is freed, the `ppermute` will be reading from garbage memory\!

How do we extend `x`’s lifetime to the `ppermute_done`? Well we can introduce a data dependency\! We need to modify our kernels a little bit to make this happen.

First, we rewrite `ppermute_start` to return `x`, aliasing it through the kernel.

```py
def ppermute_start_kernel(
    in_ref, send_sem, recv_sem, out_ref, _, *, axis_name,
):
  axis_size = jax.lax.psum(1, axis_name)
  left_neighbor = jax.lax.rem(
      jax.lax.axis_index(axis_name) - 1 + axis_size, axis_size
  )
  right_neighbor = jax.lax.rem(jax.lax.axis_index(axis_name) + 1, axis_size)
  barrier_sem = pltpu.get_barrier_semaphore()
  pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor)
  pltpu.semaphore_wait(barrier_sem, 1)
  pltpu.make_async_remote_copy(
      in_ref, out_ref, send_sem, recv_sem, device_id=right_neighbor
  ).start()

def ppermute_start(x, *, axis_name) -> tuple[Semaphore, Semaphore, Array, Array]:
  send_sem, recv_sem, x, out = pl.pallas_call(
      functools.partial(ppermute_start_kernel, axis_name=axis_name),
      out_shape=(
          pltpu.SemaphoreType.DMA(()),
          pltpu.SemaphoreType.DMA(()),
          jax.ShapeDtypeStruct(
              x.shape,
              dtype=x.dtype,
          ),
	   jax.ShapeDtypeStruct(
              x.shape,
              dtype=x.dtype,
          ),
      ),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.ANY),
      ],
      out_specs=(
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.ANY),
          pl.BlockSpec(memory_space=pltpu.ANY),
      ),
      input_output_aliases={0:2}
  )(x)
  return send_sem, recv_sem, x, out
```

We then have `ppermute_done` take in `x` and do nothing with it.

```py
def ppermute_done_kernel(_, ref, send_sem, recv_sem, _):
  pltpu.make_async_copy(ref, ref, send_sem).wait()
  pltpu.make_async_copy(ref, ref, recv_sem).wait()

def ppermute_done(send_sem, recv_sem, x, out) ->Array:
  out = pl.pallas_call(
      ppermute_done_kernel,
      out_shape=(
          jax.ShapeDtypeStruct(
              out.shape,
              dtype=out.dtype,
          ),
      ),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.ANY),
          pl.BlockSpec(memory_space=pltpu.ANY),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
      input_output_aliases={1:0}
  )(x, out, send_sem, recv_sem)
  return out

```

Now when we write

```py
def f(x):
  *sems, x ,out = ppermute_start(x)
  z = x + 1
  y = ppermute_done(*sems, x, out)
  return y, z
```

XLA can no longer free `x` because it is an input to `ppermute_done`\! This means that `x`’s lifetime is tied to the `ppermute` and this code is now correct.

### Defensive copies

XLA, in its buffer assignment pass, analyzes which buffers are aliased to each other and inserts copies whenever an operation that aliases one of its inputs is not the final consumer of that input.

#### Background

Here’s a simple example. Let’s say we have an op `add_one_inplace` which takes in an array and adds one, but promises to do it in-place.

The following code would be legal.

```py
def f():
  x = jnp.arange(...)
  y = add_one_inplace(x)  return y
```

However, if `x` had a separate consumer as well, the program may not execute correctly.

```py
def f():
  x = jnp.arange(...)
  y = add_one_inplace(x)
  return y, x * 2 # another x consumer!
```

This is because `x * 2` operates on the original `x` but `add_one_inplace` clobbers the value in `x`. `x * 2` needs to make sure to read the original values of `x`, not the ones after we’ve incremented it by 1\. XLA notices this and inserts a `copy` op (which is semantically the identity but the input and output buffers will be different).

```py
def f(x):
  x2 = copy(x)
  y = add_one_inplace(x2)
  return y, x * 2
```

This pass in XLA ensures correctness in the presence of ops that perform in-place updates by forcing them to effectively be out-of-place with `copy` ops.

#### Copies with downstream ops

Let’s revisit our example where we add 1 while `ppermute`ing.

```py
def f(x):
  fut = ppermute_start(x)
  z = x + 1
  y = ppermute_done(fut)
  return y, z
```

If we unpack the future into its components, we’ll see the the aliasing patterns:

```py
def f(x):
  *sems, x2, y = ppermute_start(x)
  z = x + 1
  y = ppermute_done((*sems, x2, y))
  return y, z
```

We know that `x` is left unchanged by `ppermute_start` (that is, `x` is identical to `x2`), but XLA does not. In fact, it looks like our `add_one_inplace` example to XLA, where it conservatively assumes that `ppermute_start` mutated `x` and `x2` is the new aliased result. Therefore, when we do `z = x + 1`, we run into a consumer of the original buffer. XLA therefore introduces a copy\!

```py
def f(x):
  x2 = copy(x)
  *sems, x2, y = ppermute_start(x2)
  z = x + 1
  y = ppermute_done((*sems, x2, y))
  return y, z
```

This copy is unnecessary because we know that `x2` is unchanged from `x`. In order to remove this copy, we’d need some mechanism to inform XLA we are just forwarding a value. However, in the absence of that we can rewrite our program a bit to explicitly use `x2` instead of `x`.

```py
def f(x):
  *sems, x2, y = ppermute_start(x)
  z = x2 + 1
  y = ppermute_done((*sems, x2, y))
  return y, z
```

Now, XLA doesn’t see a separate consumer of `x` so no more copy is introduced. However, this comes at a major downside in that it forces us to unpack the future coming from `ppermute_start`. It couples the lifetime problem to the copying problem.

#### Loop aliasing

Let’s consider a slightly more advanced example. Let’s implement a function that uses a `while_loop` with `ppermute` to send values around a ring.

```py
def f(x):
  def body(i, x):
    fut = ppermute_start(x)
    y = ppermute_done(fut)
    return y
  return fori_loop(0, 8, body, x)
```

One implementation detail of `fori_loop` is that the inputs and outputs buffers are automatically aliased to each other. Note that we are setting up some additional aliasing in the `ppermute_start` and `ppermute_done` ops. Let’s run our own “buffer assignment” by coloring each of the values in the program to determine how many unique buffers we need.

First, we’ll unpack the `fut` tuple that has the aliased `x` and `out` buffers.

```py
def f(x):
  def body(i, x):
    *sems, x, y = ppermute_start(x)
    y = ppermute_done(*sems, x, y)
    return y
  return fori_loop(0, 8, body, x)
```

Let’s now color each of the values according to the unique buffer they are assigned. We have the input/output aliasing coming from `fori_loop`, the `x` aliasing coming from `ppermute_start` and the `y` aliasing coming from `ppermute_done`.

```py
def f(x):
  def body(i, x):
    *sems, x, y = ppermute_start(x)
    y = ppermute_done((*sems, x, y))
    return y
  return fori_loop(0, 8, body, x)
```

If you run the alias analysis, you’ll find that all of the buffers have been colored the same\! Intuitively, this is problematic because if we are doing a loop of `ppermute`s, we can’t write into the same buffer we are sending into. We generally need an extra (i.e. a “double”) buffer to receive, and then usually we will switch the send/recv buffers on the next iteration. What XLA will do in practice is that it will observe the buffer reuse and defensively insert a copy.

```py
def f(x):
  def body(i, x):
    x = copy(x)
    *sems, x, y = ppermute_start(x)
    y = ppermute_done((*sems, x, y))
    return y
  return fori_loop(0, 8, body, x)
```

This copy means `x` and `y` are no longer aliased to each other and the program will be correct. However, do we need this copy? How do we introduce a double buffer to avoid expensive copies each iteration? The answer is unrolling\!

We’ll manually unroll our code.

```py
def f(x):
  def body(i, x):
    *sems, x, x2 = ppermute_start(x)
    x2 = ppermute_done((*sems, x, x2))
    
    *sems, x2, y = ppermute_start(x2)
    y = ppermute_done((*sems, x2, y))
    return y
  return fori_loop(0, 4, body, x)
```

Now if we were to run the same alias analysis, we’ll find that the buffers all no longer alias to each other and that we won’t need to insert defensive copies to be correct.

Therefore, the simple solution to removing these copies is to use `fori_loop` with `unroll >= 2`.

```py
def f(x):
  def body(i, x):
    fut = ppermute_start(x)
    y = ppermute_done(fut)
    return y
  return fori_loop(0, 8, body, x, unroll=2)
```

That’s sufficient to implement this loop without extra copies\!

#### Passing futures across loop boundaries

Let’s now look at an even more advanced example. We’ll implement the same program as before but stagger the loop, where we begin the `ppermute` in a prologue before the loop, and wait on the `ppermute` at the beginning of the loop.

```py
def f(x):
  fut = ppermute_start(x)
  def body(i, fut):
    x = ppermute_done(fut)
    fut = ppermute_start(x)
    return fut
  fut = fori_loop(0, 7, body, fut)
  return ppermute_done(fut)
```

In this example, rather than passing a value `x` from one loop to another we are passing a future value.

Let’s unpack the future again to see what’s happening.

```py
def f(x):
  fut = ppermute_start(x)
  def body(i, fut):
    *sems, x, out = fut
    x = ppermute_done((*sems, x, out))
    (*sems, x, out) = ppermute_start(x)
    return (*sems, x, out)
  (*sems, x, out) = fori_loop(0, 7, body, x)
  return ppermute_done((*sems, x, out))
```

So we’re explicitly threading the semaphores, the input buffer, and the target output buffer as a loop carry. What happens if we run alias analysis now? Well, we’ll run into the same aliasing issue as in the previous section where `x` and `out` will be aliased to each other. XLA will introduce a copy.

```py
def f(x):
  fut = ppermute_start(x)
  def body(i, fut):
    *sems, x, out = fut
    out = copy(out)
    x = ppermute_done((*sems, x, out))
    (*sems, x, out) = ppermute_start(x)
    return (*sems, x, out)
  (*sems, x, out) = fori_loop(0, 7, body, x)
  return ppermute_done((*sems, x, out))
```

In this case, we inserted a copy on `out`. However, this is a really bad scenario because `out` is being actively copied into\! Even if we insert a copy on `x`, we will also run into issues because then `x`’s lifetime will not extend to the `ppermute_done`. This is very very bad\! We will not only get copies, but we will also get incorrect results\!

The solution, as we observed before, is to avoid the copies by avoiding aliasing all the buffers via unrolling. So, if we do:

```py
def f(x):
  fut = ppermute_start(x)
  def body(i, fut):
    x = ppermute_done(fut)
    fut = ppermute_start(x)
    return fut
  fut = fori_loop(0, 7, body, x, unroll=2)
  return ppermute_done(fut)
```

our program should now be correct.

### Putting it all together

So we’ve come up with some rules of thumb:

1. If we have operations dependent on the input value to the `ppermute`, unpack the future to use the aliased value instead of the original value.  
2. Use `unroll >= 2` when doing `ppermute`s in a loop body.

Let’s combine everything into one function that does `ppermute`s in a loop and accumulates the result. 

```py
def f(x):
  out = jnp.zeros_like(x)
  fut = (*sems, x, out) = ppermute_start(x)
  out = out + x
  def body(i, carry):
    out, fut = carry
    x = ppermute_done(fut)
    fut = (*sems, x, out) = ppermute_start(x)
    out = out + x
    return out, fut
  out, fut = fori_loop(0, 7, body, (out, fut), unroll=2)
  return out, ppermute_done(fut)
```

Note that in this example, we don’t need `optimization_barrier`s because the loop boundary acts as a scheduling barrier, splitting up the `start`s and `done`s.

That’s it, we are done\! This will be the official API for doing async ops in Pallas. Thank you everyone\! Mission accomplished\!

***OR IS IT?***

## Revenge of the State

While it seems we have worked around copies and incorrectness issues by using some clever tricks, we are still in an awkward position. This API is powerful, but has many many footguns and caveats. There are likely far many more edge cases we will need to deal with that  even require deep knowledge of XLA to predict or understand. Should we release an API like this? Or is there an alternative?

Well, the answer may have been in front of us this whole time.

Let’s run through this whole exercise one more time, *except*, let’s write the stateful version. This means each of our custom async ops now operate on `Ref`s instead of values.

```py
def ppermute_start_stateful(x_ref, y_ref) -> tuple[Semaphore, Semaphore]:
  ...

def ppermute_done_stateful(send_sem, recv_sem, x_ref, y_ref) -> None:
  ...
```

Let’s assume we can implement these in Pallas and see what our new programs will look like. Let’s start with a basic collective permute:

```py
def f(x):
  x_ref = make_ref(x)
  y_ref = make_ref(zeros_like(x))
  fut = ppermute_start_stateful(x_ref, y_ref)
  ppermute_done_stateful(*fut, x_ref, y_ref)
  return y_ref[...]
```

It’s a little bit more verbose than our original value-based version, but it has a few key differences. The first is that we create an “empty” `Ref` to receive the result of the `ppermute`, unlike the value-based version, which creates a value for us. One neat thing is that the lifetime of `x_ref` is clear here: it lives until `ppermute_done_stateful`. We don’t need to “sneak” the `x` value into the op like we did before.

Another difference becomes more clear when we try adding an op between the `start/done`.

```py
def f(x):
  x_ref = make_ref(x)
  y_ref = make_ref(zeros_like(x))
  fut = ppermute_start_stateful(x_ref, y_ref)
  x_ref[...] += 1
  ppermute_done_stateful(*fut, x_ref, y_ref)
  return y_ref[...]
```

Before, we ran into scheduling ambiguity, where XLA could re-order the add w.r.t. the `ppermute`. With stateful semantics, we actually add in an ordering constraint\! `x_ref[...] += 1` mutates `x_ref` so it can’t be moved wrt to `ppermute_done_stateful`. JAX can inject these scheduling constraints as part of the lowering to HLO. 

The final key difference is evident when we try our loop examples.

```py
def f(x):
  x_ref = make_ref(x)
  y_ref = make_ref(zeros_like(x))
  def body(i, _):
    fut = ppermute_start_stateful(x_ref, y_ref)
    ppermute_done_stateful(*fut, x_ref, y_ref)
    # Now switch to y_ref -> x_ref
    fut = ppermute_start_stateful(y_ref, x_ref)
    ppermute_done_stateful(*fut, y_ref, x_ref)
  fori_loop(0, 8 // 2, body, None)
  return x_ref[...]
```

Because of the requirement that we have a separate buffer ready to receive the `ppermute`, we were forced to write our code in such a way that unrolls it\! There is no way to write the version in XLA that requires copying because that would involve a `ppermute` that sends from a `Ref` into itself, which doesn’t really make sense.

To handle this without the manual unrolling, we’d create a scratch buffer with a leading `2` dimension that acts as the send/recv target across iterations, switching each one. This is the same pattern we use internally in Pallas kernels when writing manually overlapped kernels.

The realization here is that being stateful forces us to deal with a lot of the issues that pop up with value semantics earlier on. We define them away\!

1. Scheduling \- stateful ops that have `Ref`s as inputs force an ordering of our program. Note that this will schedule operations on the same `Ref` wrt to each other. We might also need an `opt_barrier_stateful` to enforce more ordering constraints.  
2. Lifetimes \- `Ref` lifetimes can be scoped via `run_state` or could be inputs to stateful ops.  
3. Defensive copies \- Using `Ref`s forces us to handle buffer assignment “manually” and the lowering can ensure the aliasing works out to avoid any copies.

Another important fundamental limitation is that we eventually stage out an HLO program where the live buffers and semaphores are represented as array value types. XLA does not provide guarantees about buffer lifetimes or which memory spaces they live in for these intermediate values. *Therefore, it is possible XLA can copy array values even if they are actively being copied into by Pallas kernels.* This is easy to verify in HLO but it is a sharp edge of using custom calls to represent asynchronous operations in HLO.

## Conclusion

We’ve gone over some tricky challenges when it comes to async ops in Pallas and JAX. `Ref`s seem like a promising way of representing these ops that circumvents some of the issues that come up with value semantics. However, a downside is that it puts stateful JAX front and center, which we haven’t done yet outside of Pallas. It’s worth thinking whether we should educate users about stateful ops, or provide a more dangerous API. We also don’t know if everything we want to do is expressible via `Ref`s as well. We should also brainstorm alternatives to state to flesh out the design space. For example, what if XLA offered a first-class futures API that respected lifetimes, and it could automatically do things like double buffer loops with futures in them? That might be a viable alternative but the tradeoff would be giving more control to the compiler vs explicit control from the user.
