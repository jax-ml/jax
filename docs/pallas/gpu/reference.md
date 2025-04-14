# Writing Mosaic GPU kernels with Pallas

This page is a reference for the most important features of the Pallas:MGPU backend.
It's not a tutorial and as such we do not expect everyone to read it top to bottom.
Still, it is worth going over
just to familiarise yourself with some patterns you can find in other tutorials.

In the following examples, we're going to assume the following imports are in scope:
```python
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
```

## What is a GPU?

Technically, the NVIDIA GPU architecture looks as follows: the GPU is partitioned into
_streaming multiprocessors_ (SMs). The way this manifests in the CUDA programming model
is that each _CUDA thread block_ (or CTA) is scheduled on exactly one SM, but multiple
blocks can be scheduled onto a single SM at a time.

Each SM contains a chunk of fast memory called _shared memory_ (SMEM) and 4 subdivisions,
each containing a _warp scheduler_ and compute units (ALU, TensorCore, ...).
This is also reflected in the CUDA programs: each _warp_ (a group of consecutive 32 CUDA
threads in a block) is assigned to one of those subdivisions in a round-robin fashion.
Similarly to blocks, each warp is assigned to exactly one subdivision (it never migrates),
but multiple warps can be assigned to the same SM subdivision. At each clock cycle, the
warp scheduler from each subdivision tries to select one of its resident warps to execute
the next instruction.

<center><img alt="A diagram of one NVIDIA SM" src="../../_images/nvidia_sm.svg" style="width:60%; min-width: 400px;"></center>

Going further, recent CUDA versions also outline the concept of a _warpgroup_, which are
4 consecutive warps. Knowing how the hardware looks like, we can see where this is comming
from: 4 consecutive warps occupy the 4 quarters of an SM and let us issue instructions
that utilize the whole SM.

> A GPU can be viewed in many different ways and in here we want to focus on a slightly
  simplified model that is very TensorCore-centric. This should help you navigate the
  complexities of writing kernels involving the TensorCore, but keep in mind that the
  real picture is more complicated.

For our purposes, TensorCore operations have grown so big that it no longer makes much
sense to follow the CUDA model. As such, to us, a GPU is a collection of single-threaded cores
(SMs) with one thread of Pallas:MGPU corresponding to a CUDA warpgroup. In this model, each
operation you perform in the kernel occupies the whole CUDA warpgroup, and its constituent
warps always run in lockstep (modulo the jitter from hardware scheduling) and never take
different paths through control flow (with the small exception of `core_map` that we will
discuss later). One notable addition here is that we still allow you to co-schedule multiple
of those Pallas-level threads on the same SM so that they can cooperate and communicate
through shared memory (we relize that by putting them in the same CUDA block).

> From now on, whenever we say "thread", we refer to the Pallas thread, not a CUDA thread/lane.

> This is very similar to a programming model popularized by [Triton](https://triton-lang.org/),
  but as you will see there are a few differences. Mosaic GPU tends to be more low level,
  which usually means you will have to put in more work, but it also puts you more in control.
  In our view both approaches have their merits and we encourage you to pick the backend that
  suits your needs the best! Pallas supports and will continue to support Triton as an alternative
  GPU backend.

### In-order execution & using multiple hardware units

Unlike more complicated CPU architectures GPU only support in-order execution. That, however,
does not mean that at any given time only a single instruction is running! Each SM quarter
has multiple independent functional units: TensorCore, Arithmetic logic unit (ALU),
Load/Store (LSU), Special function unit (SFU).  If the first instruction targets one of the
units and is followed by another one (that does not use the result of the first one), then the
warp scheduler can issue the second one before the first one completes. This is often referred
to as instruction-level parallelism (ILP) and is a common theme in modern TensorCore kernels:
TensorCore operations are so big and take so many cycles to complete, that it is a waste to not
try to use other units in the meantime.

To extend this even further, we can take advantage of this hardware-unit-level parallelism by
allowing multiple Pallas threads to run concurrently. If one of the threads primarily
occupies the ALU, while another one primarily issues TensorCore related instructions, we can
take advantage of the efficient context switching built into the warp schedulers to keep both
units busy. This is one of the core idea behind algorithms such as [FlashAttention 3](https://arxiv.org/abs/2407.08608)
or [CUTLASS ping-pong matmul kernels](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/).

For more information on how warp scheduling and instruction issue works, we recommend reading
[Analyzing Modern NVIDIA GPU cores](https://arxiv.org/abs/2503.20481).

### Memory spaces

TODO: GMEM, SMEM, RMEM, (maybe TMEM)

## Array layouts and reference transforms

TODO

## MMA (TensorCore)

In this section, we focus on how Pallas:MGPU kernels can utilize the TensorCore unit.
The programming interface of the TensorCore changes significantly between different
NVIDIA GPU generations, which is why the lowest-level interfaces differ in Pallas:MGPU as well.

Each MMA operation is associated with three operands:
* the accumulator `D` of shape `(M, N)`,
* the left input `A` of shape `(M, K)`,
* the right input `B` of shape `(K, N)`.
All operands must have the same element type.

Each use of MMA involves a few steps:
1. Allocating the space for the accumulator (MMA implicitly performs `D += A @ B`)
2. Preparing the `A` and `B` operands
3. Issuing the operation
4. Waiting for the operation to complete
5. Reading out the result

Steps 2.-4. are usually performed in a loop over the contraction dimension (`K`).

### Memory space of `A` and `B` operands

The `A` and `B` operands are generally best passed in through SMEM, where they can
be conveniently loaded using `plgpu.copy_gmem_to_smem`. For those operands to be
compatible with MMA operations, they need to have the appropriate tiling and swizzling
transforms specified upon their allocation. For all currently supported generations,
the TensorCore requires the data to be laid out into row-major 2D tiles of shape
`(8, swizzle_elems)`, where `swizzle_elems` is derived by dividing the swizzle by the
element type bytewidth.  The currently supported swizzles are: 128, 64, and 32. Larger
swizzles are preferrable as they improve the performance of GMEM-to-SMEM copies.

```python
def mma_transforms(shape_dtype: jax.ShapeDtypeStruct):
  assert len(shape_dtype.shape) == 2
  if shape_dtype.shape[0] % 8:
    raise ValueError("Number of rows must be divisible by 8")
  for swizzle_bytes in (128, 64, 32):
    swizzle_elems = swizzle_bytes // shape_dtype.dtype.itemsize
    if shape_dtype.shape[-1] % swizzle_elems == 0:
      return (plgpu.TilingTransform((8, swizzle_elems)),
              plgpu.SwizzleTransform(swizzle_bytes))
  raise ValueError("Failed to find transforms for the specified window type")
```

If the operands need to be transformed, the `A` operand can be passed in through a different
memory space (architecture dependent, see below). The `B` operand _must_ be located in SMEM.

### Transposed operands

When performing MMA on 16-bit operands, the TensorCore can automatically transpose the
input data. For example, the `A` reference is allowed to be of shape `(K, M)`, but it
has to be transposed before passing it into the mma function. For example:
```python
assert acc_ref.shape == (M, N) and a_ref.shape == (K, M) and b_ref.shape == (K, N)
a_ref_t = plgpu.transpose_ref(a_ref, (1, 0))
assert a_ref_t.shape == (M, K)  # The shape expected by plgpu.wgmma
plgpu.wgmma(acc, a_ref_t, b_ref)
```
An analogous operation is allowed on the `B` reference in this case too.

### Hopper (`wgmma`)

In this section, we cover the basics of using the Hopper-generation TensorCores, exposed in
PTX as the [`wgmma.mma_async` instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).

#### Allocating the accumulator

In the Hopper hardware architecture the accumulator is allocated in registers, but in Pallas
it is modeled as a mutable reference, as each MMA operation accumulates in-place.
There are two ways to allocate the accumulator.

To create a zero-initialized accumulator you can use `pl.run_scoped` with a
`plgpu.ACC((m, n), dtype)` type.
```python
def compute(acc_ref):
  ...
  return acc_ref[...]
output = pl.run_scoped(compute, plgpu.ACC((m, n), jnp.float32))
```
Dereferencing the accumulator reference, as seen in the end of the `compute` function will
implicitly await all outstanding WGMMA operations.

If you'd like to initialize it with an existing array, you can use `pl.run_state` with
`plgpu.ACC.init(init_array)`:
```python
def compute(acc_ref):
  ...
  return # pl.run_state only returns the final value of the accumulator
output = pl.run_state(compute)(plgpu.ACC.init(init_array))
```
If `pl.run_state` has accumulator operands, it implicitly awaits all outstanding WGMMA
operations before returning the final values.

#### Preparing the `A` and `B` operands

As discussed above, we recommend passing in `A` and `B` through shared memory. In this
case the correct tiling and swizzling transforms must be specified.

`plgpu.wgmma` additionally allows passing in `A` through registers (i.e. not an SMEM
reference but as a regular JAX array). This mode, however, comes with a number of
significant drawbacks and it is very difficult to ensure sufficient synchronization to
make this safe.

TODO: Explain the conditions under which it is acceptable to do this.

#### Issuing the operation

The supported MMA shapes are such that:
* `M` is divisible by 64
* `N` is divisible by 8 and smaller than 256
* `K` is a multiple of `swizzle` divided by the bytewidth of element type

The currently supported data types are: `jnp.float32`, `jnp.bfloat16` and `jnp.float16`.
The accumulator `D` must be a `jnp.float32`, with the exception of `jnp.float16` inputs,
in which case it is allowed to be `jnp.float16` as well.

#### Waiting for the operation to complete

Each `plgpu.wgmma` call implicitly synchronizes with all previous `plgpu.wgmma` calls, such
that once control returns from it, we guarantee that no WGMMA other than the last issued
one is still running. As such, any SMEM regions that were read by previously issued WGMMA
instructions can be reused. This is especially relevant for pipelining WGMMA with async memory copies:
```python
buffers = 3  # In reality you might want even more
assert a_smem.shape == (buffers, m, k)
assert b_smem.shape == (buffers, k, n)
assert acc_ref.shape == (m, n)

def fetch_a_b(ki, slot):
  a_slice = ... # Replace with the right M/K slice
  b_slice = ... # Replace with the right K/N slice
  plgpu.copy_gmem_to_smem(a_gmem.at[a_slice], a_smem.at[slot], a_loaded.at[slot])
  plgpu.copy_gmem_to_smem(b_gmem.at[b_slice], b_smem.at[slot], b_loaded.at[slot])

def loop_body(i, _):
  slot = jax.lax.rem(i, buffers)
  plgpu.barrier_wait(a_loaded.at[slot])
  plgpu.barrier_wait(b_loaded.at[slot])
  plgpu.wgmma(acc_ref, a_smem.at[slot], b_smem.at[slot])
  # We know that only the last issued WGMMA is running, so we can issue a async load in
  # into the other buffer
  load_i = i + buffers - 1
  load_slot = jax.lax.rem(load_i, buffers)
  @pl.when(jnp.logical_and(load_i >= buffers, load_i < num_steps))
  def _do_fetch():
    fetch_a_b(load_i, slot)
for slot in range(buffers):
  fetch_a_b(slot, slot)
jax.lax.fori_loop(0, num_steps, loop_body, None)
```

### Blackwell (`tcgen05`)

While Mosaic GPU supports `tcgen05` MMA instructions, exposing this capability to Pallas
is still work in progress. Stay tuned!

## Using `core_map`

TODO

## Synchronization structures and primitives

In this section, we go over the most important functions and data structures
used for synchronization between threads and also some asynchronous operations.

### `commit_smem`

Regular reads/writes to references are guaranteed to produce values consistent
with the sequential program order. For example, in the following program, it is
guaranteed that `value` is equal to `value2`.
```python
ref[...] = value
value2 = ref[...]
```

This guarantee, however, does not extend to asynchronous primitives such as async
copies or MMA operations. To make the SMEM writes visible to those primitives, you
are required to explicitly synchronize with them using the `plgpu.commit_smem()` function.

For example:
```python
smem_ref[...] = value
plgpu.commit_smem()
plgpu.copy_smem_to_gmem(smem_ref, ...)
```
or:
```python
smem_ref[...] = value
plgpu.commit_smem()
plgpu.wgmma(smem_ref, ...)
```

Failing to call this function is likely to cause subtle data races, due to those asynchronous
hardware units reading stale data from SMEM. Unfortunately, this function is relatively expensive,
which is why we rely on you, the user, to insert it in the minimal number of places where it's necessary.

### `Barrier`

This is essentially a thin wrapper around an array of PTX `mbarrier` types and is
passed in as a reference. All functions involving barriers expect to only get a single
barrier argument, and so if the reference contains multiple, you have to extract one
of them explicitly using `barriers.at[index]`. `Barrier`s are always allocated in SMEM
and as such have relatively low overheads. Each barrier can be configured to complete
after a fixed number of "arrivals" (by default 1).

To block a thread until a barrier completes, use the following function:
```python
plgpu.barrier_wait(barrier)
```

There are three operations that can complete a barrier:

> It is critical to ensure that the synchronization scheme makes it impossible for two
  barrier completions to happen without a call to `plgpu.barrier_wait` in between them.
  For example, if you use `Barrier`s to synchronize two producer/consumer threads, you
  need to perform barrier synchronization going both ways to introduce "backpressure"
  that will stop one thread from arriving twice before the other one had a chance to await.
  Failing to satisfy this will corrupt the data structure and can cause surprising failures
  (including CUDA runtime errors). See below for an example of a valid program with two threads.

> Another critical restriction is that the number of barrier completions must equal the
  number of barrier waits throughout the barrier's lifetime. It is not allowed to end a scoped
  allocation of a barrier when it has an unawaited completion. Otherwise, when it is
  reused by the compiler, leaving it in this state can cause problems downstream.

> Finally, it is crucial to ensure that each thread that ever waits on a `Barrier`
  takes part in all `wait` operations on it. It is not allowed to e.g. await every
  other completion of a barrier from one thread, and all other completions from another
  one. Doing so will lead to deadlocks. To recap: when a `Barrier` is used to wait in
  some thread, it must observe every single completion of that barrier (by waiting on it).


  Note that the `Barrier` can receive arrivals from any source, without restrictions.

#### Asynchronous GMEM-to-SMEM copies

When an asynchronous GMEM-to-SMEM copy is being executed by the TMA engine, it will
post progress updates to the barrier given to `plgpu.copy_gmem_to_smem`. Once the copy
is complete, the barrier will complete one arrival as well.

#### Explicit arrival (cross-thread synchronization)

Any thread can explicitly arrival on a barrier using the following function:
```python
plgpu.barrier_arrive(barrier)
```

This is especially useful when synchronizing two threads that are in producer/consumer
roles. In this case, we recommend allocating two arrays of `Barrier`s, with size equal
to the size of the "queue" used to pass data between the two threads. For example,
assume one thread continues writing tiles of an array to SMEM while another thread
reads them. We triple-buffer the SMEM region to allow more asynchrony between the two
threads:

```python
tid = jax.lax.axis_index("thread")
assert queue.shape == (buffering, *item_shape)
assert produced.shape == consumed.shape == (buffering,)

def thread0_body(i, _):
  slot = jax.lax.rem(i, buffering)
  @pl.when(i >= buffering)
  def _await_consumed():
    plgpu.barrier_wait(consumed.at[slot])  # Wait for consumption of the value before overwriting it
  # Option 1: Compute the next value
  queue[slot] = produce()
  plgpu.barrier_arrive(produced.at[slot])  # Signal the value is ready
  # Option 2: Produce the value through async_copy
  # plgpu.copy_gmem_to_smem(..., queue.at[slot], barrier=produced.at[slot])
pl.when(tid == 0)(lambda: jax.lax.fori_loop(0, steps, thread0_body, None))

def thread1_body(i, _):
  slot = jax.lax.rem(i, buffering)
  plgpu.barrier_wait(produced.at[slot])  # Wait for the value to be ready
  consume(queue[slot])  # Load and compute
  plgpu.barrier_arrive(consumed.at[slot])  # Signal that the value is consumed
pl.when(tid == 1)(lambda: jax.lax.fori_loop(0, steps, thread1_body, None))
```

#### Awaiting `tcgen05` TensorCore instructions

While Mosaic GPU supports `tcgen05` MMA instructions, exposing this capability to Pallas
is still work in progress. Stay tuned!

### `ClusterBarrier`

TODO

### `Semaphore`

TODO

## Asynchronous copies

TODO

## Inline Mosaic GPU

TODO

## Compiler parameters

TODO
