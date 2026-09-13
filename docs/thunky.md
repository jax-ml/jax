# Thunky User Guide

`thunky` is an experimental Python DSL for programming the **GPU runtime command schedule** directly from JAX.

When programming GPUs in JAX, there are three levels of abstraction:

1. **Declarative array math (`@jax.jit`)**: Purely declarative—you write the math (`jnp.dot`, `jnp.sin`), and the compiler figures out how to run it (kernel fusion, memory allocation, stream scheduling, and collective overlap).
2. **GPU kernels (`Pallas` / `Mosaic GPU` / `Triton`)**: You write device code that executes on GPU streaming multiprocessors (warps, registers, shared memory, Tensor Cores, TMA).
3. **GPU command schedules (`thunky`)**: Between declarative array math and individual GPU kernels is the **runtime schedule**—the sequence of kernel launches, cuBLAS/cuDNN calls, async memory copies, multi-stream synchronization events, device control flow, and NCCL collectives dispatched to the GPU.

In XLA:GPU, this runtime schedule is represented as a sequence of **thunks**. Each thunk is an atomic GPU command executed by the runtime: *"launch this kernel on stream 0"*, *"start an NCCL AllGather on stream 1"*, *"wait for stream 1 on stream 0"*, or *"call cuBLAS GEMM"*.

`thunky` lets you author this GPU command schedule imperatively using mutable buffer references (`Ref`s), named GPU streams, and explicit collective primitives. `thunky` programs compile ahead-of-time into XLA's GPU execution graph and compose with both `@jax.jit` and custom Mosaic GPU / PTX kernels.

| Level | Tool | Abstraction | What You Control |
| :--- | :--- | :--- | :--- |
| **Math** | `@jax.jit` / `@shard_map` | Declarative array math | Array operations (`jnp.dot`, `lax.psum`); compiler decides how to run them (fusion, memory allocation, stream ordering) |
| **Schedule** | `@thunky.jit` | GPU runtime command sequence | Buffer lifetimes (`Ref`), kernel & library call ordering (`call_jax`), CUDA streams (`async_start`/`async_done`), device control flow, and NCCL collectives |
| **Kernel** | `Pallas` / `Mosaic GPU` | Intra-kernel SM execution | Thread/warp scheduling, shared memory, registers, TMA transfers, and Tensor Core instructions inside a single GPU kernel |

---

## Hello World: Mutable Buffers (`Ref`s) and Basic Commands

Unlike standard JAX functions that take immutable `jax.Array`s and return new arrays, `@thunky.jit` functions operate imperatively on **mutable device buffers** represented as `Ref`s. A `thunky` function mutates its buffer arguments in-place and returns `None`.

Let's write a simple `thunky` program that zeroes a buffer, fills a slice with a 32-bit pattern, and copies data between buffers:

```python
import jax
import jax.numpy as jnp
from jax.experimental import thunky
import numpy as np

@thunky.jit
def init_and_copy(src_ref, dst_ref):
  # Zero the entire destination buffer (cudaMemsetAsync)
  thunky.memzero(dst_ref)
  # Copy the first 4 elements from src_ref into dst_ref (cudaMemcpyAsync)
  thunky.copy(src_ref.at[:4], dst_ref.at[:4])
  # Fill the last 4 elements with float32 1.0
  thunky.memset(dst_ref.at[4:], np.float32(1.0))
```

### Calling a `@thunky.jit` Function from JAX

When calling a `@thunky.jit` function from inside a JAX function (`@jax.jit`):

- **Inside the `@thunky.jit` function**, every parameter arrives as a `Ref` (plus any trailing `Ref` parameters from `scratch_shapes`), and the function returns `None`.
- **Mutated arguments** (outputs or in-place updates) must be passed by the caller as `Ref`s—created with `jax.new_ref(...)` (for example, `jax.new_ref(jnp.empty(...))` or `jax.new_ref(x)`), or `.at[...]` views of a `Ref`. After the `@thunky.jit` call returns, read the result back as a `jax.Array` using `out_ref[...]`. Passing a plain `jax.Array` for an argument that is mutated in-place raises a `TypeError` at trace time.
- **Read-only arguments** can be passed either as `Ref`s or directly as plain `jax.Array`s (which `thunky` automatically wraps in a `Ref` for the body of the `@thunky.jit` function).

```python
@jax.jit
def run(x):
  # Allocate an output buffer as a Ref:
  out_ref = jax.new_ref(jnp.empty_like(x))
  # `x` is read-only so it can be passed as a jax.Array;
  # `out_ref` is mutated in-place so it must be passed as a Ref.
  init_and_copy(x, out_ref)
  # Dereference `out_ref` back into a jax.Array:
  return out_ref[...]

x = jnp.arange(8, dtype=jnp.float32)
print(run(x))
# [0. 1. 2. 3. 1. 1. 1. 1.]
```

### Temporary Workspace Buffers (`scratch_shapes`)

Many GPU programs require temporary scratch buffers (for example, staging buffers for double-buffered communication). You can request temporary buffers via `scratch_shapes` on `@thunky.jit`. Scratch buffers are passed as additional trailing `Ref` arguments to your function:

```python
@thunky.jit(scratch_shapes=[jax.ShapeDtypeStruct((4,), jnp.float32)])
def swap_halves(buf_ref, scratch_ref):
  thunky.copy(buf_ref.at[:4], scratch_ref)
  thunky.copy(buf_ref.at[4:], buf_ref.at[:4])
  thunky.copy(scratch_ref, buf_ref.at[4:])
```

---

## Mixing `thunky` and JAX: `thunky.call_jax`

`thunky` and JAX functions compose in any combination:

- **Calling `thunky` from JAX**: Call a `@thunky.jit` function inside `@jax.jit`, passing `jax.new_ref(...)` buffers for any arguments mutated in-place and reading outputs back via `ref[...]` (read-only inputs may be passed as either `Ref`s or `jax.Array`s).
- **Calling JAX from `thunky`**: Call a standard JAX function inside a `@thunky.jit` function using `thunky.call_jax(fn, *args)`.
- **Calling `thunky` from `thunky`**: Call one `@thunky.jit` function directly inside another `@thunky.jit` function.

### Calling JAX from `thunky` (`thunky.call_jax`)

`thunky.call_jax(fn, *args)` compiles `fn` into GPU operations and splices them directly into the enclosing `thunky` function. It accepts both mutable `Ref`s and `jax.Array` values:

- Arguments passed as `Ref`s (or `.at[...]` views) arrive inside `fn` as mutable `Ref`s, which can be read with `ref[...]` and written in-place with `ref[...] = value`.
- Arguments passed as `jax.Array`s (for example, `ref[:]`) arrive inside `fn` as standard `jax.Array` values.
- If `fn` returns one or more `jax.Array` values, `thunky.call_jax` allocates output buffers and returns the corresponding `Ref`s to the enclosing `thunky` function.

**Mutating `Ref` arguments in-place:**

```python
def scale_and_add(x_ref, y_ref, out_ref):
  out_ref[...] = jnp.sin(x_ref[...]) * 2.0 + y_ref[...]

@thunky.jit
def pipeline(x_ref, y_ref, out_ref):
  thunky.memzero(out_ref)
  thunky.call_jax(scale_and_add, x_ref, y_ref, out_ref)
```

**Passing `jax.Array` values and returning new `Ref`s:**

```python
@thunky.jit
def matmul_and_bias(a_ref, b_ref, out_ref):
  # a_ref[:] and b_ref[:] are passed to the lambda as standard jax.Arrays;
  # the returned jax.Array is wrapped in a newly allocated Ref `c_ref`.
  c_ref = thunky.call_jax(lambda a, b: a @ b, a_ref[:], b_ref[:])

  # Mix value input (c_ref[:]) and in-place Ref output (out_ref):
  def add_bias(c_val, out_r):
    out_r[...] = c_val + 5.0

  thunky.call_jax(add_bias, c_ref[:], out_ref)
```

### Nested `@thunky.jit` Functions

You can call one `@thunky.jit` function directly inside another. The inner function's command sequence is spliced into the outer function, and any `scratch_shapes` declared by inner functions are automatically hoisted into the outer allocation plan:

```python
def double_from_scratch(tmp_ref, x_ref):
  x_ref[...] = tmp_ref[...] * 2.0

@thunky.jit(scratch_shapes=[jax.ShapeDtypeStruct((4,), jnp.float32)])
def helper_step(x_ref, tmp_ref):
  thunky.copy(x_ref, tmp_ref)
  thunky.call_jax(double_from_scratch, tmp_ref, x_ref)

@thunky.jit
def outer_fn(x_ref):
  helper_step(x_ref.at[:4])
  helper_step(x_ref.at[4:])
```

---

## Zero-Copy Buffer Views & Slicing

In `thunky`, calling `.at[slice]` on a `Ref` creates a **zero-copy buffer view** (`TransformedRef`) representing a byte offset and sub-shape within the underlying allocation. No memory is copied when creating or passing a view:

```python
def matmul_chunk(a_ref, b_ref, out_ref):
  out_ref[...] = a_ref[...] @ b_ref[...]

@thunky.jit
def tiled_gemm(a_ref, b_ref, out_ref):
  # Compute top half: out[:4, :] = a[:4, :] @ b
  thunky.call_jax(matmul_chunk, a_ref.at[:4, :], b_ref, out_ref.at[:4, :])
  # Compute bottom half: out[4:, :] = a[4:, :] @ b
  thunky.call_jax(matmul_chunk, a_ref.at[4:, :], b_ref, out_ref.at[4:, :])
```

Because XLA:GPU buffer allocations are flat, unstrided byte ranges (`offset_bytes`, `size_bytes`), `.at[...]` only supports slices that form a **contiguous byte range** in row-major memory:

- **Supported**: Slicing or integer-indexing along the leading dimension with unit stride (`ref.at[start:stop]`, `ref.at[start:stop, :]`, or `ref.at[i]`), as well as chaining along sub-leading dimensions after earlier dimensions are fixed by integer indexing (`ref.at[i].at[start:stop]`).
- **Unsupported**: Slicing along trailing dimensions (`ref.at[:, 2:4]`) or using non-unit strides (`ref.at[::2]`) raises a `ValueError` at trace time.

For contiguous memory copies between slices, you can also use direct Python slice assignment on `Ref`s inside `@thunky.jit`:

```python
@thunky.jit
def copy_slices(src_ref, dst_ref):
  dst_ref[:4] = src_ref[4:8]  # Lowers to a zero-copy slice + thunky.copy
```

---

## Runtime Control Flow

### Conditional Execution (`thunky.cond` and `thunky.switch`)

`thunky.cond` and `thunky.switch` execute device-side conditionals (captured as conditional nodes inside a CUDA Graph). The GPU evaluates the scalar predicate directly in device memory and branches without returning control to the Python host. Both branch functions are zero-argument closures that capture any needed buffer `Ref`s:

```python
def negate_in_place(x_ref):
  x_ref[...] = -x_ref[...]

@thunky.jit
def maybe_negate(pred_ref, data_ref):
  thunky.cond(
      pred_ref,
      lambda: thunky.call_jax(negate_in_place, data_ref),
      lambda: None,  # No-op false branch
  )

@jax.jit
def run_cond(pred, x):
  data_ref = jax.new_ref(x)
  maybe_negate(pred, data_ref)
  return data_ref[...]
```

For multi-way branching indexed by an `int32` scalar buffer, use `thunky.switch(index_ref, [branch0, branch1, ...])`.

### Device Loops (`thunky.while_loop`)

`thunky.while_loop(cond_buf, cond_fn, body_fn)` executes a device-driven loop (captured as a loop node inside a CUDA Graph) without host synchronization:

- `cond_buf` is a scalar `bool` buffer `Ref` in GPU memory (often allocated via `scratch_shapes`).
- `cond_fn()` is a zero-argument closure that updates `cond_buf`.
- `body_fn()` is a zero-argument closure that executes one iteration of the loop body while `cond_buf` is true.

```python
def check_positive(i_ref, cond_out_ref):
  cond_out_ref[...] = i_ref[...] > 0

def decrement_and_double(i_ref, v_ref):
  i_ref[...] = i_ref[...] - 1
  v_ref[...] = v_ref[...] * 2.0

@thunky.jit(scratch_shapes=jax.ShapeDtypeStruct((), jnp.bool_))
def power_of_two_loop(iters_ref, val_ref, cond_buf):
  thunky.while_loop(
      cond_buf,
      lambda: thunky.call_jax(check_positive, iters_ref, cond_buf),
      lambda: thunky.call_jax(decrement_and_double, iters_ref, val_ref),
  )

@jax.jit
def run_loop(iters, val):
  iters_ref = jax.new_ref(iters)
  val_ref = jax.new_ref(val)
  power_of_two_loop(iters_ref, val_ref)
  return val_ref[...]
```

### Nested Control Flow: Mixing `while_loop` and `cond`

`thunky.cond` and `thunky.while_loop` can be nested arbitrarily—for example, placing a `thunky.cond` inside a `thunky.while_loop` body to compute the Collatz stopping time (number of steps until `n` reaches `1`, applying `n // 2` if even and `3 * n + 1` if odd):

```python
def check_gt_one(n_ref, cond_ref):
  cond_ref[...] = n_ref[...] > jnp.int32(1)

def check_is_even(n_ref, even_ref):
  even_ref[...] = (n_ref[...] % jnp.int32(2)) == jnp.int32(0)

def collatz_even(n_ref):
  n_ref[...] = n_ref[...] // jnp.int32(2)

def collatz_odd(n_ref):
  n_ref[...] = jnp.int32(3) * n_ref[...] + jnp.int32(1)

def increment_steps(steps_ref):
  steps_ref[...] = steps_ref[...] + jnp.int32(1)

@thunky.jit(
    scratch_shapes=(
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.bool_),
        jax.ShapeDtypeStruct((), jnp.bool_),
    )
)
def collatz_length(n_in_buf, steps_out_buf, n_buf, loop_cond_buf, is_even_buf):
  thunky.copy(n_in_buf, n_buf)
  thunky.memzero(steps_out_buf)

  def loop_body():
    thunky.call_jax(check_is_even, n_buf, is_even_buf)
    thunky.cond(
        is_even_buf,
        lambda: thunky.call_jax(collatz_even, n_buf),
        lambda: thunky.call_jax(collatz_odd, n_buf),
    )
    thunky.call_jax(increment_steps, steps_out_buf)

  thunky.while_loop(
      loop_cond_buf,
      lambda: thunky.call_jax(check_gt_one, n_buf, loop_cond_buf),
      loop_body,
  )

@jax.jit
def run_collatz(n):
  steps_ref = jax.new_ref(jnp.int32(0))
  collatz_length(n, steps_ref)
  return steps_ref[...]

print(run_collatz(jnp.int32(27)))
# 111
```

---

## Multi-Stream Concurrency (`async_start` and `async_done`)

Independent operations can be overlapped across concurrent CUDA streams—most commonly overlapping **communication** (NCCL collectives on a dedicated communication stream) with **computation** (GEMMs on computation streams) using cross-stream CUDA events (`cudaEventRecord` / `cudaStreamWaitEvent`).

In `thunky`, streams are selected by integer `stream_id` and `stream_kind` (`"computation"` or `"communication"`):

- `token = thunky.async_start(fn, *, stream_id=0, stream_kind="computation")` records an event on the current stream, waits for that event on the target stream, launches `fn()` on the target stream, and returns a synchronization token representing completion of `fn()` on that stream.
- `thunky.async_done(token)` records a CUDA event on `token`'s stream and makes the current stream wait for that event (`cudaStreamWaitEvent`) before launching subsequent work.

```python
def compute_exp(x_ref, out_ref):
  out_ref[...] = jnp.exp(x_ref[...])

def compute_cos(y_ref, out_ref):
  out_ref[...] = jnp.cos(y_ref[...])

@thunky.jit
def concurrent_branches(a_ref, b_ref, out_a_ref, out_b_ref):
  # Fork work onto computation stream 0 and computation stream 1
  tok_a = thunky.async_start(
      lambda: thunky.call_jax(compute_exp, a_ref, out_a_ref),
      stream_id=0,
  )
  tok_b = thunky.async_start(
      lambda: thunky.call_jax(compute_cos, b_ref, out_b_ref),
      stream_id=1,
  )

  # Join both streams back to the main stream
  thunky.async_done(tok_a)
  thunky.async_done(tok_b)
```

---

## Custom GPU Kernels & FFI (`ptx_kernel`, `mosaic_gpu_kernel`, `custom_call`)

In addition to splicing JAX computations via `thunky.call_jax`, `thunky` can directly invoke custom low-level GPU kernels and XLA FFI custom call handlers—raw PTX assembly (`thunky.ptx_kernel`), **Mosaic GPU** kernels (`thunky.mosaic_gpu_kernel`), or registered XLA FFI handlers (`thunky.custom_call`).

### Launching Raw PTX Kernels (`thunky.ptx_kernel`)

`thunky.ptx_kernel` launches a PTX kernel directly onto the active CUDA stream. You provide the buffer `Ref` arguments, a `written` boolean list indicating which buffers are mutated in-place, the kernel symbol name, PTX source string, launch grid/block dimensions, and optional dynamic shared memory size (`shmem_bytes`):

```python
ADD_ONE_PTX = """
.version 7.0
.target sm_80
.address_size 64

.visible .entry add_one_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr
) {
    .reg .pred %p0;
    .reg .f32 %f<2>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd0, %r1, 4;
    add.u64 %rd1, %rd1, %rd0;
    add.u64 %rd2, %rd2, %rd0;

    ld.global.f32 %f0, [%rd1];
    add.f32 %f1, %f0, 0f3F800000;  // + 1.0f
    st.global.f32 [%rd2], %f1;
    ret;
}
"""

@thunky.jit
def run_ptx_add_one(in_ref, out_ref):
  thunky.ptx_kernel(
      in_ref,
      out_ref,
      written=[False, True],
      kernel_name="add_one_kernel",
      ptx=ADD_ONE_PTX,
      grid_dim=(1, 1, 1),
      block_dim=(4, 1, 1),
  )
```

### Launching Mosaic GPU Kernels (`thunky.mosaic_gpu_kernel`)

`thunky.mosaic_gpu_kernel` compiles a Mosaic GPU kernel function (supporting Tensor Memory Accelerator (TMA) async copies, shared memory barriers, and WGMMA instructions) and embeds its launch directly into the `thunky` function.

You can freely interleave `thunky.mosaic_gpu_kernel` with `thunky.call_jax` or stream primitives in the same `@thunky.jit` program:

```python
from jax.experimental.mosaic import gpu as mgpu

shape = (128, 128)
dtype = jnp.float32

def tma_copy_kernel(ctx, src, dst, scratch):
  smem, barrier = scratch
  # Asynchronous TMA copy from global memory (src) to shared memory (smem)
  ctx.async_copy(src_ref=src, dst_ref=smem, barrier=barrier)
  barrier.wait()
  # Asynchronous TMA copy from shared memory (smem) back to global memory (dst)
  ctx.async_copy(src_ref=smem, dst_ref=dst)
  ctx.await_async_copy(0)

def scale_and_shift(a_ref, out_ref):
  out_ref[...] = a_ref[...] * 3.0 + 1.0

@thunky.jit(scratch_shapes=jax.ShapeDtypeStruct(shape, dtype))
def mosaic_and_jax_pipeline(in_ref, out_ref, intermediate_ref):
  # Step 1: Launch custom Mosaic GPU TMA kernel into intermediate_ref
  thunky.mosaic_gpu_kernel(
      tma_copy_kernel,
      grid=(1, 1, 1),
      block=(128, 1, 1),
      in_shape=jax.ShapeDtypeStruct(shape, dtype),
      out_shape=jax.ShapeDtypeStruct(shape, dtype),
      smem_scratch_shape=(
          jax.ShapeDtypeStruct(shape, dtype),
          mgpu.TMABarrier(),
      ),
      operands=[in_ref],
      results=[intermediate_ref],
  )
  # Step 2: Post-process intermediate_ref using standard JAX math
  thunky.call_jax(scale_and_shift, intermediate_ref, out_ref)
```

### Calling XLA FFI Handlers (`thunky.custom_call`)

`thunky.custom_call` invokes any registered XLA FFI custom call handler on input (`operands`) and output (`results`) `Ref`s, passing typed attributes via `backend_config`.

For example, you can call JAX's registered cuSOLVER Cholesky factorization handler (`"cusolver_potrf_ffi"`) directly on a symmetric matrix buffer:

```python
# Ensure JAX's cuSOLVER FFI handlers are registered:
_ = jnp.linalg.cholesky

@thunky.jit(scratch_shapes=jax.ShapeDtypeStruct((), jnp.int32))
def cholesky_in_place(a_ref, info_ref):
  # Pass `a_ref` as both operand and result to factorize strictly in-place.
  # Because `a_ref` is symmetric and cuSOLVER uses column-major layout,
  # lower=False writes the column-major upper triangle, which is the row-major
  # lower-triangular factor L such that A = L @ L.T.
  thunky.custom_call(
      "cusolver_potrf_ffi",
      operands=[a_ref],
      results=[a_ref, info_ref],
      backend_config={"lower": False},
  )

@jax.jit
def run_cholesky(a):
  a_ref = jax.new_ref(a)
  cholesky_in_place(a_ref)
  return jnp.tril(a_ref[...])

m = jnp.array([[4.0, 12.0, -16.0],
               [12.0, 37.0, -43.0],
               [-16.0, -43.0, 98.0]], dtype=jnp.float32)
print(run_cholesky(m))
# [[ 2.  0.  0.]
#  [ 6.  1.  0.]
#  [-8.  5.  3.]]
```

---

## Multi-GPU Programming: Manual Mode (`shard_map`) & Collectives

### The Per-GPU Mental Model

When writing a `@thunky.jit` function, you are **always writing per-GPU code**—just like writing a single rank's function in MPI + CUDA/NCCL. Buffer shapes inside `thunky` are the per-GPU local shard shapes, and every command you emit executes on each GPU in your mesh.

Because `thunky` operates on per-GPU buffers, multi-GPU `thunky` functions must be invoked inside `@jax.shard_map` (JAX's single-program multiple-data (SPMD) per-device execution mode). Calling `thunky` inside an automatic sharding context (`with jax.set_mesh(mesh):` without `shard_map`) raises a `ValueError`.

### Explicit `thunky` Collective Primitives

`thunky` provides direct bindings to NCCL collective and point-to-point operations:

| Primitive | Signature | Description |
| :--- | :--- | :--- |
| **`all_reduce`** | `thunky.all_reduce(src_ref, dst_ref, op="sum", axis_name="x")` | Reduces `src_ref` across `axis_name` (`"sum"`, `"prod"`, `"min"`, `"max"`) into `dst_ref` (`ncclAllReduce`) |
| **`all_gather`** | `thunky.all_gather(src_ref, dst_ref, axis_name="x", axis=0)` | Gathers shards from `src_ref` along `axis` across `axis_name` into `dst_ref` (`ncclAllGather`) |
| **`reduce_scatter`** | `thunky.reduce_scatter(src_ref, dst_ref, op="sum", axis_name="x", axis=0)` | Reduces `src_ref` across `axis_name` and scatters slices along `axis` into `dst_ref` (`ncclReduceScatter`) |
| **`all_to_all`** | `thunky.all_to_all(src_ref, dst_ref, replica_groups=((0, 1),))` | Splits `src_ref` and exchanges chunks across `replica_groups` into `dst_ref` |
| **`collective_permute`** | `thunky.collective_permute(src_ref, dst_ref, source_target_pairs=[(0, 1), (1, 0)])` | Point-to-point send/receive between explicit `(source_rank, target_rank)` pairs (`ncclSend` / `ncclRecv`) |
| **`collective_group`** | `thunky.collective_group(fn)` | Brackets nested collectives inside zero-argument closure `fn` within `ncclGroupStart()` / `ncclGroupEnd()` so NCCL fuses them into a single launch |

```python
from jax.sharding import Mesh, PartitionSpec as P
from jax import shard_map

mesh = Mesh(jax.devices()[:2], ("x",))

@thunky.jit
def sum_across_gpus(x_ref, out_ref):
  thunky.all_reduce(x_ref, out_ref, reduction="sum", replica_groups=((0, 1),))

@jax.jit
@shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
def run_all_reduce(x_local):
  out_ref = jax.new_ref(jnp.zeros_like(x_local))
  sum_across_gpus(x_local, out_ref)
  return out_ref[...]
```

### Grouping Collectives (`thunky.collective_group`)

In CUDA/NCCL, issuing multiple independent `ncclSend`/`ncclRecv` or collective calls inside `ncclGroupStart()` and `ncclGroupEnd()` aggregates them into a single NCCL kernel launch, allowing NCCL to drive multiple NVLink links concurrently.

In `thunky`, wrapping multiple collective calls inside `thunky.collective_group(fn)` brackets them in a single `ncclGroupStart()` / `ncclGroupEnd()` scope:

```python
@thunky.jit
def grouped_exchange_and_reduce(a_ref, b_ref, out_a_ref, out_b_ref):
  thunky.collective_group(
      lambda: (
          thunky.all_reduce(a_ref, out_a_ref, reduction="sum", replica_groups=((0, 1),)),
          thunky.collective_permute(
              b_ref, out_b_ref, source_target_pairs=((0, 1), (1, 0))
          ),
      )
  )
```

### Calling JAX Collectives Inside `thunky.call_jax`

Because `thunky` runs inside `@shard_map`, any function passed to `thunky.call_jax` automatically inherits the per-device SPMD context. You can freely use JAX collectives like `jax.lax.psum` or `jax.lax.all_gather` (even inside a `@jax.jit`-decorated helper) without needing a redundant `@shard_map` decorator on the helper function:

```python
@jax.jit
def jax_collective_helper(a_ref, out_ref):
  out_ref[...] = jax.lax.psum(a_ref[...], "x") * 2.0

@thunky.jit
def splice_jax_collective(x_ref, out_ref):
  thunky.call_jax(jax_collective_helper, x_ref, out_ref)
```

---

## Putting It All Together: Ring Collective Matmul

This example implements **AllGather + Matrix Multiplication** ($$C = \text{AllGather}(A) \times B$$) across $$N$$ GPUs in a ring, overlapping point-to-point communication with GEMM computation.

### Problem Setup

Suppose we have $$N$$ GPUs along mesh axis `"x"`:

- Matrix $$A$$ of global shape $$(N \cdot M_{\text{block}}, K)$$, partitioned across GPUs by rows (`P("x", None)`). Each GPU rank $$d \in \{0, \dots, N-1\}$$ starts with row block $$A_d$$ of shape $$(M_{\text{block}}, K)$$.
- Matrix $$B$$ of global shape $$(K, N \cdot N_{\text{block}})$$, partitioned across GPUs by columns (`P(None, "x")`). Each GPU rank $$d$$ holds column slice $$B_d$$ of shape $$(K, N_{\text{block}})$$.
- Each GPU rank $$d$$ computes its output column slice $$C_d = A \times B_d$$ of shape $$(N \cdot M_{\text{block}}, N_{\text{block}})$$, where row block $$r$$ of $$C_d$$ is $$A_r \times B_d$$.

Running an `ncclAllGather` on $$A$$ followed by a single cuBLAS GEMM leaves the GPU compute units idle during the network transfer. Instead, we pipeline the transfer and compute across two CUDA streams: a **communication stream** (`stream_kind="communication"`) for NCCL point-to-point shifts (`thunky.collective_permute`) and a **computation stream** (`stream_kind="computation"`) for cuBLAS GEMMs (`thunky.call_jax`).

### Step 1: Unidirectional Ring Collective Matmul

In a unidirectional ring, at each step rank $$i$$ sends its current $$(M_{\text{block}}, K)$$ chunk of $$A$$ to rank $$(i + 1) \bmod N$$ (`source_target_pairs = [(i, (i + 1) % N) ...]`). After $$s$$ shifts, rank $$d$$ holds chunk $$A_{(d - s) \bmod N}$$, multiplies it by $$B_d$$, and writes the result into row block $$(d - s) \bmod N$$ of `c_out_buf`.

Rather than allocating $$N - 1$$ temporary buffers, we double-buffer using a fixed workspace of **2 ping-pong buffers** (`ring_bufs[0]` and `ring_bufs[1]`):

1. **Prologue**: Enqueue up to 2 initial shifts (`shift 1` into `ring_bufs[0]`, `shift 2` into `ring_bufs[1]`) on communication stream 0, and launch local `GEMM 0` ($$A_d \times B_d$$) on compute stream 0 concurrently.
2. **Steady-state pipeline**: At each step $$s = 1 \dots N - 1$$:
   - Compute stream 0 waits for `shift s` to finish (`thunky.async_done(tok_comm[s])`) and runs `GEMM s` reading from `ring_bufs[(s - 1) % 2]`.
   - Once `GEMM s` finishes reading `ring_bufs[(s - 1) % 2]`, calling `thunky.async_start` for `shift s + 2` records a cross-stream event wait (`cudaStreamWaitEvent`) so communication stream 0 safely reuses `ring_bufs[(s - 1) % 2]` while compute stream 0 moves on to `GEMM s + 1` on `ring_bufs[s % 2]`.

```python
import functools
import jax
from jax import shard_map
from jax.experimental import thunky
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

def make_ring_collective_matmul(mesh, m_block, k_dim, n_block, dtype):
  num_devices = mesh.size
  # Unidirectional ring: rank i sends to rank (i + 1) % N
  cw_pairs = tuple((i, (i + 1) % num_devices) for i in range(num_devices))

  # Double-buffered ping-pong workspace: at most 2 buffers of shape (m_block, k_dim)
  num_bufs = min(2, num_devices - 1)
  scratch_specs = tuple(
      jax.ShapeDtypeStruct((m_block, k_dim), dtype) for _ in range(num_bufs)
  )

  def matmul_scatter(lhs_ref, rhs_ref, dev_id_ref, c_out_ref, *, step):
    c_block = jnp.matmul(lhs_ref[...], rhs_ref[...])
    row_offset = ((dev_id_ref[...] - step) % num_devices) * m_block
    c_out_ref[...] = jax.lax.dynamic_update_slice(
        c_out_ref[...], c_block, (row_offset, jnp.int32(0))
    )

  @thunky.jit(scratch_shapes=scratch_specs)
  def ring_collective_matmul_prog(
      a_local_buf,
      b_local_buf,
      dev_id_buf,
      c_out_buf,
      *ring_bufs,
  ):
    tok_comm = {}

    def launch_shift(step):
      src = a_local_buf if step == 1 else ring_bufs[(step - 2) % num_bufs]
      dst = ring_bufs[(step - 1) % num_bufs]
      tok_comm[step] = thunky.async_start(
          lambda s_buf=src, d_buf=dst: thunky.collective_permute(
              s_buf, d_buf, source_target_pairs=cw_pairs
          ),
          stream_id=0,
          stream_kind="communication",
      )

    # Prologue: enqueue up to 2 initial shifts on communication stream 0
    for s in range(1, min(3, num_devices)):
      launch_shift(s)

    # Step 0: compute local GEMM 0 (A_d @ B_d) on compute stream 0 concurrently
    # with the initial ring transfers.
    thunky.call_jax(
        functools.partial(matmul_scatter, step=0),
        a_local_buf,
        b_local_buf,
        dev_id_buf,
        c_out_buf,
    )

    # Steady state: wait for shift s, compute GEMM s on buf[(s - 1) % 2],
    # and enqueue shift s + 2 into the newly freed buffer.
    for s in range(1, num_devices):
      thunky.async_done(tok_comm[s])
      thunky.call_jax(
          functools.partial(matmul_scatter, step=s),
          ring_bufs[(s - 1) % num_bufs],
          b_local_buf,
          dev_id_buf,
          c_out_buf,
      )
      if s + 2 < num_devices:
        launch_shift(s + 2)

  @jax.jit
  @shard_map(
      mesh=mesh,
      in_specs=(P("x", None), P(None, "x")),
      out_specs=P(None, "x"),
      check_vma=False,
  )
  def ring_collective_matmul_fn(a_local, b_local):
    dev_id = jnp.int32(jax.lax.axis_index("x"))
    c_out_ref = jax.new_ref(
        jax.lax.empty((m_block * num_devices, n_block), dtype=dtype)
    )
    ring_collective_matmul_prog(a_local, b_local, dev_id, c_out_ref)
    return c_out_ref[...]

  return ring_collective_matmul_fn
```

### Step 2: Bidirectional Ring with Grouped Collectives & Fused GEMMs

We can extend the unidirectional schedule in two ways while keeping the workspace bounded to **2 ping-pong buffers**:

1. **Bidirectional transfers**: Grouping a clockwise shift ($$i \to (i + 1) \bmod N$$) and a counter-clockwise shift ($$i \to (i - 1) \bmod N$$) inside `thunky.collective_group` (`ncclGroupStart()` / `ncclGroupEnd()`) drives both ring directions simultaneously and reduces the communication schedule to $$\lfloor (N - 1) / 2 \rfloor$$ paired steps (plus one single step when $$N$$ is even).
2. **Fused GEMM per pair**: At each paired step $$s$$, rank $$d$$ receives two chunks of $$A$$: $$A_{(d - s) \bmod N}$$ (clockwise) and $$A_{(d + s) \bmod N}$$ (counter-clockwise). By allocating 2 ping-pong buffers of shape $$(2 \cdot M_{\text{block}}, K)$$ (`pair_bufs[0]` and `pair_bufs[1]`) and passing sub-buffer views of their top half (`.at[:m_block]`) and bottom half (`.at[m_block:]`) to `collective_permute`, both transfers land adjacent in memory so cuBLAS multiplies the $$(2 \cdot M_{\text{block}}, K)$$ buffer by $$B_d$$ in a single GEMM call.

```python
def make_bidirectional_ring_collective_matmul(
    mesh, m_block, k_dim, n_block, dtype
):
  num_devices = mesh.size
  num_pairs = (num_devices - 1) // 2
  has_mid = (num_devices % 2) == 0

  cw_pairs = tuple((i, (i + 1) % num_devices) for i in range(num_devices))
  ccw_pairs = tuple((i, (i - 1) % num_devices) for i in range(num_devices))

  # Double-buffered ping-pong workspace: at most 2 buffers of shape (2 * m_block, k_dim)
  if num_pairs > 0:
    num_bufs = min(2, num_pairs)
    scratch_specs = tuple(
        jax.ShapeDtypeStruct((2 * m_block, k_dim), dtype)
        for _ in range(num_bufs)
    )
  else:
    num_bufs = 1
    scratch_specs = (jax.ShapeDtypeStruct((m_block, k_dim), dtype),)

  def single_matmul_scatter(lhs_ref, rhs_ref, dev_id_ref, c_out_ref, *, step):
    c_block = jnp.matmul(lhs_ref[...], rhs_ref[...])
    row_offset = ((dev_id_ref[...] - step) % num_devices) * m_block
    c_out_ref[...] = jax.lax.dynamic_update_slice(
        c_out_ref[...], c_block, (row_offset, jnp.int32(0))
    )

  def pair_matmul_scatter(
      pair_lhs_ref, rhs_ref, dev_id_ref, c_out_ref, *, step
  ):
    # Single cuBLAS GEMM of shape (2 * m_block, k_dim) @ (k_dim, n_block)
    c_pair = jnp.matmul(pair_lhs_ref[...], rhs_ref[...])
    row_cw = ((dev_id_ref[...] - step) % num_devices) * m_block
    row_ccw = ((dev_id_ref[...] + step) % num_devices) * m_block
    c_out = jax.lax.dynamic_update_slice(
        c_out_ref[...], c_pair[:m_block], (row_cw, jnp.int32(0))
    )
    c_out_ref[...] = jax.lax.dynamic_update_slice(
        c_out, c_pair[m_block:], (row_ccw, jnp.int32(0))
    )

  @thunky.jit(scratch_shapes=scratch_specs)
  def bidir_ring_collective_matmul_prog(
      a_local_buf,
      b_local_buf,
      dev_id_buf,
      c_out_buf,
      *pair_bufs,
  ):
    tok_comm = {}
    total_comm_steps = num_pairs + (1 if has_mid else 0)

    def launch_comm_step(step):
      if step <= num_pairs:
        src_cw = (
            a_local_buf
            if step == 1
            else pair_bufs[(step - 2) % num_bufs].at[:m_block]
        )
        dst_cw = pair_bufs[(step - 1) % num_bufs].at[:m_block]
        src_ccw = (
            a_local_buf
            if step == 1
            else pair_bufs[(step - 2) % num_bufs].at[m_block:]
        )
        dst_ccw = pair_bufs[(step - 1) % num_bufs].at[m_block:]

        def comm_pair(scw=src_cw, dcw=dst_cw, sccw=src_ccw, dccw=dst_ccw):
          thunky.collective_group(
              lambda: (
                  thunky.collective_permute(
                      scw, dcw, source_target_pairs=cw_pairs
                  ),
                  thunky.collective_permute(
                      sccw, dccw, source_target_pairs=ccw_pairs
                  ),
              )
          )

        tok_comm[step] = thunky.async_start(
            comm_pair,
            stream_id=0,
            stream_kind="communication",
        )
      else:
        # Middle step (for even N): single clockwise transfer into top half of ping-pong buffer
        src_mid = (
            pair_bufs[(num_pairs - 1) % num_bufs].at[:m_block]
            if num_pairs > 0
            else a_local_buf
        )
        dst_mid = (
            pair_bufs[(step - 1) % num_bufs].at[:m_block]
            if num_pairs > 0
            else pair_bufs[0]
        )
        tok_comm[step] = thunky.async_start(
            lambda s_buf=src_mid, d_buf=dst_mid: thunky.collective_permute(
                s_buf, d_buf, source_target_pairs=cw_pairs
            ),
            stream_id=0,
            stream_kind="communication",
        )

    # Prologue: enqueue up to 2 initial communication steps
    for s in range(1, min(3, total_comm_steps + 1)):
      launch_comm_step(s)

    # Step 0: compute local GEMM 0 on compute stream 0
    thunky.call_jax(
        functools.partial(single_matmul_scatter, step=0),
        a_local_buf,
        b_local_buf,
        dev_id_buf,
        c_out_buf,
    )

    # Steady state: wait for step s, run GEMM s, enqueue step s + 2
    for s in range(1, total_comm_steps + 1):
      thunky.async_done(tok_comm[s])
      if s <= num_pairs:
        thunky.call_jax(
            functools.partial(pair_matmul_scatter, step=s),
            pair_bufs[(s - 1) % num_bufs],
            b_local_buf,
            dev_id_buf,
            c_out_buf,
        )
      else:
        mid_buf = (
            pair_bufs[(s - 1) % num_bufs].at[:m_block]
            if num_pairs > 0
            else pair_bufs[0]
        )
        thunky.call_jax(
            functools.partial(single_matmul_scatter, step=s),
            mid_buf,
            b_local_buf,
            dev_id_buf,
            c_out_buf,
        )
      if s + 2 <= total_comm_steps:
        launch_comm_step(s + 2)

  @jax.jit
  @shard_map(
      mesh=mesh,
      in_specs=(P("x", None), P(None, "x")),
      out_specs=P(None, "x"),
      check_vma=False,
  )
  def bidir_ring_collective_matmul_fn(a_local, b_local):
    dev_id = jnp.int32(jax.lax.axis_index("x"))
    c_out_ref = jax.new_ref(
        jax.lax.empty((m_block * num_devices, n_block), dtype=dtype)
    )
    bidir_ring_collective_matmul_prog(a_local, b_local, dev_id, c_out_ref)
    return c_out_ref[...]

  return bidir_ring_collective_matmul_fn
```

When XLA's command buffer pass is enabled (`--xla_gpu_enable_command_buffer=+COLLECTIVES`), this multi-stream schedule—including the grouped NCCL sends/receives, cross-stream CUDA event waits, and cuBLAS GEMM calls—is captured into a single **CUDA Graph**.

