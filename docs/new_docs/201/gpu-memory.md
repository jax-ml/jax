---
nosearch: true
---

(jax-201-gpu-memory-allocation)=
# GPU memory allocation

<!--* freshness: { reviewed: '2026-07-10' } *-->

**JAX will preallocate 75% of the total GPU memory when the first JAX
operation is run.** Preallocating minimizes allocation overhead and memory
fragmentation, but can sometimes cause out-of-memory (OOM) errors. If your JAX
process fails with OOM, the following environment variables can be used to
override the default behavior:

**`XLA_PYTHON_CLIENT_PREALLOCATE=false`**

This disables the preallocation behavior. JAX will instead allocate GPU
memory as needed, potentially decreasing the overall memory usage. However,
this behavior is more prone to GPU memory fragmentation, meaning a JAX
program that uses most of the available GPU memory may OOM with
preallocation disabled.

**`XLA_CLIENT_MEM_FRACTION=.XX`**

If preallocation is enabled, this makes JAX preallocate XX% of the total
GPU memory, instead of the default 75%. Lowering the amount preallocated
can fix OOMs that occur when the JAX program starts. (You may see the old
name `XLA_PYTHON_CLIENT_MEM_FRACTION` in existing code; it is deprecated in
favor of `XLA_CLIENT_MEM_FRACTION`.)

**`XLA_PYTHON_CLIENT_ALLOCATOR=platform`**

This makes JAX allocate exactly what is needed on demand, and deallocate
memory that is no longer needed (note that this is the only configuration
that will deallocate GPU memory, instead of reusing it). This is very slow,
so is not recommended for general use, but may be useful for running with
the minimal possible GPU memory footprint or debugging OOM failures.

## Common causes of OOM failures

**Running multiple JAX processes concurrently.**

Either use `XLA_CLIENT_MEM_FRACTION` to give each process an
appropriate amount of memory, or set
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

**Running JAX and GPU TensorFlow concurrently.**

TensorFlow also preallocates by default, so this is similar to running
multiple JAX processes concurrently.

One solution is to use CPU-only TensorFlow (e.g. if you're only doing data
loading with TF). You can prevent TensorFlow from using the GPU with the
command `tf.config.experimental.set_visible_devices([], "GPU")`.

Alternatively, use `XLA_CLIENT_MEM_FRACTION` or
`XLA_PYTHON_CLIENT_PREALLOCATE`. There are also similar options to
configure TensorFlow's GPU memory allocation; see
[Using GPUs: Limiting GPU memory growth](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth).

**Running JAX on the display GPU.**

Use `XLA_CLIENT_MEM_FRACTION` or `XLA_PYTHON_CLIENT_PREALLOCATE`.

**Poor choices by the automatic rematerialization pass.**

Sometimes disabling XLA's automatic rematerialization HLO pass is
favorable, to avoid poor remat choices by the compiler. The pass can be
enabled or disabled by setting
`jax.config.update('jax_compiler_enable_remat_pass', True)` (or `False`).
Enabling or disabling it produces different trade-offs between compute and
memory; the algorithm is basic, and you can often get a better trade-off
by disabling the pass and rematerializing manually with
[the `jax.remat` API](https://docs.jax.dev/en/latest/jep/11830-new-remat-checkpoint.html).

## Experimental allocator features

Features here are experimental and must be tried with caution.

**`XLA_PYTHON_CLIENT_ALLOCATOR=vmm`**

This uses CUDA's virtual memory management (VMM) allocator
(`cudaDeviceAddressVmmAllocator`). This is a CUDA-only experimental
allocator that provides fine-grained virtual memory control. It does not
preallocate memory; use `XLA_CLIENT_MEM_FRACTION` to control the fraction
of GPU memory used.

**`TF_GPU_ALLOCATOR=cuda_malloc_async`**

This replaces XLA's own BFC memory allocator with
[`cudaMallocAsync`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html).
This removes the big fixed preallocation and uses a memory pool that
grows. The expected benefit is no need to set
`XLA_CLIENT_MEM_FRACTION`.

The risks are:

- memory fragmentation is different, so if you are close to the limit, the
  exact OOM case due to fragmentation will be different;
- the allocation time won't all be paid at the start, but incurred as the
  memory pool needs to grow, so you could experience less speed stability
  at the start (and for benchmarks it will be even more important to
  ignore the first few iterations).

The risks can be mitigated by preallocating a significant chunk and still
getting the benefit of a growing memory pool, via
`TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=N`. If `N` is `-1`, it
preallocates the same amount as the default; otherwise it's the size in
bytes to preallocate.
