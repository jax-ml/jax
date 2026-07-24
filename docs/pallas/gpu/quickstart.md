---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(pallas-gpu-quickstart)=
# Quickstart: GPU

This quickstart shows you how to write Pallas kernels for NVIDIA GPUs
using Mosaic GPU. The examples target Hopper (H100) GPUs, but the core
concepts - memory spaces, grids, pipelining - apply to all supported GPU
generations.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
```

## The programming model

In Mosaic GPU, each Pallas "thread" corresponds to a warpgroup
(4 warps = 128 CUDA threads). You write straight-line code that operates
on arrays; the warpgroup executes it in lockstep, so there are no
individual CUDA threads to manage.

## GPU memory spaces

Pallas kernels access memory through **Refs**, JAX's mutable array
references. On GPU, each Ref lives in a specific **memory space**:

- **GMEM** (global memory / HBM): large, slow. Used for kernel inputs
  and outputs.
- **SMEM** (shared memory): small, fast, per-SM. Shared between
  threads within a block, used to stage data for Tensor Core ops.
- **TMEM** (tensor memory): fast, per-SM. Used for Tensor Core ops
  (only available on Blackwell GPUs).

The typical data flow for Tensor Core work on Hopper is:
GMEM → SMEM → Tensor Cores → registers → SMEM → GMEM
(on Blackwell, TMEM replaces SMEM for staging Tensor Core inputs and outputs).
Simpler ALU work (like elementwise operations) bypasses SMEM and Tensor Cores
entirely: GMEM → registers → GMEM. Shared memory is primarily needed if
threads within a block must exchange or reuse data.

## Your first kernel

Here's a kernel that fills an output array with a constant:

```python
@plgpu.kernel(out_type=jax.ShapeDtypeStruct((128,), jnp.float32))
def fill_42(o_ref):
  o_ref[...] = jnp.full_like(o_ref, 42.0)

result = fill_42()  # [42.0, 42.0, ...]
```

`plgpu.kernel` allocates the output buffer, runs the computation on-device, and
returns a JAX array.

### Parallel work with a grid

To process a larger array, add a **grid**. Each grid point runs as a
separate CUDA block in parallel across SMs:

```python
@plgpu.kernel(
    out_type=jax.ShapeDtypeStruct((1024,), jnp.float32),
    grid=(8,),
    grid_names=('i',),
)
def iota(o_ref):
  i = jax.lax.axis_index('i')
  o_ref[pl.ds(i * 128, 128)] = jnp.arange(128, dtype=jnp.float32) + i * 128

result = iota()  # [0.0, 1.0, ..., 1023.0]
```

`jax.lax.axis_index(name)` tells you which grid block you're in - use it to
determine your slice of the output. `pl.ds(start, size)` creates a dynamic
slice of the given `size`, and is equivalent to `start:start+size`.

## Pipelining: a matmul kernel

The examples above don't need pipelining, but anything that hits the
Tensor Cores - matmuls, attention, etc. - should overlap GMEM↔SMEM
transfers with compute. Otherwise, the Tensor Cores sit idle waiting for
data.

`plgpu.emit_pipeline` handles this. It takes:

- A **sequential grid**: how many pipeline steps to run (typically
  over the contraction dimension).
- **`BlockSpec`s**: how to slice your inputs per step.
- A **body** function: the computation to run at each step.

The outer `plgpu.kernel` grid handles the parallel work, and
`emit_pipeline` handles the sequential reduction within each block.

Here's a matmul for Hopper GPUs:

```{note}
This example uses `wgmma`, which is specific to Hopper GPUs. For Blackwell, see
[Blackwell Matrix Multiplication](blackwell_matmul.md), which uses `tcgen05`
instead.
```

```python
def matmul(a, b, tile_m=128, tile_n=128, tile_k=64, out_dtype=jnp.float16):
  m, k = a.shape
  _, n = b.shape

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((m, n), out_dtype),
      scratch_types=dict(
          o_smem=plgpu.SMEM((tile_m, tile_n), out_dtype),
          acc=plgpu.ACC((tile_m, tile_n), jnp.float32),
      ),
      grid=(m // tile_m, n // tile_n),
      grid_names=('m', 'n'),
  )
  def kernel(a_gmem, b_gmem, o_gmem, o_smem, acc):
    pid_m = jax.lax.axis_index('m')
    pid_n = jax.lax.axis_index('n')

    def body(_, a_smem, b_smem):
      plgpu.wgmma(acc, a_smem, b_smem)
      plgpu.wgmma_wait(1)  # Keep one wgmma in flight.

    plgpu.emit_pipeline(
        body,
        grid=(k // tile_k,),
        in_specs=[
            plgpu.BlockSpec(
                (tile_m, tile_k), lambda ki: (pid_m, ki), delay_release=1
            ),
            plgpu.BlockSpec(
                (tile_k, tile_n), lambda ki: (ki, pid_n), delay_release=1
            ),
        ],
        max_concurrent_steps=2,
    )(a_gmem, b_gmem)

    # Drain: move the accumulated result to GMEM via SMEM.
    o_smem[...] = acc[...].astype(out_dtype)
    plgpu.commit_smem()  # Make the SMEM write visible to the TMA engine.
    plgpu.copy_smem_to_gmem(
        o_smem,
        o_gmem.at[pl.ds(pid_m * tile_m, tile_m),
                  pl.ds(pid_n * tile_n, tile_n)],
    )
    plgpu.wait_smem_to_gmem(0)  # Wait for all copies to finish.

  return kernel(a, b)
```

A few things to note:

- The **parallel grid** (`plgpu.kernel(..., grid=...)`) maps one CUDA block
  per output tile.
- The **sequential grid** (`emit_pipeline(..., grid=...)`) is the pipeline
  loop over the K dimension.
- **`scratch_types`** specifies temporary memory allocations needed for each
  parallel grid point, such as shared memory buffers (`SMEM`) or Tensor Core
  accumulators (`ACC`). Each key in the dictionary is passed as a keyword
  argument to the kernel function.
- **`plgpu.ACC`** is the Tensor Core accumulator. `wgmma` accumulates
  into it asynchronously.
- **`delay_release=1`** tells the pipeline to keep an extra buffer alive.
  Without this, the pipeline releases the corresponding buffer immediately,
  allowing the next iteration to overwrite input data while `wgmma` is still
  reading it.

## What's next

- [Mosaic GPU Pipelining](pipelining.md) - pipelining in depth, including warp
  specialization
- [Mosaic GPU Reference](reference.md) - full API reference (memory spaces,
  layouts, Tensor Core ops)
