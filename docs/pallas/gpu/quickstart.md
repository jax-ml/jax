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

This quickstart covers the basics of writing Pallas kernels for NVIDIA
GPUs using the Mosaic GPU backend. You'll need a **Hopper (H100) or
Blackwell (B200)** GPU.

```python
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
```

## The programming model

In Mosaic GPU, each Pallas "thread" corresponds to a **warpgroup**
(4 warps = 128 CUDA threads). You write straight-line code that operates
on arrays, and the warpgroup executes it in lockstep. You don't manage
individual CUDA threads.

## GPU memory spaces

Pallas kernels interact with memory through **Refs** — JAX's mutable
array references. On GPU, each Ref lives in a specific **memory space**:

- **GMEM** (global memory / HBM) — large, slow. Where kernel inputs
  and outputs live.
- **SMEM** (shared memory) — small, fast, per-SM. Shared between
  threads within a block, used to stage data for TensorCore ops.
- **Registers** — fastest. Where every JAX array inside your kernel
  body lives.

The typical data flow for TensorCore work is: GMEM → SMEM → TensorCores
→ registers → SMEM → GMEM. For simpler ALU work, you can load from SMEM
into registers and compute directly.

## Your first kernel with {func}`~jax.experimental.pallas.mosaic_gpu.kernel`

A kernel that fills an output array:

```python
@jax.jit
@functools.partial(
    plgpu.kernel,
    out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
)
def fill_42(o_ref):
  o_ref[...] = jnp.full_like(o_ref, 42.0)

result = fill_42()  # [42.0, 42.0, ...]
```

{func}`~jax.experimental.pallas.mosaic_gpu.kernel` allocates the output, launches the kernel, and gives you
back a JAX array.

### Parallel work with a grid

To process a larger array, add a **grid**. Each grid point becomes a
separate CUDA block, running in parallel across SMs:

```python
@jax.jit
@functools.partial(
    plgpu.kernel,
    out_shape=jax.ShapeDtypeStruct((1024,), jnp.float32),
    grid=(8,),
    grid_names=("i",),
)
def iota(o_ref):
  i = pl.program_id(0)
  o_ref[pl.ds(i * 128, 128)] = jnp.arange(128, dtype=jnp.float32) + i * 128

result = iota()  # [0.0, 1.0, ..., 1023.0]
```

`pl.program_id(axis)` tells you which grid block you're in — use it to
figure out your slice of the output.

## Pipelining: a matmul kernel

The examples above don't need pipelining. But for anything involving the
TensorCores (matmuls, attention, etc.), you want to overlap GMEM↔SMEM
transfers with compute. Otherwise the TensorCores sit idle while data is
being copied.

`plgpu.emit_pipeline` handles this. You give it:

- A **sequential grid** — how many pipeline steps to run (typically
  over the contraction dimension).
- **BlockSpecs** — how to slice your inputs per step.
- A **body** — the compute for one step.

The outer `plgpu.kernel` grid handles the parallel work, and
`emit_pipeline` handles the sequential reduction within each block.

Here's a Hopper matmul:

```python
def matmul(a, b, tile_m=128, tile_n=128, swizzle=128):
  m, k = a.shape
  _, n = b.shape
  dtype = jnp.float16
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  tile_k = swizzle_elems

  # TilingTransform + SwizzleTransform arrange data in the layout
  # that wgmma expects. These will be inferred automatically in the
  # future — for now, you specify them by hand.
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )

  def kernel(a_gmem, b_gmem, o_gmem, o_smem, acc):
    pid_m = pl.program_id(0)
    pid_n = pl.program_id(1)

    def body(_, a_smem, b_smem):
      plgpu.wgmma(acc, a_smem, b_smem)
      plgpu.wgmma_wait(1)  # keep one wgmma in flight

    plgpu.emit_pipeline(
        body,
        grid=(k // tile_k,),
        in_specs=[
            plgpu.BlockSpec(
                (tile_m, tile_k), lambda ki: (pid_m, ki),
                transforms=transforms,
            ),
            plgpu.BlockSpec(
                (tile_k, tile_n), lambda ki: (ki, pid_n),
                transforms=transforms,
            ),
        ],
        max_concurrent_steps=2,
        delay_release=1,
    )(a_gmem, b_gmem)

    # Drain: accumulator → SMEM → GMEM
    o_smem[...] = acc[...].astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(
        o_smem,
        o_gmem.at[pl.ds(pid_m * tile_m, tile_m),
                  pl.ds(pid_n * tile_n, tile_n)],
    )
    plgpu.wait_smem_to_gmem(0)

  return plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      scratch_shapes=dict(
          o_smem=plgpu.SMEM((tile_m, tile_n), dtype),
          acc=plgpu.ACC((tile_m, tile_n), jnp.float32),
      ),
      grid=(m // tile_m, n // tile_n),
      grid_names=("m", "n"),
  )(a, b)
```

A few things to note:

- The **outer grid** (`plgpu.kernel(..., grid=...)`) is the parallel CUDA
  grid — one block per output tile.
- The **inner grid** (`emit_pipeline(..., grid=...)`) is the sequential
  pipeline loop over the K dimension.
- **`plgpu.ACC`** is the TensorCore accumulator. `wgmma` accumulates
  into it asynchronously.
- **`delay_release=1`** tells the pipeline to keep an extra buffer alive
  so `wgmma` can still read from it. Without this, you get silent data
  races.
- The **transforms** are boilerplate for now — they tell the hardware how
  to lay out data in SMEM for the TensorCores. We're working on inferring
  these automatically.

## What's next

- [Mosaic GPU Pipelining](pipelining.md) — pipelining in depth, including warp
  specialization
- [Mosaic GPU Reference](reference.md) — full API reference (memory spaces,
  layouts, TensorCore ops)
- [Blackwell Matrix Multiplication](blackwell_matmul.md) — matrix multiplication on Blackwell with `tcgen05`
