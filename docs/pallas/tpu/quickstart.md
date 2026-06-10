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

(pallas-tpu-quickstart)=
# Quickstart: TPU

This quickstart covers the basics of writing Pallas kernels for TPU:
memory spaces, {func}`~jax.experimental.pallas.kernel`, and pipelining.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
```

## TPU memory spaces

Pallas kernels interact with memory through **Refs** — JAX's mutable
array references. On TPU, each Ref lives in a specific **memory space**:

- **HBM** — large but slow. Kernel inputs and outputs live here.
- **VMEM** — small but fast. Computation happens here.

You can't compute directly on HBM Refs. Data has to be copied to VMEM
first, and results copied back. Here's a complete kernel that adds two
arrays, with explicit memory movement:

```python
def add_kernel(x_ref, y_ref, o_ref, x_vmem, y_vmem, o_vmem):
  # HBM → VMEM
  pltpu.sync_copy(x_ref, x_vmem)
  pltpu.sync_copy(y_ref, y_vmem)

  # Compute in VMEM
  o_vmem[...] = x_vmem[...] + y_vmem[...]

  # VMEM → HBM
  pltpu.sync_copy(o_vmem, o_ref)
```

The `x_vmem`, `y_vmem`, `o_vmem` arguments are **scratch buffers** —
temporary VMEM allocations that the kernel uses as working space. We'll
see how to allocate them next.

## Running a kernel with {func}`~jax.experimental.pallas.kernel`

`pl.kernel` is how you go from a kernel function to something callable
with JAX arrays. It creates a **TensorCore mesh** and converts your
arrays into Refs:

```python
def add(x: jax.Array, y: jax.Array) -> jax.Array:
  tc_mesh = pltpu.create_tensorcore_mesh(axis_name='core')
  return pl.kernel(
      add_kernel,
      out_type=jax.ShapeDtypeStruct.like(x),
      mesh=tc_mesh,
      scratch_types=[
          pltpu.VMEM.like(x),  # x_vmem
          pltpu.VMEM.like(x),  # y_vmem
          pltpu.VMEM.like(x),  # o_vmem
      ],
  )(x, y)
```

`out_type` tells `pl.kernel` to allocate an output Ref with the same shape
and dtype as `x`. The `scratch_types` allocate the VMEM buffers that get
passed as extra arguments to the kernel.

If your chip has multiple TensorCores (e.g. TPU v5p has 2), the kernel
runs on all of them. As written, both cores will redundantly perform the exact
same addition. In a real kernel, use `jax.lax.axis_index('core')` inside the
kernel to figure out which core you're on and slice the refs to split the work.

## Making it fast: pipelining

The kernel above works, but it's leaving performance on the table. It copies
all data in, computes, then copies out — sequentially. The TensorCores
sit idle while memory is moving, and vice versa.

**Pipelining** fixes this by breaking the work into blocks and overlapping
memory transfers with compute. While the TensorCores process block N, the
DMA engine is already fetching block N+1.

`pltpu.emit_pipeline` handles the double-buffering machinery for you.
You provide:

- A **grid** — how many blocks to process.
- **BlockSpecs** — how to slice the arrays into blocks.
- A **body** — the compute for one block.

```python
def add_body(x_vmem, y_vmem, o_vmem):
  o_vmem[...] = x_vmem[...] + y_vmem[...]

def add_pipeline_kernel(x_hbm, y_hbm, o_hbm):
  pltpu.emit_pipeline(
      add_body,
      grid=(x_hbm.shape[0] // 128, x_hbm.shape[1] // 128),
      in_specs=[
          pl.BlockSpec(block_shape=(128, 128), index_map=lambda i, j: (i, j)),
          pl.BlockSpec(block_shape=(128, 128), index_map=lambda i, j: (i, j)),
      ],
      out_specs=pl.BlockSpec(
          block_shape=(128, 128), index_map=lambda i, j: (i, j)
      ),
      core_axis_name='core',
      dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
  )(x_hbm, y_hbm, o_hbm)

def add_matrices_pipelined(x: jax.Array, y: jax.Array) -> jax.Array:
  tc_mesh = pltpu.create_tensorcore_mesh(axis_name='core')
  return pl.kernel(
      add_pipeline_kernel,
      out_type=jax.ShapeDtypeStruct.like(x),
      mesh=tc_mesh,
  )(x, y)
```

Notice the body is just the compute — no `sync_copy`, no scratch allocation.
`emit_pipeline` takes care of the memory movement and buffer management.

We also passed `core_axis_name='core'` and `dimension_semantics`. This tells
the pipeline which grid dimensions can run independently on different cores.
`pltpu.PARALLEL` means the first dimension of the grid is distributed across
the TensorCores, splitting the work automatically!

## What's next

- [TPU Details](details) — the full picture of TPU memory spaces and supported ops
- [TPU Pipelining](pipelining.md) — deeper dive into pipelining, reductions, and accumulation
- [Matrix Multiplication](matmul.md) — putting it all together in a full kernel
