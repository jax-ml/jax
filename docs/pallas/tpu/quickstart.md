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
memory spaces, `pl.kernel`, and pipelining.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
```

## TPU memory spaces

Pallas kernels interact with memory through **Refs** -- JAX's mutable
array references. On TPU, each Ref lives in a specific **memory space**:

- **HBM**: large but slow. Kernel inputs and outputs live here.
- **VMEM**: small but fast. Computation happens here.

You can't compute directly on HBM Refs. Data has to be copied to VMEM
first, and results copied back.

## Your first kernel

Here's a kernel that fills an output array with a constant. We allocate a
temporary VMEM buffer, write the constant there, and copy it to the output
HBM Ref.

```python
@pl.kernel(
    out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
    mesh=pltpu.TensorCoreMesh(axis_name='core'),
    scratch_types=dict(o_vmem=pltpu.VMEM((128,), jnp.float32)),
)
def fill_42(o_ref, o_vmem):
  # Compute in VMEM
  o_vmem[...] = jnp.full_like(o_vmem, 42.0)

  # VMEM → HBM (blocks until the transfer completes)
  pltpu.sync_copy(o_vmem, o_ref)

result = fill_42()  # [42.0, 42.0, ...]
```

`pl.kernel` wraps a kernel function to make it callable with standard JAX arrays.
It handles allocating the necessary Refs and executing the kernel on-device:

- `out_type` allocates an output Ref in HBM for the result.
- `scratch_types` allocates the VMEM buffers (like `o_vmem`) that get passed
  as extra arguments to the kernel.

```{note}
If your chip has multiple TensorCores (e.g. TPU v5p has 2), the kernel
runs on all of them. As written, both cores will redundantly perform the exact
same computation. In a real kernel, you would distribute work across cores
either manually or by using pipelining.
```

### Distributing work across TensorCores

To process a larger array, we can distribute work across multiple TensorCores
manually. Each core runs a separate invocation of the kernel.

```python
def iota() -> jax.Array:
  tpu_info = pltpu.get_tpu_info()

  @pl.kernel(
      out_type=jax.ShapeDtypeStruct((128 * tpu_info.num_cores,), jnp.float32),
      mesh=pltpu.TensorCoreMesh(axis_name='core'),
      scratch_types=dict(o_vmem=pltpu.VMEM((128,), jnp.float32)),
  )
  def kernel(o_ref, o_vmem):
    i = jax.lax.axis_index('core')

    # Compute our chunk in VMEM
    o_vmem[...] = jnp.arange(128, dtype=jnp.float32) + i * 128

    # Copy back to our slice of HBM
    pltpu.sync_copy(o_vmem, o_ref.at[pl.ds(i * 128, 128)])

  return kernel()

result = iota()  # [0.0, 1.0, 2.0, ...]
```

`jax.lax.axis_index('core')` tells you which TensorCore you're on -- use it to
determine your slice of the output.

## Making it fast: pipelining

The examples above work, but are inefficient. They copy data in, compute,
then copy out -- each `sync_copy` blocks until the transfer completes.
The TensorCores sit idle while memory is moving, and vice versa.

**Pipelining** fixes this by breaking the work into blocks and overlapping
memory transfers with compute. While the TensorCores process block N, the
DMA engine is already fetching block N+1.

`pltpu.emit_pipeline` handles the double-buffering machinery for you.
It takes:

- A **grid**: how many pipeline blocks to process.
- **`BlockSpec`s**: how to slice the arrays into blocks.
- A **body** function: the computation to run for each block.

Here's an elementwise addition kernel using pipelining:

```python
def add_matrices_pipelined(x: jax.Array, y: jax.Array) -> jax.Array:
  @pl.kernel(
      out_type=jax.ShapeDtypeStruct.like(x),
      mesh=pltpu.TensorCoreMesh(axis_name='core'),
  )
  def kernel(x_hbm, y_hbm, o_hbm):
    def add_body(x_vmem, y_vmem, o_vmem):
      o_vmem[...] = x_vmem[...] + y_vmem[...]

    pltpu.emit_pipeline(
        add_body,
        grid=(x_hbm.shape[0] // 128, x_hbm.shape[1] // 128),
        in_specs=[
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        core_axis_name='core',
        dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
    )(x_hbm, y_hbm, o_hbm)
  return kernel(x, y)

x = 2 * jnp.ones((256, 256))
y = 3 * jnp.ones((256, 256))
add_matrices_pipelined(x, y)
# Array([[5., 5., 5., ..., 5., 5., 5.],
#        [5., 5., 5., ..., 5., 5., 5.],
#        [5., 5., 5., ..., 5., 5., 5.],
#        ...,
#        [5., 5., 5., ..., 5., 5., 5.],
#        [5., 5., 5., ..., 5., 5., 5.],
#        [5., 5., 5., ..., 5., 5., 5.]], dtype=float32)
```

Notice the body is just the compute -- no `sync_copy`, no scratch allocation.
`emit_pipeline` takes care of the memory movement and buffer management.

We also passed `core_axis_name='core'` and `dimension_semantics`. This tells
the pipeline how to map the grid to the hardware:
- `pltpu.PARALLEL` means the first dimension of the grid is distributed across
  the TensorCores, splitting the work automatically.
- `pltpu.ARBITRARY` indicates that Pallas cannot make assumptions about data
  independence along this dimension; it therefore cannot be parallelized across
  cores and is instead executed sequentially on a single core.

## What's next

- [TPU Details](details) -- the full picture of TPU memory spaces and supported ops
- [TPU Pipelining](pipelining.md) -- deeper dive into pipelining, reductions, and accumulation
- [Matrix Multiplication](matmul.md) -- putting it all together in a full kernel
