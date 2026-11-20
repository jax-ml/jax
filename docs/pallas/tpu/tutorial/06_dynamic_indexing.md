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

(pallas-tpu-tutorial-06-dynamic-indexing)=
# Chapter 6: Dynamic Computation and Block Sparsity

<!--* freshness: { reviewed: '2026-11-11' } *-->

Dynamic and sparse computation is a major reason to write custom Pallas kernels
over simply using JAX/XLA. It is generally difficult to express programs that
perform a dynamic amount of computation in XLA due to static array shapes. In
this chapter, we will learn how to write kernels that can early stop and kernels
that can dynamically skip over computation and blocks of memory.

To get maximum performance out of a TPU, we want to process data in large blocks
to keep the MXU and VPU fully utilized. **Block sparsity** gives us the best of
both worlds: we keep the computation dense and efficient within each block,
while dynamically skipping over blocks that contain only zeros. This provides
runtime dynamism while still playing to the strengths of the hardware.

## Example: Ragged Batched Addition

Another common use case for block sparsity is handling **ragged dimensions**,
where the size of a dimension varies at runtime. For example, consider a batched
matrix addition where the number of valid matrices in the batch is dynamic, and
we want to skip processing the padding at the end of the batch.

In a {func}`pl.kernel <jax.experimental.pallas.kernel>`, we can close over
scalar JAX arrays (like the valid size of a ragged dimension) and use them to
dynamically control our execution. Note that closing over generic non-scalar
arrays is not allowed inside of a `pl.kernel`.

Here is an example of a batched matrix addition on `(batch, m, n)` inputs where
the leading batch dimension is ragged. We define the kernel inline so it can
close over the scalar `ragged_size` JAX array and use it to compute the grid
size for `emit_pipeline` dynamically:

```{code-cell}
from functools import partial
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def ragged_add(
    x: jax.Array, y: jax.Array, ragged_size: jax.Array
) -> jax.Array:
  def kernel(x_hbm: jax.Ref, y_hbm: jax.Ref, o_hbm: jax.Ref):
    def body(x_vmem: jax.Ref, y_vmem: jax.Ref, o_vmem: jax.Ref):
      o_vmem[...] = x_vmem[...] + y_vmem[...]

    # Use the scalar ragged_size to determine the batch grid size dynamically!
    pltpu.emit_pipeline(
        body,
        grid=(ragged_size, x_hbm.shape[1] // 128, x_hbm.shape[2] // 128),
        in_specs=[
            pl.BlockSpec((1, 128, 128), lambda b, i, j: (b, i, j)),
            pl.BlockSpec((1, 128, 128), lambda b, i, j: (b, i, j)),
        ],
        out_specs=pl.BlockSpec((1, 128, 128), lambda b, i, j: (b, i, j)),
        core_axis_name='core',
        dimension_semantics=(
            pltpu.ARBITRARY,
            pltpu.PARALLEL,
            pltpu.ARBITRARY,
        ),
    )(x_hbm, y_hbm, o_hbm)

  tc_mesh = pltpu.TensorCoreMesh(axis_name='core')

  return pl.kernel(
      kernel,
      out_type=jax.typeof(x),
      mesh=tc_mesh,
  )(x, y)
```

In this example, `emit_pipeline` executes only for the first `ragged_size`
matrices along the leading batch dimension, skipping any padded batch entries
beyond `ragged_size` (note that skipped slices in the newly allocated output
buffer remain uninitialized unless you pre-initialize an output `Ref`).

## Dynamic Block Indexing

We can also parameterize memory accesses using array metadata (e.g., start/stop
indices or non-zero block indices).

We can achieve this by:

1.  Passing the metadata in HBM to the kernel.
2.  Allocating a scratch buffer in **SMEM** using `scratch_types`.
3.  Issuing a `sync_copy` to move the metadata from HBM to SMEM.
4.  Using the SMEM `Ref` (or values read from it) to parameterize slices that
    drive DMAs.

## Example: Block Dynamic Slice

Here is a basic example that demonstrates this pattern. We will implement a
block-aligned dynamic slice kernel which extracts a block out of a larger 2D
array based on user-specified start indices.

```{code-cell}
def dynamic_slice_kernel(
    starts_hbm: jax.Ref,
    x_hbm: jax.Ref,
    o_hbm: jax.Ref,
    starts_smem: jax.Ref,
    *,
    sizes: tuple[int, int],
):
  # 1. Copy metadata (starts) from HBM to SMEM
  pltpu.sync_copy(starts_hbm, starts_smem)
  # 2. Construct Ref dynamic slice using SMEM metadata
  block_i = starts_smem[0] // sizes[0]
  block_j = starts_smem[1] // sizes[1]
  x_slc = (
      jax.ds(block_i * sizes[0], sizes[0]),
      jax.ds(block_j * sizes[1], sizes[1]),
  )
  # 3. Issue sync copy from HBM -> HBM
  pltpu.sync_copy(x_hbm.at[x_slc], o_hbm)


def block_dynamic_slice(
    x: jax.Array, starts: jax.Array, sizes: tuple[int, int]
) -> jax.Array:
  tc_mesh = pltpu.TensorCoreMesh(axis_name='core', num_cores=1)
  return pl.kernel(
      partial(dynamic_slice_kernel, sizes=sizes),
      out_type=jax.ShapeDtypeStruct(shape=sizes, dtype=x.dtype),
      mesh=tc_mesh,
      scratch_types=[pltpu.SMEM.like(starts)],  # SMEM buffer for starts
  )(starts, x)
```

This example involves a single synchronous copy. Next, let's see how to use SMEM
metadata to drive a full pipeline.

### Example: Pipelined Dynamic Indexing

In more advanced cases, we want the metadata to determine an access pattern in a
full pipeline. We can achieve this by defining our `index_map` functions
*inside* the kernel so they can close over the SMEM reference and read the
indices to control the pipeline.

Here is an example of a block-gather kernel on a 3D `(num_source_blocks, bm, bn)`
input that uses `emit_pipeline`. We pass a 1D array of block indices to gather,
copy them to SMEM, and use them in `x_map` to fetch the requested blocks:

```{code-cell}
def gather_pipeline_kernel(
    indices_hbm: jax.Ref,
    x_hbm: jax.Ref,
    o_hbm: jax.Ref,
    indices_smem: jax.Ref,
):
  # 1. Copy metadata (indices) from HBM to SMEM
  pltpu.sync_copy(indices_hbm, indices_smem)

  # 2. Define index map closing over SMEM
  def x_map(i):
    # Look up the actual block index in SMEM
    block_idx = indices_smem[i]
    return (block_idx, 0, 0)

  def body(x_vmem: jax.Ref, o_vmem: jax.Ref):
    o_vmem[...] = x_vmem[...]

  block_shape = (1, x_hbm.shape[1], x_hbm.shape[2])

  # 3. Emit pipeline using the dynamic index map
  pltpu.emit_pipeline(
      body,
      grid=(indices_hbm.shape[0],),
      in_specs=[pl.BlockSpec(block_shape, x_map)],
      out_specs=pl.BlockSpec(block_shape, lambda i: (i, 0, 0)),
  )(x_hbm, o_hbm)


def block_gather(x: jax.Array, indices: jax.Array) -> jax.Array:
  tc_mesh = pltpu.TensorCoreMesh(axis_name='core', num_cores=1)
  num_blocks = indices.shape[0]
  return pl.kernel(
      gather_pipeline_kernel,
      out_type=jax.ShapeDtypeStruct(
          shape=(num_blocks, *x.shape[1:]), dtype=x.dtype
      ),
      mesh=tc_mesh,
      scratch_types=[pltpu.SMEM.like(indices)],  # SMEM buffer for indices
  )(indices, x)
```

```{seealso} Where to go deeper
* {ref}`Block-Sparse Kernels and Scalar Prefetch <pallas_scalar_prefetch_guide>`: Full Block-COO Sparse × Dense matmul and Block-Sparse Output Mask case studies.
* {ref}`TPU Pipelining <pallas_tpu_pipelining>`: Dynamic block shapes (`pl.BoundedSlice`) and lookahead prefetch (`use_lookahead=True`).
* {ref}`Working with Refs <pallas_refs>`: Dynamic `.at[pl.ds(...)]` slicing views and TPU DMA alignment rules.
```
