---
jupytext:
  formats: ipynb,md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "ZHuzXqQ-9JUQ"}

(pallas_scalar_prefetch_guide)=

# Scalar Prefetch and Block-Sparse Computation

In this tutorial, we will cover the basics of block-sparse computing in Pallas. Sparse computation is a major reason to write custom Pallas kernels over simply using JAX/XLA, since it is generally difficult to express programs that perform a dynamic amount of computation in XLA due to static array shapes. In this tutorial we will learn how to stage small metadata arrays into SMEM ("scalar prefetch") inside `pl.kernel` and close over them in `pltpu.emit_pipeline` `BlockSpec` index maps to dynamically skip over computation and blocks of memory.

```{code-cell}
---
executionInfo:
  elapsed: 56
  status: ok
  timestamp: 1726001133029
  user:
    displayName: Justin Fu
    userId: '17543197034567316452'
  user_tz: 420
id: ibeIs_6QFMAM
outputId: d72edb91-4529-4650-c9e9-b96788608635
---
import timeit
import numpy as np
import jax
from jax import numpy as jnp
from jax import lax
from jax.experimental import checkify
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

assert "TPU" in jax.devices()[0].device_kind, "Please run this notebook with TPU devices."
print("Running on", jax.devices()[0].device_kind)
```

+++ {"id": "FIDGpPTEIcOa"}

## Dynamic Block Indexing with SMEM Scalar Prefetch

We will use SMEM staging ("scalar prefetch") to write sparse kernels. Scalar prefetch loads a small metadata or block-index array from HBM into SMEM ("scalar memory") before launching `pltpu.emit_pipeline`. Because the metadata is already in SMEM before the pipeline starts, both the `index_map` of each `pl.BlockSpec` and the pipeline step body can close over those SMEM `Ref`s directly to perform data-dependent block indexing!

With `pl.kernel` and `pltpu.emit_pipeline`, the pattern has three steps:

1. **Allocate SMEM scratch buffers** on `pl.kernel` via `scratch_types=[pltpu.SMEM.like(indices), ...]`.
2. **Copy metadata from HBM to SMEM** before starting the pipeline via `pltpu.sync_copy(indices_hbm_ref, indices_smem_ref)`.
3. **Close over the SMEM `Ref`s** inside your `BlockSpec` `index_map` functions and pipeline `body`:

```python
@pl.kernel(
    out_type=...,
    mesh=pltpu.TensorCoreMesh(axis_name='core', num_cores=1),
    scratch_types=[pltpu.SMEM.like(block_indices)],
)
def kernel(indices_hbm_ref, x_hbm_ref, o_hbm_ref, indices_smem_ref):
  # 1. Stage metadata into SMEM before starting the pipeline.
  pltpu.sync_copy(indices_hbm_ref, indices_smem_ref)

  # 2. Close over `indices_smem_ref` in `index_map` (and/or `body`).
  def x_index_map(*grid_indices):
    return (indices_smem_ref[grid_indices[0]], 0)

  def body(x_vmem_ref, o_vmem_ref):
    ...

  pltpu.emit_pipeline(
      body,
      grid=...,
      in_specs=[pl.BlockSpec(..., x_index_map)],
      out_specs=pl.BlockSpec(...),
  )(x_hbm_ref, o_hbm_ref)
```

```{note}
In older `pl.pallas_call` code, scalar prefetch was expressed via `pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=...)`, which threaded extra `prefetch_refs` arguments into every `index_map` and kernel signature. Staging SMEM `Ref`s explicitly with `pl.kernel(..., scratch_types=[pltpu.SMEM(...)])` and Python closures in `pltpu.emit_pipeline` is cleaner and avoids signature acoplamiento.
```

+++ {"id": "pA8RmHEA2HN3"}

## Example: Block Dynamic Slice with Scalar Prefetch

Let's begin with a basic example that demonstrates SMEM scalar prefetch. We will implement a block-aligned dynamic slice kernel which extracts a block out of a larger array based on runtime indices:

1. Outside of the kernel, we compute the block index to extract as: `block_idx = jnp.array([starts[0] // sizes[0], starts[1] // sizes[1]])`.

2. Inside `pl.kernel`, we copy `block_idx` from HBM into an SMEM scratch `Ref` (`idx_smem_ref`) via `pltpu.sync_copy`.

3. In our `pltpu.emit_pipeline` `index_map`, we close over `idx_smem_ref` and return `(idx_smem_ref[0], idx_smem_ref[1])`.

Of course, this kernel is limited in that our slice sizes must fit inside of a kernel block (limited by VMEM size) and we can only start on size-aligned indices. A more advanced kernel can decouple the pipeline block size from the slice size or use `pl.ds` / `pl.BoundedSlice` for unaligned slices.

```{code-cell}
---
executionInfo:
  elapsed: 143
  status: ok
  timestamp: 1726003877561
  user:
    displayName: Justin Fu
    userId: '17543197034567316452'
  user_tz: 420
id: FWeTBlEYlCGD
outputId: 4b04a441-c97c-4d0d-d167-c60d4d31fd2e
---
@checkify.checkify
@jax.jit(static_argnums=(2,))
def block_dynamic_slice(x, starts, sizes):
  # Checkify inserts a runtime assert that starts are divisible by block size.
  checkify.check(starts[0] % sizes[0] == 0, "Starts must be divisible by size.")
  checkify.check(starts[1] % sizes[1] == 0, "Starts must be divisible by size.")
  block_idx = jnp.array([starts[0] // sizes[0], starts[1] // sizes[1]], dtype=jnp.int32)
  mesh = pltpu.TensorCoreMesh(axis_name='core', num_cores=1)

  @pl.kernel(
      out_type=jax.ShapeDtypeStruct(shape=sizes, dtype=x.dtype),
      mesh=mesh,
      scratch_types=[pltpu.SMEM.like(block_idx)],
  )
  def kernel(idx_hbm_ref, x_hbm_ref, o_hbm_ref, idx_smem_ref):
    pltpu.sync_copy(idx_hbm_ref, idx_smem_ref)

    def body(x_vmem_ref, o_vmem_ref):
      o_vmem_ref[...] = x_vmem_ref[...]

    pltpu.emit_pipeline(
        body,
        grid=(1, 1),
        in_specs=[pl.BlockSpec(sizes, lambda i, j: (idx_smem_ref[0], idx_smem_ref[1]))],
        out_specs=pl.BlockSpec(sizes, lambda i, j: (0, 0)),
    )(x_hbm_ref, o_hbm_ref)

  return kernel(block_idx, x)

shape = (512, 512)
x = jnp.reshape(jnp.arange(np.prod(shape), dtype=jnp.int32), shape)
err, result = block_dynamic_slice(x, starts=(128, 256), sizes=(128, 128))
err.throw()
ref = lax.dynamic_slice(x, start_indices=(128, 256), slice_sizes=(128, 128))
diff = jnp.max(jnp.abs(result - ref))
print("Error |result - lax.dynamic_slice| =", diff)
```

+++ {"id": "K2dod4lkoifa"}

## Sparse Kernels: Representing Sparse Data

Before we dive into implementing sparse kernels, let's first review how sparse matrices are represented. While there are several popular formats for storing sparse matrices, we will be following a blocked variant of the coordinate-list format (COO) in which we will store a matrix as a list of `(block_index, block_data)` pairs. All blocks that are not explicitly stored in the list are assumed to be zero, meaning we can save a significant amount of memory if there are many zero blocks in the matrix.

The following figure demonstrates how we convert a 4x4 dense matrix (left) into a block-COO format (right) with a block size of 2x2. Note that in the sparse format, we can avoid explicitly storing the upper-right block which consists of all zero elements.

![block_coo](../../_static/pallas/sparse/block_coo.svg)

We will use the following helper function to sample a block-sparse matrix. It returns a dense matrix used for checking our results, as well as a list of block data and indices for each axis.

```{code-cell}
:id: 1gLiSvgIYUEx

def generate_block_sparse_mat(key, M, N, blk_M, blk_N, p=0.2, dtype=jnp.float32):
  """Returns a sampled matrix and its block-sparse representation.

  Args:
    key: RNG Key.
    M: Major array dimension.
    N: Minor array dimension.
    blk_M: Block size along M dimension.
    blk_N: Block size along N dimension.
    p: Probability that a block will be non-zero.
    dtype: dtype of the sampled matrix.

  Returns:
    dense_mat: A (M, N) dense sampled array.
    block_data: A (num_blocks, blk_M, blk_N) array of data blocks representing
      the non-zero blocks of the matrix.
    indices_i: A (num_blocks,) array of block indices for the first axis.
    indices_j: A (num_blocks,) array of block indices for the second axis.
  """
  mask_key, blocks_key = jax.random.split(key)
  num_blocks = (M // blk_M, N // blk_N)
  # We first sample a block mask, denoting which blocks are nonzero.
  block_mask = jax.random.bernoulli(mask_key, p=p, shape=num_blocks)
  num_blocks = int(jnp.sum(block_mask))
  indices = jnp.where(block_mask)
  # For each non-zero block, we sample a block of random values.
  block_data = jax.random.uniform(blocks_key,
                                  shape=(num_blocks, blk_M, blk_N),
                                  dtype=dtype)
  # For checking purposes, create the dense version of the sparse matrix.
  dense_mat = jnp.zeros((M, N), dtype=dtype)
  for blk in range(num_blocks):
    idx_i = indices[0][blk]
    idx_j = indices[1][blk]
    slice_i = slice(idx_i * blk_M, (idx_i + 1) * blk_M)
    slice_j = slice(idx_j * blk_N, (idx_j + 1) * blk_N)
    dense_mat = dense_mat.at[slice_i, slice_j].set(block_data[blk])
  return dense_mat, block_data, indices[0], indices[1]
```

+++ {"id": "eFyoZSTOH9Fk"}

## Example: Sparse @ Dense Matrix Multiplication

In our first example, we will multiply a sparse LHS matrix with a dense RHS matrix to produce a dense output.

We will structure our pipeline grid with 2 loops — the outer loop over the columns of the RHS/output, and the inner loop over the sparse blocks of the LHS. During each inner loop iteration, we load one block from the LHS and look up the corresponding block in the RHS using the block index of the contracting dimension (`K`) staged in SMEM. We multiply the two blocks together and accumulate into the output block. One outer loop iteration computes a result for an entire column as depicted by the following diagram:

![sparse_matmul](../../_static/pallas/sparse/sparse_matmul.svg)

It is important that we group the block indices by row (e.g. `[0, 0, 1, 2, 3, 3]`) before we pass them into the kernel for two reasons. First, in our kernel we need to know when to initially zero-out the accumulator, which is straightforward if row indices are grouped. Second, Pallas's pipeline emitter keeps an output block in VMEM while consecutive iterations target the same block index and writes it back to HBM when the block index changes; revisiting the same output block on non-consecutive iterations would overwrite previous results. Also, because completely empty rows are never visited by the sparse grid, we pre-initialize a mutable output `Ref` to zeros via `jax.new_ref(jnp.zeros(...))`, mutate it in place inside `pl.kernel`, and call `jax.freeze`.

```{code-cell}
---
executionInfo:
  elapsed: 673
  status: ok
  timestamp: 1725919879291
  user:
    displayName: Justin Fu
    userId: '17543197034567316452'
  user_tz: 420
id: WfyV2WWhjsyA
outputId: fa4d4fff-bc6b-4dc9-ac14-63276ca14131
---
M = N = K = 16384
blk_M = blk_N = blk_K = 512

(X_dense, X_blocks, indices_i, indices_k) = generate_block_sparse_mat(
    jax.random.key(0), M, K, blk_M, blk_K, p=0.1, dtype=jnp.bfloat16
)
num_blocks = X_blocks.shape[0]
Y = jax.random.uniform(jax.random.key(1), shape=(K, N), dtype=jnp.bfloat16)


@jax.jit
def sparse_dense_matmul(indices_i, indices_k, x_blocks, y):
  # Pre-zero the output Ref so rows with zero non-zero blocks remain 0.
  o_ref = jax.new_ref(jnp.zeros((M, N), dtype=jnp.bfloat16))
  mesh = pltpu.TensorCoreMesh(axis_name='core')

  @pl.kernel(
      out_type=None,
      mesh=mesh,
      scratch_types=[
          pltpu.SMEM.like(indices_i),
          pltpu.SMEM.like(indices_k),
          pltpu.VMEM((blk_M, blk_N), dtype=jnp.float32),
      ],
  )
  def dsd_kernel(
      idxs_i_hbm, idxs_k_hbm, x_hbm, y_hbm, o_hbm,
      idxs_i_smem, idxs_k_smem, accum_scratch,
  ):
    """A DSD (Dense = Sparse @ Dense) matmul kernel."""
    pltpu.sync_copy(idxs_i_hbm, idxs_i_smem)
    pltpu.sync_copy(idxs_k_hbm, idxs_k_smem)

    def body(x_vmem, y_vmem, o_vmem):
      blk_idx = pl.program_id(1)
      is_start = blk_idx == 0
      changed_blocks = (
          idxs_i_smem[blk_idx] != idxs_i_smem[jnp.maximum(blk_idx - 1, 0)]
      )

      @pl.when(is_start | changed_blocks)
      def _():
        accum_scratch[...] = jnp.zeros_like(accum_scratch)

      accum_scratch[...] += jnp.dot(
          x_vmem[0, :, :], y_vmem[...], preferred_element_type=jnp.float32
      )

      next_block_change = (
          idxs_i_smem[blk_idx]
          != idxs_i_smem[jnp.minimum(blk_idx + 1, num_blocks - 1)]
      )
      is_end = blk_idx == (num_blocks - 1)

      @pl.when(is_end | next_block_change)
      def _():
        o_vmem[...] = accum_scratch[...].astype(o_vmem.dtype)

    pltpu.emit_pipeline(
        body,
        grid=(N // blk_N, num_blocks),
        in_specs=[
            pl.BlockSpec((1, blk_M, blk_K), lambda j, blk_idx: (blk_idx, 0, 0)),
            pl.BlockSpec((blk_K, blk_N), lambda j, blk_idx: (idxs_k_smem[blk_idx], j)),
        ],
        out_specs=pl.BlockSpec(
            (blk_M, blk_N), lambda j, blk_idx: (idxs_i_smem[blk_idx], j)
        ),
        core_axis_name='core',
        dimension_semantics=('parallel', 'arbitrary'),
    )(x_hbm, y_hbm, o_hbm)

  dsd_kernel(indices_i, indices_k, x_blocks, y, o_ref)
  return jax.freeze(o_ref)


result = sparse_dense_matmul(indices_i, indices_k, X_blocks, Y)
ref = X_dense @ Y
diff = jnp.abs(ref - result)
print('mean |result - ref|:', jnp.mean(diff))
```

+++ {"id": "2KDgPKF2tUjq"}

We can do a quick benchmark to compare the performance of our sparse kernel compared to a dense matmul in JAX. On a TPU v5e chip, this kernel achieves a roughly ~6x speed increase compared to the theoretical 10x from the sparsity factor.

There are a few main tips for performance here, mainly centered around reducing the communication overhead between HBM/VMEM:
- Using `dtype=jnp.bfloat16` is critical for performance since it reduces memory bandwidth by half.
- Using larger block sizes also helps, since matrix multiply is an $O(N^3)$ compute and $O(N^2)$ memory operation. As $N$ grows larger, the kernel becomes compute-bound. However, a counter-argument to this in practice is that smaller block sizes also enables data to be more sparse, so this is a parameter that should be selected carefully.

```{code-cell}
---
executionInfo:
  elapsed: 6576
  status: ok
  timestamp: 1725919886762
  user:
    displayName: Justin Fu
    userId: '17543197034567316452'
  user_tz: 420
id: CkzjqnekpZbx
outputId: 1ae9031e-705a-4d05-f8b9-d09623918300
---
# Benchmark Sparse Pallas kernel vs reference JAX implementation

def benchmark(f, ntrials: int = 100):
  def run(*args, **kwargs):
    # Compile function first
    jax.block_until_ready(f(*args, **kwargs))
    # Time function
    result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                           number=ntrials)
    time = result / ntrials
    return time
  return run


n_trials = 100

time = benchmark(sparse_dense_matmul, n_trials)(indices_i, indices_k, X_blocks, Y)
print("Sparse Kernel: %.3f ms (avg over %d trials)" % (time * 1000, n_trials))

ref_impl = jax.jit(lambda x, y: x @ y)
time = benchmark(ref_impl, n_trials)(X_dense, Y)
print("Reference: %.3f ms (avg over %d trials)" % (time * 1000, n_trials))
```

+++ {"id": "Q1KKd5vTCwnB"}

## Sparse Access Patterns on Dense Data

In our previous example we considered the case when the data itself is sparse. This manifested itself in the kernel structure as a dimension in the kernel grid that was dynamic and looped over the number of nonzero blocks (`num_blocks`).

A second useful programming pattern emerges when the underlying data is dense, but we wish to perform sparse computation over it. Our kernel grid in this case will be dense, but we wish to skip over some blocks in the grid as indicated by a block-sparse mask. This type of programming pattern commonly arises when using masks in many machine learning applications, such as causal or local masks in self-attention. In these cases, we can entirely skip over computation in blocks where the mask is zeroed-out. Examples of this programming pattern can be found in the Splash Attention and Grouped Matrix Multiplication kernels located in `jax/experimental/pallas/ops/tpu`, or in PyTorch's [FlexAttention](https://pytorch.org/blog/flexattention/).

The main performance consideration with dealing with a sparse access pattern on dense data is the interaction with pipelining. On any given kernel iteration, the Pallas pipeline emitter will attempt to prefetch the next block of data by calling the `index_map` for each `BlockSpec` on the next iteration of the grid. However, if our computation is sparse we may be skipping the computation for the next block in the grid, so we need some method to tell the pipeline instead begin fetching the *next block that we are not skipping*. In order to do this, we need to construct *prefetch maps* which contains indices to the next non-skipped block of data for each kernel input. The following diagram illustrates how a prefetch map could be constructed for a block-sparse mask that is stored in a COO-like format.

![prefetch_map](../../_static/pallas/sparse/prefetch_map.svg)
*Left: A sparse access pattern, where the color blue denotes blocks with non-zero masks that we need to compute. Right: The prefetch map, where each element of the array contains the index of the next non-zero block data.*

Once the prefetch map has been constructed and staged into SMEM (`prefetch_map_smem`), we close over it in the `index_map` function of the `BlockSpec`:

```python
def mask_index_map(i, j, k):
  next_nonzero_block = prefetch_map_smem[i, j]
  return (next_nonzero_block, 0, 0)
```

We can construct similar index maps for the other inputs to the kernel. For dense inputs you will most likely need to construct prefetch maps which point to the next non-zero block index in the grid. Our next example will provide an example of using these prefetch maps.

+++ {"id": "ii7rzL5YIA8-"}

## Example: Dense @ Dense Matrix Multiplication with a Block-Sparse Output Mask

+++ {"id": "ecjiqWfA2RlV"}

In our next example we will cover dense matrix multiplication fused with a sparse output mask using a prefetch map to improve pipelining performance. We will use the mask to selectively skip computing output blocks that are zeroed-out, therefore saving on computation costs.

As we will be working with a sparse mask, we will begin by implementing a function that converts an `M x N` mask stored in dense format into a block-sparse format. We additionally need to compute prefetch maps to help the pipeline emitter know which block to fetch next. In total, our `sparsify_mask` function computes:
- A `block_mask` of shape `(num_M_blocks, num_N_blocks)` indicating if a block is all-zeros (value `0`) or contains non-zero elements (value `1`). If the `block_mask` has a value of 0 we can skip computing the block in the kernel.
- A `prefetch_mask` array of shape `(num_M_blocks, num_N_blocks)` consisting of indices into `mask_data` for the next non-zero block.
- A `prefetch_i` array of shape `(num_M_blocks, num_N_blocks)` consisting of the next non-masked `i` index of the mask.
- A `prefetch_j` array of shape `(num_M_blocks, num_N_blocks)` consisting of the next non-masked `j` index of the mask.
- A `mask_data` array of shape `(num_blocks, blk_M, blk_N)` containing data for non-zero blocks of the mask.

```{code-cell}
:id: 19zGcliL2SJy

def sparsify_mask(mask: jax.Array,
                  block_shape: tuple[int, int]):
  """Preprocesses a mask into a sparse representation.

  Args:
    mask: A boolean or integer array of shape [M, N]
    block_shape: The size of a single block.

  Returns:
    block_mask: A block_grid-shaped array indicating whether a block
      is all-zeros (0) or contains non-zero elements (1).
    prefetch_mask: A block_grid-shaped array of integers indicating the index of
      the next non-zero block.
    prefetch_i: Next non-masked i block index.
    prefetch_j: Next non-masked j block index.
    mask_data: A (num_blocks, *block_shape) array containing
      the data for non-zero blocks of the mask.
  """
  M, N = mask.shape
  bm, bn = block_shape

  block_mask = jnp.zeros((M // bm, N // bn), dtype=mask.dtype)
  mask_types_finder = []
  mask_data = []

  next_mask_type_idx = 0
  prefetch_mask = jnp.zeros_like(block_mask)
  next_i = (M // bm) - 1
  next_j = (N // bn) - 1
  prefetch_i = jnp.zeros_like(block_mask)
  prefetch_j = jnp.zeros_like(block_mask)
  for i in range((M // bm) - 1, -1, -1):
    for j in range((N // bn) - 1, -1, -1):
      mask_block = mask[i * bm :(i + 1) * bm,
                        j * bn :(j + 1) * bn]
      is_nonzero = jnp.any(mask_block)
      if is_nonzero:
        try:
          type_index = mask_types_finder.index(str(mask_block))
        except ValueError:
          type_index = len(mask_types_finder)
          mask_types_finder.append(str(mask_block))
          mask_data.append(mask_block)
        next_mask_type_idx = type_index
        next_i = i
        next_j = j
      block_mask = block_mask.at[i, j].set(is_nonzero)
      prefetch_mask = prefetch_mask.at[i, j].set(next_mask_type_idx)
      prefetch_i = prefetch_i.at[i, j].set(next_i)
      prefetch_j = prefetch_j.at[i, j].set(next_j)
  return block_mask, prefetch_mask, prefetch_i, prefetch_j, jnp.stack(mask_data)
```

+++ {"id": "w4b7ckKq67Xw"}

In terms of the structure of the kernel, we use the same 3-D pipeline grid over `(M // blk_M, N // blk_N, K // blk_K)` as standard matrix multiplication, staging `block_mask`, `prefetch_mask`, `prefetch_i`, and `prefetch_j` into SMEM before `pltpu.emit_pipeline`. Within each step, we check `block_mask_smem[i, j]` to see if the mask for the current output block is all zeros; if so, we skip the MXU dot product entirely.

```{code-cell}
---
executionInfo:
  elapsed: 5374
  status: ok
  timestamp: 1725919713252
  user:
    displayName: Justin Fu
    userId: '17543197034567316452'
  user_tz: 420
id: 4YQ9OmbTCSjT
outputId: 2d752609-34f2-4059-e8ba-4d80afe8cb26
---
M = N = K = 16384
blk_M = blk_N = 512
blk_K = 1024

X = jax.random.normal(jax.random.key(0), shape=(M, K), dtype=jnp.bfloat16)
Y = jax.random.normal(jax.random.key(1), shape=(K, N), dtype=jnp.bfloat16)
mask = jnp.tril(jnp.ones((M, N), dtype=jnp.int32))
block_mask, prefetch_mask, prefetch_i, prefetch_j, sparse_mask_data = (
    sparsify_mask(mask, (blk_M, blk_N))
)


@jax.jit
def run_sparse_mask_matmul(
    block_mask, prefetch_mask, prefetch_i, prefetch_j, x, y, mask_data
):
  mesh = pltpu.TensorCoreMesh(axis_name='core', num_cores=1)

  @pl.kernel(
      out_type=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
      mesh=mesh,
      scratch_types=[
          pltpu.SMEM.like(block_mask),
          pltpu.SMEM.like(prefetch_mask),
          pltpu.SMEM.like(prefetch_i),
          pltpu.SMEM.like(prefetch_j),
          pltpu.VMEM((blk_M, blk_N), dtype=jnp.float32),
      ],
  )
  def kernel(
      bmask_hbm, pmask_hbm, pi_hbm, pj_hbm, x_hbm, y_hbm, mdata_hbm, o_hbm,
      bmask_smem, pmask_smem, pi_smem, pj_smem, accum_scratch,
  ):
    pltpu.sync_copy(bmask_hbm, bmask_smem)
    pltpu.sync_copy(pmask_hbm, pmask_smem)
    pltpu.sync_copy(pi_hbm, pi_smem)
    pltpu.sync_copy(pj_hbm, pj_smem)

    def x_map(i, j, k):
      # Zero-out the k index if the mask is zero, to avoid constantly fetching
      # new blocks in the inner loop for blocks we are skipping.
      k_fetch = (bmask_smem[i, j] != 0) * k
      return (pi_smem[i, j], k_fetch)

    def y_map(i, j, k):
      k_fetch = (bmask_smem[i, j] != 0) * k
      return (k_fetch, pj_smem[i, j])

    def mask_map(i, j, k):
      del k
      return (pmask_smem[i, j], 0, 0)

    def body(x_vmem, y_vmem, mask_vmem, o_vmem):
      i, j, k = pl.program_id(0), pl.program_id(1), pl.program_id(2)
      should_compute = bmask_smem[i, j] != 0

      @pl.when(k == 0)
      def _():
        o_vmem[...] = jnp.zeros_like(o_vmem)
        accum_scratch[...] = jnp.zeros_like(accum_scratch)

      @pl.when(should_compute)
      def _():
        accum_scratch[...] += jnp.dot(
            x_vmem[...], y_vmem[...], preferred_element_type=jnp.float32
        )

        @pl.when(k == pl.num_programs(2) - 1)
        def _():
          o_vmem[...] = (mask_vmem[0, ...] * accum_scratch[...]).astype(
              o_vmem.dtype
          )

    pltpu.emit_pipeline(
        body,
        grid=(M // blk_M, N // blk_N, K // blk_K),
        in_specs=[
            pl.BlockSpec((blk_M, blk_K), x_map),
            pl.BlockSpec((blk_K, blk_N), y_map),
            pl.BlockSpec((1, blk_M, blk_N), mask_map),
        ],
        out_specs=pl.BlockSpec((blk_M, blk_N), lambda i, j, k: (i, j)),
    )(x_hbm, y_hbm, mdata_hbm, o_hbm)

  return kernel(
      block_mask, prefetch_mask, prefetch_i, prefetch_j, x, y, mask_data
  )


result = run_sparse_mask_matmul(
    block_mask, prefetch_mask, prefetch_i, prefetch_j, X, Y, sparse_mask_data
)

ref = mask * (X @ Y)
diff = jnp.abs(ref - result)
print('mean |result - ref|:', jnp.mean(diff))
```

+++ {"id": "uutNGgjZGGhB"}

Now let's compare performance versus a naive dense implementation. On TPU v5e, we achieve around a ~1.8x speed increase with the sparse kernel, compared to a theoretical best-case of 2x from using a lower triangular mask and only visiting half of the possible outputs.

We would generally expect performance to get closer to the theoretical peak as our inputs get larger, since a few of the main reasons why we don't exactly reach theoretical performance are:
- We skip slightly less than half of computation since the blocks along the diagonal are mixed 0s and 1s, and for mixed blocks we need to compute the entire block. With larger inputs, our overhead for mixed blocks becomes smaller relative to the overall computation.
- The pipeline bubble also accounts for a less percentage of the overall runtime as inputs become larger.

```{code-cell}
---
executionInfo:
  elapsed: 8877
  status: ok
  timestamp: 1725917397452
  user:
    displayName: Justin Fu
    userId: '17543197034567316452'
  user_tz: 420
id: MAT9JjGNvsx8
outputId: a32d56fb-a71b-4007-c6a5-e5270dcaa6cf
---
n_trials = 100

time = benchmark(run_sparse_mask_matmul, n_trials)(
    block_mask, prefetch_mask, prefetch_i, prefetch_j, X, Y, sparse_mask_data
)
print("Sparse Kernel: %.3f ms (avg over %d trials)" % (time * 1000, n_trials))

ref_impl = jax.jit(lambda mask, x, y: mask * (x @ y))
time = benchmark(ref_impl, n_trials)(mask, X, Y)
print("Reference: %.3f ms (avg over %d trials)" % (time * 1000, n_trials))
```
