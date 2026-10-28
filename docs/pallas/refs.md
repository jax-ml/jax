(pallas_refs)=

# Working with `Ref`s

<!--* freshness: { reviewed: '2026-10-27' } *-->

Unlike standard JAX functions, which take and return immutable `jax.Array`
values, Pallas kernels operate on *`Ref`* objects representing mutable buffers
in device memory (see {ref}`jax-101-refs` for how `Ref`s work in JAX more
generally). Rather than returning outputs as values, kernels receive input and
output `Ref`s and mutate the output `Ref`s in-place:

```python
def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y
```

A `Ref` exposes array-like metadata attributes—`x_ref.shape`, `x_ref.dtype`,
`x_ref.ndim`, `x_ref.size`, and `x_ref.memory_space`—but you cannot pass a `Ref`
directly to `jax.numpy` or `jax.lax` math operations. Instead, you explicitly
read values out of input `Ref`s into immutable `jax.Array`s (representing
registers in Pallas), compute on those `jax.Array`s, and write the result back
to an output `Ref`.

## Reading, writing, and dynamic slicing

* **Reading from `Ref`s:** Indexing a `Ref` (for example `x_ref[...]`,
  `x_ref[:]`, or `x_ref[0, :128]`) reads its contents into an immutable
  `jax.Array`.
* **Writing to `Ref`s:** Assigning to an indexed `Ref` (for example
  `o_ref[...] = x + y` or `o_ref[...] += x`) mutates the underlying memory
  buffer in-place. You can also exchange a value and return the previous
  contents using `old_val = o_ref.swap(new_val)` (or `jax.ref.swap`). Note that
  `.swap` lowers to a regular non-atomic load followed by a store—it is not a
  hardware atomic read-modify-write operation.
* **Dynamic slicing with `pl.ds`:** Standard Python slice syntax like
  `x_vmem[start : start + 8, :]` fails when `start` is a traced runtime scalar
  (such as `jax.lax.axis_index` or a loop index) because JAX requires slice
  sizes to be statically known at compile time. Use
  {func}`pl.ds(start, size) <jax.experimental.pallas.ds>` (alias `pl.dslice`) to
  slice a static `size` starting from a dynamic `start` offset:

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    mesh=pltpu.TensorCoreMesh(axis_name='core', num_cores=1),
    scratch_types=dict(
        x_vmem=pltpu.VMEM((32, 128), jnp.float32),
        o_vmem=pltpu.VMEM((8, 128), jnp.float32),
    ),
)
def sum_tiles_kernel(x_ref, o_ref, x_vmem, o_vmem):
  pltpu.sync_copy(x_ref, x_vmem)
  o_vmem[...] = jax.lax.fori_loop(
      0, 4, lambda i, acc: acc + x_vmem[pl.ds(i * 8, 8), :],
      jnp.zeros((8, 128), dtype=jnp.float32),
  )
  pltpu.sync_copy(o_vmem, o_ref)
```

## Zero-copy views and `Ref` transforms

Instead of immediately reading or writing memory, Pallas lets you construct
zero-copy **`TransformedRef` views** that alias the same underlying buffer.

### Slicing views with `.at`

Calling `x_ref.at[slice]` does not read or copy data; it returns a new `Ref`
view pointing to a slice of the original buffer (for example `ref.at[0:128]`
for the first 128 elements, `ref.at[::2]` for a strided view, or
`ref.at[pl.ds(i * 128, 128)]` for a dynamic tile).

Once you have a sliced `Ref`, you can read or write it with standard indexing,
compose multiple slices, or pass the view to a helper function or DMA copy that
expects a `Ref`:

```python
def add_sliced_kernel(x_ref, y_ref, o_ref):
  small_mid = x_ref.shape[0] // 2

  x_left = x_ref.at[:small_mid]
  x_right = x_ref.at[small_mid:]
  y_left = y_ref.at[:small_mid]
  y_right = y_ref.at[small_mid:]

  # The output shape is (4 * small_mid,).
  large_mid = 2 * small_mid
  o_ref.at[:large_mid][:small_mid] = x_left[...] + y_left[...]
  o_ref.at[:large_mid][small_mid:] = x_left[...] + y_right[...]
  o_ref.at[large_mid:][:small_mid] = x_right[...] + y_left[...]
  o_ref.at[large_mid:][small_mid:] = x_right[...] + y_right[...]
```

Note that `x_ref.at[slice][...]` is equivalent to `x_ref[slice]`.

### Shape, transpose, and bitcast views

`Ref`s also support zero-copy shape, axis, and dtype transforms that return new
`TransformedRef` views without moving data:

* **`ref.reshape(*shape)`:** Reinterprets static buffer dimensions with the same
  element count (for example, `x_ref.reshape(4, 32, 128)`).
* **`ref.transpose(permutation)` / `ref.T`:** Permutes axes lazily as a view
  (for example, `a_ref.transpose((1, 0))` or passing `b_smem.T` to a
  matrix-multiply instruction).
* **`ref.bitcast(dtype)`:** Reinterprets the element dtype in-place without
  numerical conversion. Same-bitwidth bitcasts (such as `jnp.float32` to
  `jnp.int32`) preserve the shape; changing bitwidth on a 2D+ buffer scales
  `shape[-2]` proportionally (for example, bitcasting an `(8, 128)` `int32` TPU
  buffer to `jnp.bfloat16` produces a `(16, 128)` view).

```{note}
Backend support for `Ref` transforms varies: on Mosaic GPU, `ref.bitcast` is not
supported, and `ref.reshape` / `ref.transpose` are not supported on Blackwell
`TMEM` buffers; on TPU, `ref.transpose` / `ref.T` is not supported.
```

## Memory spaces

Unlike standard JAX `Ref`s, every Pallas `Ref` resides in a hardware **memory
space** (`ref.memory_space`), which determines both its capacity/bandwidth and
which operations are legal on it:

* **TPU (`jax.experimental.pallas.tpu` as `pltpu`):**
  * `pltpu.HBM`: High-bandwidth device memory. Kernel inputs and outputs live
    here by default; TensorCore compute cannot read or write `HBM` directly, so
    data is moved via `pltpu.sync_copy` / `pltpu.async_copy` or
    `pltpu.emit_pipeline`.
  * `pltpu.VMEM`: Fast on-chip vector scratchpad SRAM where vector and matrix
    computations take place.
  * `pltpu.SMEM`: Fast on-chip scalar SRAM for scalar metadata, dynamic loop
    bounds, and indirect indices.
* **Mosaic GPU (`jax.experimental.pallas.mosaic_gpu` as `plgpu`):**
  * `plgpu.GMEM`: Global GPU memory (HBM) for kernel inputs and outputs.
  * `plgpu.SMEM`: Per-SM shared memory used for staging tiles across threads in
    a block and feeding Hopper `wgmma` Tensor Core operations.
  * `plgpu.TMEM`: Tensor Memory on Blackwell GPUs for MMA accumulators and
    operands.

See {ref}`pallas-tpu-quickstart` and {ref}`pallas-gpu-quickstart` for end-to-end
examples of moving data across memory spaces on each backend.

## Allocating scratch `Ref`s

When a kernel needs temporary on-chip buffers (such as `VMEM` staging buffers,
accumulators, or semaphores), you can allocate them in two ways:

1. **Kernel-level `scratch_types`:** Pass a dictionary (for keyword arguments)
   or sequence (for positional arguments after outputs) of memory-space types to
   {func}`@pl.kernel <jax.experimental.pallas.kernel>` or `@plgpu.kernel`:

```python
@pl.kernel(
    out_type=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    mesh=pltpu.TensorCoreMesh(axis_name='core', num_cores=1),
    scratch_types=dict(buf_vmem=pltpu.VMEM((8, 128), jnp.float32)),
)
def scale_kernel(x_ref, o_ref, buf_vmem):
  pltpu.sync_copy(x_ref, buf_vmem)
  buf_vmem[...] = buf_vmem[...] * 2.0
  pltpu.sync_copy(buf_vmem, o_ref)
```

2. **Scoped allocations (`pl.run_scoped` / `@pl.with_scoped`):** Allocate
   temporary `Ref`s whose lifetime is restricted to an inner scope inside a
   kernel body:

```python
@pl.with_scoped(tmp_vmem=pltpu.VMEM((8, 128), jnp.float32))
def stage_scale(x_ref, o_ref, tmp_vmem):
  pltpu.sync_copy(x_ref, tmp_vmem)
  tmp_vmem[...] = tmp_vmem[...] * 2.0
  pltpu.sync_copy(tmp_vmem, o_ref)
```
