(pallas_refs)=

# Working with Refs

<!--* freshness: { reviewed: '2026-10-27' } *-->

Unlike standard JAX functions, which take and return immutable `jax.Array`
values, Pallas kernels operate on *`Ref`* objects representing mutable buffers
in device memory. Rather than returning outputs as values, kernels receive input
and output `Ref`s and mutate the output `Ref`s in-place:

```python
def add_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y
```

A `Ref` exposes array-like metadata attributes—`x_ref.shape`, `x_ref.dtype`,
`x_ref.ndim`, `x_ref.size`, and `x_ref.memory_space`—but you cannot pass a `Ref`
directly to `jax.numpy` or `jax.lax` math operations. Instead, you explicitly
read values out of input `Ref`s into immutable `jax.Array`s (representing
registers in Pallas), compute on those `jax.Array`s, and write the result back
to an output `Ref` (or exchange values via `old_val = o_ref.swap(new_val)`,
which lowers to a regular non-atomic load followed by a store).

`Ref`s are a general JAX feature—for how `Ref`s work outside of kernels, see:
* {ref}`Refs: mutable arrays <jax-101-refs>` for creating `Ref`s
  (`jax.new_ref`, `jax.freeze`), NumPy-style indexed reads and writes
  (including dynamic slicing with {func}`jax.ds`), purity rules, and aliasing
  restrictions.
* {ref}`In-place updates with refs <jax-201-jit-refs>` for mutating buffers
  in-place across `jax.jit` boundaries.
* {ref}`Autodiff with refs <jax-301-refs>` for differentiating functions that
  use `Ref`s.

The rest of this guide covers `Ref` features specific to Pallas kernels.

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

When a kernel needs temporary on-chip buffers (such as `VMEM`/`SMEM` staging
buffers, accumulators, barriers, or semaphores), you can allocate them in two
ways:

1. **Kernel-level `scratch_types`:** Pass a dictionary (for keyword arguments)
   or sequence (for positional arguments after outputs) of memory-space types to
   {func}`@pl.kernel <jax.experimental.pallas.kernel>` or `@plgpu.kernel`:

`````{tab-set}
````{tab-item} TPU
:sync: tpu

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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
````
````{tab-item} GPU
:sync: gpu

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

@plgpu.kernel(
    out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
    scratch_types=dict(
        buf_smem=plgpu.SMEM((128,), jnp.float32),
        bar=plgpu.Barrier(),
    ),
)
def scale_kernel(x_gmem, o_gmem, buf_smem, bar):
  plgpu.copy_gmem_to_smem(x_gmem, buf_smem, bar)
  plgpu.barrier_wait(bar)
  buf_smem[...] = buf_smem[...] * 2.0
  plgpu.commit_smem()
  plgpu.copy_smem_to_gmem(buf_smem, o_gmem)
  plgpu.wait_smem_to_gmem(0)
```
````
`````

2. **Scoped allocations (`pl.run_scoped` / `@pl.with_scoped`):** Allocate
   temporary `Ref`s whose lifetime is restricted to an inner scope inside a
   kernel body:

`````{tab-set}
````{tab-item} TPU
:sync: tpu

```python
@pl.with_scoped(tmp_vmem=pltpu.VMEM((8, 128), jnp.float32))
def stage_scale(x_ref, o_ref, tmp_vmem):
  pltpu.sync_copy(x_ref, tmp_vmem)
  tmp_vmem[...] = tmp_vmem[...] * 2.0
  pltpu.sync_copy(tmp_vmem, o_ref)
```
````
````{tab-item} GPU
:sync: gpu

```python
@pl.with_scoped(
    tmp_smem=plgpu.SMEM((128,), jnp.float32),
    bar=plgpu.Barrier(),
)
def stage_scale(x_gmem, o_gmem, tmp_smem, bar):
  plgpu.copy_gmem_to_smem(x_gmem, tmp_smem, bar)
  plgpu.barrier_wait(bar)
  tmp_smem[...] = tmp_smem[...] * 2.0
  plgpu.commit_smem()
  plgpu.copy_smem_to_gmem(tmp_smem, o_gmem)
  plgpu.wait_smem_to_gmem(0)
```
````
`````

## Zero-copy views and `Ref` transforms

Instead of immediately reading or writing memory, Pallas lets you construct
zero-copy **`TransformedRef` views** that alias the same underlying buffer.

### Slicing views with `.at`

Indexing `x_ref.at[slice]` does not read or copy data; it returns a new `Ref`
view pointing to a slice of the original buffer (for example, `x_ref.at[:128]`
for the first 128 elements, or `x_ref.at[pl.ds(i * 128, 128)]` for the 128
elements starting at `i * 128`).

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

When slicing a tile at a runtime grid, core, or loop index (where Python
`start : start + tile` fails under tracing), use
{func}`pl.ds(start, size) <jax.experimental.pallas.ds>` (`pl.dslice`, a
re-export of {func}`jax.ds` covered in {ref}`jax-101-refs`). Because `pltpu.HBM`
and `plgpu.GMEM` `Ref`s cannot be dereferenced directly with `[...]`, passing
`.at[pl.ds(...)]` views to DMA copy primitives is the standard way to stage
individual tiles:

`````{tab-set}
````{tab-item} TPU
:sync: tpu

```python
@pl.kernel(
    out_type=jax.ShapeDtypeStruct((16, 128), jnp.float32),
    mesh=pltpu.TensorCoreMesh(axis_name='core', num_cores=2),
    scratch_types=dict(buf_vmem=pltpu.VMEM((8, 128), jnp.float32)),
)
def scale_tiles_kernel(x_hbm, o_hbm, buf_vmem):
  core_idx = jax.lax.axis_index('core')
  tile = pl.ds(core_idx * 8, 8)
  pltpu.sync_copy(x_hbm.at[tile], buf_vmem)
  buf_vmem[...] = buf_vmem[...] * 2.0
  pltpu.sync_copy(buf_vmem, o_hbm.at[tile])
```
````
````{tab-item} GPU
:sync: gpu

```python
@plgpu.kernel(
    out_type=jax.ShapeDtypeStruct((256,), jnp.float32),
    grid=(2,),
    grid_names=('x',),
    scratch_types=dict(
        buf_smem=plgpu.SMEM((128,), jnp.float32),
        bar=plgpu.Barrier(),
    ),
)
def scale_tiles_kernel(x_gmem, o_gmem, buf_smem, bar):
  pid = jax.lax.axis_index('x')
  tile = pl.ds(pid * 128, 128)
  plgpu.copy_gmem_to_smem(x_gmem.at[tile], buf_smem, bar)
  plgpu.barrier_wait(bar)
  buf_smem[...] = buf_smem[...] * 2.0
  plgpu.commit_smem()
  plgpu.copy_smem_to_gmem(buf_smem, o_gmem.at[tile])
  plgpu.wait_smem_to_gmem(0)
```
````
`````

Dynamic `start` offsets and `size` lengths must satisfy backend tile and DMA
alignment rules (for example, `(8, 128)` register/DMA alignment on TPU and
tile/swizzle boundaries on Mosaic GPU; wrap dynamic offsets in
{func}`pl.multiple_of(start, tile) <jax.experimental.pallas.multiple_of>` when
the compiler cannot statically prove divisibility), and `.at` views only support
unit-stride (`stride=1`) slices.

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
