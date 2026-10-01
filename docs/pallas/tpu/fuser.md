# Fuser API: Fuse operations around kernels

This guide walks through the Pallas Fuser API. This API allows user to fuse arbitrary JAX operations into existing Pallas kernels, without making additional hand-written kernel variants.

## The Idea of Kernel Fusion

By default, a Pallas kernel takes and outputs HBM refs. This means ops immediately before and after them may take extra HBM round trips, even if they are very simple (e.g., element-wise ops).

The `pallas.fuser` API avoids this by folding the surrounding JAX operations into a kernel's HBM <-> VMEM pipeline directly. This allows you to only hand-write the complicated core part into a Pallas kernel, and iterate quickly on the more trival computations (activation functions, etc).

A sample API would look like:

```python
@fuser.fuse
def fused_kernel(x):
  x_inner = input_jax_ops(x)
  y_inner = a_pallas_kernel(x_inner)  # The real Pallas kernel
  y = output_jax_ops(y_inner)
  return y
```

And in the actual computation, both the input and output JAX ops will happen inside the pipeline, on the VMEM level. In other words, the core Pallas kernel will be expanded to include these additional ops.

```text
         HBM inputs (`x`)
              |
              |  Pipelined DMA copy-in
              v
        Input fusion ops +
The main Pallas kernel pipeline body +
       Output fusion ops
              |
              |  Pipelined DMA copy-out
              v
          HBM outputs (`y`)
```

### But... How?

JAX sees your full fused kernel expression as three parts: the core Pallas kernel call, the computations happening before it (input fusion), and after (output fusion).

From the perspective inside a kernel, these extra information are needed to carry out the fusions:

* The actual computation ops of the input/output fusions.

* The additional arrays involved in the fusions.
  * This includes normal JAX arrays, and prefetched index scalars.

* The expected `BlockSpec`s of all the arrays involved in the fused kernel.
  * This includes the block sizes, and the index map to access the blocks.

The core Pallas kernel must contain some extra API calls to programmatically derive these information and execute the fusion stages. The next section will go through them in detail.

```python
import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
```

## Kernel Writer Side API

It falls on the kernel writers to carry out the fusions. We will use a simple pipelined kernel of `f(x) = 2 * x` to demonstrate this end-to-end.

### The in-pipeline body

The function `double_body` below is the function body inside our kernel pipeline. Here's where input and output fusions actually occur. That means it needs some extra args:

* `x_fusion_refs`, `y_fusion_refs`: The extra data refs used for the input and output fusions.

* `x_fn`, `y_fn`: The actual fusion computations, as function callbacks.

See the code below for the exact execution API.

Note that the fusion functions take and return **values** instead of refs - this is because JAX computation graphs are value-based, unlike Pallas kernel functions. Some extra boilerplates are needed to convert between these two types.

```python
def double_body(
    x_fusion_refs,
    y_fusion_refs,
    y_out_ref,
    *,
    x_fn: Any,
    y_fn: Any,
):
  pids = (pl.program_id(0), pl.program_id(1))

  # Carry out the input fusion and derive the real x.
  x_fusion_vals = jax.tree.map(lambda r: r.get(), x_fusion_refs)
  x = x_fn(pids, (), x_fusion_vals)

  # Main computation body.
  y = 2 * x

  # Carry out the output fusion.
  y_fusion_vals = jax.tree.map(lambda r: r.get(), y_fusion_refs)
  y = y_fn(pids, (), y_fusion_vals, y)

  # Set the final output to the pipelined body's output ref.
  jax.tree.map(lambda r, v: r.set(v), y_out_ref, y)
```

### The `fusible` kernel body

Write your kernel normally first - figure out your pipeline's `grid`, `BlockSpec`s, and output type. Your end result should be a Pallas kernel function created by `pl.kernel` or `pl.pallas_call`.

Now, wrap the function with `fuser.fusible`. Adjust it to take `fuser.Fusion` instances as inputs, which represents all the input and output fusions.

Inside the function body, add API calls to programmically figure out the right values and BlockSpecs to be fed into the core Pallas kernel:

* `fuser.get_fusion_values`: Extract all the needed input JAX arrays (as values) and the computation op out from each fusion.

* `fuser.pull_block_spec`: For each input JAX array, use the computation op and the core kernel's `grid` and `block_spec` to derive its real `BlockSpec` at the kernel input time.

  * This is called "pull" because it's "pulling" the core block spec upwards and finding its corresponding version at the beginning of a fusion.

  * This step is needed for both input and output fusions, because both may contain extra input arrays.

* `fuser.push_block_spec`: For each kernel output (one per output fusion), push it down to derive its real `BlockSpec` at the kernel output time.

  * In the example below, the push and pull of the output fusion is combined in `fuser.push_pull_block_spec`.

After that, use the new block specs to construct the real kernel, using `emit_pipeline`.

```python
@fuser.fusible
def double(
    x: fuser.Fusion[[], jax.Array],
    y: fuser.Fusion[[jax.Array], jax.Array] | None = None,
    *,
    bm: int = 128,
    bn: int = 128,
) -> jax.Array:
  m, n = x.shape
  grid = (m // bm, n // bn)
  spec = pl.BlockSpec((bm, bn), lambda i, j: (i, j))
  y_type = jax.ShapeDtypeStruct((m, n), x.dtype)

  if y is None:
    y = lambda out: out

  # 1. Extract closed-over arrays from input and output fusions.
  x_fn, x_fusion_values, _ = fuser.get_fusion_values(x)
  y_fn, y_fusion_values, _ = fuser.get_fusion_values(y, y_type)

  # 2. Pull BlockSpecs backward through input fusion.
  x_fn, (x_fusion_specs,), _ = fuser.pull_block_spec(
      x_fn, spec, grid_len=len(grid)
  )(x_fusion_values)

  # 3. Push and pull BlockSpecs through output fusion.
  y_fn, y_fusion_specs, _, y_out_type, y_out_spec = (
      fuser.push_pull_block_spec(y_fn, spec, grid_len=len(grid))(
          y_fusion_values, y_type
      )
  )

  # 4. Launch with pl.kernel + pltpu.emit_pipeline.
  def body(x_fusion_refs, y_fusion_refs, y_out_ref):
    pltpu.emit_pipeline(
        functools.partial(double_body, x_fn=x_fn, y_fn=y_fn),
        grid=grid,
        in_specs=(x_fusion_specs, y_fusion_specs),
        out_specs=(y_out_spec,),
        core_axis_name='core',
        dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
    )(x_fusion_refs, y_fusion_refs, y_out_ref)

  return pl.kernel(
      body,
      out_type=(y_out_type,),
      mesh=pltpu.TensorCoreMesh(axis_name='core'),
  )(x_fusion_values, y_fusion_values)[0]
```

## User Side API (`@fuser.fuse`)

On user side, write the fused kernel like a normal JAX function, and call the fusible kernel within.

```python
x = jax.random.normal(jax.random.key(0), (256, 256), jnp.float32)
a = jax.random.normal(jax.random.key(1), (256, 256), jnp.float32)

@jax.jit
@fuser.fuse
def fused_double(x):
  y = double(x + a)      # `x + a` is an input fusion
  return jax.nn.relu(y)  # `relu(y)` is an output fusion

out = fused_double(x)
expected = jax.nn.relu(2 * (x + a))
np.testing.assert_allclose(out, expected, atol=1e-5)
```

### Shape-changing fusions

Under the hood, Pallas fuser tracks how a JAX primitive makes dimension changes on the oprands, and define rules for transforming the block specs. This means it's possible to use shape-changing primitives in fusions.

The example below shows how the correct block specs were derived given an input/output fusion.

```python
a = jax.random.normal(jax.random.key(0), (256, 256), jnp.float32)
b = jax.random.normal(jax.random.key(1), (4, 256, 256), jnp.float32)

def input_fusion(a, b):
  b2 = jax.lax.dynamic_slice(b, (2, 0, 0), (1, 256, 256))[0]  # (256, 256)
  return jnp.concatenate([a, b2], axis=0)                     # (512, 256)

def output_fusion(y):
  return y.T                                                  # (256, 512)

def block_idx(spec, i, j):
  return tuple(int(c) for c in spec.index_map(i, j))


# Before fusion: the BlockSpec `double` uses for x and y on a (4, 2) grid.
spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
assert block_idx(spec, 3, 1) == (3, 1)

# After pulling through the input fusion: one BlockSpec per captured array.
x_fn, x_fusion_values, _ = fuser.get_fusion_values(
    lambda: input_fusion(a, b)
)
assert [v.shape for v in x_fusion_values] == [(4, 256, 256), (256, 256)]
_, ((b_spec, a_spec),), _ = fuser.pull_block_spec(x_fn, spec, grid_len=2)(
    x_fusion_values
)
assert b_spec.block_shape == (None, 128, 128)  # dynamic_slice squeezes dim 0
assert block_idx(b_spec, 3, 1) == (2, 1, 1)    # slice index 2, row 3 - 2 = 1
assert a_spec.block_shape == (128, 128)
assert block_idx(a_spec, 1, 1) == (1, 1)       # top half reads `a` directly
assert block_idx(a_spec, 3, 1) == (1, 1)       # bottom half: clamped, unused

# After pushing through the output fusion: transposed output and index map.
y_type = jax.ShapeDtypeStruct((512, 256), jnp.float32)
y_fn, y_fusion_values, _ = fuser.get_fusion_values(output_fusion, y_type)
_, _, _, out_type, out_spec = fuser.push_pull_block_spec(
    y_fn, spec, grid_len=2
)(y_fusion_values, y_type)
assert out_type.shape == (256, 512)
assert block_idx(out_spec, 3, 0) == (0, 3)


# Real-run verification
@jax.jit
@fuser.fuse
def fused_double(a, b):
  return output_fusion(double(input_fusion(a, b)))

out = fused_double(a, b)
expected = output_fusion(2 * input_fusion(a, b))
np.testing.assert_allclose(out, expected, atol=1e-5)
```

For a more complex example, including a pipelined matmul with a K-reduction grid, scalar prefetch, and many input/output fusions, see [tpu_fusible_matmul_test.py](https://github.com/jax-ml/jax/blob/main/tests/pallas/tpu_fusible_matmul_test.py).

## Other Topics

### Scalar Prefetch in Index Maps

When an input fusion uses a dynamic array as an index (`x_stack[idx]`), `get_fusion_values` will identify and extract it as part of the `scalar_prefetch` values being returned.

If the kernel wants to handle it in the scalar-prefetch way, it needs to allocate additional SMEM scratch spaces, do the sync copy, and update the derived block specs with the scalar prefetched buffers using `block_spec_with_prefetch`.

```python
def double_body_with_sp(
    x_scalar_prefetch,
    x_fusion_refs,
    o_ref,
    *,
    x_fn: Any,
):
  pids = (pl.program_id(0), pl.program_id(1))
  x_fusion_vals = jax.tree.map(lambda r: r.get(), x_fusion_refs)
  x = x_fn(pids, (x_scalar_prefetch,), x_fusion_vals)
  o_ref[...] = 2 * x


@functools.partial(fuser.fusible, output_fusion_prefix=False)
def double_with_sp(
    x: fuser.Fusion[[], jax.Array],
    _out_fusion: None = None,
    *,
    bm: int = 128,
    bn: int = 128,
) -> jax.Array:
  m, n = x.shape
  grid = (m // bm, n // bn)
  spec = pl.BlockSpec((bm, bn), lambda i, j, *_: (i, j))

  x_fn, x_fusion_values, x_sp = fuser.get_fusion_values(x)
  x_fn, (x_fusion_specs,), _ = fuser.pull_block_spec(
      x_fn,
      spec,
      scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(0),
      grid_len=len(grid),
  )(x_fusion_values)

  scalar_prefetch = jax.tree.map(lambda s: s[None], (x_sp,))

  def body(x_fusion_refs, sp_refs, out_ref, sp_smem_refs):
    pltpu.sync_copy(sp_refs, sp_smem_refs)
    in_specs_ = fuser.block_spec_with_prefetch((x_fusion_specs,), sp_smem_refs)
    pltpu.emit_pipeline(
        functools.partial(double_body_with_sp, *sp_smem_refs, x_fn=x_fn),
        grid=grid,
        in_specs=in_specs_,
        out_specs=spec,
        core_axis_name='core',
        dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
    )(x_fusion_refs, out_ref)

  return pl.kernel(
      body,
      out_type=jax.ShapeDtypeStruct((m, n), x.dtype),
      mesh=pltpu.TensorCoreMesh(axis_name='core'),
      scratch_types=[jax.tree.map(pltpu.SMEM.like, scalar_prefetch)],
  )(x_fusion_values, scalar_prefetch)


x_stack = jax.random.normal(jax.random.key(0), (4, 256, 256), jnp.float32)


@jax.jit
@fuser.fuse
def fused_double_with_sp(x_stack, idx):
  return double_with_sp(x_stack[idx])


out_sp = fused_double_with_sp(x_stack, 3)
np.testing.assert_allclose(out_sp, x_stack[3] * 2.0, atol=1e-5)
```

### Custom Fusion

Pallas traces a fusion as a program of JAX primitives and derives block specs from that trace. `@fuser.custom_fusion` wraps a normal JAX function into a custom fusion op, allowing you to bypass some primitive tracing and apply additional code:

* `def_pull_block_spec` (required) / `def_push_block_spec` (optional, for output fusions): Define how block transforms and `BlockSpec`s are propagated.

* `def_eval_rule` (required): Define how the op is rewritten per block `(ctx, *tile_args) -> (tile_out,)` when pulled into the kernel.

* `def_pallas_impl` (optional): Define an alternative implementation lowered by Pallas when the custom op itself is called inside the kernel (e.g., from `def_eval_rule`).

```python
@fuser.custom_fusion
def add_row(x, row):
  # Normal JAX implementation (and shape/dtype inference).
  return x + row

@add_row.def_pull_block_spec
def _(out_transforms):
  (t,) = out_transforms
  row_t = t.replace(
      block_shape=(1, *t.block_shape[1:]),
      block_index_transform=lambda *idxs: (0, *t.block_index_transform(*idxs)[1:]),
  )
  return (t, row_t)

@add_row.def_push_block_spec
def _(in_specs):
  x_spec, _ = in_specs
  return (x_spec,)

@add_row.def_eval_rule
def _(ctx, x_tile, row_tile):
  del ctx
  return (add_row(x_tile, row_tile),)

@add_row.def_pallas_impl
def _(x_tile, row_tile):
  return x_tile + jnp.broadcast_to(row_tile, x_tile.shape)

@jax.jit
@fuser.fuse
def double_with_row(x, row):
  y = double(add_row(x, row))  # input fusion
  return add_row(y, row)       # output fusion

row = jax.random.normal(jax.random.key(3), (1, 256), jnp.float32)
out_row = double_with_row(x, row)
expected_row = add_row(2 * add_row(x, row), row)
np.testing.assert_allclose(out_row, expected_row, atol=1e-5)
```
