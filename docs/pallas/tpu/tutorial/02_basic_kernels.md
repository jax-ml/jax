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

(pallas-tpu-tutorial-02-basic-kernels)=
# Chapter 2: Basic Kernels

<!--* freshness: { reviewed: '2026-11-11' } *-->

To understand how to write custom kernels, we first need to bridge the gap
between high-level JAX and low-level hardware. At its core, kernel writing
involves two main conceptual shifts: moving from immutable values to mutable
memory references, and explicitly mapping those computations onto a hardware
mesh.

## From Values to References (Refs)

In standard JAX, you program with **values** (typically `jax.Array`). These
arrays are immutable and represent logical mathematical tensors. When you write
a normal JAX function, you don't worry about where the arrays are allocated, how
they are laid out in memory, or how to safely mutate them in place. The compiler
abstracts all of that away for you.

When stepping down a level of abstraction, you do not operate on logical values
directly; instead, you operate on **References** (`Ref`).

A `Ref` is essentially a mutable array. The inputs you operate on are `Ref`s,
representing allocated memory buffers. Your job is to explicitly read data from
these input `Ref`s—and crucially, when you read from a `Ref`, you get back an
immutable JAX value. You can then perform computations on these values, and
write the resulting values back into an output `Ref`.

```{code-cell}
import jax


# Standard JAX: Immutable logical values
def add_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  return x + y


# Mutable memory references
def add_refs(x_ref, y_ref, o_ref):
  # 1. Read immutable values out of refs
  x = x_ref[...]
  y = y_ref[...]

  # 2. Compute and write immutable value into ref
  o_ref[...] = x + y
```

To learn more about `Ref`s and how they work in JAX and Pallas, refer to
{ref}`jax-101-refs` and {ref}`Working with Refs <pallas_refs>`.

## Refs with Memory Spaces

In addition to being a mutable array, in Pallas every `Ref` is associated with a
specific **memory space**. This is a Pallas-specific concept that allows you to
control where data resides on the hardware, and does not exist in standard JAX.

These memory spaces directly correspond to the TPU memory hierarchy we
introduced in {ref}`Chapter 1 <pallas-tpu-tutorial-01-tpus>` (HBM, VMEM, and
SMEM).

When you write a kernel function, the input `Ref`s you receive generally reside
in **HBM** (High Bandwidth Memory). As we discussed previously, the TPU's
compute units (like the MXU and VPU) cannot access HBM directly.

Therefore, you cannot perform math directly on these HBM `Ref`s. Instead, the
operations you can perform on an HBM `Ref` are essentially limited to
asynchronous DMA copies. A typical kernel will allocate temporary `Ref`s that
reside in **VMEM** or **SMEM**, issue a DMA copy to move the data from the HBM
`Ref` into the VMEM `Ref`, perform the computation on the VMEM `Ref`, and
finally issue another DMA copy to move the result back to an output HBM `Ref`.

## Mapping Computation with `pl.kernel`

Now that we have a way to operate on memory buffers, how do we execute this code
on the hardware? This is where Pallas and the
{func}`pl.kernel <jax.experimental.pallas.kernel>` API come in. It acts as a
function transformation—similar to `jax.shard_map`—that both maps this function
across cores and automatically converts JAX arrays into `Ref`s.

For example, if you are mapping across cores, a TPU v5p has 2 TensorCores per
chip, while a TPU v6e has only 1 TensorCore per chip.

```{code-cell}
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Construct the TensorCore mesh
tc_mesh = pltpu.TensorCoreMesh(axis_name='core')


def kernel(x_ref: jax.Ref, y_ref: jax.Ref, o_ref: jax.Ref):
  # Runs on all TensorCores simultaneously
  ...


def fn_explicit(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.kernel(
      kernel,
      out_type=jax.typeof(x),  # out_type allocates the output Ref returned by fn_explicit
      mesh=tc_mesh,
  )(x, y)
```

Or equivalently, using `pl.kernel` as a decorator:

```{code-cell}
def fn_decorator(x: jax.Array, y: jax.Array) -> jax.Array:
  @pl.kernel(out_type=jax.typeof(x), mesh=tc_mesh)
  def kernel(x_ref: jax.Ref, y_ref: jax.Ref, o_ref: jax.Ref):
    # Runs on all TensorCores simultaneously
    ...

  return kernel(x, y)
```

If you are familiar with `jax.shard_map`, you already understand the fundamental
paradigm here. `pl.kernel` runs the provided `kernel` function on a **mesh** of
cores, just as `shard_map` runs a function on a mesh of devices. The primary
difference is that while `shard_map` operates on JAX values, `pl.kernel`
automatically promotes them to `Ref`s within the kernel body.

By combining these two concepts—`Ref`s for memory manipulation and `pl.kernel`
for mesh execution—we gain the granular control needed to maximize hardware
utilization.

## A Complete Kernel with Memory Movement

Now let's put it all together and write a complete kernel that moves data
between HBM and VMEM and performs a computation.

To do this, we need to allocate scratch memory in VMEM. We can do this by
passing the `scratch_types` argument to `pl.kernel`. We can use
`pltpu.VMEM.like` to allocate VMEM buffers that match the shape and dtype of our
inputs.

Here is a complete example of a working add kernel:

```{code-cell}
def add_kernel(
    x_ref: jax.Ref,
    y_ref: jax.Ref,
    o_ref: jax.Ref,
    x_vmem: jax.Ref,
    y_vmem: jax.Ref,
    o_vmem: jax.Ref,
):
  # 1. Sync copy from HBM to VMEM
  pltpu.sync_copy(x_ref, x_vmem)
  pltpu.sync_copy(y_ref, y_vmem)

  # 2. Read from VMEM, compute, and write to VMEM
  x = x_vmem[...]
  y = y_vmem[...]
  o_vmem[...] = x + y

  # 3. Sync copy from VMEM back to HBM
  pltpu.sync_copy(o_vmem, o_ref)


def fn(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.kernel(
      add_kernel,
      out_type=jax.typeof(x),
      mesh=tc_mesh,
      scratch_types=[
          pltpu.VMEM.like(x),
          pltpu.VMEM.like(x),
          pltpu.VMEM.like(x),
      ],
  )(x, y)
```

In this example:

* We pass `scratch_types` to `pl.kernel`, which allocates three VMEM buffers.
* These scratch buffers are passed as trailing arguments to our `add_kernel`
  function (`x_vmem`, `y_vmem`, `o_vmem`).
* We use `pltpu.sync_copy` to move data between HBM and VMEM.
* Finally, we perform the computation on the VMEM buffers and copy the result
  back to the output HBM reference.

## Parallelizing Across Cores

If you run the above example on a device with multiple cores (like a TPU v5p
with 2 TensorCores), you might notice that it is doing redundant work. Both
cores will execute the exact same instructions, read the same data from HBM, and
write the same result back to HBM. While this is not a race condition (because
both cores write identical bits), it is a waste of compute resources.

To fix this, we can use the `axis_name='core'` that we specified when creating
our TensorCore mesh. We can use `jax.lax.axis_index('core')` to determine which
core is currently executing the kernel, and then split the work among them.

Here is how we can update the example to split the work along the leading
dimension across cores:

```{code-cell}
def parallel_add_kernel(
    x_ref: jax.Ref,
    y_ref: jax.Ref,
    o_ref: jax.Ref,
    x_vmem: jax.Ref,
    y_vmem: jax.Ref,
    o_vmem: jax.Ref,
):
  core_idx = jax.lax.axis_index('core')
  num_cores = jax.lax.axis_size('core')

  # Calculate the slice size for this core
  slice_size = x_ref.shape[0] // num_cores
  start_idx = core_idx * slice_size
  slc = pl.ds(start_idx, slice_size)

  # 1. Sync copy only this core's slice from HBM to VMEM
  pltpu.sync_copy(x_ref.at[slc], x_vmem)
  pltpu.sync_copy(y_ref.at[slc], y_vmem)

  # 2. Read from VMEM, compute, and write to VMEM
  x = x_vmem[...]
  y = y_vmem[...]
  o_vmem[...] = x + y

  # 3. Sync copy from VMEM back to this core's slice in HBM
  pltpu.sync_copy(o_vmem, o_ref.at[slc])


def fn_parallel(x: jax.Array, y: jax.Array) -> jax.Array:
  # Construct the TensorCore mesh
  tc_mesh = pltpu.TensorCoreMesh(axis_name='core')

  # Allocate VMEM space for only this core's slice
  slice_shape = (x.shape[0] // tc_mesh.num_cores, *x.shape[1:])

  return pl.kernel(
      parallel_add_kernel,
      out_type=jax.typeof(x),
      mesh=tc_mesh,
      scratch_types=[
          pltpu.VMEM(slice_shape, x.dtype),
          pltpu.VMEM(slice_shape, x.dtype),
          pltpu.VMEM(slice_shape, x.dtype),
      ],
  )(x, y)
```

By using `x_ref.at[slc]`, each core reads a disjoint slice of the input data into
its local VMEM, processes it, and writes it back to a disjoint slice of the
output HBM. On a 2-TensorCore chip, this effectively doubles our throughput by
utilizing both cores!

```{seealso} Where to go deeper
* {ref}`Working with Refs <pallas_refs>` — complete guide to `Ref` indexing,
  dynamic slicing with `pl.ds`, transformed views, and scratch allocation.
* {doc}`Writing TPU kernels with Pallas </pallas/tpu/details>` — reference for
  multi-core `TensorCoreMesh` configurations, VMEM/SMEM rules, and TPU vector
  tiling.
* {ref}`TPU Quickstart <pallas-tpu-quickstart>` — compact overview of
  `pl.kernel`, `pltpu.sync_copy`, and `pltpu.emit_pipeline`.
```
