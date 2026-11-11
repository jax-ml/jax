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

(pallas-tpu-tutorial-05-aliasing-mutation)=
# Chapter 5: Aliasing and Mutation

<!--* freshness: { reviewed: '2026-11-11' } *-->

In standard JAX, functions are pure and arrays are immutable. Kernels, however,
operate on data represented as mutable references (`Ref`s). This allows you to
write kernels that mutate data in place.

However, working with `Ref`s on TPUs comes with an important restriction: **you
cannot read or write directly to `Ref`s that reside in HBM**. Attempting to do
`x = x_ref[...]` or `o_ref[...] = x` on an HBM reference is not supported.
Instead, you must use memory operations like `pltpu.sync_copy` to move data
between HBM and VMEM (or SMEM). Once the data is in VMEM, you can read and write
it directly.

{func}`pl.kernel <jax.experimental.pallas.kernel>` supports two main ways to
work with mutable references:

1.  **Passed-in Refs:** References passed as explicit arguments to the kernel.
2.  **Closed-over Refs:** References defined in the scope outside the kernel
    that the kernel function closes over.

## Mutating Passed-in Refs

You can pass references into a kernel and mutate them. Because of the HBM
restriction, we can't directly perform any compute on them, so for now we'll
just issue an HBM -> HBM synchronous copy.

```{code-cell}
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fn(x: jax.Array) -> jax.Array:
  # Create Refs outside the kernel
  x_ref = jax.new_ref(x)
  o_ref = jax.empty_ref(jax.typeof(x))

  def run(x_ref: jax.Ref, o_ref: jax.Ref):
    pltpu.sync_copy(x_ref, o_ref)

  mesh = pltpu.TensorCoreMesh(axis_name='core', num_cores=1)
  # Pass Refs directly into the kernel
  pl.kernel(run, mesh=mesh)(x_ref, o_ref)  # Mutates o_ref!
  return jax.freeze(o_ref)
```

## Closing over `Ref`s

Another option with `pl.kernel` is the ability to close over `Ref`s defined
outside the kernel function.

In these cases, the closed-over references often reside in HBM, so we must use
`sync_copy` to interact with them, or first copy them into VMEM before
performing operations.

```{code-cell}
def fn_closing_over(x: jax.Array) -> jax.Array:
  # Create Refs outside the kernel
  x_ref = jax.new_ref(x)
  o_ref = jax.empty_ref(jax.typeof(x))

  def kernel(x_vmem: jax.Ref, o_vmem: jax.Ref):
    # We close over x_ref and o_ref from the outer scope
    pltpu.sync_copy(x_ref, x_vmem)
    o_vmem[...] = x_vmem[...] + 1
    pltpu.sync_copy(o_vmem, o_ref)

  tc_mesh = pltpu.TensorCoreMesh(axis_name='core', num_cores=1)
  pl.kernel(
      kernel,
      mesh=tc_mesh,
      scratch_types=[pltpu.VMEM.like(x), pltpu.VMEM.like(x)],
  )()  # Mutates o_ref!

  return jax.freeze(o_ref)
```

## Aliasing with `emit_pipeline`

A subtle but critical edge case arises when you pass the same HBM `Ref` multiple
times into `emit_pipeline`—for example, when trying to perform an in-place
operation where a buffer is both an input and an output.

Even though you pass the *same* HBM `Ref` as multiple arguments to
`emit_pipeline`, it will allocate **distinct, separate buffers in VMEM** for
each argument in the body function. They are *not* aliased in VMEM.

Consider an attempt to add 1 to a matrix in place:

```{code-cell}
def add_one_body(x_vmem: jax.Ref, o_vmem: jax.Ref):
  # Inside the body, x_vmem and o_vmem are DISTINCT buffers in VMEM,
  # even though they are backed by the same HBM Ref.
  o_vmem[...] = x_vmem[...] + 1


def inplace_add_kernel(x_hbm: jax.Ref):
  pltpu.emit_pipeline(
      add_one_body,
      grid=(x_hbm.shape[0] // 128, x_hbm.shape[1] // 128),
      in_specs=[pl.BlockSpec((128, 128), lambda i, j: (i, j))],
      out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
  )(x_hbm, x_hbm)  # Passing x_hbm twice!
```

Because `x_vmem` and `o_vmem` are separate buffers:

1.  `x_vmem` will contain the value copied from HBM at the start of the pipeline
    stage.
2.  `o_vmem` is a separate buffer that will be copied *back* to HBM at the end
    of the stage.
3.  At any point during the pipeline, they may contain different values.

You must be careful not to assume that writing to `o_vmem` immediately updates
`x_vmem` or that they share the same memory space in VMEM. They are independent
staging areas for the same underlying HBM data.

Furthermore, if the pipeline **revisits** the same HBM location across different
iterations (e.g., reading from and writing to overlapping regions), there are no
guarantees about when writes are flushed back to HBM. Consequently, you cannot
rely on a "read-after-write" being up-to-date if they target the same location.
This lack of guaranteed ordering can lead to correctness issues if there is
revisiting in the pipeline.

```{seealso} Where to go deeper
* {ref}`Working with Refs <pallas_refs>`: Pallas `Ref` views (`.at`, `reshape`, `transpose`, `bitcast`), memory spaces, and `scratch_types`.
* {ref}`Stateful computations in JAX <jax-101-refs>`: Creating, mutating, and freezing `Ref`s (`jax.new_ref`, `jax.empty_ref`, `jax.freeze`).
* {ref}`Pallas Async Operations <pallas_async>`: Buffer lifetimes, aliasing rules, and defensive copies across asynchronous kernels.
```
