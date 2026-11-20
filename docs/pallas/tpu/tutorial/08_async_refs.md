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

(pallas-tpu-tutorial-08-async-refs)=
# Chapter 8: Asynchronous Operations and the Ref API

<!--* freshness: { reviewed: '2026-11-11' } *-->

In previous chapters, we learned how to write Pallas kernels that perform
asynchronous memory transfers (both local and remote) *inside* the kernel. While
this approach is powerful, it requires us to embed all computation that overlaps
with communication directly within the same kernel.

In this chapter, we will explore **decomposed** asynchronous operations. The
goal is to expose APIs in Pallas that allow us to explicitly overlap computation
and communication *across multiple kernels*. This enables a more modular
programming style where we can start a communication operation in one kernel,
perform arbitrary JAX computation in between, and then wait for the
communication to complete in a separate kernel.

## Synchronous Copy Kernel

Historically, when we write a Pallas kernel that involves asynchronous copying,
we combine the `start` and `wait` operations in the same kernel. While we can
overlap computation with communication *within* the kernel itself, the kernel as
a whole behaves synchronously with respect to the rest of the JAX program.

As we saw in the previous chapter, we can use the
{func}`@pl.kernel <jax.experimental.pallas.kernel>` API and specify semaphores
as `scratch_types`. Here is an example of a fused kernel that performs an
asynchronous copy and waits for it to complete before exiting:

```{code-cell}
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

tc_mesh = pltpu.TensorCoreMesh(axis_name="tc", num_cores=1)


@pl.kernel(
    mesh=tc_mesh,
    out_type=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    scratch_types=[pltpu.SemaphoreType.DMA],
)
def fused_async_copy_kernel(x_ref, y_ref, sem):
  # Construct the async copy descriptor
  copy_op = pltpu.make_async_copy(x_ref, y_ref, sem)

  # Start the asynchronous copy from x_ref to y_ref
  copy_op.start()

  # (Optional) Overlap with other localized computation here

  # Wait for the copy to complete
  copy_op.wait()


# Invoking the kernel in JAX
def f(x):
  return fused_async_copy_kernel(x)
```

## Decomposed Asynchronous Kernels

To overlap these memory transfers with other operations across the entire JAX
program, we can decompose the start and wait steps into separate kernels using
the `pl.kernel` API.

### The Challenge: Extending State Lifetimes

Decomposing an asynchronous operation introduces a new challenge: internal state
that used to live entirely within a single kernel must now survive across kernel
boundaries. Specifically, the source memory `Ref`, the destination `Ref`, and
the semaphore allocated in the `start` kernel must be preserved and passed along
to the `done` kernel.

### The Mechanism: `out_type` vs `scratch_types`

In the fused synchronous kernel, we allocated semaphores by specifying them in
the `scratch_types` argument of `@pl.kernel`. However, resources allocated via
`scratch_types` have a lifetime that is strictly scoped to the execution of that
kernel's body. Once the kernel finishes, they are deallocated or reused.

To extend the lifetime of these resources, we specify them in the `out_type`
argument instead. The `out_type` parameter supports internal TPU types like
`pltpu.VMEM` and `pltpu.SemaphoreType`. By defining them as output types, the
`start` kernel returns these buffers back to the host JAX program, allowing us
to wrap them in `jax.new_ref(..., memory_space=...)` and pass them into
subsequent kernels.

```{note}
Notice that in the code below, we pass `Ref` objects (created via
`jax.new_ref()`) directly into `pl.kernel` as inputs. Because `pl.kernel`
returns plain JAX values on exit, wrapping non-HBM outputs such as DMA
semaphores or VMEM buffers with `jax.new_ref(..., memory_space=...)` preserves
their TPU memory space for downstream kernels. Finally, we call `jax.freeze()`
on the modified output `Ref` to convert it back into an immutable JAX array.
```

```{code-cell}
# 1. Decomposed start kernel
def copy_start_kernel(x_ref, o_ref, sem):
  # Start the asynchronous copy
  pltpu.make_async_copy(x_ref, o_ref, sem).start()


# 2. Decomposed done kernel
def copy_done_kernel(x_ref, o_ref, sem):
  # Reconstruct the async copy operation and wait on the passed semaphore
  pltpu.make_async_copy(x_ref, o_ref, sem).wait()


@jax.jit
def overlapped_computation(x):
  tc_mesh = pltpu.TensorCoreMesh(axis_name="tc", num_cores=1)

  # Create a mutable reference for our input
  x_ref = jax.new_ref(x)

  # Call the start kernel, specifying out_type to allocate our
  # destination buffer and synchronization semaphore.
  out, sem = pl.kernel(
      copy_start_kernel,
      mesh=tc_mesh,
      out_type=(jax.typeof(x), pltpu.SemaphoreType.DMA(())),
  )(x_ref)

  # Wrap the returned buffers into mutable references
  out_ref = jax.new_ref(out)
  sem_ref = jax.new_ref(sem, memory_space=pltpu.SEMAPHORE)

  # (Arbitrary JAX operations can run here overlapped with the DMA!)

  # Call the done kernel, passing our tracked references
  pl.kernel(copy_done_kernel, mesh=tc_mesh)(x_ref, out_ref, sem_ref)

  # Freeze the mutated output reference back into a standard JAX Array
  return jax.freeze(out_ref)
```

## Prefetching into VMEM

Because `out_type` supports internal TPU memory spaces, we can adapt our
decomposed kernels to prefetch data directly into VMEM. This is useful when you
want to schedule a memory transfer from HBM into VMEM while overlapping with
other operations.

To prefetch into VMEM, we update `out_type` to use `pltpu.VMEM.like(x)` and pass
`memory_space=pltpu.VMEM` when wrapping `vmem_out` in `jax.new_ref`:

```{code-cell}
@jax.jit
def prefetch_vmem_computation(x):
  tc_mesh = pltpu.TensorCoreMesh(axis_name="tc", num_cores=1)
  x_ref = jax.new_ref(x)

  # Allocate the destination directly in pltpu.VMEM via out_type
  vmem_out, sem = pl.kernel(
      copy_start_kernel,
      mesh=tc_mesh,
      out_type=(pltpu.VMEM.like(x), pltpu.SemaphoreType.DMA(())),
  )(x_ref)

  vmem_ref = jax.new_ref(vmem_out, memory_space=pltpu.VMEM)
  sem_ref = jax.new_ref(sem, memory_space=pltpu.SEMAPHORE)

  # Wait for the prefetch into VMEM to finish
  pl.kernel(copy_done_kernel, mesh=tc_mesh)(x_ref, vmem_ref, sem_ref)

  @pl.kernel(mesh=tc_mesh)
  def add_one():
    # Subsequent compute kernels can operate directly on Refs in VMEM.
    vmem_ref[...] += 1

  add_one()

  return jax.freeze(vmem_ref)
```

```{seealso} Where to go deeper
* {ref}`Pallas Async Operations <pallas_async>`: Design note on XLA async scheduling, state lifetime extension, and loop futures.
* {ref}`Working with Refs <pallas_refs>`: Managing Pallas `Ref` memory spaces across kernel invocations.
* {ref}`Stateful computations in JAX <jax-101-refs>` and {ref}`JIT and Refs <jax-201-jit-refs>`: Core `jax.new_ref` and `jax.freeze` semantics.
```
