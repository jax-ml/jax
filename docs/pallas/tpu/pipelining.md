---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  name: python3
---

(pallas_tpu_pipelining)=

+++ {"id": "teoJ_fUwlu0l"}

# Pipelining

<!--* freshness: { reviewed: '2024-04-08' } *-->

+++ {"id": "gAJDZh1gBh-h"}

In this guide we'll cover how memory spaces in TPU work and how to write
pipelines in Pallas that overlap memory I/O with compute.

```{code-cell}
:id: ejAVO6ikUUuF

#@title Imports

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
```

+++ {"id": "TWKESTKAlyjT"}

(tpu_and_its_memory_spaces)=

## TPU and its memory spaces

+++

A TPU and its TensorCore consist of memory spaces (where arrays can reside),
registers (which temporarily store scalar and array values) and compute units
(that do computation with values in registers).
Below is a diagram of a TPU in which `x` and `y` are arrays that live in
high-bandwidth memory (HBM):

![TPU Memory Space Cartoon.png](data:image/png;base64,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FhER0UcJWkRERB8laBEREX2UoEVERPRRghYREdFHCVpEREQfJWgRERF9lKBFRET0UYIWERHRRwlaREREHyVoERERfZSgRURE9FGCFhER0UcJWkRERB8laBEREX2UoEVERPRRghYREdFHCVpEREQfJWgRERF9lKBFRET0UYIWERHRRwlaREREHyVoERERfZSgRURE9FGCFhER0UcJWkRERB8laBEREX2UoEVERPRRghYREdFHCVpEREQfJWgRERF9lKBFRET0UYIWERHRRwlaREREHyVoERERfZSgRURE9FGCFhER0UcJWkRERB8laBEREX2UoEVERPRRghYREdFHCVpEREQfJWgRERF9lKBFRET0UYIWERHRRwlaREREHyVoERERfZSgRURE9FGCFhER0UcJWkRERB8laBEREX2UoEVERPRRghYREdFHCVpEREQfJWgRERF9lKBFRET0UYIWERHRRwlaREREHyVoERERfZSgRURE9FGCFhER0UcJWkRERB8laBEREX2UoEVERPRRghaxjbp06VI1fPhw13zb7du3q6FDh6qPP/64bt78+fOteU7Hjh2rZs2apfbt2+dax5QpU6wyq1atci1zumTJEqvca6+95lqGiPElaBHbqFOnTlVFRUWu+barV69WsVhMHTp0qG7euHHjrNc4g3bAgAGqU6dOVtlFixbVW0fPnj2t+fLTXL/txYsXVWlpqVUuXvAjYsMStIht1AcN2s6dO7vKSlhOnDhRFRQUqE8//bRuvgRshw4drPXs37/f9Tpx48aN1nIpR9AiepegRWyjNmfQirt377bKb926tW6eBK10LUuIzpgxw/UaUZZXVFSowYMHE7SIDyBBi9hGbe6g3bRpk1X+gw8+qJsnQTthwgRVVVVldQ9Ly9f5mjNnzqjCwkL15ptvWl3QBC2id1ssaAOBgLpy5YrrDSBiwz5o0JaXl6tPPvmkntL9K6FqBqXMk9fs2rXLWpeUcy5fvny5ys/PV8ePH1eVlZWu1yNifCX3dNDeNTPRFyKRyL3z58+73gQiNqwErYTf/TSD1lxuK6H62Wef1fsbdtBKZSDdw/K7c7m0YocMGWL9TtAieldyTzc0vzYz0Rf0iX7n5MmTrjeBiA0rQSuDl+RnQ8o104aCVkYYv/fee/WU23O6dOmi+vfvb534dnk7aOX3BQsWWC1oO4yPHj1qrV9azjJN0CJ698SJE/d0i/YzMxN9obi4+IacuOabQMSGfdCu48au0X700UdWcEug2vOcQSvdw9JNXF1dbU3PmzdP6fO2LpgJWkTvHjx48KoO2kNmJvpCRUXFORmMYb4JRGzY5g5aUYJ15MiR9aad3cXSTTxw4ECrK1lawPJAC3sZQYvo3VWrVh3OyMh4x8xEX+jfv/+2xYsXu94EIjZscwethGdZWZkaP3583TwzaO11yiAo+blz5866ZQQtonf1P6u7AoHACDMTfWHw4MFTJ0+e7HoTiNiwzRm0ly9ftm7RkfJr1qypm28GrXQTyz21cktP165d690pQNAiereiomJn+/bto2Ym+oI+oX9ZXl5+z3wTiNiwDxq0ch1WRhA7tR/BKPfLOsPTDFpRykjZN954o958ghbRu6FQ6Lxu0f5HMxN9QZ+kP4xGo3dOnTrleiOI6FaCdNq0aa75tnv37rXC+MSJE3XzpMvXHJ0szpkzp143sK0MeJLXOOfJAy3kNTI4yjlfBlG99dZbrnUgYsMePnz4im7NnjTz0Fe6dOly1L5VABERMZnV/7BKt/FUMwt9pXv37p3l3j/zzSAiIiab0Wj044yMjEfMLPSVtLS0P9F/+LY8P9V8Q4iIiMni3r17L+jW7Cc6+n5gZqHvlJeXv++8YR4RETHZ7NGjx24dtD3MDGwRsrOz/7GgoOD2559/7npjiIiIie7Ro0e/DAQCl4LB4L8xM7DFKCkp2bds2TLXm0NEREx0e/XqtT8jI6O3mX0tSiQS+XleXt53Z8+edb1BRETERHXbtm1y3+zp7OzsPzWzr8UpLi5+s6qqigdYICJiUiiXRKPR6Bndmn3GzLxWQSf+X+jE/6ahm+gRERETzREjRhxr3779AjPvWpVwOPxkfn7+zdOnT7veMCIiYqK4YsWKc7oB+UlWVtafm1nX6sRisQn9+vW7eenSJdcbR0REbOvu27fvWjAYvKL9qZlxbYK0tLTfzc3NXT9q1KhbzoedIyIitnUPHjz4XTgcvqpD9jdmvrUpdNj+kQ7bD6qqqr4jbBERMRE8dOjQnUgkciUjIyPdzLU2ifRrZ2dn79ctW7qRERGxTbtly5abuiX7RcKErI20bHXYru3bt+91BkghImJbdNGiRd/okJWvwPtXM8cSArlmG41GR8disevc+oOIiG1F+T71Pn36XNEhuzcQCPx7M78SjvT09MezsrK+nDx58i2eIIWIiK3l5cuX1bJly25HIpFvgsHg4EceeeT3zMxKWGofajFTe12ejcwXESAiYkv6/vvvq06dOn2tG37b2uztO81BRkbGP+Xm5m7Iycm5sWDBgnt8ny0iIvrlhQsXVHV1terQocM3uqEnT3t6zsylpEX+m8jLy1sSDodvDhw48Prq1asVoYuIiA+rXKJct26dPEZRRhPf1FmzIRAI/NLMoZShXbt2P9I7IFxQULBZdkhJScm1yZMn3128eLHatGmTfBegOn78uLp48aJrZyIiYmoqmSDZcOzYMSsrlixZol555RVVWlp6PRQK3dKZsjsSiRSmpaX9pZk7qc4PdOj+XAduuW7mv15YWLg3JyfnsvaLzMzMO7rJrxARESUTJBtyc3OvFBcX79eZMT8rK6urzpB/ljtezHABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABIav5/PEJoH3//LlQAAAAASUVORK5CYII=)

Let's talk about the components of this diagram in more detail:

* **Memory spaces**: A TPU has high-bandwidth memory (HBM) which is what we
  often think of as "device memory".
  There is also vector memory (VMEM),
  a cache meant for storing vector and array values, and scalar memory (SMEM),
  a cache designed to store scalar values.
* **Registers**: A TensorCore has two main types of registers: vector
  registers (VREGs) store array values, and scalar registers (SREGs) store
  scalar values.
  Values can be loaded into memory from their respective caches (VMEM for
  VREGs and SMEM for SREGs).
* **Compute units**: A TensorCore has a scalar unit, vector unit (VPU) and
  matrix unit (MXU) that can do numerical computation.
  Compute units operate on values that live in SREGs and VREGs and output
  values into those registers as well.

In order to do a vectorized computation on our values `x` and `y` that live
in HBM, we need to:

1. Copy the values `x` and `y` into VMEM.
2. Load the values from VMEM into VREGs.
3. Execute the computation using the VPU or MXU, storing the output in VREGs.
4. Store the values in the output VREGs into VMEM.
5. Copy the output values in VMEM back to HBM.

+++ {"id": "TzctMbNsn3vc"}

Let's implement a Pallas function that does just that!

```{code-cell}
:id: 2IXQxNWrKJyb
:outputId: d62eb493-5f92-4496-f113-d3cd24cb0b9f

def add_matrices_kernel(x_vmem_ref, y_vmem_ref, z_vmem_ref):
  # Load x and y from VMEM into VREGs
  x_vregs = x_vmem_ref[:, :]
  y_vregs = y_vmem_ref[:, :]
  # Execute a vectorized add
  z_vregs = x_vregs + y_vregs
  # Store the output values in VREGs back into VMEM
  z_vmem_ref[:, :] = z_vregs


def add_matrices(x: jax.Array, y: jax.Array) -> jax.Array:
  # pallas_call will first allocate scratch buffers for `x` and `y` in VMEM.
  # It will then copy `x` and `y` from HBM into VMEM.
  z = pl.pallas_call(
      add_matrices_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)
  # pallas_call will also copy the output from VMEM back into HBM.
  return z


x, y = jnp.ones((512, 512)), jnp.ones((512, 512))
add_matrices(x, y)
```

+++ {"id": "HMENNLy8okCL"}

We've written two functions: `add_matrices_kernel` and `add_matrices`.

`add_matrices_kernel` operates using `Ref`s that live in VMEM.
Loading from a VMEM `Ref` produces a value that lives in VREGs.
Values in VREGs behave like `jax.Array`s in that we can use `jnp` and
`jax.lax` operations on them to produce new values that live in VREGs.
When we produce the values we'd like to return, we store them in the output
VMEM `Ref`.

The `add_matrices` function acts on `jax.Array`s and returns a `jax.Array`.
Inside it, we pass `x` and `y` into `pallas_call`.
`pallas_call` is responsible for copying `x` and `y` into VMEM and for
allocating the VMEM buffers that the kernel operates on (including allocating
`z_vmem_ref`, the output VMEM buffer).
After the kernel function is finished running, `pallas_call` will also copy
the value in `z_vmem_ref` to HBM, resulting in an output `jax.Array`.

+++ {"id": "5kWr-1tKpYro"}

## Constraints of using VMEM/SMEM

Pallas exposes access to lower level memory spaces like VMEM and SMEM but
writing kernels utilizing them adds some considerations.

1. Memory capacity. VMEM and SMEM are *small*! VMEM on v4 TPUs is only 16MiB
  and SMEM ranges in the tens to hundreds of KiB.
  If our arrays are too big, we won't even be able to fit them into VMEM at all.
  For reference, a `f32[2048, 2048]` array is 16MiB, so our above kernel won't
  scale beyond moderately sized arrays.

2. Memory bandwidth. Copying to/from HBM and VMEM takes a long time, at least
  compared to most compute instructions.
  The `add_matrices` function above will likely spend more time copying
  between HBM and VMEM than actually performing the addition itself.

With these two constraints in mind, we'll have to rethink our strategy for
getting performance out of our TPUs.

+++ {"id": "_NTqvlbetB3P"}

## Primer: Pipelining

Pipelining our computation offers a way of dealing with both the memory
capacity and bandwidth constraints in one fell swoop.
What do we mean by pipelining?

The goal is: *in parallel* copy to/from HBM and VMEM *while* utilizing our
compute units.
Naively this is difficult because in our program above we copy *all* of `x`
and `y` before we start doing any compute with them, creating a dependence
between the copy and the compute.

However, if we can chunk up our computation into several subcomputations
(e.g. when we add two matrices, we can express that as addition of "blocks"
of the original matrices together), we can now overlap the copies of one of
those subcomputations with the compute of the other. Let's walk through a
simple example:

Let's say we split our arrays `x` and `y` into `x1, x2` and `y1, y2` (for
example, split along the leading axis, resulting in two `(256, 512)` arrays
for each input.
We can now execute the following pipelined computation.

1. Copy `x1` and `y1` into VMEM.
1. Start copying `x2` and `y2` into VMEM
2. Load `x1, y1` from VMEM into VREGs.
3. Execute the `z1 = x1 + y1` using the compute units.
4. Store `z1` into VMEM.
5. Start copying `z1` from VMEM back into HBM.
6. Wait until `x2, y2` have been copied into VMEM.
7. Load `x2, y2` from VMEM into VREGs.
8. Execute the `z2 = x2 + y2` using the compute units.
9. Store `z2` into VMEM.
10. Wait until `z1` is copied into HBM.
10. Start copying `z2` from VMEM back into HBM.
10. Wait until `z2` is copied into HBM.

Any time we are doing compute here, we are asynchronously copying something.
This means that some of the time spent copying is not wasted.

The two most important numbers for determining how efficient a pipelined
computation are a) how many floating point operations (FLOPs) we need to
execute and b) how many bytes we need to copy to execute that computation.
The ratio of these two (FLOPs/memory usage) is called the
*arithmetic intensity* of an operation and determines if our pipeline will
be compute bound or memory bound.

+++ {"id": "gutx7y8uvZKH"}

## Pipelining in Pallas

+++ {"id": "U-dPTjlBverB"}

How do we implement a pipeline like the one above in Pallas?
It seems like a complex sequence of asynchronous data operations and
executing kernels that would be a pain to implement manually.
Fear not! Pallas offers an API for expressing pipelines without too much
boilerplate, namely through `grid`s and `BlockSpec`s.

See how in the above pipelined example, we are executing the same logic
multiple times: steps 3-5 and 8-10 both execute the same operations,
only on different inputs.
The {func}`jax.experimental.pallas.pallas_call` provides a way to
execute a kernel multiple times, by using the `grid` argument.
See {ref}`pallas_grid`.

We also use {class}`jax.experimental.pallas.BlockSpec` to specify
how to construct the input of each kernel invocation.
See {ref}`pallas_blockspec`.

In the pipelining example above, we had `(512, 512)`-shaped arrays and
split them along the leading dimension into two `(256, 512)`-shaped arrays.
In this pipeline, our `BlockSpec.block_shape` would be `(256, 512)`.
On the 1st iteration we'd
like to select `x1` and on the second iteration we'd like to use `x2`.
This can be expressed with the following `index_map`:

```python
def x_index_map(i):
  return (i, 0)
```

We'd then construct the `BlockSpec`:
```python
block_spec = pl.BlockSpec((256, 512), x_index_map)
```

The `BlockSpec`s for `y` and `z` will be the same as the one for `x`.

+++ {"id": "noybOKghzjwG"}

### Putting it together

We provide these arguments to `pallas_call` via `grid`, `in_specs` and
`out_specs` (`in_specs` corresponds to the tuple of positional arguments,
and `out_specs` corresponds to the output).

```{code-cell}
:id: ehKAYAwIojfv
:outputId: 504bab29-83f3-4e1f-8664-1860ad15b6de

def add_matrices_pipelined(x: jax.Array, y: jax.Array) -> jax.Array:
  block_spec = pl.BlockSpec((256, 512), lambda i: (i, 0))
  return pl.pallas_call(
      add_matrices_kernel,
      out_shape=x,
      in_specs=[block_spec, block_spec],
      out_specs=block_spec,
      grid=(2,)
  )(x, y)

add_matrices_pipelined(x, y)
```

+++ {"id": "rkytgIZYzz4t"}

We've only added a little bit of code to our original function to add
automatic pipelining but the `BlockSpec`s and `grid` do a lot of heavy
lifting!

How does it work? Well, the `BlockSpec`s provide enough information to start
*prefetching* blocks of our input from HBM into VMEM.
For example, if we are starting iteration `i` of our `grid`, we can pass
`i + 1` into the `index_map` functions to obtain the blocks needed for the
next iteration. We can then start an asynchronous copy for those blocks.
Similarly for outputs, we can wait for the outputs of the previous iteration
to be copied before starting the copy for the current iteration's outputs.

+++ {"id": "7Xtz9oMs0ZRL"}

### Parameterizing a pipeline

+++ {"id": "esY4GcIB0bqQ"}

It's common to parameterize the block shapes in our kernel. Block sizes are
perhaps the most important parameter to tune when optimizing the performance
of Pallas kernels! They give us control over the pipeline (for example,
picking smaller blocks adds more iterations to our pipelined loop where each
iteration has less work to do).

Furthermore, we could also carve up the inputs and outputs along the 2nd
dimension (we are only splitting along the first right now). Let's write a
more general kernel that handles both of these features.

```{code-cell}
:id: VartelFd0YfY

def add_matrices_pipelined_2d(
    x: jax.Array, y: jax.Array, *, bm: int = 256, bn: int = 256
) -> jax.Array:
  m, n = x.shape
  block_spec = pl.BlockSpec((bm, bn), lambda i, j: (i, j))
  return pl.pallas_call(
      add_matrices_kernel,
      out_shape=x,
      in_specs=[block_spec, block_spec],
      out_specs=block_spec,
      grid=(m // bm, n // bn),
  )(x, y)

np.testing.assert_array_equal(
    add_matrices_pipelined_2d(x, y, bm=256, bn=256), x + y
)
np.testing.assert_array_equal(
    add_matrices_pipelined_2d(x, y, bm=128, bn=128), x + y
)
np.testing.assert_array_equal(
    add_matrices_pipelined_2d(x, y, bm=512, bn=512), x + y
)
```

+++ {"id": "KrfeYwaW1QA-"}

## Handling reductions

+++ {"id": "P3SqEKDe3Mar"}

How would you implement something like `jnp.sum` using `pallas_call`?
Specifically, we'd like to pipeline across the reduction dimension.

Take the example of reducing a `(8, 512, 512)`-shaped array to a
`(512, 512)`-shaped one.

```{code-cell}
:id: JoT-ZKEk1R7l
:outputId: fd842223-98a5-4e5c-87fc-5dadc94da4fa

x = jnp.ones((8, 512, 512))
jnp.sum(x, axis=0)
```

+++ {"id": "5O3ByvuT3iyC"}

To do this using `pallas_call`, we could use a grid of size `(8,)` and in
each iteration `i` load `x[i]` into VMEM.
Then we could add `x[i]` to an output VMEM buffer. Let's implement this
naively first.

```{code-cell}
:id: hqvv_WRQ3bvP
:outputId: 200648d2-3f4d-4d1a-b95a-d2c1352cd7b8

# Warning: this implementation is incorrect!

def naive_sum_kernel(x_ref, o_ref):
  o_ref[...] += x_ref[...]

def naive_sum(x: jax.Array) -> jax.Array:
  grid, *out_shape = x.shape
  return pl.pallas_call(
      naive_sum_kernel,
      grid=grid,
      # None in `block_shape` means we pick a size of 1 and squeeze it away
      in_specs=[pl.BlockSpec((None, *out_shape), lambda i: (i, 0, 0))],
      out_specs=pl.BlockSpec(out_shape, lambda i: (0, 0)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
  )(x)
naive_sum(x)
```

+++ {"id": "Kv9qJYJY4jbK"}

Notice how we've set up the `BlockSpec`s: we're loading the entirety of
the `(512, 512)` dimension into VMEM (no pipelining there) but selecting
the `i`-th dimension of `x` each iteration in the `index_map`.
We are using a `None` for that dimension in the block shape, which indicates
that we are selecting a singleton dimension from `x` that we would like
to squeeze away in the kernel.
Therefore, `x_ref` is `(512, 512)`-shaped in VMEM as well.

`out_spec` uses `lambda i: (0, 0)` as its `index_map`, indicating that
`o_ref` is unchanged over the course of the pipeline.
This means that we can update its value each iteration by reading from and
writing to it. Or can it?
Actually there is one catch: *`o_ref` is initially garbage*, meaning we'll
be accumulating into garbage.
This will result in the overall function outputting the incorrect value!

Therefore, **whenever we do a reduction in a kernel, we need to make sure
to initialize the `Ref` that is storing the reduced value**.
We can accomplish this by conditionally writing a value to `out_ref`
when we're on iteration 0.
We can do this with the helper function `pl.when`, a convenience wrapper
around `jax.lax.cond`, and `pl.program_id`,
which queries which iteration in a grid axis we are in.

```{code-cell}
:id: JXN2RthX5cSw
:outputId: 195df19b-a889-479b-95b6-1fb7281f1518

def sum_kernel(x_ref, o_ref):
  @pl.when(pl.program_id(axis=0) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)

  o_ref[...] += x_ref[...]

def sum(x: jax.Array) -> jax.Array:
  grid, *out_shape = x.shape
  return pl.pallas_call(
      sum_kernel,
      grid=grid,
      # None in `block_shape` means we pick a size of 1 and squeeze it away
      in_specs=[pl.BlockSpec((None, *out_shape), lambda i: (i, 0, 0))],
      out_specs=pl.BlockSpec(out_shape, lambda i: (0, 0)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype)
  )(x)

sum(x)
```

+++ {"id": "2828qXBI5ksZ"}

This `sum` function now outputs the correct values!

One last thing to note about reductions in Pallas are that **they must be
done in the minormost (rightmost) dimensions of our grid** (our grid is
1-dimensional in the above example so we are reducing over its minormost
dimension). This is because the pipeline that Pallas generates using
the `BlockSpec`s, `grid` and kernel function *does not read outputs back
from HBM*.
Once you've written an output value back to HBM you cannot revisit it.
Therefore, you cannot do a reduction across a grid dimension that has any
revisiting and therefore all reductions need to happen in the rightmost
dimensions.

+++ {"id": "KvPFez9N8cKJ"}

(pallas_tpu_megacore)=

## TPUs in Megacore configuration

+++ {"id": "0f4HAVzQ8n71"}

Some TPU chips have two TensorCores but appear as one device to JAX users.
This is called "megacore".
The separate TensorCores have their own separate VMEM, VREGs, SMEM, SREGs
and compute units but *share HBM*.

![TPU Memory Space Cartoon (Megacore).png](data:image/png;base64,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)

Conceptually, TPUs in Megacore behave like very simple GPUs, i.e. they have
only two threads.
How do we modify our kernels to utilize both TensorCores simultaneously?

The basic idea is that if we have embarrassingly parallel dimensions in our
computation, we can split up those dimensions across the TensorCores.
We can indicate which dimensions are parallelizable by providing an
annotation to `pallas_call` called `dimension_semantics`.

```{code-cell}
:id: nQNa8RaQ-TR1
:outputId: 385ed87c-d95c-466c-af77-df3845c979f2

def add_matrices_pipelined_megacore(x: jax.Array, y: jax.Array) -> jax.Array:
  block_spec = pl.BlockSpec((256, 512), lambda i: (i, 0))
  return pl.pallas_call(
      add_matrices_kernel,
      out_shape=x,
      in_specs=[block_spec, block_spec],
      out_specs=block_spec,
      grid=(2,),
      compiler_params=pltpu.TPUCompilerParams(dimension_semantics=("parallel",))
  )(x, y)

x, y = jnp.ones((512, 512)), jnp.ones((512, 512))
add_matrices_pipelined_megacore(x, y)
```

+++ {"id": "xG51AiUC-8cl"}

`dimension_semantics` should be a tuple of same length as `grid` where each
entry is either `"parallel"` or `"arbitrary"`. `"parallel"` indicates to Pallas that the iterations of the for loop corresponding to that dimension can be executed independently without affecting the correctness of the program. `"arbitrary"` indicates to Pallas that there can be no assumptions made about this grid dimension and it therefore cannot be parallelized.

By specifying `dimension_semantics`, we now execute the kernel
simultaneously on each TensorCore. Pallas will handle splitting up the grid
automatically.

> Note that Megacore is only currently available on TPU `v4` and TPU `v5p`. Supplying `dimension_semantics` annotations is a no-op on other platforms, but *not* specifying it will result in only one TensorCore being used (even if there are more than one available).

+++ {"id": "1ZJ2rV5W8FAe"}

## Conclusion

In this guide we covered how to express TPU pipelines using `pallas_call`,
`grid` and `BlockSpec`s. We covered how to express nested loops via a
multi-dimensional grid and how to handle reductions by initialize our
accumulators at the beginning of the reduction.
We also learned how to handle Megacore by adding annotations to the kernel.

Exercises left to the reader:
* Try implementing a `sum` kernel that pipelines the other dimensions as well
* Add megacore support to the `add` kernel and the `sum` kernel as well.
