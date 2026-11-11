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

(pallas-tpu-tutorial-04-matmul)=
# Chapter 4: Matrix Multiplication

<!--* freshness: { reviewed: '2026-11-11' } *-->

Building on the pipelining concepts introduced in the previous chapter, we will
now look at how to implement matrix multiplication using Pallas. We'll also go
over how to think about matmul performance on TPU and how to template a matmul
kernel to fuse in operations.

```{code-cell}
from collections.abc import Callable
from functools import partial
import jax
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
```

## Background

Matrix multiplication is a fundamental linear algebra operation at the heart of
modern deep learning and language modeling. We'd like to make matmuls as speedy
as possible using specialized accelerators like TPUs, which have specialized
units for fast matrix multiplication.

To effectively utilize TPUs for matrix multiplication, we'll need to cover a few
background concepts: block matrix multiplication, tiling and pipelining.

### Block Matrix Multiplication

Let's say we want to implement `matmul(x, y)` which generically multiplies an
`(m, k)` array with a `(k, n)` array, but with a twist. We're only allowed to
use the primitive `matmul_small` which multiplies small matrices (say `m, k, n <=
256`). How could we do it?

A nice property of matrix multiplication is that each block of the output can be
expressed as the sum of several smaller matrix multiplications of row blocks and
column blocks of the inputs. Formally, if we have input arrays $x \in
\mathbb{R}^{m \times k}$ and $y \in \mathbb{R}^{k \times n}$ and output $z \in
\mathbb{R}^{m \times n}$, we decompose them into blocks along the dimensions of
size $b_m, b_k, b_n$.

For example, $x$ would be decomposed as:

$$
\begin{bmatrix}
x_{0, 0} & \cdots & x_{0, i_k} \\
x_{1, 0} & \cdots & x_{1, i_k} \\
\vdots & \ddots & \vdots \\
x_{i_m, 0} & \cdots & x_{i_m, i_k} \\
\end{bmatrix}
$$

where $x_{ik} \in \mathbb{R}^{b_m \times b_k}$. (We can similarly decompose $y$
and $z$.)

For a particular output block $z_{ij}$, we can compute it as

$$
z_{ij} = \sum_k x_{ik} y_{kj}
$$

Therefore, each output block $z_{ij}$ is the sum of several smaller block matrix
multiplications $x_{ik} y_{kj}$. Here's how we'd implement this algorithm in
NumPy:

```{code-cell}
def matmul_small(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  m, k, n = x.shape[0], x.shape[1], y.shape[1]
  assert x.shape[1] == y.shape[0]
  assert m <= 256
  assert k <= 256
  assert n <= 256
  return np.matmul(x, y)


def block_matmul(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bm: int = 256,
    bk: int = 256,
    bn: int = 256,
) -> np.ndarray:
  m, k = x.shape
  _, n = y.shape

  z = np.zeros((m, n), dtype=x.dtype)
  for m_i in range(m // bm):
    for n_i in range(n // bn):
      for k_i in range(k // bk):
        m_slice = slice(m_i * bm, (m_i + 1) * bm)
        k_slice = slice(k_i * bk, (k_i + 1) * bk)
        n_slice = slice(n_i * bn, (n_i + 1) * bn)
        x_block = x[m_slice, k_slice]
        y_block = y[k_slice, n_slice]
        z[m_slice, n_slice] += matmul_small(x_block, y_block)
  return z
```

Our `block_matmul` function should now work on inputs larger than 256 (assuming
that our input dimensions are evenly divisible by 256).

```{code-cell}
m, k, n = 4096, 4096, 4096
x = np.random.uniform(size=(m, k)).astype(np.float32)
y = np.random.uniform(size=(k, n)).astype(np.float32)
np.testing.assert_allclose(x @ y, block_matmul(x, y), atol=1e-5, rtol=1e-5)
```

`block_matmul` decomposes a matrix multiplication into many smaller ones by
observing that each output chunk of size `(bm, bn)` can be computed by
accumulating several `(bm, bk) x (bk, bn)` size matrix multiplications.

TPUs do matmuls just like this! They natively support small matrix
multiplication akin to `matmul_small`, so to utilize this hardware when doing
bigger matrix multiplications, we will apply the `block_matmul` decomposition.

### Tiling and Pipelining

As covered in the previous chapter, to make sure our compute units are always
working and never stalled by memory transfers, we overlap the memory transfers
for the next iteration of a kernel with the current one.

In Pallas, we specify that via {ref}`BlockSpecs <pallas_blockspec>` and a
{ref}`grid <pallas_grid>`. Note that we already have a nested `for` loop in the
block matrix multiplication algorithm. We can specify that in Pallas via a
`grid`, and specify the slices in the block matrix multiplication via
`BlockSpec`s.

## Your first matrix multiplication kernel

Putting it all together, here's an implementation of a block matrix
multiplication kernel that pipelines the memory transfers with the compute using
{func}`pl.kernel <jax.experimental.pallas.kernel>` and `pltpu.emit_pipeline`. We
create a 3D grid corresponding to the 3-nested loop in the NumPy code. To
maximize performance on multi-core TPUs, we also parallelize the execution
across cores.

The last dimension of the grid corresponds to the contraction dimension of the
matrix multiply and is a reduction dimension, so we need to be sure to
initialize the accumulator. We use `scratch_types` in `pl.kernel` to allocate a
scratch buffer in VMEM for accumulation.

```{code-cell}
def matmul_kernel(
    x_hbm: jax.Ref,
    y_hbm: jax.Ref,
    z_hbm: jax.Ref,
    acc_vmem: jax.Ref,
    *,
    bm: int,
    bk: int,
    bn: int,
):
  m, k = x_hbm.shape
  _, n = y_hbm.shape

  def matmul_body(x_vmem: jax.Ref, y_vmem: jax.Ref, z_vmem: jax.Ref):
    k_index = pl.program_id(2)

    # Initialize accumulator on the first step
    @pl.when(k_index == 0)
    def _():
      acc_vmem[...] = jnp.zeros_like(acc_vmem)

    # Accumulate matrix multiplication
    acc_vmem[...] += x_vmem[...] @ y_vmem[...]

    # Write back to HBM on the last step
    @pl.when(k_index == pl.num_programs(2) - 1)
    def _():
      z_vmem[...] = acc_vmem[...].astype(z_vmem.dtype)

  pltpu.emit_pipeline(
      matmul_body,
      grid=(m // bm, n // bn, k // bk),
      in_specs=[
          pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
          pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
      ],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
      core_axis_name='core',
      dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
  )(x_hbm, y_hbm, z_hbm)


def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
) -> jax.Array:
  m, k = x.shape
  _, n = y.shape

  tc_mesh = pltpu.TensorCoreMesh(axis_name='core')

  return pl.kernel(
      partial(matmul_kernel, bm=bm, bk=bk, bn=bn),
      out_type=jax.ShapeDtypeStruct((m, n), x.dtype),
      mesh=tc_mesh,
      scratch_types=[pltpu.VMEM((bm, bn), jnp.float32)],
  )(x, y)


m, k, n = 4096, 4096, 4096
k1, k2 = random.split(random.key(0), 2)
x = random.normal(k1, (m, k), dtype=jnp.float32)
y = random.normal(k2, (k, n), dtype=jnp.float32)
np.testing.assert_allclose(x @ y, matmul(x, y), atol=1e-2, rtol=1e-2)
```

## Matrix multiplication performance

Let's think about how to analyze matrix multiplication performance. When we
think about matmul performance, we typically care about two things: the total
number of floating point operations (FLOPs) and the amount of memory bandwidth
usage. As we saw earlier, in order to use the efficient compute units on TPUs,
we need to copy our inputs from HBM into VMEM, closer to the compute units. This
copying to and from HBM takes time and an efficient kernel hopefully spends most
of its time actually computing, as opposed to waiting for these transfers.
Memory bandwidth measures the rate of this data transfer.

The number of FLOPs in an `(m, k) x (k, n)` matrix multiplication is
(approximately) `2 * m * k * n`.

The minimum amount of memory bandwidth usage for a matrix multiply (assuming
`float32`) is the total size of the inputs (copying into VMEM) plus the size of
the output (copying into HBM). Thus the minimum bandwidth usage is `(m * k + k *
n + m * n) * 4 bytes/float32`. Memory usage can be greater if we re-read the
inputs multiple times, which is often the case.

One observation is that the number of matmul FLOPs is cubic in its inputs
whereas the minimum bandwidth usage is quadratic in its inputs. Intuitively,
this means that FLOPs grow faster than bandwidth usage, meaning that the bigger
our matmul is, the more compute we have relative to copying.

Now that we can calculate the total number of FLOPs and (minimum) memory
bandwidth usage of a matrix multiplication, let's see what a real TPU can
handle.

Suppose we are on a chip with 197 TFLOP/s of compute and 819 GB/s of memory
bandwidth. By looking at the ratio of these numbers (called the arithmetic
intensity), we can get a bound on how low this "FLOPs / memory bandwidth usage"
ratio can get before we become IO bound (about 240 FLOPs/byte).

This basic calculation tells us roughly how efficiently we'll be able to use our
MXUs. If our matmul op intensity is below what our chip is capable of, then our
computation will be *memory bound*, i.e. our compute units will be idling while
waiting for values to be transferred. If the matmul intensity is higher than
what the chip is capable of, then we will be *compute bound*.

Because matmul FLOPs are cubic in their input sizes and memory bandwidth usage
is quadratic, we expect that we will get compute bound as we get bigger and
bigger.

### `bfloat16` matrix multiplication

To make it easier for matrix multiplication to be compute bound on TPU, we could
also use a smaller dtype for our inputs and outputs. The previous example used
`float32` inputs and outputs, but TPUs also support the `bfloat16` data type for
matrix multiplication. On some chips, we will have the same FLOP/s (or higher)
while *halving our memory bandwidth usage*. This makes it much easier to be
compute bound for smaller matrices.

Let's add `bf16` support to our matrix multiplication kernel.

The native MXU `bf16` matmul routine takes two input `bf16` matrices and
accumulates them in `f32`. We trigger this routine by passing
`preferred_element_type=jnp.float32` into `jnp.dot`. We also need an accumulator
`Ref` in `f32`, which we downcast back to `bf16` before writing it back to HBM.
This way we don't lose precision, don't do extra casting, and still retain the
`bf16` memory bandwidth savings.

```{code-cell}
def matmul_bf16_kernel(
    x_hbm: jax.Ref,
    y_hbm: jax.Ref,
    z_hbm: jax.Ref,
    acc_vmem: jax.Ref,
    *,
    bm: int,
    bk: int,
    bn: int,
):
  m, k = x_hbm.shape
  _, n = y_hbm.shape

  def matmul_body(x_vmem: jax.Ref, y_vmem: jax.Ref, z_vmem: jax.Ref):
    k_index = pl.program_id(2)

    @pl.when(k_index == 0)
    def _():
      acc_vmem[...] = jnp.zeros_like(acc_vmem)

    # Use jnp.dot with preferred_element_type=jnp.float32 for bf16 -> f32
    acc_vmem[...] += jnp.dot(
        x_vmem[...], y_vmem[...], preferred_element_type=jnp.float32
    )

    @pl.when(k_index == pl.num_programs(2) - 1)
    def _():
      z_vmem[...] = acc_vmem[...].astype(z_vmem.dtype)

  pltpu.emit_pipeline(
      matmul_body,
      grid=(m // bm, n // bn, k // bk),
      in_specs=[
          pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
          pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
      ],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
      core_axis_name='core',
      dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
  )(x_hbm, y_hbm, z_hbm)


def matmul_bf16(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
) -> jax.Array:
  m, k = x.shape
  _, n = y.shape

  tc_mesh = pltpu.TensorCoreMesh(axis_name='core')

  return pl.kernel(
      partial(matmul_bf16_kernel, bm=bm, bk=bk, bn=bn),
      out_type=jax.ShapeDtypeStruct((m, n), x.dtype),
      mesh=tc_mesh,
      scratch_types=[pltpu.VMEM((bm, bn), jnp.float32)],
  )(x, y)
```

## Performance of pipelined kernels

Our above analysis about FLOPs vs memory usage applies at a coarse scale, i.e.
when we are looking at the size of the total matrix multiplication. However,
remember that in practice, we are pipelining the execution of a blocked matrix
multiplication, meaning we have a loop in which we are doing matrix
multiplication with smaller blocks.

This means that we actually care about the FLOPs vs memory bandwidth usage of
each individual instance of the kernel, not the global FLOPs vs memory bandwidth
usage.

Therefore, the block sizes `bm`, `bk`, `bn` are extremely important for
performance. Even if we have the largest matrices in the world, if we pick very
small `bm`, `bk`, and `bn`, we will be memory bound because each time we invoke
the kernel we will have too few FLOPs to hide the memory transfers happening in
the background.

The intuition should therefore be: to be compute bound, make the blocks as big
as possible! There are two main constraints:

1.  **VMEM usage:** The bigger our blocks, the more VMEM we use. With large
    enough blocks, we will run out.
2.  **Pipeline bubbles:** The larger our blocks are relative to the matrix size,
    the fewer loop iterations we will have in our pipeline. This will make the
    size of the bubbles at the beginning and end of the pipeline larger relative
    to the total pipeline and this overhead can be nontrivial.

## Templating the matrix multiplication

Now that we have a basic matrix multiplication kernel, we can try fusing
operations into it.

### Fused activation function

Fusing in an activation is very common. This makes sure we don't follow an
efficient, compute-bound matmul kernel with a slow memory-bound activation
kernel.

```{code-cell}
def matmul_fused_kernel(
    x_hbm: jax.Ref,
    y_hbm: jax.Ref,
    z_hbm: jax.Ref,
    acc_vmem: jax.Ref,
    *,
    bm: int,
    bk: int,
    bn: int,
    activation: Callable[[jax.Array], jax.Array],
):
  m, k = x_hbm.shape
  _, n = y_hbm.shape

  def matmul_body(x_vmem: jax.Ref, y_vmem: jax.Ref, z_vmem: jax.Ref):
    k_index = pl.program_id(2)

    @pl.when(k_index == 0)
    def _():
      acc_vmem[...] = jnp.zeros_like(acc_vmem)

    acc_vmem[...] += jnp.dot(
        x_vmem[...], y_vmem[...], preferred_element_type=jnp.float32
    )

    @pl.when(k_index == pl.num_programs(2) - 1)
    def _():
      # Apply activation before writing back to HBM
      z_vmem[...] = activation(acc_vmem[...]).astype(z_vmem.dtype)

  pltpu.emit_pipeline(
      matmul_body,
      grid=(m // bm, n // bn, k // bk),
      in_specs=[
          pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
          pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
      ],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
      core_axis_name='core',
      dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
  )(x_hbm, y_hbm, z_hbm)


def matmul_fused(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    activation: Callable[[jax.Array], jax.Array] = lambda x: x,
) -> jax.Array:
  m, k = x.shape
  _, n = y.shape

  tc_mesh = pltpu.TensorCoreMesh(axis_name='core')

  return pl.kernel(
      partial(
          matmul_fused_kernel, bm=bm, bk=bk, bn=bn, activation=activation
      ),
      out_type=jax.ShapeDtypeStruct((m, n), x.dtype),
      mesh=tc_mesh,
      scratch_types=[pltpu.VMEM((bm, bn), jnp.float32)],
  )(x, y)
```

## Conclusion

In this chapter, we covered how to write efficient matrix multiplications on TPU
using Pallas. We discussed blocked matrix multiplication and pipelining, how to
analyze the performance of a TPU matmul, and how to write an efficient `bf16`
matrix multiplication. We concluded with templating the matrix multiplication to
support a fused activation function.

```{seealso} Where to go deeper
* {doc}`Matrix Multiplication on TPU </pallas/tpu/matmul>`: Fused RHS transpose via `lax.dot_general`, systematic block-size tuning, and mixed-precision benchmarks.
* {doc}`Kernel Fusion with Pallas Fuser </pallas/tpu/fuser>`: Automatically fusing arbitrary JAX prologue/epilogue operations into matmul kernels.
* {doc}`Writing TPU kernels with Pallas </pallas/tpu/details>`: MXU register tiling (`8×128`), supported data types, and operation costs.
```
