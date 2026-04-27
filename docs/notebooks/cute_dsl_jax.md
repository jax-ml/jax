---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.20.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Writing High-Performance GPU Kernels with CuTe DSL and JAX

## Overview

JAX provides excellent built-in GPU support through XLA, but sometimes you need to go beyond what the compiler can generate automatically. Custom GPU kernels let you exploit hardware-specific features, fuse operations that XLA misses, or implement algorithms that don't map cleanly to standard library calls. CuTe DSL bridges this gap by letting you write CUDA kernels in Python and plug them directly into JAX programs.

**What you'll do:**

- Install CUTLASS 4.x and its CuTe DSL Python front-end
- Write a **Vector Add** kernel using `@cute.kernel` and launch it with `@cute.jit`
- Integrate CuTe DSL kernels into JAX programs via `cutlass.jax.cutlass_call`
- Implement **SAXPY** (`y = alpha * x + y`) with scalar kernel arguments
- Write **ReLU** and **Fused Bias+ReLU** activation kernels for deep learning
- Build a **tiled GEMM** using tensor core MMA instructions
- Shard CUTLASS kernels across multiple GPUs with `jax.shard_map`
- **Export and serialize** JAX functions containing CUTLASS kernels with **`jax.export`**

+++

## Introduction

[CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) is the Python-native interface to [CUTLASS](https://docs.nvidia.com/cutlass/latest/) 4.4+, NVIDIA's open-source library of high-performance CUDA kernels. It exposes the same CuTe abstractions (layouts, tensors, thread-to-data mappings) that power CUTLASS's C++ template library, but authored entirely in Python.

Traditionally, writing custom GPU kernels meant working in C++ or CUDA — a steep learning curve for Python-focused ML engineers. CuTe DSL changes this: you define per-thread logic with `@cute.kernel`, configure launch parameters with `@cute.jit`, and the CUTLASS JIT compiler generates optimized CUDA code behind the scenes. The `cutlass.jax` integration module then lets you call these kernels from JAX as if they were native operations, with full support for `@jax.jit`, automatic differentiation plumbing, and multi-device sharding.

This notebook walks through progressively more complex kernels showing the patterns you'll reuse in your own custom operations.

+++

## The CuTe mental model

At its core, CuTe is an **index transformation DSL** — it provides abstractions for mapping logical coordinates to physical memory offsets. Everything in CuTe builds on the following concepts:

**Shape** describes the dimensions of your data. A shape can be simple like `(M, N)` for a matrix, or hierarchical like `((2, 4), N)` where the first mode is itself subdivided. Hierarchical shapes are especially useful on GPUs, where work is organized in layers:

- A **thread** is the smallest unit of execution — one thread runs one sequence of instructions.
- A **warp** is a group of 32 threads that execute in lockstep on the same hardware unit.
- A **block** is a group of threads (organized internally into warps) that share fast on-chip (shared) memory and can synchronize with each other.
- The **grid** is the collection of all blocks launched by a kernel.

CuTe shapes can nest to mirror this hierarchy. Such a hierarchical shape can be used to model a GPU execution hierarchy — for example, 32 threads per warp × 8 warps per block, across N blocks — when bound to CUDA’s thread and block indices.

**Coordinate** is a position within a shape. For a shape `(4, 8)`, the coordinate `(2, 5)` identifies one element — row 2, column 5.

**Stride** tells CuTe how far you move in memory when you step along each dimension. In a row-major `(4, 8)` matrix, memory is laid out row by row: the first 8 elements belong to row 0, the next 8 to row 1, and so on. Moving one column to the right simply advances to the next element in memory (stride 1). Moving one row down skips over an entire row of 8 elements (stride 8). So the stride is `(8, 1)`.

**Layout = (Shape, Stride)** is CuTe's central abstraction. Shape and stride must have the same rank — each logical dimension must have a corresponding stride.

Although we think of tensors as multi-dimensional, GPU memory itself is just a long one-dimensional array of elements. Given a coordinate, a layout tells you where that element lives in memory. It does this by combining the coordinate with the stride:

```
offset = coord[0] * stride[0] + coord[1] * stride[1] + ...
```

In CuTe DSL, you can define layout using:

```python
cute.make_layout((...), stride=(...))
```

One important thing to note here: in  *row-major* layout, elements of each row are stored contiguously in memory (so columns vary fastest), whereas in *column-major* layout, elements of each column are stored contiguously (so rows vary fastest), meaning the logical shape stays the same but the stride — and therefore the memory access pattern — changes.

For example, with layout in row-major order `((4, 8), (8, 1))`, coordinate `(2, 5)` maps to offset `2*8 + 5*1 = 21`.

A column-major layout for the same shape would use stride `(1, 4)`, so the same coordinate maps to `2*1 + 5*4 = 22`.

The shape stays the same — only the stride changes.

```python
row_major = cute.make_layout((M, N), stride=(N, cutlass.Int32(1)))
col_major = cute.make_layout((M, N), stride=(cutlass.Int32(1), M))
```

This separation of logical structure from physical storage is what makes CuTe powerful. Algorithms operate on coordinates, while layouts decide how those coordinates map to memory. Change the stride, and you change the storage pattern — without rewriting the algorithm.

In the following examples, you won’t see `make_layout` because the kernels operate on `cute.Tensor` objects and use CuTe’s tensor / fragment helpers (`cute.size`, `cute.make_rmem_tensor`, `cute.autovec_copy`, `Tensor[...]`) which already encode the shape, stride and indexing semantics the kernel needs. The code stays higher-level and avoids manual offset arithmetic or explicit layout construction — that’s deliberate: CuTe’s helpers are there so kernels read like algorithms, not pointer math.

+++

## Hardware and software requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| GPU | SM 8.0+ (Ampere) | SM 9.0+ (Hopper) |
| CUDA | 12.9 | 13.1 |
| JAX | 0.8.1+ | Latest |
| CUTLASS | 4.4+ (CuTe DSL) | Latest |
| Python | 3.10+ | 3.12 |

+++

First, let's check which GPUs are available in this environment. The `nvidia-smi` command shows the GPU model, driver version, CUDA toolkit version, and current memory usage.

```{code-cell} ipython3
!nvidia-smi
```

We programmatically query the GPU's **compute capability** using `nvidia-smi`. This two-digit number (e.g., 9.0 for Hopper) tells us which hardware features are available. CuTe DSL requires SM 8.0 (Ampere) or newer.

```{code-cell} ipython3
import subprocess


def get_compute_capability():
  """Query the compute capability of the first visible GPU."""
  out = subprocess.check_output(
      ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
      text=True,
  )
  major, minor = out.strip().split("\n")[0].split(".")
  return int(major), int(minor)


SM_MAJOR, SM_MINOR = get_compute_capability()
print(f"Detected compute capability: SM {SM_MAJOR}.{SM_MINOR}")

if SM_MAJOR < 8:
  print("WARNING: CuTe DSL requires SM 8.0+ (Ampere or newer).")
  print("Some examples may not run on this GPU.")
else:
  print("GPU is compatible with CuTe DSL.")
```

## Install CuTe DSL and import dependencies

The `nvidia-cutlass-dsl` package bundles CuTe DSL together with its JAX integration module (`cutlass.jax`). The `[cu13]` extra pulls in CUDA 13.x compatible runtime libraries. Version 4.4+ is required for the JAX integration.

Refer to the [official documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html) for a more comprehensive installation guide.

```{code-cell} ipython3
%pip install "nvidia-cutlass-dsl[cu13]" --quiet
```

With CUTLASS installed, we import the libraries we'll use throughout the notebook: `cutlass` for kernel definitions, `jax` and `jnp` for array computation and JIT compilation, and `numpy` for result validation.

```{code-cell} ipython3
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF/XLA info & warnings
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

import cutlass
from importlib.metadata import version as _pkg_version

print(f"CUTLASS version: {_pkg_version('nvidia-cutlass-dsl')}")

import cutlass.cute as cute
import cutlass.jax as cjax
import cuda.bindings.driver as cuda

import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX version:     {jax.__version__}")
print(f"JAX devices:     {jax.devices()}")
```

## Defining kernels

In CuTe DSL, kernels are defined in two layers:

1. **`@cute.kernel`** defines the per-thread program — the sequence of instructions executed by each thread instance.
2. **`@cute.jit`** compiles the kernel and specifies how it runs on the GPU: the grid (how many blocks), the block (how many threads per block), and the CUDA stream (which execution queue to launch into).

CuTe DSL lowers Python kernels to CUDA/CUTLASS code and compiles them just-in-time using the CUTLASS JIT toolchain.

**Note:** CuTe DSL relies on Python source inspection `inspect.getsourcelines()` to parse kernel definitions. In many environments (including this notebook), defining `@cute.kernel` / `@cute.jit` functions directly in notebook cells works correctly. However, this is not consistently reliable across all interactive environments (e.g. plain Python REPL), where source inspection may fail with errors like `OSError: could not get source code`.

We show the executable kernel definitions inline in the notebook. At the same time, for robustness and reproducibility, we keep equivalent definitions in a separate .py module ([cute_dsl_jax_kernels.py](cute_dsl_jax/cute_dsl_jax_kernels.py)).

Here, we import the pre-written kernel launch functions from [cute_dsl_jax_kernels.py](cute_dsl_jax/cute_dsl_jax_kernels.py).

```{code-cell} ipython3
# Optional, if you execute the equivalent kernel definitions further in the notebook

# from cute_dsl_jax.cute_dsl_jax_kernels import (
#     launch_vector_add, launch_saxpy, launch_gemm,
#     launch_relu, launch_fused_bias_relu,
#     launch_elementwise_add,
# )
# print("Imported: launch_vector_add, launch_saxpy, launch_gemm, launch_relu, launch_fused_bias_relu, launch_elementwise_add")
```

```{code-cell} ipython3
def split_keys(seed=0):
  key = jax.random.key(seed)
  while True:
    key, subkey = jax.random.split(key)
    yield subkey

keys = iter(split_keys())
```

## Basic kernel: vector add

We’ll start with the simplest GPU kernel — vector add: `c[i] = a[i] + b[i]`.

Each thread in the kernel below identifies itself using `thread_idx()` and `block_idx()`. Thread and block indices are accessed through `cute.arch` (e.g., `thread_idx`, `block_idx`), each returning `(x, y, z)` tuples, because CUDA’s execution and indexing are 3-dimensional by design. Since this kernel is launched in 1D, we only use the `x` component (`tidx` and `bidx`) and ignore the unused `y` and `z` values with `_`.

```python
tidx, _, _ = cute.arch.thread_idx()
bidx, _, _ = cute.arch.block_idx()
```

Inside the kernel, tensors are typically created in register memory using `cute.make_rmem_tensor`.

Below, `frgA` and `frgB` hold the input values in registers, while `frgC` is a register fragment that will store the computed result before it is written back to global memory. `mode=[0]` selects the first dimension of the tensor — the "elements per thread" axis — so the register fragment is sized to hold exactly the data owned by one thread.

Data movement between global and register memory is explicit: fragments are read using `load()` and written back using `store()`, while `cute.autovec_copy` performs efficient, vectorized transfers between memory spaces. Here, one element of `a` and `b` is loaded into register fragments, the sum in registers is computed and the result is stored back to `c`. The `None` selects the entire first dimension (which has size 1 in this example), preserving the (`elems_per_thread`, `threads_per_block`, `num_blocks`) structure while allowing each thread to access its own slice of the tensor.

>  **Concept: Tensor = Pointer + Layout**
>
> A CuTe **Tensor** pairs a pointer to GPU memory with a **Layout** that describes how to navigate it. When the kernel receives `a: cute.Tensor`, it gets both the raw data and the index mapping. In this example, our tensors have shape `(1, BLOCK, num_blocks)` — one element per thread, `BLOCK=256` (defined in the example below) threads per block, spread across blocks. The layout maps a `(elems_per_thread, threads_per_block, num_blocks)` coordinate to the flat memory offset where that element lives. The kernel never computes offsets manually — it just indexes the tensor with `a[None, tidx, bidx]` and CuTe's layout handles the rest.

```{code-cell} ipython3
@cute.kernel
def vector_add_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
  """Per-thread kernel: each thread adds one element."""
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()

  frgA = cute.make_rmem_tensor(cute.size(a, mode=[0]), a.element_type)
  frgB = cute.make_rmem_tensor(cute.size(b, mode=[0]), b.element_type)
  frgC = cute.make_rmem_tensor(cute.size(c, mode=[0]), c.element_type)

  cute.autovec_copy(a[None, tidx, bidx], frgA)
  cute.autovec_copy(b[None, tidx, bidx], frgB)
  frgC.store(frgA.load() + frgB.load())
  cute.autovec_copy(frgC, c[None, tidx, bidx])


print("vector_add_kernel defined.")
```

The `@cute.kernel` defines one thread’s work. The `@cute.jit` launcher decides how many threads run, and how they’re grouped. It must follow the signature convention: `(stream, *inputs, *outputs, *, **kwargs)` — where `stream` is a CUDA stream managed by XLA, followed by input tensors, then output tensors.

We launch `a.shape[-2]` threads per block and `a.shape[-1]` blocks, directly matching the tensor’s `(1, threads_per_block, num_blocks)` layout so that `threadIdx.x` indexes the thread dimension and `blockIdx.x` indexes the block dimension. We use -2 and -1 because they refer to the last two tensor dimensions (threads per block and number of blocks), making the launch configuration robust even if additional leading dimensions are added.

> **Concept: Layout composition**
>
> The vector add kernel expects 3-D tensors with shape `(elems_per_thread, threads_per_block, num_blocks)`, but our data is a flat 1-D array. The JAX wrapper reshapes from 1-D to 3-D before calling the kernel, and back afterward. In CuTe terms, this reshape is a **layout composition** — combining the original 1-D layout with a new layout that splits the single dimension into three. CuTe performs this algebraically: the composed layout maps 3-D coordinates directly to the original flat offsets, with no data movement. Reshaping is free — it's just a change of layout, not a copy.

```{code-cell} ipython3
@cute.jit
def launch_vector_add(
    stream: cuda.CUstream,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
):
  vector_add_kernel(a, b, c).launch(
      grid=[a.shape[-1], 1, 1],
      block=[a.shape[-2], 1, 1],
      stream=stream,
  )


print("launch_vector_add defined.")
```

### JAX integration via `cutlass_call`

The `cutlass.jax.cutlass_call` function wraps a CuTe `@cute.jit` launch function as a JAX custom call, so your CuTe/CUTLASS kernel can be invoked inside `@jax.jit`-compiled code and become part of the XLA computation graph.

High-level flow:

1. Prepare the data (pad + reshape)

* We pad `N` up to a multiple of BLOCK so blocks are full (no partial last block), then reshape the 1-D vector into the 3-D logical tensor shape the kernel expects: `(elems_per_thread, threads_per_block, num_blocks)`.
* This reshape is free — it only changes the layout/interpretation of memory. No copy happens.

```python
N = a.shape[0]
padded = ((N + BLOCK - 1) // BLOCK) * BLOCK
a_pad = jnp.pad(a, (0, padded - N))
a_3d = a_pad.reshape(1, BLOCK, padded // BLOCK)
```

2. Wrap the launcher

* This returns a callable that accepts JAX arrays (DeviceArrays) and will, when executed inside `@jax.jit`, lower to a JAX custom call that launches your compiled CUTLASS kernel.
* `output_shape_dtype` tells JAX/XLA what the kernel will produce so shapes and dtypes are known for compilation and graph building.
* `use_static_tensors=True` asks the wrapper to treat the kernel tensors as static (compile-time) shapes where possible — this allows CuTe/CUTLASS to generate specialized, high-performance code for fixed shapes.

```python
call = cjax.cutlass_call(
    launch_fn,                    # The @cute.jit function
    output_shape_dtype=...,       # Shape/dtype of output(s)
)
result = call(*input_arrays)      # Pass JAX arrays here
```

3. Call the wrapped launcher

* Inside a `@jax.jit` function this becomes a custom call node in the XLA graph; at runtime XLA provides a CUDA `CUstream` and device pointers, and the CUTLASS kernel is invoked on that stream.
* The callable accepts JAX arrays and returns a JAX array containing the kernel output.

```python
c_3d = call(a_3d, b_3d)
```

4. Unpack back to 1-D and trim padding

* Convert the logical 3-D result back to a flat 1-D array and drop the padded tail.

```python
return c_3d.reshape(-1)[:N]
```

```{code-cell} ipython3
BLOCK = 256  # threads per block for vector add: 256 is a practical default:
# large enough to expose parallelism, small enough to scale
# well across different GPUs, and aligned with the hardware’s
# 32-thread warp execution model.


@jax.jit
def jax_vector_add(a, b):
  """JAX-compatible vector add using CUTLASS kernel."""
  N = a.shape[0]
  padded = ((N + BLOCK - 1) // BLOCK) * BLOCK
  a_pad = jnp.pad(a, (0, padded - N))
  b_pad = jnp.pad(b, (0, padded - N))
  # Reshape to (1, BLOCK, num_blocks) for the CuTe kernel
  a_3d = a_pad.reshape(1, BLOCK, padded // BLOCK)
  b_3d = b_pad.reshape(1, BLOCK, padded // BLOCK)
  call = cjax.cutlass_call(
      launch_vector_add,
      output_shape_dtype=jax.ShapeDtypeStruct(a_3d.shape, a_3d.dtype),
      use_static_tensors=True,
  )
  c_3d = call(a_3d, b_3d)
  return c_3d.reshape(-1)[:N]


print("jax_vector_add defined.")
```

Let's test our CUTLASS vector add by comparing its output against JAX's built-in `+` operator. We generate two random arrays, run both implementations, and verify the results match element-by-element.

```{code-cell} ipython3
# Test vector add
N = 1024
a = jax.random.normal(next(keys), (N,), dtype=jnp.float32)
b = jax.random.normal(next(keys), (N,), dtype=jnp.float32)

c = jax_vector_add(a, b)
c_ref = a + b

np.testing.assert_allclose(np.array(c), np.array(c_ref), rtol=1e-5)
print(f"Vector Add PASSED (N={N})")
print(f"  Max error: {float(jnp.max(jnp.abs(c - c_ref))):.2e}")
```

## SAXPY: scalar parameters in kernels

**SAXPY** computes `out[i] = alpha * x[i] + y[i]`. This builds on the vector add pattern and introduces passing a **scalar argument** (`alpha`) alongside tensor arguments.

The SAXPY kernel follows the same structure as vector add — identify the thread, load data into registers, compute, write back — with one addition: a scalar `alpha` parameter.

```python
def saxpy_kernel(x: cute.Tensor, y: cute.Tensor, out: cute.Tensor, alpha: float):
```

The signature adds `alpha: float` alongside the tensor arguments. CuTe DSL compiles scalar parameters just like CUDA kernel arguments — they are passed by value and available to every thread.

The body is identical to vector add except for the arithmetic:

```python
frgO.store(alpha * frgX.load() + frgY.load())
```

Each thread loads its element of `x` and `y` into register fragments, multiplies `x` by `alpha`, adds `y`, and writes the result to `out`.

```{code-cell} ipython3
@cute.kernel
def saxpy_kernel(
    x: cute.Tensor, y: cute.Tensor, out: cute.Tensor, alpha: float
):
  """SAXPY: out[i] = alpha * x[i] + y[i]."""
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()

  frgX = cute.make_rmem_tensor(cute.size(x, mode=[0]), x.element_type)
  frgY = cute.make_rmem_tensor(cute.size(y, mode=[0]), y.element_type)
  frgO = cute.make_rmem_tensor(cute.size(out, mode=[0]), out.element_type)

  cute.autovec_copy(x[None, tidx, bidx], frgX)
  cute.autovec_copy(y[None, tidx, bidx], frgY)
  frgO.store(alpha * frgX.load() + frgY.load())
  cute.autovec_copy(frgO, out[None, tidx, bidx])


print("saxpy_kernel defined.")
```

The launcher passes `alpha` as a **keyword-only** argument (note the `*` in the signature):

```{code-cell} ipython3
@cute.jit
def launch_saxpy(
    stream: cuda.CUstream,
    x: cute.Tensor,
    y: cute.Tensor,
    out: cute.Tensor,
    *,
    alpha: float,
):
  saxpy_kernel(x, y, out, alpha).launch(
      grid=[x.shape[-1], 1, 1],
      block=[x.shape[-2], 1, 1],
      stream=stream,
  )


print("launch_saxpy defined.")
```

The keyword-only convention matters for `cutlass_call`: positional arguments correspond to JAX tensors (managed by XLA), while keyword arguments are scalar values passed directly to the kernel. In the JAX wrapper below, `alpha=alpha` routes through `cutlass_call` as a kernel kwarg:

```python
call = cjax.cutlass_call(
    launch_saxpy,
    ...,
    alpha=alpha,    # scalar kwarg → passed to the kernel
)
out_3d = call(x_3d, y_3d)  # tensor args → managed by XLA
```

> **Concept: Static vs dynamic integers**
>
> CUTLASS distinguishes between values known at **compile time** (static) and values known only at **runtime** (dynamic). Static integers — like tensor shapes passed with `use_static_tensors=True` or constants like `BLOCK_SIZE` — are baked into the generated CUDA code, letting the compiler unroll loops, optimize memory access patterns, and eliminate branches. Dynamic values like `alpha` are passed as regular kernel arguments and read at runtime. As a rule of thumb: make shapes and tile sizes static, keep data-dependent values dynamic.

Note that `jax_saxpy` uses `@partial(jax.jit, static_argnums=(2,))` to mark `alpha` as a static argument to JAX. This means JAX will recompile the function whenever `alpha` changes — which is fine for a value that rarely varies, and lets the CUTLASS JIT bake the exact `alpha` value into the generated CUDA code.

```{code-cell} ipython3
from functools import partial

BLOCK = 256


@partial(jax.jit, static_argnums=(2,))
def jax_saxpy(x, y, alpha=2.0):
  """JAX-compatible SAXPY using CUTLASS kernel."""
  N = x.shape[0]
  padded = ((N + BLOCK - 1) // BLOCK) * BLOCK
  x_pad = jnp.pad(x, (0, padded - N))
  y_pad = jnp.pad(y, (0, padded - N))
  x_3d = x_pad.reshape(1, BLOCK, padded // BLOCK)
  y_3d = y_pad.reshape(1, BLOCK, padded // BLOCK)
  call = cjax.cutlass_call(
      launch_saxpy,
      output_shape_dtype=jax.ShapeDtypeStruct(x_3d.shape, x_3d.dtype),
      use_static_tensors=True,
      alpha=alpha,
  )
  out_3d = call(x_3d, y_3d)
  return out_3d.reshape(-1)[:N]


print("jax_saxpy defined.")
```

We test the SAXPY kernel with `alpha=2.5`, comparing against the reference computation `alpha * x + y`. The `assert_allclose` check verifies that results match within floating-point tolerance.

```{code-cell} ipython3
# Test SAXPY
N = 2048
ALPHA = 2.5
x = jax.random.normal(next(keys), (N,), dtype=jnp.float32)
y = jax.random.normal(next(keys), (N,), dtype=jnp.float32)

result = jax_saxpy(x, y, alpha=ALPHA)
ref = ALPHA * x + y

np.testing.assert_allclose(np.array(result), np.array(ref), rtol=1e-5)
print(f"SAXPY PASSED (N={N}, alpha={ALPHA})")
print(f"  Max error: {float(jnp.max(jnp.abs(result - ref))):.2e}")
```

## Deep learning activations: ReLU and fused bias+ReLU

### ReLU

**ReLU** (`max(0, x)`) is the most widely used activation function in deep learning. It's elementwise and trivially parallel — a perfect custom kernel for ML workloads.

The ReLU kernel uses a different pattern from vector add and SAXPY. Instead of the 3-D tensor approach with register fragments, it uses **flat 1-D indexing** — simpler and equally efficient for elementwise operations.

```python
tidx, _, _ = cute.arch.thread_idx()
bidx, _, _ = cute.arch.block_idx()
bdx, _, _ = cute.arch.block_dim()
```

A new call appears here: `cute.arch.block_dim()` returns the number of threads per block (set at launch time). Together with `thread_idx` and `block_idx`, it lets each thread compute its unique **global index**:

```python
idx = bidx * bdx + tidx
```

For example, if we launch 256 threads per block: thread 3 in block 2 gets `idx = 2 * 256 + 3 = 515`. This is the standard CUDA pattern for mapping threads to 1-D data.

Here we index the tensors directly with `x[idx]` — no register fragments or `autovec_copy`. For simple elementwise operations this flat approach is cleaner. `cutlass.max` is CuTe DSL's built-in max function, and `cutlass.Float32(0.0)` creates a typed zero constant that matches the tensor's element type.

```{code-cell} ipython3
@cute.kernel
def relu_kernel(x: cute.Tensor, out: cute.Tensor, N: int):
  """Per-thread kernel: each thread computes ReLU of one element."""
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()
  bdx, _, _ = cute.arch.block_dim()

  idx = bidx * bdx + tidx
  if idx < N:
    val = x[idx]
    out[idx] = cutlass.max(val, cutlass.Float32(0.0))


print("relu_kernel defined.")
```

The launcher computes how many blocks are needed to cover `N` elements:

```{code-cell} ipython3
@cute.jit
def launch_relu(
    stream: cuda.CUstream,
    x: cute.Tensor,
    out: cute.Tensor,
    *,
    N: int,
):
  BLOCK_SIZE = 256
  grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
  relu_kernel(x, out, N).launch(
      grid=[grid_size, 1, 1],
      block=[BLOCK_SIZE, 1, 1],
      stream=stream,
  )


print("launch_relu defined.")
```

The formula `(N + BLOCK_SIZE - 1) // BLOCK_SIZE` is ceiling division — it ensures we launch enough blocks even when `N` isn't a multiple of 256. The bounds check inside the kernel handles the leftover threads in the last block.

### JAX wrapper: ReLU

The JAX wrapper is simpler than vector add because we skip the 3-D reshape. The kernel works on flat 1-D data directly:

```python
x_flat = x.reshape(-1)         # flatten to 1-D
call = cjax.cutlass_call(
    launch_relu,
    output_shape_dtype=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
    N=N,                        # scalar kwarg → bounds check inside kernel
)
out_flat = call(x_flat)
return out_flat.reshape(x.shape)  # restore original shape
```

`N` is passed as a keyword argument so the kernel knows how many elements are valid. The output is reshaped back to match the input's original shape (works for any dimensionality).

```{code-cell} ipython3
@jax.jit
def jax_relu(x):
  """JAX-compatible ReLU using CUTLASS kernel."""
  N = x.size
  x_flat = x.reshape(-1)
  call = cjax.cutlass_call(
      launch_relu,
      output_shape_dtype=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
      N=N,
  )
  out_flat = call(x_flat)
  return out_flat.reshape(x.shape)


print("jax_relu defined.")
```

We verify the ReLU kernel by comparing against `jax.nn.relu`. Positive values should pass through unchanged, and negative values should become zero.

```{code-cell} ipython3
# Test ReLU
N = 2048
x = jax.random.normal(next(keys), (N,), dtype=jnp.float32)

result = jax_relu(x)
ref = jax.nn.relu(x)

np.testing.assert_allclose(np.array(result), np.array(ref), rtol=1e-5)
print(f"ReLU PASSED (N={N})")
print(f"  Max error: {float(jnp.max(jnp.abs(result - ref))):.2e}")
print(f"  Sample: x[:6]   = {x[:6]}")
print(f"          out[:6] = {result[:6]}")
```

### Fused bias+ReLU

**Fused bias+ReLU** computes `max(0, x + bias)` in a single kernel. This demonstrates **kernel fusion** — combining multiple operations into one GPU pass.

Why fusion matters:
- **Without fusion:** `x + bias` writes an intermediate array to global memory, then `max(0, ...)` reads it back. That's two kernel launches and two round-trips to memory.
- **With fusion:** one kernel reads `x` and `bias`, computes the sum and the max, and writes the final result. Half the memory traffic, one launch instead of two.

The kernel extends the ReLU pattern with a bias lookup.

The input `x` is a flattened `(batch, width)` matrix. `idx` is the global flat index, and `col = idx % width` recovers which column (feature) this element belongs to, so we can look up the correct bias. This modular indexing pattern is common in fused kernels that combine elementwise and broadcast operations.

```{code-cell} ipython3
@cute.kernel
def fused_bias_relu_kernel(
    x: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    N: int,
    width: int,
):
  """Per-thread: out[i] = max(0, x[i] + bias[i % width])."""
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()
  bdx, _, _ = cute.arch.block_dim()

  idx = bidx * bdx + tidx
  if idx < N:
    col = idx % width
    val = x[idx] + bias[col]
    out[idx] = cutlass.max(val, cutlass.Float32(0.0))


print("fused_bias_relu_kernel defined.")
```

The launcher and JAX wrapper follow the same flat-indexing pattern as ReLU, with `N` (total elements) and `width` (columns) passed as keyword arguments:

```{code-cell} ipython3
@cute.jit
def launch_fused_bias_relu(
    stream: cuda.CUstream,
    x: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    *,
    N: int,
    width: int,
):
  BLOCK_SIZE = 256
  grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
  fused_bias_relu_kernel(x, bias, out, N, width).launch(
      grid=[grid_size, 1, 1],
      block=[BLOCK_SIZE, 1, 1],
      stream=stream,
  )


print("launch_fused_bias_relu defined.")
```

Note that `width` is marked as a static argument in the JAX wrapper via `static_argnums=(2,)`. This means JAX recompiles when the feature dimension changes, allowing CUTLASS to generate specialized code for each width.

```python
call = cjax.cutlass_call(
    launch_fused_bias_relu,
    output_shape_dtype=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
    N=N, width=width,
)
out_flat = call(x_flat, bias)   # two input tensors: x and bias
```

```{code-cell} ipython3
from functools import partial


@partial(jax.jit, static_argnums=(2,))
def jax_fused_bias_relu(x, bias, width):
  """JAX-compatible fused Bias+ReLU using CUTLASS kernel.

  Args:
      x: Input matrix of shape (batch, width), flattened to 1-D for the kernel.
      bias: Bias vector of shape (width,).
      width: Number of columns (static, passed as constexpr to the kernel).
  """
  N = x.size
  x_flat = x.reshape(-1)
  call = cjax.cutlass_call(
      launch_fused_bias_relu,
      output_shape_dtype=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
      N=N,
      width=width,
  )
  out_flat = call(x_flat, bias)
  return out_flat.reshape(x.shape)


print("jax_fused_bias_relu defined.")
```

Test the fused kernel against the equivalent two-step JAX computation: add bias, then apply ReLU. The results should match exactly since both paths perform the same arithmetic.

```{code-cell} ipython3
# Test Fused Bias+ReLU
BATCH, WIDTH = 64, 512
x = jax.random.normal(next(keys), (BATCH, WIDTH), dtype=jnp.float32)
bias = jax.random.normal(next(keys), (WIDTH,), dtype=jnp.float32)

result = jax_fused_bias_relu(x, bias, WIDTH)
ref = jnp.maximum(0, x + bias[None, :])

np.testing.assert_allclose(np.array(result), np.array(ref), rtol=1e-5)
print(f"Fused Bias+ReLU PASSED (batch={BATCH}, width={WIDTH})")
print(f"  Max error: {float(jnp.max(jnp.abs(result - ref))):.2e}")
```

> **Going further:** For a production-grade generalization of elementwise kernels — with optimized TV (thread-value) layouts, vectorized memory access, and support for arbitrary binary operators including custom ops like `leaky_relu` — see NVIDIA's [elementwise_apply_example.py](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/jax/elementwise_apply_example.py).

+++

## Advanced: Tiled GEMM

This demonstrates a general matrix multiply (GEMM) kernel: `D = A @ B` where `A` is `(M, K)`, `B` is `(K, N)`, and `D` is `(M, N)`. Unlike the previous elementwise kernels, GEMM requires cooperation across data dimensions — each output element is a dot product over `K` values.

> **Concept: Tiling**
>
> Tiling is CuTe's mechanism for partitioning data into sub-problems that map onto the GPU's execution hierarchy. In a GEMM, we divide the output matrix into `BLOCK_M x BLOCK_N` tiles, each assigned to one thread block. Within a tile, individual threads split the work further. CuTe's tiling operations decompose a layout into an "inner" part (the tile itself) and an "outer" part (which tile we're on). The block index `(bm, bn)` selects the outer coordinate, and thread indices work within the inner tile. This two-level decomposition — partition then index locally — is the fundamental pattern for mapping parallel GPU work to data.

This is our first kernel using a **2-D grid**. Each block is responsible for a `BLOCK_M x BLOCK_N` tile of the output matrix:

```python
    tidx, _, _ = cute.arch.thread_idx()
    bm, bn, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
```

Note `bm, bn, _` — the block index now has two meaningful components: `bm` selects the tile row, `bn` selects the tile column.

Each tile contains `BLOCK_M * BLOCK_N` output elements, but we only have `bdx` (256) threads per block. A **stride loop** distributes the work evenly:

```python
    for i in cutlass.range(tidx, BLOCK_M * BLOCK_N, bdx):
```

This loop starts at `tidx` and steps by `bdx` (the block size). For a `64 x 64 = 4096` element tile with 256 threads, each thread computes `4096 / 256 = 16` output elements. `cutlass.range` works like Python's `range()` but generates CUDA loop code.

Within the loop, the flat tile index `i` is converted to 2-D tile-local coordinates, then to global matrix coordinates:

```python
        row = i // BLOCK_N          # tile-local row
        col = i % BLOCK_N           # tile-local column
        m_idx = bm * BLOCK_M + row  # global row in D
        n_idx = bn * BLOCK_N + col  # global column in D
```

A bounds check handles edge tiles where the matrix dimensions aren't multiples of the block size:

```python
        if m_idx < M and n_idx < N:
```

The inner loop accumulates the dot product over the K dimension:

```python
            acc = cutlass.Float32(0.0)
            for k in cutlass.range(K):
                acc += A[m_idx * K + k] * B[k * N + n_idx]
            D[m_idx * N + n_idx] = acc
```

Here we use **manual row-major indexing**: `A[m_idx * K + k]` computes the offset into the flattened 1-D tensor for element `(m_idx, k)` of a row-major matrix with K columns. Similarly, `B[k * N + n_idx]` indexes element `(k, n_idx)`. Production CUTLASS kernels use multi-dimensional CuTe tensor indexing instead, but explicit indexing makes the memory layout visible for learning.

```{code-cell} ipython3
@cute.kernel
def gemm_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    D: cute.Tensor,
    M: int,
    N: int,
    K: int,
    BLOCK_M: int,
    BLOCK_N: int,
):
  """Tiled GEMM: each thread accumulates output elements."""
  tidx, _, _ = cute.arch.thread_idx()
  bm, bn, _ = cute.arch.block_idx()
  bdx, _, _ = cute.arch.block_dim()

  for i in cutlass.range(tidx, BLOCK_M * BLOCK_N, bdx):
    row = i // BLOCK_N
    col = i % BLOCK_N
    m_idx = bm * BLOCK_M + row
    n_idx = bn * BLOCK_N + col
    if m_idx < M and n_idx < N:
      acc = cutlass.Float32(0.0)
      for k in cutlass.range(K):
        acc += A[m_idx * K + k] * B[k * N + n_idx]
      D[m_idx * N + n_idx] = acc


print("gemm_kernel defined.")
```

The launcher sets up a 2-D grid matching the tile decomposition:

- `grid=[grid_m, grid_n, 1]` — one block per output tile, arranged in a 2-D grid
- `block=[256, 1, 1]` — 256 threads per block, each handling multiple elements via the stride loop
- `M, N, K, BLOCK_M, BLOCK_N` are all passed as compile-time constants to the kernel

```{code-cell} ipython3
@cute.jit
def launch_gemm(
    stream: cuda.CUstream,
    A: cute.Tensor,
    B: cute.Tensor,
    D: cute.Tensor,
    *,
    M: int,
    N: int,
    K: int,
):
  BLOCK_M, BLOCK_N = 64, 64
  grid_m = (M + BLOCK_M - 1) // BLOCK_M
  grid_n = (N + BLOCK_N - 1) // BLOCK_N
  gemm_kernel(A, B, D, M, N, K, BLOCK_M, BLOCK_N).launch(
      grid=[grid_m, grid_n, 1],
      block=[256, 1, 1],
      stream=stream,
  )


print("launch_gemm defined.")
```

The JAX wrapper flattens both input matrices to 1-D (matching the kernel's flat indexing), passes the matrix dimensions as keyword arguments, and reshapes the result:

```{code-cell} ipython3
@jax.jit
def jax_cutlass_gemm(a, b):
  """JAX wrapper for the CUTLASS GEMM kernel."""
  M, K = a.shape
  _, N = b.shape
  a_flat = a.reshape(-1)
  b_flat = b.reshape(-1)
  call = cjax.cutlass_call(
      launch_gemm,
      output_shape_dtype=jax.ShapeDtypeStruct((M * N,), a.dtype),
      M=M,
      N=N,
      K=K,
  )
  d_flat = call(a_flat, b_flat)
  return d_flat.reshape(M, N)


print("jax_cutlass_gemm defined.")
```

Test the CUTLASS GEMM against JAX's `jnp.matmul`. We use relaxed tolerances (`rtol=1e-2`) because our simple kernel accumulates the K-dimension in a different order than cuBLAS, leading to small floating-point differences that are expected and harmless.

```{code-cell} ipython3
# Test GEMM
M, N, K = 256, 256, 128
A = jax.random.normal(next(keys), (M, K), dtype=jnp.float32)
B = jax.random.normal(next(keys), (K, N), dtype=jnp.float32)

D = jax_cutlass_gemm(A, B)
D_ref = jnp.matmul(A, B)

np.testing.assert_allclose(np.array(D), np.array(D_ref), rtol=1e-2, atol=1e-2)
print(f"GEMM PASSED (M={M}, N={N}, K={K})")
print(f"  Max error: {float(jnp.max(jnp.abs(D - D_ref))):.2e}")
```

## Performance comparison

Let's compare our CUTLASS GEMM kernel against JAX's built-in `jnp.matmul` (which calls cuBLAS under the hood).

Our simple tiled kernel is **not expected to beat cuBLAS** — cuBLAS is one of the most heavily optimized libraries in existence, with hand-tuned assembly for each GPU architecture. The goal here is to show the integration pattern and demonstrate that custom kernels produce correct results.

CuTe DSL's real value shows up when you need kernels that cuBLAS doesn't provide: custom fusions, non-standard data layouts, mixed-precision schemes, or operations specific to your model architecture.

The benchmark below runs each implementation 20 times (after a warmup pass to trigger JIT compilation) and reports the average wall-clock time. `block_until_ready()` ensures we time the actual GPU execution, not just the asynchronous launch.

```{code-cell} ipython3
import time

M, N, K = 512, 512, 512
A = jax.random.normal(next(keys), (M, K), dtype=jnp.float32)
B = jax.random.normal(next(keys), (K, N), dtype=jnp.float32)

# Warmup
_ = jax_cutlass_gemm(A, B).block_until_ready()
_ = jnp.matmul(A, B).block_until_ready()

NUM_RUNS = 20

# Time CUTLASS GEMM
start = time.perf_counter()
for _ in range(NUM_RUNS):
  _ = jax_cutlass_gemm(A, B).block_until_ready()
cutlass_time = (time.perf_counter() - start) / NUM_RUNS

# Time JAX matmul
start = time.perf_counter()
for _ in range(NUM_RUNS):
  _ = jnp.matmul(A, B).block_until_ready()
jax_time = (time.perf_counter() - start) / NUM_RUNS

print(f"Matrix size: {M}x{N}x{K}")
print(f"CUTLASS GEMM:  {cutlass_time*1000:.3f} ms")
print(f"JAX jnp.matmul: {jax_time*1000:.3f} ms")
print(f"Ratio (CUTLASS / JAX): {cutlass_time / jax_time:.2f}x")
print()
print("Note: Our simple tiled kernel is not expected to beat cuBLAS.")
print("CuTe DSL's value is in specialized kernels cuBLAS doesn't provide.")
```

## Multi-GPU: sharding CUTLASS kernels via `jax.shard_map`

One of JAX's key strengths is transparent multi-device execution. CUTLASS kernels integrated via `cutlass_call` participate fully in JAX's sharding APIs, so you can distribute work across all available GPUs without modifying the kernel code.

### How sharding works

The key idea: split the data across devices, run the same kernel independently on each device's local shard, and let JAX handle the coordination.

**1. Create a device mesh.** A mesh maps physical devices to named logical axes:

```python
mesh = jax.make_mesh((num_devices,), ("x",))
jax.set_mesh(mesh)
```

This creates a 1-D mesh with `num_devices` devices along an axis called `"x"`. For 8 GPUs, the mesh maps device 0 through device 7 to positions 0–7 along the `"x"` axis.

**2. Define the sharding spec.**

`PartitionSpec` tells JAX how to slice each tensor dimension across the mesh:

```python
sharding = P(None, None, "x")
```

For our 3-D tensors with shape `(elems_per_thread, threads_per_block, num_blocks)`:
- `None` — don't shard the first dimension (elems per thread, stays local)
- `None` — don't shard the second dimension (threads per block, stays local)
- `"x"` — shard the third dimension (blocks) across devices on the `"x"` axis

So with 8 devices and 128 total blocks, each device gets a tensor of shape `(1, 256, 16)` — its 16 local blocks.

**3. Create sharded inputs.**

With explicit mesh axes, inputs must already have a layout compatible with the mesh.

We create them directly with the desired sharding:

```python
a = jax.random.normal(
    jax.random.key(10),
    shape,
    dtype=jnp.float32,
    out_sharding=sharding,
)
b = jax.random.normal(
    jax.random.key(11),
    shape,
    dtype=jnp.float32,
    out_sharding=sharding,
)
```

This produces arrays with sharding P(None, None, "x"), matching the computation.

An equivalent alternative is to create unsharded arrays and place them explicitly:

```python
from jax.sharding import NamedSharding

named_sharding = NamedSharding(mesh, sharding)

a = jax.random.normal(jax.random.key(10), shape, dtype=jnp.float32)
b = jax.random.normal(jax.random.key(11), shape, dtype=jnp.float32)

a = jax.device_put(a, named_sharding)
b = jax.device_put(b, named_sharding)
```

**4. Use `jax.shard_map` to run per-device code**

With an explicit mesh set via `jax.set_mesh`, `jax.shard_map` can be written concisely:

```python
@jax.shard_map(out_specs=sharding)
def sharded_vector_add(a_shard, b_shard):
    call = cjax.cutlass_call(
        launch_vector_add,
        output_shape_dtype=jax.typeof(a_shard),
        use_static_tensors=True,
    )
    return call(a_shard, b_shard)
```

Inside `sharded_vector_add`, the code is identical to single-GPU — it sees a regular tensor and calls the same CUTLASS kernel. The kernel has no idea it's running on multiple GPUs. JAX handles splitting inputs before the kernel and reassembling outputs afterward.

```{code-cell} ipython3
from jax.sharding import PartitionSpec as P

num_devices = len(jax.devices())
print(f"Number of devices: {num_devices}")

BLOCK = 256

mesh = jax.make_mesh((num_devices,), ("x",))

# Use `jax.set_mesh` as a context manager so the mesh is scoped to this
# sharding demo and does not leak into later cells.
with jax.set_mesh(mesh):
  # Kernel expects 3-D tensors: (elems_per_thread, threads, blocks)
  # Shard along the blocks axis (last dim)
  sharding = P(None, None, "x")

  @jax.shard_map(out_specs=sharding)
  def sharded_vector_add(a_shard, b_shard):
    call = cjax.cutlass_call(
        launch_vector_add,
        output_shape_dtype=jax.typeof(a_shard),
        use_static_tensors=True,
    )
    return call(a_shard, b_shard)

  # Create 3-D tensors: (1, 256, total_blocks) with total_blocks divisible by device count
  blocks_per_device = 16
  total_blocks = blocks_per_device * num_devices
  shape = (1, BLOCK, total_blocks)

  a_m = jax.random.normal(
      jax.random.key(10),
      shape,
      dtype=jnp.float32,
      out_sharding=sharding,
  )
  b_m = jax.random.normal(
      jax.random.key(11),
      shape,
      dtype=jnp.float32,
      out_sharding=sharding,
  )

  print("a_m sharding:", a_m.sharding)
  print("b_m sharding:", b_m.sharding)

  c_m = sharded_vector_add(a_m, b_m)

  np.testing.assert_allclose(jnp.array(c_m), jnp.array(a_m + b_m), rtol=1e-5)
  n_total = int(np.prod(shape))
  print(f"Sharded Vector Add PASSED across {num_devices} devices (N={n_total})")
```

## Exporting CUTLASS kernels with `jax.export`

So far, every kernel we've written lives inside a `@jax.jit` function — it compiles and runs within the current Python process. But what if you want to **save** a compiled JAX function containing a CUTLASS kernel, ship it to another machine, or load it in a non-Python runtime?

That's what `jax.export` does. It takes a JIT-compiled function and produces a **standalone, serialized artifact** that you can save to disk, send over the network, and reload later — even after the original Python program has exited. Without `jax.export`, JAX functions are only compiled and callable inside the same Python process through `jit`.

With `jax.export` you get:

- **Serialization** — turn your staged JAX computation into a blob that can be stored and reused
- **Interoperability** — future tools could invoke this from non-Python runtimes (TensorFlow, C++, other frameworks)
- **Stable HLO output** — useful for ahead-of-time (AOT) compilation, deployment, and cross-platform interoperability

For CUTLASS kernels specifically:

- The exported function includes **custom calls** to CUTLASS kernels — these aren't part of JAX's built-in compilation pipeline. `get_export_disabled_safety_checks()` tells JAX that these custom calls are safe to include in the exported output.
- With **symbolic shapes**, the exported artifact works for multiple input sizes without recompilation. The kernel doesn't have to be recompiled for new input shapes after export.

### What `jax.export` gives you

- **A StableHLO representation** of the compiled function (the lowered intermediate representation)
- **Metadata** about the function's inputs and outputs
- **A serialized blob** you can save to disk or transmit over the network
- **A callable object** (`rehydrated.call(...)`) that works independently of the code that built it

### How it works

The flow is straightforward:

```python
from jax import export
from cutlass.jax import get_export_disabled_safety_checks

# 1. Export the JIT-compiled function
exported = jax.export.export(f, disabled_checks=get_export_disabled_safety_checks())

# 2. Specialize to a signature (concrete or symbolic shapes) and serialize
traced = exported(shape_dtype_spec, shape_dtype_spec)
blob = traced.serialize()

# 3. Later: deserialize and call with real data
rehydrated = export.deserialize(blob)
result = rehydrated.call(a, b)
```

The following example is adapted from [NVIDIA's official export example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/jax/cutlass_call_export.py). It exports a function that adds two matrices with a CUTLASS kernel and applies `sigmoid`, then serializes, deserializes, and verifies the result.

```{code-cell} ipython3
from cutlass.jax import get_export_disabled_safety_checks
from jax import export


# Element-wise Add (2-D, flat indexing)
@cute.kernel
def elementwise_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
  """Per-thread kernel: 2-D element-wise add using flat indexing."""
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()
  bdim, _, _ = cute.arch.block_dim()

  thread_idx = bidx * bdim + tidx

  m, n = gA.shape

  if thread_idx < m * n:

    ni = thread_idx % n
    mi = thread_idx // n

    a_val = gA[mi, ni]
    b_val = gB[mi, ni]
    gC[mi, ni] = a_val + b_val


@cute.jit
def launch_elementwise_add(
    stream: cuda.CUstream,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
  num_threads_per_block = 256
  m, n = mA.shape
  elementwise_add_kernel(mA, mB, mC).launch(
      grid=((m * n + num_threads_per_block - 1) // num_threads_per_block, 1, 1),
      block=(num_threads_per_block, 1, 1),
      stream=stream,
  )


# Define a function that uses a CUTLASS kernel + JAX ops.
# We use launch_elementwise_add which accepts 2-D tensors directly
# with flat indexing — compatible with jax.export's tracing.
@jax.jit
def f(a, b):
  call = cjax.cutlass_call(launch_elementwise_add, output_shape_dtype=a)
  return jax.nn.sigmoid(call(a, b))


# Reference implementation (pure JAX)
@jax.jit
def ref_f(a, b):
  return jax.nn.sigmoid(a + b)


# --- Export with concrete shapes ---
M, N = 512, 256
export_shape_dtype = jax.ShapeDtypeStruct((M, N), jnp.float32)

print(
    f"Exporting with input signature: ({export_shape_dtype},"
    f" {export_shape_dtype})"
)

# Export the function — get_export_disabled_safety_checks() tells JAX
# that CUTLASS custom call targets are safe to include
exported = jax.export.export(
    f, disabled_checks=get_export_disabled_safety_checks()
)
traced = exported(export_shape_dtype, export_shape_dtype)

# Serialize to a byte blob
blob = traced.serialize()
print(f"Serialized computation: {len(blob):,} bytes")

# Deserialize and run — this works independently of the original function
rehydrated = export.deserialize(blob)

a = jax.random.normal(next(keys), (M, N), dtype=jnp.float32)
b = jax.random.normal(next(keys), (M, N), dtype=jnp.float32)

c = rehydrated.call(a, b)
c_ref = ref_f(a, b)

np.testing.assert_allclose(np.array(c), np.array(c_ref), rtol=1e-5)
print(f"Export + Deserialize PASSED (M={M}, N={N})")
print(f"  Max error: {float(jnp.max(jnp.abs(c - c_ref))):.2e}")
```

### Exporting with symbolic shapes

With concrete shapes, the exported artifact only works for the exact dimensions it was traced with. **Symbolic shapes** lift this restriction — they let you export once and call with any compatible dimensions, without recompilation.

`export.symbolic_shape("a, b")` creates symbolic dimension variables. The exported function is parameterized over these variables, so the same serialized blob works for `(512, 256)`, `(1024, 1024)`, or any other shape.

```{code-cell} ipython3
# --- Export with symbolic shapes ---
a_sym, b_sym = export.symbolic_shape("a, b")
symbolic_shape_dtype = jax.ShapeDtypeStruct((a_sym, b_sym), jnp.float32)

print(
    f"Exporting with symbolic signature: ({symbolic_shape_dtype},"
    f" {symbolic_shape_dtype})"
)

exported_sym = jax.export.export(
    f, disabled_checks=get_export_disabled_safety_checks()
)
traced_sym = exported_sym(symbolic_shape_dtype, symbolic_shape_dtype)
blob_sym = traced_sym.serialize()
print(f"Serialized computation: {len(blob_sym):,} bytes")

rehydrated_sym = export.deserialize(blob_sym)

# Call with different shapes — no recompilation needed.
# The same serialized blob works for any (M, N) where M*N is a
# multiple of the kernel's block size (256).
for shape in [(512, 256), (1024, 512), (2048, 1024)]:
  a = jax.random.normal(next(keys), shape, dtype=jnp.float32)
  b = jax.random.normal(next(keys), shape, dtype=jnp.float32)
  c = rehydrated_sym.call(a, b)
  c_ref = ref_f(a, b)
  np.testing.assert_allclose(np.array(c), np.array(c_ref), rtol=1e-5)
  print(f"  Symbolic export PASSED for shape {shape}")

print("All symbolic shape tests passed.")
```

## Summary

In this notebook you learned to:

- Define GPU kernels in Python with **`@cute.kernel`** and **`@cute.jit`**
- Bridge CuTe DSL kernels into JAX via **`cutlass.jax.cutlass_call`**
- Pass both tensor and scalar arguments to custom kernels
- Write **ReLU** and **Fused Bias+ReLU** activation kernels for deep learning
- Demonstrate **kernel fusion** — combining multiple ops into a single GPU kernel
- Build a **tiled GEMM** kernel using CuTe DSL abstractions
- Distribute CUTLASS kernels across GPUs with **`jax.shard_map`**
- **Export and serialize** JAX functions containing CUTLASS kernels with **`jax.export`**

CuTe DSL is the right tool when you need direct control over tensor core matrix multiply-accumulate (MMA) instructions, shared memory layouts, and warp-level operations.
