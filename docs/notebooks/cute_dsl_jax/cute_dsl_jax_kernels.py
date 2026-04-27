# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import cuda.bindings.driver as cuda  # pyrefly: ignore

"""
CuTe DSL kernels used by the ``cute_dsl_jax.ipynb`` notebook.

This module defines GPU kernels written in CuTe DSL (CUTLASS 4.x Python DSL)
that are called from JAX via ``cutlass.jax.cutlass_call``. ``cutlass_call`` is a
JAX primitive that triggers compilation of the kernel during lowering and embeds
it into the HLO computation, so XLA can launch it efficiently without callback
to Python.

Kernels provided:

- ``vector_add``        — element-wise c = a + b (3-D CuTe layout)
- ``saxpy``             — y = alpha * x + y
- ``relu``              — element-wise ReLU with flat indexing
- ``fused_bias_relu``   — fused bias addition + ReLU
- ``gemm``              — tiled matrix multiplication
- ``elementwise_add``   — 2-D element-wise add (flat indexing, ``jax.export``-compatible)

The notebook imports these kernels and wraps each one with ``cutlass_call``
inside ``@jax.jit`` functions. See ``cute_dsl_jax.ipynb`` for usage, validation,
and step-by-step explanations.

This module is imported by the notebook and by ``cute_dsl_jax_kernels.py``. It can also
be run directly to validate every kernel:

.. code-block:: bash

    # Interactive notebook (recommended for learning)
    jupyter lab cute_dsl_jax.ipynb

    # Full demo as a standalone script
    python cute_dsl_jax_kernels.py
"""


# ------------------------------------------------------------------ #
#  Vector Add: c = a + b                                             #
# ------------------------------------------------------------------ #
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


@cute.jit
def launch_vector_add(
    stream: cuda.CUstream,
    a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
):
    vector_add_kernel(a, b, c).launch(
        grid=[a.shape[-1], 1, 1],  # pyrefly: ignore
        block=[a.shape[-2], 1, 1],  # pyrefly: ignore
        stream=stream,
    )


# ------------------------------------------------------------------ #
#  SAXPY: y = alpha * x + y                                          #
# ------------------------------------------------------------------ #
@cute.kernel
def saxpy_kernel(x: cute.Tensor, y: cute.Tensor, out: cute.Tensor, alpha: float):
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


@cute.jit
def launch_saxpy(
    stream: cuda.CUstream,
    x: cute.Tensor, y: cute.Tensor, out: cute.Tensor,
    *, alpha: float,
):
    saxpy_kernel(x, y, out, alpha).launch(
        grid=[x.shape[-1], 1, 1],  # pyrefly: ignore
        block=[x.shape[-2], 1, 1],  # pyrefly: ignore
        stream=stream,
    )


# ------------------------------------------------------------------ #
#  ReLU: out = max(0, x)                                             #
# ------------------------------------------------------------------ #
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


@cute.jit
def launch_relu(
    stream: cuda.CUstream,
    x: cute.Tensor, out: cute.Tensor,
    *, N: int,
):
    BLOCK_SIZE = 256
    grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    relu_kernel(x, out, N).launch(
        grid=[grid_size, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


# ------------------------------------------------------------------ #
#  Fused Bias + ReLU: out = max(0, x + bias[col])                    #
# ------------------------------------------------------------------ #
@cute.kernel
def fused_bias_relu_kernel(
    x: cute.Tensor, bias: cute.Tensor, out: cute.Tensor, N: int, width: int,
):
    """Per-thread: out[i] = max(0, x[i] + bias[i % width])."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()

    idx = bidx * bdx + tidx
    if idx < N:
        col = idx % width
        val = x[idx] + bias[col]  # pyrefly: ignore
        out[idx] = cutlass.max(val, cutlass.Float32(0.0))


@cute.jit
def launch_fused_bias_relu(
    stream: cuda.CUstream,
    x: cute.Tensor, bias: cute.Tensor, out: cute.Tensor,
    *, N: int, width: int,
):
    BLOCK_SIZE = 256
    grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_bias_relu_kernel(x, bias, out, N, width).launch(
        grid=[grid_size, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


# ------------------------------------------------------------------ #
#  GEMM: D = A @ B                                                   #
# ------------------------------------------------------------------ #
@cute.kernel
def gemm_kernel(
    A: cute.Tensor, B: cute.Tensor, D: cute.Tensor,
    M: int, N: int, K: int, BLOCK_M: int, BLOCK_N: int,
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
                acc += A[m_idx * K + k] * B[k * N + n_idx]  # pyrefly: ignore
            D[m_idx * N + n_idx] = acc


@cute.jit
def launch_gemm(
    stream: cuda.CUstream,
    A: cute.Tensor, B: cute.Tensor, D: cute.Tensor,
    *, M: int, N: int, K: int,
):
  BLOCK_M, BLOCK_N = 64, 64
  grid_m = (M + BLOCK_M - 1) // BLOCK_M
  grid_n = (N + BLOCK_N - 1) // BLOCK_N
  gemm_kernel(A, B, D, M, N, K, BLOCK_M, BLOCK_N).launch(
        grid=[grid_m, grid_n, 1],
        block=[256, 1, 1],
        stream=stream,
    )


# ------------------------------------------------------------------ #
#  Element-wise Add (2-D, flat indexing)                             #
# ------------------------------------------------------------------ #
@cute.kernel
def elementwise_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
  """Per-thread kernel: 2-D element-wise add using flat indexing."""
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()
  bdim, _, _ = cute.arch.block_dim()

  thread_idx = bidx * bdim + tidx

  m, n = gA.shape  # pyrefly: ignore

  if thread_idx < m * n:  # pyrefly: ignore

    ni = thread_idx % n
    mi = thread_idx // n

    a_val = gA[mi, ni]
    b_val = gB[mi, ni]
    gC[mi, ni] = a_val + b_val  # pyrefly: ignore


@cute.jit
def launch_elementwise_add(
    stream: cuda.CUstream,
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
):
    num_threads_per_block = 256
    m, n = mA.shape  # pyrefly: ignore
    elementwise_add_kernel(mA, mB, mC).launch(
        grid=((m * n + num_threads_per_block - 1) // num_threads_per_block, 1, 1),  # pyrefly: ignore
        block=(num_threads_per_block, 1, 1),
        stream=stream,
    )


# ------------------------------------------------------------------ #
#  Self-tests                                                         #
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import jax
    import jax.numpy as jnp
    import numpy as np

    def split_keys(seed=0):
      key = jax.random.key(seed)
      while True:
        key, subkey = jax.random.split(key)
        yield subkey

    keys = iter(split_keys())

    BLOCK = 256
    N_BLOCKS = 4

    # ── Vector Add ────────────────────────────────────────────────────
    # 3-D CuTe layout: (elems_per_thread, threads_per_block, num_blocks)
    a = jax.random.normal(next(keys), (1, BLOCK, N_BLOCKS), dtype=jnp.float32)
    b = jax.random.normal(next(keys), (1, BLOCK, N_BLOCKS), dtype=jnp.float32)
    call = cjax.cutlass_call(
        launch_vector_add,
        output_shape_dtype=jax.ShapeDtypeStruct(a.shape, a.dtype),
        use_static_tensors=True,
    )
    c = jax.jit(call)(a, b)
    np.testing.assert_allclose(np.array(c), np.array(a + b), rtol=1e-5, atol=1e-5)
    print('vector_add:       PASSED')

    # ── SAXPY ─────────────────────────────────────────────────────────
    x = jax.random.normal(next(keys), (1, BLOCK, N_BLOCKS), dtype=jnp.float32)
    y = jax.random.normal(next(keys), (1, BLOCK, N_BLOCKS), dtype=jnp.float32)
    alpha = 2.5
    call = cjax.cutlass_call(
        launch_saxpy,
        output_shape_dtype=jax.ShapeDtypeStruct(x.shape, x.dtype),
        use_static_tensors=True,
        alpha=alpha,
    )
    out = jax.jit(call)(x, y)
    np.testing.assert_allclose(np.array(out), np.array(alpha * x + y), rtol=1e-5, atol=1e-5)
    print('saxpy:            PASSED')

    # ── ReLU ──────────────────────────────────────────────────────────
    N_ELEM = BLOCK * N_BLOCKS
    x = jax.random.normal(next(keys), (N_ELEM,), dtype=jnp.float32)
    call = cjax.cutlass_call(
        launch_relu,
        output_shape_dtype=jax.ShapeDtypeStruct(x.shape, x.dtype),
        N=N_ELEM,
    )
    out = jax.jit(call)(x)
    np.testing.assert_allclose(np.array(out), np.array(jnp.maximum(x, 0)), rtol=1e-5, atol=1e-5)
    print('relu:             PASSED')

    # ── Fused Bias + ReLU ─────────────────────────────────────────────
    ROWS, COLS = 16, 64
    x = jax.random.normal(next(keys), (ROWS * COLS,), dtype=jnp.float32)
    bias = jax.random.normal(next(keys), (COLS,), dtype=jnp.float32)
    call = cjax.cutlass_call(
        launch_fused_bias_relu,
        output_shape_dtype=jax.ShapeDtypeStruct(x.shape, x.dtype),
        N=ROWS * COLS, width=COLS,
    )
    out = jax.jit(call)(x, bias)
    ref = jnp.maximum(x.reshape(ROWS, COLS) + bias, 0).reshape(-1)
    np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-5, atol=1e-5)
    print('fused_bias_relu:  PASSED')

    # ── GEMM ──────────────────────────────────────────────────────────
    M, N, K = 128, 128, 64
    A = jax.random.normal(next(keys), (M * K,), dtype=jnp.float32)
    B = jax.random.normal(next(keys), (K * N,), dtype=jnp.float32)
    call = cjax.cutlass_call(
        launch_gemm,
        output_shape_dtype=jax.ShapeDtypeStruct((M * N,), A.dtype),
        M=M, N=N, K=K,
    )
    D = jax.jit(call)(A, B)
    ref = A.reshape(M, K) @ B.reshape(K, N)
    np.testing.assert_allclose(np.array(D.reshape(M, N)), np.array(ref), rtol=1e-2, atol=1e-2)
    print('gemm:             PASSED')

    # ── Elementwise Add (2-D) ─────────────────────────────────────────
    M, N = 16, 256
    a = jax.random.normal(next(keys), (M, N), dtype=jnp.float32)
    b = jax.random.normal(next(keys), (M, N), dtype=jnp.float32)
    call = cjax.cutlass_call(
        launch_elementwise_add,
        output_shape_dtype=jax.ShapeDtypeStruct(a.shape, a.dtype),
    )
    c = jax.jit(call)(a, b)
    np.testing.assert_allclose(np.array(c), np.array(a + b), rtol=1e-5, atol=1e-5)
    print('elementwise_add:  PASSED')

    print('\nAll kernels passed.')
