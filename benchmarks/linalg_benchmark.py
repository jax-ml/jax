# Copyright 2020 The JAX Authors.
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

"""Benchmarks for JAX linear algebra functions."""

import functools
import google_benchmark
import jax
import jax.lax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

SQUARE_CASES = [
    (1, 100, np.float32),
    (100, 100, np.float32),
    (1, 1000, np.float32),
    (1000, 10, np.float32),
    (1, 100, np.complex64),
]

NON_SQUARE_CASES = [
    (1, 100, 500, np.float32),
    (10, 100, 500, np.float32),
    (1, 500, 100, np.float32),
    (1, 100, 500, np.complex64),
]


def _rand_matrix(shape, dtype):
  x = np.random.randn(*shape).astype(dtype)
  if np.issubdtype(dtype, np.complexfloating):
    x += 1j * np.random.randn(*shape).astype(dtype)
  return x


def svd_benchmark(b, m, n, dtype, state):
  x = _rand_matrix((b, m, n), dtype)
  run_fn = jax.jit(jnp.linalg.svd)
  jax.block_until_ready(run_fn(x))
  while state:
    jax.block_until_ready(run_fn(x))


for b, m, n, dtype in NON_SQUARE_CASES:
  google_benchmark.register(
      google_benchmark.option.measure_process_cpu_time()(
          functools.partial(svd_benchmark, b, m, n, dtype)
      ),
      name=f"svd_{b}_{m}_{n}_{dtype.__name__}",
  )


def qr_benchmark(b, m, n, dtype, state):
  x = _rand_matrix((b, m, n), dtype)
  run_fn = jax.jit(jnp.linalg.qr)
  jax.block_until_ready(run_fn(x))
  while state:
    jax.block_until_ready(run_fn(x))


for b, m, n, dtype in NON_SQUARE_CASES:
  google_benchmark.register(
      google_benchmark.option.measure_process_cpu_time()(
          functools.partial(qr_benchmark, b, m, n, dtype)
      ),
      name=f"qr_{b}_{m}_{n}_{dtype.__name__}",
  )


def lu_benchmark(b, m, n, dtype, state):
  x = _rand_matrix((b, m, n), dtype)
  run_fn = jax.jit(jax.vmap(jax.scipy.linalg.lu))
  jax.block_until_ready(run_fn(x))
  while state:
    jax.block_until_ready(run_fn(x))


for b, m, n, dtype in NON_SQUARE_CASES:
  google_benchmark.register(
      google_benchmark.option.measure_process_cpu_time()(
          functools.partial(lu_benchmark, b, m, n, dtype)
      ),
      name=f"lu_{b}_{m}_{n}_{dtype.__name__}",
  )


def cholesky_benchmark(b, n, dtype, state):
  x = _rand_matrix((b, n, n), dtype)
  x = np.matmul(x, x.transpose(0, 2, 1).conj()) + 0.05 * np.eye(n, dtype=dtype)
  run_fn = jax.jit(jnp.linalg.cholesky)
  jax.block_until_ready(run_fn(x))
  while state:
    jax.block_until_ready(run_fn(x))


for b, n, dtype in SQUARE_CASES:
  google_benchmark.register(
      google_benchmark.option.measure_process_cpu_time()(
          functools.partial(cholesky_benchmark, b, n, dtype)
      ),
      name=f"cholesky_{b}_{n}_{dtype.__name__}",
  )


def solve_triangular_benchmark(b, n, dtype, state):
  a = np.triu(_rand_matrix((b, n, n), dtype))
  b_vec = _rand_matrix((b, n, n), dtype)

  run_fn = jax.jit(jax.scipy.linalg.solve_triangular)
  jax.block_until_ready(run_fn(a, b_vec))
  while state:
    jax.block_until_ready(run_fn(a, b_vec))


for b, n, dtype in SQUARE_CASES:
  google_benchmark.register(
      google_benchmark.option.measure_process_cpu_time()(
          functools.partial(solve_triangular_benchmark, b, n, dtype)
      ),
      name=f"solve_triangular_{b}_{n}_{dtype.__name__}",
  )


def tridiagonal_solve_benchmark(b, n, dtype, state):
  dl = _rand_matrix((b, n), dtype)
  d = _rand_matrix((b, n), dtype) + 2.0
  du = _rand_matrix((b, n), dtype)
  b_vec = _rand_matrix((b, n, 1), dtype)

  run_fn = jax.jit(jax.lax.linalg.tridiagonal_solve)
  jax.block_until_ready(run_fn(dl, d, du, b_vec))
  while state:
    jax.block_until_ready(run_fn(dl, d, du, b_vec))


for b, n, dtype in SQUARE_CASES:
  google_benchmark.register(
      google_benchmark.option.measure_process_cpu_time()(
          functools.partial(tridiagonal_solve_benchmark, b, n, dtype)
      ),
      name=f"tridiagonal_solve_{b}_{n}_{dtype.__name__}",
  )


def eigh_benchmark(b, n, dtype, state):
  x = _rand_matrix((b, n, n), dtype)
  x = x + x.transpose(0, 2, 1).conj()
  run_fn = jax.jit(jnp.linalg.eigh)
  jax.block_until_ready(run_fn(x))
  while state:
    jax.block_until_ready(run_fn(x))


for b, n, dtype in SQUARE_CASES:
  google_benchmark.register(
      google_benchmark.option.measure_process_cpu_time()(
          functools.partial(eigh_benchmark, b, n, dtype)
      ),
      name=f"eigh_{b}_{n}_{dtype.__name__}",
  )


def schur_benchmark(b, n, dtype, state):
  x = _rand_matrix((b, n, n), dtype)
  run_fn = jax.jit(jax.vmap(jax.scipy.linalg.schur))
  jax.block_until_ready(run_fn(x))
  while state:
    jax.block_until_ready(run_fn(x))


for b, n, dtype in SQUARE_CASES:
  google_benchmark.register(
      google_benchmark.option.measure_process_cpu_time()(
          functools.partial(schur_benchmark, b, n, dtype)
      ),
      name=f"schur_{b}_{n}_{dtype.__name__}",
  )


def hessenberg_benchmark(b, n, dtype, state):
  x = _rand_matrix((b, n, n), dtype)
  run_fn = jax.jit(jax.vmap(jax.scipy.linalg.hessenberg))
  jax.block_until_ready(run_fn(x))
  while state:
    jax.block_until_ready(run_fn(x))


for b, n, dtype in SQUARE_CASES:
  google_benchmark.register(
      google_benchmark.option.measure_process_cpu_time()(
          functools.partial(hessenberg_benchmark, b, n, dtype)
      ),
      name=f"hessenberg_{b}_{n}_{dtype.__name__}",
  )

if __name__ == "__main__":
  google_benchmark.main()
