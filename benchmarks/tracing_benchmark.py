# Copyright 2025 The JAX Authors.
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

"""Benchmarks for Jax tracing."""
import functools

import google_benchmark
import jax
from jax import random
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
import jax.numpy as jnp
import numpy as np


def clear_caches(state):
  state.pause_timing()
  jax.clear_caches()
  state.resume_timing()


def make_mqa_splash_attention_fn_and_args():
  seed = 0
  key = random.key(seed)
  k1, k2, k3 = random.split(key, 3)

  q_seq_len = 1024
  kv_seq_len = 1024
  num_q_heads = 2
  head_dim_qk = 128
  head_dim_v = 128
  dtype = np.dtype("float32")

  q = random.uniform(k1, (num_q_heads, q_seq_len, head_dim_qk), dtype=dtype)
  k = random.uniform(k2, (kv_seq_len, head_dim_qk), dtype=dtype)
  v = random.uniform(k3, (kv_seq_len, head_dim_v), dtype=dtype)

  mask = mask_lib.NumpyMask(
      mask_lib.make_random_mask((q_seq_len, kv_seq_len), sparsity=0.5, seed=0)
  )
  mask = mask_lib.MultiHeadMask(tuple(mask for _ in range(num_q_heads)))
  block_sizes = splash.BlockSizes.get_default()

  return (
      jax.jit(
          splash.make_splash_mqa_single_device(mask, block_sizes=block_sizes)
      )
  ), (q, k, v)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_pallas_mqa_splash_attention_trace(state):
  attn, (q, k, v) = make_mqa_splash_attention_fn_and_args()

  while state:
    _ = attn.trace(q, k, v)
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_pallas_mqa_splash_attention_trace_no_cache_clear(state):
  attn, (q, k, v) = make_mqa_splash_attention_fn_and_args()

  while state:
    _ = attn.trace(q, k, v)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_pallas_mqa_splash_attention_lower(state):
  attn, (q, k, v) = make_mqa_splash_attention_fn_and_args()
  traced = attn.trace(q, k, v)

  while state:
    _ = traced.lower(lowering_platforms=("tpu",))
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_pallas_mqa_splash_attention_lower_no_cache_clear(state):
  attn, (q, k, v) = make_mqa_splash_attention_fn_and_args()
  traced = attn.trace(q, k, v)

  while state:
    _ = traced.lower(lowering_platforms=("tpu",))


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_jnp_dot_trace(state):
  fn = jax.jit(jnp.dot)
  while state:
    _ = fn.trace(jnp.arange(1024), jnp.arange(1024))
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_jnp_dot_trace_no_cache_clear(state):
  fn = jax.jit(jnp.dot)
  while state:
    _ = fn.trace(jnp.arange(1024), jnp.arange(1024))


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_jnp_concat_trace(state):
  fn = jax.jit(functools.partial(jnp.concat, axis=0))
  while state:
    _ = fn.trace((jnp.ones((1024, 1)), jnp.ones((1024, 1))))
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_jnp_concat_trace_no_cache_clear(state):
  fn = jax.jit(functools.partial(jnp.concat, axis=0))
  while state:
    _ = fn.trace((jnp.ones((1024, 1)), jnp.ones((1024, 1))))


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
# NOTE(dsuo): Linear spacing so it's easier to eyeball historical plots.
@google_benchmark.option.arg(1)
@google_benchmark.option.dense_range(128, 896, 128)
def test_num_multiply_eqns_trace(state):
  fns = [lambda x: x * x for _ in range(state.range(0))]
  fn = jax.jit(functools.reduce(lambda a, b: (lambda x: a(b(x))), fns))
  while state:
    _ = fn.trace(jnp.ones((1024,)))
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
# NOTE(dsuo): Linear spacing so it's easier to eyeball historical plots.
@google_benchmark.option.arg(1)
@google_benchmark.option.dense_range(128, 896, 128)
def test_num_multiply_eqns_trace_no_cache_clear(state):
  fns = [lambda x: x * x for _ in range(state.range(0))]
  fn = jax.jit(functools.reduce(lambda a, b: (lambda x: a(b(x))), fns))
  while state:
    _ = fn.trace(jnp.ones((1024,)))


def _custom_root_function(depth):
  def root(a):
    def f(x):
      for _ in range(depth):
        x = x + 0.01 * jnp.sin(x)
      return x - a

    def solve(f, x):
      def step(state):
        i, x = state
        y, dy = jax.jvp(f, (x,), (jnp.ones_like(x),))
        return i + 1, x - y / dy
      return jax.lax.while_loop(lambda s: s[0] < 8, step, (0, x))[1]

    return jax.lax.custom_root(f, jnp.ones_like(a), solve,
                               lambda g, y: y / g(jnp.ones_like(y)))
  return root


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg(0)
@google_benchmark.option.arg(32)
@google_benchmark.option.arg(128)
def test_custom_root_trace(state):
  fn = jax.jit(_custom_root_function(state.range(0)))
  arg = jax.ShapeDtypeStruct((32,), np.float32)
  while state:
    # Include expansion to lojax so deferring work alone is not a speedup.
    _ = fn.trace(arg).lojax
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg(0)
@google_benchmark.option.arg(32)
@google_benchmark.option.arg(128)
def test_custom_root_grad_trace(state):
  root = _custom_root_function(state.range(0))
  fn = jax.jit(jax.grad(lambda a: root(a).sum()))
  arg = jax.ShapeDtypeStruct((32,), np.float32)
  while state:
    _ = fn.trace(arg).lojax
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg(0)
@google_benchmark.option.arg(32)
@google_benchmark.option.arg(128)
def test_custom_root_grad_eager(state):
  root = _custom_root_function(state.range(0))
  fn = jax.grad(lambda a: root(a).sum())
  arg = jnp.full((32,), 0.5, dtype=jnp.float32)
  jax.block_until_ready(fn(arg))
  while state:
    jax.block_until_ready(fn(arg))


def _trace_grad(state, loss, *args):
  fn = jax.jit(jax.grad(loss))
  while state:
    _ = fn.trace(*args).lojax
    clear_caches(state)


def _fem_compliance(n):
  """Nested-scan FEM assembly, per-row constraints, dense solve."""
  n_dof = (n + 1) ** 2
  # Element stiffness matrix [p, p + 1, q, q + 1], q = p + n + 1.
  ke_ref = jnp.array([[4., -1., -1., -2.], [-1., 4., -2., -1.],
                      [-1., -2., 4., -1.], [-2., -1., -1., 4.]]) / 6

  def add_slice(x, start, update):
    old = jax.lax.dynamic_slice(x, start, update.shape)
    return jax.lax.dynamic_update_slice(x, old + update, start)

  def compliance(theta):
    def element_row(carry, row):
      def element(carry, elem):
        (K, f), j, t = carry, *elem
        blocks = ((row[0] * (n + 1) + j, 0), ((row[0] + 1) * (n + 1) + j, 2))
        for r, a in blocks:
          for c, b in blocks:
            K = add_slice(K, (r, c), jnp.exp(t) * ke_ref[a:a + 2, b:b + 2])
          f = add_slice(f, (r,), jnp.full(2, 0.25 / n**2))
        return (K, f), None
      return jax.lax.scan(element, carry, (jnp.arange(n), row[1]))[0], None

    init = (jnp.zeros((n_dof, n_dof)), jnp.zeros(n_dof))
    K, f = jax.lax.scan(element_row, init, (jnp.arange(n), theta))[0]
    for i in range(0, n_dof, n + 1):
      K = K.at[i, :].set(0.0).at[:, i].set(0.0).at[i, i].set(1.0)
      f = f.at[i].set(0.0)
    C = jnp.zeros((n, n_dof))
    for k in range(1, n + 1):
      C = C.at[k - 1, k].set(1.0).at[k - 1, n * (n + 1) + k].set(-1.0)
    A = jnp.block([[K, C.T], [C, jnp.zeros((n, n))]])
    u = jnp.linalg.solve(A, jnp.concatenate([f, jnp.zeros(n)]))
    return f @ u[:n_dof]
  return compliance


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg(4)
@google_benchmark.option.arg(16)
@google_benchmark.option.arg(32)
def test_fem_assembly_constrained_grad_trace(state):
  n = state.range(0)
  theta = jax.ShapeDtypeStruct((n, n), np.float32)
  _trace_grad(state, _fem_compliance(n), theta)


def _galerkin_rom_loss(n_steps, dt=1e-2):
  """Einsums interleaved with nonlinear ops."""
  def loss(params, a):
    phi, w, u1, u2, u3, core = params

    def step(a, _):
      u = jnp.einsum("qn,bn->bq", phi, a)
      proj = jnp.einsum("q,qn,bq->bn", w, phi, jnp.tanh(u) * u)
      triad = jnp.einsum("ix,jy,kz,xyz,bj,bk->bi", u1, u2, u3, core, a, a)
      jac = jnp.einsum("q,qi,qj,bq->bij", w, phi, phi, 1 - jnp.tanh(u)**2)
      damping = jax.nn.softplus(jnp.einsum("bi,bij,bj->b", a, jac, a)
                                - jnp.einsum("bii->b", jac))
      return a + dt * (proj + triad - damping[:, None] * a), None

    return jnp.sum(jax.lax.scan(step, a, length=n_steps)[0] ** 2)
  return loss


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_einsum_galerkin_rom_grad_trace(state):
  b, n, q, r = 16, 24, 32, 6
  spec = lambda *shape: jax.ShapeDtypeStruct(shape, np.float32)
  params = (spec(q, n), spec(q), spec(n, r), spec(n, r), spec(n, r),
            spec(r, r, r))
  _trace_grad(state, _galerkin_rom_loss(100), params, spec(b, n))


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg(1)
@google_benchmark.option.arg(64)
@google_benchmark.option.arg(512)
def test_tree_stack_grad_trace(state):
  leaf = lambda *shape: jax.ShapeDtypeStruct(shape, np.float32)
  tree = {
      "mass": leaf(),
      "state": {"position": leaf(3), "velocity": leaf(3),
                "modes": (leaf(8, 3), leaf(8, 3))},
      "fields": [leaf(), leaf(), leaf(4)],
  }

  def loss(trees):
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *trees)
    return sum(jnp.sum(x**2) for x in jax.tree.leaves(stacked))
  _trace_grad(state, loss, [tree] * state.range(0))

if __name__ == "__main__":
  google_benchmark.main()
