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
"""Tests and benchmarks for N-GPU Ring Collective Matmul using `thunky`."""

import functools
import os
import time

if "--xla_gpu_enable_command_buffer" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_gpu_enable_command_buffer=+COLLECTIVES"
  ).strip()

from absl.testing import absltest
import jax
from jax import shard_map
from jax._src import config
from jax._src import test_util as jtu
from jax.experimental import thunky
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

config.parse_flags_with_absl()

try:
  from jax.experimental.pallas.ops.gpu import collective_matmul_mgpu
except ImportError:
  collective_matmul_mgpu = None


def make_ring_collective_matmul(mesh, m_block, k_dim, n_block, dtype):
  num_devices = mesh.size
  # Unidirectional ring: rank i sends to rank (i + 1) % N
  cw_pairs = tuple((i, (i + 1) % num_devices) for i in range(num_devices))

  # Double-buffered ping-pong workspace: at most 2 buffers of shape (m_block, k_dim)
  num_bufs = min(2, num_devices - 1)
  scratch_specs = tuple(
      jax.ShapeDtypeStruct((m_block, k_dim), dtype) for _ in range(num_bufs)
  )

  def matmul_scatter(lhs_ref, rhs_ref, dev_id_ref, c_out_ref, *, step):
    c_block = jnp.matmul(lhs_ref[...], rhs_ref[...])
    row_offset = ((dev_id_ref[...] - step) % num_devices) * m_block
    c_out_ref[...] = jax.lax.dynamic_update_slice(
        c_out_ref[...], c_block, (row_offset, jnp.int32(0))
    )

  @thunky.jit(scratch_shapes=scratch_specs)
  def ring_collective_matmul_prog(
      a_local_buf,
      b_local_buf,
      dev_id_buf,
      c_out_buf,
      *ring_bufs,
  ):
    tok_comm = {}

    def launch_shift(step):
      src = a_local_buf if step == 1 else ring_bufs[(step - 2) % num_bufs]
      dst = ring_bufs[(step - 1) % num_bufs]
      tok_comm[step] = thunky.async_start(
          lambda s_buf=src, d_buf=dst: thunky.collective_permute(
              s_buf, d_buf, source_target_pairs=cw_pairs
          ),
          stream_id=0,
          stream_kind="communication",
      )

    # Prologue: enqueue up to 2 initial shifts on communication stream 0
    for s in range(1, min(3, num_devices)):
      launch_shift(s)

    # Step 0: compute local GEMM 0 (A_d @ B_d) on compute stream 0 concurrently
    # with the initial ring transfers.
    thunky.call_jax(
        functools.partial(matmul_scatter, step=0),
        a_local_buf,
        b_local_buf,
        dev_id_buf,
        c_out_buf,
    )

    # Steady state: wait for shift s, compute GEMM s on buf[(s - 1) % 2],
    # and enqueue shift s + 2 into the newly freed buffer.
    for s in range(1, num_devices):
      thunky.async_done(tok_comm[s])
      thunky.call_jax(
          functools.partial(matmul_scatter, step=s),
          ring_bufs[(s - 1) % num_bufs],
          b_local_buf,
          dev_id_buf,
          c_out_buf,
      )
      if s + 2 < num_devices:
        launch_shift(s + 2)

  @jax.jit
  @shard_map(
      mesh=mesh,
      in_specs=(P("x", None), P(None, "x")),
      out_specs=P(None, "x"),
      check_vma=False,
  )
  def ring_collective_matmul_fn(a_local, b_local):
    dev_id = jnp.int32(jax.lax.axis_index("x"))
    c_out_ref = jax.new_ref(
        jax.lax.empty((m_block * num_devices, n_block), dtype=dtype)
    )
    ring_collective_matmul_prog(a_local, b_local, dev_id, c_out_ref)
    return c_out_ref[...]

  return ring_collective_matmul_fn


def make_bidirectional_ring_collective_matmul(
    mesh, m_block, k_dim, n_block, dtype
):
  num_devices = mesh.size
  num_pairs = (num_devices - 1) // 2
  has_mid = (num_devices % 2) == 0

  cw_pairs = tuple((i, (i + 1) % num_devices) for i in range(num_devices))
  ccw_pairs = tuple((i, (i - 1) % num_devices) for i in range(num_devices))

  # Double-buffered ping-pong workspace: at most 2 buffers of shape (2 * m_block, k_dim)
  # (or 1 buffer of shape (m_block, k_dim) when num_devices == 2).
  if num_pairs > 0:
    num_bufs = min(2, num_pairs)
    scratch_specs = tuple(
        jax.ShapeDtypeStruct((2 * m_block, k_dim), dtype)
        for _ in range(num_bufs)
    )
  else:
    num_bufs = 1
    scratch_specs = (jax.ShapeDtypeStruct((m_block, k_dim), dtype),)

  def single_matmul_scatter(lhs_ref, rhs_ref, dev_id_ref, c_out_ref, *, step):
    c_block = jnp.matmul(lhs_ref[...], rhs_ref[...])
    row_offset = ((dev_id_ref[...] - step) % num_devices) * m_block
    c_out_ref[...] = jax.lax.dynamic_update_slice(
        c_out_ref[...], c_block, (row_offset, jnp.int32(0))
    )

  def pair_matmul_scatter(
      pair_lhs_ref, rhs_ref, dev_id_ref, c_out_ref, *, step
  ):
    # Single cuBLAS GEMM of shape (2 * m_block, k_dim) @ (k_dim, n_block)
    c_pair = jnp.matmul(pair_lhs_ref[...], rhs_ref[...])
    row_cw = ((dev_id_ref[...] - step) % num_devices) * m_block
    row_ccw = ((dev_id_ref[...] + step) % num_devices) * m_block
    c_out = jax.lax.dynamic_update_slice(
        c_out_ref[...], c_pair[:m_block], (row_cw, jnp.int32(0))
    )
    c_out_ref[...] = jax.lax.dynamic_update_slice(
        c_out, c_pair[m_block:], (row_ccw, jnp.int32(0))
    )

  @thunky.jit(scratch_shapes=scratch_specs)
  def bidir_ring_collective_matmul_prog(
      a_local_buf,
      b_local_buf,
      dev_id_buf,
      c_out_buf,
      *pair_bufs,
  ):
    tok_comm = {}
    total_comm_steps = num_pairs + (1 if has_mid else 0)

    def launch_comm_step(step):
      if step <= num_pairs:
        src_cw = (
            a_local_buf
            if step == 1
            else pair_bufs[(step - 2) % num_bufs].at[:m_block]
        )
        dst_cw = pair_bufs[(step - 1) % num_bufs].at[:m_block]
        src_ccw = (
            a_local_buf
            if step == 1
            else pair_bufs[(step - 2) % num_bufs].at[m_block:]
        )
        dst_ccw = pair_bufs[(step - 1) % num_bufs].at[m_block:]

        def comm_pair(scw=src_cw, dcw=dst_cw, sccw=src_ccw, dccw=dst_ccw):
          thunky.collective_group(
              lambda: (
                  thunky.collective_permute(
                      scw, dcw, source_target_pairs=cw_pairs
                  ),
                  thunky.collective_permute(
                      sccw, dccw, source_target_pairs=ccw_pairs
                  ),
              )
          )

        tok_comm[step] = thunky.async_start(
            comm_pair,
            stream_id=0,
            stream_kind="communication",
        )
      else:
        # Middle step (for even N): single clockwise transfer into top half of ping-pong buffer
        src_mid = (
            pair_bufs[(num_pairs - 1) % num_bufs].at[:m_block]
            if num_pairs > 0
            else a_local_buf
        )
        dst_mid = (
            pair_bufs[(step - 1) % num_bufs].at[:m_block]
            if num_pairs > 0
            else pair_bufs[0]
        )
        tok_comm[step] = thunky.async_start(
            lambda s_buf=src_mid, d_buf=dst_mid: thunky.collective_permute(
                s_buf, d_buf, source_target_pairs=cw_pairs
            ),
            stream_id=0,
            stream_kind="communication",
        )

    # Prologue: enqueue up to 2 initial communication steps
    for s in range(1, min(3, total_comm_steps + 1)):
      launch_comm_step(s)

    # Step 0: compute local GEMM 0 on compute stream 0
    thunky.call_jax(
        functools.partial(single_matmul_scatter, step=0),
        a_local_buf,
        b_local_buf,
        dev_id_buf,
        c_out_buf,
    )

    # Steady state: wait for step s, run GEMM s, enqueue step s + 2
    for s in range(1, total_comm_steps + 1):
      thunky.async_done(tok_comm[s])
      if s <= num_pairs:
        thunky.call_jax(
            functools.partial(pair_matmul_scatter, step=s),
            pair_bufs[(s - 1) % num_bufs],
            b_local_buf,
            dev_id_buf,
            c_out_buf,
        )
      else:
        mid_buf = (
            pair_bufs[(s - 1) % num_bufs].at[:m_block]
            if num_pairs > 0
            else pair_bufs[0]
        )
        thunky.call_jax(
            functools.partial(single_matmul_scatter, step=s),
            mid_buf,
            b_local_buf,
            dev_id_buf,
            c_out_buf,
        )
      if s + 2 <= total_comm_steps:
        launch_comm_step(s + 2)

  @jax.jit
  @shard_map(
      mesh=mesh,
      in_specs=(P("x", None), P(None, "x")),
      out_specs=P(None, "x"),
      check_vma=False,
  )
  def bidir_ring_collective_matmul_fn(a_local, b_local):
    dev_id = jnp.int32(jax.lax.axis_index("x"))
    c_out_ref = jax.new_ref(
        jax.lax.empty((m_block * num_devices, n_block), dtype=dtype)
    )
    bidir_ring_collective_matmul_prog(a_local, b_local, dev_id, c_out_ref)
    return c_out_ref[...]

  return bidir_ring_collective_matmul_fn


class ThunkyCollectiveMatmulTest(jtu.JaxTestCase):

  def test_ring_collective_matmul_correctness(self):
    """Verifies numerical correctness of the Unidirectional Ring Collective Matmul."""
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest("Requires >= 2 GPUs")

    dtype = jnp.bfloat16
    m_block, k_dim, n_block = 512, 1024, 512
    mesh = Mesh(np.array(jax.devices()), ("x",))

    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(
        (m_block * num_devices, k_dim), dtype=np.float32
    ).astype(dtype)
    b_np = rng.standard_normal(
        (k_dim, n_block * num_devices), dtype=np.float32
    ).astype(dtype)

    a = jax.device_put(a_np, jax.sharding.NamedSharding(mesh, P("x", None)))
    b = jax.device_put(b_np, jax.sharding.NamedSharding(mesh, P(None, "x")))
    expected = a_np @ b_np

    plain_ring_fn = make_ring_collective_matmul(
        mesh, m_block, k_dim, n_block, dtype
    )
    out_plain = plain_ring_fn(a, b)
    np.testing.assert_allclose(
        np.asarray(out_plain, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=2e-2,
        atol=2e-1,
    )

    compiled = plain_ring_fn.lower(a, b).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_bidirectional_ring_collective_matmul_correctness(self):
    """Verifies numerical correctness of the Bidirectional Ring Collective Matmul."""
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest("Requires >= 2 GPUs")

    dtype = jnp.bfloat16
    m_block, k_dim, n_block = 512, 1024, 512
    mesh = Mesh(np.array(jax.devices()), ("x",))

    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(
        (m_block * num_devices, k_dim), dtype=np.float32
    ).astype(dtype)
    b_np = rng.standard_normal(
        (k_dim, n_block * num_devices), dtype=np.float32
    ).astype(dtype)

    a = jax.device_put(a_np, jax.sharding.NamedSharding(mesh, P("x", None)))
    b = jax.device_put(b_np, jax.sharding.NamedSharding(mesh, P(None, "x")))
    expected = a_np @ b_np

    bidir_ring_fn = make_bidirectional_ring_collective_matmul(
        mesh, m_block, k_dim, n_block, dtype
    )
    out_bidir = bidir_ring_fn(a, b)
    np.testing.assert_allclose(
        np.asarray(out_bidir, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=2e-2,
        atol=2e-1,
    )

    compiled = bidir_ring_fn.lower(a, b).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_benchmark_collective_matmul(self):
    """Benchmarks Thunky Ring Collective Matmul against XLA:GPU and Mosaic:GPU."""
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest("Requires >= 2 GPUs")

    dtype = jnp.bfloat16
    mesh_exp = jax.make_mesh(
        (num_devices,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    mesh_std = Mesh(np.array(jax.devices()), ("x",))

    sizes = [
        (512, 4096, 512),
        (1024, 4096, 1024),
        (2048, 4096, 2048),
        (4096, 4096, 4096),
    ]

    print("\n" + "=" * 95)
    print(
        f"{'M_shard x K x N_shard':<22} | {'Method':<28} | {'Latency (us)':<14}"
        f" | {'TFLOPS/GPU':<12}"
    )
    print("-" * 95)

    for m_block, k_dim, n_block in sizes:
      flops_per_gpu = 2.0 * (m_block * num_devices) * k_dim * n_block

      rng = np.random.default_rng(42)
      a_np = rng.standard_normal(
          (m_block * num_devices, k_dim), dtype=np.float32
      ).astype(dtype)
      b_np = rng.standard_normal(
          (k_dim, n_block * num_devices), dtype=np.float32
      ).astype(dtype)

      a_std = jax.device_put(
          a_np, jax.sharding.NamedSharding(mesh_std, P("x", None))
      )
      b_std = jax.device_put(
          b_np, jax.sharding.NamedSharding(mesh_std, P(None, "x"))
      )

      a_exp = jax.device_put(
          a_np, jax.sharding.NamedSharding(mesh_exp, P("x", None))
      )
      b_exp = jax.device_put(
          b_np, jax.sharding.NamedSharding(mesh_exp, P(None, "x"))
      )

      def bench(fn, lhs, rhs, name, iters=20):
        for _ in range(5):
          out = fn(lhs, rhs)
        jax.block_until_ready(out)
        t0 = time.perf_counter()
        for _ in range(iters):
          out = fn(lhs, rhs)
        jax.block_until_ready(out)
        dt = (time.perf_counter() - t0) / iters
        us = dt * 1e6
        tflops = (flops_per_gpu / dt) / 1e12
        shape_str = f"{m_block}x{k_dim}x{n_block}"
        print(f"{shape_str:<22} | {name:<28} | {us:12.1f} us | {tflops:10.2f}")
        return us, tflops, out

      # 1. XLA:GPU Explicit sharding context
      with jax.set_mesh(mesh_exp):

        @jax.jit
        def xla_explicit_fn(lhs, rhs):
          return jnp.matmul(lhs, rhs, out_sharding=P(None, "x"))

        bench(xla_explicit_fn, a_exp, b_exp, "XLA:GPU (Explicit matmul)")

      # 2. XLA:GPU shard_map (all_gather + matmul)
      @jax.jit
      @shard_map(
          mesh=mesh_std,
          in_specs=(P("x", None), P(None, "x")),
          out_specs=P(None, "x"),
          check_vma=False,
      )
      def xla_ag_fn(lhs, rhs):
        return jax.lax.all_gather(lhs, "x", axis=0, tiled=True) @ rhs

      bench(xla_ag_fn, a_std, b_std, "XLA:GPU (shmap AllGather)")

      # 3. Mosaic:GPU collective matmul
      if collective_matmul_mgpu is not None:
        config = collective_matmul_mgpu.TuningConfig(
            tile_m=128,
            tile_n=128,
            tile_k=64,
            max_concurrent_steps=2,
            grid_minor_dim=collective_matmul_mgpu.MatmulDimension.N,
            grid_tile_width=1,
            wg_dimension=collective_matmul_mgpu.MatmulDimension.N,
        )
        with jax.set_mesh(mesh_exp):

          @jax.jit
          @shard_map(
              mesh=mesh_exp,
              in_specs=(P("x", None), P(None, "x")),
              out_specs=P(None, "x"),
              check_vma=False,
          )
          def mgpu_fn(lhs, rhs):
            return collective_matmul_mgpu.all_gather_lhs_matmul(
                lhs, rhs, axis_name="x", config=config, dtype=dtype
            )

          try:
            bench(mgpu_fn, a_exp, b_exp, "Mosaic:GPU (all_gather_lhs)")
          except Exception as e:
            print(
                f"{m_block}x{k_dim}x{n_block:<12} |"
                f" {'Mosaic:GPU (all_gather_lhs)':<28} | SKIPPED ({e})"
            )

      # 4. Thunky Plain Unidirectional Ring Collective Matmul
      plain_ring_fn = make_ring_collective_matmul(
          mesh_std, m_block, k_dim, n_block, dtype
      )
      bench(plain_ring_fn, a_std, b_std, "Thunky (Plain Ring MM)")

      # 5. Thunky Bidirectional Ring Collective Matmul
      bidir_ring_fn = make_bidirectional_ring_collective_matmul(
          mesh_std, m_block, k_dim, n_block, dtype
      )
      bench(bidir_ring_fn, a_std, b_std, "Thunky (Bidir Ring MM)")

      profile_dir = os.environ.get("THUNKY_PROFILE_DIR")
      if profile_dir and m_block == 4096:
        plain_dir = profile_dir + "_plain"
        bidir_dir = profile_dir + "_bidir"
        with jax.profiler.trace(plain_dir):
          for _ in range(10):
            out = plain_ring_fn(a_std, b_std)
          jax.block_until_ready(out)
        print(f"Saved plain ring xprof trace to {plain_dir}")
        with jax.profiler.trace(bidir_dir):
          for _ in range(10):
            out = bidir_ring_fn(a_std, b_std)
          jax.block_until_ready(out)
        print(f"Saved bidir ring xprof trace to {bidir_dir}")

      print("-" * 95)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
