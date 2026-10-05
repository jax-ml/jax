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

"""Multi-GPU collective execution tests for `thunky`."""

import os

from absl.testing import absltest
import jax
from jax import shard_map
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import jaxlib_extension_version

if jaxlib_extension_version >= 503:
  from jax._src.thunky import thunky
else:
  thunky = None  # pyrefly: ignore
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

config.parse_flags_with_absl()


def _jax_executable_to_mlir_text(compiled: jax.stages.Compiled) -> str:
  with thunky.make_ir_context():
    return str(thunky.jax_executable_to_mlir(compiled))


class ThunkyCollectiveTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jaxlib_extension_version < 503:
      self.skipTest("Requires jaxlib_extension_version >= 503")
    if jax.default_backend() != "gpu":
      self.skipTest("ThunkyCollectiveTest requires a GPU backend")
    if len(jax.devices()) < 2:
      self.skipTest("Requires at least 2 GPUs")

  def test_all_reduce(self):
    """Verifies thunky.all_reduce executes AllReduceThunk across 2 GPUs."""
    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def ar_thunky(src_buf, dst_buf):
      thunky.all_reduce(
          src_buf,
          dst_buf,
          "x",
          reduction="sum",
      )

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_ar(x_local):
      out_ref = jax.new_ref(jnp.zeros_like(x_local))
      ar_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32
    )
    res = run_ar(x)
    expected = np.array(
        [[11.0, 22.0, 33.0, 44.0], [11.0, 22.0, 33.0, 44.0]], dtype=np.float32
    )
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)

    compiled = run_ar.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_all_gather(self):
    """Verifies thunky.all_gather executes AllGatherThunk across 2 GPUs."""
    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def ag_thunky(src_buf, dst_buf):
      thunky.all_gather(
          src_buf,
          dst_buf,
          "x",
      )

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_ag(x_local):
      out_ref = jax.new_ref(jnp.zeros((1, 8), dtype=x_local.dtype))
      ag_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32
    )
    res = run_ag(x)
    expected_row = np.array(
        [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32
    )
    expected = np.stack([expected_row, expected_row], axis=0)
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)

    compiled = run_ag.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_reduce_scatter(self):
    """Verifies thunky.reduce_scatter executes ReduceScatterThunk across 2 GPUs."""
    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def rs_thunky(src_buf, dst_buf):
      thunky.reduce_scatter(
          src_buf,
          dst_buf,
          "x",
          reduction="sum",
      )

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_rs(x_local):
      out_ref = jax.new_ref(jnp.zeros((1, 4), dtype=x_local.dtype))
      rs_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        ],
        dtype=jnp.float32,
    )
    res = run_rs(x)
    expected = np.array(
        [[11.0, 22.0, 33.0, 44.0], [55.0, 66.0, 77.0, 88.0]], dtype=np.float32
    )
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)

    compiled = run_rs.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_all_to_all(self):
    """Verifies thunky.all_to_all executes AllToAllThunk across 2 GPUs."""
    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def a2a_thunky(src_buf, dst_buf):
      thunky.all_to_all(
          src_buf,
          dst_buf,
          "x",
          has_split_dimension=True,
      )

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_a2a(x_local):
      out_ref = jax.new_ref(jnp.zeros_like(x_local))
      a2a_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ],
        dtype=jnp.float32,
    )
    res = run_a2a(x)
    expected = np.array(
        [
            [[1.0, 2.0], [10.0, 20.0]],
            [[3.0, 4.0], [30.0, 40.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)

    compiled = run_a2a.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_all_to_all_buffer_sequences(self):
    """Verifies all_to_all with has_split_dimension=False executes across 2 GPUs."""
    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def tuple_all_to_all(src0_ref, src1_ref, dst0_ref, dst1_ref):
      thunky.all_to_all(
          (src0_ref, src1_ref),
          (dst0_ref, dst1_ref),
          "x",
          has_split_dimension=False,
      )

    @jax.jit
    @shard_map(
        mesh=mesh,
        in_specs=(P("x"), P("x")),
        out_specs=(P("x"), P("x")),
        check_vma=False,
    )
    def run_tuple_a2a(s0, s1):
      r_s0 = jax.new_ref(s0)
      r_s1 = jax.new_ref(s1)
      r_d0 = jax.new_ref(jnp.zeros_like(s0))
      r_d1 = jax.new_ref(jnp.zeros_like(s1))
      tuple_all_to_all(r_s0, r_s1, r_d0, r_d1)
      return r_d0[...], r_d1[...]

    x0 = jnp.array([[1.0, 2.0], [10.0, 20.0]], dtype=jnp.float32)
    x1 = jnp.array([[3.0, 4.0], [30.0, 40.0]], dtype=jnp.float32)
    y0, y1 = run_tuple_a2a(x0, x1)
    expected_y0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    expected_y1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    self.assertArraysAllClose(np.asarray(y0), expected_y0)
    self.assertArraysAllClose(np.asarray(y1), expected_y1)

  def test_collective_permute(self):
    """Verifies thunky.collective_permute executes CollectivePermuteThunk across 2 GPUs."""
    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def cp_thunky(src_buf, dst_buf):
      thunky.collective_permute(
          src_buf,
          dst_buf,
          "x",
          source_target_pairs=((0, 1), (1, 0)),
      )

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_cp(x_local):
      out_ref = jax.new_ref(jnp.zeros_like(x_local))
      cp_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32
    )
    res = run_cp(x)
    expected = np.array(
        [[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32
    )
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)

    compiled = run_cp.lower(x).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

  def test_collective_permute_with_non_participating_ranks(self):
    """Verifies thunky.collective_permute executes with non-participating ranks across 2 GPUs."""
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("x",))

    @thunky.jit
    def permute_one_way(src_ref, dst_ref):
      thunky.collective_permute(
          src_ref,
          dst_ref,
          "x",
          source_target_pairs=((0, 1),),
      )

    @jax.jit
    @shard_map(
        mesh=mesh,
        in_specs=(P("x"), P("x")),
        out_specs=P("x"),
        check_vma=False,
    )
    def run_one_way(src, dst):
      s_ref = jax.new_ref(src)
      d_ref = jax.new_ref(dst)
      permute_one_way(s_ref, d_ref)
      return d_ref[...]

    src_arr = jnp.arange(8, dtype=jnp.float32).reshape(2, 4) + 10.0
    dst_arr = jnp.zeros((2, 4), dtype=jnp.float32)
    out = run_one_way(src_arr, dst_arr)
    self.assertArraysAllClose(np.asarray(out[1]), np.asarray(src_arr[0]))

  def test_call_jax_with_collective(self):
    """Verifies JaxExecutableToMlir and call_jax round-trip a JAX function containing collectives."""
    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    def jax_collective_fn(a_ref, out_ref):
      out_ref[...] = jax.lax.psum(a_ref[...], "x") * 2.0

    @jax.jit
    def jitted_jax_collective_fn(a_ref, out_ref):
      out_ref[...] = jax.lax.psum(a_ref[...], "x") * 2.0

    @thunky.jit
    def splice_collective_thunky(x_buf, out_buf):
      thunky.call_jax(jax_collective_fn, x_buf, out_buf)

    @thunky.jit
    def splice_jitted_collective_thunky(x_buf, out_buf):
      thunky.call_jax(jitted_jax_collective_fn, x_buf, out_buf)

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_spliced(x_local):
      out_ref = jax.new_ref(jnp.zeros_like(x_local))
      splice_collective_thunky(x_local, out_ref)
      return out_ref[...]

    @jax.jit
    @shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    def run_spliced_jitted(x_local):
      out_ref = jax.new_ref(jnp.zeros_like(x_local))
      splice_jitted_collective_thunky(x_local, out_ref)
      return out_ref[...]

    x = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32
    )
    expected = np.array(
        [[22.0, 44.0, 66.0, 88.0], [22.0, 44.0, 66.0, 88.0]], dtype=np.float32
    )
    np.testing.assert_allclose(np.asarray(run_spliced(x)), expected, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(run_spliced_jitted(x)), expected, rtol=1e-5
    )

  def test_executable_to_mlir_multi_operand_collectives_execution(self):
    """Verifies combined multi-operand collective thunks execute across 2 GPUs."""
    mesh = jax.sharding.Mesh(jax.devices()[:2], ("x",))

    def _jax_two_pperms(a_ref, b_ref, out_a_ref, out_b_ref):
      out_a_ref[...] = jax.lax.ppermute(
          a_ref[...], "x", perm=[(0, 1), (1, 0)]
      )
      out_b_ref[...] = jax.lax.ppermute(
          b_ref[...], "x", perm=[(0, 1), (1, 0)]
      )

    @thunky.jit
    def thunky_two_pperms(a_buf, b_buf, out_a_buf, out_b_buf):
      thunky.call_jax(_jax_two_pperms, a_buf, b_buf, out_a_buf, out_b_buf)

    @jax.jit
    @shard_map(
        mesh=mesh,
        in_specs=(P("x"), P("x")),
        out_specs=(P("x"), P("x")),
        check_vma=False,
    )
    def run_on_mesh(a, b):
      out_a = jax.new_ref(jnp.zeros((16,), dtype=jnp.float32))
      out_b = jax.new_ref(jnp.zeros((32,), dtype=jnp.float32))
      thunky_two_pperms(jax.new_ref(a), jax.new_ref(b), out_a, out_b)
      return out_a[...], out_b[...]

    a_in = jnp.arange(32, dtype=jnp.float32)
    b_in = jnp.arange(64, dtype=jnp.float32) * 10.0
    res_a, res_b = run_on_mesh(a_in, b_in)
    np.testing.assert_allclose(
        np.asarray(res_a[:16]), np.asarray(a_in[16:]), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(res_b[:32]), np.asarray(b_in[32:]), rtol=1e-5
    )

  def test_async_collective_compute_overlap(self):
    """Verifies overlapping a collective on a communication stream with compute on a computation stream."""
    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))

    def compute_kernel(y_ref, out_ref):
      out_ref[...] = y_ref[...] * 3.0 + 1.0

    @thunky.jit
    def overlap_thunky(x_buf, y_buf, ar_out_buf, comp_out_buf):
      def comm_work():
        thunky.all_reduce(
            x_buf,
            ar_out_buf,
            "x",
            reduction="sum",
        )

      # Launch AllReduce asynchronously on the communication stream
      comm_tok = thunky.async_start(
          comm_work, stream_id=0, stream_kind="communication"
      )
      # Run compute kernel concurrently on the main execution stream
      thunky.call_jax(compute_kernel, y_buf, comp_out_buf)
      # Synchronize communication stream back to the main stream
      thunky.async_done(comm_tok)

    @jax.jit
    @shard_map(
        mesh=mesh,
        in_specs=(P("x"), P("x")),
        out_specs=(P("x"), P("x")),
        check_vma=False,
    )
    def run_overlap(x_local, y_local):
      ar_out_ref = jax.new_ref(jnp.zeros_like(x_local))
      comp_out_ref = jax.new_ref(jnp.zeros_like(y_local))
      overlap_thunky(x_local, y_local, ar_out_ref, comp_out_ref)
      return ar_out_ref[...], comp_out_ref[...]

    x = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=jnp.float32
    )
    y = jnp.array(
        [[5.0, 6.0, 7.0, 8.0], [50.0, 60.0, 70.0, 80.0]], dtype=jnp.float32
    )
    ar_res, comp_res = run_overlap(x, y)

    expected_ar = np.array(
        [[11.0, 22.0, 33.0, 44.0], [11.0, 22.0, 33.0, 44.0]], dtype=np.float32
    )
    expected_comp = np.asarray(y) * 3.0 + 1.0
    np.testing.assert_allclose(np.asarray(ar_res), expected_ar, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(comp_res), expected_comp, rtol=1e-5)

    compiled = run_overlap.lower(x, y).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())
    mlir_text = _jax_executable_to_mlir_text(compiled)
    self.assertIn("thunky.all_reduce", mlir_text)
    self.assertIn("is_communication = true", mlir_text)

  def test_ring_all_gather_collective_matmul(self):
    """Verifies an N-GPU Ring AllGather(A) @ B collective GEMM overlapping CollectivePermute with matmul into sliced output buffers."""
    num_devices = len(jax.devices())
    mesh = Mesh(np.array(jax.devices()), ("x",))
    m_block = int(os.environ.get("THUNKY_M_BLOCK", "8"))
    k_dim = int(os.environ.get("THUNKY_K_DIM", "32"))
    n_block = int(os.environ.get("THUNKY_N_BLOCK", "8"))
    avoid_d2h = os.environ.get("THUNKY_AVOID_D2H", "0") == "1"
    multistream = os.environ.get("THUNKY_MULTISTREAM", "0") == "1"
    dtype_str = os.environ.get("THUNKY_DTYPE", "bf16")
    dtype = {"f32": jnp.float32, "bf16": jnp.bfloat16, "f16": jnp.float16}[
        dtype_str
    ]

    def matmul_fn(lhs_ref, rhs_ref, out_ref):
      out_ref[...] = jnp.matmul(lhs_ref[...], rhs_ref[...])

    if multistream:

      @thunky.jit(
          scratch_shapes=(
              [
                  jax.ShapeDtypeStruct((m_block, k_dim), dtype)
                  for _ in range(num_devices - 1)
              ],
              jax.ShapeDtypeStruct((num_devices * m_block, n_block), dtype),
          )
      )
      def ring_all_gather_matmul_thunky(
          a_local_buf,
          b_local_buf,
          dev_id_buf,
          c_out_buf,
          a_scratch_bufs,
          c_ring_buf,
      ):
        bufs = [a_local_buf] + list(a_scratch_bufs)
        c_ring_slices = [
            c_ring_buf.at[s * m_block : (s + 1) * m_block]
            for s in range(num_devices)
        ]
        ring_pairs = tuple(
            (i, (i + 1) % num_devices) for i in range(num_devices)
        )

        tok_a = {}
        tok_c = {}

        def start_comm(step_idx):
          return thunky.async_start(
              lambda src=bufs[step_idx - 1], dst=bufs[step_idx]: (
                  thunky.collective_permute(
                      src, dst, "x", source_target_pairs=ring_pairs
                  )
              ),
              stream_id=0,
              stream_kind="communication",
          )

        def start_gemm(step_idx):
          return thunky.async_start(
              lambda a_buf=bufs[step_idx], out_slice=c_ring_slices[step_idx]: (
                  thunky.call_jax(matmul_fn, a_buf, b_local_buf, out_slice)
              ),
              stream_id=(step_idx % 2),
              stream_kind="computation",
          )

        # Queue all N-1 collective permutes back-to-back on communication stream
        for s in range(1, num_devices):
          tok_a[s] = start_comm(s)

        # Launch Gemm 0 on computeA immediately
        tok_c[0] = start_gemm(0)

        # Each subsequent Gemm s waits only on tok_a[s] (arrival of bufs[s])
        for s in range(1, num_devices):
          thunky.async_done(tok_a[s])
          tok_c[s] = start_gemm(s)

        # Wait for all compute streams before final coalesced reorder
        for s in range(num_devices):
          thunky.async_done(tok_c[s])

        def unroll_fn(c_ring_ref, dev_id_ref, c_out_ref):
          d = dev_id_ref[...]
          indices = (d - jnp.arange(num_devices, dtype=jnp.int32)) % num_devices
          c_out_ref[...] = (
              c_ring_ref[...]
              .reshape(num_devices, m_block, n_block)[indices]
              .reshape(m_block * num_devices, n_block)
          )

        thunky.call_jax(unroll_fn, c_ring_buf, dev_id_buf, c_out_buf)

    else:

      @thunky.jit(
          scratch_shapes=(
              [jax.ShapeDtypeStruct((m_block, k_dim), dtype) for _ in range(2)],
          )
      )
      def ring_all_gather_matmul_thunky(
          a_local_buf, b_local_buf, dev_id_buf, c_out_buf, scratch_bufs
      ):
        # Zero-copy views into the N row blocks of c_out_buf (total shape (m_block * N, n_block))
        c_slices = [
            c_out_buf.at[r * m_block : (r + 1) * m_block]
            for r in range(num_devices)
        ]
        ring_pairs = tuple(
            (i, (i + 1) % num_devices) for i in range(num_devices)
        )

        for step in range(num_devices):
          curr_a = a_local_buf if step == 0 else scratch_bufs[(step - 1) % 2]

          # Overlap CollectivePermute for the next ring step with the current step's GEMM
          if step < num_devices - 1:
            next_a = scratch_bufs[step % 2]
            comm_tok = thunky.async_start(
                lambda src=curr_a, dst=next_a: thunky.collective_permute(
                    src, dst, "x", source_target_pairs=ring_pairs
                ),
                stream_id=0,
                stream_kind="communication",
            )

          if avoid_d2h:
            # Compute GEMM and write into row slice `(dev_id - step) % num_devices` on device
            # without host-device synchronization (avoiding ConditionalThunk MemcpyD2H).
            def make_step_matmul_fn(s):
              def _step_fn(lhs_ref, rhs_ref, dev_id_ref, c_out_ref):
                block = jnp.matmul(lhs_ref[...], rhs_ref[...])
                row_idx = (dev_id_ref[...] - s) % num_devices
                c_out_ref[jax.ds(row_idx * m_block, m_block), :] = block

              return _step_fn

            thunky.call_jax(
                make_step_matmul_fn(step),
                curr_a,
                b_local_buf,
                dev_id_buf,
                c_out_buf,
            )
          else:
            # At step `step`, device `d` holds row block `(d - step) % num_devices` of A.
            # Switch on dev_id_buf (int32) to write GEMM output directly into that row slice of C.
            thunky.switch(
                dev_id_buf,
                [
                    lambda c_slice=c_slices[
                        (d - step) % num_devices
                    ], a_buf=curr_a: (
                        thunky.call_jax(matmul_fn, a_buf, b_local_buf, c_slice)
                    )
                    for d in range(num_devices)
                ],
            )

          if step < num_devices - 1:
            thunky.async_done(comm_tok)

    @jax.jit
    @shard_map(
        mesh=mesh,
        in_specs=(P("x", None), P(None, "x")),
        out_specs=P(None, "x"),
        check_vma=False,
    )
    def run_collective_matmul(a_local, b_local):
      dev_id = jnp.int32(jax.lax.axis_index("x"))
      c_out_ref = jax.new_ref(
          jnp.zeros((m_block * num_devices, n_block), dtype=dtype)
      )
      ring_all_gather_matmul_thunky(a_local, b_local, dev_id, c_out_ref)
      return c_out_ref[...]

    rng = np.random.default_rng(42)
    a_np = rng.standard_normal(
        (m_block * num_devices, k_dim), dtype=np.float32
    ).astype(dtype)
    b_np = rng.standard_normal(
        (k_dim, n_block * num_devices), dtype=np.float32
    ).astype(dtype)

    a = jax.device_put(a_np, jax.sharding.NamedSharding(mesh, P("x", None)))
    b = jax.device_put(b_np, jax.sharding.NamedSharding(mesh, P(None, "x")))

    res = run_collective_matmul(a, b)
    jax.block_until_ready(res)

    trace_dir = os.environ.get("JAX_TRACE_DIR")
    if trace_dir:
      with jax.profiler.trace(trace_dir):
        for _ in range(5):
          res = run_collective_matmul(a, b)
          jax.block_until_ready(res)

    expected = np.asarray(a_np, dtype=np.float32) @ np.asarray(
        b_np, dtype=np.float32
    )
    if dtype == jnp.float32:
      tol = 1e-4 * np.sqrt(k_dim / 32.0)
    else:
      tol = 2e-2 * np.sqrt(k_dim / 32.0)
    np.testing.assert_allclose(
        np.asarray(res, dtype=np.float32), expected, rtol=tol, atol=tol
    )

    compiled = run_collective_matmul.lower(a, b).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())
    mlir_text = _jax_executable_to_mlir_text(compiled)
    self.assertIn("thunky.collective_permute", mlir_text)
    self.assertIn("is_communication = true", mlir_text)
    if not avoid_d2h and not multistream:
      self.assertIn("thunky.cond", mlir_text)

  def test_collective_group_multi_gpu(self):
    """Tests thunky.collective_group fusing multi-GPU collectives in a single NCCL group."""
    devices = jax.devices()[:2]
    mesh = Mesh(np.array(devices), ("x",))

    @thunky.jit
    def grouped_collectives_prog(a_buf, b_buf, out_a_buf, out_b_buf):
      thunky.collective_group(
          lambda: (
              thunky.all_reduce(
                  a_buf, out_a_buf, "x", reduction="sum"
              ),
              thunky.collective_permute(
                  b_buf, out_b_buf, "x", source_target_pairs=((0, 1), (1, 0))
              ),
          )
      )

    @jax.jit
    @shard_map(
        mesh=mesh,
        in_specs=(P("x"), P("x")),
        out_specs=(P("x"), P("x")),
        check_vma=False,
    )
    def run_grouped_collectives(a_local, b_local):
      out_a = jax.new_ref(jnp.zeros_like(a_local))
      out_b = jax.new_ref(jnp.zeros_like(b_local))
      grouped_collectives_prog(a_local, b_local, out_a, out_b)
      return out_a[...], out_b[...]

    # Shard across 2 GPUs: device 0 has [1, 2], device 1 has [3, 4]
    a = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    # device 0 has [10, 20], device 1 has [30, 40]
    b = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)

    res_a, res_b = run_grouped_collectives(a, b)
    # all_reduce sum: both devices get [1+3, 2+4] = [4, 6]
    expected_a = np.array([4.0, 6.0, 4.0, 6.0], dtype=np.float32)
    # collective_permute swap: device 0 gets [30, 40], device 1 gets [10, 20]
    expected_b = np.array([30.0, 40.0, 10.0, 20.0], dtype=np.float32)

    np.testing.assert_allclose(np.asarray(res_a), expected_a, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(res_b), expected_b, rtol=1e-5)

    compiled = run_grouped_collectives.lower(a, b).compile()
    self.assertIn("thunky.inline_module", compiled.as_text())

    mlir_text = _jax_executable_to_mlir_text(compiled)
    self.assertIn("thunky.collective_group", mlir_text)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
