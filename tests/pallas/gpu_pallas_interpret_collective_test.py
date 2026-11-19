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

"""Tests for collective (2-CTA) operations in GPU interpret mode."""

from absl.testing import absltest
import jax
from jax import lax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call as mosaic_interpret
from jax._src.pallas.mosaic_gpu.interpret.params import InterpretGPUParams as InterpretParams
import jax.experimental.pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()

_ROWS = 16  # Rows of the whole window; each of the 2 CTAs loads half.
_COLS = 128
_PARTITIONED = plgpu.CopyPartition.PARTITIONED(0)


def _x():
  return jnp.arange(_ROWS * _COLS, dtype=jnp.float32).reshape(_ROWS, _COLS)


def _races_found():
  return mosaic_interpret.get_races().races_found


# Per CTA: the accumulator is (M, 2N), the LHS (M, K) and the RHS (K, N).
M, N, K = 128, 64, 64


def _inputs():
  rng = np.random.default_rng(0)
  a = jnp.asarray(rng.normal(size=(2 * M, K)), jnp.float16)
  b = jnp.asarray(rng.normal(size=(K, 2 * N)), jnp.float16)
  return a, b


def _expected(a, b):
  return np.matmul(np.asarray(a, np.float32), np.asarray(b, np.float32))


@jtu.thread_unsafe_test_class()
class GpuPallasInterpretCollectiveTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("CPU-only test")
    mosaic_interpret.gpu_callbacks.reset_gpu_interpret_mode_state()

  def _kernel(self, body, out_shape, scratch_types, num_threads=None):
    return plgpu.kernel(
        body,
        out_type=out_shape,
        interpret=InterpretParams(detect_races=True, on_race="warn"),
        scratch_types=scratch_types,
        cluster=(2,),
        cluster_names=("c",),
        num_threads=num_threads,
        thread_name=None if num_threads is None else "wg",
    )

  @jtu.parameterized.parameters(
      "none", "leader_skips_copy_wait", "follower_skips_cluster_wait"
  )
  def test_partitioned_copy(self, race):
    # Each CTA loads its half of `x`. Only the leader can wait for the copy.
    def body(x_gmem, out_gmem, smem, barrier, cluster_barrier):
      cid = lax.axis_index("c")
      plgpu.copy_gmem_to_smem(
          x_gmem, smem, barrier, collective_axes="c",
          leader_tracked=_PARTITIONED,
      )
      if race != "leader_skips_copy_wait":
        @pl.when(cid == 0)
        def _():
          plgpu.barrier_wait(barrier)
      plgpu.barrier_arrive(cluster_barrier)
      if race != "follower_skips_cluster_wait":
        plgpu.barrier_wait(cluster_barrier)
      else:
        @pl.when(cid == 0)
        def _():
          plgpu.barrier_wait(cluster_barrier)
      out_gmem[pl.ds(cid * (_ROWS // 2), _ROWS // 2)] = smem[...]

    kernel = self._kernel(
        body,
        jax.ShapeDtypeStruct((_ROWS, _COLS), jnp.float32),
        dict(
            smem=plgpu.SMEM((_ROWS // 2, _COLS), jnp.float32),
            barrier=plgpu.Barrier(num_arrivals=1),
            cluster_barrier=plgpu.ClusterBarrier(collective_axes=("c",)),
        ),
    )
    x = _x()
    out = kernel(x)
    self.assertEqual(_races_found(), race != "none")
    np.testing.assert_array_equal(out, x)

  @jtu.parameterized.parameters(False, True)
  def test_completion_does_not_release_follower_history(self, follower_writes):
    def body(x_gmem, out_gmem, flag_gmem, smem, barrier):
      cid = lax.axis_index("c")
      writer = 1 if follower_writes else 0

      @pl.when(cid == writer)
      def _():
        flag_gmem[...] = jnp.ones_like(flag_gmem)

      plgpu.copy_gmem_to_smem(
          x_gmem, smem, barrier, collective_axes="c",
          leader_tracked=_PARTITIONED,
      )

      @pl.when(cid == 0)
      def _():
        plgpu.barrier_wait(barrier)
        out_gmem[...] = flag_gmem[...]

    kernel = self._kernel(
        body,
        (
            jax.ShapeDtypeStruct((8, _COLS), jnp.float32),
            jax.ShapeDtypeStruct((8, _COLS), jnp.float32),
        ),
        dict(
            smem=plgpu.SMEM((_ROWS // 2, _COLS), jnp.float32),
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
    )
    kernel(_x())
    self.assertEqual(_races_found(), follower_writes)

  @jtu.parameterized.parameters(False, True)
  def test_leader_history_reaches_other_waiters(self, waits):
    def body(x_gmem, out_gmem, flag_gmem, smem, barrier):
      cid = lax.axis_index("c")
      wg = lax.axis_index("wg")

      @pl.when(wg == 0)
      def _():
        @pl.when(cid == 0)
        def _():
          flag_gmem[...] = jnp.ones_like(flag_gmem)

        plgpu.copy_gmem_to_smem(
            x_gmem, smem, barrier, collective_axes="c",
            leader_tracked=_PARTITIONED,
        )

      @pl.when((wg == 1) & (cid == 0))
      def _():
        if waits:
          plgpu.barrier_wait(barrier)
        out_gmem[...] = flag_gmem[...]

      if not waits:
        @pl.when((wg == 0) & (cid == 0))
        def _():
          plgpu.barrier_wait(barrier)

    kernel = self._kernel(
        body,
        (
            jax.ShapeDtypeStruct((8, _COLS), jnp.float32),
            jax.ShapeDtypeStruct((8, _COLS), jnp.float32),
        ),
        dict(
            smem=plgpu.SMEM((_ROWS // 2, _COLS), jnp.float32),
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
        num_threads=2,
    )
    kernel(_x())
    self.assertEqual(_races_found(), not waits)

  @jtu.parameterized.parameters(False, True)
  def test_follower_refill(self, synchronized):
    def body(x_gmem, out_gmem, smem, barrier, cluster_barrier):
      cid = lax.axis_index("c")
      for i in range(2):
        if i == 1 and synchronized:
          plgpu.barrier_wait(cluster_barrier)
        plgpu.copy_gmem_to_smem(
            x_gmem.at[i], smem, barrier, collective_axes="c",
            leader_tracked=_PARTITIONED,
        )

        @pl.when(cid == 0)
        def _():
          plgpu.barrier_wait(barrier)

        if i == 0:
          plgpu.barrier_arrive(cluster_barrier)
      if not synchronized:
        plgpu.barrier_wait(cluster_barrier)

      @pl.when(cid == 0)
      def _():
        out_gmem[...] = jnp.zeros_like(out_gmem)

    kernel = self._kernel(
        body,
        jax.ShapeDtypeStruct((8, _COLS), jnp.float32),
        dict(
            smem=plgpu.SMEM((_ROWS // 2, _COLS), jnp.float32),
            barrier=plgpu.Barrier(num_arrivals=1),
            cluster_barrier=plgpu.ClusterBarrier(collective_axes=("c",)),
        ),
    )
    kernel(jnp.stack([_x(), _x() + 1]))
    self.assertEqual(_races_found(), not synchronized)

  def test_mismatched_copies_raise(self):
    def body(x_gmem, y_gmem, out_gmem, smem, barrier):
      cid = lax.axis_index("c")

      @pl.when(cid == 0)
      def _():
        plgpu.copy_gmem_to_smem(
            x_gmem, smem, barrier, collective_axes="c",
            leader_tracked=_PARTITIONED,
        )

      @pl.when(cid == 1)
      def _():
        plgpu.copy_gmem_to_smem(
            y_gmem, smem, barrier, collective_axes="c",
            leader_tracked=_PARTITIONED,
        )

      del out_gmem  # Unused.

    kernel = self._kernel(
        body,
        jax.ShapeDtypeStruct((8, _COLS), jnp.float32),
        dict(
            smem=plgpu.SMEM((_ROWS // 2, _COLS), jnp.float32),
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
    )
    with self.assertRaisesRegex(Exception, "must issue the same collective"):
      kernel(_x(), _x())

  def test_copy_issued_by_one_cta_raises(self):
    def body(x_gmem, out_gmem, smem, barrier):
      @pl.when(lax.axis_index("c") == 0)
      def _():
        plgpu.copy_gmem_to_smem(
            x_gmem, smem, barrier, collective_axes="c",
            leader_tracked=_PARTITIONED,
        )

      del out_gmem  # Unused.

    kernel = self._kernel(
        body,
        jax.ShapeDtypeStruct((8, _COLS), jnp.float32),
        dict(
            smem=plgpu.SMEM((_ROWS // 2, _COLS), jnp.float32),
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
    )
    with self.assertRaisesRegex(Exception, "not issued by every CTA"):
      kernel(_x())

  @jtu.parameterized.parameters(False, True)
  def test_predicated_off_copy_still_arrives(self, collective):
    def body(x_gmem, out_gmem, smem, barrier):
      cid = lax.axis_index("c")
      smem[...] = jnp.zeros_like(smem)
      plgpu.commit_smem()
      plgpu.copy_gmem_to_smem(
          x_gmem.at[pl.ds(0, _ROWS)] if collective
          else x_gmem.at[pl.ds(cid * (_ROWS // 2), _ROWS // 2)],
          smem,
          barrier,
          collective_axes="c" if collective else None,
          leader_tracked=_PARTITIONED if collective else None,
          predicate=cid < 0,
      )

      @pl.when((cid == 0) | (not collective))
      def _():
        plgpu.barrier_wait(barrier)
        out_gmem[pl.ds(cid * (_ROWS // 2), _ROWS // 2)] = smem[...]

      if collective:
        @pl.when(cid == 1)
        def _():
          out_gmem[pl.ds(_ROWS // 2, _ROWS // 2)] = jnp.zeros(
              (_ROWS // 2, _COLS), jnp.float32
          )

    kernel = self._kernel(
        body,
        jax.ShapeDtypeStruct((_ROWS, _COLS), jnp.float32),
        dict(
            smem=plgpu.SMEM((_ROWS // 2, _COLS), jnp.float32),
            barrier=plgpu.Barrier(num_arrivals=1),
        ),
    )
    out = kernel(_x())
    self.assertFalse(_races_found())
    np.testing.assert_array_equal(out, np.zeros((_ROWS, _COLS), np.float32))

  def _mma_kernel(self, body):
    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((2 * M, 2 * N), jnp.float32),
        interpret=InterpretParams(detect_races=True, on_race="warn"),
        scratch_types=dict(
            a_smem=plgpu.SMEM((M, K), jnp.float16),
            b_smem=plgpu.SMEM((K, N), jnp.float16),
            acc_tmem=plgpu.TMEM((M, 2 * N), jnp.float32, collective=True),
            load_barrier=plgpu.Barrier(num_arrivals=2),
            mma_barrier=plgpu.Barrier(orders_tensor_core=True),
            ready=plgpu.ClusterBarrier(collective_axes=("c",)),
        ),
        cluster=(2,),
        cluster_names=("c",),
    )

  @staticmethod
  def _load(a_gmem, b_gmem, a_smem, b_smem, load_barrier):
    # CTA i loads rows i*M of the LHS and columns i*N of the RHS.
    plgpu.copy_gmem_to_smem(
        a_gmem, a_smem, load_barrier, collective_axes="c",
        leader_tracked=plgpu.CopyPartition.PARTITIONED(0),
    )
    plgpu.copy_gmem_to_smem(
        b_gmem, b_smem, load_barrier, collective_axes="c",
        leader_tracked=plgpu.CopyPartition.PARTITIONED(1),
    )

  @jtu.parameterized.product(
      arrive_via=["mma", "commit"],
      race=[
          None,
          "follower_overwrites_rhs",
          "follower_reads_acc_early",
          "leader_skips_load_wait",
      ],
  )
  def test_collective_mma(self, arrive_via, race):
    def body(a_gmem, b_gmem, out_gmem, a_smem, b_smem, acc_tmem,
             load_barrier, mma_barrier, ready):
      cid = lax.axis_index("c")
      self._load(a_gmem, b_gmem, a_smem, b_smem, load_barrier)

      @pl.when(cid == 0)
      def _():
        if race != "leader_skips_load_wait":
          plgpu.barrier_wait(load_barrier)
        if race == "follower_overwrites_rhs":
          # Tells the follower that its RHS has landed, but not that the MMA
          # has read it.
          plgpu.barrier_arrive(ready)
        plgpu.tcgen05_mma(
            acc_tmem, a_smem, b_smem,
            mma_barrier if arrive_via == "mma" else None,
            accumulate=False, collective_axis="c",
        )
        if arrive_via == "commit":
          plgpu.tcgen05_commit_arrive(mma_barrier, collective_axis="c")
        if race == "leader_skips_load_wait":
          plgpu.barrier_wait(load_barrier)

      if race == "follower_overwrites_rhs":
        @pl.when(cid == 1)
        def _():
          plgpu.barrier_arrive(ready)
          plgpu.barrier_wait(ready)
          b_smem[...] = jnp.zeros_like(b_smem)

      if race != "follower_reads_acc_early":
        plgpu.barrier_wait(mma_barrier)
      out_gmem[pl.ds(cid * M, M)] = plgpu.async_load_tmem(acc_tmem)
      plgpu.wait_load_tmem()
      if race == "follower_reads_acc_early":
        plgpu.barrier_wait(mma_barrier)

    a, b = _inputs()
    out = self._mma_kernel(body)(a, b)
    self.assertEqual(_races_found(), race is not None)
    if race is None:
      np.testing.assert_allclose(out, _expected(a, b), rtol=1e-5, atol=1e-5)

  @jtu.parameterized.parameters("tcgen05_mma", "tcgen05_commit_arrive")
  def test_follower_issuing_collective_op_raises(self, op):
    def body(a_gmem, b_gmem, out_gmem, a_smem, b_smem, acc_tmem,
             load_barrier, mma_barrier, ready):
      del out_gmem, ready  # Unused.
      self._load(a_gmem, b_gmem, a_smem, b_smem, load_barrier)

      @pl.when(lax.axis_index("c") == 0)
      def _():
        plgpu.barrier_wait(load_barrier)

      if op == "tcgen05_mma":
        plgpu.tcgen05_mma(
            acc_tmem, a_smem, b_smem, mma_barrier, accumulate=False,
            collective_axis="c",
        )
      else:
        plgpu.tcgen05_commit_arrive(mma_barrier, collective_axis="c")

    with self.assertRaisesRegex(Exception, "issued by the follower CTA"):
      self._mma_kernel(body)(*_inputs())

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
