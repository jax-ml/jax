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

import functools

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call as mosaic_interpret
from jax._src.pallas.mosaic_gpu.interpret.params import InterpretGPUParams as InterpretParams
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

M = N = K = 128
ACC_SHAPE = (M, N)  # jnp.float32
LHS_SHAPE = (M, K)  # jnp.float16
RHS_SHAPE = (K, N)  # jnp.float16

class WarpSpecializeHelper:
  """
  Helper for making multi-threaded kernels generic over whether they're
  warp-specialized or not.
  """

  def __init__(self, warp_specialize: bool):
    self.warp_specialize = warp_specialize

  def thread_count(self, n):
    if n > 4:
      raise ValueError(f'A warp only has 4 threads, but got {n}')
    return 1 if self.warp_specialize else n

  def maybe_warp_specialize(self, *, thread_name):
    if self.warp_specialize:
      return plgpu.warp_map
    else:
      def decorator(f):
        f(jax.lax.axis_index(thread_name))
      return decorator



@jtu.thread_unsafe_test_class()
class TCGen05RaceDetectionTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    mosaic_interpret.gpu_callbacks.reset_gpu_interpret_mode_state()

    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

    self.num_devices = jax.device_count()
    if self.num_devices > 1:
      self.skipTest(f'requires 1 device, found {self.num_devices}')

  # ----------------------------------------------------------------------
  # TMEM store/load waits (tcgen05.st/ld + tcgen05.wait::{st,ld})
  # ----------------------------------------------------------------------

  @jtu.parameterized.product(
      first_op=['store', 'load'],
      wait=['none', 'commit_tmem', 'wait_load_tmem', 'commit_arrive'],
  )
  def test_store_load_synchronization(self, first_op, wait):
    # A store followed by an overlapping load (or vice versa) is only ordered
    # by the wait matching the *first* operation: commit_tmem awaits stores,
    # wait_load_tmem awaits loads. tcgen05_commit_arrive tracks only MMAs and
    # SMEM->TMEM copies, so it does not help here at all.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            tmem_ref=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, tmem_ref, barrier_ref):
      if first_op == 'store':
        plgpu.async_store_tmem(tmem_ref, jnp.full(ACC_SHAPE, 42.0, jnp.float32))
      else:
        out_ref[...] = plgpu.async_load_tmem(tmem_ref)

      if wait == 'commit_tmem':
        plgpu.commit_tmem()
      elif wait == 'wait_load_tmem':
        plgpu.wait_load_tmem()
      elif wait == 'commit_arrive':
        plgpu.tcgen05_commit_arrive(barrier_ref)
        plgpu.barrier_wait(barrier_ref)

      if first_op == 'store':
        out_ref[...] = plgpu.async_load_tmem(tmem_ref)
      else:
        plgpu.async_store_tmem(tmem_ref, jnp.full(ACC_SHAPE, 42.0, jnp.float32))
        plgpu.commit_tmem()

    correct = (first_op == 'store' and wait == 'commit_tmem') or (
        first_op == 'load' and wait == 'wait_load_tmem'
    )

    out = _kernel()
    if correct and first_op == 'store':
      self.assertArraysEqual(out, jnp.full(ACC_SHAPE, 42.0, jnp.float32))
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  @jtu.parameterized.product(
      commit=[False, True],
      store_target=['lhs', 'acc'],
  )
  def test_store_then_mma_requires_commit_tmem(self, commit, store_target):
    # An MMA is not ordered after a pending async_store_tmem to its LHS
    # operand (read-after-write) or to its accumulator (write-after-write,
    # and read-after-write when accumulate=True) without commit_tmem.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            lhs_tmem=plgpu.TMEM(LHS_SHAPE, jnp.float16, packed=True),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, acc_tmem, lhs_tmem, b_smem, barrier_ref):
      if store_target == 'lhs':
        plgpu.async_store_tmem(lhs_tmem, jnp.zeros(LHS_SHAPE, jnp.float16))
        accumulate = False
      else:
        plgpu.async_store_tmem(acc_tmem, jnp.zeros(ACC_SHAPE, jnp.float32))
        accumulate = True
      if commit:
        plgpu.commit_tmem()
      plgpu.tcgen05_mma(
          acc_tmem, lhs_tmem, b_smem, barrier_ref, accumulate=accumulate
      )
      plgpu.barrier_wait(barrier_ref)
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    _kernel()
    correct = commit
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  @jtu.parameterized.product(
      commit=[False, True],
  )
  def test_overlapping_stores_require_commit_tmem(self, commit):
    # Two async stores are not pipelined with each other (st -> st is not a
    # pipelined pair), so overlapping stores race without commit_tmem between.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            tmem_ref=plgpu.TMEM(ACC_SHAPE, jnp.float32),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, tmem_ref):
      plgpu.async_store_tmem(tmem_ref, jnp.full(ACC_SHAPE, 1.0, jnp.float32))
      if commit:
        plgpu.commit_tmem()
      plgpu.async_store_tmem(tmem_ref, jnp.full(ACC_SHAPE, 2.0, jnp.float32))
      plgpu.commit_tmem()
      out_ref[...] = plgpu.async_load_tmem(tmem_ref)

    out = _kernel()
    correct = commit
    if correct:
      self.assertArraysEqual(out, jnp.full(ACC_SHAPE, 2.0, jnp.float32))
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  @jtu.parameterized.product(
      wait=[False, True],
      overwriter=['mma', 'copy'],
  )
  def test_tmem_load_then_overwrite_requires_wait(self, wait, overwriter):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            src_smem=plgpu.SMEM(ACC_SHAPE, jnp.float32),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, acc_tmem, a_smem, b_smem, src_smem, barrier_ref):
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)
      if wait:
        plgpu.wait_load_tmem()
      if overwriter == 'mma':
        plgpu.tcgen05_mma(
            acc_tmem, a_smem, b_smem, barrier_ref, accumulate=False
        )
      else:
        plgpu.async_copy_smem_to_tmem(src_smem, acc_tmem)
        plgpu.tcgen05_commit_arrive(barrier_ref)
      plgpu.barrier_wait(barrier_ref)

    _kernel()
    correct = wait
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  def test_tmem_load_write_is_synchronous(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            tmem_ref=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            smem_ref=plgpu.SMEM(ACC_SHAPE, jnp.float32),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, tmem_ref, smem_ref):
      smem_ref[...] = plgpu.async_load_tmem(tmem_ref)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref, out_ref)
      plgpu.wait_smem_to_gmem(0)

    _kernel()
    self.assertFalse(mosaic_interpret.get_races().races_found)

  def test_overlapping_loads_do_not_race(self):
    # loads only read TMEM, so unordered overlapping loads are fine.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            tmem_ref=plgpu.TMEM(ACC_SHAPE, jnp.float32),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, tmem_ref):
      a = plgpu.async_load_tmem(tmem_ref)
      b = plgpu.async_load_tmem(tmem_ref)
      out_ref[...] = a + b

    _kernel()
    self.assertFalse(mosaic_interpret.get_races().races_found)

  # ----------------------------------------------------------------------
  # MMA completion (tcgen05.mma + tcgen05.commit)
  # ----------------------------------------------------------------------

  @jtu.parameterized.parameters(
      ('mma_barrier', False),
      ('mma_barrier', True),
      ('commit_arrive', False),
      ('commit_arrive', True),
      # arrive='none' with wait=True would deadlock, so it is not tested.
      ('none', False),
  )
  def test_reading_mma_accumulator_requires_arrive_and_wait(self, arrive, wait):
    # The accumulator may only be read after the MMA's completion was observed
    # by waiting on a barrier that the MMA arrives on -- either passed to
    # tcgen05_mma directly, or registered via tcgen05_commit_arrive.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, acc_tmem, a_smem, b_smem, barrier_ref):
      plgpu.tcgen05_mma(
          acc_tmem,
          a_smem,
          b_smem,
          barrier_ref if arrive == 'mma_barrier' else None,
          accumulate=False,
      )
      if arrive == 'commit_arrive':
        plgpu.tcgen05_commit_arrive(barrier_ref)
      if wait:
        plgpu.barrier_wait(barrier_ref)
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)
      # Cleanly await the MMA before the kernel ends (after the racy load).
      if arrive == 'none':
        plgpu.tcgen05_commit_arrive(barrier_ref)
      if not wait:
        plgpu.barrier_wait(barrier_ref)

    _kernel()
    correct = arrive != 'none' and wait
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  def test_mmas_on_same_accumulator_are_pipelined(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, acc_tmem, a_smem, b_smem, barrier_ref):
      plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, accumulate=False)
      plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    _kernel()
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(
      load_first=[False, True],
  )
  def test_commit_arrive_tracks_only_prior_mmas(self, load_first):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc1_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            acc2_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            barrier1=plgpu.Barrier(orders_tensor_core=True),
            barrier2=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(
        out_ref, acc1_tmem, acc2_tmem, a_smem, b_smem, barrier1, barrier2
    ):
      plgpu.tcgen05_mma(acc1_tmem, a_smem, b_smem, accumulate=False)
      plgpu.tcgen05_commit_arrive(barrier1)
      plgpu.tcgen05_mma(acc2_tmem, a_smem, b_smem, accumulate=False)
      plgpu.barrier_wait(barrier1)
      if load_first:
        out_ref[...] = plgpu.async_load_tmem(acc1_tmem)
      else:
        out_ref[...] = plgpu.async_load_tmem(acc2_tmem)
      # Cleanly await the second MMA before the kernel ends.
      plgpu.tcgen05_commit_arrive(barrier2)
      plgpu.barrier_wait(barrier2)

    _kernel()
    correct = load_first
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  @jtu.parameterized.product(
      wait=[False, True],
  )
  def test_store_to_accumulator_while_mma_in_flight(self, wait):
    # st -> mma is not pipelined and neither is mma -> st: storing to the
    # accumulator of an unawaited MMA is a write-after-write race.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, acc_tmem, a_smem, b_smem, barrier_ref):
      plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, barrier_ref, accumulate=False)
      if wait:
        plgpu.barrier_wait(barrier_ref)
      plgpu.async_store_tmem(acc_tmem, jnp.zeros(ACC_SHAPE, jnp.float32))
      plgpu.commit_tmem()
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)
      if not wait:
        plgpu.barrier_wait(barrier_ref)

    _kernel()
    correct = wait
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  @jtu.parameterized.product(
      warp_specialize=[WarpSpecializeHelper(False), WarpSpecializeHelper(True)],
      use_barrier_in_mma=[False, True],
  )
  def test_cross_thread_mmas_pipelined(self, warp_specialize, use_barrier_in_mma):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            mma_barrier0=plgpu.Barrier(orders_tensor_core=True),
            thread_barrier=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
        num_threads=warp_specialize.thread_count(2),
        thread_name='t',
    )
    def _kernel(
        out_ref,
        acc_tmem,
        a_smem,
        b_smem,
        mma_barrier0,
        thread_barrier,
    ):
      @warp_specialize.maybe_warp_specialize(thread_name='t')
      def _(warp_id):

        @pl.when(warp_id == 0)
        def _():
          if use_barrier_in_mma:
            plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, thread_barrier, accumulate=False)
          else:
            plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, accumulate=False)
            plgpu.barrier_arrive(thread_barrier)

        @pl.when(warp_id == 1)
        def _():
          plgpu.barrier_wait(thread_barrier)
          plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, mma_barrier0)
          plgpu.barrier_wait(mma_barrier0)
          out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    _kernel()
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(
      warp_specialize=[WarpSpecializeHelper(False), WarpSpecializeHelper(True)],
  )
  def test_cross_thread_mma_pipeline_multiparent_has_race(
      self, warp_specialize
  ):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            mma_barrier0=plgpu.Barrier(orders_tensor_core=True),
            mma_barrier1=plgpu.Barrier(orders_tensor_core=True),
            thread_barrier=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
        num_threads=warp_specialize.thread_count(2),
        thread_name='t',
    )
    def _kernel(
        _out_ref,
        acc_tmem,
        a_smem,
        b_smem,
        mma_barrier0,
        mma_barrier1,
        thread_barrier,
    ):
      @warp_specialize.maybe_warp_specialize(thread_name='t')
      def _(warp_id):

        @pl.when(warp_id == 0)
        def _():
          plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, accumulate=False)
          plgpu.barrier_arrive(thread_barrier)
          plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, mma_barrier0)
          plgpu.barrier_wait(mma_barrier0)

        @pl.when(warp_id == 1)
        def _():
          plgpu.barrier_wait(thread_barrier)
          plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, mma_barrier1)
          plgpu.barrier_wait(mma_barrier1)

    _kernel()
    self.assertTrue(mosaic_interpret.get_races().races_found)

  # ----------------------------------------------------------------------
  # SMEM operands of MMAs and SMEM->TMEM copies (async proxy)
  # ----------------------------------------------------------------------

  @jtu.parameterized.product(
      commit=[False, True],
  )
  def test_smem_write_then_mma_requires_commit_smem(self, commit):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, acc_tmem, a_smem, b_smem, barrier_ref):
      a_smem[...] = jnp.ones(LHS_SHAPE, jnp.float16)
      if commit:
        plgpu.commit_smem()
      plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, barrier_ref, accumulate=False)
      plgpu.barrier_wait(barrier_ref)
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    _kernel()
    correct = commit
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  @jtu.parameterized.product(
      wait=[False, True],
      writer=['store', 'tma'],
  )
  def test_overwriting_smem_operand_while_mma_in_flight(self, wait, writer):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            gmem_ref=plgpu.GMEM(RHS_SHAPE, jnp.float16),
            mma_barrier=plgpu.Barrier(orders_tensor_core=True),
            tma_barrier=plgpu.Barrier(num_arrivals=1),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(
        _out_ref, acc_tmem, a_smem, b_smem, gmem_ref, mma_barrier, tma_barrier
    ):
      plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, mma_barrier, accumulate=False)
      if wait:
        plgpu.barrier_wait(mma_barrier)
      if writer == 'store':
        b_smem[...] = jnp.zeros(RHS_SHAPE, jnp.float16)
      else:
        plgpu.copy_gmem_to_smem(gmem_ref, b_smem, tma_barrier)
        plgpu.barrier_wait(tma_barrier)
      if not wait:
        plgpu.barrier_wait(mma_barrier)

    _kernel()
    correct = wait
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  @jtu.parameterized.product(
      wait=[False, True],
  )
  def test_overwriting_tmem_operand_while_mma_in_flight(self, wait):
    # Same as above, but for an LHS operand living in TMEM: storing to it
    # while the MMA may still be reading it is a race.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            lhs_tmem=plgpu.TMEM(LHS_SHAPE, jnp.float16, packed=True),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(_out_ref, acc_tmem, lhs_tmem, b_smem, barrier_ref):
      plgpu.tcgen05_mma(
          acc_tmem, lhs_tmem, b_smem, barrier_ref, accumulate=False
      )
      if wait:
        plgpu.barrier_wait(barrier_ref)
      plgpu.async_store_tmem(lhs_tmem, jnp.zeros(LHS_SHAPE, jnp.float16))
      plgpu.commit_tmem()
      if not wait:
        plgpu.barrier_wait(barrier_ref)

    _kernel()
    correct = wait
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  @jtu.parameterized.product(
      wait=[False, True],
  )
  def test_overwriting_copy_source_smem(self, wait):
    # The SMEM source of async_copy_smem_to_tmem is read asynchronously and
    # may only be overwritten after the copy was awaited through
    # tcgen05_commit_arrive + barrier_wait.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            tmem_ref=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            src_smem=plgpu.SMEM(ACC_SHAPE, jnp.float32),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(_out_ref, tmem_ref, src_smem, barrier_ref):
      plgpu.async_copy_smem_to_tmem(src_smem, tmem_ref)
      if wait:
        plgpu.tcgen05_commit_arrive(barrier_ref)
        plgpu.barrier_wait(barrier_ref)
      src_smem[...] = jnp.zeros(ACC_SHAPE, jnp.float32)
      if not wait:
        plgpu.tcgen05_commit_arrive(barrier_ref)
        plgpu.barrier_wait(barrier_ref)

    _kernel()
    correct = wait
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  # ----------------------------------------------------------------------
  # SMEM->TMEM copies (tcgen05.cp) and pipelining
  # ----------------------------------------------------------------------

  def test_copy_to_tmem_then_mma_is_pipelined(self):
    # Negative test: cp -> mma is a pipelined pair, so an MMA may consume the
    # TMEM operand written by a preceding async_copy_smem_to_tmem with no
    # synchronization in between.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            lhs_tmem=plgpu.TMEM(LHS_SHAPE, jnp.float16, packed=True),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, acc_tmem, lhs_tmem, a_smem, b_smem, barrier_ref):
      plgpu.async_copy_smem_to_tmem(a_smem, lhs_tmem)
      plgpu.tcgen05_mma(
          acc_tmem, lhs_tmem, b_smem, barrier_ref, accumulate=False
      )
      plgpu.barrier_wait(barrier_ref)
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    _kernel()
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(
      wait=[False, True],
  )
  def test_reading_copied_tmem_requires_commit_arrive(self, wait):
    # cp -> ld is *not* a pipelined pair: loading the destination of an
    # SMEM->TMEM copy requires awaiting the copy first.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            tmem_ref=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            src_smem=plgpu.SMEM(ACC_SHAPE, jnp.float32),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, tmem_ref, src_smem, barrier_ref):
      plgpu.async_copy_smem_to_tmem(src_smem, tmem_ref)
      if wait:
        plgpu.tcgen05_commit_arrive(barrier_ref)
        plgpu.barrier_wait(barrier_ref)
      out_ref[...] = plgpu.async_load_tmem(tmem_ref)
      if not wait:
        plgpu.tcgen05_commit_arrive(barrier_ref)
        plgpu.barrier_wait(barrier_ref)

    _kernel()
    correct = wait
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  @jtu.parameterized.product(
      second_op=['store', 'copy'],
  )
  def test_unordered_overwrites_of_copied_tmem(self, second_op):
    # Neither cp -> st nor cp -> cp are pipelined pairs, so a second write to
    # the destination of a pending SMEM->TMEM copy is a write-after-write
    # race.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            tmem_ref=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            src_smem=plgpu.SMEM(ACC_SHAPE, jnp.float32),
            barrier_ref=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(_out_ref, tmem_ref, src_smem, barrier_ref):
      plgpu.async_copy_smem_to_tmem(src_smem, tmem_ref)
      if second_op == 'store':
        plgpu.async_store_tmem(tmem_ref, jnp.zeros(ACC_SHAPE, jnp.float32))
        plgpu.commit_tmem()
      else:
        plgpu.async_copy_smem_to_tmem(src_smem, tmem_ref)
      plgpu.tcgen05_commit_arrive(barrier_ref)
      plgpu.barrier_wait(barrier_ref)

    _kernel()
    self.assertTrue(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(
      wait=[False, True],
  )
  def test_copy_into_accumulator_while_mma_in_flight(self, wait):
    # The pipelining rule is one-directional: cp -> mma is pipelined, but
    # mma -> cp is not. Copying into the accumulator of an unawaited MMA is a
    # race even from the same thread.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            src_smem=plgpu.SMEM(ACC_SHAPE, jnp.float32),
            barrier1=plgpu.Barrier(orders_tensor_core=True),
            barrier2=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(
        _out_ref, acc_tmem, a_smem, b_smem, src_smem, barrier1, barrier2
    ):
      plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, accumulate=False)
      if wait:
        plgpu.tcgen05_commit_arrive(barrier1)
        plgpu.barrier_wait(barrier1)
      plgpu.async_copy_smem_to_tmem(src_smem, acc_tmem)
      plgpu.tcgen05_commit_arrive(barrier2)
      plgpu.barrier_wait(barrier2)

    _kernel()
    correct = wait
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  # ----------------------------------------------------------------------
  # Cross-thread synchronization
  # ----------------------------------------------------------------------

  @jtu.parameterized.product(
      producer_op=['store', 'load'],
      producer_waits=[False, True],
      consumer_waits=[False, True],
      warp_specialize=[WarpSpecializeHelper(False), WarpSpecializeHelper(True)],
  )
  def test_cross_thread_handoff_requires_producer_wait(
      self, producer_op, producer_waits, consumer_waits, warp_specialize
  ):
    # Handing TMEM off to another thread requires the *producer* to await its
    # own asynchronous accesses before signaling: barrier arrival does not
    # carry the completion of pending loads/stores. The consumer calling
    # commit_tmem/wait_load_tmem itself does not help, because those waits
    # only cover operations issued by the calling thread.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            tmem_ref=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            thread_barrier=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
        num_threads=warp_specialize.thread_count(2),
        thread_name='t',
    )
    def _kernel(out_ref, tmem_ref, thread_barrier):
      @warp_specialize.maybe_warp_specialize(thread_name='t')
      def _(warp_id):

        @pl.when(warp_id == 0)
        def _():
          if producer_op == 'store':
            plgpu.async_store_tmem(
                tmem_ref, jnp.full(ACC_SHAPE, 1.0, jnp.float32)
            )
            if producer_waits:
              plgpu.commit_tmem()
          else:
            out_ref[...] = plgpu.async_load_tmem(tmem_ref)
            if producer_waits:
              plgpu.wait_load_tmem()
          plgpu.barrier_arrive(thread_barrier)

        @pl.when(warp_id == 1)
        def _():
          plgpu.barrier_wait(thread_barrier)
          if consumer_waits:
            if producer_op == 'store':
              plgpu.commit_tmem()
            else:
              plgpu.wait_load_tmem()
          if producer_op == 'store':
            out_ref[...] = plgpu.async_load_tmem(tmem_ref)
          else:
            plgpu.async_store_tmem(
                tmem_ref, jnp.full(ACC_SHAPE, 2.0, jnp.float32)
            )
            plgpu.commit_tmem()

    _kernel()
    correct = producer_waits
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  @jtu.parameterized.product(
      warp_specialize=[WarpSpecializeHelper(False), WarpSpecializeHelper(True)],
  )
  def test_commit_arrive_does_not_cover_other_threads_mma(
      self, warp_specialize
  ):
    # tcgen05_commit_arrive tracks only the MMAs issued by the calling
    # thread. A consumer thread cannot use it to await a producer thread's
    # MMA.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            thread_barrier=plgpu.Barrier(orders_tensor_core=True),
            mma_barrier=plgpu.Barrier(orders_tensor_core=True),
            producer_barrier=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
        num_threads=warp_specialize.thread_count(2),
        thread_name='t',
    )
    def _kernel(
        out_ref,
        acc_tmem,
        a_smem,
        b_smem,
        thread_barrier,
        mma_barrier,
        producer_barrier,
    ):
      @warp_specialize.maybe_warp_specialize(thread_name='t')
      def _(warp_id):

        @pl.when(warp_id == 0)
        def _():
          plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, accumulate=False)
          plgpu.barrier_arrive(thread_barrier)
          plgpu.tcgen05_commit_arrive(producer_barrier)
          plgpu.barrier_wait(producer_barrier)

        @pl.when(warp_id == 1)
        def _():
          plgpu.barrier_wait(thread_barrier)
          # This produces a vacuous arrival, but does not observe completion of
          # the producer thread's MMA.
          plgpu.tcgen05_commit_arrive(mma_barrier)
          plgpu.barrier_wait(mma_barrier)
          out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    _kernel()
    self.assertTrue(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(
      warp_specialize=[WarpSpecializeHelper(False), WarpSpecializeHelper(True)],
  )
  def test_cross_thread_copy_then_mma_is_pipelined(self, warp_specialize):
    # Negative test (PTX "pipelined instructions, different thread"): cp ->
    # mma stays pipelined across threads when the threads synchronize through
    # an orders_tensor_core barrier, with no completion wait needed for the
    # copy.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            lhs_tmem=plgpu.TMEM(LHS_SHAPE, jnp.float16, packed=True),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            thread_barrier=plgpu.Barrier(orders_tensor_core=True),
            mma_barrier=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
        num_threads=warp_specialize.thread_count(2),
        thread_name='t',
    )
    def _kernel(
        out_ref, acc_tmem, lhs_tmem, a_smem, b_smem, thread_barrier, mma_barrier
    ):
      @warp_specialize.maybe_warp_specialize(thread_name='t')
      def _(warp_id):

        @pl.when(warp_id == 0)
        def _():
          plgpu.async_copy_smem_to_tmem(a_smem, lhs_tmem)
          plgpu.barrier_arrive(thread_barrier)

        @pl.when(warp_id == 1)
        def _():
          plgpu.barrier_wait(thread_barrier)
          plgpu.tcgen05_mma(
              acc_tmem, lhs_tmem, b_smem, mma_barrier, accumulate=False
          )
          plgpu.barrier_wait(mma_barrier)
          out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    _kernel()
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(
      warp_specialize=[WarpSpecializeHelper(False), WarpSpecializeHelper(True)],
  )
  def test_mma_barrier_wait_synchronizes_consumer_thread(self, warp_specialize):
    # Negative test (PTX "non-pipelined instructions, different thread"): the
    # mbarrier completion mechanism works across threads. A consumer waiting
    # on the barrier the producer's MMA arrives on may read the accumulator.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            mma_barrier=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
        num_threads=warp_specialize.thread_count(2),
        thread_name='t',
    )
    def _kernel(out_ref, acc_tmem, a_smem, b_smem, mma_barrier):
      @warp_specialize.maybe_warp_specialize(thread_name='t')
      def _(warp_id):

        @pl.when(warp_id == 0)
        def _():
          plgpu.tcgen05_mma(
              acc_tmem, a_smem, b_smem, mma_barrier, accumulate=False
          )

        @pl.when(warp_id == 1)
        def _():
          plgpu.barrier_wait(mma_barrier)
          out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    _kernel()
    self.assertFalse(mosaic_interpret.get_races().races_found)

  # ----------------------------------------------------------------------
  # Barrier arrival aliasing
  # ----------------------------------------------------------------------

  def test_tma_arrival_does_not_signal_mma_completion(self):
    # A barrier shared between a TMA copy and an MMA can complete its wait on
    # whichever arrives first, so the wait proves nothing about the MMA:
    # reading the accumulator after the first wait is a race.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            data_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            gmem_ref=plgpu.GMEM(RHS_SHAPE, jnp.float16),
            barrier_ref=plgpu.Barrier(num_arrivals=1, orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(
        out_ref, acc_tmem, a_smem, b_smem, data_smem, gmem_ref, barrier_ref
    ):
      plgpu.copy_gmem_to_smem(gmem_ref, data_smem, barrier_ref)
      plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, barrier_ref, accumulate=False)
      # This wait may be satisfied by the TMA's arrival instead of the MMA's.
      plgpu.barrier_wait(barrier_ref)
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)
      # Observe the barrier's second phase before the kernel ends.
      plgpu.barrier_wait(barrier_ref)

    # The interpreter may flag this bug either as a data race or as a barrier
    # phase-invariant violation (the second arrival can complete phase 1
    # before the wait observes phase 0).
    try:
      _kernel()
      flagged = mosaic_interpret.get_races().races_found
    except Exception as e:
      if 'barrier' not in str(e).lower():
        raise
      flagged = True
    self.assertTrue(flagged)

  @jtu.parameterized.product(wait_on=[0, 1, 2])
  def test_can_commit_mma_to_multiple_barriers(self, wait_on):
    # The completion of an MMA can be used to satisfy the arrival condition of
    # multiple barriers as long as they have the orders_tensor_core=True.
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            barrier=plgpu.Barrier(orders_tensor_core=True, num_barriers=3),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(a, b, out_ref, acc_tmem, a_smem, b_smem, barrier):
      a_smem[...] = a[...]
      b_smem[...] = b[...]
      plgpu.commit_smem()
      plgpu.tcgen05_mma(
          acc_tmem, a_smem, b_smem, barrier.at[0], accumulate=False
      )
      plgpu.tcgen05_commit_arrive(barrier.at[1])
      plgpu.tcgen05_commit_arrive(barrier.at[2])

      plgpu.barrier_wait(barrier.at[wait_on])
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)
      for i in range(3):
        if i != wait_on:
          plgpu.barrier_wait(barrier.at[i])

    a = jax.random.uniform(
        jax.random.key(0), shape=LHS_SHAPE, dtype=jnp.float16
    )
    b = jax.random.uniform(
        jax.random.key(1), shape=RHS_SHAPE, dtype=jnp.float16
    )
    result = _kernel(a, b)
    expected = jnp.dot(a, b, preferred_element_type=jnp.float32)
    np.testing.assert_allclose(result, expected)
    self.assertFalse(mosaic_interpret.get_races().races_found)

  def test_can_deallocate_tmem_while_mma_active_on_different_tmem(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            mma_barrier=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(a, b, out_ref, acc_tmem, a_smem, b_smem, mma_barrier):
      a_smem[...] = a[...]
      b_smem[...] = b[...]
      plgpu.commit_smem()

      @functools.partial(
          pl.run_scoped,
          dummy_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
      )
      def _(dummy_tmem):
        plgpu.tcgen05_mma(
            acc_tmem, a_smem, b_smem, mma_barrier, accumulate=False
        )

      plgpu.barrier_wait(mma_barrier)
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    a = jax.random.uniform(
        jax.random.key(0), shape=LHS_SHAPE, dtype=jnp.float16
    )
    b = jax.random.uniform(
        jax.random.key(1), shape=RHS_SHAPE, dtype=jnp.float16
    )
    result = _kernel(a, b)
    expected = jnp.dot(a, b, preferred_element_type=jnp.float32)
    np.testing.assert_allclose(result, expected)
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(
      warp_specialize=[WarpSpecializeHelper(False), WarpSpecializeHelper(True)],
  )
  def test_can_pipeline_with_multiple_parents(self, warp_specialize):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc_smem=plgpu.SMEM(ACC_SHAPE, jnp.float32),
            acc_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            a_tmem=plgpu.TMEM(LHS_SHAPE, jnp.float16, packed=True),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            thread_barrier=plgpu.Barrier(
                num_arrivals=2, orders_tensor_core=True
            ),
            mma_barrier=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
        num_threads=warp_specialize.thread_count(3),
        thread_name='t',
    )
    def _kernel(
        acc_init,
        a,
        b,
        out_ref,
        acc_smem,
        acc_tmem,
        a_tmem,
        a_smem,
        b_smem,
        thread_barrier,
        mma_barrier,
    ):
      @warp_specialize.maybe_warp_specialize(thread_name='t')
      def _(warp_id):

        @pl.when(warp_id == 0)
        def _():
          a_smem[...] = a[...]
          plgpu.commit_smem()
          plgpu.async_copy_smem_to_tmem(a_smem, a_tmem)
          a[...] = jnp.zeros(LHS_SHAPE, jnp.float16)
          plgpu.barrier_arrive(thread_barrier)

        @pl.when(warp_id == 1)
        def _():
          acc_smem[...] = acc_init[...]
          plgpu.commit_smem()
          plgpu.async_copy_smem_to_tmem(acc_smem, acc_tmem)
          plgpu.barrier_arrive(thread_barrier)

        @pl.when(warp_id == 2)
        def _():
          plgpu.barrier_wait(thread_barrier)
          b_smem[...] = b[...]
          plgpu.commit_smem()
          plgpu.tcgen05_mma(acc_tmem, a_tmem, b_smem, mma_barrier)
          plgpu.barrier_wait(mma_barrier)
          out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    a = jax.random.uniform(
        jax.random.key(0), shape=LHS_SHAPE, dtype=jnp.float16
    )
    b = jax.random.uniform(
        jax.random.key(1), shape=RHS_SHAPE, dtype=jnp.float16
    )
    acc_init = jnp.full(ACC_SHAPE, 1.0, jnp.float32)
    result = _kernel(acc_init, a, b)
    expected = acc_init + jnp.dot(a, b, preferred_element_type=jnp.float32)
    np.testing.assert_allclose(result, expected)
    self.assertFalse(mosaic_interpret.get_races().races_found)

  def test_can_pipeline_with_multiple_children(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct(ACC_SHAPE, jnp.float32),
        scratch_types=dict(
            acc1_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            acc2_tmem=plgpu.TMEM(ACC_SHAPE, jnp.float32),
            lhs_tmem=plgpu.TMEM(LHS_SHAPE, jnp.float16, packed=True),
            a_smem=plgpu.SMEM(LHS_SHAPE, jnp.float16),
            b1_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            b2_smem=plgpu.SMEM(RHS_SHAPE, jnp.float16),
            mma_barrier1=plgpu.Barrier(orders_tensor_core=True),
            mma_barrier2=plgpu.Barrier(orders_tensor_core=True),
        ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(
        a,
        b1,
        b2,
        out_ref,
        acc1_tmem,
        acc2_tmem,
        lhs_tmem,
        a_smem,
        b1_smem,
        b2_smem,
        mma_barrier1,
        mma_barrier2,
    ):
      a_smem[...] = a[...]
      b1_smem[...] = b1[...]
      b2_smem[...] = b2[...]
      plgpu.commit_smem()
      plgpu.async_copy_smem_to_tmem(a_smem, lhs_tmem)
      plgpu.tcgen05_mma(
          acc1_tmem, lhs_tmem, b1_smem, mma_barrier1, accumulate=False
      )
      plgpu.tcgen05_mma(
          acc2_tmem, lhs_tmem, b2_smem, mma_barrier2, accumulate=False
      )
      plgpu.barrier_wait(mma_barrier1)
      plgpu.barrier_wait(mma_barrier2)
      out_ref[...] = plgpu.async_load_tmem(acc1_tmem) + plgpu.async_load_tmem(
          acc2_tmem
      )

    a = jax.random.uniform(
        jax.random.key(0), shape=LHS_SHAPE, dtype=jnp.float16
    )
    b1 = jax.random.uniform(
        jax.random.key(1), shape=RHS_SHAPE, dtype=jnp.float16
    )
    b2 = jax.random.uniform(
        jax.random.key(2), shape=RHS_SHAPE, dtype=jnp.float16
    )
    result = _kernel(a, b1, b2)
    expected = jnp.dot(a, b1, preferred_element_type=jnp.float32) + jnp.dot(
        a, b2, preferred_element_type=jnp.float32
    )
    np.testing.assert_allclose(result, expected)
    self.assertFalse(mosaic_interpret.get_races().races_found)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
