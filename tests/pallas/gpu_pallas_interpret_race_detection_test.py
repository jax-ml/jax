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

jax.config.parse_flags_with_absl()


@jtu.thread_unsafe_test_class()
class InterpretTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    mosaic_interpret.gpu_callbacks.reset_gpu_interpret_mode_state()

    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

    self.num_devices = jax.device_count()
    if self.num_devices > 1:
      self.skipTest(f'requires 1 device, found {self.num_devices}')

  @jtu.parameterized.product(
      wait_barrier=[True, False],
  )
  def test_async_gmem_ops_not_visible_unless_synchronized(self, wait_barrier):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((), jnp.int32),
            barrier_ref=plgpu.Barrier(num_arrivals=1)
            ),
        interpret=InterpretParams(detect_races=True)
    )
    def _kernel(out_ref, smem_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(out_ref, smem_ref, barrier_ref)
      if wait_barrier:
        plgpu.barrier_wait(barrier_ref)
      plgpu.commit_smem()
      out_ref[...] = 42

    out = _kernel()
    if wait_barrier:
      self.assertEqual(out, 42)
      self.assertFalse(mosaic_interpret.get_races().races_found)
    else:
      self.skipTest('Need to implement gmem_commit_clock to detect this')
      self.assertTrue(mosaic_interpret.get_races().races_found)


  @jtu.parameterized.product(
      a=[False, True],
      b=[False, True],
      c=[False, True],
      d=[False, True],
  )
  def test_finds_races_in_single_thread_later_gmem_to_smem_copy(self, a, b, c, d):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((), jnp.int32),
            barrier_ref=plgpu.Barrier(num_arrivals=1)
            ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(in_ref, out_ref, smem_ref, barrier_ref):
      if a: plgpu.commit_smem()
      smem_ref[...] = 100
      if b: plgpu.commit_smem()
      plgpu.copy_gmem_to_smem(in_ref, smem_ref, barrier_ref)
      if c: plgpu.commit_smem()
      plgpu.barrier_wait(barrier_ref)
      if d: plgpu.commit_smem()
      out_ref[...] = smem_ref[...]

    correct = b

    out = _kernel(jnp.int32(42))
    if correct:
      self.assertEqual(out, 42)
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)


  @jtu.parameterized.product(
      wait_barrier=[False, True],
  )
  def test_finds_races_in_single_thread_earlier_gmem_to_smem_copy(self, wait_barrier):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            gmem_ref=plgpu.GMEM((), jnp.int32),
            smem_ref=plgpu.SMEM((), jnp.int32),
            barrier_ref=plgpu.Barrier(num_arrivals=1)
            ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, gmem_ref, smem_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(gmem_ref, smem_ref, barrier_ref)
      if wait_barrier:
        plgpu.barrier_wait(barrier_ref)
      smem_ref[...] = 42
      out_ref[...] = smem_ref[...]

    correct = wait_barrier
    out = _kernel()
    if correct:
      self.assertEqual(out, 42)
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)


  @jtu.parameterized.product(
      a=[False, True],
      b=[False, True],
      c=[False, True],
      d=[False, True],
      e=[False, True],
      f=[False, True],
  )
  def test_finds_races_in_multi_thread_gmem_to_smem_copy(self, a, b, c, d, e, f):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((), jnp.int32),
            thread_barrier=plgpu.Barrier(num_arrivals=1),
            copy_barrier=plgpu.Barrier(num_arrivals=1),
            ),
        interpret=InterpretParams(detect_races=True),
        num_threads=2,
        thread_name='t',
    )
    def _kernel(in_ref, out_ref, smem_ref, thread_barrier, copy_barrier):
      @pl.when(jax.lax.axis_index('t') == 0)
      def _():
        if a: plgpu.commit_smem()
        smem_ref[...] = 100
        if b: plgpu.commit_smem()
        plgpu.copy_gmem_to_smem(in_ref, smem_ref, copy_barrier)
        if c: plgpu.commit_smem()
        plgpu.barrier_arrive(thread_barrier)

      @pl.when(jax.lax.axis_index('t') == 1)
      def _():
        if d: plgpu.commit_smem()
        plgpu.barrier_wait(thread_barrier)
        if e: plgpu.commit_smem()
        plgpu.barrier_wait(copy_barrier)
        if f: plgpu.commit_smem()
        out_ref[...] = smem_ref[...]

    correct = b
    out = _kernel(jnp.int32(42))
    if correct:
      self.assertEqual(out, 42)
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)

  @jtu.parameterized.product(
      a=[False, True],
      b=[False, True],
      c=[False, True],
      d=[False, True],
      e=[False, True],
      f=[False, True],
  )
  def test_finds_races_in_cross_thread_gmem_to_smem_copy(self, a, b, c, d, e, f):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((), jnp.int32),
            thread_barrier=plgpu.Barrier(num_arrivals=1),
            copy_barrier=plgpu.Barrier(num_arrivals=1),
            ),
        interpret=InterpretParams(detect_races=True),
        num_threads=2,
        thread_name='t',
    )
    def _kernel(in_ref, out_ref, smem_ref, thread_barrier, copy_barrier):
      @pl.when(jax.lax.axis_index('t') == 0)
      def _():
        if a: plgpu.commit_smem()
        smem_ref[...] = 100
        if b: plgpu.commit_smem()
        plgpu.barrier_arrive(thread_barrier)

      @pl.when(jax.lax.axis_index('t') == 1)
      def _():
        if c: plgpu.commit_smem()
        plgpu.barrier_wait(thread_barrier)
        if d: plgpu.commit_smem()
        plgpu.copy_gmem_to_smem(in_ref, smem_ref, copy_barrier)
        if e: plgpu.commit_smem()
        plgpu.barrier_wait(copy_barrier)
        if f: plgpu.commit_smem()
        out_ref[...] = smem_ref[...]

    correct = b or d
    out = _kernel(jnp.int32(42))
    if correct:
      self.assertEqual(out, 42)
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)


  @jtu.parameterized.product(
      a=[False, True],
      b=[False, True],
      c=[False, True],
  )
  def test_finds_races_in_single_thread_smem_to_gmem_copy(self, a, b, c):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            gmem_ref=plgpu.GMEM((), jnp.int32),
            smem_ref=plgpu.SMEM((), jnp.int32),
            ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, gmem_ref, smem_ref):
      if a: plgpu.commit_smem()
      smem_ref[...] = 42
      if b: plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref, gmem_ref)
      if c: plgpu.commit_smem()
      plgpu.wait_smem_to_gmem(0)
      out_ref[...] = gmem_ref[...]

    correct = b
    out = _kernel()
    if correct:
      self.assertEqual(out, 42)
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)


  def test_finds_races_in_non_fully_waited_smem_to_gmem_copy(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem1=plgpu.SMEM((), jnp.int32),
            smem2=plgpu.SMEM((), jnp.int32),
            gmem1=plgpu.GMEM((), jnp.int32),
            gmem2=plgpu.GMEM((), jnp.int32),
            ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, smem1, smem2, gmem1, gmem2):
      smem1[...] = 1
      smem2[...] = 2
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem1, gmem1)
      plgpu.copy_smem_to_gmem(smem2, gmem2)
      plgpu.wait_smem_to_gmem(1)
      out_ref[...] = gmem1[...] + gmem2[...]
      plgpu.wait_smem_to_gmem(0)

    _kernel()
    self.assertTrue(mosaic_interpret.get_races().races_found)


  def test_allows_rewaiting_smem_to_gmem_copies(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((3,), jnp.int32),
            gmem_ref=plgpu.GMEM((3,), jnp.int32),
            ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(in_ref, out_ref, smem_ref, gmem_ref):
      smem_ref[...] = in_ref[...]
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref.at[0], gmem_ref.at[0])
      plgpu.copy_smem_to_gmem(smem_ref.at[1], gmem_ref.at[1])
      plgpu.copy_smem_to_gmem(smem_ref.at[2], gmem_ref.at[2])
      plgpu.wait_smem_to_gmem(1)
      plgpu.wait_smem_to_gmem(2)
      out_ref[...] = gmem_ref[0] + gmem_ref[1]
      plgpu.wait_smem_to_gmem(0)
      out_ref[...] = gmem_ref[0] + gmem_ref[1] + gmem_ref[2]

    _kernel(jnp.array([1, 2, 3], dtype=jnp.int32))
    self.assertFalse(mosaic_interpret.get_races().races_found)


  @jtu.parameterized.product(
      wait_only_once=[False, True],
      check_location=list(range(4))
  )
  def test_finds_races_when_smem_to_gmem_copy_waits_read_only(self, wait_only_once, check_location):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2,), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((2,), jnp.int32),
            ),
        interpret=InterpretParams(detect_races=True)
    )
    def _kernel(in_ref, out_ref, smem_ref):
      smem_ref[...] = in_ref[...]
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref.at[0], out_ref.at[0])
      plgpu.copy_smem_to_gmem(smem_ref.at[1], out_ref.at[1])

      # There are 4 possible write locations. Some should cause a race, some
      # shouldn't
      attempt_race_locations = [smem_ref.at[0], out_ref.at[0], smem_ref.at[1], out_ref.at[1]]

      plgpu.wait_smem_to_gmem(1, wait_read_only=True)
      if not wait_only_once:
        plgpu.wait_smem_to_gmem(1)
      attempt_race_locations[check_location][...] = 3

      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    _out = _kernel(jnp.array([1, 2], dtype=jnp.int32))

    skip = (wait_only_once and check_location == 1)
    if skip:
      self.skipTest("Not supported until gmem_commit clock is implemented.")

    correct = (wait_only_once and check_location == 0) or (
        not wait_only_once and check_location in [0, 1]
    )
    self.assertEqual(mosaic_interpret.get_races().races_found, not correct)


  @jtu.parameterized.product(
      check_read=[False, True],
  )
  def test_wait_smem_to_gmem_isolated_per_thread(self, check_read):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem=plgpu.SMEM((), jnp.int32),
            barrier=plgpu.Barrier(num_arrivals=1),
            cleanup_barrier=plgpu.Barrier(num_arrivals=1),
            ),
        interpret=InterpretParams(detect_races=True),
        num_threads=2,
        thread_name='t',
    )
    def _kernel(out_ref, smem, barrier, cleanup_barrier):
      tid = jax.lax.axis_index('t')
      @pl.when(tid == 0)
      def _():
        plgpu.copy_smem_to_gmem(smem, out_ref)
        plgpu.barrier_arrive(barrier)
        plgpu.barrier_wait(cleanup_barrier)
        plgpu.wait_smem_to_gmem(0)

      @pl.when(tid == 1)
      def _():
        plgpu.barrier_wait(barrier)
        plgpu.wait_smem_to_gmem(0)
        if check_read:
          smem[...] = 43
        else:
          out_ref[...] = 43
        plgpu.barrier_arrive(cleanup_barrier)

    _kernel()
    self.assertTrue(mosaic_interpret.get_races().races_found)


  def properly_waited_smem_to_gmem_visible_across_threads(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((), jnp.int32),
            gmem_ref=plgpu.GMEM((), jnp.int32),
            barrier=plgpu.Barrier(num_arrivals=1),
            ),
        interpret=InterpretParams(detect_races=True),
        num_threads=2,
        thread_name='t',
    )
    def _kernel(out_ref, smem_ref, gmem_ref, barrier):
      tid = jax.lax.axis_index('t')
      @pl.when(tid == 0)
      def _():
        smem_ref[...] = 42
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(smem_ref, gmem_ref)
        plgpu.wait_smem_to_gmem(0)
        plgpu.barrier_arrive(barrier)
      @pl.when(tid == 1)
      def _():
        plgpu.barrier_wait(barrier)
        out_ref[...] = gmem_ref[...]

    out = _kernel()
    self.assertFalse(mosaic_interpret.get_races().races_found)
    self.assertEqual(out, 42)


  def test_finds_race_in_double_gmem_to_smem(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((2,), jnp.int32),
            gmem_ref=plgpu.GMEM((2,), jnp.int32),
            barrier=plgpu.Barrier(num_arrivals=2),
            ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(_out_ref, smem_ref, gmem_ref, barrier):
      plgpu.commit_smem()
      plgpu.copy_gmem_to_smem(gmem_ref, smem_ref, barrier)
      plgpu.copy_gmem_to_smem(gmem_ref, smem_ref, barrier)

    _kernel()
    self.assertTrue(mosaic_interpret.get_races().races_found)


  def test_finds_race_in_double_smem_to_gmem(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((2,), jnp.int32),
            gmem_ref=plgpu.GMEM((2,), jnp.int32),
            ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(_out_ref, smem_ref, gmem_ref):
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref, gmem_ref)
      plgpu.copy_smem_to_gmem(smem_ref, gmem_ref)
      plgpu.wait_smem_to_gmem(0)

    _kernel()
    self.assertTrue(mosaic_interpret.get_races().races_found)


  def test_finds_exception_in_unawaited_smem_to_gmem(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2,), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((2,), jnp.int32),
            ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(_out_ref, smem_ref):
      plgpu.copy_smem_to_gmem(smem_ref, _out_ref)

    with self.assertRaisesRegex(
        Exception,
        r'Not all copy_smem_to_gmem read-side operations completed before'
    ):
      _kernel()


  def test_unawaited_smem_to_gmem_write_side_ok(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2,), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((2,), jnp.int32),
            ),
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(_out_ref, smem_ref):
      plgpu.copy_smem_to_gmem(smem_ref, _out_ref)
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    _kernel()



if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
