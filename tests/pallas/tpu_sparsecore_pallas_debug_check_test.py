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
"""SparseCore Pallas tests with runtime assertions.

Runtime assertions halt TPU execution, which can cause subsequent tests to get
stuck. Therefore, each test with a failing assertion should run in a separate
process. By separating these tests from the rest, we can set the shard count
such that each test runs in its own shard.

The test class in this file makes an attempt to detect the simple scenario where
there are more test methods in the module than shards.
"""

import functools
import os
import sys
import unittest

from absl.testing import absltest
from absl import flags
import jax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


jax.config.parse_flags_with_absl()


class DebugCheckTest(jtu.JaxTestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    total_shards = int(os.environ.get("TEST_TOTAL_SHARDS", -1))
    if total_shards == -1:
      raise unittest.SkipTest("Tests can only be run with Bazel.")

    loader = unittest.TestLoader()
    test_cases = loader.loadTestsFromModule(
        sys.modules['__main__']
    ).countTestCases()
    if test_cases > total_shards:
      raise RuntimeError(
          "Each test with a failing assertion should be in a separate test"
          " shard because they put the hardware in a halt state, causing"
          " subsequent tests to fail. Make sure sharding is enabled and the"
          f" shard count is at least {test_cases}."
      )

  def setUp(self):
    if not jtu.is_device_tpu(5, "p") and not jtu.is_device_tpu_at_least(6):
      self.skipTest("SparseCore only supported on TPU v5p+")

    super().setUp()

  def test_scalar_debug_check(self):
    if not jtu.is_device_tpu_at_least(7):
      # TODO: b/469486032 - Figure out why the test gets stuck on v5p, v6e.
      self.skipTest("Fails on v5p and v6e.")

    x = jnp.arange(8)

    @pl.kernel(
        out_shape=x,
        mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1),
    )
    def kernel(o_hbm_ref):
      @functools.partial(
          pl.run_scoped,
          sem=pltpu.SemaphoreType.DMA,
      )
      def _(sem):
        pltpu.async_copy(o_hbm_ref, o_hbm_ref, sem).wait()
        pl.debug_check(True, "Check success!")
        pl.debug_check(False, "Check failure!")

    with pl.enable_debug_checks(), self.assertRaises(
        jax.errors.JaxRuntimeError
    ) as error:
      jax.block_until_ready(kernel())

    self.assertNotIn("Check success!", str(error.exception))
    self.assertIn("Check failure!", str(error.exception))
    self.assertIn(
        "check at DebugCheckTest.test_scalar_debug_check", str(error.exception)
    )

  def test_vector_debug_check(self):
    x = jnp.arange(8)

    @functools.partial(
        pl.pallas_call,
        out_shape=x,
        compiler_params=pltpu.CompilerParams(
            kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE
        ),
    )
    def kernel(_):
      pl.debug_check(True, "Check success!")
      pl.debug_check(False, "Check failure!")

    with pl.enable_debug_checks(), self.assertRaises(
        jax.errors.JaxRuntimeError
    ) as error:
      jax.block_until_ready(kernel())

    self.assertNotIn("Check success!", str(error.exception))
    if jtu.is_cloud_tpu() and jtu.is_device_tpu_at_least(7):
      self.assertIn("Check failure!", str(error.exception))
      self.assertIn(
          "check at DebugCheckTest.test_vector_debug_check",
          str(error.exception),
      )

  def test_trigger_bounds_checker(self):
    if "xla_sc_assert_level" in flags.FLAGS:
      # The test crashes the process anyway, so no need to be clean.
      flags.FLAGS.xla_sc_assert_level = "all-loads-stores"
    else:
      self.skipTest("TODO: Find another way to enable bounds checking.")
    if jtu.is_device_tpu(7, "x"):
      self.skipTest("TODO(b/478798643): Fails on v7x")

    x = jnp.arange(8, dtype=jnp.int32)
    # Index 8 is out-of-bounds.
    indices = jnp.array([0, 1, 2, 3, 4, 5, 6, 8], dtype=jnp.int32)

    @functools.partial(
        pl.pallas_call,
        out_shape=x,
        compiler_params=pltpu.CompilerParams(
            kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE
        ),
    )
    def kernel(x_ref, indices_ref, o_ref):
      o_ref[...] = plsc.load_gather(x_ref, [indices_ref[...]])

    # We expect this to fail with a runtime error from the bounds checker.
    with self.assertRaisesRegex(
        jax.errors.JaxRuntimeError,
        "Trying to perform an indexed vector load from out of bounds address.",
    ):
      jax.block_until_ready(kernel(x, indices))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
