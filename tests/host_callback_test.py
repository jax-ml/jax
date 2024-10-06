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

from __future__ import annotations

from unittest import SkipTest

from absl.testing import absltest

import jax
from jax.experimental import host_callback as hcb
from jax._src import xla_bridge
from jax._src import test_util as jtu

import numpy as np

jax.config.parse_flags_with_absl()


class HostCallbackCallTest(jtu.JaxTestCase):
  """Tests for hcb.call"""

  def setUp(self):
    # skipping here skips teardown, so do this before super().setUp().
    if jtu.test_device_matches(["gpu"]) and jax.device_count() > 1:
      raise SkipTest("host_callback broken on multi-GPU platforms (#6447)")
    if xla_bridge.using_pjrt_c_api():
      raise SkipTest("host_callback not implemented in PJRT C API")
    super().setUp()
    self.enter_context(jtu.ignore_warning(
      category=DeprecationWarning, message="The host_callback APIs are deprecated"))
    self.enter_context(jtu.ignore_warning(
      category=DeprecationWarning, message="backend and device argument"))

  def tearDown(self) -> None:
    jax.effects_barrier()
    super().tearDown()

  def test_call_simple(self):

    def f_outside(x):
      return 2 * x

    def fun(x):
      y = hcb.call(f_outside, x + 1, result_shape=x)
      return 3 * (1 + y)

    arg = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    self.assertAllClose(3 * (1 + 2 * (arg + 1)), fun(arg))


  @jtu.sample_product(
    dtype=[dtype for dtype in jtu.dtypes.all if dtype != np.bool_],
  )
  def test_call_types(self, dtype=np.float64):

    def f_outside(x):
      # Use x + x to ensure that the result type is the same
      return x + x

    def fun(x):
      return hcb.call(f_outside, x + x, result_shape=x)

    arg = np.arange(24, dtype=dtype).reshape((2, 3, 4))
    self.assertAllClose(arg + arg + arg + arg, fun(arg), check_dtypes=True)

  def test_call_types_bool(self, dtype=np.float64):

    def f_outside(x):
      return np.invert(x)

    def fun(x):
      return hcb.call(f_outside, x, result_shape=x)

    arg = self.rng().choice(a=[True, False], size=(2, 3, 4))
    self.assertAllClose(np.invert(arg), fun(arg))

  def test_call_tuples(self):

    def f_outside(args):
      x, y = args
      return y, x  # Swap the tuple

    def fun(x):
      xy = hcb.call(f_outside, (x, x + 1), result_shape=(x, x))
      return 2 * xy[0] + 3 * xy[1]

    arg = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    self.assertAllClose(2 * (arg + 1) + 3 * arg, fun(arg))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
