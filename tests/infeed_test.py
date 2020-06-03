# Copyright 2019 Google LLC
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


import threading

from absl.testing import absltest
import jax
from jax import lax, numpy as np
from jax.config import config
from jax.lib import xla_client
import jax.test_util
import numpy as onp

config.parse_flags_with_absl()
FLAGS = config.FLAGS

class InfeedTest(jax.test_util.JaxTestCase):

  def testInfeed(self):
    @jax.jit
    def f(x):
      token = lax.create_token(x)
      (y,), token = lax.infeed(
          token, shape=(jax.ShapedArray((3, 4), np.float32),))
      (z,), _ = lax.infeed(
          token, shape=(jax.ShapedArray((3, 1, 1), np.float32),))
      return x + y + z

    x = onp.float32(1.5)
    y = onp.reshape(onp.arange(12, dtype=onp.float32), (3, 4)) # onp.random.randn(3, 4).astype(onp.float32)
    z = onp.random.randn(3, 1, 1).astype(onp.float32)
    device = jax.local_devices()[0]
    device.transfer_to_infeed((y,))
    device.transfer_to_infeed((z,))
    self.assertAllClose(f(x), x + y + z)

  def testInfeedThenOutfeed(self):
    @jax.jit
    def f(x):
      token = lax.create_token(x)
      y, token = lax.infeed(
          token, shape=jax.ShapedArray((3, 4), np.float32))
      token = lax.outfeed(token, y + onp.float32(1))
      return lax.tie_in(token, x - 1)

    x = onp.float32(7.5)
    y = onp.random.randn(3, 4).astype(onp.float32)
    execution = threading.Thread(target=lambda: f(x))
    execution.start()
    device = jax.local_devices()[0]
    device.transfer_to_infeed((y,))
    out, = device.transfer_from_outfeed(
      xla_client.shape_from_pyval((y,)).with_major_to_minor_layout_if_absent())
    execution.join()
    self.assertAllClose(out, y + onp.float32(1))

  def testInfeedThenOutfeedInALoop(self):
    def doubler(_, token):
      y, token = lax.infeed(
          token, shape=jax.ShapedArray((3, 4), np.float32))
      return lax.outfeed(token, y * onp.float32(2))

    @jax.jit
    def f(n):
      token = lax.create_token(n)
      token = lax.fori_loop(0, n, doubler, token)
      return lax.tie_in(token, n)

    device = jax.local_devices()[0]
    n = 10
    execution = threading.Thread(target=lambda: f(n))
    execution.start()
    for _ in range(n):
      x = onp.random.randn(3, 4).astype(onp.float32)
      device.transfer_to_infeed((x,))
      y, = device.transfer_from_outfeed(xla_client.shape_from_pyval((x,))
                                        .with_major_to_minor_layout_if_absent())
      self.assertAllClose(y, x * onp.float32(2))
    execution.join()


if __name__ == '__main__':
  absltest.main()
