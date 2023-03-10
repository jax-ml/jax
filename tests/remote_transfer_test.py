# Copyright 2021 The JAX Authors.
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
"""Tests for cross host device transfer."""

from absl.testing import absltest
import unittest
import numpy as np

import jax
from jax._src import test_util as jtu

from jax.config import config

config.parse_flags_with_absl()


@jtu.pytest_mark_if_available('multiaccelerator')
class RemoteTransferTest(jtu.JaxTestCase):

  # TODO(jheek): this test crashes on multi-GPU.
  @jtu.skip_on_devices("gpu")
  def test_remote_transfer(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Remote transfer requires at least 2 devices")
    if config.jax_array:
      raise unittest.SkipTest("Array does not have xla_shape method since "
                              "it is deprecated.")
    dev_a, dev_b = jax.local_devices()[:2]
    if "libtpu" in jax.local_devices()[0].client.platform_version:
      raise unittest.SkipTest("Test does not yet work on cloud TPU")
    send_buf = jax.device_put(np.ones((32,)), dev_a)
    shapes = [send_buf.xla_shape()]
    (tag, recv_buf), = dev_b.client.make_cross_host_receive_buffers(
        shapes, dev_b)
    status, dispatched = send_buf.copy_to_remote_device(tag)
    self.assertIsNone(status)
    self.assertTrue(dispatched)
    self.assertArraysEqual(send_buf, recv_buf)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
