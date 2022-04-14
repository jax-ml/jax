# Copyright 2021 Google LLC
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


class RemoteTransferTest(jtu.JaxTestCase):

  def test_remote_transfer(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Remote transfer requires at lest 2 devices")
    dev_a, dev_b = jax.local_devices()[:2]
    if not hasattr(dev_a.client, "make_cross_host_receive_buffers"):
      # TODO(jheek) remove this once a new version of JAX lib is released
      raise unittest.SkipTest("jax-lib doesn't include cross host APIs")
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
