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
"""Tests for allow/disallow_host_transfer context manager."""

import unittest
from absl.testing import absltest
from absl.testing import parameterized
from functools import partial
import numpy as np
import pickle

import jax
import jax._src.test_util as jtu
import jax.numpy as jnp

from jax.config import config
config.parse_flags_with_absl()


class DisallowHostTransfersTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      ("str", str),
      ("pickle", pickle.dumps),
      ('np_add', partial(np.add, 1)),
      ('np_asarray', np.asarray),
  )
  def test_disallow_host_transfer(self, func):
    # TODO(rmcilroy): Decide whether cpu platform should behave like gpu/tpu.
    if (jax.default_backend() == "cpu"): return

    device_buffer_1 = jnp.ones(10)
    with jax.disallow_host_transfers():
      with self.assertRaises(RuntimeError):
        func(device_buffer_1)

    device_buffer_2 = jnp.ones(10)
    with jax.disallow_host_transfers():
      with jax.allow_host_transfers():
        func(device_buffer_2)

  def test_nesting(self):
    # TODO(rmcilroy): Decide whether cpu platform should behave like gpu/tpu.
    if (jax.default_backend() == "cpu"): return

    device_buffer = jnp.ones(10)
    with jax.disallow_host_transfers():
      with jax.allow_host_transfers():
        with jax.disallow_host_transfers():
          with self.assertRaises(RuntimeError):
            str(device_buffer)
        # At this scope, should succeed.
        str(device_buffer)

  @jax.disallow_host_transfers()
  def test_annotation(self):
    # TODO(rmcilroy): Decide whether cpu platform should behave like gpu/tpu.
    if (jax.default_backend() == "cpu"): return

    device_buffer = jnp.ones(10)
    with self.assertRaises(RuntimeError):
      str(device_buffer)

  def test_device_get_always_allowed(self):
    device_buffer = jnp.ones(10)
    with jax.disallow_host_transfers():
      str(jax.device_get(device_buffer))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
