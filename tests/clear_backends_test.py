# Copyright 2022 The JAX Authors.
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
"""Tests for release_backend_clients."""

import unittest

from absl.testing import absltest

import jax
from jax import config
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax._src.lib import xla_extension_version

config.parse_flags_with_absl()


class ClearBackendsTest(jtu.JaxTestCase):

  def test_clear_backends(self):
    if xla_extension_version < 164 and xb.using_pjrt_c_api():
      raise unittest.SkipTest('test crashes runtime with PJRT C API')

    g = jax.jit(lambda x, y: x * y)
    self.assertEqual(g(1, 2), 2)
    self.assertNotEmpty(xb.get_backend().live_executables())
    jax.clear_backends()
    self.assertEmpty(xb.get_backend().live_executables())
    self.assertEqual(g(1, 2), 2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
