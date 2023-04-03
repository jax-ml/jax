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
"""Tests for GlobalDeviceArray."""

import contextlib
import os
import unittest
from absl.testing import absltest
import numpy as np

import jax
from jax._src import core
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax.experimental.pjit import pjit
from jax.experimental.serialize_executable import (
    serialize, deserialize_and_load)
from jax.sharding import PartitionSpec as P

from jax.config import config
config.parse_flags_with_absl()

prev_xla_flags = None

with contextlib.suppress(ImportError):
  import pytest
  pytestmark = pytest.mark.multiaccelerator


# Run all tests with 8 CPU devices.
def setUpModule():
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=8")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xb.get_backend.cache_clear()

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xb.get_backend.cache_clear()


class JaxAotTest(jtu.JaxTestCase):

  def check_for_compile_options(self):
    example_exe = jax.jit(lambda x: x * x).lower(
        core.ShapedArray(
            (2, 2), dtype=np.float32)).compile()._executable.xla_executable

    # Skip if CompileOptions is not available. This is true on
    # CPU/GPU/Cloud TPU for now.
    try:
      example_exe.compile_options()
    except Exception as e:
      if str(e) == 'UNIMPLEMENTED: CompileOptions not available.':
        raise unittest.SkipTest('Serialization not supported')
      raise e

  def test_pickle_pjit_lower(self):
    self.check_for_compile_options()

    def fun(x):
      return x * x

    with jax.sharding.Mesh(np.array(jax.devices()), ('data',)):
      lowered = pjit(
          fun, in_shardings=P('data'), out_shardings=P(None, 'data')
      ).lower(core.ShapedArray(shape=(8, 8), dtype=np.float32))

    def verify_serialization(lowered):
      serialized, in_tree, out_tree = serialize(lowered.compile())
      compiled = deserialize_and_load(serialized, in_tree, out_tree)
      self.assertEqual(compiled.as_text(), lowered.compile().as_text())

    verify_serialization(lowered)
    verify_serialization(jax.jit(lambda x: x * x).lower(np.arange(100)))
    verify_serialization(
        jax.pmap(lambda x: x * x).lower(
            np.zeros((len(jax.devices()), 4), dtype=np.float32)))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
