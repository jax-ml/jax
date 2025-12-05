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
"""Tests for backwards compatibility of serialization of JAX exports.

Whenever we change the serialization format for jax.export.Exported
(see file jax.export.serialization), we should first save a serialization
of the current format and add a test that it can be deserialized and it has
the expected behavior.

To add a new test:

  * Create a new test method, with a function to be serialized that exercises
    the feature you want to test, and a call to self.export_and_serialize.
  * Run the test. This will save the serialized data in
    TEST_UNDECLARED_OUTPUTS_DIR (or "/tmp/back_compat_testdata" if not set).
  * Copy the test data defined in the output file, to the file
    jax._src.internal_test_util.export_back_compat_test_data.export_{name}.py.
  * Add a new import statement to this file to import that module
  * Add a new subtest to the test that deserializes the data and checks that it
    has the expected behavior.

This process will ensure that the saved serialized export can be read by
future code version (backward compatibility of the deserializer). To check
forward compatibility you'd have to try to attempt to run the serialized
export with a previous code version. You can do this manually.
"""
import datetime
import logging
import os
import re

from absl.testing import absltest
import numpy as np

# ruff: noqa: F401
try:
  import flatbuffers
  CAN_SERIALIZE = True
except (ModuleNotFoundError, ImportError):
  CAN_SERIALIZE = False

import jax
from jax._src import config
from jax._src.export import _export
from jax._src.export.serialization import _SERIALIZATION_VERSION
from jax.sharding import PartitionSpec as P
from jax._src import test_util as jtu

from jax._src.internal_test_util.export_back_compat_test_data import export_with_specified_sharding
from jax._src.internal_test_util.export_back_compat_test_data import export_with_unspecified_sharding

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class CompatTest(jtu.JaxTestCase):

  def setUp(self):
    if not CAN_SERIALIZE:
      self.skipTest("Serialization not available")

  def export_and_serialize(self, fun, *args,
                           vjp_order=0,
                           **kwargs) -> bytearray:
    """Export and serialize a function.

    The test data is saved in TEST_UNDECLARED_OUTPUTS_DIR (or
    "/tmp/back_compat_testdata" if not set) and should be copied as explained
    in the module docstring.
    """
    exp = _export.export(fun)(*args, **kwargs)
    serialized = exp.serialize(vjp_order=vjp_order)
    updated_testdata = f"""
# Pasted from the test output (see export_serialization_back_compat_test.py module docstring)
data_v{_SERIALIZATION_VERSION}_{datetime.date.today().strftime('%Y_%m_%d')} = dict(
    testdata_version=1,
    serialization_version={_SERIALIZATION_VERSION},
    exported_serialized={serialized!r},
)  # End paste

"""
    # Replace the word that should not appear.
    updated_testdata = re.sub(r"google.", "googlex", updated_testdata)
    output_dir = os.getenv("TEST_UNDECLARED_OUTPUTS_DIR",
                           "/tmp/back_compat_testdata")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"export_{self._testMethodName}.py")
    logging.info("Writing the updated serialized Exported at %s", output_file)
    with open(output_file, "w") as f:
      f.write(updated_testdata)
    return serialized

  @jtu.parameterized_filterable(
    kwargs=[
      dict(version=version)
      for version in ["current", "v4", "v5"]
    ]
  )
  def test_with_specified_sharding(self, version):
    a = np.arange(16 * 4, dtype=np.float32).reshape((16, 4))
    with jtu.create_mesh((2,), "x") as mesh:
      @jax.jit(in_shardings=(jax.sharding.NamedSharding(mesh, P("x", None),),),
               out_shardings=jax.sharding.NamedSharding(mesh, P(None, "x")))
      def f(b):
        return b * 2.

      a = jax.device_put(a, jax.sharding.NamedSharding(mesh, P("x", None)))
      if version == "current":
        serialized = self.export_and_serialize(f, a)
      elif version == "v4":
        serialized = export_with_specified_sharding.data_v4_2025_11_23["exported_serialized"]
      elif version == "v5":
        serialized = export_with_specified_sharding.data_v5_2025_12_05["exported_serialized"]

      out = _export.deserialize(serialized).call(a)
      self.assertAllClose(out, a * 2.)
      self.assertEqual(out.addressable_shards[0].index, (slice(None), slice(0, 2)))
      self.assertEqual(out.addressable_shards[1].index, (slice(None), slice(2, 4)))


  @jtu.parameterized_filterable(
    kwargs=[
      dict(version=version)
      for version in ["current", "v4", "v5"]
    ]
  )
  def test_with_unspecified_sharding(self, version):
    a = np.arange(16 * 4, dtype=np.float32).reshape((16, 4))

    # Output sharding is not specified
    with jtu.create_mesh((2,), "x") as mesh:
      @jax.jit(in_shardings=(jax.sharding.NamedSharding(mesh, P("x", None),),))
      def f(b):
        return b * 2.

      a = jax.device_put(a, jax.sharding.NamedSharding(mesh, P("x", None)))
      if version == "current":
        serialized = self.export_and_serialize(f, a)
      elif version == "v4":
        serialized = export_with_unspecified_sharding.data_v4_2025_11_23["exported_serialized"]
      elif version == "v5":
        serialized = export_with_unspecified_sharding.data_v5_2025_12_05["exported_serialized"]

      out = _export.deserialize(serialized).call(a)
      self.assertAllClose(out, a * 2.)
      self.assertEqual(out.addressable_shards[0].index, (slice(0, 8), slice(None)))
      self.assertEqual(out.addressable_shards[1].index, (slice(8, 16), slice(None)))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
