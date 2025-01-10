# Copyright 2024 The JAX Authors.
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

from absl import logging
from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import xla_extension_version  # pylint: disable=g-importing-member
import numpy as np

try:
  import numpy.dtypes as np_dtypes  # pylint: disable=g-import-not-at-top
except ImportError:
  np_dtypes = None  # type: ignore

config.parse_flags_with_absl()
jtu.request_cpu_devices(2)


class StringArrayTest(jtu.JaxTestCase):

  # @jtu.run_on_devices("cpu")
  def test_wip(self):
    host_devices = jax.devices("cpu")
    logging.info("2DO host_devices: %s", host_devices)
    logging.info("2DO host_devices[0]: %s", host_devices[0].device_kind)
    if xla_extension_version < 304:
      self.skipTest("Skipping test because the XLA version is too old.")

    if not hasattr(np_dtypes, "StringDType"):
      self.skipTest(
          "Skipping test because the numpy.dtype.StringDType is not available."
      )

    # sharding = jax.sharding.PositionalSharding(host_devices)
    # a = np.array([b"abc", b"defgh"], dtype=np.bytes_)
    # numpy_string_array = np.array([b"abc", b"defgh"], dtype=np.object_)
    numpy_string_array = np.array(
        ["abcdefghijklmnopqrstuvwxyz", "cba"], dtype=np_dtypes.StringDType()  # type: ignore
    )
    logging.info(
        "2DO type(numpy_string_array): %s; dtype: %s",
        type(numpy_string_array),
        numpy_string_array.dtype,
    )
    logging.info("2DO numpy_string_array: %s", numpy_string_array)

    # ja = jax.device_put(a, device=host_devices[0]) # Add a test for device_put.
    jax_string_array = jnp.asarray(numpy_string_array, device=host_devices[0])
    logging.info(
        "2DO type(ja): %s; dtype: %s",
        type(jax_string_array),
        jax_string_array.dtype,
    )
    jax_string_array.block_until_ready()
    logging.info("2DO sharding: %s", jax_string_array.sharding)
    logging.info("2DO change here is picked up")

    # read = client.experimental_device_get(jax_string_array)
    read = jax.device_get(jax_string_array)
    logging.info("2DO type(read): %s; dtype: %s", type(read), read.dtype)
    logging.info("2DO read: %s", read)
    self.assertEqual(read.dtype, np_dtypes.StringDType())  # type: ignore
    np.testing.assert_array_equal(read, numpy_string_array)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
