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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import xla_extension_version
import numpy as np

config.parse_flags_with_absl()
jtu.request_cpu_devices(2)


class StringArrayTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if xla_extension_version < 311:
      self.skipTest(
          "Skipping this test because the current XLA extension version:"
          f" {xla_extension_version} is older than 309, the oldest version with"
          " string array support."
      )
    if not hasattr(np.dtypes, "StringDType"):
      self.skipTest(
          "Skipping this test because the numpy.dtype.StringDType is not"
          " available."
      )

  def make_test_string_array(self, device=None):
    """Makes and returns a simple 2x1 string array on the first CPU device."""
    if device is None:
      cpu_devices = jax.devices("cpu")
      if len(cpu_devices) < 1:
        self.skipTest(
            "Skipping this test because no CPU devices are available."
        )
      device = cpu_devices[0]

    numpy_string_array = np.array(
        ["abcd", "efgh"], dtype=np.dtypes.StringDType()  # type: ignore
    )
    jax_string_array = jax.device_put(numpy_string_array, device=device)
    jax_string_array.block_until_ready()
    return jax_string_array

  @parameterized.named_parameters(
      ("asarray", True),
      ("device_put", False),
  )
  @jtu.run_on_devices("cpu")
  def test_single_device_array(self, asarray):
    cpu_devices = jax.devices("cpu")
    if len(cpu_devices) < 1:
      self.skipTest("Skipping this test because no CPU devices are available.")

    numpy_string_array = np.array(
        ["abcdefghijklmnopqrstuvwxyz", "cba"], dtype=np.dtypes.StringDType()  # type: ignore
    )
    if asarray:
      jax_string_array = jnp.asarray(numpy_string_array, device=cpu_devices[0])
    else:
      jax_string_array = jax.device_put(
          numpy_string_array, device=cpu_devices[0]
      )
    jax_string_array.block_until_ready()

    array_read_back = jax.device_get(jax_string_array)
    self.assertEqual(array_read_back.dtype, np.dtypes.StringDType())  # type: ignore
    np.testing.assert_array_equal(array_read_back, numpy_string_array)

  @parameterized.named_parameters(
      ("asarray", True),
      ("device_put", False),
  )
  @jtu.run_on_devices("cpu")
  def test_multi_device_array(self, asarray):
    cpu_devices = jax.devices("cpu")
    if len(cpu_devices) < 2:
      self.skipTest(
          f"Skipping this test because only {len(cpu_devices)} host"
          " devices are available. Need at least 2."
      )

    numpy_string_array = np.array(
        [["abcd", "efgh"], ["ijkl", "mnop"]], dtype=np.dtypes.StringDType()  # type: ignore
    )
    mesh = jax.sharding.Mesh(np.array(cpu_devices)[:2].reshape((2, 1)), ("x", "y"))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x", "y")
    )

    if asarray:
      jax_string_array = jnp.asarray(numpy_string_array, device=sharding)
    else:
      jax_string_array = jax.device_put(numpy_string_array, device=sharding)
    jax_string_array.block_until_ready()

    array_read_back = jax.device_get(jax_string_array)
    self.assertEqual(array_read_back.dtype, np.dtypes.StringDType())  # type: ignore
    np.testing.assert_array_equal(array_read_back, numpy_string_array)

  @jtu.run_on_devices("cpu")
  def test_dtype_conversions(self):
    cpu_devices = jax.devices("cpu")
    if len(cpu_devices) < 1:
      self.skipTest("Skipping this test because no CPU devices are available.")

    # Explicitly specifying the dtype should work with StringDType numpy arrays.
    numpy_string_array = np.array(
        ["abcd", "efgh"], dtype=np.dtypes.StringDType()  # type: ignore
    )
    jax_string_array = jnp.asarray(
        numpy_string_array,
        device=cpu_devices[0],
        dtype=np.dtypes.StringDType(),
    )  # type: ignore
    jax_string_array.block_until_ready()

    # Cannot make a non-StringDType array from a StringDType numpy array.
    with self.assertRaisesRegex(
        TypeError,
        r"Cannot make an array with dtype bfloat16 from an object with dtype"
        r" StringDType.*",
    ):
      jnp.asarray(
          numpy_string_array,
          device=cpu_devices[0],
          dtype=jnp.bfloat16,
      )

    # Cannot make a StringDType array from a numeric numpy array.
    numpy_int_array = np.arange(2, dtype=np.int32)
    with self.assertRaisesRegex(
        TypeError,
        r"Cannot make an array with dtype StringDType.*from an object with"
        r" dtype int32.",
    ):
      jnp.asarray(
          numpy_int_array,
          device=cpu_devices[0],
          dtype=np.dtypes.StringDType(),  # type: ignore
      )

  @parameterized.named_parameters(
      ("asarray", True),
      ("device_put", False),
  )
  @jtu.skip_on_devices("cpu")
  def test_string_array_cannot_be_non_cpu_devices(self, asarray):
    devices = jax.devices()
    if len(devices) < 1:
      self.skipTest("Skipping this test because no devices are available.")

    numpy_string_array = np.array(
        ["abcdefghijklmnopqrstuvwxyz", "cba"], dtype=np.dtypes.StringDType()  # type: ignore
    )
    with self.assertRaisesRegex(
        TypeError, "String arrays can only be sharded to CPU devices"
    ):
      if asarray:
        jax_string_array = jnp.asarray(numpy_string_array, device=devices[0])
      else:
        jax_string_array = jax.device_put(numpy_string_array, device=devices[0])
      jax_string_array.block_until_ready()

  def test_jit_fails_with_string_arrays(self):
    f = jax.jit(lambda x: x)
    input_array = self.make_test_string_array()
    self.assertRaisesRegex(
        TypeError,
        r"Argument.*is not a valid JAX type.",
        lambda: f(input_array),
    )

  def test_grad_fails_with_string_arrays(self):
    f = jax.grad(lambda x: x)
    input_array = self.make_test_string_array()
    self.assertRaisesRegex(
        TypeError,
        r"Argument.*is not a valid JAX type.",
        lambda: f(input_array),
    )

  def test_vmap_without_jit_works_with_string_arrays(self):
    f = jax.vmap(lambda x: x)
    input_array = self.make_test_string_array()
    output_array = f(input_array)
    self.assertEqual(output_array.dtype, input_array.dtype)
    np.testing.assert_array_equal(output_array, input_array)

  def test_vmap_with_jit_fails_with_string_arrays(self):
    f = jax.vmap(lambda x: x + jnp.arange(2))
    input_array = self.make_test_string_array()
    self.assertRaisesRegex(
        ValueError,
        r".*StringDType.*is not a valid dtype",
        lambda: f(input_array),
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
