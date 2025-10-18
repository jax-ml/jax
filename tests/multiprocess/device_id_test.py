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

import jax
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu


class DeviceIdTest(jt_multiprocess.MultiProcessTest):

  def testDeviceIds(self):
    # TODO(phawkins): TPU process IDs won't necessarily match the global
    # process index.
    if not jtu.test_device_matches(["tpu"]):
      self.assertEqual(
          jax.process_index(),
          jt_multiprocess.MULTIPROCESS_TEST_WORKER_ID.value,
      )
    self.assertLen(
        jax.devices(),
        jt_multiprocess.NUM_PROCESSES.value * jax.local_device_count(),
    )
    self.assertEqual(
        jax.local_devices()[0].process_index,
        jax.process_index(),
    )

  def testPrimitive(self):
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
      self.assertEqual(2, jax.lax.neg(jax.lax.neg(2)))

  def testJit(self):
    """Verifies that local computation works inside a distributed job."""
    x = jax.device_put(1)
    self.assertEqual(x, 1)
    y = jax.jit(lambda x: x + 1)(x)
    self.assertEqual(y, 2)

  # TODO(phawkins): this test CHECK-fails on TPU.
  @jtu.skip_on_devices("tpu")
  def testNonaddressableDeviceToDevicePut(self):
    source_device = jax.local_devices(backend="cpu")[0]
    x = jax.device_put(0, source_device)
    for device in jax.devices():
      if device.process_index != jax.process_index():
        with self.assertRaisesRegex(
            RuntimeError,
            "(Cannot copy array to non-addressable device.*|.*is not a local"
            " device.*)",
        ):
          jax.device_put(x, device)

  def testDefaultDevicePlatformString(self):
    with jax.default_device("cpu"):
      result = jax.jit(lambda x: x + 1)(1)
    self.assertEqual(result.device.platform, "cpu")
    self.assertEqual(result.device, jax.local_devices(backend="cpu")[0])

    result = jax.jit(lambda x: x + 1)(1)
    self.assertEqual(result.device.platform, jax.default_backend())
    self.assertEqual(result.device, jax.local_devices()[0])

  # def testCrossProcessReduceScatter(self):
  #   i = multiprocess_test.MULTIPROCESS_TEST_WORKER_ID.value
  #   n = multiprocess_test.NUM_PROCESSES.value
  #   f = jax.pmap(
  #       lambda x: lax.psum_scatter(
  #           x,
  #           "i",
  #       ),
  #       axis_name="i",
  #   )
  #   x = np.arange(n * n).reshape(n, n)
  #   out = f(x[i : i + 1])
  #   expected = np.sum(x, axis=0)
  #   np.testing.assert_allclose(expected[i : i + 1], out)


if __name__ == "__main__":
  jt_multiprocess.main()
