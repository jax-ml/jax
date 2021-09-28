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

import time
import warnings

from absl.testing import absltest
from jax._src import test_util as jtu
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc

mock = absltest.mock


class XlaBridgeTest(jtu.JaxTestCase):

  def test_set_device_assignment_no_partition(self):
    compile_options = xb.get_compile_options(
        num_replicas=4, num_partitions=1, device_assignment=[0, 1, 2, 3])
    expected_device_assignment = ("Computations: 1 Replicas: 4\nComputation 0: "
                                  "0 1 2 3 \n")
    self.assertEqual(compile_options.device_assignment.__repr__(),
                     expected_device_assignment)

  def test_set_device_assignment_with_partition(self):
    compile_options = xb.get_compile_options(
        num_replicas=2, num_partitions=2, device_assignment=[[0, 1], [2, 3]])
    expected_device_assignment = ("Computations: 2 Replicas: 2\nComputation 0: "
                                  "0 2 \nComputation 1: 1 3 \n")
    self.assertEqual(compile_options.device_assignment.__repr__(),
                     expected_device_assignment)

  def test_parameter_replication_default(self):
    c = xb.make_computation_builder("test")
    _ = xb.parameter(c, 0, xc.Shape.array_shape(xc.PrimitiveType.F32, ()))
    built_c = c.Build()
    assert "replication" not in built_c.as_hlo_text()

  def test_parameter_replication(self):
    c = xb.make_computation_builder("test")
    _ = xb.parameter(c, 0, xc.Shape.array_shape(xc.PrimitiveType.F32, ()), "",
                     False)
    built_c = c.Build()
    assert "parameter_replication={false}" in built_c.as_hlo_text()

  def test_local_devices(self):
    self.assertNotEmpty(xb.local_devices())
    with self.assertRaisesRegex(ValueError, "Unknown process_index 100"):
      xb.local_devices(100)
    with self.assertRaisesRegex(RuntimeError, "Unknown backend foo"):
      xb.local_devices(backend="foo")

  def test_timer_tpu_warning(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      def _mock_tpu_client():
        time_to_wait = 5
        start = time.time()
        while not w:
          if time.time() - start > time_to_wait:
            raise ValueError(
                "This test should not hang for more than "
                f"{time_to_wait} seconds.")
          time.sleep(0.1)

        self.assertLen(w, 1)
        msg = str(w[-1].message)
        self.assertIn("Did you run your code on all TPU hosts?", msg)

      with mock.patch("jax._src.lib.xla_client.make_tpu_client",
                      side_effect=_mock_tpu_client):
        xb.tpu_client_timer_callback(0.01)


class GetBackendTest(jtu.JaxTestCase):

  class _DummyBackend:
    def __init__(self, platform, device_count):
      self.platform = platform
      self._device_count = device_count

    def device_count(self):
      return self._device_count

    def process_index(self):
      return 0

    def local_devices(self):
      return []

  def _register_factory(self, platform: str, priority, device_count=1):
    xb.register_backend_factory(
        platform, lambda: self._DummyBackend(platform, device_count),
        priority=priority)

  def setUp(self):
    self._orig_factories = xb._backend_factories
    xb._backend_factories = {}
    self._save_backend_state()
    self._reset_backend_state()

    # get_backend logic assumes CPU platform is always present.
    self._register_factory("cpu", 0)

  def tearDown(self):
    xb._backend_factories = self._orig_factories
    self._restore_backend_state()

  def _save_backend_state(self):
    self._orig_backends = xb._backends
    self._orig_backends_errors = xb._backends_errors
    self._orig_default_backend = xb._default_backend

  def _reset_backend_state(self):
    xb._backends = {}
    xb._backends_errors = {}
    xb._default_backend = None
    xb.get_backend.cache_clear()

  def _restore_backend_state(self):
    xb._backends = self._orig_backends
    xb._backends_errors = self._orig_backends_errors
    xb._default_backend = self._orig_default_backend
    xb.get_backend.cache_clear()

  def test_default(self):
    self._register_factory("platform_A", 20)
    self._register_factory("platform_B", 10)

    backend = xb.get_backend()
    self.assertEqual(backend.platform, "platform_A")
    # All backends initialized.
    self.assertEqual(len(xb._backends), len(xb._backend_factories))

  def test_specific_platform(self):
    self._register_factory("platform_A", 20)
    self._register_factory("platform_B", 10)

    backend = xb.get_backend("platform_B")
    self.assertEqual(backend.platform, "platform_B")
    # All backends initialized.
    self.assertEqual(len(xb._backends), len(xb._backend_factories))

  def test_unknown_backend_error(self):
    with self.assertRaisesRegex(RuntimeError, "Unknown backend foo"):
      xb.get_backend("foo")

  def test_backend_init_error(self):
    def factory():
      raise RuntimeError("I'm not a real backend")

    xb.register_backend_factory("error", factory, priority=10)
    # No error raised if there's a fallback backend.
    default_backend = xb.get_backend()
    self.assertEqual(default_backend.platform, "cpu")

    with self.assertRaisesRegex(RuntimeError, "I'm not a real backend"):
      xb.get_backend("error")

  def test_no_devices(self):
    self._register_factory("no_devices", -10, device_count=0)
    default_backend = xb.get_backend()
    self.assertEqual(default_backend.platform, "cpu")
    with self.assertRaisesRegex(
        RuntimeError,
        "Backend 'no_devices' failed to initialize: "
        "Backend 'no_devices' provides no devices."):
      xb.get_backend("no_devices")

    self._reset_backend_state()

    self._register_factory("no_devices2", 10, device_count=0)
    default_backend = xb.get_backend()
    self.assertEqual(default_backend.platform, "cpu")
    with self.assertRaisesRegex(
        RuntimeError,
        "Backend 'no_devices2' failed to initialize: "
        "Backend 'no_devices2' provides no devices."):
      xb.get_backend("no_devices2")

  def test_factory_returns_none(self):
    xb.register_backend_factory("none", lambda: None, priority=10)
    default_backend = xb.get_backend()
    self.assertEqual(default_backend.platform, "cpu")
    with self.assertRaisesRegex(
        RuntimeError,
        "Backend 'none' failed to initialize: "
        "Could not initialize backend 'none'"):
      xb.get_backend("none")

  def cpu_fallback_warning(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      xb.get_backend()
      self.assertLen(w, 1)
      msg = str(w[-1].message)
      self.assertIn("No GPU/TPU found, falling back to CPU", msg)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
