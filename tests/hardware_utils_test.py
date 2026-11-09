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

"""Unit tests for jax._src.hardware_utils.

These tests verify the hardware utility functions for GPU detection
and shared memory checking.
"""

from unittest.mock import Mock, patch
from absl.testing import absltest
from jax._src import config
from jax._src import test_util as jtu

from jax._src.hardware_utils import (
  TpuVersion,
  num_available_amd_gpus as count_amd_gpus_impl,
  num_available_tpu_chips_and_device_id,
)
from jax._src.hardware_utils import get_shm_size_in_mb as get_shm_size_impl

config.parse_flags_with_absl()


class TestCountAmdGpus(absltest.TestCase):
  """Test GPU counting logic exists and has correct signature."""

  def test_count_amd_gpus_callable(self):
    """count_amd_gpus should be callable."""
    self.assertTrue(callable(count_amd_gpus_impl))

  def test_count_amd_gpus_with_stop_at(self):
    """count_amd_gpus should accept stop_at parameter."""
    # Just verify it can be called with stop_at - don't check the result
    # since that would require real GPUs or complex mocking
    count = count_amd_gpus_impl(stop_at=1)
    self.assertIsInstance(count, int)
    self.assertGreaterEqual(count, 0)

  def test_count_amd_gpus_returns_int(self):
    """count_amd_gpus should return an integer."""
    count = count_amd_gpus_impl()
    self.assertIsInstance(count, int)
    self.assertGreaterEqual(count, 0)


class TestGetShmSize(absltest.TestCase):
  """Test shared memory size checking."""

  def test_dev_shm_exists(self):
    """Should return shm size in MB when /dev/shm exists."""
    with patch("os.path.exists", return_value=True):
      with patch("os.statvfs", create=True) as mock_statvfs:
        mock_stat = Mock()
        mock_stat.f_blocks = 128 * 1024
        mock_stat.f_frsize = 1024
        mock_statvfs.return_value = mock_stat

        size_mb = get_shm_size_impl()
        self.assertEqual(size_mb, 128.0)

  def test_dev_shm_not_exists(self):
    """Should return 0 when /dev/shm doesn't exist."""
    with patch("os.path.exists", return_value=False):
      size = get_shm_size_impl()
      self.assertEqual(size, 0)

  def test_statvfs_exception(self):
    """Should return 0 on statvfs exception."""
    with patch("os.path.exists", return_value=True):
      with patch("os.statvfs", side_effect=OSError("Error"), create=True):
        size = get_shm_size_impl()
        self.assertEqual(size, 0)


class TestNumAvailableTpuChips(absltest.TestCase):
  """Test TPU chip counting logic."""

  def test_ignore_secondary_pci_functions(self):
    """Verify dual virtual functions (e.g. .0 and .1 on TPU 7x) only count .0."""
    fake_paths = [
        "/sys/bus/pci/devices/0000:00:07.0/vendor",
        "/sys/bus/pci/devices/0000:00:07.1/vendor",
        "/sys/bus/pci/devices/0000:00:08.0/vendor",
        "/sys/bus/pci/devices/0000:00:08.1/vendor",
    ]

    def fake_read_text(path_obj):
      path_str = str(path_obj)
      if "vendor" in path_str:
        return "0x1ae0"
      elif "device" in path_str:
        return "0x0076"
      return ""

    with patch("glob.glob", return_value=fake_paths):
      with patch(
          "pathlib.Path.read_text", side_effect=fake_read_text, autospec=True
      ):
        num_chips, tpu_version = num_available_tpu_chips_and_device_id()
        self.assertEqual(num_chips, 2)
        self.assertEqual(tpu_version, TpuVersion.tpu7x)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
