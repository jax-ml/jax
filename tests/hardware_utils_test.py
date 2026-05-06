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

import unittest
from unittest.mock import Mock, patch
from jax._src import config

from jax._src.hardware_utils import (
  num_available_amd_gpus as count_amd_gpus_impl,
)
from jax._src.hardware_utils import get_shm_size_in_mb as get_shm_size_impl

config.parse_flags_with_absl()


class TestCountAmdGpus(unittest.TestCase):
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


class TestGetShmSize(unittest.TestCase):
  """Test shared memory size checking."""

  def test_dev_shm_exists(self):
    """Should return shm size in MB when /dev/shm exists."""
    with patch("os.path.exists", return_value=True):
      with patch("os.statvfs") as mock_statvfs:
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
      with patch("os.statvfs", side_effect=OSError("Error")):
        size = get_shm_size_impl()
        self.assertEqual(size, 0)


if __name__ == "__main__":
  unittest.main()
