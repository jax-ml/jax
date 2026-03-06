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

"""Unit tests for ROCm plugin GPU counting and shm checking logic.

These tests mock filesystem operations to test the logic without requiring
real AMD GPUs or specific filesystem structures.
"""

import pathlib
import unittest
from unittest.mock import Mock, patch

# Import the actual functions from jax._src.hardware_utils
from jax._src.hardware_utils import count_amd_gpus as count_amd_gpus_impl
from jax._src.hardware_utils import get_shm_size as get_shm_size_impl


class TestCountAmdGpus(unittest.TestCase):
    """Test GPU counting logic with mocked filesystem."""

    def test_no_kfd_nodes_path_returns_zero(self):
        """No KFD nodes path should return 0."""
        mock_path = Mock(spec=pathlib.Path)
        mock_path.exists.return_value = False
        with patch('pathlib.Path', return_value=mock_path):
            count = count_amd_gpus_impl()
            self.assertEqual(count, 0)

    def test_single_valid_gpu(self):
        """Single GPU with valid simd_count should return 1."""
        # Mock the KFD path
        mock_kfd_path = Mock(spec=pathlib.Path)
        mock_kfd_path.exists.return_value = True

        # Mock a single node
        mock_node = Mock(spec=pathlib.Path)
        mock_props = Mock(spec=pathlib.Path)
        mock_props.exists.return_value = True

        mock_stat = Mock()
        mock_stat.st_size = 1024
        mock_props.stat.return_value = mock_stat
        mock_props.read_text.return_value = "simd_count 10\n"

        mock_node.__truediv__ = Mock(return_value=mock_props)
        mock_kfd_path.iterdir.return_value = [mock_node]

        with patch('pathlib.Path', return_value=mock_kfd_path):
            count = count_amd_gpus_impl()
            self.assertEqual(count, 1)

    def test_multiple_gpus_with_stop_at(self):
        """Multiple GPUs with stop_at should limit count."""
        mock_kfd_path = Mock(spec=pathlib.Path)
        mock_kfd_path.exists.return_value = True

        nodes = []
        for i in range(3):
            mock_node = Mock(spec=pathlib.Path)
            mock_props = Mock(spec=pathlib.Path)
            mock_props.exists.return_value = True

            mock_stat = Mock()
            mock_stat.st_size = 1024
            mock_props.stat.return_value = mock_stat
            mock_props.read_text.return_value = f"simd_count {i+1}\n"

            mock_node.__truediv__ = Mock(return_value=mock_props)
            nodes.append(mock_node)

        mock_kfd_path.iterdir.return_value = nodes

        with patch('pathlib.Path', return_value=mock_kfd_path):
            # Should count all 3
            count = count_amd_gpus_impl()
            self.assertEqual(count, 3)

            # Should stop at 2
            count = count_amd_gpus_impl(stop_at=2)
            self.assertEqual(count, 2)

    def test_zero_simd_count_not_counted(self):
        """Node with simd_count=0 should not be counted."""
        mock_kfd_path = Mock(spec=pathlib.Path)
        mock_kfd_path.exists.return_value = True

        mock_node = Mock(spec=pathlib.Path)
        mock_props = Mock(spec=pathlib.Path)
        mock_props.exists.return_value = True

        mock_stat = Mock()
        mock_stat.st_size = 1024
        mock_props.stat.return_value = mock_stat
        mock_props.read_text.return_value = "simd_count 0\n"

        mock_node.__truediv__ = Mock(return_value=mock_props)
        mock_kfd_path.iterdir.return_value = [mock_node]

        with patch('pathlib.Path', return_value=mock_kfd_path):
            count = count_amd_gpus_impl()
            self.assertEqual(count, 0)

    def test_file_too_large_skipped(self):
        """Files larger than 16KB should be skipped."""
        mock_kfd_path = Mock(spec=pathlib.Path)
        mock_kfd_path.exists.return_value = True

        mock_node = Mock(spec=pathlib.Path)
        mock_props = Mock(spec=pathlib.Path)
        mock_props.exists.return_value = True

        mock_stat = Mock()
        mock_stat.st_size = 20 * 1024  # 20KB
        mock_props.stat.return_value = mock_stat

        mock_node.__truediv__ = Mock(return_value=mock_props)
        mock_kfd_path.iterdir.return_value = [mock_node]

        with patch('pathlib.Path', return_value=mock_kfd_path):
            count = count_amd_gpus_impl()
            self.assertEqual(count, 0)

    def test_nonexistent_properties_file(self):
        """Non-existent properties files should be skipped."""
        mock_kfd_path = Mock(spec=pathlib.Path)
        mock_kfd_path.exists.return_value = True

        mock_node = Mock(spec=pathlib.Path)
        mock_props = Mock(spec=pathlib.Path)
        mock_props.exists.return_value = False

        mock_node.__truediv__ = Mock(return_value=mock_props)
        mock_kfd_path.iterdir.return_value = [mock_node]

        with patch('pathlib.Path', return_value=mock_kfd_path):
            count = count_amd_gpus_impl()
            self.assertEqual(count, 0)


class TestGetShmSize(unittest.TestCase):
    """Test shared memory size checking."""

    def test_dev_shm_exists(self):
        """Should return shm size in MB when /dev/shm exists."""
        with patch('os.path.exists', return_value=True):
            with patch('os.statvfs') as mock_statvfs:
                mock_stat = Mock()
                mock_stat.f_blocks = 128 * 1024
                mock_stat.f_frsize = 1024
                mock_statvfs.return_value = mock_stat

                size_mb = get_shm_size_impl()
                self.assertEqual(size_mb, 128.0)

    def test_dev_shm_not_exists(self):
        """Should return None when /dev/shm doesn't exist."""
        with patch('os.path.exists', return_value=False):
            size = get_shm_size_impl()
            self.assertIsNone(size)

    def test_statvfs_exception(self):
        """Should return 0 on statvfs exception."""
        with patch('os.path.exists', return_value=True):
            with patch('os.statvfs', side_effect=OSError("Error")):
                size = get_shm_size_impl()
                self.assertEqual(size, 0)


if __name__ == '__main__':
    unittest.main()
