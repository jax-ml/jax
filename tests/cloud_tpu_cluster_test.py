# Copyright 2026 The JAX Authors.
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

from __future__ import annotations

import os
import socket
import tempfile
from unittest import mock

from absl.testing import absltest
from jax._src import test_util as jtu
from jax._src.clusters.cloud_tpu_cluster import GkeTpuCluster


@jtu.thread_unsafe_test_class()  # Modifies ``os.environ``.
class CloudTpuClusterTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.orig_env = os.environ.copy()

  def tearDown(self):
    os.environ.clear()
    os.environ.update(self.orig_env)
    super().tearDown()

  def test_read_from_valid_file(self):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write('  10.0.0.1:8476,10.0.0.2:8476 \n')
      temp_path = f.name

    try:
      os.environ['TPU_PROCESS_ADDRESSES_PATH'] = temp_path
      os.environ.pop('TPU_PROCESS_ADDRESSES', None)
      os.environ.pop('TPU_WORKER_HOSTNAMES', None)

      self.assertEqual(
          GkeTpuCluster._get_worker_host_names_env_var(),
          '10.0.0.1:8476,10.0.0.2:8476',
      )
      self.assertEqual(
          GkeTpuCluster._get_worker_list_in_slice(),
          ['10.0.0.1:8476', '10.0.0.2:8476'],
      )
    finally:
      os.remove(temp_path)

  def test_precedence_over_process_addresses_env(self):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write('10.0.0.1:8476')
      temp_path = f.name

    try:
      os.environ['TPU_PROCESS_ADDRESSES_PATH'] = temp_path
      os.environ['TPU_PROCESS_ADDRESSES'] = '10.0.0.2:8476'
      os.environ.pop('TPU_WORKER_HOSTNAMES', None)

      self.assertEqual(
          GkeTpuCluster._get_worker_host_names_env_var(), '10.0.0.1:8476'
      )
    finally:
      os.remove(temp_path)

  def test_precedence_over_worker_hostnames_env(self):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write('10.0.0.1:8476')
      temp_path = f.name

    try:
      os.environ['TPU_PROCESS_ADDRESSES_PATH'] = temp_path
      os.environ.pop('TPU_PROCESS_ADDRESSES', None)
      os.environ['TPU_WORKER_HOSTNAMES'] = 'worker-0,worker-1'

      self.assertEqual(
          GkeTpuCluster._get_worker_host_names_env_var(), '10.0.0.1:8476'
      )
    finally:
      os.remove(temp_path)

  def test_file_does_not_exist(self):
    nonexistent_path = '/tmp/nonexistent_tpu_addresses_path_test_12345.txt'
    if os.path.exists(nonexistent_path):
      os.remove(nonexistent_path)

    os.environ['TPU_PROCESS_ADDRESSES_PATH'] = nonexistent_path
    os.environ.pop('TPU_PROCESS_ADDRESSES', None)
    os.environ.pop('TPU_WORKER_HOSTNAMES', None)

    with self.assertRaises(RuntimeError) as ctx:
      GkeTpuCluster._get_worker_host_names_env_var()
    self.assertIn(
        'Failed to read TPU process addresses file', str(ctx.exception)
    )

  def test_empty_file_raises_value_error(self):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write('   \n  \t ')
      temp_path = f.name

    try:
      os.environ['TPU_PROCESS_ADDRESSES_PATH'] = temp_path
      os.environ.pop('TPU_PROCESS_ADDRESSES', None)
      os.environ.pop('TPU_WORKER_HOSTNAMES', None)

      with self.assertRaises(ValueError) as ctx:
        GkeTpuCluster._get_worker_host_names_env_var()
      self.assertIn('TPU process addresses file is empty', str(ctx.exception))
    finally:
      os.remove(temp_path)

  def test_no_fallback_on_failure(self):
    nonexistent_path = '/tmp/nonexistent_tpu_addresses_path_test_12345.txt'
    if os.path.exists(nonexistent_path):
      os.remove(nonexistent_path)

    os.environ['TPU_PROCESS_ADDRESSES_PATH'] = nonexistent_path
    os.environ['TPU_PROCESS_ADDRESSES'] = '10.0.0.2:8476'
    os.environ.pop('TPU_WORKER_HOSTNAMES', None)

    with self.assertRaises(RuntimeError):
      GkeTpuCluster._get_worker_host_names_env_var()

  def test_fallback_to_process_addresses(self):
    os.environ.pop('TPU_PROCESS_ADDRESSES_PATH', None)
    os.environ['TPU_PROCESS_ADDRESSES'] = '10.0.0.1:8476,10.0.0.2:8476'
    os.environ.pop('TPU_WORKER_HOSTNAMES', None)

    self.assertEqual(
        GkeTpuCluster._get_worker_host_names_env_var(),
        '10.0.0.1:8476,10.0.0.2:8476',
    )

  def test_fallback_to_worker_hostnames(self):
    os.environ.pop('TPU_PROCESS_ADDRESSES_PATH', None)
    os.environ.pop('TPU_PROCESS_ADDRESSES', None)
    os.environ['TPU_WORKER_HOSTNAMES'] = 'worker-0,worker-1'

    self.assertEqual(
        GkeTpuCluster._get_worker_host_names_env_var(), 'worker-0,worker-1'
    )

  def test_empty_path_env_var_ignored(self):
    os.environ['TPU_PROCESS_ADDRESSES_PATH'] = ''
    os.environ['TPU_PROCESS_ADDRESSES'] = '10.0.0.1:8476'
    os.environ.pop('TPU_WORKER_HOSTNAMES', None)

    self.assertEqual(
        GkeTpuCluster._get_worker_host_names_env_var(), '10.0.0.1:8476'
    )

  def test_none_when_no_env_set(self):
    os.environ.pop('TPU_PROCESS_ADDRESSES_PATH', None)
    os.environ.pop('TPU_PROCESS_ADDRESSES', None)
    os.environ.pop('TPU_WORKER_HOSTNAMES', None)

    self.assertIsNone(GkeTpuCluster._get_worker_host_names_env_var())

  def test_wait_for_coordinator_uses_getaddrinfo_ipv6(self):
    with mock.patch('socket.getaddrinfo') as mock_getaddrinfo:
      GkeTpuCluster.wait_for_coordinator('fd00::1', timeout_secs=5)
      mock_getaddrinfo.assert_called_once_with('fd00::1', None)

  def test_wait_for_coordinator_timeout(self):
    with (
        mock.patch(
            'socket.getaddrinfo',
            side_effect=socket.gaierror(
                -5, 'No address associated with hostname'
            ),
        ),
        mock.patch('time.sleep'),
    ):
      with self.assertRaises(RuntimeError) as ctx:
        GkeTpuCluster.wait_for_coordinator(
            'tpu-job-0.tpu-svc', timeout_secs=0.01
        )
      self.assertIn(
          'Failed to recognize coordinator address tpu-job-0.tpu-svc',
          str(ctx.exception),
      )




if __name__ == '__main__':
  absltest.main()
