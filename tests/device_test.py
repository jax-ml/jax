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

import os
import time


def _device_info(device):
  return {
      'repr': repr(device),
      'id': getattr(device, 'id', None),
      'process_index': getattr(device, 'process_index', None),
      'coords': getattr(device, 'coords', None),
      'core_on_chip': getattr(device, 'core_on_chip', None),
  }


def _run_tpu_core_split_diagnostic():
  import jax
  import jax.numpy as jnp
  import numpy as np

  shard_status_file = os.environ.get('TEST_SHARD_STATUS_FILE')
  if shard_status_file:
    with open(shard_status_file, 'a', encoding='utf-8'):
      pass

  print('JAX TPU core split Bazel diagnostic', flush=True)
  env_keys = (
      'TEST_RUN_NUMBER',
      'TEST_SHARD_INDEX',
      'TEST_TOTAL_SHARDS',
      'TEST_SHARD_STATUS_FILE',
      'TPU_VISIBLE_DEVICES',
      'TPU_VISIBLE_CHIPS',
      'TPU_CHIPS_PER_PROCESS_BOUNDS',
      'TPU_PROCESS_BOUNDS',
      'JAX_PLATFORMS',
      'TPU_TOPOLOGY',
      'TPU_TOPOLOGY_ALT',
      'TPU_TOPOLOGY_WRAP',
      'TPU_WORKER_ID',
      'TPU_CHIPS_PER_HOST_BOUNDS',
      'TPU_HOST_BOUNDS',
      'CHIPS_PER_HOST_BOUNDS',
      'HOST_BOUNDS',
      'TPU_WORKER_HOSTNAMES',
      'TPU_ACCELERATOR_TYPE',
      'TPU_RUNTIME_METRICS_PORTS',
      'VBAR_CONTROL_SERVICE_URL',
  )
  for key in env_keys:
    print(f'{key}: {os.environ.get(key)}', flush=True)
  print('default backend:', jax.default_backend(), flush=True)
  print('process count:', jax.process_count(), flush=True)
  print('process index:', jax.process_index(), flush=True)
  print('device count:', jax.device_count(), flush=True)
  local_devices = jax.local_devices()
  print('local devices:', [_device_info(d) for d in local_devices], flush=True)
  if len(local_devices) != 1:
    raise SystemExit(
        f'Expected exactly one local TPU device; got {len(local_devices)}'
    )

  @jax.jit
  def _compute(x):
    return jnp.sum((x + 1.0) * (x + 2.0))

  x = jax.device_put(np.arange(16, dtype=np.float32), local_devices[0])
  result = _compute(x).block_until_ready()
  actual = float(jax.device_get(result))
  expected = float(sum((i + 1) * (i + 2) for i in range(16)))
  print('compute result:', actual, flush=True)
  if actual != expected:
    raise SystemExit(f'Expected compute result {expected}; got {actual}')
  time.sleep(float(os.environ.get('JAX_TPU_CORE_SPLIT_DIAGNOSTIC_SLEEP', '5')))
  print('JAX TPU core split Bazel diagnostic finished', flush=True)


if os.environ.get('JAX_TPU_CORE_SPLIT_DIAGNOSTIC') == '1':
  _run_tpu_core_split_diagnostic()
  raise SystemExit(0)


from absl.testing import absltest
import jax
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class DeviceTest(jtu.JaxTestCase):

  def test_repr(self):
    device = jax.devices()[0]

    if jtu.is_device_cuda():
      self.assertEqual(device.platform, 'gpu')
      self.assertEqual(repr(device), 'CudaDevice(id=0)')
    elif jtu.is_device_rocm():
      self.assertEqual(device.platform, 'gpu')
      self.assertEqual(repr(device), 'RocmDevice(id=0)')
    elif jtu.test_device_matches(['tpu']):
      self.assertEqual(device.platform, 'tpu')
      self.assertEqual(
          repr(device),
          'TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)',
      )
    elif jtu.test_device_matches(['oneapi']):
      self.assertEqual(device.platform, 'oneapi')
      self.assertEqual(repr(device), 'OneapiDevice(id=0)')
    elif jtu.test_device_matches(['cpu']):
      self.assertEqual(device.platform, 'cpu')
      self.assertEqual(repr(device), 'CpuDevice(id=0)')

  def test_str(self):
    device = jax.devices()[0]

    if jtu.is_device_cuda():
      self.assertEqual(str(device), 'cuda:0')
    elif jtu.is_device_rocm():
      self.assertEqual(str(device), 'rocm:0')
    elif jtu.test_device_matches(['tpu']):
      self.assertEqual(str(device), 'TPU_0(process=0,(0,0,0,0))')
    elif jtu.test_device_matches(['oneapi']):
      self.assertEqual(str(device), 'oneapi:0')
    elif jtu.test_device_matches(['cpu']):
      # TODO(phawkins): remove TFRT_CPU_0 once jaxlib 0.10 is the minimum
      # version.
      self.assertIn(str(device), ['cpu:0', 'TFRT_CPU_0'])


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
