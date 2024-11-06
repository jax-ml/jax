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

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

jax.config.parse_flags_with_absl()

NUM_SLICES = 2
NUM_HOSTS_PER_SLICE = 4


@jtu.with_config(
  jax_mock_gpu_topology=f"{NUM_SLICES}x{NUM_HOSTS_PER_SLICE}x1",
  jax_cuda_visible_devices="0")
class MockGPUTopologyTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest("Mocking devices only works on the GPU backend.")
    super().setUp()

  @jtu.skip_under_pytest("Test must run in an isolated process")
  def testMockDeviceCount(self):
    self.assertEqual(jax.device_count(), NUM_SLICES * NUM_HOSTS_PER_SLICE)

  @jtu.skip_under_pytest("Test must run in an isolated process")
  def testMockWithSharding(self):
    mesh = jax.sharding.Mesh(jax.devices(), ('x',))
    f = jax.jit(jnp.sum,
                in_shardings=NamedSharding(mesh, P('x')),
                out_shardings=NamedSharding(mesh, P()))

    f_lowered = f.lower(jnp.arange(16))
    hlo = f_lowered.compiler_ir()

    mocked_count = NUM_SLICES * NUM_HOSTS_PER_SLICE
    self.assertIn(f'num_partitions = {mocked_count}', str(hlo))
    self.assertIn(
        f'sharding = "{{devices=[{mocked_count}]<=[{mocked_count}]}}"',
        str(hlo)
    )

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
