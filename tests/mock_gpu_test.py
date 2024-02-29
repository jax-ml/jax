# Copyright 2023 The JAX Authors.
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

from functools import partial
import math

from absl.testing import absltest
import jax
from jax import config
from jax._src import test_util as jtu
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

config.parse_flags_with_absl()


class MockGPUTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    jax.config.update('use_mock_gpu_client', True)

  def tearDown(self):
    jax.config.update('use_mock_gpu_client', False)
    jax.config.update('mock_num_gpus', 1)
    super().tearDown()

  def testMockWithSharding(self):
    num_shards = 16
    jax.config.update('mock_num_gpus', num_shards)
    mesh = jtu.create_global_mesh((num_shards,), ('x',))
    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, P('x',)),
        out_shardings=NamedSharding(mesh, P('x',)),
    )
    def f(x, y):
      z = x @ y
      return z @ y

    shape = (64, 64)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    y = x + 1
    f_lowered = f.lower(x, y)
    hlo = f_lowered.compiler_ir()
    self.assertIn('sharding = "{devices=[16,1]<=[16]}"', str(hlo))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
