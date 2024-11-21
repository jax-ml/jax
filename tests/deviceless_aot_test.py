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
"""Tests for deviceless AOT compilation on GPU."""

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax.experimental import topologies
from jax.experimental.serialize_executable import (
    deserialize_and_load,
    serialize,
)
from jax.lib.xla_client import get_topology_for_devices
import jax.numpy as jnp
from jax._src import xla_bridge
import multiprocessing

class DevicelessGpuAotTest(jtu.JaxTestCase):

  def compile_worker(target_config):
    # This function is supposed to run in a fresh process.
    # Check that backends are not initialized yet.
    assert not xla_bridge.backends_are_initialized()

    with jtu.global_config_context(jax_platforms="cpu"):
      topo = topologies.get_topology_desc(
        platform="cuda",
        target_config=target_config,
        topology="1x1x1")

      sharding = jax.sharding.SingleDeviceSharding(topo.devices[0])

      # Function to compile.
      @jax.jit
      def fn(x):
        return jnp.sum(x * x)

      # Provide input shape(s).
      x_shape = jax.ShapeDtypeStruct(
                shape=(2, 2),
                dtype=jnp.dtype('float32'),
                sharding=sharding)

      # Lower and compile.
      compiled = fn.lower(x_shape).compile()

      # Serialize the compilation results.
      serialized, in_tree, out_tree = serialize(compiled)

      return serialized

  def test_serialize_deserialize_execute(self):
    target_config = get_topology_for_devices(jax.devices()).target_config

    # Call the compilation in a different process so that we
    # can start JAX without the GPU platform there.
    multiprocessing.set_start_method("spawn")
    pool = multiprocessing.Pool(processes = 1)
    [serialized] = pool.map(DevicelessGpuAotTest.compile_worker,
                            [target_config])
    pool.close()

    # Provide the input pytree structure (0 stands for leaf).
    _, in_tree = jax.tree_util.tree_flatten(((0,), {}))
    # Provide the output pytree structure (here just one JAX array).
    _, out_tree = jax.tree_util.tree_flatten(0)
    # Deserialize the function.
    compiled = deserialize_and_load(serialized, in_tree, out_tree)

    # Call the deserialized function.
    result = compiled(jnp.array([[0., 1.], [2., 3.]]))

    self.assertEqual(result, 14)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
