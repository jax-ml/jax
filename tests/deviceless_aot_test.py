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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax.experimental import topologies, serialize_executable
from jaxlib import _jax as _xla

jax.config.parse_flags_with_absl()


class DevicelessAOTTest(jtu.JaxTestCase):
  def compile_without_devices(target_config):
    # No backend should be initialized
    # Topology will initialize a compile only backend
    assert not xb.backends_are_initialized()
    with jtu.global_config_context(jax_platforms="cpu"):
      topology = topologies.get_topology_desc(
        platform="cuda",
        target_config=target_config,
        topology="1x1x1",
      )
      mesh = topologies.make_mesh(topo=topology, mesh_shape=(1,), axis_names=("x",))

      x = jax.ShapeDtypeStruct(
        shape=(2, 2), 
        dtype=jnp.float32, 
        sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
      )

      def f(x):
        return jnp.sum(x * x)

      compiled = jax.jit(f).lower(x).compile()
      serialized_executable, _, _ = serialize_executable.serialize(compiled)
      return serialized_executable

  def test_deviceless_aot(self):
    mp.set_start_method('spawn')
    target_config = _xla.get_topology_for_devices(jax.devices()).target_config
    with ProcessPoolExecutor(max_workers=1) as executor:
      future = executor.submit(DevicelessAOTTest.compile_without_devices, target_config)
      serialized_executable = future.result()
    
    _, in_tree = jax.tree.flatten(((0,), {}))
    _, out_tree = jax.tree.flatten(0)
    compiled = serialize_executable.deserialize_and_load(
      serialized_executable, 
      in_tree, 
      out_tree,
      backend="cuda",
      execution_devices=jax.devices()[:1],
    )
    input = jnp.array([[0., 1.], [2., 3.]], dtype=jnp.float32, device=jax.devices()[0])
    result = compiled(input)
    self.assertEqual(result, 14.)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
