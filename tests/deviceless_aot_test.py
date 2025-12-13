import contextlib
import unittest
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from absl.testing import absltest
import jax
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax.experimental import topologies
from jax.experimental.serialize_executable import (
    deserialize_and_load,
    serialize,
)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

jax.config.parse_flags_with_absl()

prev_xla_flags = None

with contextlib.suppress(ImportError):
  import pytest
  pytestmark = pytest.mark.multiaccelerator

def _compile_without_devices(target_config):
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
      serialized_executable, _, _ = serialize(compiled)
      return serialized_executable

class JaxDevicelessAotTest(jtu.JaxTestCase):
  @jtu.run_on_devices('gpu')
  def test_deviceless_aot_compile(self):
    mp.set_start_method('spawn')
    target_config = xc.get_topology_for_devices(jax.devices()).target_config
    with ProcessPoolExecutor(max_workers=1) as executor:
      future = executor.submit(_compile_without_devices, target_config)
      serialized_executable = future.result()

    _, in_tree = jax.tree.flatten(((0,), {}))
    _, out_tree = jax.tree.flatten(0)
    compiled = deserialize_and_load(
        serialized_executable, 
        in_tree, 
        out_tree, 
        backend="cuda", 
        execution_devices=jax.devices()[:1]
    )
    input = jnp.array([[0., 1.], [2., 3.]], dtype=jnp.float32, device=jax.devices()[0])
    result = compiled(input)
    self.assertEqual(result, 14.)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())