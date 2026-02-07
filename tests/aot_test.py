# Copyright 2021 The JAX Authors.
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

import contextlib
import unittest
from absl.testing import absltest
import jax
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import test_util as jtu
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


class JaxAotTest(jtu.JaxTestCase):

  @jtu.run_on_devices('tpu', 'gpu')
  def test_pickle_jit_lower(self):
    def fun(x):
      return x * x

    with jax.set_mesh(jax.sharding.Mesh(np.array(jax.devices()), ('data',))):
      lowered = jax.jit(
          fun, in_shardings=P('data'), out_shardings=P(None, 'data')
      ).lower(core.ShapedArray(shape=(8, 8), dtype=np.float32))

    def verify_serialization(lowered):
      serialized, in_tree, out_tree = serialize(lowered.compile())
      compiled = deserialize_and_load(serialized, in_tree, out_tree)
      self.assertEqual(compiled.as_text(), lowered.compile().as_text())

    verify_serialization(lowered)
    verify_serialization(jax.jit(lambda x: x * x).lower(np.arange(100)))
    verify_serialization(
        jax.pmap(lambda x: x * x).lower(
            np.zeros((len(jax.devices()), 4), dtype=np.float32)))

  @jtu.skip_on_devices("tpu")  # TODO(phawkins): This test is segfaulting on TPU
  def test_topology_jit_serialize(self):
    try:
      aot_topo = topologies.get_topology_desc(
          platform=jax.devices()[0].platform
      )
    except (ValueError, NotImplementedError) as e:
      assert ('topology_name is not specified' in str(e) or
              'topology not implemented' in str(e))
      raise unittest.SkipTest('PJRT Topology not supported')

    if jtu.TEST_WITH_PERSISTENT_COMPILATION_CACHE.value:
      raise unittest.SkipTest('Compilation caching not yet supported.')
    if jtu.is_device_cuda():
      raise unittest.SkipTest('Broken on GPU: b/442353988')

    @jax.jit
    def fn(x):
      return x * x

    def lower_and_load(mesh):
      s = jax.sharding.NamedSharding(mesh, P('x', 'y'))
      x_shape = jax.ShapeDtypeStruct(
          shape=(16, 16),
          dtype=jnp.dtype('float32'),
          sharding=s)
      lowered = fn.lower(x_shape)
      serialized, in_tree, out_tree = serialize(lowered.compile())
      compiled = deserialize_and_load(serialized, in_tree, out_tree)
      return compiled

    ref_topo = topologies.get_attached_topology()
    n = max(1, len(ref_topo.devices) // 2)
    mesh_shape = (len(ref_topo.devices) // n, n)

    ref_mesh = topologies.make_mesh(ref_topo, mesh_shape, ('x', 'y'))
    aot_mesh = topologies.make_mesh(aot_topo, mesh_shape, ('x', 'y'))
    self.assertEqual(
        lower_and_load(ref_mesh).as_text(), lower_and_load(aot_mesh).as_text()
    )

  def test_get_topology_from_devices(self):
    try:
      aot_topo = topologies.get_topology_desc(
          platform=jax.devices()[0].platform
      )
    except (ValueError, NotImplementedError) as e:
      assert ('topology_name is not specified' in str(e) or
              'topology not implemented' in str(e))
      raise unittest.SkipTest('PJRT Topology not supported')

    topo = xc.get_topology_for_devices(aot_topo.devices)
    self.assertEqual(
        topo.platform_version, aot_topo.devices[0].client.platform_version
    )

  def test_lower_as_text_with_and_without_debug_info(self):
    def my_function(x):
      return jnp.sin(x)

    lowered = jax.jit(my_function).lower(42.)
    stablehlo = lowered.as_text("stablehlo", debug_info=True)
    self.assertRegex(stablehlo, r"sine.* loc")
    stablehlo = lowered.as_text("stablehlo")
    self.assertNotRegex(stablehlo, r"sine.* loc")

    hlo = lowered.as_text("hlo", debug_info=True)
    self.assertRegex(hlo, r'sine.*metadata=.*[stack_frame_id|source_file]=.*')
    hlo = lowered.as_text("hlo")
    self.assertNotRegex(
        hlo, r'sine.*metadata=.*[stack_frame_id|source_file]=.*'
    )

  def test_constants_in_lowering_in_aot(self):
    const_size = 100
    const = jax.random.uniform(jax.random.key(0), (const_size,),
                               dtype=np.float32)

    def my_function(x):
      return jnp.sin(x) + const

    lowered = jax.jit(my_function).lower(np.full_like(const, 42., dtype=const.dtype))
    stablehlo = lowered.as_text("stablehlo")
    if config.use_simplified_jaxpr_constants.value:
      self.assertNotRegex(stablehlo, rf"stablehlo.constant dense.*tensor<{const_size}x")
      self.assertLen(lowered._lowering.const_args, 1)
      self.assertIs(lowered._lowering.const_args[0], const)
    else:
      self.assertRegex(stablehlo, rf"stablehlo.constant dense.*tensor<{const_size}x")
      self.assertLen(lowered._lowering.const_args, 0)

  def test_with_constants(self):
    const = jnp.arange(16.) + 42.  # A distinctive shape and value

    @jax.jit
    def f(x):
      return const[0:8] + x

    inp = jnp.arange(8.)
    compiled = f.lower(inp).compile()
    self.assertLen(compiled.args_info[0], 1)  # Not including const_args
    self.assertLen(compiled.in_avals[0], 1)
    if config.use_simplified_jaxpr_constants.value:
      self.assertLen(compiled._params.const_args, 1)
      self.assertIs(compiled._params.const_args[0], const)
    else:
      self.assertLen(compiled._params.const_args, 0)
    self.assertArraysEqual(compiled(inp), const[0:8] + inp)
    self.assertCacheMisses(lambda: compiled(inp), cpp=0, aot_call=0)

  @jtu.parameterized_filterable(
      kwargs=[
          dict(use_np=use_np, lower=lower, compile=compile, exec=exec)
            for use_np in (False, True)
            for lower in (False, True)
            for compile in (False, True)
            for exec in (False, True)
  ])
  def test_with_constants_enable_x64(self, *, use_np, lower, compile, exec):
    # Closed-over constant is 64-bit. Each of lowering, compilation, and
    # execution can be run in 64-bit or 32-bit mode.
    with config.enable_x64(True):
      arange = np.arange if use_np else jnp.arange
      const = arange(8, dtype=np.int64) + 42

      @jax.jit
      def f(x):
        return lax.convert_element_type(const, np.float32) + x

    inp = np.arange(8., dtype=np.float32)
    with config.enable_x64(True) if lower else contextlib.nullcontext():
      lowered = f.lower(inp)
    with config.enable_x64(True) if compile else contextlib.nullcontext():
      compiled = lowered.compile()

    def run():
      with config.enable_x64(True) if exec else contextlib.nullcontext():
        return compiled(inp)

    self.assertLen(compiled.args_info[0], 1)  # Not including const_args
    self.assertLen(compiled.in_avals[0], 1)
    if config.use_simplified_jaxpr_constants.value:
      self.assertLen(compiled._params.const_args, 1)
      self.assertLen(compiled._executable.in_avals, 2)
      expected_dtype = np.int64
      if not config.enable_x64.value and use_np and not lower:
        expected_dtype = np.int32
      self.assertEqual(compiled._executable.in_avals[0].dtype, expected_dtype)

      if expected_dtype is np.int64:  # Otherwise, we made a copy of the const
        if use_np:
          self.assertIs(np.asarray(compiled._params.const_args[0]), const)
        else:
          self.assertIs(compiled._params.const_args[0], const)
    else:
      self.assertLen(compiled._params.const_args, 0)
      self.assertLen(compiled._executable.in_avals, 1)

    # In some cases we expect errors: in 32-bit mode, lowered with 64-bit mode
    # and execute in 32-bit mode.
    if (config.use_simplified_jaxpr_constants.value and
        not config.enable_x64.value and
        use_np and lower and not exec):
      with self.assertRaisesRegex(
          xc.XlaRuntimeError,
          "got buffer with incompatible size"):
        run()
      return

    self.assertArraysEqual(run(),
                           lax.convert_element_type(const, inp.dtype) + inp)
    # Trigger cache hit
    self.assertCacheMisses(run, cpp=0, aot_call=0)

  def test_with_ref_constants(self):
    x_ref = core.new_ref(0)

    @jax.jit
    def f(x):
      x_ref[...] += x

    f_lowered = f.lower(1)
    with self.assertRaisesRegex(ValueError, 'serialize with a closed-over'):
      serialized, in_tree, out_tree = serialize(f_lowered.compile())

  @jtu.run_on_devices('gpu', 'tpu')
  def test_mismatched_backends_raises(self):
    @jax.jit
    def f(x):
      return x * 2

    x = jnp.arange(1)
    f_lowered = f.lower(x)
    serialized, in_tree, out_tree = serialize(f_lowered.compile())
    with self.assertRaisesRegex(
        ValueError,
        'Execution devices belong to a client other than `backend`'):
      deserialize_and_load(serialized, in_tree, out_tree, backend='cpu',
                           execution_devices=jax.devices()[:1])

  @jtu.run_on_devices('gpu')
  def test_deviceless_aot_compile(self):
    target_config = xc.get_topology_for_devices(jax.devices()).target_config
    gpu_platform = jax.devices()[0].platform  # Capture before switching to cpu
    with jtu.global_config_context(jax_platforms="cpu"):
      topology = topologies.get_topology_desc(
        platform=gpu_platform,
        target_config=target_config,
        topology="1x1x1",
      )
      assert topology.devices[0].client.runtime_type == "compile_only_runtime"
      mesh = topologies.make_mesh(topo=topology, mesh_shape=(1,), axis_names=("x",))
      x = jax.ShapeDtypeStruct(
        shape=(2, 2),
        dtype=jnp.float32,
        sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
      )
      compiled = jax.jit(lambda x: jnp.sum(x * x)).lower(x).compile()
      serialized_executable, _, _ = serialize(compiled)

    _, in_tree = jax.tree.flatten(((0,), {}))
    _, out_tree = jax.tree.flatten(0)
    compiled = deserialize_and_load(
        serialized_executable,
        in_tree,
        out_tree,
        backend=gpu_platform,
        execution_devices=jax.devices()[:1]
    )
    input = jnp.array([[0., 1.], [2., 3.]], dtype=jnp.float32, device=jax.devices()[0])
    result = compiled(input)
    self.assertEqual(result, 14.)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
