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
import re
import tempfile
import unittest
from absl.testing import absltest
import jax
from jax import lax
from jax._src import compilation_cache
from jax._src import config
from jax._src import core
from jax._src import monitoring
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


def _get_compile_only_topology_for_attached_devices():
  devices = jax.devices()
  platform = devices[0].platform
  if platform == 'tpu' and xb.make_pjrt_tpu_topology.__module__.endswith(
      'xla_bridge'
  ):
    device_kind = devices[0].device_kind
    topo_prefix = topologies._DEVICE_KIND_TO_TOPOLOGY[device_kind][0].split(
        '='
    )[0]
    chips_per_host_bounds = tuple(
        max(d.coords[i] for d in devices) + 1 for i in range(3)
    )
    num_chips = (
        chips_per_host_bounds[0]
        * chips_per_host_bounds[1]
        * chips_per_host_bounds[2]
    )
    topology_name = (
        f'{topo_prefix}:'
        f'{chips_per_host_bounds[0]}x{chips_per_host_bounds[1]}x{chips_per_host_bounds[2]}'
    )
    kwargs = {'chips_per_host_bounds': chips_per_host_bounds}
    # Match the chip_config_name that libtpu sets on live Cloud TPU hosts.
    if device_kind == 'TPU v4' and len(devices) == num_chips:
      kwargs['chip_config_name'] = 'megacore'
    elif device_kind in ('TPU v5', 'TPU v5p', 'TPU v6 lite', 'TPU v6e'):
      kwargs['chip_config_name'] = 'megachip_tccontrol'
    return topologies.get_topology_desc(
        topology_name=topology_name,
        platform='tpu',
        **kwargs,
    )
  return topologies.get_topology_desc(platform=platform)


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

    # Unlike CPU and GPU, TPU lowerings retain debug info in text format. We'll
    # keep function-level info since that gives us confidence that things match,
    # but line/column level will never match so let's rip them out.
    def normalize_mlir_locations(ir_text: str) -> str:
      lines_removed = re.sub(r'line=(\d+)', 'line=stripped', ir_text)
      columns_removed = re.sub(
          r'column=(\d+)', 'column=stripped', lines_removed)
      return columns_removed

    self.assertEqual(
        normalize_mlir_locations(lower_and_load(ref_mesh).as_text()),
        normalize_mlir_locations(lower_and_load(aot_mesh).as_text())
    )

  def test_get_topology_from_devices(self):
    try:
      aot_topo = _get_compile_only_topology_for_attached_devices()
    except (ValueError, NotImplementedError) as e:
      assert ('topology_name is not specified' in str(e) or
              'topology not implemented' in str(e))
      raise unittest.SkipTest('PJRT Topology not supported')

    topo = xc.get_topology_for_devices(aot_topo.devices)
    ref_topo = xc.get_topology_for_devices(jax.devices())
    self.assertEqual(
        topo.platform_version, aot_topo.devices[0].client.platform_version
    )
    self.assertEqual(topo.platform_version, ref_topo.platform_version)
    self.assertEqual(
        aot_topo.devices[0].client.platform_version,
        jax.devices()[0].client.platform_version,
    )
    self.assertFalse(topo.platform_version.startswith('PJRT C API'))
    self.assertEqual(topo.fingerprint(), ref_topo.fingerprint())

  @jtu.run_on_devices('tpu')
  @jtu.thread_unsafe_test()
  def test_topology_persistent_compilation_cache(self):
    compilation_cache.reset_cache()
    self.addCleanup(compilation_cache.reset_cache)
    cache_dir = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(config.enable_compilation_cache(True))
    self.enterContext(config.raise_persistent_cache_errors(True))
    self.enterContext(config.persistent_cache_min_compile_time_secs(0))
    self.enterContext(config.persistent_cache_min_entry_size_bytes(0))
    self.enterContext(config.compilation_cache_check_contents(False))
    self.enterContext(config.compilation_cache_dir(cache_dir))

    events = []
    monitoring.register_event_listener(events.append)
    self.addCleanup(monitoring.unregister_event_listener, events.append)

    @jax.jit
    def fn(x):
      return x * x + 1.0

    tpu_topo = topologies.get_attached_topology()
    n = max(1, len(tpu_topo.devices) // 2)
    mesh_shape = (len(tpu_topo.devices) // n, n)

    # AOT compile on the CPU device targeting a compile-only TPU topology.
    aot_topo = _get_compile_only_topology_for_attached_devices()
    aot_mesh = topologies.make_mesh(aot_topo, mesh_shape, ('x', 'y'))
    aot_sharding = jax.sharding.NamedSharding(aot_mesh, P('x', 'y'))
    x_shape = jax.ShapeDtypeStruct(
        shape=(16, 16), dtype=jnp.float32, sharding=aot_sharding
    )
    with jax.default_device(jax.devices('cpu')[0]):
      compiled = fn.lower(x_shape).compile()

    # Verify the executable targets the compile-only TPU devices and was cached.
    self.assertEqual(aot_topo.devices[0].platform, 'tpu')
    self.assertEqual(
        aot_topo.devices[0].client.runtime_type, 'compile_only_runtime'
    )
    self.assertEqual(
        compiled.output_shardings.device_set, set(aot_topo.devices)
    )
    self.assertEqual(events.count('/jax/compilation_cache/cache_misses'), 1)
    self.assertEqual(events.count('/jax/compilation_cache/cache_hits'), 0)

    # Clear the in-memory JIT cache and run on the real attached TPU devices.
    fn.clear_cache()
    tpu_mesh = topologies.make_mesh(tpu_topo, mesh_shape, ('x', 'y'))
    tpu_sharding = jax.sharding.NamedSharding(tpu_mesh, P('x', 'y'))
    x_np = np.arange(256, dtype=np.float32).reshape(16, 16)
    x = jax.device_put(x_np, tpu_sharding)
    result = fn(x)

    # Verify the TPU run hit the persistent cache and executed on the TPUs.
    self.assertEqual(events.count('/jax/compilation_cache/cache_hits'), 1)
    self.assertEqual(events.count('/jax/compilation_cache/cache_misses'), 1)
    self.assertEqual(result.sharding.device_set, set(jax.devices('tpu')))
    self.assertArraysEqual(result, x_np * x_np + 1.0)

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
    self.assertEqual(compiled.args_info[0][0]._aval.shape, inp.shape)
    self.assertLen(compiled.in_avals[0], 1)
    self.assertEqual(compiled.in_avals[0][0].shape, inp.shape)
    self.assertLen(compiled.input_shardings[0], 1)
    self.assertLen(compiled.input_formats[0], 1)
    if config.use_simplified_jaxpr_constants.value:
      self.assertLen(compiled._params.const_args, 1)
      self.assertIs(compiled._params.const_args[0], const)
    else:
      self.assertLen(compiled._params.const_args, 0)
    self.assertArraysEqual(compiled(inp), const[0:8] + inp)
    self.assertCacheMisses(lambda: compiled(inp), cpp=0, aot_call=0)

  @jtu.skip_on_flag("jax_use_simplified_jaxpr_constants", False)
  def test_with_small_constants(self):
    const1 = np.ones((4,), dtype=np.int32)  # Will be embedded
    const2 = jnp.ones((16,), dtype=np.int32)
    @jax.jit
    def f():
      return const1 + const2[:const1.shape[0]]
    with config.embedded_constants_max_bytes(const1.nbytes):
      compiled = f.lower().compile()
    self.assertLen(compiled._params.const_args, 1)
    self.assertIs(compiled._params.const_args[0], const2)

  def test_with_constants_and_dce(self):
    const = jnp.arange(16.) + 42.  # A distinctive shape and value

    @jax.jit
    def f(x, y_dead):  # y is DCEed
      z = const[0:8] + x
      return z

    inp = jnp.arange(8.)
    y_dead = jnp.arange(24.)
    compiled = f.lower(inp, y_dead).compile()
    # `compiled.` fields include dead args, but not the const_args
    self.assertLen(compiled.args_info[0], 2)
    self.assertEqual(compiled.args_info[0][0]._aval.shape, inp.shape)
    self.assertEqual(compiled.args_info[0][1]._aval.shape, y_dead.shape)
    self.assertLen(compiled.in_avals[0], 2)
    self.assertEqual(compiled.in_avals[0][0].shape, inp.shape)
    self.assertEqual(compiled.in_avals[0][1].shape, y_dead.shape)
    self.assertLen(compiled.input_shardings[0], 2)
    self.assertLen(compiled.input_formats[0], 2)
    if config.use_simplified_jaxpr_constants.value:
      self.assertLen(compiled._params.const_args, 1)
      self.assertIs(compiled._params.const_args[0], const)
    else:
      self.assertLen(compiled._params.const_args, 0)
    self.assertArraysEqual(compiled(inp, y_dead), const[0:8] + inp)
    self.assertCacheMisses(lambda: compiled(inp, y_dead), cpp=0, aot_call=0)

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
      const = arange(16, dtype=np.int64) + 42

      @jax.jit
      def f(x):
        return lax.convert_element_type(const, np.float32) + x

    inp = np.arange(16., dtype=np.float32)
    with config.enable_x64(True) if lower else contextlib.nullcontext():
      lowered = f.lower(inp)
    with config.enable_x64(True) if compile else contextlib.nullcontext():
      compiled = lowered.compile()

    def run():
      with config.enable_x64(True) if exec else contextlib.nullcontext():
        return compiled(inp)

    self.assertLen(compiled.args_info[0], 1)  # Not including const_args
    self.assertLen(compiled.in_avals[0], 1)
    self.assertLen(compiled.input_shardings[0], 1)
    self.assertLen(compiled.input_formats[0], 1)
    if config.use_simplified_jaxpr_constants.value:
      self.assertLen(compiled._params.const_args, 1)
      self.assertLen(compiled._executable.in_avals, 2)
      expected_dtype = np.int64
      if not config.enable_x64.value and use_np and not lower:
        expected_dtype = np.int32
      self.assertEqual(compiled._executable.in_avals[0].dtype, expected_dtype)

      if expected_dtype is np.int64:  # Otherwise, we made a copy of the const
        if not use_np:
          self.assertIs(compiled._params.const_args[0], const)
    else:
      self.assertLen(compiled._params.const_args, 0)
      self.assertLen(compiled._executable.in_avals, 1)

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

  @jtu.run_on_devices('gpu')
  def test_cross_compile_with_real_gpu(self):
    """Test that cross-compilation uses the real GPU for autotuning."""
    target_config = xc.get_topology_for_devices(jax.devices()).target_config
    gpu_platform = jax.devices()[0].platform

    # Create a compile-only topology, but DON'T switch to CPU so the real
    # GPU backend remains available for cross-compilation with autotuning.
    topology = topologies.get_topology_desc(
        platform=gpu_platform,
        target_config=target_config,
        topology="1x1x1",
    )
    assert topology.devices[0].client.runtime_type == "compile_only_runtime"
    mesh = topologies.make_mesh(
        topo=topology, mesh_shape=(1,), axis_names=("x",)
    )
    x = jax.ShapeDtypeStruct(
        shape=(2, 2),
        dtype=jnp.float32,
        sharding=jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x")
        ),
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
        execution_devices=jax.devices()[:1],
    )
    input = jnp.array(
        [[0., 1.], [2., 3.]], dtype=jnp.float32, device=jax.devices()[0]
    )
    result = compiled(input)
    self.assertEqual(result, 14.)

  @jtu.run_on_devices("tpu", "gpu")
  def test_topology_serialize_deserialize_aot_compile_reload(self):
    """Tests getting builtin topology, serializing, deserializing, compiling on it, and reloading executable on local devices."""
    if jtu.TEST_WITH_PERSISTENT_COMPILATION_CACHE.value:
      raise unittest.SkipTest("Compilation caching not yet supported.")

    orig_topo = topologies.TopologyDescription(jax.devices())
    serialized_topo = orig_topo.serialize()
    self.assertIsInstance(serialized_topo, bytes)
    self.assertNotEmpty(serialized_topo)

    restored_topo = topologies.TopologyDescription.deserialize(serialized_topo)
    self.assertLen(restored_topo.devices, len(orig_topo.devices))
    self.assertEqual(
        restored_topo.devices[0].client.runtime_type, "compile_only_runtime"
    )
    self.assertEqual(
        [d.id for d in restored_topo.devices],
        [d.id for d in orig_topo.devices],
    )

    mesh = jax.sharding.Mesh(np.array(restored_topo.devices[:1]), ("x",))
    x = jax.ShapeDtypeStruct(
        shape=(2, 2),
        dtype=jnp.float32,
        sharding=jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x")
        ),
    )
    compiled = jax.jit(lambda x: jnp.sum(x * x)).lower(x).compile()
    serialized_exec, in_tree, out_tree = serialize(compiled)

    reloaded = deserialize_and_load(
        serialized_exec,
        in_tree,
        out_tree,
        backend=jax.devices()[0].platform,
        execution_devices=jax.devices()[:1],
    )

    inp = jnp.array(
        [[0.0, 1.0], [2.0, 3.0]], dtype=jnp.float32, device=jax.devices()[0]
    )
    result = reloaded(inp)
    self.assertEqual(result, 14.0)

  def test_serialized_topology_invalid_bytes_raises_error(self):
    """Tests that passing invalid serialized protobuf bytes raises ValueError."""
    with self.assertRaisesRegex(
        ValueError,
        "Failed to parse PjRtTopologyDescriptionProto from serialized bytes",
    ):
      topologies.TopologyDescription.deserialize(b"invalid_proto_bytes")

  def test_get_executable_version(self):
    """Tests LoadedExecutable.get_executable_version() returns non-empty version bytes."""
    x = jnp.ones((2, 2), dtype=jnp.float32)
    compiled = jax.jit(lambda x: x + 1).lower(x).compile()
    py_exec = compiled.runtime_executable()
    version_bytes = py_exec.get_executable_version()
    self.assertIsInstance(version_bytes, bytes)
    self.assertNotEmpty(version_bytes)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
