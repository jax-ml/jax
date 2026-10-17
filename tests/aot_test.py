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
import functools
import re
import unittest
from absl.testing import absltest
import jax
from jax._src import compilation_cache
from jax._src import config
from jax._src import core
from jax._src import monitoring
from jax._src import test_util as jtu
from jax._src.lib import xla_client as xc
from jax.experimental import topologies
from jax.experimental.serialize_executable import (
    deserialize_and_load,
    serialize,
)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.stages import Cache, Lowered, Traced, set_cache
from jax._src.stages import Stage
import numpy as np


class InMemoryCache(Cache):
  """Simple in-memory cache for testing JAX stages."""

  def __init__(self):
    self._store: dict[str, Stage] = {}
    self.hits: int = 0
    self.misses: int = 0

  def get(self, key: str) -> Stage | None:
    if key in self._store:
      self.hits += 1
      return self._store[key]
    self.misses += 1
    return None

  def put(self, key: str, value: Stage) -> None:
    self._store[key] = value

  def reset_stats(self) -> None:
    self.hits = 0
    self.misses = 0

  def __len__(self) -> int:
    return len(self._store)

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
      aot_topo = topologies.get_topology_desc(
          platform=jax.devices()[0].platform
      )
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
    if jax.devices()[0].platform != "gpu":
      self.assertEqual(
          aot_topo.devices[0].client.platform_version,
          jax.devices()[0].client.platform_version,
      )
    self.assertEqual(topo.fingerprint(), ref_topo.fingerprint())

  @jtu.run_on_devices('tpu')
  @jtu.thread_unsafe_test()
  def test_topology_persistent_compilation_cache(self):
    compilation_cache.reset_cache()
    self.addCleanup(compilation_cache.reset_cache)
    cache_dir = self.create_tempdir().full_path
    with (
        jtu.global_config_context(jax_logging_level='DEBUG'),
        config.enable_compilation_cache(True),
        config.raise_persistent_cache_errors(True),
        config.persistent_cache_min_compile_time_secs(0),
        config.persistent_cache_min_entry_size_bytes(0),
        config.compilation_cache_check_contents(False),
        config.compilation_cache_dir(cache_dir),
    ):
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
      aot_topo = topologies.get_topology_desc(
          platform=jax.devices()[0].platform
      )
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
      self.assertEqual(events.count('/jax/compilation_cache/cache_misses'), 1)
      self.assertEqual(events.count('/jax/compilation_cache/cache_hits'), 1)
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
        return const, x

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

    out_const, out_inp = run()
    self.assertArraysEqual(out_const, const)
    self.assertArraysEqual(out_inp, inp)
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


class LoweredStageCacheTest(jtu.JaxTestCase):
  """Tests for Lowered stage caching with @jax.jit(cache_key=...) and jax.stages.Cache."""

  def test_lowered_cache_cold_start_and_hit(self):
    trace_count = 0

    @jax.jit(cache_key="my_func")
    def f(x):
      nonlocal trace_count
      trace_count += 1
      return x * 2.0 + 1.0

    cache = InMemoryCache()
    x = jnp.arange(4, dtype=jnp.float32)

    with set_cache(cache):
      # 1. Cold start: executes and caches Lowered
      res1 = f(x)
      self.assertArraysEqual(res1, x * 2.0 + 1.0)
      self.assertEqual(trace_count, 1)
      self.assertEqual(cache.hits, 0)
      self.assertEqual(cache.misses, 1)

      # 2. In-memory cache hit on the instance
      res2 = f(x)
      self.assertArraysEqual(res2, x * 2.0 + 1.0)
      self.assertEqual(trace_count, 1)

      # 3. New jit instance with same key: hits Lowered directly, skips tracing entirely!
      trace_count_new = 0

      @jax.jit(cache_key="my_func")
      def f_new(x):
        nonlocal trace_count_new
        trace_count_new += 1
        return x * 2.0 + 1.0

      res3 = f_new(x)
      self.assertArraysEqual(res3, x * 2.0 + 1.0)
      self.assertEqual(trace_count_new, 0)  # Never traced!
      self.assertEqual(cache.hits, 1)

  def test_lowered_cache_lower_method_hit(self):
    trace_count = 0

    @jax.jit(cache_key="lower_method_func")
    def f(x):
      nonlocal trace_count
      trace_count += 1
      return x * 3.0

    cache = InMemoryCache()
    x = jnp.ones(4, dtype=jnp.float32)

    with set_cache(cache):
      # Run to populate Lowered cache
      f(x)
      self.assertEqual(trace_count, 1)

      # Calling .lower() on a new function instance hits Lowered cache directly
      @jax.jit(cache_key="lower_method_func")
      def f_new(x):
        raise RuntimeError("Should never trace on Lowered cache hit!")

      lowered = f_new.lower(x)
      self.assertIsInstance(lowered, Lowered)
      self.assertEqual(cache.hits, 1)
      compiled = lowered.compile()
      self.assertArraysEqual(compiled(x), x * 3.0)

  def test_lowered_cache_eval_shape_hit(self):
    @jax.jit(cache_key="eval_shape_func")
    def f(x):
      return x * 3.0

    cache = InMemoryCache()
    x = jnp.ones((2, 3), dtype=jnp.float32)

    with set_cache(cache):
      # Run to populate Lowered cache
      f(x)
      self.assertEqual(cache.hits, 0)
      self.assertEqual(cache.misses, 1)

      # Calling .eval_shape() on a new function instance hits Lowered cache directly
      @jax.jit(cache_key="eval_shape_func")
      def f_new(x):
        raise RuntimeError("Should never trace on Lowered cache hit!")

      out_shape = f_new.eval_shape(x)
      self.assertEqual(cache.hits, 1)
      self.assertEqual(out_shape.shape, (2, 3))
      self.assertEqual(out_shape.dtype, jnp.float32)

  def test_grad_of_function_with_cached_lowered_does_not_fail(self):
    @jax.jit(cache_key="grad_with_cached_lowered")
    def f(x):
      return jnp.sum(x**2)

    cache = InMemoryCache()
    x_shape = jax.ShapeDtypeStruct((3,), jnp.float32)
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

    with set_cache(cache):
      # Populate Lowered cache with ShapeDtypeStruct
      lowered = f.lower(x_shape)
      self.assertIsInstance(lowered, Lowered)
      self.assertEqual(cache.hits, 0)
      self.assertEqual(cache.misses, 1)

      # Under jax.grad, trace_state_clean is False, so it must trace cleanly and not hit Lowered cache
      @jax.jit(cache_key="grad_with_cached_lowered")
      def f2(x):
        return jnp.sum(x**2)

      grad_f2 = jax.grad(f2)
      res = grad_f2(x)
      self.assertArraysAllClose(res, 2.0 * x)

  def test_fingerprint_separation(self):
    trace_count = 0

    @jax.jit(cache_key="fp_sep", static_argnames=("k",))
    def f(x, *, k: int = 1):
      nonlocal trace_count
      trace_count += 1
      return x * k

    cache = InMemoryCache()
    x4 = jnp.ones(4, dtype=jnp.float32)
    x8 = jnp.ones(8, dtype=jnp.float32)

    with set_cache(cache):
      f(x4, k=1)
      self.assertEqual(trace_count, 1)

      # Different shape -> cache miss, re-traces and caches separate Lowered
      f(x8, k=1)
      self.assertEqual(trace_count, 2)

      # Different static arg -> cache miss, re-traces
      f(x4, k=2)
      self.assertEqual(trace_count, 3)

      # New instance with (x4, k=1) hits cached Lowered
      @jax.jit(cache_key="fp_sep", static_argnames=("k",))
      def f_new(x, *, k: int = 1):
        raise RuntimeError("Should not trace!")

      self.assertArraysEqual(f_new(x4, k=1), x4)
      self.assertEqual(cache.hits, 1)

  def test_cold_start_lower_populates_cache_and_hits_subsequent(self):
    trace_count = 0

    @jax.jit(cache_key="cold_lower_test")
    def f(x):
      nonlocal trace_count
      trace_count += 1
      return x * 7.0

    cache = InMemoryCache()
    x = jnp.ones(4, dtype=jnp.float32)

    with set_cache(cache):
      # 1. Cold-start .lower(x) without f(x) first: must populate Lowered cache
      lowered = f.lower(x)
      self.assertIsInstance(lowered, Lowered)
      self.assertEqual(trace_count, 1)
      self.assertEqual(cache.hits, 0)
      self.assertEqual(cache.misses, 1)

      # 2. Subsequent call on new jit instance with same key hits cached Lowered
      @jax.jit(cache_key="cold_lower_test")
      def f_new(x):
        raise RuntimeError("Should not trace!")

      res = f_new(x)
      self.assertArraysEqual(res, x * 7.0)
      self.assertEqual(cache.hits, 1)

      # 3. Subsequent .lower(x) on another jit instance also hits cached Lowered
      @jax.jit(cache_key="cold_lower_test")
      def f_lower_only(x):
        raise RuntimeError("Should not trace!")

      lowered2 = f_lower_only.lower(x)
      self.assertIsInstance(lowered2, Lowered)
      self.assertEqual(cache.hits, 2)

  def test_call_then_lower_on_same_instance_hits_cache(self):
    trace_count = 0

    @jax.jit(cache_key="same_instance_test")
    def f(x):
      nonlocal trace_count
      trace_count += 1
      return x * 5.0

    cache = InMemoryCache()
    x = jnp.ones(4, dtype=jnp.float32)

    with set_cache(cache):
      # 1. f(x) runs, traces and populates Lowered cache
      f(x)
      self.assertEqual(trace_count, 1)
      self.assertEqual(cache.hits, 0)
      self.assertEqual(cache.misses, 1)

      # 2. Calling f.lower(x) on the SAME instance hits Lowered cache
      lowered = f.lower(x)
      self.assertIsInstance(lowered, Lowered)
      self.assertEqual(trace_count, 1)
      self.assertEqual(cache.hits, 1)

      # 3. Calling .lower(x) on a NEW instance with same key hits Lowered cache
      @jax.jit(cache_key="same_instance_test")
      def f_new(x):
        return x * 5.0

      lowered2 = f_new.lower(x)
      self.assertIsInstance(lowered2, Lowered)
      self.assertEqual(cache.hits, 2)

  def test_closed_over_array_constants_cached(self):
    const_arr = jnp.array([10.0, 20.0], dtype=jnp.float32)

    @jax.jit(cache_key="closed_over_const_fn")
    def f(x):
      return x + const_arr

    cache = InMemoryCache()
    x = jnp.ones(2, dtype=jnp.float32)

    with set_cache(cache):
      res = f(x)
      self.assertArraysEqual(res, jnp.array([11.0, 21.0], dtype=jnp.float32))
      self.assertLen(cache, 1)

      lowered = f.lower(x)
      self.assertIsInstance(lowered, Lowered)
      self.assertEqual(cache.hits, 1)

  def test_donated_invars_and_shardings_preserved_on_cache_hit(self):
    @functools.partial(jax.jit, donate_argnums=(0,), cache_key="donate_test")
    def f(x, y):
      return x + y

    cache = InMemoryCache()
    x = jnp.ones(4, dtype=jnp.float32)
    y = jnp.ones(4, dtype=jnp.float32)

    with set_cache(cache):
      # First run lowers and caches
      lowered1 = f.lower(x, y)
      self.assertTrue(lowered1.args_info[0][0].donated)
      self.assertFalse(lowered1.args_info[0][1].donated)
      self.assertEqual(cache.hits, 0)
      self.assertEqual(cache.misses, 1)

      # Second run on a new jit instance with the same key
      @functools.partial(jax.jit, donate_argnums=(0,), cache_key="donate_test")
      def f2(x, y):
        raise RuntimeError("Should hit cache!")

      # Verify that Lowered cache hit preserves donated_invars
      lowered2 = f2.lower(x, y)
      self.assertTrue(lowered2.args_info[0][0].donated)
      self.assertFalse(lowered2.args_info[0][1].donated)
      self.assertEqual(cache.hits, 1)

  def test_stage_compute_cache_key(self):
    from jax._src.stages import _compute_cache_key
    cache = InMemoryCache()
    with set_cache(cache):
      key = _compute_cache_key(Lowered, "my_user_key", ("meta1", 42))
      self.assertIsNotNone(key)
      self.assertTrue(key.startswith("my_user_key:Lowered:"))
      self.assertIn("Lowered", key)

  def test_stage_can_cache(self):
    @jax.jit
    def f(x):
      return x * 2.0

    x = jnp.ones(2, dtype=jnp.float32)
    traced = f.trace(x)
    self.assertFalse(traced.can_cache)

    lowered = f.lower(x)
    self.assertTrue(lowered.can_cache)

    compiled = lowered.compile()
    self.assertFalse(compiled.can_cache)

  def test_cache_hit_then_grad_and_nested_jit(self):
    @functools.partial(jax.jit, cache_key="grad_nested_test_fn")
    def f(x):
      return jnp.sum(x ** 2)

    cache = InMemoryCache()
    x = jnp.array([2.0, 3.0], dtype=jnp.float32)

    with set_cache(cache):
      # 1. First run populates the cache
      res1 = f(x)
      self.assertEqual(res1, 13.0)
      self.assertIn(cache.misses, (1, 2))

      # 2. Calling jax.grad(f)(x) on the same function instance must NOT reuse dummy Jaxpr!
      grad_f = jax.grad(f)
      df = grad_f(x)
      self.assertArraysAllClose(df, jnp.array([4.0, 6.0], dtype=jnp.float32))

      # 3. A new jit instance with the same key hits the Lowered stage cache
      @functools.partial(jax.jit, cache_key="grad_nested_test_fn")
      def f2(x):
        raise RuntimeError("Should hit cache!")

      lowered2 = f2.lower(x)
      self.assertIsInstance(lowered2, Lowered)
      self.assertEqual(cache.hits, 1)

      # 4. Calling f inside an outer jit must also trace and succeed cleanly!
      @jax.jit
      def outer(x):
        return f(x) + 1.0

      self.assertEqual(outer(x), 14.0)

  def test_infer_params_cache_frame_reuse(self):
    @functools.partial(jax.jit, cache_key="infer_params_reuse_fn")
    def f(x):
      return x * 3.0

    cache = InMemoryCache()
    x = jnp.array([1.0, 2.0], dtype=jnp.float32)

    with set_cache(cache):
      # Case 1: eval_shape first, then call on the same instance
      s = f.eval_shape(x)
      self.assertEqual(s.shape, (2,))
      res = f(x)
      self.assertArraysEqual(res, jnp.array([3.0, 6.0], dtype=jnp.float32))

      # Case 2: hit on new instance f2, then f2.lower(x) and f2.eval_shape(x)
      @functools.partial(jax.jit, cache_key="infer_params_reuse_fn")
      def f2(x):
        raise RuntimeError("Should hit cache!")

      res2 = f2(x)
      self.assertArraysEqual(res2, jnp.array([3.0, 6.0], dtype=jnp.float32))
      l2 = f2.lower(x)
      self.assertIsInstance(l2, Lowered)
      s2 = f2.eval_shape(x)
      self.assertEqual(s2.shape, (2,))

      # Verify that f2.trace(x) returns Traced carrying cached_lowered
      traced2 = f2.trace(x)
      self.assertIsInstance(traced2, Traced)
      self.assertIs(traced2.cached_lowered, l2)
      self.assertIs(traced2._params.get("cached_lowered"), l2)
      self.assertEqual(traced2.out_info.shape, (2,))

  def test_closed_over_tracer_under_cache(self):
    cache = InMemoryCache()
    @jax.jit(cache_key="closed_tracer_fn")
    def inner(x, y):
      return jnp.sum(x * y)

    def outer(y):
      return jax.jit(lambda x: inner(x, y), cache_key="inner_closure")(jnp.ones(2))

    with set_cache(cache):
      grad_val = jax.grad(outer)(3.0)
      self.assertEqual(grad_val, 2.0)

    with set_cache(cache):
      grad_val = jax.grad(outer)(3.0)
      self.assertEqual(grad_val, 2.0)

  def test_set_cache_nesting_error(self):
    cache1 = InMemoryCache()
    cache2 = InMemoryCache()
    msg = (
        "Nesting set_cache contexts is not currently supported. If you need"
        " this functionality, please contact dsuo@ or file an issue."
    )
    with set_cache(cache1):
      with self.assertRaisesRegex(ValueError, re.escape(msg)):
        with set_cache(cache2):
          pass

    # Verify that cache is reset after exiting the outer context
    with set_cache(cache2):
      pass


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
