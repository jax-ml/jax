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
import logging
from typing import Any, Callable, Sequence
import unittest

from absl.testing import absltest
import jax
from jax import lax
from jax._src import aot
from jax._src import api
from jax._src import aot_util
from jax._src import config
from jax._src import core
from jax._src import pjit
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
  @jtu.run_on_devices("tpu", "gpu")
  def test_pickle_jit_lower(self):
    def fun(x):
      return x * x

    with jax.set_mesh(jax.sharding.Mesh(np.array(jax.devices()), ("data",))):
      lowered = jax.jit(
        fun, in_shardings=P("data"), out_shardings=P(None, "data")
      ).lower(core.ShapedArray(shape=(8, 8), dtype=np.float32))

    def verify_serialization(lowered):
      serialized, in_tree, out_tree = serialize(lowered.compile())
      compiled = deserialize_and_load(serialized, in_tree, out_tree)
      self.assertEqual(compiled.as_text(), lowered.compile().as_text())

    verify_serialization(lowered)
    verify_serialization(jax.jit(lambda x: x * x).lower(np.arange(100)))
    verify_serialization(
      jax.pmap(lambda x: x * x).lower(
        np.zeros((len(jax.devices()), 4), dtype=np.float32)
      )
    )

  @jtu.skip_on_devices("tpu")  # TODO(phawkins): This test is segfaulting on TPU
  def test_topology_jit_serialize(self):
    try:
      aot_topo = topologies.get_topology_desc(
        platform=jax.devices()[0].platform
      )
    except NotImplementedError:
      raise unittest.SkipTest("PJRT Topology not supported")

    if jtu.TEST_WITH_PERSISTENT_COMPILATION_CACHE.value:
      raise unittest.SkipTest("Compilation caching not yet supported.")
    if jtu.is_device_cuda():
      raise unittest.SkipTest("Broken on GPU: b/442353988")

    @jax.jit
    def fn(x):
      return x * x

    def lower_and_load(mesh):
      s = jax.sharding.NamedSharding(mesh, P("x", "y"))
      x_shape = jax.ShapeDtypeStruct(
        shape=(16, 16), dtype=jnp.dtype("float32"), sharding=s
      )
      lowered = fn.lower(x_shape)
      serialized, in_tree, out_tree = serialize(lowered.compile())
      compiled = deserialize_and_load(serialized, in_tree, out_tree)
      return compiled

    ref_topo = topologies.get_attached_topology()
    n = max(1, len(ref_topo.devices) // 2)
    mesh_shape = (len(ref_topo.devices) // n, n)

    ref_mesh = topologies.make_mesh(ref_topo, mesh_shape, ("x", "y"))
    aot_mesh = topologies.make_mesh(aot_topo, mesh_shape, ("x", "y"))
    self.assertEqual(
      lower_and_load(ref_mesh).as_text(), lower_and_load(aot_mesh).as_text()
    )

  def test_get_topology_from_devices(self):
    try:
      aot_topo = topologies.get_topology_desc(
        platform=jax.devices()[0].platform
      )
    except NotImplementedError:
      raise unittest.SkipTest("PJRT Topology not supported")

    topo = xc.get_topology_for_devices(aot_topo.devices)
    self.assertEqual(
      topo.platform_version, aot_topo.devices[0].client.platform_version
    )

  def test_lower_as_text_with_and_without_debug_info(self):
    def my_function(x):
      return jnp.sin(x)

    lowered = jax.jit(my_function).lower(42.0)
    stablehlo = lowered.as_text("stablehlo", debug_info=True)
    self.assertRegex(stablehlo, r"sine.* loc")
    stablehlo = lowered.as_text("stablehlo")
    self.assertNotRegex(stablehlo, r"sine.* loc")

    hlo = lowered.as_text("hlo", debug_info=True)
    self.assertRegex(hlo, r"sine.*metadata=.*source_file=.*")
    hlo = lowered.as_text("hlo")
    self.assertNotRegex(hlo, r"sine.*metadata=.*source_file=.*")

  def test_constants_in_lowering_in_aot(self):
    const_size = 100
    const = jax.random.uniform(
      jax.random.key(0), (const_size,), dtype=np.float32
    )

    def my_function(x):
      return jnp.sin(x) + const

    lowered = jax.jit(my_function).lower(
      np.full_like(const, 42.0, dtype=const.dtype)
    )
    stablehlo = lowered.as_text("stablehlo")
    if config.use_simplified_jaxpr_constants.value:
      self.assertNotRegex(
        stablehlo, rf"stablehlo.constant dense.*tensor<{const_size}x"
      )
      self.assertLen(lowered._lowering.const_args, 1)
      self.assertIs(lowered._lowering.const_args[0], const)
    else:
      self.assertRegex(
        stablehlo, rf"stablehlo.constant dense.*tensor<{const_size}x"
      )
      self.assertLen(lowered._lowering.const_args, 0)

  def test_with_constants(self):
    const = jnp.arange(16.0) + 42.0  # A distinctive shape and value

    @jax.jit
    def f(x):
      return const[0:8] + x

    inp = jnp.arange(8.0)
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
    ]
  )
  def test_with_constants_enable_x64(self, *, use_np, lower, compile, exec):
    # Closed-over constant is 64-bit. Each of lowering, compilation, and
    # execution can be run in 64-bit or 32-bit mode.
    with config.enable_x64(True):
      arange = np.arange if use_np else jnp.arange
      const = arange(8, dtype=np.int64) + 42

      @jax.jit
      def f(x):
        return lax.convert_element_type(const, np.float32) + x

    inp = np.arange(8.0, dtype=np.float32)
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
    if (
      config.use_simplified_jaxpr_constants.value
      and not config.enable_x64.value
      and use_np
      and lower
      and not exec
    ):
      with self.assertRaisesRegex(
        xc.XlaRuntimeError, "got buffer with incompatible size"
      ):
        run()
      return

    self.assertArraysEqual(
      run(), lax.convert_element_type(const, inp.dtype) + inp
    )
    # Trigger cache hit
    self.assertCacheMisses(run, cpp=0, aot_call=0)

  def test_with_ref_constants(self):
    x_ref = core.new_ref(0)

    @jax.jit
    def f(x):
      x_ref[...] += x

    f_lowered = f.lower(1)
    with self.assertRaisesRegex(ValueError, "serialize with a closed-over"):
      serialized, in_tree, out_tree = serialize(f_lowered.compile())

  @jtu.run_on_devices("gpu", "tpu")
  def test_mismatched_backends_raises(self):
    @jax.jit
    def f(x):
      return x * 2

    x = jnp.arange(1)
    f_lowered = f.lower(x)
    serialized, in_tree, out_tree = serialize(f_lowered.compile())
    with self.assertRaisesRegex(
      ValueError, "Execution devices belong to a client other than `backend`"
    ):
      deserialize_and_load(
        serialized,
        in_tree,
        out_tree,
        backend="cpu",
        execution_devices=jax.devices()[:1],
      )


@jtu.thread_unsafe_test_class()
class ComponentTest(jtu.JaxTestCase):
  @contextlib.contextmanager
  def make_in_memory_cache(self):
    cache = aot_util.Cache()
    with aot_util.component_cache(cache):
      yield
      jax.clear_caches()

  # TODO(dsuo): It would be nice to have a way to grab the pjit jaxpr cache
  # key easily.
  def get_jaxpr_key(self, fun: Callable[..., Any]) -> Callable[..., Any] | None:
    for key in pjit._create_pjit_jaxpr.cache_keys():
      f = key if not hasattr(key, "__wrapped__") else key.__wrapped__
      if f == fun:
        return key

  def validate_cache_states(
    self,
    fun: Callable[..., Any],
    num_jaxpr_entries: int,
    num_jaxpr_hits: int | Sequence[int],
    num_trace_hits: int,
    num_trace_misses: int,
    num_wrapper_hits: int,
    num_disk_hits: int,
  ):
    cache = aot.get_cache()
    component_key = fun.component_key  # type: ignore

    # Verify component key exists in disk cache.
    self.assertIn(component_key, cache.cache_keys())
    # Verify the number of wrapper cache hits.
    self.assertEqual(
      aot_util._wrapper_cache.cache_info()[component_key]["hits"],
      num_wrapper_hits,
    )

    # Verify the number of disk hits.
    self.assertEqual(cache.cache_info(component_key)["hits"], num_disk_hits)

    jaxpr_key = self.get_jaxpr_key(fun.fun)
    jaxpr_cache = pjit._create_pjit_jaxpr.cache_get(jaxpr_key)
    jaxpr_hit = pjit._create_pjit_jaxpr.hit_get(jaxpr_key)
    if isinstance(num_jaxpr_hits, int):
      num_jaxpr_hits = (num_jaxpr_hits,)

    # Verify fun exists in the jaxpr cache.
    self.assertIsNotNone(jaxpr_key)

    # Verify number of entries in jaxpr cache for fun.
    self.assertEqual(len(jaxpr_cache), num_jaxpr_entries)

    # Verify number of hits for each entry.
    self.assertEqual(tuple(jaxpr_hit.values()), num_jaxpr_hits)

    # Verify the number of hits and misses we expect.
    self.assertEqual(
      pjit._infer_params_cached.cache_info().hits, num_trace_hits
    )
    self.assertEqual(
      pjit._infer_params_cached.cache_info().misses, num_trace_misses
    )

  # NOTE(dsuo): Disable checks because otherwise we check jaxprs in (at least)
  # four places and makes reasoning about cache hits and misses harder.
  # 1. After the initial abstract eval.
  # 2. Before converting const vars.
  # 3. After lifting the jaxpr.
  # 4. After DCE.
  @config.enable_checks(False)
  def test_component_basic(self):
    with self.make_in_memory_cache():
      cache = aot.get_cache()

      @aot.component(key="f")
      def f(x):
        return x + 1.0

      self.assertEqual(f(1.0), 2.0)
      self.validate_cache_states(
        f,
        # Make sure there is only one entry for f.fun. If there are more, then
        # it means the lowering rule missed.
        num_jaxpr_entries=1,
        # There should be no hits in the jaxpr cache for f.fun because we've
        # only just created it.
        num_jaxpr_hits=0,
        # We should have 1 hit from the trace in lowering.
        num_trace_hits=1,
        # We should have 4 misses: add, equal, f, and call_wrapped.
        num_trace_misses=4,
        # We shouldn't have hit the wrapper cache yet.
        num_wrapper_hits=0,
        # We get 1 hit on the disk cache during the lowering rule. However, this
        # hit is for an incomplete CacheEntry; only avals_out were populated
        # and not the lowered module. The lowering rule updates the CacheEntry
        # with the lowered module.
        num_disk_hits=1,
      )

      @aot.component(key="f")
      def g(x):
        raise NotImplementedError

      self.assertEqual(f.fun, g.fun)
      self.assertEqual(g(1.0), 2.0)
      # Cache state should remain unchanged except we grabbed the wrapped fun.
      self.validate_cache_states(g, 1, 0, 1, 4, 1, 1)

  @config.enable_checks(False)
  def test_component_in_function(self):
    with self.make_in_memory_cache():
      cache = aot.get_cache()

      @aot.component(key="f")
      def f(x):
        return x + 1.0

      @jax.jit
      def g(x):
        return f(x) + 1.0

      self.assertEqual(f(1.0), 2.0)

      # We should have the same cache states as in test_component_basic.
      self.validate_cache_states(f, 1, 0, 1, 4, 1)

      logging.info("\n\n\n")

      # 1 hit when lowering g. g is not a component, so doesn't look up
      # CacheEntry during abstract_eval.
      self.assertEqual(g(1.0), 3.0)
      # We incur one more missed trace for g and
      self.validate_cache_states(f, 1, 0, 2, 6, 2)
      # # Make sure we didn't add any new entries for f.fun.
      # num_entries = len(list(self.get_pjit_jaxpr_entry(pjit_key)))
      # self.assertEqual(num_entries, 1)
      # self.assertEqual(cache.info(f.component_key)["hits"], 2)

  @config.enable_checks(False)
  def test_jit_of_component(self):
    with self.make_in_memory_cache():
      cache = aot.get_cache()

      @jax.jit
      @aot.component(key="f")
      def f(x):
        return x + 1.0

      # Create cache entry when abstract_eval f. 1 hit when lowering f.
      self.assertEqual(f(1.0), 2.0)
      # Make sure the underlying function f.fun exists in the jaxpr cache.
      pjit_key = self.get_jaxpr_key(f.fun)
      self.assertIsNotNone(pjit_key)
      # Make sure there is only one entry for f.fun. If there are more, then it
      # means the lowering rule missed.
      num_entries = len(list(self.get_pjit_jaxpr_entry(pjit_key)))
      self.assertEqual(num_entries, 1)

      @aot.component(key="f")
      def g(x):
        raise NotImplementedError

      # We ignore g's implementation because it was turned into a component with
      # key "f".
      self.assertEqual(f.fun, g.fun)
      self.assertEqual(g(1.0), 2.0)
      # Confirm we still have just one entry in the jaxpr cache.
      num_entries = len(list(self.get_pjit_jaxpr_entry(pjit_key)))
      self.assertEqual(num_entries, 1)
      # We get two additional hits on the disk cache during abstract eval and
      # lowering for g.
      self.assertEqual(cache.info(f.component_key)["hits"], 3)

  @config.enable_checks(False)
  def test_component_of_jit(self):
    with self.make_in_memory_cache():
      cache = aot.get_cache()

      @aot.component(key="f")
      @jax.jit
      def f(x):
        return x + 1.0

      # Create cache entry when abstract_eval f. 1 hit when lowering f.
      self.assertEqual(f(1.0), 2.0)
      # Make sure the underlying function f.fun exists in the jaxpr cache.
      pjit_key = self.get_jaxpr_key(f.fun)
      self.assertIsNotNone(pjit_key)
      # Make sure there is only one entry for f.fun. If there are more, then it
      # means the lowering rule missed.
      num_entries = len(list(self.get_pjit_jaxpr_entry(pjit_key)))
      self.assertEqual(num_entries, 1)

      @aot.component(key="f")
      def g(x):
        raise NotImplementedError

      # We ignore g's implementation because it was turned into a component with
      # key "f".
      self.assertEqual(f.fun, g.fun)
      self.assertEqual(g(1.0), 2.0)
      # Confirm we still have just one entry in the jaxpr cache.
      num_entries = len(list(self.get_pjit_jaxpr_entry(pjit_key)))
      self.assertEqual(num_entries, 1)
      # We get two additional hits on the disk cache during abstract eval and
      # lowering for g.
      self.assertEqual(cache.info(f.component_key)["hits"], 3)

  @config.enable_checks(False)
  def test_explicit_lowering(self):
    with self.make_in_memory_cache():
      cache = aot.get_cache()

      @aot.component(key="f")
      def f(x):
        return x + 1.0

      lowered = f.lower(jax.ShapeDtypeStruct((), "float32"))
      self.assertEqual(cache.keys(), [f.component_key])

      pjit_key = self.get_jaxpr_key(f.fun)
      self.assertIsNotNone(pjit_key)
      # Make sure there is only one entry for f.fun. If there are more, then it
      # means the lowering rule missed.
      num_entries = len(list(self.get_pjit_jaxpr_entry(pjit_key)))
      self.assertEqual(num_entries, 1)

      @aot.component(key="f")
      def g(x):
        raise NotImplementedError

      lowered = g.lower(jax.ShapeDtypeStruct((), "float32"))
      self.assertEqual(cache.info(f.component_key)["hits"], 3)

  @config.enable_checks(False)
  def test_vmap_of_component(self):
    with self.make_in_memory_cache():
      cache = aot.get_cache()

      @aot.component(key="f")
      def f(x):
        logging.info("running!")
        return x + 1.0

      vmapped_f = jax.vmap(f)

      # TODO(dsuo): How to put component_key on vmapped_f? This is just a hack.
      vmapped_key = aot_util.ComponentKey.vmap(aot_util.ComponentKey("f"))

      self.assertArraysEqual(vmapped_f(jnp.ones((8,))), jnp.ones((8,)) + 1.0)
      self.assertEqual(cache.keys(), [f.component_key, vmapped_key])
      # self.assertEqual(aot_util._traced_cache[f.component_key].hits, 0)
      # self.assertEqual(aot_util._traced_cache[vmapped_key].hits, 1)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
