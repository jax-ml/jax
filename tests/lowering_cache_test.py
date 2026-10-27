# Copyright 2026 The JAX Authors.
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

from __future__ import annotations

import tempfile

from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import lowering_cache
from jax._src import path as pathlib
from jax._src import stages
from jax._src import test_util as jtu
from jax._src.cache_interface import CacheInterface
import numpy as np

config.parse_flags_with_absl()


class InMemoryCache(CacheInterface):
  """In-memory cache implementing CacheInterface for testing."""

  _path = pathlib.Path()

  def __init__(self):
    self._data: dict[str, bytes] = {}

  def get(self, key: str) -> bytes | None:
    return self._data.get(key)

  def put(self, key: str, value: bytes) -> None:
    self._data[key] = value

  def clear(self) -> None:
    self._data.clear()

  def __len__(self) -> int:
    return len(self._data)


@jtu.thread_unsafe_test_class()
class LoweringCacheTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    lowering_cache.reset_cache()
    self._orig_dir = config.lowering_cache_dir.value
    self._orig_enabled = config.enable_lowering_cache.value
    self._orig_explain = config.explain_cache_misses.value

  def tearDown(self):
    lowering_cache.reset_cache()
    config.config.update("jax_lowering_cache_dir", self._orig_dir)
    config.config.update("jax_enable_lowering_cache", self._orig_enabled)
    config.config.update("jax_explain_cache_misses", self._orig_explain)
    super().tearDown()

  def test_cache_key_determinism(self):
    def f(x, y):
      return x + y

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)

    jit_f = jax.jit(f)
    ji = jit_f._jit_info
    (arg_sig, avals, ctx_mesh), _, _ = jax._src.pjit._parse_jit_inputs(
        f, ji, (x, y), {}
    )

    key1 = lowering_cache.get_cache_key(f, ji, arg_sig, avals, ctx_mesh)
    key2 = lowering_cache.get_cache_key(f, ji, arg_sig, avals, ctx_mesh)
    self.assertEqual(key1, key2)

  def test_cache_key_sensitivity_to_shape(self):
    def f(x):
      return x * 2

    jit_f = jax.jit(f)
    ji = jit_f._jit_info

    x1 = np.ones((2, 3), dtype=np.float32)
    x2 = np.ones((2, 4), dtype=np.float32)

    (sig1, avals1, mesh1), _, _ = jax._src.pjit._parse_jit_inputs(
        f, ji, (x1,), {}
    )
    (sig2, avals2, mesh2), _, _ = jax._src.pjit._parse_jit_inputs(
        f, ji, (x2,), {}
    )

    key1 = lowering_cache.get_cache_key(f, ji, sig1, avals1, mesh1)
    key2 = lowering_cache.get_cache_key(f, ji, sig2, avals2, mesh2)
    self.assertNotEqual(key1, key2)

  def test_cache_key_sensitivity_to_dtype(self):
    def f(x):
      return x * 2

    jit_f = jax.jit(f)
    ji = jit_f._jit_info

    x1 = np.ones((2, 3), dtype=np.float32)
    x2 = np.ones((2, 3), dtype=np.int32)

    (sig1, avals1, mesh1), _, _ = jax._src.pjit._parse_jit_inputs(
        f, ji, (x1,), {}
    )
    (sig2, avals2, mesh2), _, _ = jax._src.pjit._parse_jit_inputs(
        f, ji, (x2,), {}
    )

    key1 = lowering_cache.get_cache_key(f, ji, sig1, avals1, mesh1)
    key2 = lowering_cache.get_cache_key(f, ji, sig2, avals2, mesh2)
    self.assertNotEqual(key1, key2)

  def test_cache_key_sensitivity_to_code(self):
    def f1(x):
      return x + 1

    def f2(x):
      return x + 2

    x = np.array([1, 2], dtype=np.int32)

    jit_f1 = jax.jit(f1)
    jit_f2 = jax.jit(f2)

    (sig1, avals1, mesh1), _, _ = jax._src.pjit._parse_jit_inputs(
        f1, jit_f1._jit_info, (x,), {}
    )
    (sig2, avals2, mesh2), _, _ = jax._src.pjit._parse_jit_inputs(
        f2, jit_f2._jit_info, (x,), {}
    )

    key1 = lowering_cache.get_cache_key(
        f1, jit_f1._jit_info, sig1, avals1, mesh1
    )
    key2 = lowering_cache.get_cache_key(
        f2, jit_f2._jit_info, sig2, avals2, mesh2
    )
    self.assertNotEqual(key1, key2)

  def test_cache_key_sensitivity_to_custom_hook(self):
    def f(x):
      return x + 1

    x = np.array([1, 2], dtype=np.int32)
    jit_f = jax.jit(f)
    (sig, avals, mesh), _, _ = jax._src.pjit._parse_jit_inputs(
        f, jit_f._jit_info, (x,), {}
    )

    key_orig = lowering_cache.get_cache_key(
        f, jit_f._jit_info, sig, avals, mesh
    )

    orig_hook = lowering_cache.custom_hook
    try:
      lowering_cache.custom_hook = lambda: "cl/99999999"
      key_custom = lowering_cache.get_cache_key(
          f, jit_f._jit_info, sig, avals, mesh
      )
      self.assertNotEqual(key_orig, key_custom)
    finally:
      lowering_cache.custom_hook = orig_hook

  def test_lowered_serialize_deserialize_roundtrip(self):
    def f(x, y):
      return jnp.sin(x) + jnp.cos(y)

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    lowered = jax.jit(f).lower(x, y)
    blob = lowering_cache.serialize_lowered(lowered)
    self.assertIsInstance(blob, bytes)
    self.assertNotEmpty(blob)

    loaded = lowering_cache.deserialize_lowered(blob)
    self.assertIsInstance(loaded, stages.Lowered)

    # Compile both and verify numerical parity
    out_orig = lowered.compile()(x, y)
    out_loaded = loaded.compile()(x, y)
    self.assertArraysAllClose(out_orig, out_loaded)

  def test_lowering_cache_get_put(self):
    cache = InMemoryCache()
    lowering_cache._cache = cache
    lowering_cache._cache_initialized = True
    config.config.update("jax_enable_lowering_cache", True)

    def f(x):
      return x * 3.0 + 1.0

    x = np.array([1.0, 2.0], dtype=np.float32)
    lowered = jax.jit(f).lower(x)

    key = "test_key_001"
    lowering_cache.put_lowered(key, lowered)
    self.assertIn(key, cache._data)

    retrieved = lowering_cache.get_lowered(key)
    self.assertIsNotNone(retrieved)
    out = retrieved.compile()(x)
    self.assertArraysAllClose(out, f(x))

  def test_jit_bypasses_tracing_on_lowering_cache_hit(self):
    trace_count = 0

    def f(x):
      nonlocal trace_count
      trace_count += 1
      return x * 2.0 + 5.0

    with tempfile.TemporaryDirectory() as tmpdir:
      lowering_cache.set_cache_dir(tmpdir)
      config.config.update("jax_enable_lowering_cache", True)

      x = np.array([10.0, 20.0], dtype=np.float32)

      # 1. First execution: cache miss, function is traced once
      f_jit = jax.jit(f)
      res1 = f_jit(x)
      self.assertEqual(trace_count, 1)
      self.assertArraysAllClose(res1, np.array([25.0, 45.0], dtype=np.float32))

      # 2. Reset in-memory JIT cache, but keep persistent lowering cache
      f_jit.clear_cache()
      jax._src.pjit._infer_params_cached.cache_clear()

      # 3. Second execution in fresh JIT wrapper: cache hit, function NOT traced
      f_jit_new = jax.jit(f)
      res2 = f_jit_new(x)
      self.assertEqual(trace_count, 1)  # Tracing was bypassed!
      self.assertArraysAllClose(res2, res1)

  def test_lower_bypasses_tracing_on_lowering_cache_hit(self):
    trace_count = 0

    def f(x):
      nonlocal trace_count
      trace_count += 1
      return jnp.exp(x)

    with tempfile.TemporaryDirectory() as tmpdir:
      lowering_cache.set_cache_dir(tmpdir)
      config.config.update("jax_enable_lowering_cache", True)

      x = np.array([1.0, 2.0], dtype=np.float32)

      # First lower call: traces once and stores to cache
      lowered1 = jax.jit(f).lower(x)
      self.assertEqual(trace_count, 1)

      # Clear in-memory caches
      jax._src.pjit._infer_params_cached.cache_clear()

      # Second lower call: loads from lowering cache, no tracing
      lowered2 = jax.jit(f).lower(x)
      self.assertEqual(trace_count, 1)  # Tracing was bypassed!

      # Compiled outputs match
      out1 = lowered1.compile()(x)
      out2 = lowered2.compile()(x)
      self.assertArraysAllClose(out1, out2)

  def test_persistent_cache_survives_reset(self):
    trace_count = 0

    def f(x):
      nonlocal trace_count
      trace_count += 1
      return x**2 + 3

    with tempfile.TemporaryDirectory() as tmpdir:
      lowering_cache.set_cache_dir(tmpdir)
      config.config.update("jax_enable_lowering_cache", True)

      x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
      res1 = jax.jit(f)(x)
      self.assertEqual(trace_count, 1)

      # Simulate complete process restart by resetting lowering_cache module state
      lowering_cache.reset_cache()
      jax._src.pjit._infer_params_cached.cache_clear()

      # Re-initialize pointing to the same cache directory
      lowering_cache.set_cache_dir(tmpdir)
      config.config.update("jax_enable_lowering_cache", True)

      res2 = jax.jit(f)(x)
      self.assertEqual(trace_count, 1)  # Tracing was bypassed via disk cache!
      self.assertArraysAllClose(res1, res2)

  def test_lowering_cache_disabled_flag(self):
    trace_count = 0

    def f(x):
      nonlocal trace_count
      trace_count += 1
      return x + 1

    with tempfile.TemporaryDirectory() as tmpdir:
      lowering_cache.set_cache_dir(tmpdir)
      config.config.update(
          "jax_enable_lowering_cache", False
      )  # Explicitly disabled

      x = np.array([1, 2], dtype=np.int32)
      f_jit = jax.jit(f)
      f_jit(x)
      self.assertEqual(trace_count, 1)

      f_jit.clear_cache()
      jax._src.pjit._infer_params_cached.cache_clear()
      f_jit(x)
      # Because cache is disabled, tracing happens again
      self.assertEqual(trace_count, 2)

  def test_can_cache_clean(self):
    def f(x):
      return x * 2.0

    lowered = jax.jit(f).lower(np.array([1.0], dtype=np.float32))
    self.assertTrue(lowering_cache.can_cache(lowered))

  def test_can_cache_rejects_uncacheable(self):
    class DummyLowering:

      def __init__(self):
        self.compile_args = {
            "unordered_effects": ["custom_effect_1"],
        }

    class DummyLowered:

      def __init__(self):
        self._lowering = DummyLowering()

    dummy_lowered = DummyLowered()
    self.assertFalse(lowering_cache.can_cache(dummy_lowered))

  def test_explain_cannot_cache_multi_reasons(self):
    class MeshComputation:

      def __init__(self):
        self.compile_args = {
            "unordered_effects": ["eff_1"],
            "ordered_effects": ["eff_2"],
            "host_callbacks": ["cb_1"],
            "keepalive": [object()],
            "mut": [1],
        }

    class DummyLowered:

      def __init__(self):
        self._lowering = MeshComputation()

    dummy = DummyLowered()
    reasons = lowering_cache.explain_cannot_cache(dummy)
    self.assertEqual(len(reasons), 5)
    self.assertTrue(any("unordered_effects" in r for r in reasons))
    self.assertTrue(any("ordered_effects" in r for r in reasons))
    self.assertTrue(any("host_callbacks" in r for r in reasons))
    self.assertTrue(any("keepalive" in r for r in reasons))
    self.assertTrue(any("mutable arguments" in r for r in reasons))

  def test_can_cache_logs_warning_on_explain_cache_misses(self):
    class DummyLowering:

      def __init__(self):
        self._name = "test_func"
        self.compile_args = {
            "unordered_effects": ["eff_1"],
            "host_callbacks": ["cb_1"],
        }

    class DummyLowered:

      def __init__(self):
        self._lowering = DummyLowering()

    dummy = DummyLowered()
    with config.explain_cache_misses(True):
      with self.assertLogs(lowering_cache.logger, level="WARNING") as cm:
        can = lowering_cache.can_cache(dummy)
        self.assertFalse(can)
        self.assertTrue(any("Cannot cache lowered" in msg for msg in cm.output))
        self.assertTrue(any("unordered_effects" in msg for msg in cm.output))
        self.assertTrue(any("host_callbacks" in msg for msg in cm.output))

  def test_put_lowered_failure_logs_warning_on_explain_cache_misses(self):
    class FailingCache(InMemoryCache):

      def put(self, key: str, value: bytes) -> None:
        raise RuntimeError("Disk quota exceeded")

    lowering_cache._cache = FailingCache()
    lowering_cache._cache_initialized = True
    config.config.update("jax_enable_lowering_cache", True)

    def f(x):
      return x * 2.0

    lowered = jax.jit(f).lower(np.array([1.0], dtype=np.float32))

    with config.explain_cache_misses(True):
      with config.raise_persistent_cache_errors(False):
        with self.assertLogs(lowering_cache.logger, level="WARNING") as cm:
          lowering_cache.put_lowered(
              "test_fail_key", lowered, fun_name="my_func"
          )
          self.assertTrue(
              any(
                  "Failed to store lowering cache entry for my_func" in msg
                  for msg in cm.output
              )
          )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
