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
import glob
import logging
import math
import os
import shutil
import tempfile

from absl.testing import absltest
import jax
from jax._src import api
from jax._src import compilation_cache as cc
from jax._src import config
from jax._src import monitoring
from jax._src import pjit
from jax._src import profiler
from jax._src import test_util as jtu
from jax.experimental import profiler as exp_profiler
from jax.experimental.serialize_executable import (
    deserialize_and_load,
    serialize,
)
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np

jax.config.parse_flags_with_absl()


@jtu.pytest_mark_if_available('multiaccelerator')
class PgleTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest('Profile-guideded latency estimation only supported on GPU')

    cc.set_cache_dir(None)
    cc.reset_cache()

  def tearDown(self):
    cc.set_cache_dir(None)
    super().tearDown()

  def testPGLEProfilerGetFDOProfile(self):
    mesh = jtu.create_mesh((2,), ('x',))

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
        compiler_options={'xla_gpu_enable_latency_hiding_scheduler': 'True'},
    )
    def f(x, y):
      return x @ y

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    y = x + 1

    with config.pgle_profiling_runs(0):
      f_lowered = f.lower(x, y)
      compiled = f_lowered.compile()

    pgle_profiler = profiler.PGLEProfiler(1, 90)
    with config.enable_pgle(False):
      with profiler.PGLEProfiler.trace(pgle_profiler):
        compiled(x, y)

    fdo_profile = pgle_profiler.consume_fdo_profile()
    self.assertIsNotNone(fdo_profile)
    self.assertIn(b'custom', fdo_profile)

  def testPGLEProfilerGetFDOProfileLarge(self):
    mesh = jtu.create_mesh((2,), ('x',))
    its = 500

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
        compiler_options={
            'xla_gpu_enable_latency_hiding_scheduler': 'True',
            # TODO(patrios): Remove this flag once b/376647494 is fixed.
            'xla_gpu_graph_min_graph_size': '100000',
        },
    )
    def f(x):
      agg = x
      for _ in range(its):
        agg = agg @ x
      return agg

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)

    pgle_profiler = profiler.PGLEProfiler(1, 90)
    with config.enable_pgle(False):
      with profiler.PGLEProfiler.trace(pgle_profiler):
        f(x)
    fdo_profile = pgle_profiler.consume_fdo_profile()
    self.assertEqual(fdo_profile.count(b'custom'), its)

  def get_fdo_profiles(self, dump_dir):
    jit_f_fdo_profiles = [
        x
        for x in os.listdir(dump_dir)
        if 'jit_f' in x and x.endswith('.fdo_profile')
    ]
    return jit_f_fdo_profiles

  def testAutoPgle(self):
    mesh = jtu.create_mesh((2,), ('x',))

    with tempfile.TemporaryDirectory() as dump_dir:
      @partial(
          jax.jit,
          in_shardings=NamedSharding(mesh, PartitionSpec('x')),
          out_shardings=NamedSharding(mesh, PartitionSpec('x')),
          compiler_options={
              'xla_gpu_enable_latency_hiding_scheduler': 'True',
              # TODO(patrios): Remove this flag once b/376647494 is fixed.
              'xla_gpu_graph_min_graph_size': '100000',
              'xla_dump_to': dump_dir,
              'xla_gpu_experimental_dump_fdo_profiles': 'True'
          },
      )
      def f(x):
        return x * 2

      shape = (16, 16)
      x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
      expected = x * 2

      with config.pgle_profiling_runs(2), config.enable_pgle(True):
        # Run 1: Module should be compiled without FDO. Two modules are expected
        # One is the funtion f, the other one is multi slice module
        with jtu.count_jit_compilation_cache_miss() as cache_miss_count:
          self.assertArraysEqual(f(x), expected)
        self.assertEqual(cache_miss_count(), 2)

        # Run 2: Second PGLE run. Profile should be empty.
        with jtu.count_jit_compilation_cache_miss() as cache_miss_count:
          self.assertArraysEqual(f(x), expected)
        self.assertEqual(cache_miss_count(), 2)
        fdo_profiles_before_pgle = self.get_fdo_profiles(dump_dir)
        # One for before and one for after optimization.
        self.assertLen(fdo_profiles_before_pgle, 2)
        # The FDO profile file should be empty.
        self.assertEqual(
            os.path.getsize(os.path.join(dump_dir, fdo_profiles_before_pgle[0])), 0)

        # Run 3: The module should be recompiled with FDO profiles
        with jtu.count_jit_compilation_cache_miss() as cache_miss_count:
          self.assertArraysEqual(f(x), expected)
        self.assertEqual(cache_miss_count(), 2)
        fdo_profiles_after_pgle = self.get_fdo_profiles(dump_dir)
        # One for before and one for after optimization.
        self.assertLen(fdo_profiles_after_pgle, 4)

        for fdo_profile in fdo_profiles_after_pgle:
          if fdo_profile not in fdo_profiles_before_pgle:
            self.assertGreater(
                os.path.getsize(os.path.join(dump_dir, fdo_profile)), 0
            )

        # Run 4: Fast-path should be used after PGLE is done
        with jtu.count_jit_compilation_cache_miss() as cache_miss_count:
          self.assertArraysEqual(f(x), expected)
        self.assertLess(cache_miss_count(), 2)

  def testAutoPgleWithAot(self):
    @jax.jit
    def f(x):
      return x * 2

    x = jnp.arange(1)
    expected = x * 2

    f_lowered = f.lower(x)
    serialized, in_tree, out_tree = serialize(f_lowered.compile())
    compiled = deserialize_and_load(serialized, in_tree, out_tree)

    with config.pgle_profiling_runs(1), config.enable_pgle(True):
      # Run 1
      with jtu.count_jit_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(compiled(x), expected)
      self.assertEqual(cache_miss_count(), 0)

      # Run 2
      with jtu.count_jit_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(compiled(x), expected)
      self.assertEqual(cache_miss_count(), 0)

  def testAutoPgleWithPersistentCache(self):
    its = 50
    mesh = jtu.create_mesh((2,), ('x',))

    with tempfile.TemporaryDirectory() as dump_dir:
      @partial(
          jax.jit,
          in_shardings=NamedSharding(mesh, PartitionSpec('x')),
          out_shardings=NamedSharding(mesh, PartitionSpec('x')),
          compiler_options={
              'xla_gpu_enable_latency_hiding_scheduler': 'True',
              # TODO(patrios): Remove this flag once b/376647494 is fixed.
              'xla_gpu_graph_min_graph_size': '100000',
              'xla_dump_to': dump_dir,
              'xla_gpu_experimental_dump_fdo_profiles': 'True'
          },
      )
      def f(x):
        agg = x
        for _ in range(its):
          agg = agg @ x
        return agg

      shape = (16, 16)
      x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)

      with (config.enable_compilation_cache(True),
            config.enable_pgle(True),
            config.raise_persistent_cache_errors(True),
            config.raise_persistent_cache_errors(True),
            config.persistent_cache_min_entry_size_bytes(0),
            config.persistent_cache_min_compile_time_secs(0),
            config.pgle_profiling_runs(2),
            tempfile.TemporaryDirectory() as cache_dir):
        cc.reset_cache()
        cc.set_cache_dir(cache_dir)
        # Run 1: Module should be compiled without FDO
        with jtu.count_jit_compilation_cache_miss() as cache_miss_count:
          f(x)
        self.assertGreater(cache_miss_count(), 0)

        # Non-pgle profiled version of module should be saved
        non_pgle_profiled_files = os.listdir(cache_dir)
        self.assertNotEmpty(non_pgle_profiled_files)

        # Run 2: Compilation should not be called
        with jtu.count_jit_compilation_cache_miss() as cache_miss_count:
          f(x)
        self.assertGreater(cache_miss_count(), 0)

        fdo_profiles_before_pgle = self.get_fdo_profiles(dump_dir)
        # Run 3: Module should be compiled with FDO and stored to persistent cache
        with jtu.count_jit_compilation_cache_miss() as cache_miss_count:
          f(x)
        self.assertGreater(cache_miss_count(), 0)

        # Check if FDO profile file of the biggest module is not empty
        fdo_profiles_after_pgle = [
            x
            for x in self.get_fdo_profiles(dump_dir)
            if x not in fdo_profiles_before_pgle
        ]
        self.assertNotEmpty(fdo_profiles_after_pgle)

        # Check if FDO profile file in dump directory is not empty
        for fdo_profile in fdo_profiles_after_pgle:
          self.assertGreater(
              os.path.getsize(os.path.join(dump_dir, fdo_profile)), 0
          )

        for pgle_profiler in pjit._pgle_profiler_dict.values():
          self.assertTrue(pgle_profiler.is_enabled())
          self.assertTrue(pgle_profiler.is_fdo_consumed())

        files_after_pgle_profile = os.listdir(cache_dir)
        self.assertGreater(
            len(files_after_pgle_profile), len(non_pgle_profiled_files)
        )

        # Removing non-pgle profiled module from cache to check that later pgle
        # profiled version will be used.
        for non_pgle_file in non_pgle_profiled_files:
          path = os.path.join(cache_dir, non_pgle_file)
          if os.path.isfile(path):
            os.remove(path)
          elif os.path.isdir(path):
            shutil.rmtree(path)

        api.clear_caches()
        pjit._pgle_profiler_dict.clear()

        # Run 4: Persistent compilation cache should be hit PGLE profiler should
        # be disabled
        cache_hit = 0
        def check_if_cache_hit(event):
          nonlocal cache_hit
          if event == '/jax/compilation_cache/cache_hits':
            cache_hit += 1

        monitoring.register_event_listener(check_if_cache_hit)
        f(x)
        monitoring._unregister_event_listener_by_callback(check_if_cache_hit)

        self.assertGreater(cache_hit, 0)

  def testPassingFDOProfile(self):
    mesh = jtu.create_mesh((2,), ('x',))

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
        compiler_options={'xla_gpu_enable_latency_hiding_scheduler': 'True'},
    )
    def f(x, y):
      return x @ y

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    y = x + 1

    with config.pgle_profiling_runs(0):
      f_lowered = f.lower(x, y)
      compiled = f_lowered.compile()

    with tempfile.TemporaryDirectory() as cache_dir:
      jax.profiler.start_trace(cache_dir)
      compiled(x, y)
      jax.profiler.stop_trace()
      directories = glob.glob(os.path.join(cache_dir, 'plugins/profile/**/'))
      directories = [d for d in directories if os.path.isdir(d)]
      rundir = directories[-1]
      logging.info('rundir: %s', rundir)
      fdo_profile = exp_profiler.get_profiled_instructions_proto(rundir)

    if jtu.test_device_matches(['gpu']) and jtu.is_device_cuda():
      self.assertIn(b'custom', fdo_profile)

    logging.info('fdo_profile: %s', fdo_profile)
    # Test pass fdo_profile as compiler_options API works.
    f_lowered.compile(compiler_options={'fdo_profile': fdo_profile})


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
