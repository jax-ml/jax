# Copyright 2021 Google LLC
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
import hashlib
import os
import random
import tempfile
import unittest
from unittest import SkipTest

from absl.testing import absltest
from jax.experimental import PartitionSpec as P
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental.maps import xmap
from jax.experimental.pjit import pjit
import jax
from jax import jit, lax, pmap
from jax._src.util import prod
import jax._src.test_util as jtu
import jax._src.lib
import numpy as np

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

class CompilationCacheTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jtu.device_under_test() != "tpu":
        raise SkipTest("serialize executable only works on TPU")
    if jax._src.lib.xla_bridge.get_backend().runtime_type == "tfrt":
        raise SkipTest("the new TFRT runtime does not support serialization")

  def tearDown(self):
      super().tearDown()
      cc._cache = None

  @unittest.skipIf(jax._src.lib.version < (0, 1, 68), "fails with earlier jaxlibs")
  def test_compile_options(self):
    compile_options_not_filled = jax._src.lib.xla_bridge.get_compile_options(
                      num_replicas=1, num_partitions=1)
    compile_options_filled = self.filled_compile_options()
    filled_hash1 = self.get_hashed_value(cc._hash_compile_options, compile_options_filled)
    filled_hash2 = self.get_hashed_value(cc._hash_compile_options, compile_options_filled)
    not_filled_hash3 = self.get_hashed_value(cc._hash_compile_options, compile_options_not_filled)
    self.assertEqual(filled_hash1, filled_hash2)
    self.assertNotEqual(filled_hash1, not_filled_hash3)

  def test_executable_build_options(self):
    compile_options_not_filled = jax._src.lib.xla_bridge.get_compile_options(
                      num_replicas=1, num_partitions=1)
    compile_options_filled = self.filled_compile_options()
    filled_hash1 = self.get_hashed_value(cc._hash_executable_build_options,
                   compile_options_filled.executable_build_options)
    filled_hash2 = self.get_hashed_value(cc._hash_executable_build_options,
                   compile_options_filled.executable_build_options)
    not_filled_hash3 = self.get_hashed_value(cc._hash_executable_build_options,
                       compile_options_not_filled.executable_build_options)
    self.assertEqual(filled_hash1, filled_hash2)
    self.assertNotEqual(filled_hash1, not_filled_hash3)

  def test_debug_options(self):
    compile_options = jax._src.lib.xla_bridge.get_compile_options(
                      num_replicas=1, num_partitions=1)
    hash1 = self.get_hashed_value(cc._hash_debug_options,
                   compile_options.executable_build_options.debug_options)
    hash2 = self.get_hashed_value(cc._hash_debug_options,
                   compile_options.executable_build_options.debug_options)
    self.assertEqual(hash1, hash2)
    new_debug_options = self.create_new_debug_options(compile_options.executable_build_options.debug_options)
    hash3 = self.get_hashed_value(cc._hash_debug_options, new_debug_options)
    self.assertNotEqual(hash1, hash3)

  def test_hash_platform(self):
    hash1 = self.get_hashed_value(cc._hash_platform, jax._src.lib.xla_bridge.get_backend())
    hash2 = self.get_hashed_value(cc._hash_platform, jax._src.lib.xla_bridge.get_backend())
    self.assertEqual(hash1, hash2)
    if jax._src.lib.xla_bridge.get_backend().platform != "cpu":
        cpu_backend = jax._src.lib.xla_bridge.get_backend("cpu")
        hash3 = self.get_hashed_value(cc._hash_platform, cpu_backend)
        self.assertNotEqual(hash1, hash3)

  def test_hash_int(self):
    hash1 = self.get_hashed_value(cc._hash_int, 90)
    hash2 = self.get_hashed_value(cc._hash_int, 8)
    hash3 = self.get_hashed_value(cc._hash_int, 8)
    self.assertEqual(hash2, hash3)
    self.assertNotEqual(hash1, hash2)

  def test_hash_bool(self):
    hash1 = self.get_hashed_value(cc._hash_bool, False)
    hash2 = self.get_hashed_value(cc._hash_bool, True)
    hash3 = self.get_hashed_value(cc._hash_bool, True)
    self.assertEqual(hash2, hash3)
    self.assertNotEqual(hash1, hash2)

  def test_hash_string(self):
    hash1 = self.get_hashed_value(cc._hash_string, "foo")
    hash2 = self.get_hashed_value(cc._hash_string, "bar")
    hash3 = self.get_hashed_value(cc._hash_string, "bar")
    self.assertEqual(hash2, hash3)
    self.assertNotEqual(hash1, hash2)

  def test_same_hash_key(self):
    computation = jax.xla_computation(lambda x, y: x + y)(1, 1)
    compile_options = jax._src.lib.xla_bridge.get_compile_options(
                       num_replicas=1, num_partitions=1)
    backend = jax._src.lib.xla_bridge.get_backend()
    self.assertEqual(cc.get_cache_key(computation, compile_options, backend),
                     cc.get_cache_key(computation, compile_options, backend))

  def test_different_hash_key(self):
    computation = jax.xla_computation(lambda x, y: x + y)(1, 1)
    compile_options_not_filled = jax._src.lib.xla_bridge.get_compile_options(
                       num_replicas=1, num_partitions=1)
    compile_options_filled = self.filled_compile_options()
    backend = jax._src.lib.xla_bridge.get_backend()
    self.assertNotEqual(cc.get_cache_key(computation, compile_options_not_filled, backend),
                        cc.get_cache_key(computation, compile_options_filled, backend))

  def test_different_computations(self):
    computation1 = jax.xla_computation(lambda x, y: x + y)(1, 1)
    computation2 = jax.xla_computation(lambda x, y: x * y)(2, 2)
    compile_options = jax._src.lib.xla_bridge.get_compile_options(
                       num_replicas=1, num_partitions=1)
    backend = jax._src.lib.xla_bridge.get_backend()
    self.assertNotEqual(cc.get_cache_key(computation1, compile_options, backend),
                        cc.get_cache_key(computation2, compile_options, backend))

  def test_get_no_executable(self):
      with tempfile.TemporaryDirectory() as tmpdir:
          cc.initialize_cache(tmpdir)
          computation = jax.xla_computation(lambda x, y: x + y)(1, 1)
          compile_options = jax._src.lib.xla_bridge.get_compile_options(
                               num_replicas=1, num_partitions=1)
          backend = jax._src.lib.xla_bridge.get_backend()
          self.assertEqual(cc.get_executable(computation, compile_options, backend), None)

  def test_diff_executables(self):
      with tempfile.TemporaryDirectory() as tmpdir:
          cc.initialize_cache(tmpdir)
          computation1 = jax.xla_computation(lambda x, y: x + y)(1, 1)
          computation2 = jax.xla_computation(lambda x, y: x * y)(2, 2)
          compile_options = jax._src.lib.xla_bridge.get_compile_options(
                                num_replicas=1, num_partitions=1)
          backend = jax._src.lib.xla_bridge.get_backend()
          executable1 = backend.compile(computation1, compile_options)
          executable2 = backend.compile(computation2, compile_options)
          cc.put_executable(computation1, compile_options, executable1, backend)
          cc.put_executable(computation2, compile_options, executable2, backend)
          self.assertNotEqual(cc.get_executable(computation1, compile_options, backend),
                              cc.get_executable(computation2, compile_options, backend))

  def test_put_executable(self):
      with tempfile.TemporaryDirectory() as tmpdir:
          cc.initialize_cache(tmpdir)
          computation = jax.xla_computation(lambda x, y: x + y)(1, 1)
          compile_options = jax._src.lib.xla_bridge.get_compile_options(
                               num_replicas=1, num_partitions=1)
          backend = jax._src.lib.xla_bridge.get_backend()
          executable = backend.compile(computation, compile_options)
          cc.put_executable(computation, compile_options, executable, backend)
          deserialized_executable = cc.get_executable(computation, compile_options, backend)
          inputs_to_executable = (np.array(1, dtype=np.int32), np.array(2, dtype=np.int32))
          expected = jax._src.lib.xla_client.execute_with_python_values(executable, inputs_to_executable, backend)
          actual = jax._src.lib.xla_client.execute_with_python_values(deserialized_executable, inputs_to_executable, backend)
          self.assertEqual(expected, actual)

  def test_pmap(self):
      with tempfile.TemporaryDirectory() as tmpdir:
          cc.initialize_cache(tmpdir)
          f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i')
          x = np.arange(jax.device_count(), dtype=np.int64)
          f(x)
          files_in_directory = len(os.listdir(tmpdir))
          self.assertEqual(files_in_directory, 1)
          x = np.arange(jax.device_count(), dtype=np.float32)
          f(x)
          files_in_directory = len(os.listdir(tmpdir))
          self.assertEqual(files_in_directory, 2)
          #TODO: create a test for calling pmap with the same input more than once

  def test_jit(self):
      with tempfile.TemporaryDirectory() as tmpdir:
          cc.initialize_cache(tmpdir)
          f = jit(lambda x: x*x)
          f(1)
          files_in_directory = len(os.listdir(tmpdir))
          self.assertEqual(files_in_directory, 1)
          f(1.0)
          files_in_directory = len(os.listdir(tmpdir))
          self.assertEqual(files_in_directory, 2)

  @jtu.with_mesh([('x', 2)])
  def test_pjit(self):
      with tempfile.TemporaryDirectory() as tmpdir:
          cc.initialize_cache(tmpdir)
          @partial(pjit,
                   in_axis_resources=(P('x'), P('x')),
                   out_axis_resources=None)
          def f(x, y):
              return x + y

          shape = (8, 8)
          x = np.arange(prod(shape), dtype=np.int64).reshape(shape)
          f(x, x + 1)
          files_in_directory = len(os.listdir(tmpdir))
          self.assertEqual(files_in_directory, 1)
          x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
          f(x, x + 1)
          files_in_directory = len(os.listdir(tmpdir))
          self.assertEqual(files_in_directory, 2)

  @jtu.with_mesh([('x', 2)])
  def test_xmap(self):
      with tempfile.TemporaryDirectory() as tmpdir:
          cc.initialize_cache(tmpdir)
          def f(x):
              return x * 2
          devices = np.array(jax.local_devices()[:2])
          if devices.size < 2:
              raise SkipTest("Test requires 2 devices")
          x = np.arange(8, dtype=np.int64).reshape((2, 2, 2))
          xmap(f, in_axes=['a', ...], out_axes=['a', ...],
               axis_resources={'a': 'x'})(x)
          files_in_directory = len(os.listdir(tmpdir))
          self.assertEqual(files_in_directory, 1)
          x = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
          xmap(f, in_axes=['a', ...], out_axes=['a', ...],
               axis_resources={'a': 'x'})(x)
          files_in_directory = len(os.listdir(tmpdir))
          self.assertEqual(files_in_directory, 2)

  def create_new_debug_options(self, debug_options_obj):
    debug_options_obj.xla_cpu_enable_fast_math = False
    debug_options_obj.xla_cpu_fast_math_honor_infs = False
    debug_options_obj.xla_cpu_fast_math_honor_nans = False
    debug_options_obj.xla_cpu_fast_math_honor_division = False
    debug_options_obj.xla_cpu_fast_math_honor_functions = False
    debug_options_obj.xla_gpu_enable_fast_min_max = False
    debug_options_obj.xla_backend_optimization_level = random.randint(0, 10)
    debug_options_obj.xla_cpu_enable_xprof_traceme = False
    debug_options_obj.xla_llvm_disable_expensive_passes = False
    debug_options_obj.xla_test_all_input_layouts = False
    return debug_options_obj

  def filled_compile_options(self):
    compile_options = jax._src.lib.xla_client.CompileOptions()
    compile_options.num_replicas = 1
    compile_options.num_partitions = 1
    shape = jax._src.lib.xla_client.Shape.array_shape(np.dtype(np.float32), [2])
    shape_array = [shape, shape]
    compile_options.argument_layouts = shape_array
    compile_options.executable_build_options.result_layout = shape

    device_assignment = jax._src.lib.xla_client.DeviceAssignment.create(np.ndarray(shape=(2,2)))
    compile_options.device_assignment = device_assignment
    compile_options.executable_build_options.device_assignment = device_assignment
    return compile_options

  def get_hashed_value(self, hash_function, hash_function_input):
    hash_obj = hashlib.sha256()
    hash_function(hash_obj, hash_function_input)
    return hash_obj.digest().hex()

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
