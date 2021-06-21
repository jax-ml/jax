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

from absl.testing import absltest
import hashlib
from jax.experimental.compilation_cache import compilation_cache as cc
import jax
import jax.test_util as jtu
import numpy as np
import random
import unittest

class CompilationCacheTest(jtu.JaxTestCase):

  @unittest.skipIf(jax.lib.version < (0, 1, 68), "fails with earlier jaxlibs")
  def test_compile_options(self):
    compile_options_not_filled = jax.lib.xla_bridge.get_compile_options(
                      num_replicas=1, num_partitions=1)
    compile_options_filled = self.filled_compile_options()
    filled_hash1 = self.get_hashed_value(cc._hash_compile_options, compile_options_filled)
    filled_hash2 = self.get_hashed_value(cc._hash_compile_options, compile_options_filled)
    not_filled_hash3 = self.get_hashed_value(cc._hash_compile_options, compile_options_not_filled)
    self.assertEqual(filled_hash1, filled_hash2)
    self.assertNotEqual(filled_hash1, not_filled_hash3)

  @unittest.skipIf(jax.lib.version < (0, 1, 68), "fails with earlier jaxlibs")
  def test_executable_build_options(self):
    compile_options_not_filled = jax.lib.xla_bridge.get_compile_options(
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
    compile_options = jax.lib.xla_bridge.get_compile_options(
                      num_replicas=1, num_partitions=1)
    hash1 = self.get_hashed_value(cc._hash_debug_options,
                   compile_options.executable_build_options.debug_options)
    hash2 = self.get_hashed_value(cc._hash_debug_options,
                   compile_options.executable_build_options.debug_options)
    self.assertEqual(hash1, hash2)
    new_debug_options = self.create_new_debug_options(compile_options.executable_build_options.debug_options)
    hash3 = self.get_hashed_value(cc._hash_debug_options, new_debug_options)
    self.assertNotEqual(hash1, hash3)

  @unittest.skipIf(jax.lib.version < (0, 1, 68), "fails with earlier jaxlibs")
  def test_hash_platform(self):
    hash1 = self.get_hashed_value(cc._hash_platform, jax.lib.xla_bridge.get_backend())
    hash2 = self.get_hashed_value(cc._hash_platform, jax.lib.xla_bridge.get_backend())
    self.assertEqual(hash1, hash2)
    if jax.lib.xla_bridge.get_backend().platform != "cpu":
        cpu_backend = jax.lib.xla_bridge.get_backend("cpu")
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

  @unittest.skipIf(jax.lib.version < (0, 1, 68), "fails with earlier jaxlibs")
  def test_same_hash_key(self):
    computation = jax.xla_computation(lambda x, y: x + y)(1, 1)
    compile_options = jax.lib.xla_bridge.get_compile_options(
                       num_replicas=1, num_partitions=1)
    self.assertEqual(cc.get_cache_key(computation, compile_options),
                     cc.get_cache_key(computation, compile_options))

  @unittest.skipIf(jax.lib.version < (0, 1, 68), "fails with earlier jaxlibs")
  def test_different_hash_key(self):
    computation = jax.xla_computation(lambda x, y: x + y)(1, 1)
    compile_options_not_filled = jax.lib.xla_bridge.get_compile_options(
                       num_replicas=1, num_partitions=1)
    compile_options_filled = self.filled_compile_options()
    self.assertNotEqual(cc.get_cache_key(computation, compile_options_not_filled),
                        cc.get_cache_key(computation, compile_options_filled))

  @unittest.skipIf(jax.lib.version < (0, 1, 68), "fails with earlier jaxlibs")
  def test_different_computations(self):
    computation1 = jax.xla_computation(lambda x, y: x + y)(1, 1)
    computation2 = jax.xla_computation(lambda x, y: x * y)(2, 2)
    compile_options = jax.lib.xla_bridge.get_compile_options(
                       num_replicas=1, num_partitions=1)
    self.assertNotEqual(cc.get_cache_key(computation1, compile_options),
                        cc.get_cache_key(computation2, compile_options))

  def create_new_debug_options(self, debug_options_obj):
    debug_options_obj.xla_cpu_enable_fast_math = False
    debug_options_obj.xla_cpu_fast_math_honor_infs = False
    debug_options_obj.xla_cpu_fast_math_honor_nans = False
    debug_options_obj.xla_cpu_fast_math_honor_division = False
    debug_options_obj.xla_cpu_fast_math_honor_functions = False
    debug_options_obj.xla_gpu_enable_fast_min_max = False
    debug_options_obj.xla_backend_optimization_level = random.randint(1, 10)
    debug_options_obj.xla_cpu_enable_xprof_traceme = False
    debug_options_obj.xla_llvm_disable_expensive_passes = False
    debug_options_obj.xla_test_all_input_layouts = False
    return debug_options_obj

  def filled_compile_options(self):
    compile_options = jax.lib.xla_client.CompileOptions()
    compile_options.num_replicas = 1
    compile_options.num_partitions = 1
    shape = jax.lib.xla_client.Shape.array_shape(np.dtype(np.float32), [2])
    shape_array = [shape, shape]
    compile_options.argument_layouts = shape_array
    compile_options.executable_build_options.result_layout = shape

    device_assignment = jax.lib.xla_client.DeviceAssignment.create(np.ndarray(shape=(2,2)))
    compile_options.device_assignment = device_assignment
    compile_options.executable_build_options.device_assignment = device_assignment
    return compile_options

  def get_hashed_value(self, hash_function, hash_function_input):
    hash_obj = hashlib.sha256()
    hash_function(hash_obj, hash_function_input)
    return hash_obj.digest().hex()

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
