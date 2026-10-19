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

import hashlib
import os
import re
import sys
import threading
import unittest
from typing import cast as type_cast

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax._src import cache_key
from jax._src import compiler
from jax._src import config
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.lib import _jax
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec as P
from jax._src.sharding_impls import NamedSharding
from jax._src.custom_partitioning import custom_partitioning


config.parse_flags_with_absl()


class _FakeDistributedClient:
  """The key-value store of a distributed runtime client, shared by the
  processes a test runs as threads, counting the calls it answers."""

  def __init__(self):
    self._values: dict[str, str] = {}
    self._changed = threading.Condition()
    self.sets = 0
    self.gets = 0

  def key_value_set(self, key: str, value: str,
                    allow_overwrite: bool = False) -> None:
    with self._changed:
      if key in self._values and not allow_overwrite:
        raise ValueError(f"{key} is already set")
      self._values[key] = value
      self.sets += 1
      self._changed.notify_all()

  def blocking_key_value_get(self, key: str, timeout_in_ms: int) -> str:
    with self._changed:
      if not self._changed.wait_for(lambda: key in self._values,
                                    timeout_in_ms / 1000):
        raise TimeoutError(key)
      self.gets += 1
      return self._values[key]


class CacheKeyTest(jtu.JaxTestCase):

  def test_serialized_compile_options(self):
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    hash1 = self.get_hashed_value(
        cache_key._hash_serialized_compile_options, compile_options
    )
    debug_options = compile_options.executable_build_options.debug_options
    debug_options.xla_force_host_platform_device_count = 2
    debug_options.xla_dump_to = "foo"
    debug_options.xla_dump_hlo_module_re = "bar"
    debug_options.xla_dump_hlo_pass_re = "baz"
    debug_options.xla_dump_hlo_as_text = True
    debug_options.xla_dump_hlo_as_proto = True
    debug_options.xla_dump_hlo_as_dot = True
    debug_options.xla_dump_hlo_as_url = True
    debug_options.xla_dump_hlo_as_html = True
    debug_options.xla_dump_fusion_visualization = True
    debug_options.xla_dump_hlo_snapshots = True
    debug_options.xla_dump_max_hlo_modules = True
    debug_options.xla_dump_module_metadata = True
    debug_options.xla_dump_compress_protos = True
    debug_options.xla_dump_hlo_as_long_text = True
    debug_options.xla_dump_disable_metadata = True
    debug_options.xla_dump_hlo_pipeline_re = "xyzzy"
    debug_options.xla_gpu_experimental_autotune_cache_mode = 2
    hash2 = self.get_hashed_value(
        cache_key._hash_serialized_compile_options, compile_options
    )
    self.assertEqual(hash1, hash2)

  @jtu.skip_on_devices("cpu")
  def test_hash_accelerator_devices(self):
    devices = np.array([[jax.local_devices()[0]]])

    dev_hash1 = self.get_hashed_value(cache_key._hash_devices, devices)
    dev_hash2 = self.get_hashed_value(cache_key._hash_devices, devices)
    self.assertEqual(dev_hash1, dev_hash2)

    acc_hash1 = self.get_hashed_value(
        cache_key._hash_accelerator_config, devices)
    acc_hash2 = self.get_hashed_value(
        cache_key._hash_accelerator_config, devices)
    self.assertEqual(acc_hash1, acc_hash2)

  def test_processes_of_a_computation_read_one_set_of_fingerprints(self):
    # Two processes whose topologies fingerprint apart, as GPUs with and
    # without NVLink do on one host, each publish their own fingerprint and
    # accelerators and read both, so they decide on one key for a
    # computation they share.
    client = _FakeDistributedClient()
    read = {}

    def process(process_id, fingerprint):
      fingerprints = cache_key._TopologyFingerprints(
          client, process_id, timeout_ms=60_000)
      read[process_id] = fingerprints.of(
          [0, 1], (fingerprint, "gpu cuda 12090: RTX 3090 8.6 82"))

    threads = [threading.Thread(target=process, args=arguments)
               for arguments in ((0, 7), (1, 11))]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    pool = {0: (7, "gpu cuda 12090: RTX 3090 8.6 82"),
            1: (11, "gpu cuda 12090: RTX 3090 8.6 82")}
    self.assertEqual(read, {0: pool, 1: pool})

  def test_a_process_publishes_once_and_reads_each_peer_once(self):
    # The exchange costs a run one publish and one read per peer, however
    # many computations it compiles and whichever processes they span.
    client = _FakeDistributedClient()
    for process_id in (1, 2, 3):
      client.key_value_set(cache_key._fingerprint_key(process_id),
                           f"{process_id} cpu")
    fingerprints = cache_key._TopologyFingerprints(client, 0, timeout_ms=1)
    for process_ids in ([0, 1], [0, 1, 2, 3], [0, 2], [0, 1, 2, 3]):
      fingerprints.of(process_ids, (0, "cpu"))
    self.assertEqual((client.sets, client.gets), (3 + 1, 3))

  def test_processes_linked_differently_share_one_key(self):
    # The same accelerators, fingerprinted apart by their links: every
    # process hashes the same sorted set, whichever process it is.
    self.assertEqual(
        cache_key._shared_fingerprints({0: (11, "gpu a"), 1: (7, "gpu a")}),
        [7, 11])
    self.assertEqual(
        cache_key._shared_fingerprints({1: (7, "gpu a"), 0: (11, "gpu a")}),
        [7, 11])

  def test_processes_that_fingerprint_alike_keep_their_key(self):
    # One fingerprint in the set: the key hashes its eight bytes, as a
    # single process's does, so a pool of like processes (a TPU slice, a
    # cluster of like GPU hosts) keeps the key it had.
    self.assertEqual(
        cache_key._shared_fingerprints({0: (7, "gpu a"), 1: (7, "gpu a"),
                                        2: (7, "gpu a")}),
        [7])

  def test_processes_with_different_accelerators_share_no_key(self):
    # Different GPU architectures, or runtimes, must not share an
    # executable: the computation gets no key, which callers of the key
    # take as compiling without the cache, and the error names each group.
    with self.assertRaisesRegex(
        _jax.JaxRuntimeError,
        r"processes \[0, 2\]: gpu cuda 12090: RTX 3090 8.6 82; "
        r"processes \[1\]: gpu cuda 12090: RTX 4080 8.9 76"):
      cache_key._shared_fingerprints({
          0: (7, "gpu cuda 12090: RTX 3090 8.6 82"),
          1: (9, "gpu cuda 12090: RTX 4080 8.9 76"),
          2: (8, "gpu cuda 12090: RTX 3090 8.6 82")})

  def test_a_process_names_its_platform_and_devices(self):
    backend = xla_bridge.get_backend()
    accelerators = cache_key._accelerators(backend)
    self.assertStartsWith(accelerators, backend.platform)
    self.assertIn(jax.local_devices()[0].device_kind, accelerators)
    self.assertNotIn("\n", accelerators)
    if jtu.test_device_matches(["cuda"]):
      self.assertIn(f" {jax.local_devices()[0].compute_capability} ",
                    accelerators)
      self.assertIn(" cudnn ", accelerators)

  def test_hash_platform(self):
    hash1 = self.get_hashed_value(
        cache_key._hash_platform, xla_bridge.get_backend()
    )
    hash2 = self.get_hashed_value(
        cache_key._hash_platform, xla_bridge.get_backend()
    )
    self.assertEqual(hash1, hash2)
    if xla_bridge.get_backend().platform != "cpu":
      cpu_backend = xla_bridge.get_backend("cpu")
      hash3 = self.get_hashed_value(cache_key._hash_platform, cpu_backend)
      self.assertNotEqual(hash1, hash3)

  def test_hash_string(self):
    hash1 = self.get_hashed_value(cache_key._hash_string, "foo")
    hash2 = self.get_hashed_value(cache_key._hash_string, "bar")
    hash3 = self.get_hashed_value(cache_key._hash_string, "bar")
    self.assertEqual(hash2, hash3)
    self.assertNotEqual(hash1, hash2)

  def test_same_key(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    self.assertEqual(
        cache_key.get(computation, devices, compile_options, backend),
        cache_key.get(computation, devices, compile_options, backend),
    )

  def test_different_key(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options_not_filled = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    compile_options_filled = self.filled_compile_options()
    backend = xla_bridge.get_backend()
    self.assertNotEqual(
        cache_key.get(
            computation, devices, compile_options_not_filled, backend
        ),
        cache_key.get(computation, devices, compile_options_filled, backend),
    )

  @jtu.thread_unsafe_test()
  def test_custom_hook(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    original_custom_hook = cache_key.custom_hook
    cache_key.custom_hook = lambda: "hook1"
    key1 = cache_key.get(computation, devices, compile_options, backend)
    cache_key.custom_hook = lambda: "hook2"
    key2 = cache_key.get(computation, devices, compile_options, backend)
    cache_key.custom_hook = original_custom_hook
    self.assertNotEqual(key1, key2)

  def test_different_computations(self):
    computation1 = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    computation2 = jax.jit(lambda x, y: x * y).lower(2, 2).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    self.assertNotEqual(
        cache_key.get(computation1, devices, compile_options, backend),
        cache_key.get(computation2, devices, compile_options, backend),
    )

  # TODO(phawkins): this test flakes if test concurrency is enabled.
  @jtu.thread_unsafe_test()
  @jtu.ignore_warning(category=DeprecationWarning,
                      message='`with mesh:` context manager')
  def test_custom_partitioning_ptr_removal(self):
    def _partition(mesh, arg_shapes, result_shape):
      arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
      result_shardings = NamedSharding(mesh, arg_shapes[0].sharding.spec)
      return mesh, jax.numpy.add, result_shardings, arg_shardings

    def _infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      return NamedSharding(mesh, arg_shapes[0].sharding.spec)

    @custom_partitioning
    def _cp_add(x, y):
      return jax.numpy.add(x, y)

    _cp_add.def_partition(
      infer_sharding_from_operands=_infer_sharding_from_operands,
      partition=_partition,
      sharding_rule='..., ... -> ...')

    devices = np.asarray(jax.devices())
    with Mesh(devices, ('x',)) as m:
      computation = jax.jit(
        _cp_add,
        in_shardings=(NamedSharding(m, P('x')),
                      NamedSharding(m, P('x'))),
                      out_shardings=NamedSharding(m, P('x'))
      ).lower(
        jax.ShapeDtypeStruct([1024], dtype=jax.numpy.float32),
        jax.ShapeDtypeStruct([1024], dtype=jax.numpy.float32),
      ).compiler_ir()
      pattern = (
          r'stablehlo\.custom_call @CustomSPMDPartitioning\('
          r'(.*?)\) \{'
          r'(.*?backend_config\s*=\s*"([^"]*)".*?)'
          r'\}'
      )
      with computation.context:
        updated_module = cache_key._remove_custom_partitioning_callbacks(
            type_cast(ir.Module, computation.operation.clone()),
        )
        bcs = [
            match[2]
            for match in re.findall(pattern, str(updated_module), re.DOTALL)
        ]
        for bc in bcs:
          self.assertEqual(bc, "REMOVED")

      compile_options = compiler.get_compile_options(
          num_replicas=1, num_partitions=1
      )
      backend = xla_bridge.get_backend()
      hash_without_callback_ptrs = cache_key.get(
          computation,
          devices,
          compile_options,
          backend,
          ignore_custom_partitioning=True,
      )
      expected_hash = cache_key.get(
          updated_module, devices, compile_options, backend
      )
      self.assertEqual(expected_hash, hash_without_callback_ptrs)

  def test_different_device_assignment(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options_1 = compiler.get_compile_options(
        num_replicas=1, num_partitions=1, device_assignment=np.array([[0]])
    )
    compile_options_2 = compiler.get_compile_options(
        num_replicas=1, num_partitions=1, device_assignment=np.array([[1]])
    )
    backend = xla_bridge.get_backend()
    hash_1 = cache_key.get(computation, devices, compile_options_1, backend)
    hash_2 = cache_key.get(computation, devices, compile_options_2, backend)
    if backend.platform == "gpu":
      self.assertEqual(hash_1, hash_2)
    else:
      self.assertNotEqual(hash_1, hash_2)

  @parameterized.parameters([False, True])
  @jtu.thread_unsafe_test()  # env vars are not thread-safe
  def test_identical_computations_different_metadata(self, include_metadata):
    f = lambda x, y: lax.mul(lax.add(x, y), 2)
    g = lambda x, y: lax.mul(lax.add(x, y), 2)
    assert id(f) != id(g)
    computation1 = jax.jit(f).lower(1, 1).compiler_ir()
    computation2 = jax.jit(g).lower(2, 3).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    with config.compilation_cache_include_metadata_in_key(include_metadata):
      key1 = cache_key.get(computation1, devices, compile_options, backend)
      key2 = cache_key.get(computation2, devices, compile_options, backend)
    self.assertEqual(include_metadata, key1 != key2)

  @parameterized.parameters([False, True])
  def test_identical_pallas_kernels_different_metadata(self, include_metadata):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Pallas TPU lowering requires TPU backend")

    # Same kernel body at different source lines; same __name__ to avoid kernel_name diff
    def make_caller_a():
      def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0
      return jax.jit(lambda x: pl.pallas_call(
          kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x))

    def make_caller_b():
      def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0
      return jax.jit(lambda x: pl.pallas_call(
          kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x))

    x = np.ones((128,), dtype=np.float32)
    computation1 = make_caller_a().lower(x).compiler_ir()
    computation2 = make_caller_b().lower(x).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    with config.compilation_cache_include_metadata_in_key(include_metadata):
      key1 = cache_key.get(computation1, devices, compile_options, backend)
      key2 = cache_key.get(computation2, devices, compile_options, backend)
    self.assertEqual(include_metadata, key1 != key2)

  @jtu.thread_unsafe_test()  # env vars are not thread-safe
  def test_xla_flags(self):
    if jtu.is_device_tpu(version=4):
      raise unittest.SkipTest("TODO(b/240151176)")

    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()

    orig_xla_flags = os.getenv("XLA_FLAGS")
    orig_argv = sys.argv
    try:
      os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
      key1 = cache_key.get(computation, devices, compile_options, backend)
      os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=1"
      key2 = cache_key.get(computation, devices, compile_options, backend)
      self.assertNotEqual(key1, key2)

      os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
      key3 = cache_key.get(computation, devices, compile_options, backend)
      self.assertEqual(key1, key3)

      # Test flag in _xla_flags_to_exclude_from_cache_key
      os.environ["XLA_FLAGS"] = (
          "--xla_gpu_autotune_level=0 --xla_force_host_platform_device_count=8"
      )
      key4 = cache_key.get(computation, devices, compile_options, backend)
      self.assertEqual(key1, key4)

      # Test flags given on command line
      del os.environ["XLA_FLAGS"]
      sys.argv.append("--xla_gpu_autotune_level=0")
      key5 = cache_key.get(computation, devices, compile_options, backend)
      self.assertEqual(key1, key5)
      sys.argv.append("--xla_force_host_platform_device_count=8")
      self.assertEqual(key1, key5)

    finally:
      if orig_xla_flags is not None:
        os.environ["XLA_FLAGS"] = orig_xla_flags
      elif os.getenv("XLA_FLAGS") is not None:
        del os.environ["XLA_FLAGS"]
      sys.argv = orig_argv

  @jtu.thread_unsafe_test()  # env vars are not thread-safe
  def test_libtpu_init_args(self):
    if jtu.is_device_tpu(version=4):
      raise unittest.SkipTest("TODO(b/240151176)")

    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()

    orig_libtpu_init_args = os.getenv("LIBTPU_INIT_ARGS")
    orig_argv = sys.argv
    try:
      os.environ["LIBTPU_INIT_ARGS"] = (
          "--xla_spmd_threshold_for_windowed_einsum_mib=0"
      )
      key1 = cache_key.get(computation, devices, compile_options, backend)
      os.environ["LIBTPU_INIT_ARGS"] = (
          "--xla_spmd_threshold_for_windowed_einsum_mib=1"
      )
      key2 = cache_key.get(computation, devices, compile_options, backend)
      self.assertNotEqual(key1, key2)

    finally:
      if orig_libtpu_init_args is not None:
        os.environ["LIBTPU_INIT_ARGS"] = orig_libtpu_init_args
      elif os.getenv("LIBTPU_INIT_ARGS") is not None:
        del os.environ["LIBTPU_INIT_ARGS"]
      sys.argv = orig_argv

  def filled_compile_options(self):
    compile_options = xla_client.CompileOptions()
    compile_options.num_replicas = 1
    compile_options.num_partitions = 1
    shape = xla_client.Shape.array_shape(np.dtype(np.float32), [2])
    shape_array = [shape, shape]
    compile_options.argument_layouts = shape_array
    compile_options.executable_build_options.result_layout = shape

    device_assignment = xla_client.DeviceAssignment.create(
        np.arange(4).reshape(2, 2)
    )
    compile_options.device_assignment = device_assignment
    compile_options.executable_build_options.device_assignment = (
        device_assignment
    )
    compile_options.executable_build_options.fdo_profile = b"test_profile"
    return compile_options

  def get_hashed_value(
      self, hash_function, hash_function_input1, hash_function_input2=None):
    hash_obj = hashlib.sha256()
    if hash_function_input2 is not None:
      hash_function(hash_obj, hash_function_input1, hash_function_input2)
    else:
      hash_function(hash_obj, hash_function_input1)
    return hash_obj.digest().hex()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
