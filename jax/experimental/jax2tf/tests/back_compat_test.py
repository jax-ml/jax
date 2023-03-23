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
"""Tests for backwards compatibility of custom calls.

Since we have to guarantee 6 months of backward compatibility for the
JAX serialized format, we need to guarantee that custom calls continue to
work as before. We test this here.

The tests in this file refer to the test data in ./back_compat_testdata.
There is one test for each version of a custom call target, e.g.,
`test_ducc_fft` tests the FFT custom calls on CPU.
Only custom call targets tested here should be listed in
jax_export._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE. All other custom
call targets will result in an error when encountered during serialization.

Once we stop using a custom call target in JAX, you can remove it from the
_CUSTOM_CALL_TARGETS_GUARANTEED_STABLE and you can add a comment to the
test here to remove it after 6 months.

** To create a new test **

Write the JAX function `func` that exercises the custom call `foo_call` you
want, then pick some inputs, and then add this to the new test to get started.

  def test_foo_call(self):
    def func(...): ...
    inputs = (...,)  # Tuple of nd.array, keep it small, perhaps generate the
                     # inputs in `func`.
    data = dataclasses.replace(dummy_data, inputs=inputs,
                               platform=default_jax_backend())
    self.run_one_test(func, data)

The test will fail, but will save to a file the test data you will need. The
file name will be printed in the logs. Create a new
file ./back_compat_testdata/cuda_foo_call.py and paste the test data that
you will see printed in the logs. You may want to
edit the serialization string to remove any pathnames that may be included at
the end, or gxxxxx3 at the beginning.

Name the literal `data_YYYYY_MM_DD` to include the date of serializaton
(for readability only). Then add here:

  from jax.experimental.jax2tf.tests.back_compat_testdata import foo_call
  def test_foo_call(self):
    def func(...): ...
    data = load_testdata(foo_call.data_YYYY_MM_DD)
    self.run_one_test(func, data)

"""
import dataclasses
import datetime
from functools import partial
import itertools
import os
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Sequence

# from absl import logging
from absl.testing import absltest, parameterized
from absl import logging

import numpy as np
# Import some NumPy symbols so that we can parse repr(ndarray).
from numpy import array, float32

import jax
from jax.config import config
from jax import lax
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests.back_compat_testdata import cpu_ducc_fft
from jax.experimental.jax2tf.tests.back_compat_testdata import cpu_lapack_geqrf
from jax.experimental.jax2tf.tests.back_compat_testdata import cpu_lapack_syev
from jax.experimental.jax2tf.tests.back_compat_testdata import cuda_cusolver_geqrf
from jax.experimental.jax2tf.tests.back_compat_testdata import cuda_cusolver_syev
from jax.experimental.jax2tf.tests.back_compat_testdata import cuda_threefry2x32
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Eigh
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Lu
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Qr
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Sharding

from jax.experimental import pjit
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax._src import core
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb

import tensorflow as tf  # type: ignore[import]

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla as tfxla  # type: ignore[import]
# pylint: enable=g-direct-tensorflow-import


config.parse_flags_with_absl()


def default_jax_backend() -> str:
  # Canonicalize to turn into "cuda" or "rocm"
  return xb.canonicalize_platform(jax.default_backend())

CURRENT_TESTDATA_VERSION = 1

@dataclasses.dataclass
class CompatTestData:
  testdata_version: int
  platform: str  # One of: "cpu", "tpu", "cuda", "rocm"
  custom_call_targets: List[str]
  serialized_date: datetime.date  # e.g., datetime.date(2023, 3, 9)
  inputs: Sequence[np.ndarray]
  expected_outputs: Sequence[np.ndarray]
  mlir_module_text: str
  mlir_module_serialized: bytes
  xla_call_module_version: int  # The version of XlaCallModule to use for testing


# The dummy_data is used for getting started for adding a new test and for
# testing the helper functions.

# Pasted from the test output (see module docstring)
dummy_data_dict = dict(
    testdata_version = CURRENT_TESTDATA_VERSION,
    platform="cpu",
    custom_call_targets=[],
    serialized_date=datetime.date(2023, 3, 15),
    inputs=(array(0.0, dtype=float32),),
    expected_outputs=(array(0.0, dtype=float32),),
    mlir_module_text="""
  module @jit_sin {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.sine %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01\x17\x05\x01\x05\x01\x03\x05\x03\x07\x07\t\x0b\x03K5\x07\x01\x1b\x07\x0b\x13\x0b3\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0b\x03\x1b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x03\x07\x0f\x17\x07\x02\xa7\x1f\x05\r\x03\x03\x03\x07\x05\x0f\x03\x0b\x0b\x1b\r'\x0f)\x031\x113\x05\x11\x05\x13\x05\x15\x05\x17\x1d\x15\x17\x05\x19\x17\x19\xef\x01\x05\x1b\x03\x03\x1d\r\x05\x1f!#%\x1d\x1d\x1d\x1f\x1d!\x1d##\x03\x03\x03+\r\x03-/\x1d%\x1d'\x1d)\x1d+)\x01\x05\x11\x03\x01\x03\x01\t\x04A\x05\x01\x11\x01\x05\x07\x03\x01\x05\x03\x11\x01\t\x05\x03\x05\x0b\x03\x01\x01\x05\x06\x13\x03\x01\x03\x01\x07\x04\x01\x03\x03\x06\x03\x01\x05\x01\x00\x9a\x04-\x0f\x0b\x03!\x1b\x1d\x05\x1b\x83/\x1f\x15\x1d\x15\x11\x13\x15\x11\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00sine_v1\x00return_v1\x00sym_name\x00jit_sin\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(sin)/jit(main)/sin\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00\x00main\x00public\x00",
    xla_call_module_version=4,
)  # End paste


def load_testdata(testdata_dict: Dict[str, Any]) -> CompatTestData:
  if testdata_dict["testdata_version"] == CURRENT_TESTDATA_VERSION:
    return CompatTestData(**testdata_dict)
  else:
    raise NotImplementedError("testdata_version not recognized: " +
                              testdata_dict["testdata_version"])


def load_testdata_nested(testdata_nest) -> Iterable[CompatTestData]:
  # Load all the CompatTestData in a Python nest.
  if isinstance(testdata_nest, dict) and "testdata_version" in testdata_nest:
    yield load_testdata(testdata_nest)
  elif isinstance(testdata_nest, dict):
    for e in testdata_nest.values():
      yield from load_testdata_nested(e)
  elif isinstance(testdata_nest, list):
    for e in testdata_nest:
      yield from load_testdata_nested(e)
  else:
    assert False, testdata_nest

dummy_data = load_testdata(dummy_data_dict)


class CompatTest(jtu.JaxTestCase):

  def run_one_test(self, func: Callable[..., jax.Array],
                   data: CompatTestData,
                   run_tf=None,
                   rtol=None):
    """Run one compatibility test.

    Args:
      func: the JAX function to serialize and run
      data: the test data
      run_tf: (optional) a function to invoke the XlaCallModule TF op. Takes
        a TensorFlow callable and the arguments.
      rtol: relative tolerance for numerical comparisons
    """
    if default_jax_backend() != data.platform:
      self.skipTest(f"Test enabled only for {data.platform}")

    # Check that it runs in JAX native
    res_from_jax = jax.jit(func)(*data.inputs)
    if not isinstance(res_from_jax, (list, tuple)):
      res_from_jax = (res_from_jax,)
    res_from_jax = tuple(np.array(a) for a in res_from_jax)

    # Use the native exporter, to make sure we get the proper serialized module.
    exported = jax2tf.jax_export.serialize_native(
        jax.jit(func),
        [core.ShapedArray(a.shape, a.dtype) for a in data.inputs],
        lowering_platform=default_jax_backend(),
        # Must turn off strict checks because the custom calls may be unallowed.
        strict_checks=False,
    )

    module_str = str(exported.mlir_module)
    custom_call_re = r"stablehlo.custom_call\s*@([^\(]+)\("
    custom_call_targets = sorted(
        list(set(re.findall(custom_call_re, module_str)))
    )
    np.set_printoptions(threshold=sys.maxsize, floatmode="unique")
    # Print the test data to simplify updating the test
    updated_testdata = f"""
# Pasted from the test output (see back_compat_test.py module docstring)
data_{datetime.date.today().strftime('%Y_%m_%d')} = dict(
    testdata_version={CURRENT_TESTDATA_VERSION},
    platform={repr(default_jax_backend())},
    custom_call_targets={repr(custom_call_targets)},
    serialized_date={repr(datetime.date.today())},
    inputs={repr(data.inputs)},
    expected_outputs={repr(res_from_jax)},
    mlir_module_text=\"\"\"\n{module_str}\"\"\",
    mlir_module_serialized={repr(exported.mlir_module_serialized)},
    xla_call_module_version={exported.xla_call_module_version},
)  # End paste
"""
    output_dir = os.getenv("TEST_UNDECLARED_OUTPUTS_DIR",
                           "/tmp/back_compat_testdata")
    output_file = os.path.join(output_dir, f"{self._testMethodName}.py")
    logging.info("Writing the up-to-date testdata at %s", output_file)
    with open(output_file, "w") as f:
      f.write(updated_testdata)

    if rtol is not None:
      rtol = 1.e-7
    self.assertAllClose(res_from_jax, data.expected_outputs, rtol=rtol)

    res_serialized = self.run_serialized(data, run_tf=run_tf)
    logging.info("Result of serialized run is %s", res_serialized)
    self.assertAllClose(res_serialized, data.expected_outputs)
    self.assertListEqual(custom_call_targets, data.custom_call_targets)

  def run_serialized(self, data: CompatTestData, run_tf=None):
    # Run the serialized module. For now, use XlaCallModule. This has the
    # disadvantage that it brings TF and jax2tf in the picture, but has the
    # advantage that it is simple (e.g., XlaCallModule already has the
    # machinery to deserialize and run), and also it is the way users actually
    # run serialized modules today.
    # TODO(necula): come up with a JAX-native way of running serialized modules.
    tf_preferred_devices = (
        tf.config.list_logical_devices("TPU")
        + tf.config.list_logical_devices("GPU")
        + tf.config.list_logical_devices()
    )
    # We need --config=cuda build flag for TF to see the GPUs
    self.assertEqual(
        jtu.device_under_test().upper(), tf_preferred_devices[0].device_type
    )

    def f_tf(*args_tf):
      return tfxla.call_module(
          args_tf,
          version=data.xla_call_module_version,
          Tout=[r.dtype for r in res_tf],
          Sout=[r.shape for r in res_tf],
          module=data.mlir_module_serialized,
          platforms=[data.platform.upper()])

    # We need this to run the TPU code on the TPU
    with tf.device(tf_preferred_devices[0]):
      args_tf = [tf.constant(a) for a in data.inputs]
      res_tf = [tf.constant(r) for r in data.expected_outputs]
      if run_tf is not None:
        res = run_tf(f_tf, *args_tf)
      else:
        res = f_tf(*args_tf)
      return tuple(r.numpy() for r in res)

  def test_dummy(self):
    # Tests the test mechanism. Let this test run on all platforms
    platform_dummy_data = dataclasses.replace(
        dummy_data, platform=default_jax_backend())
    self.run_one_test(jnp.sin, platform_dummy_data)

  def test_detect_different_output(self):
    # Test the detection mechanism. Let this test run on all platforms
    platform_dummy_data = dataclasses.replace(
        dummy_data,
        platform=default_jax_backend(),
        expected_outputs=(np.array(2.0, dtype=np.float32),))
    with self.assertRaisesRegex(AssertionError, "Not equal to tolerance"):
      self.run_one_test(jnp.sin, platform_dummy_data)

  def test_detect_different_custom_calls(self):
    # Test the detection mechanism. Let this test run on all platforms
    platform_dummy_data = dataclasses.replace(
        dummy_data,
        platform=default_jax_backend(),
        custom_call_targets=["missing"])
    with self.assertRaisesRegex(AssertionError, "Lists differ"):
      self.run_one_test(jnp.sin, platform_dummy_data)

  def test_custom_call_coverage(self):
    targets_to_cover = set(jax2tf.jax_export._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE)
    # Add here all the testdatas that should cover the targets guaranteed
    # stable
    covering_testdatas = [
        cpu_ducc_fft.data_2023_03_17, cpu_lapack_syev.data_2023_03_17,
        cpu_lapack_geqrf.data_2023_03_17, cuda_threefry2x32.data_2023_03_15,
        cuda_cusolver_geqrf.data_2023_03_18, cuda_cusolver_syev.data_2023_03_17,
        tpu_Eigh.data, tpu_Lu.data_2023_03_21, tpu_Qr.data_2023_03_17, tpu_Sharding.data_2023_03_16]
    covering_testdatas = itertools.chain(
        *[load_testdata_nested(d) for d in covering_testdatas])
    covered_targets = set()
    for data in covering_testdatas:
      self.assertIsInstance(data, CompatTestData)
      covered_targets = covered_targets.union(data.custom_call_targets)

    not_covered = targets_to_cover.difference(covered_targets)
    self.assertEmpty(not_covered)

  def test_ducc_fft(self):
    def func(x):
      return lax.fft(x, fft_type="fft", fft_lengths=(4,))

    data = load_testdata(cpu_ducc_fft.data_2023_03_17)
    self.run_one_test(func, data)

  @staticmethod
  def eigh_harness(shape, dtype):
    # In order to keep inputs small, we construct the input programmatically
    operand = jnp.reshape(jnp.arange(np.prod(shape), dtype=dtype), shape)
    # Make operand self-adjoint
    operand = (operand + jnp.conj(jnp.swapaxes(operand, -1, -2))) / 2.
    return lax.linalg.eigh(jnp.tril(operand), lower=True, symmetrize_input=False)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  def test_cpu_lapack_syevd(self, dtype_name="f32"):
    # For lax.linalg.eigh
    if not config.jax_enable_x64 and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    func = lambda: CompatTest.eigh_harness((8, 8), dtype)
    data = load_testdata(cpu_lapack_syev.data_2023_03_17[dtype_name])
    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)
    self.run_one_test(func, data, rtol=rtol)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}_{variant}",
           dtype_name=dtype_name, variant=variant)
      for dtype_name in ("f32", "f64")
      # We use different custom calls for sizes <= 32
      for variant in ["syevj", "syevd"])
  def test_gpu_cusolver_syev(self, dtype_name="f32", variant="syevj"):
    # For lax.linalg.eigh
    dtype = dict(f32=np.float32, f64=np.float64)[dtype_name]
    size = dict(syevj=8, syevd=36)[variant]
    rtol = dict(f32=1e-3, f64=1e-5)
    func = lambda: CompatTest.eigh_harness((size, size), dtype)
    data = load_testdata(cuda_cusolver_syev.data_2023_03_17[f"{dtype_name}_{variant}"])
    self.run_one_test(func, data, rtol=rtol)

  def test_tpu_Eigh(self):
    # For lax.linalg.eigh
    func = lambda: CompatTest.eigh_harness((8, 8), np.float32)
    data = load_testdata(tpu_Eigh.data)
    self.run_one_test(func, data, rtol=1e-3)

  @staticmethod
  def qr_harness(shape, dtype):
    # In order to keep inputs small, we construct the input programmatically
    operand = jnp.reshape(jnp.arange(np.prod(shape), dtype=dtype), shape)
    return lax.linalg.qr(operand, full_matrices=True)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  def test_cpu_lapack_geqrf(self, dtype_name="f32"):
    # For lax.linalg.qr
    if not config.jax_enable_x64 and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    func = lambda: CompatTest.qr_harness((3, 3), dtype)
    data = load_testdata(cpu_lapack_geqrf.data_2023_03_17[dtype_name])
    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)
    self.run_one_test(func, data, rtol=rtol)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}_{batched}",
           dtype_name=dtype_name, batched=batched)
      for dtype_name in ("f32",)
      # For batched qr we use cublas_geqrf_batched
      for batched in ("batched", "unbatched"))
  def test_gpu_cusolver_geqrf(self, dtype_name="f32", batched="unbatched"):
    # For lax.linalg.qr
    dtype = dict(f32=np.float32, f64=np.float64)[dtype_name]
    rtol = dict(f32=1e-3, f64=1e-5)
    shape = dict(batched=(2, 3, 3), unbatched=(3, 3))[batched]
    func = lambda: CompatTest.qr_harness(shape, dtype)
    data = load_testdata(cuda_cusolver_geqrf.data_2023_03_18[batched])
    self.run_one_test(func, data, rtol=rtol)

  def test_tpu_Qr(self):
    # For lax.linalg.qr
    func = lambda: CompatTest.qr_harness((3, 3), np.float32)
    data = load_testdata(tpu_Qr.data_2023_03_17)
    self.run_one_test(func, data, rtol=1e-3)

  @staticmethod
  def lu_harness(shape, dtype):
    operand = jnp.reshape(jnp.arange(np.prod(shape), dtype=dtype), shape)
    return lax.linalg.lu(operand)

  def test_tpu_Lu(self):
    # For lax.linalg.lu
    func = lambda: CompatTest.lu_harness((3, 3), np.float32)
    data = load_testdata(tpu_Lu.data_2023_03_21)
    self.run_one_test(func, data, rtol=1e-3)

  def test_cu_threefry2x32(self):
    def func(x):
      return jax.random.uniform(x, (2, 4), dtype=np.float32)

    data = load_testdata(cuda_threefry2x32.data_2023_03_15)
    self.run_one_test(func, data)

  def test_sharding(self):
    # Tests "Sharding", "SPMDShardToFullShape", "SPMDFullToShardShape" on TPU
    if jtu.device_under_test() != "tpu" or len(jax.devices()) < 2:
      self.skipTest("Test runs only on TPU with at least 2 devices")

    # Must use exactly 2 devices for expected outputs from ppermute
    devices = jax.devices()[:2]
    mesh = Mesh(devices, axis_names=('a'))

    @partial(pjit.pjit,
             in_shardings=(P('a', None),), out_shardings=P('a', None))
    @partial(shard_map, mesh=mesh,
             in_specs=(P('a', None),), out_specs=P('a', None))
    def func(x):  # b: f32[2, 4]
      axis_size = lax.psum(1, 'a')
      perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
      return lax.ppermute(x, 'a', perm=perm)

    # We need these only because for now we run the serialized module with TF
    tf_tpus = tf.config.list_logical_devices("TPU")
    self.assertNotEmpty(tf_tpus)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology, computation_shape=[1, 1, 1, 2],
        num_replicas=1)
    def run_tf(f_tf, x):
      def wrapped_f_tf(x):
        return tf.compat.v1.tpu.rewrite(
            f_tf, [tf.convert_to_tensor(x)],
            device_assignment=device_assignment)
      return tf.function(wrapped_f_tf, autograph=False, jit_compile=True)(x)

    data = load_testdata(tpu_Sharding.data_2023_03_16)
    with mesh:
      self.run_one_test(func, data, run_tf=run_tf)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
