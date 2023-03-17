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
jax2tf._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE. All other custom
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

The test will fail, but will print the test data you will need. Create a new
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
import re
from typing import Callable, List, Sequence
import sys

# from absl import logging
from absl.testing import absltest
from absl import logging

import numpy as np
# Import some NumPy symbols so that we can parse repr(ndarray).
from numpy import array, float32

import jax
from jax.config import config
from jax import lax
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests.back_compat_testdata import cuda_threefry2x32
from jax.experimental.jax2tf.tests.back_compat_testdata import cpu_ducc_fft

import jax.numpy as jnp

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

def load_testdata(testdata_dict):
  if testdata_dict["testdata_version"] == CURRENT_TESTDATA_VERSION:
    return CompatTestData(**testdata_dict)
  else:
    raise NotImplementedError("testdata_version not recognized: " +
                              testdata_dict["testdata_version"])

dummy_data = load_testdata(dummy_data_dict)


class CompatTest(jtu.JaxTestCase):

  def run_one_test(self, func: Callable[..., jax.Array], data: CompatTestData):
    if default_jax_backend() != data.platform:
      self.skipTest(f"Test enabled only for {data.platform}")

    # Check that it runs in JAX native
    res_from_jax = jax.jit(func)(*data.inputs)
    if not isinstance(res_from_jax, (list, tuple)):
      res_from_jax = (res_from_jax,)
    res_from_jax = tuple(np.array(a) for a in res_from_jax)

    # Use the native exporter, to make sure we get the proper serialized module.
    exported = jax2tf.jax2tf.serialize_native(
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
    updated_testdata = f"""Computed test data for this test (paste this into the test):
# Pasted from the test output (see back_compat_test.py module docstring)
data = dict(
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
    logging.info("%s", updated_testdata)
    self.assertAllClose(res_from_jax, data.expected_outputs)

    res_from_serialized = self.run_serialized(data)
    self.assertAllClose(res_from_serialized, data.expected_outputs)
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

    # We need this to run the TPU code on the TPU
    with tf.device(tf_preferred_devices[0]):
      args_tf = [tf.constant(a) for a in data.inputs]
      res_tf = [tf.constant(r) for r in data.expected_outputs]
      res = tfxla.call_module(
          args_tf,
          version=data.xla_call_module_version,
          Tout=[r.dtype for r in res_tf],
          Sout=[r.shape for r in res_tf],
          module=data.mlir_module_serialized,
          platforms=[data.platform.upper()],
      )
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

  def test_ducc_fft(self):
    def func(x):
      return lax.fft(x, fft_type="fft", fft_lengths=(4,))

    data = load_testdata(cpu_ducc_fft.data_2023_03_17)
    self.run_one_test(func, data)

  def test_cu_threefry2x32(self):
    def func(x):
      return jax.random.uniform(x, (2, 4), dtype=np.float32)

    data = load_testdata(cuda_threefry2x32.data_2023_03_15)
    self.run_one_test(func, data)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
