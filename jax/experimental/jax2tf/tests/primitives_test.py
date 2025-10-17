# Copyright 2020 The JAX Authors.
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
"""Tests for JAX primitive coverage.

The bulk of the testing is done by `test_prim`, which is parameterized by
about 3500+ test harnesses. See `test_harnesses.py` docstring for a
description of test harnesses. That module contains also the definitions
of all the test harnesses, and a specification of which are only partially
implemented for JAX.

For each harness, we convert the JAX function to Tensorflow and then we run
it on the same inputs in "eager", "graph", or "compiled" mode and we check
that we get the same result as in JAX
(see `tf_test_util.ConvertAndCompare`).

Some harnesses need specific tolerances, or even custom equality assertions.
Also, for some harnesses we need to specify some data types that result
in Tensorflow errors (for some devices and compilation modes). These limitations
are captured as jax2tf_limitations.Jax2TfLimitation objects.

If a harness run fails with error, and a limitation that matches the device
and data types is found,
the error is logged but does not abort the test. If a harness run succeeds
and there are matching limitations, the test emits a warning. If you want to
turn these warnings into errors, you'd have to uncomment an assertion
in `tf_test_util.ConvertAndCompare`.

IMPORTANT: If you need to customize the testing of a particular primitive
conversion, you must create a class method in jax2tf_limitations.jax2tf_limitations,
with the same name as the harness.group_name (typically the same as the
primitive name). That class method should return the list of Jax2TfLimitation
objects for the harness.
See `jax2tf_limitations.limitations_for_harness`. If a group name does
not need limitations, then it must be listed in the
`jax2tf_limitations.harness_groups_no_limitations`.

"""

from typing import Any
import unittest

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import dtypes
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu

import numpy as np

config.parse_flags_with_absl()

# Import after parsing flags
from jax.experimental.jax2tf.tests import tf_test_util
from jax.experimental.jax2tf.tests.jax2tf_limitations import Jax2TfLimitation
from jax._src.internal_test_util import test_harnesses

DType = Any

REDUCE = (
    jnp.all,
    jnp.any,
    jnp.max,
    jnp.min,
    jnp.prod,
    jnp.sum,
)


@jtu.thread_unsafe_test_class()
class JaxPrimitiveTest(tf_test_util.JaxToTfTestCase):

  # This test runs for all primitive harnesses. For each primitive "xxx" the
  # test will be called "test_prim_xxx_..." and the custom parameters for
  # the test are defined in the class method "jax2tf_limitations.Jax2TfLimitation.xxx".
  # See more details in the comment at top of file and in Jax2TfLimitation class.
  # If you want to run this test for only one harness, add parameter
  # `one_containing="foo"` to parameterized below.
  @test_harnesses.parameterized(
      test_harnesses.all_harnesses,
      include_jax_unimpl=False,
      #one_containing="",
  )
  @jtu.ignore_warning(
      category=UserWarning, message="Using reduced precision for gradient.*")
  def test_prim(self, harness: test_harnesses.Harness):
    limitations = Jax2TfLimitation.limitations_for_harness(harness)
    device = jtu.device_under_test()
    limitations = tuple(filter(lambda l: l.filter(device=device,
                                                  dtype=harness.dtype), limitations))
    func_jax = harness.dyn_fun
    args = harness.dyn_args_maker(self.rng())

    if ("eigh" == harness.group_name and
        np.complex64 == harness.dtype and
        device == "tpu"):
      raise unittest.SkipTest("b/264716764: error on tf.cast from c64 to f32")

    if "eigh" == harness.group_name and device == "cpu":
      raise unittest.SkipTest(
          "Equality comparisons on eigendecompositions are not stable.")

    if device == "gpu" and "lu" in harness.fullname:
      raise unittest.SkipTest("b/269388847: lu failures on GPU")

    def skipCustomCallTest(target: str):
      raise unittest.SkipTest(
          f"TODO(b/272239584): custom call target not guaranteed stable: {target}")
    if device == "gpu":
      if "custom_linear_solve_" in harness.fullname:
        skipCustomCallTest("cusolver_geqrf, cublas_geqrf_batched")
      if "svd_shape" in harness.fullname:
        skipCustomCallTest("cusolver_gesvdj")
      if "tridiagonal_solve_shape" in harness.fullname:
        skipCustomCallTest("cusparse_gtsv2_f32, cusparse_gtsv2_f64")

    associative_scan_reductions = harness.params.get("associative_scan_reductions", False)
    try:
      with jax.jax2tf_associative_scan_reductions(associative_scan_reductions):
        self.ConvertAndCompare(func_jax, *args, limitations=limitations)
    except Exception as e:
      # TODO(b/264596006): custom calls are not registered properly with TF in OSS
      if "does not work with custom calls" in str(e):
        logging.warning("Suppressing error %s", e)
        raise unittest.SkipTest("b/264596006: custom calls in native serialization fail in TF")
      else:
        raise e

  # The rest of the test are checking special cases

  @parameterized.named_parameters(
      dict(testcase_name=f"_{f_jax.__name__}",
           f_jax=f_jax)
      for f_jax in [jnp.add, jnp.subtract, jnp.multiply, jnp.divide,
                    jnp.less, jnp.less_equal, jnp.equal, jnp.greater,
                    jnp.greater_equal, jnp.not_equal, jnp.maximum,
                    jnp.minimum])
  def test_type_promotion(self, f_jax=jnp.add):
    # We only test a few types here, as tensorflow does not support many
    # types like uint* or bool in binary ops.
    types = [dtypes.bfloat16, np.int32, np.int64, np.float32]
    for x_dtype in types:
      for y_dtype in types:
        x = np.array([1, 2], dtype=x_dtype)
        y = np.array([3, 4], dtype=y_dtype)
        self.ConvertAndCompare(f_jax, x, y)

  def test_boolean_gather(self):
    values = np.array([[True, True], [False, True], [False, False]],
                      dtype=np.bool_)
    indices = np.array([0, 1], dtype=np.int32)
    for axis in [0, 1]:
      f_jax = jax.jit(lambda v, i: jnp.take(v, i, axis=axis))  # pylint: disable=cell-var-from-loop
      self.ConvertAndCompare(f_jax, values, indices)

  def test_gather_rank_change(self):
    params = jnp.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [3.0, 3.5, 4.0]])
    indices = jnp.array([[1, 1, 2], [0, 1, 0]])
    f_jax = jax.jit(lambda i: params[i])
    self.ConvertAndCompare(f_jax, indices)

  @jtu.sample_product(f_jax=REDUCE)
  def test_reduce_ops_with_numerical_input(self, f_jax):
    values = np.array([1, 2, 3], dtype=np.float32)
    self.ConvertAndCompare(f_jax, values)

  @jtu.sample_product(op=["add", "max", "min", "multiply", "set"])
  def test_scatter_static(self, op):
    values = np.ones((5, 6), dtype=np.float32)
    update = np.float32(6.)
    f_jax = jax.jit(lambda v, u: getattr(v.at[::2, 3:], op)(u))
    self.ConvertAndCompare(f_jax, values, update)

  @jtu.sample_product(f_jax=REDUCE)
  def test_reduce_ops_with_boolean_input(self, f_jax):
    values = np.array([True, False, True], dtype=np.bool_)
    self.ConvertAndCompare(f_jax, values)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
