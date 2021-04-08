# Copyright 2020 Google LLC
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
about 2000+ test harnesses. See `primitive_harness.py` docstring for a
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

From the limitations objects, we generate a
[report](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/g3doc/primitives_with_limited_support.md).
The report has instructions for how to re-generate it.

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

import datetime
import os
from typing import Any, Dict, Tuple
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import dtypes
from jax import lax
from jax import numpy as jnp
from jax import test_util as jtu
from jax.config import config
from jax.experimental import jax2tf
from jax.interpreters import xla

import numpy as np
import tensorflow as tf  # type: ignore[import]

config.parse_flags_with_absl()

# Import after parsing flags
from jax.experimental.jax2tf.tests import tf_test_util
from jax.experimental.jax2tf.tests.jax2tf_limitations import Jax2TfLimitation
from jax.experimental.jax2tf.tests import primitive_harness

DType = Any

REDUCE = (
    jnp.all,
    jnp.any,
    jnp.max,
    jnp.min,
    jnp.prod,
    jnp.sum,
)


class JaxPrimitiveTest(tf_test_util.JaxToTfTestCase):

  # This test runs for all primitive harnesses. For each primitive "xxx" the
  # test will be called "test_prim_xxx_..." and the custom parameters for
  # the test are defined in the class method "jax2tf_limitations.Jax2TfLimitation.xxx".
  # See more details in the comment at top of file and in Jax2TfLimitation class.
  # If you want to run this test for only one harness, add parameter
  # `one_containing="foo"` to parameterized below.
  @primitive_harness.parameterized(
      primitive_harness.all_harnesses, include_jax_unimpl=False
      )
  @jtu.ignore_warning(
      category=UserWarning, message="Using reduced precision for gradient.*")
  def test_prim(self, harness: primitive_harness.Harness):
    limitations = Jax2TfLimitation.limitations_for_harness(harness)
    device = jtu.device_under_test()
    limitations = tuple(filter(lambda l: l.filter(device=device,
                                                  dtype=harness.dtype), limitations))
    func_jax = harness.dyn_fun
    args = harness.dyn_args_maker(self.rng())
    self.ConvertAndCompare(func_jax, *args, limitations=limitations)

  def test_primitive_coverage(self):
    """Fail if there are JAX primitives that are not implemented."""
    # Harvest primitives from XLA translation tables
    all_primitives = (
        set(xla.translations)
        | set(xla.backend_specific_translations["cpu"])
        | set(xla.backend_specific_translations["gpu"])
        | set(xla.backend_specific_translations["tpu"])
        | set(xla.initial_style_translations)
        | set(xla.parallel_translations))

    tf_impl = set(jax.experimental.jax2tf.jax2tf.tf_impl) | set(
        jax.experimental.jax2tf.jax2tf.tf_impl_with_avals)
    tf_not_yet_impl = set(jax.experimental.jax2tf.jax2tf.tf_not_yet_impl)

    all_primitives = tuple(sorted(all_primitives, key=str))
    for p in all_primitives:
      if p.name == "axis_index":
        continue
      if p.name in tf_not_yet_impl:
        self.assertNotIn(
            p, tf_impl)  # Should not be in both tf_impl and tf_not_yet_impl
      else:
        self.assertIn(p, tf_impl)

  def test_generate_limitations_doc(self):
    """Generates primitives_with_limited_support.md.

    See the doc for instructions.
    """

    harnesses = [
        h for h in primitive_harness.all_harnesses
        if h.filter(h, include_jax_unimpl=True)
    ]
    print(f"Found {len(harnesses)} test harnesses that work in JAX")

    def unique_hash(h: primitive_harness.Harness, l: Jax2TfLimitation):
      return (h.group_name, l.description, l.devices,
              tuple([np.dtype(d).name for d in l.dtypes]), l.modes)

    unique_limitations: Dict[Any, Tuple[primitive_harness.Harness, Jax2TfLimitation]] = {}
    for h in harnesses:
      for l in h.jax_unimplemented:
        if l.enabled:
          # Fake a Jax2TFLimitation from the Limitation
          tfl = Jax2TfLimitation(description="Not implemented in JAX: " + l.description,
                                 devices = l.devices,
                                 dtypes = l.dtypes,
                                 expect_tf_error = False,
                                 skip_tf_run = True)
          unique_limitations[hash(unique_hash(h, tfl))] = (h, tfl)
    for h in harnesses:
      for l in Jax2TfLimitation.limitations_for_harness(h):
        unique_limitations[hash(unique_hash(h, l))] = (h, l)

    print(f"Found {len(unique_limitations)} unique limitations")
    tf_error_table = [
        """
| Affected primitive | Description of limitation | Affected dtypes | Affected devices | Affected compilation modes |
| --- | --- | --- | --- | --- |"""
    ]
    tf_numerical_discrepancies_table = list(tf_error_table)  # a copy
    for h, l in sorted(
        unique_limitations.values(), key=lambda pair: unique_hash(*pair)):
      devices = ", ".join(sorted(l.devices))
      modes = ", ".join(sorted(l.modes))
      description = l.description
      if l.skip_comparison:
        description = "Numeric comparision disabled: " + description
      if l.expect_tf_error:
        description = "TF error: " + description
      if l.skip_tf_run:
        description = "TF test skipped: " + description

      if l.skip_tf_run or l.expect_tf_error:
        to_table = tf_error_table
      elif l.skip_comparison or l.custom_assert:
        to_table = tf_numerical_discrepancies_table
      else:
        continue

      to_table.append(
          f"| {h.group_name} | {description} | "
          f"{primitive_harness.dtypes_to_str(l.dtypes, empty_means_all=True)} | {devices} | {modes} |"
      )

    if not os.environ.get("JAX_OUTPUT_LIMITATIONS_DOC"):
      raise unittest.SkipTest(
          "Set JAX_OUTPUT_LIMITATIONS_DOC=1 to enable the generation of the documentation"
      )
    # The CPU has more supported types, and harnesses
    self.assertEqual("cpu", jtu.device_under_test())
    self.assertTrue(
        config.x64_enabled,
        "Documentation generation must be run with JAX_ENABLE_X64=1")

    with open(
        os.path.join(
            os.path.dirname(__file__),
            "../g3doc/primitives_with_limited_support.md.template")) as f:
      template = f.read()
    output_file = os.path.join(
        os.path.dirname(__file__),
        "../g3doc/primitives_with_limited_support.md")

    with open(output_file, "w") as f:
      f.write(template.replace("{{generation_date}}", str(datetime.date.today())) \
              .replace("{{tf_error_table}}", "\n".join(tf_error_table)) \
              .replace("{{tf_numerical_discrepancies_table}}", "\n".join(tf_numerical_discrepancies_table)) \
              )

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

  def test_integer_div(self):
    x = jnp.array([-4, -3, -1, 0, 1, 3, 6])
    y = np.int32(3)
    self.ConvertAndCompare(jnp.floor_divide, x, y)
    expected = jnp.floor_divide(x, y)
    # Try it with TF 1 as well (#5831)
    with tf.compat.v1.Session() as sess:
      tf1_res = sess.run(jax2tf.convert(jnp.floor_divide)(x, y))
      self.assertAllClose(expected, tf1_res)

  def test_disable_xla(self):

    def fun(x):
      return lax.pad(x, np.float32(0), [(-1, 0, 0), (0, 0, 0)])

    with self.assertRaisesRegex(
        NotImplementedError, "Call to pad can only be converted through "
                             "TFXLA, but XLA is disabled"):
      self.ConvertAndCompare(
          fun, np.ones((2, 3), dtype=np.float32), enable_xla=False)

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

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(testcase_name=f"_{f_jax.__name__}", f_jax=f_jax)
          for f_jax in REDUCE))
  def test_reduce_ops_with_numerical_input(self, f_jax):
    values = np.array([1, 2, 3], dtype=np.float32)
    self.ConvertAndCompare(f_jax, values)

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(testcase_name=f"_{op.__name__}", op=op) for op in (
              jax.ops.index_add,
              jax.ops.index_max,
              jax.ops.index_min,
              jax.ops.index_mul,
              jax.ops.index_update,
          )))
  def test_scatter_static(self, op):
    values = np.ones((5, 6), dtype=np.float32)
    update = np.float32(6.)
    f_jax = jax.jit(lambda v, u: op(v, jax.ops.index[::2, 3:], u))
    self.ConvertAndCompare(f_jax, values, update)

  @parameterized.named_parameters(
      jtu.cases_from_list(
          dict(testcase_name=f"_{f_jax.__name__}", f_jax=f_jax)
          for f_jax in REDUCE))
  def test_reduce_ops_with_boolean_input(self, f_jax):
    values = np.array([True, False, True], dtype=np.bool_)
    self.ConvertAndCompare(f_jax, values)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
