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
in Tensorflow errors (for some decvices and compilation modes). These limitations
are captured as Jax2TfLimitation objects.

From the limitations objects, we generate a [report](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/g3doc/primitives_with_limited_support.md).
The report has instructions for how to re-generate it.

Together with the tolerances, the limitations are captured in
Jax2TfHarnessTrait objects, one per harness.

If a harness run fails with error, and a limitation that matches the device
and data types is found,
the error is logged but does not abort the test. If a harness run succeeds
and there are matching limitations, the test emits a warning. If you want to
turn these warnings into errors, you'd have to uncomment an assertion
in `tf_test_util.ConvertAndCompare`.

IMPORTANT: If you need to customize the testing of a particular primitive
conversion, you must create a class method in Jax2TfHarnessTrait,
with the same name as the harness.group_name (typically the same as the
primitive names). That class method should return the Jax2TfHarnessTrait
for the harness. See `Jax2TfHarnessTrait.create`. If a group name does
not need a custom Jax2TfHarnessTrait, then it must be listed in the
`Jax2TfHarnessTrait.default_harness_trait`.

"""

import datetime
import itertools
import os
from typing import Any, Callable, Dict, Optional, Sequence
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
from jax.experimental.jax2tf.tests import tf_test_util
from jax.interpreters import xla

import numpy as np

config.parse_flags_with_absl()
FLAGS = config.FLAGS

# Import after parsing flags
from jax.experimental.jax2tf.tests import primitive_harness

REDUCE = (
  jnp.all,
  jnp.any,
  jnp.max,
  jnp.min,
  jnp.prod,
  jnp.sum,
)


class JaxPrimitiveTest(tf_test_util.JaxToTfTestCase):

  # See comment at top of file
  @primitive_harness.parameterized(primitive_harness.all_harnesses,
                                   include_jax_unimpl=False)
  @jtu.ignore_warning(category=UserWarning,
                      message="Using reduced precision for gradient.*")
  def test_prim(self, harness: primitive_harness.Harness):
    trait = Jax2TfHarnessTrait.create(harness)
    func_jax = harness.dyn_fun
    args = harness.dyn_args_maker(self.rng())
    self.ConvertAndCompare(
      func_jax, *args,
      limitations=lambda dut, mode: trait.get_limitations(dut, mode),
      custom_assert=trait.custom_assert,
      always_custom_assert=trait.always_custom_assert,
      atol=trait.atol,
      rtol=trait.rtol)

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
      # TODO: remove tie_in once omnistaging is on by default
      if p.name == "axis_index" or p.name == "tie_in":
        continue
      if p in tf_not_yet_impl:
        self.assertNotIn(
            p, tf_impl)  # Should not be in both tf_impl and tf_not_yet_impl
      else:
        self.assertIn(p, tf_impl)

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

  def test_disable_xla(self):
    def fun(x):
      return lax.pad(x, np.float32(0), [(-1, 0, 0), (0, 0, 0)])

    with self.assertRaisesRegex(NotImplementedError,
                                "Call to pad can only be converted through "
                                "TFXLA, but XLA is disabled"):
      self.ConvertAndCompare(fun, np.ones((2, 3), dtype=np.float32),
                             enable_xla=False)

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

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in REDUCE))
  def test_reduce_ops_with_numerical_input(self, f_jax):
    values = np.array([1, 2, 3], dtype=np.float32)
    self.ConvertAndCompare(f_jax, values)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{op.__name__}",
         op=op)
    for op in (
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

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in REDUCE))
  def test_reduce_ops_with_boolean_input(self, f_jax):
    values = np.array([True, False, True], dtype=np.bool_)
    self.ConvertAndCompare(f_jax, values)

  # test_bfloat16_constant checks that https://github.com/google/jax/issues/3942 is
  # fixed
  # TODO(bchetioui): re-enable this test once recursion issues are addressed.
  @unittest.skipIf(True, "Infinite recursion after changes in #5085")
  def test_bfloat16_constant(self):
    def jax_fn_scalar(x):
      x = x.astype(jnp.bfloat16)
      x *= 2.
      return x

    def jax_fn_array(x):
      x = x.astype(jnp.bfloat16)
      x *= np.array([1.5, 2.5, 3.5], jnp.bfloat16)
      return x

    tf_fn_scalar = jax2tf.convert(jax_fn_scalar)
    self.assertAllClose(tf_fn_scalar(1.375).numpy(), jnp.bfloat16(2.750))

    tf_fn_array = jax2tf.convert(jax_fn_array)
    self.assertAllClose(tf_fn_array(np.array([3, 4, 5])),
                        np.array([4.5, 10, 17.5], jnp.bfloat16))


  def test_generate_limitations_doc(self):
    """Generates limited_support.md. See the doc for instructions."""

    harnesses = [h for h in primitive_harness.all_harnesses
                 if h.filter(h, include_jax_unimpl=False)]
    print(f"Found {len(harnesses)} test harnesses that work in JAX")

    def unique_hash(l: Jax2TfLimitation):
      return hash((l.harness.group_name, l.description, l.for_devices,
                   l.for_dtypes, l.for_modes))
    unique_limitations : Dict[Any, Jax2TfLimitation] = {}
    for h in harnesses:
      for l in Jax2TfHarnessTrait.create(h).limitations:
        # We do not filter the limitations, we want them all
        unique_limitations[unique_hash(l)] = l

    print(f"Found {len(unique_limitations)} unique limitations")
    limited_support_table = ["""
| Affected primitive | Description of limitation | Affected dtypes | Affected devices | Affected compilation modes |
| --- | --- | --- | --- | --- | ---|"""]
    for l in sorted(unique_limitations.values(), key=lambda l: str(l.harness.group_name)):
      devices = ", ".join(sorted(l.for_devices))
      modes = ", ".join(sorted(l.for_modes))
      if l.disable_comparison:
        descr = f"Numeric comparison disabled: {l.description}"
      else:
        descr = f"TF error: {l.description}"
      limited_support_table.append(f"|{l.harness.group_name}|{descr}|{primitive_harness.dtypes_to_str(l.for_dtypes)}|{devices}|{modes}|")

    if not os.environ.get("JAX_OUTPUT_LIMITATIONS_DOC"):
      raise unittest.SkipTest("Set JAX_OUTPUT_LIMITATIONS_DOC=1 to enable the generation of the documentation")
    # The CPU has more supported types, and harnesses
    self.assertEqual("cpu", jtu.device_under_test())
    self.assertTrue(FLAGS.jax_enable_x64, "Documentation generation must be run with JAX_ENABLE_X64=1")

    with open(os.path.join(os.path.dirname(__file__),
                           '../g3doc/primitives_with_limited_support.md.template')) as f:
      template = f.read()
    output_file = os.path.join(os.path.dirname(__file__),
                               '../g3doc/primitives_with_limited_support.md')

    with open(output_file, "w") as f:
      f.write(template.replace("{{generation_date}}", str(datetime.date.today())) \
              .replace("{{limited-support-table}}", "\n".join(limited_support_table)))


class Jax2TfLimitation(primitive_harness.Limitation):
  """Specific primitive limitations for jax2tf.

  See the module docstring for details.
  """

  def __init__(self,
               harness: primitive_harness.Harness,
               description: str,
               *,
               for_devices: Sequence[str] = ("cpu", "gpu", "tpu"),
               for_dtypes=(),
               enabled: bool = True,
               # jax2tf specific
               for_modes=("eager", "graph", "compiled"),
               # If given the it does not specify a TF exception, but
               # disables the comparison
               disable_comparison=False):
    assert isinstance(harness, primitive_harness.Harness)
    super().__init__(description,
                     for_devices=for_devices,
                     for_dtypes=for_dtypes,
                     harness=harness,
                     enabled=enabled)
    if isinstance(for_modes, str):
      for_modes = (for_modes,)
    assert all(m in ["eager", "graph", "compiled"] for m in for_modes)
    if for_devices == ("tpu",):
      assert for_modes == ("compiled",), f"{for_modes}"
    self.for_modes = for_modes
    self.disable_comparison = disable_comparison


class Jax2TfHarnessTrait:
  """Specific harnesses for jax2tf.

  See comments for tf_test_util.ConvertAndCompare
  """

  def __init__(self, limitations: Sequence[Jax2TfLimitation] = (),
               custom_assert: Optional[Callable] = None,
               always_custom_assert: bool = False,
               atol=None,
               rtol=None):
    self.limitations = limitations
    self.custom_assert = custom_assert
    self.always_custom_assert = always_custom_assert
    self.atol = atol
    self.rtol = rtol

  def get_limitations(self,
                      device_under_test: str,
                      mode: str) -> Sequence[Jax2TfLimitation]:
    return [l for l in self.limitations
            if (mode in l.for_modes and l.filter(device_under_test))]

  @classmethod
  def missing_tf_kernel(cls, harness: primitive_harness.Harness,
                        *, for_dtypes,
                        for_modes=("eager", "graph"),
                        also_compiled=False,
                        for_devices=("cpu", "gpu", "tpu")) -> Jax2TfLimitation:
    if isinstance(for_modes, str):
      for_modes = (for_modes,)
    if also_compiled and "compiled" not in for_modes:
      for_modes = for_modes + ("compiled",)

    return Jax2TfLimitation(
      harness, "op not defined for dtype",
      for_dtypes=for_dtypes,
      for_devices=for_devices,
      for_modes=for_modes)

  @classmethod
  def create(cls, harness: primitive_harness.Harness) -> 'Jax2TfHarnessTrait':
    group_method = getattr(cls, harness.group_name, None)
    if harness.group_name in cls.default_trait_groups:
      assert group_method is None, (
        f"Harness group {harness.group_name} is both in 'default_trait_groups' and "
        "has a custom Jax2TfHarnessTrain.classmethod defined (see module docstring)"
      )
      return Jax2TfHarnessTrait()
    else:
      # TODO: Remove, here only for backwards compatibility
      if group_method is None:
        group_method = getattr(cls, f"{harness.group_name}", None)
      assert group_method is not None, (
        f"Harness group {harness.group_name} must be either part of "
        f"'default_trait_groups' or must have a custom Jax2TfHarnessTrain.classmethod defined (see module docstring)")
    return group_method(harness)

  # We keep here the explicit set of groups for which we use default trait
  default_trait_groups = {
    "abs", "and", "argmin", "argmax",
    "broadcast", "broadcast_in_dim",
    "ceil", "concatenate",
    "cos", "complex", "conj",
    "device_put", "dynamic_slice", "dynamic_update_slice",
    "exp", "eq", "floor", "log",
    "gather",
     "imag", "iota", "is_finite",
    "ne", "not", "or",
    "pad",
    "random_split",
    "reduce_and", "reduce_prod", "reduce_or", "reduce_sum",
    "real",
    "reshape",
    "select",
    "shift_left", "shift_right_logical", "shift_right_arithmetic",
    "sin", "slice",
    "sqrt", "squeeze", "stop_gradient", "tie_in",
    "transpose", "xor",
    "zeros_like"
  }

  @classmethod
  def helper_get_trig_custom_assert(cls, harness, np_inverse, tol):
    # E.g., np_inverse=np.cosh for lax.acosh_p):
    custom_assert = None
    if harness.dtype in [np.complex64, np.complex128]:
      def custom_assert(tst, result_jax, result_tf, *, args):
        operand, = args
        tst.assertAllClose(operand, np_inverse(result_tf), atol=tol,
                           rtol=tol)
    return custom_assert

  @classmethod
  def acos(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    tol = None
    if dtype == np.complex128:
      tol = 1e-13
    elif dtype == np.complex64:
      tol = (1e-3 if jtu.device_under_test() == "tpu"
             else 1e-4)
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.float16, dtypes.bfloat16, np.complex64],
                            for_devices=("cpu", "gpu")),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex128],
                            for_devices=("cpu", "gpu")),
    ]
    return Jax2TfHarnessTrait(
      atol=tol, rtol=tol,
      custom_assert=cls.helper_get_trig_custom_assert(harness, np.cos, tol),
      limitations=limitations)

  @classmethod
  def acosh(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    tol = None
    if jtu.device_under_test() in ["cpu", "gpu"]:
      tol = 1e-3 if dtype == np.complex64 else (1e-12 if dtype == np.complex128
                                                else tol)
    limitations = [
      Jax2TfLimitation(harness,
                       "TODO: investigate large numeric difference",
                       for_devices=("cpu,"),
                       for_dtypes=np.complex128),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[dtypes.bfloat16, np.float16],
                            for_devices=("cpu", "gpu")),
    ]
    return Jax2TfHarnessTrait(
      atol=tol, rtol=tol,
      custom_assert=cls.helper_get_trig_custom_assert(harness, np.cosh, tol),
      limitations=limitations)


  @classmethod
  def add(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(limitations=[
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint16, np.uint32],
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64],
                            for_devices=("cpu", "gpu"),
                            also_compiled=True),
    ])

  @classmethod
  # Also called add_jaxvals
  def add_any(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(limitations=[
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint16, np.uint32, np.uint64],
                            also_compiled=True)])

  @classmethod
  def asin(cls, harness: primitive_harness.Harness):
    tol = None
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.float16, dtypes.bfloat16],
                            for_devices=("cpu", "gpu")),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex64, np.complex128],
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(
      atol=tol, rtol=tol,
      custom_assert=cls.helper_get_trig_custom_assert(harness, np.sin, tol),
      limitations=limitations)

  @classmethod
  def asinh(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    tol = None
    if jtu.device_under_test() in ["cpu", "gpu"]:
      tol = 1e-12 if dtype == np.complex128 else (1e-3 if dtype == np.complex64
                                                  else tol)
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.float16, dtypes.bfloat16],
                            for_devices=("cpu", "gpu")),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex64, np.complex128],
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(
      atol=tol, rtol=tol,
      custom_assert=cls.helper_get_trig_custom_assert(harness, np.sinh, tol),
      limitations=limitations)

  @classmethod
  def atan(cls, harness: primitive_harness.Harness):
    tol = None
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.float16, dtypes.bfloat16],
                            for_devices=("cpu", "gpu")),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex64, np.complex128],
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(
      atol=tol, rtol=tol,
      custom_assert=cls.helper_get_trig_custom_assert(harness, np.tan, tol),
      limitations=limitations)

  @classmethod
  def atanh(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    tol = None
    if dtype == np.float64:
      tol = 1e-14
    elif (dtype == np.complex128 and
          jtu.device_under_test() in ["cpu", "gpu"]):
      tol = 1e-12
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.float16, dtypes.bfloat16],
                            for_devices=("cpu", "gpu"))
    ]
    return Jax2TfHarnessTrait(
      atol=tol, rtol=tol,
      custom_assert=cls.helper_get_trig_custom_assert(harness, np.tanh, tol),
      limitations=limitations)

  @classmethod
  def atan2(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(limitations=[
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.float16, dtypes.bfloat16],
                            for_devices=("cpu", "gpu"))])

  @classmethod
  def bessel_i0e(cls, harness: primitive_harness.Harness,
                 primitive=lax.bessel_i0e_p):
    return Jax2TfHarnessTrait(limitations=[
      cls.missing_tf_kernel(harness, for_dtypes=[dtypes.bfloat16],
                            for_devices=("cpu", "gpu"))
    ])

  @classmethod
  def bessel_i1e(cls, harness: primitive_harness.Harness):
    return cls.bessel_i0e(harness, lax.bessel_i1e_p)

  @classmethod
  def bitcast_convert_type(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(limitations=[
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.bool_],
                            also_compiled=True)])

  @classmethod
  def cholesky(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype

    tol = 5e-2
    # TODO(bchetioui): very high discrepancy in the float32/complex64 case
    if dtype in [np.float32, np.complex64]:
      tol = 1e-2
    # TODO(bchetioui): also high discrepancy in the float64/complex128 case
    elif dtype in [np.float64, np.complex128]:
      tol = 1e-6

    def custom_assert(tst, result_jax, result_tf, **_):
      # cholesky_p returns garbage in the strictly upper triangular part of the
      # result, so we can safely ignore that part.
      tst.assertAllClose(jnp.tril(result_jax), result_tf, atol=tol)

    limitations = [
      # See https://github.com/google/jax/pull/3775#issuecomment-659407824;
      Jax2TfLimitation(harness,
                       "function not compilable",
                       for_dtypes=[np.complex64, np.complex128],
                       for_devices=("cpu", "gpu"),
                       for_modes="compiled"),
      cls.missing_tf_kernel(harness,
                            # Interesting: on TPU, complex64 works in eager
                            # mode, but fails otherwise.
                            for_dtypes=[np.complex64, np.complex128],
                            for_devices="tpu",
                            for_modes="compiled")
    ]
    return Jax2TfHarnessTrait(custom_assert=custom_assert,
                              always_custom_assert=True,
                              limitations=limitations)

  @classmethod
  def clamp(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(limitations=[
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.int8, np.uint16, np.uint32, np.uint64],
                            also_compiled=True)])

  @classmethod
  def convert_element_type(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait()

  @classmethod
  def conv_general_dilated(self, harness: primitive_harness.Harness):
    dtype, device = harness.dtype, jtu.device_under_test()
    # if device == "gpu" and dtype in [np.complex64, np.complex128]:
    #   raise unittest.SkipTest("TODO: crash on GPU in TF")

    tol = None
    if device == "gpu":
      tol = 1e-4
    elif device == "tpu":
      tol = 1e-3
    # TODO(bchetioui): significant discrepancies in some float16 cases.
    if dtype == np.float16:
      tol = 1.
    # TODO(bchetioui): slight occasional discrepancy in float32 cases.
    elif dtype == np.float32:
      tol = 0.5 if device == "tpu" else (1e-3 if device == "gpu" else 1e-4)
    elif dtype == np.complex64 and device == "tpu":
      tol = 0.1
    # TODO(bchetioui): slight discrepancy when going through the path using
    # tf.nn.convolution.
    elif dtype == np.float64 and device == "cpu":
      tol = 1e-13

    # TODO(bchetioui): unidentified bug in compiled mode. The test that fails is
    #
    # test_conv_general_dilated_tf_conversion_path_3d_lhs=float32[1,4,28,28,1]_rhs=float32[2,3,3,1,16]_windowstrides=(1,1,1)_padding=VALID_lhsdilation=(1,1,1)_rhsdilation=(1,1,2)_dimensionnumbers=('NDHWC','DHWIO','NDHWC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=False
    #
    # with the following assertion error in TensorFlowTrace.process_primitive:
    #
    # AssertionError: conv_general_dilated: out.aval = ShapedArray(float32[1,3,24,26,16]); expected ShapedArray(float32[1,3,26,24,16])
    #
    # Deactivating this assertion is enough to pass the test, which suggests
    # that the end shape is indeed the correct one (i.e. (1,3,26,24,16)).
    # Further investigation is required to really understand this behavior,
    # which we have not managed to reproduce as a pure TF test.
    #
    # This bug is low priority since it only occurs when using a non-TFXLA
    # conversion path in compiled mode, i.e. in a context where using the
    # TFXLA path is possible.
    # if harness.name == "_tf_conversion_path_3d_lhs=float32[1,4,28,28,1]_rhs=float32[2,3,3,1,16]_windowstrides=(1,1,1)_padding=VALID_lhsdilation=(1,1,1)_rhsdilation=(1,1,2)_dimensionnumbers=('NDHWC','DHWIO','NDHWC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=False":
    #  raise unittest.SkipTest("TODO: known but unidentified bug in compiled "
    #                          "mode")

    limitations = []
    if harness.params["batch_group_count"] > 1:
      limitations.append(Jax2TfLimitation(
        harness,
        "jax2tf BUG: batch_group_count > 1 not yet converted"))
    limitations.append(Jax2TfLimitation(
      harness,
      "XLA bug in the HLO -> LLVM IR lowering",
      for_dtypes=[np.complex64, np.complex128],
      for_devices=("cpu", "gpu"),
      for_modes=("eager", "graph", "compiled")))

    return Jax2TfHarnessTrait(atol=tol, rtol=tol, limitations=limitations)

  @classmethod
  def cosh(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(limitations=[
      cls.missing_tf_kernel(harness, for_dtypes=[np.float16],
                            for_devices=("cpu", "gpu")),
    ])

  @classmethod
  def cummax(cls, harness):
    dtype = harness.dtype
    tol = 0.1 if dtype == np.float16 else (0.5 if dtype == dtypes.bfloat16
                                           else None)
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64, np.complex128],
                            for_devices=("cpu", "gpu"),
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint16, np.uint32, np.int8, np.complex64],
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(atol=tol, rtol=tol, limitations=limitations)

  @classmethod
  def cummin(cls, harness):
    dtype = harness.dtype
    tol = 0.1 if dtype == np.float16 else (0.5 if dtype == dtypes.bfloat16
                                           else None)
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64, np.complex128],
                            for_devices=("cpu", "gpu"),
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint16, np.uint32, np.int8, np.complex64],
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(atol=tol, rtol=tol, limitations=limitations)

  @classmethod
  def cumprod(cls, harness):
    dtype = harness.dtype
    tol = 0.1 if dtype == np.float16 else (0.5 if dtype == dtypes.bfloat16
                                           else None)
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64],
                            for_devices=("cpu", "gpu"),
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint32],
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(atol=tol, rtol=tol, limitations=limitations)

  @classmethod
  def cumsum(cls, harness):
    dtype = harness.dtype
    tol = 0.1 if dtype == np.float16 else (0.5 if dtype == dtypes.bfloat16
                                           else None)
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64],
                            for_devices=("cpu", "gpu"),
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex64],
                            for_devices="tpu",
                            for_modes="compiled"),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint16, np.uint32],
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(atol=tol, rtol=tol, limitations=limitations)


  @classmethod
  def custom_linear_solve(cls, harness: primitive_harness.Harness):
    tol = 1e-3
    dtype = harness.dtype
    if (dtype == np.float32 and
        jtu.device_under_test() == "tpu"):
      tol = 0.01
    limitations = [
      Jax2TfLimitation(harness,
                       "TODO: large numerical discrepancy",
                       for_dtypes=np.float32,
                       for_devices="tpu",
                       for_modes="compiled",
                       disable_comparison=True)
    ]
    return Jax2TfHarnessTrait(atol=tol, rtol=tol, limitations=limitations)

  @classmethod
  def digamma(cls, harness: primitive_harness.Harness):
    custom_assert = None
    dtype = harness.dtype
    tol = None

    if dtype == np.float64:
      tol = 1e-13

    if dtype == np.float32 and jtu.device_under_test() in ["cpu", "gpu"]:
      tol = 1e-3

    # In the bfloat16 case, TF and lax both return NaN in undefined cases.
    if not dtype == dtypes.bfloat16:
      # digamma is not defined at 0 and -1
      def custom_assert(tst, result_jax, result_tf, *, args):
        # lax.digamma returns NaN and tf.math.digamma returns inf
        arg, = args
        special_cases = (arg == 0.) | (arg == -1.)
        nr_special_cases = np.count_nonzero(special_cases)
        tst.assertAllClose(np.full((nr_special_cases,), dtype(np.nan)),
                           result_jax[special_cases])
        tst.assertAllClose(np.full((nr_special_cases,), dtype(np.inf)),
                           result_tf[special_cases])
        # non-special cases are equal
        tst.assertAllClose(result_jax[~ special_cases],
                           result_tf[~ special_cases], atol=tol,
                           rtol=tol)
    return Jax2TfHarnessTrait(
      custom_assert=custom_assert, atol=tol, rtol=tol,
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[dtypes.bfloat16],
                              for_devices=("cpu", "gpu")),
      ])

  @classmethod
  def div(cls, harness: primitive_harness.Harness):
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint8, np.uint16, np.uint32, np.uint64,
                                        np.int8, np.int16],
                            also_compiled=True),
      Jax2TfLimitation(
        harness, "TF integer division fails if divisor contains 0; JAX returns NaN",
        for_dtypes=[np.uint8, np.int8, np.uint16, np.uint32, np.uint64,
                    np.int8, np.int16, np.int32, np.int64],
        # Only the harnesses with "singularity" will have divide by 0
        enabled=("singularity" in harness.name))
    ]
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def dot_general(cls, harness: primitive_harness.Harness):
    tol, dtype = None, harness.dtype
    if dtype == dtypes.bfloat16:
      tol = 0.3
    elif dtype in [np.complex64, np.float32]:
      if jtu.device_under_test() == "tpu":
        tol = 0.1 if dtype == np.float32 else 0.3
      else:
        tol = 1e-5
    elif dtype == np.float16:
      if jtu.device_under_test() == "gpu":
        tol = 0.1
      else:
        tol = 0.01

    # np_dtype = _to_np_dtype(args[0].dtype)
    # if np_dtype in [np.bool, np.uint8, np.uint16, np.uint32, np.uint64,
    #                 np.int8]:
    #   tf_unimpl(np_dtype)
    # elif np_dtype == np.int16:
    #   # TODO(bchetioui): the path using 'einsum' is not compatible with int16
    #   # arguments on CPU/GPU, while the one using 'matmul' is (but not in
    #   # compiled mode).
    #   tf_unimpl(np_dtype, additional_msg=("only cases representable as 2D "
    #                                       "matrix multiplication can be "
    #                                       "converted properly"),
    #             devs=['CPU', 'GPU'])
    #   tf_unimpl(np_dtype, devs=['TPU'])
    # elif np_dtype in [np.int16, np.int64]:
    #   devs = ['CPU'] if np_dtype == np.int16 else ['CPU', 'GPU']
    #   tf_unimpl(np_dtype, additional_msg=("this is a problem only in compiled "
    #                                       "mode (experimental_compile=True))"),
    #             devs=devs)
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.bool_, np.uint8, np.uint16, np.uint32, np.uint64,
                                        np.int8, np.int16],
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.int64],
                            for_devices=("cpu", "gpu"),
                            for_modes="compiled"),
    ]
    return Jax2TfHarnessTrait(atol=tol, rtol=tol, limitations=limitations)

  @classmethod
  def eig(self, harness: primitive_harness.Harness):
    compute_left_eigenvectors = harness.params["compute_left_eigenvectors"]
    compute_right_eigenvectors = harness.params["compute_right_eigenvectors"]
    dtype = harness.dtype

    def custom_assert(tst, result_jax, result_tf, *, args):
      operand, = args
      inner_dimension = operand.shape[-1]

      # Test ported from tests.test.testEig
      # Norm, adjusted for dimension and type.
      def norm(x):
        norm = np.linalg.norm(x, axis=(-2, -1))
        return norm / ((inner_dimension + 1) * jnp.finfo(dtype).eps)

      def check_right_eigenvectors(a, w, vr):
        tst.assertTrue(
          np.all(norm(np.matmul(a, vr) - w[..., None, :] * vr) < 100))

      def check_left_eigenvectors(a, w, vl):
        rank = len(a.shape)
        aH = jnp.conj(a.transpose(list(range(rank - 2)) + [rank - 1, rank - 2]))
        wC = jnp.conj(w)
        check_right_eigenvectors(aH, wC, vl)

      def check_eigenvalue_is_in_array(eigenvalue, eigenvalues_array):
        tol = None
        # TODO(bchetioui): numerical discrepancies
        if dtype in [np.float32, np.complex64]:
          tol = 1e-4
        elif dtype in [np.float64, np.complex128]:
          tol = 1e-13
        closest_diff = min(abs(eigenvalues_array - eigenvalue))
        tst.assertAllClose(closest_diff, np.array(0., closest_diff.dtype),
                           atol=tol)

      all_w_jax, all_w_tf = result_jax[0], result_tf[0]
      for idx in itertools.product(*map(range, operand.shape[:-2])):
        w_jax, w_tf = all_w_jax[idx], all_w_tf[idx]
        for i in range(inner_dimension):
          check_eigenvalue_is_in_array(w_jax[i], w_tf)
          check_eigenvalue_is_in_array(w_tf[i], w_jax)

      if compute_left_eigenvectors:
        check_left_eigenvectors(operand, all_w_tf, result_tf[1])
      if compute_right_eigenvectors:
        check_right_eigenvectors(operand, all_w_tf,
                                 result_tf[1 + compute_left_eigenvectors])

    limitations = [
      Jax2TfLimitation(harness,
                       "function not compilable",
                       for_modes="compiled")
    ]
    if compute_left_eigenvectors and compute_right_eigenvectors:
      limitations.append(Jax2TfLimitation(
        harness,
        "TF Conversion of eig is not implemented when both compute_left_eigenvectors and compute_right_eigenvectors are set to True"
      ))

    return Jax2TfHarnessTrait(custom_assert=custom_assert,
                              limitations=limitations)

  @classmethod
  def eigh(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    shape = harness.params["shape"]

    def custom_assert(tst, result_jax, result_tf, *, args):
      operand, = args
      inner_dimension = operand.shape[-1]

      def check_right_eigenvectors(a, w, vr):
        tol = 1e-16
        # TODO(bchetioui): tolerance needs to be very high in compiled mode,
        # specifically for eigenvectors.
        if dtype == np.float64:
          tol = 1e-6
        elif dtype == np.float32:
          tol = 1e-2
        elif dtype in [dtypes.bfloat16, np.complex64]:
          tol = 1e-3
        elif dtype == np.complex128:
          tol = 1e-13
        tst.assertAllClose(np.matmul(a, vr) - w[..., None, :] * vr,
                           np.zeros(a.shape, dtype=vr.dtype),
                           atol=tol)

      def check_eigenvalue_is_in_array(eigenvalue, eigenvalues_array):
        tol = None
        if dtype in [dtypes.bfloat16, np.float32, np.complex64]:
          tol = 1e-3
        elif dtype in [np.float64, np.complex128]:
          tol = 1e-11
        closest_diff = min(abs(eigenvalues_array - eigenvalue))
        tst.assertAllClose(closest_diff, np.array(0., closest_diff.dtype),
                           atol=tol)

      _, all_w_jax = result_jax
      all_vr_tf, all_w_tf = result_tf

      for idx in itertools.product(*map(range, operand.shape[:-2])):
        w_jax, w_tf = all_w_jax[idx], all_w_tf[idx]
        for i in range(inner_dimension):
          check_eigenvalue_is_in_array(w_jax[i], w_tf)
          check_eigenvalue_is_in_array(w_tf[i], w_jax)

      check_right_eigenvectors(operand, all_w_tf, all_vr_tf)

    # On CPU and GPU, JAX makes custom calls
    always_custom_assert = True
    # On TPU, JAX calls xops.Eigh
    if jtu.device_under_test == "tpu":
      always_custom_assert = False

    limitations = [
      # See https://github.com/google/jax/pull/3775#issuecomment-659407824;
      Jax2TfLimitation(harness,
                       "function not compilable",
                       for_dtypes=[np.complex64, np.complex128],
                       for_modes="compiled",
                       enabled=(shape[0] > 0)),
      Jax2TfLimitation(harness,
                       "TODO: numeric discrepancies",
                       for_dtypes=[np.float64],
                       for_modes="compiled",
                       for_devices=("cpu", "gpu"),
                       disable_comparison=True),
      Jax2TfLimitation(harness,
                       "TODO: numeric discrepancies",
                       for_dtypes=[np.float16],
                       for_modes="compiled",
                       for_devices=("tpu",),
                       disable_comparison=True),
    ]
    return Jax2TfHarnessTrait(custom_assert=custom_assert,
                              always_custom_assert=always_custom_assert,
                              limitations=limitations)

  @classmethod
  def ge(cls, harness: primitive_harness.Harness):
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.bool_],
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint16, np.uint32],
                            for_devices=("cpu", "gpu")),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64],
                            for_devices=("cpu", "gpu"))
    ]
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def gt(cls, harness: primitive_harness.Harness):
    return cls.ge(harness)

  @classmethod
  def erf(cls, harness: primitive_harness.Harness):
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[dtypes.bfloat16],
                            for_devices=("cpu", "gpu")),
    ]
    tol = None
    return Jax2TfHarnessTrait(limitations=limitations, atol=tol, rtol=tol)

  @classmethod
  def erfc(cls, harness: primitive_harness.Harness):
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[dtypes.bfloat16],
                            for_devices=("cpu", "gpu"))
    ]
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def erf_inv(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    custom_assert = tol = None
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[dtypes.bfloat16, np.float16],
                            for_devices=("cpu", "gpu"))
    ]
    # TODO(necula): fix erf_inv bug on TPU
    if jtu.device_under_test() == "tpu" and dtype == np.float32:
      raise unittest.SkipTest("erf_inv bug on TPU: nan vs non-nan")
    # TODO: investigate: in the (b)float16 cases, TF and lax both return the
    # same result in undefined cases.
    if dtype == np.float64:
      tol = 1e-13

    if dtype in [np.float32, np.float64]:
      tol = 1e-4

      # erf_inv is not defined for arg <= -1 or arg >= 1
      def custom_assert(tst, result_jax, result_tf, *, args):  # noqa: F811
        arg, = args
        # for arg < -1 or arg > 1
        # lax.erf_inv returns NaN; tf.math.erf_inv return +/- inf
        special_cases = (arg < -1.) | (arg > 1.)
        nr_special_cases = np.count_nonzero(special_cases)
        tst.assertAllClose(np.full((nr_special_cases,), dtype(np.nan),
                                   dtype=dtype),
                           result_jax[special_cases])
        signs = np.where(arg[special_cases] < 0., -1., 1.)
        tst.assertAllClose(np.full((nr_special_cases,),
                                   signs * dtype(np.inf), dtype=dtype),
                           result_tf[special_cases])
        # non-special cases are equal
        tst.assertAllClose(result_jax[~ special_cases],
                           result_tf[~ special_cases], atol=tol, rtol=tol)

    return Jax2TfHarnessTrait(custom_assert=custom_assert, atol=tol, rtol=tol,
                              limitations=limitations)

  @classmethod
  def expm1(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    tol = None
    if dtype == np.float64:
      tol = 1e-5
    return Jax2TfHarnessTrait(atol=tol, rtol=tol)

  @classmethod
  def fft(cls, harness):
    # if prim is lax.fft_p:
    #   if np_dtype in [np.float64, np.complex128]:
    #     tf_unimpl(np_dtype, additional_msg=("this is a problem only in compiled "
    #                                         "mode (experimental_compile=True))"))
    # if False and len(harness.params["fft_lengths"]) > 3:
    #   if jtu.device_under_test() == "gpu":
    #     with self.assertRaisesRegex(RuntimeError,
    #                                 "FFT only supports ranks 1-3"):
    #       harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
    #   else:
    #     raise unittest.SkipTest("TF does not support >3D FFTs.")
    # else:
    tol = None if jtu.device_under_test() == "tpu" else 1e-3
    limitations = [
      Jax2TfLimitation(harness,
                       "TF function not compileable",
                       for_devices=("cpu", "gpu"),
                       for_dtypes=[np.float64, np.complex128],
                       for_modes="compiled")
    ]
    return Jax2TfHarnessTrait(atol=tol, rtol=tol, limitations=limitations)

  @classmethod
  def _pow_test_util(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    custom_assert = rtol = None
    if dtype in [np.float32, np.complex64]:
      rtol = 1e-2 if jtu.device_under_test() == "tpu" else 1e-3
    elif dtype in [np.float64, np.complex128]:
      rtol = 1e-12
    elif dtype == np.float16:
      rtol = 1
    elif dtype == dtypes.bfloat16:
      # Values get really small for large negative powers.
      rtol = 3

    if dtype in [np.complex64, np.complex128]:
      def custom_assert(tst, result_jax, result_tf, **_):
        # NaNs are mismatched, but assertAllClose will also behave weirdly for
        # complex numbers containing np.inf as one of their components. See
        # https://github.com/numpy/numpy/issues/15959 for more details.
        mask = (np.isnan(result_jax) + np.isnan(result_tf) +
                np.isinf(result_jax) + np.isinf(result_tf))
        tst.assertAllClose(result_jax[~ mask], result_tf[~ mask], rtol=rtol)

    return Jax2TfHarnessTrait(rtol=rtol, custom_assert=custom_assert,
                              always_custom_assert=True)

  @classmethod
  def igamma(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype

    # igamma is not defined when the first argument is <=0
    def custom_assert(tst, result_jax, result_tf, *, args):
      arg1, arg2 = args
      # lax.igamma returns NaN when arg1 == arg2 == 0; tf.math.igamma returns 0
      special_cases = (arg1 == 0.) & (arg2 == 0.)
      nr_special_cases = np.count_nonzero(special_cases)
      tst.assertAllClose(np.full((nr_special_cases,), np.nan, dtype=dtype),
                         result_jax[special_cases])
      tst.assertAllClose(np.full((nr_special_cases,), 0., dtype=dtype),
                         result_tf[special_cases])
      # non-special cases are equal
      tst.assertAllClose(result_jax[~ special_cases],
                         result_tf[~ special_cases])

    return Jax2TfHarnessTrait(custom_assert=custom_assert)

  @classmethod
  def igammac(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    tol = None

    if dtype == np.float64:
      tol = 1e-9
    if jtu.device_under_test() == "gpu":
      tol = 1e-3

    # igammac is not defined when the first argument is <=0
    def custom_assert(tst, result_jax, result_tf, *, args):  # noqa: F811
      arg1, arg2 = args
      # lax.igammac returns 1. when arg1 <= 0; tf.math.igammac returns NaN
      special_cases = (arg1 <= 0.) | (arg2 <= 0)
      nr_special_cases = np.count_nonzero(special_cases)
      tst.assertAllClose(np.full((nr_special_cases,), 1., dtype=dtype),
                         result_jax[special_cases])
      tst.assertAllClose(np.full((nr_special_cases,), np.nan, dtype=dtype),
                         result_tf[special_cases])
      # non-special cases are equal
      tst.assertAllClose(result_jax[~ special_cases],
                         result_tf[~ special_cases], atol=tol, rtol=tol)

    limitations = [
      Jax2TfLimitation(harness,
                       "TODO: nan vs. non-nan",
                       for_devices="tpu",
                       for_dtypes=np.float32,
                       for_modes="compiled",
                       disable_comparison=True)
    ]
    return Jax2TfHarnessTrait(custom_assert=custom_assert, atol=tol, rtol=tol,
                              limitations=limitations)

  @classmethod
  def integer_pow(cls, harness: primitive_harness.Harness):
    y = harness.params["y"]

    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint8, np.uint16, np.int8,
                                        np.int16, np.uint32, np.uint64],
                            also_compiled=True),
      # hitting rtol = nan
      Jax2TfLimitation(harness,
                       "TODO: large numeric difference",
                       for_devices="tpu",
                       for_dtypes=np.complex64,
                       for_modes="compiled",
                       enabled=(y in [1000, -1000]),
                       disable_comparison=True),
      Jax2TfLimitation(harness,
                       "TODO: large numeric difference for overflow",
                       for_dtypes=[np.int32, np.int64, np.float32],
                       enabled=(y > 10),
                       disable_comparison=True)
    ]
    trait = cls._pow_test_util(harness)
    trait.limitations = tuple(trait.limitations) + tuple(limitations)
    return trait

  @classmethod
  def pow(cls, harness: primitive_harness.Harness):
    return cls._pow_test_util(harness)

  @classmethod
  def le(cls, harness: primitive_harness.Harness):
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.bool_],
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint16, np.uint32],
                            for_devices=("cpu", "gpu")),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64],
                            for_devices=("cpu", "gpu"))
    ]
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def lt(cls, harness: primitive_harness.Harness):
    return cls.ge(harness)

  @classmethod
  def lgamma(cls, harness: primitive_harness.Harness):
    atol = rtol = None
    dtype = harness.dtype
    if dtype == np.float64:
      atol, rtol = 1e-14, 1e-11
    elif dtype == np.float32:
      atol, rtol = 1e-5, 1e-3
    return Jax2TfHarnessTrait(
      atol=atol, rtol=rtol,
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[dtypes.bfloat16],
                              for_devices=("cpu", "gpu"))
      ])

  @classmethod
  def log1p(cls, harness: primitive_harness.Harness):
    atol = rtol = None
    dtype = harness.dtype
    if dtype == np.float64:
      atol, rtol = 1e-10, 1e-10
    elif dtype == np.float32:
      atol, rtol = 1e-5, 1e-3
    return Jax2TfHarnessTrait(atol=atol, rtol=rtol)

  @classmethod
  def lu(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    tol = None
    if dtype in [np.float32, np.complex64]:
      if jtu.device_under_test() == "tpu":
        tol = 0.1
      else:
        tol = 1e-5
    if dtype in [np.float64, np.complex128]:
      tol = 1e-13

    def custom_assert(tst, result_jax, result_tf, *, args):
      operand, = args
      lu, pivots, perm = result_tf
      batch_dims = operand.shape[:-2]
      m, n = operand.shape[-2], operand.shape[-1]

      def _make_permutation_matrix(perm):
        result = []
        for idx in itertools.product(*map(range, operand.shape[:-1])):
          result += [0 if c != perm[idx] else 1 for c in range(m)]
        result = np.reshape(np.array(result, dtype=dtype), [*batch_dims, m, m])
        return result

      k = min(m, n)
      l = jnp.tril(lu, -1)[..., :, :k] + jnp.eye(m, k, dtype=dtype)
      u = jnp.triu(lu)[..., :k, :]
      p_mat = _make_permutation_matrix(perm)

      tst.assertArraysEqual(lax.linalg.lu_pivots_to_permutation(pivots, m),
                            perm)
      tst.assertAllClose(jnp.matmul(p_mat, operand), jnp.matmul(l, u),
                         atol=tol, rtol=tol)

    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex64],
                            for_devices="tpu",
                            for_modes="compiled")
    ]
    return Jax2TfHarnessTrait(custom_assert=custom_assert,
                              always_custom_assert=True,
                              limitations=limitations)

  @classmethod
  def _min_max_test_util(cls, harness: primitive_harness.Harness):
    # TODO(bchetioui): discrepancies between TF & JAX when comparing with NaN;
    # JAX always returns NaN, while TF returns the value NaN is compared with.
    def custom_assert(tst, result_jax, result_tf, **_):
      mask = np.isnan(result_jax)
      tst.assertAllClose(result_jax[~ mask], result_tf[~ mask])

    # TODO(bchetioui): figure out why we need always_custom_assert=True
    always_custom_assert = True
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.bool_, np.int8, np.complex64,
                                        np.uint16, np.uint32, np.uint64],
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex128],
                            for_devices=("cpu", "gpu"),
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(custom_assert=custom_assert,
                              always_custom_assert=always_custom_assert,
                              limitations=limitations)

  @classmethod
  def max(cls, harness: primitive_harness.Harness):
    return cls._min_max_test_util(harness)

  @classmethod
  def min(cls, harness: primitive_harness.Harness):
    return cls._min_max_test_util(harness)

  @classmethod
  def mul(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(limitations=[
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint32, np.uint64],
                            also_compiled=True)])

  @classmethod
  def neg(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(limitations=[
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint8, np.uint16, np.uint32, np.uint64],
                            also_compiled=True)
    ])

  @classmethod
  def nextafter(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.float16, dtypes.bfloat16],

                              also_compiled=True)])

  @classmethod
  def population_count(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.uint32, np.uint64],
                              for_devices=("cpu", "gpu"))])

  @classmethod
  def qr(cls, harness: primitive_harness.Harness):
    # See https://github.com/google/jax/pull/3775#issuecomment-659407824;
    #     # experimental_compile=True breaks for complex types.
    # TODO: see https://github.com/google/jax/pull/3775#issuecomment-659407824.
    # - for now, the performance of the HLO QR implementation called when
    #   compiling with TF is expected to have worse performance than the
    #   custom calls made in JAX.
    return Jax2TfHarnessTrait(
      limitations=[cls.missing_tf_kernel(harness,
                                         for_dtypes=[dtypes.bfloat16],
                                         for_devices="tpu",
                                         for_modes=("compiled",))],
      atol=1e-5, rtol=1e-5)

  @classmethod
  def random_gamma(cls, harness: primitive_harness.Harness):
    tol = 1e-5
    if jtu.device_under_test() == "tpu":
      tol = 1e-3
    return Jax2TfHarnessTrait(atol=tol, rtol=tol)

  @classmethod
  def reduce_max(cls, harness: primitive_harness.Harness):
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex64],
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex128],
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def reduce_min(cls, harness: primitive_harness.Harness):
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex64],
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex128],
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def reduce_window_add(cls, harness):
    assert "add" == harness.params["computation"].__name__
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint16, np.uint32],
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex64],
                            for_devices="tpu",
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64],
                            for_devices=("cpu", "gpu"),
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def reduce_window_mul(cls, harness):
    assert "mul" == harness.params["computation"].__name__
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint32],
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64],
                            for_devices=("cpu", "gpu"),
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def reduce_window_min(cls, harness):
    assert "min" == harness.params["computation"].__name__
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint32, np.uint16, np.bool_,
                                        np.complex64, np.int8],
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64, np.complex128],
                            for_devices=("cpu", "gpu"),
                            also_compiled=True)
    ]
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def reduce_window_max(cls, harness):
    assert "max" == harness.params["computation"].__name__
    dtype = harness.dtype
    init_value = harness.params["init_value"]
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint32, np.bool_,
                                        np.complex64],
                            also_compiled=True),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64, np.complex128],
                            for_devices=("cpu", "gpu"),
                            also_compiled=True),
      Jax2TfLimitation(harness,
                       "TF kernel missing, except when the initial_value is the minimum for the dtype",
                       for_dtypes=[np.uint16, np.int8],
                       enabled=((dtype == np.uint16 and init_value != 0) or
                                (dtype == np.int8 and init_value != -128)))

    ]
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def regularized_incomplete_beta(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    # TODO: https://www.tensorflow.org/api_docs/python/tf/math/betainc only
    # supports float32/64 tests.
    tol = None
    if dtype == np.float64:
      tol = 1e-14
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.float16, dtypes.bfloat16],
                              also_compiled=True)],
      atol=tol, rtol=tol)

  @classmethod
  def rem(cls, harness: primitive_harness.Harness):
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint8, np.uint16, np.uint32, np.uint64,
                                        np.int8, np.int16],
                            also_compiled=True),
      Jax2TfLimitation(
        harness, "TF integer division fails if divisor contains 0; JAX returns NaN",
        for_dtypes=[np.uint8, np.int8, np.uint16, np.uint32, np.uint64,
                    np.int8, np.int16, np.int32, np.int64],
        # Only the harnesses with "singularity" will have divide by 0
        enabled=("singularity" in harness.name)),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.float16])
    ]
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def rev(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.uint32, np.uint64],
                              also_compiled=True)])

  @classmethod
  def round(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[dtypes.bfloat16],
                              for_devices=("cpu", "gpu"))])

  @classmethod
  def rsqrt(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[dtypes.bfloat16],
                              for_devices=("cpu", "gpu"))
      ])

  @classmethod
  def scatter_add(cls, harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.int8, np.uint16, np.uint32,
                                          np.uint64,
                                          np.complex64, np.bool_],
                              also_compiled=True)
      ])

  @classmethod
  def scatter_max(cls, harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.int8, np.uint16, np.uint32,
                                          np.uint64,
                                          np.complex64, np.complex128, np.bool_],
                              also_compiled=True)
      ])

  @classmethod
  def scatter_min(cls, harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.int8, np.uint16, np.uint32,
                                          np.complex64, np.bool_, np.uint64,
                                          np.complex128],
                              also_compiled=True)
      ])

  @classmethod
  def scatter_mul(cls, harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.int8, np.uint16, np.uint32,
                                          np.uint64,
                                          np.complex64, np.bool_],
                              also_compiled=True)
      ])

  # lax.select_and_gather_add_p:
  #   # TODO: the conversion is only supported for float16/float32 on CPU/GPU,
  #   # and float16 on TPU. This is because we do not implement a precision
  #   # reduction in the case where packing 2 n-bit values together results in
  #   # more than the maximum number of bits allowed on the platform (64 on
  #   # CPU/GPU, 32 on TPU). This could be fixed by implementing a variadic
  #   # reduce_window in tfxla, or we can require the user to reduce the
  #   # precision of their arrays manually based on the platform they run on.
  #   devices_and_max_bits = [ (["CPU", "GPU"], 64)
  #                          , (["TPU"], 32)
  #                          ]
  #   for devs, max_bits in devices_and_max_bits:
  #     if dtypes.finfo(np_dtype).bits * 2 > max_bits:
  #       # TODO: getting an exception "XLA encountered an HLO for which this
  #       # rewriting is not implemented"
  #       tf_unimpl(np_dtype, devs=devs)
  @classmethod
  def select_and_gather_add(cls, harness):
    return Jax2TfHarnessTrait(limitations=[
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.float32],
                            for_devices="tpu",
                            for_modes="compiled"),
      Jax2TfLimitation(harness,
                       "jax2tf unimplemented",
                       for_dtypes=[np.float64],
                       for_devices=("cpu", "gpu")),
    ])

  @classmethod
  def select_and_scatter_add(cls, harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.uint32, np.uint16],
                              also_compiled=True),
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.uint64],
                              for_devices=("cpu", "gpu"),
                              also_compiled=True)
      ])

  @classmethod
  def sign(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.uint32, np.uint16, np.int16,
                                          np.int8, np.uint8, np.uint64],
                              for_modes=("eager", "graph", "compiled",))])

  @classmethod
  def sinh(cls, harness: primitive_harness.Harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.float16],
                              for_devices=("cpu", "gpu"))])

  @classmethod
  def sort(cls, harness: primitive_harness.Harness):
    if (jtu.device_under_test() == "gpu" and
        len(harness.arg_descriptors) == 4 and
        not harness.params["is_stable"]):
      # TODO: fix the TF GPU test
      raise unittest.SkipTest("GPU tests are running TF on CPU")

    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.complex64, np.complex128],
                            for_devices=("cpu", "gpu")),
      Jax2TfLimitation(harness, "TODO: XlaSort does not support more than 2 arrays",
                       enabled=harness.params["nb_arrays"] > 2),
      Jax2TfLimitation(harness, "TODO: XlaSort does not support sorting axis",
                       enabled=harness.params["dimension"] != len(np.shape(harness.arg_descriptors[0])) - 1),
    ]

    # if prim is lax.sort_p:
    #   if np_dtype == np.bool_ and len(args) == 2:
    #     tf_unimpl(np_dtype, additional_msg=(
    #       "sorting 2 arrays where the first one is an array of booleans is not "
    #       "supported for XlaSort"))
    #   if kwargs["is_stable"]:
    #     tf_unimpl(additional_msg="stable sort not implemented for XlaSort")
    #   if kwargs["dimension"] != len(np.shape(args[0])) - 1:
    #     tf_unimpl(additional_msg="only sorting on last dimension is supported "
    #                              "for XlaSort")
    #   if len(args) > 2:
    #     tf_unimpl(additional_msg=(
    #       "sorting more than 2 arrays is not supported for XlaSort"))
    return Jax2TfHarnessTrait(limitations=limitations)

  @classmethod
  def sub(cls, harness):
    return Jax2TfHarnessTrait(
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.uint64],
                              also_compiled=True)
      ])

  @classmethod
  def svd(cls, harness: primitive_harness.Harness):
    # TODO: slow test
    tol = 1e-4

    def custom_assert(tst, r_jax, r_tf, *, args):
      def _reconstruct_operand(result, is_tf: bool):
        # Reconstructing operand as documented in numpy.linalg.svd (see
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
        s, u, v = result
        U = u[..., :s.shape[-1]]
        V = v[..., :s.shape[-1], :]
        S = s[..., None, :]
        return jnp.matmul(U * S, V), s.shape, u.shape, v.shape

      if harness.params["compute_uv"]:
        r_jax_reconstructed = _reconstruct_operand(r_jax, False)
        r_tf_reconstructed = _reconstruct_operand(r_tf, True)
        tst.assertAllClose(r_jax_reconstructed, r_tf_reconstructed,
                           atol=tol, rtol=tol)
      else:
        tst.assertAllClose(r_jax, r_tf, atol=tol, rtol=tol)

    # if prim is lax.linalg.svd_p:
    #   if np_dtype in [dtypes.bfloat16]:
    #     # TODO: SVD on TPU for bfloat16 seems to work for JAX but fails for TF
    #     tf_unimpl(np_dtype, devs=["TPU"])
    #   elif np_dtype in [np.complex64, np.complex128]:
    #     # TODO: on CPU and GPU "No registered 'Svd' OpKernel for XLA_CPU_JIT
    #     # devices". Works on JAX because JAX uses a custom implementation.
    #     # There exists a XlaSvd operation that could replace tf.linalg.svd in
    #     # these cases but complex numbers support is not implemented in XLA yet,
    #     # and the API of XlaSvd is different than the one in JAX/TF, which also
    #     # limits its useability (e.g. no full_matrices argument, …).
    #     additional_msg = ("this works on JAX because JAX uses a custom "
    #                       "implementation")
    #     tf_unimpl(np_dtype, additional_msg=additional_msg, devs=["CPU", "GPU"])
    limitations = [
      Jax2TfLimitation(harness,
                       "function not compilable",
                       for_dtypes=[np.complex64, np.complex128],
                       for_devices=("cpu", "gpu"),
                       for_modes=("compiled",)),
      cls.missing_tf_kernel(harness,
                            for_dtypes=[dtypes.bfloat16],
                            for_devices="tpu",
                            for_modes="compiled")
    ]
    return Jax2TfHarnessTrait(atol=tol, rtol=tol,
                              custom_assert=custom_assert,
                              always_custom_assert=True,
                              limitations=limitations)

  @classmethod
  def tan(cls, harness):
    atol = rtol = None
    dtype, dut = harness.dtype, jtu.device_under_test()
    if dut == "tpu" and dtype == np.complex64:
      atol, rtol = 1e-4, 1e-5
    elif dut in ["cpu", "gpu"]:
      atol, rtol = ((1e-3, 1e-3) if dtype == np.complex64 else
                    ((1e-12, 1e-12) if dtype == np.complex128 else
                     (atol, rtol)))
    return Jax2TfHarnessTrait(atol=atol, rtol=rtol)

  @classmethod
  def tanh(cls, harness):
    atol = rtol = None
    dtype = harness.dtype
    if dtype == np.complex128:
      atol, rtol = 1e-7, 1e-7
    elif dtype == np.complex64:
      atol, rtol = 1e-4, 1e-5
    return Jax2TfHarnessTrait(atol=atol, rtol=rtol)

  @classmethod
  def top_k(cls, harness):
    dtype = harness.dtype
    limitations = [
      cls.missing_tf_kernel(harness,
                            for_dtypes=[np.uint64, np.int64],
                            for_devices=("cpu", "gpu"),
                            for_modes="compiled")
    ]
    custom_assert = None
    if dtype in jtu.dtypes.all_inexact:
      def custom_assert(tst, result_jax, result_tf, **_):
        assert len(result_jax) == len(result_tf)
        # TODO: TF and JAX sort [inf, nan] differently.
        first_arr_jax, first_arr_tf = result_jax[0], result_tf[0]
        if np.all(first_arr_jax == first_arr_tf):
          for arr_jax, arr_tf in zip(result_jax, result_tf):
            tst.assertArraysEqual(arr_jax, arr_tf)
        else:
          mask_jax, mask_tf = np.isnan(first_arr_jax), np.isnan(first_arr_tf)
          tst.assertArraysEqual(first_arr_jax[~ mask_jax],
                                first_arr_tf[~ mask_tf])

    return Jax2TfHarnessTrait(custom_assert=custom_assert,
                              limitations=limitations)

  @classmethod
  def triangular_solve(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype
    tol = None
    if dtype == np.float32:
      tol = 5e-3

    return Jax2TfHarnessTrait(
      atol=tol, rtol=tol,
      limitations=[
        cls.missing_tf_kernel(harness,
                              for_dtypes=[dtypes.bfloat16],
                              also_compiled=True),
        cls.missing_tf_kernel(harness,
                              for_dtypes=[np.float16],
                              for_modes=("eager", "graph")),
      ])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
