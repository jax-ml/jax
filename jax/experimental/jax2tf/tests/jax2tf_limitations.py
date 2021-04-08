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
"""See primitives_test docstring for how the Jax2TfLimitations are used"""

import itertools
import numpy as np
from typing import Any, Callable, Optional, Sequence, Union

from jax._src import dtypes
from jax import lax
from jax import numpy as jnp

from jax.experimental.jax2tf.tests import primitive_harness

DType = Any


class Jax2TfLimitation(primitive_harness.Limitation):
  """Specific primitive limitations for jax2tf.

  See the primitive_test module docstring for details.
  """
  def __init__(
      self,
      description: str,
      *,
      devices: Union[str, Sequence[str]] = ("cpu", "gpu", "tpu"),
      dtypes: Union[DType, Sequence[DType]] = (),
      enabled: bool = True,
      # jax2tf specific
      modes=("eager", "graph", "compiled"),
      skip_tf_run=False,
      expect_tf_error: bool = True,
      skip_comparison=False,
      custom_assert: Optional[Callable] = None,
      tol=None):
    """See the primitive_harness.Limitation common arguments.

    Args :
      modes: one of "eager", "graph", "compiled"
      skip_tf_run: if set will skip the TF execution. Use this sparingly,
        prefer `expect_tf_error`. Use only when the test cannot recover from
        the TF error.
      expect_tf_error: if set, then expect a TF error in the given mode when
        executing the result of jax2tf conversion. If not set, then the
        limitation must have a custom_assert or non-default tol.
      skip_comparison: skips the numeric comparison.
      tol: a tolerance to use for both atol and rtol. We will use the maximum
        tolerance over all the applicable limitations, irrespective of their
        order.
      custom_assert: if given, then execute as
        `custom_assert(tst, result_jax, result_tf, args=args, tol=tol)`, where
        `tst` is the current TestCase instance, and args are the input
        arguments that the harness created. The `tol` is the maximum tolerance
        based on the applicable limitations.
        `result_tf` is already converted to NumPy arrays.
    """
    super().__init__(
        description,
        devices=devices,
        dtypes=dtypes,
        enabled=enabled)
    if isinstance(modes, str):
      modes = (modes,)
    assert all(m in ["eager", "graph", "compiled"] for m in modes)
    self.modes = modes
    self.expect_tf_error = expect_tf_error
    self.skip_tf_run = skip_tf_run
    self.custom_assert = custom_assert
    self.tol = tol
    self.skip_comparison = skip_comparison


  def get_max_tolerance_limitation(
      self, limitations: Sequence["Jax2TfLimitation"]) -> Optional["Jax2TfLimitation"]:
    """Pick the tolerance limitation that establishes the maximum tolerance"""
    # TODO: it would be best if the limitations with tolerance are mutually exclusive
    # and we don't have to compute the maximum
    # TODO: we made this an instance method only so that we don't have to import
    # this module from tf_test.util.
    max_tol_lim = None
    for l in limitations:
      if l.tol is not None:
        if max_tol_lim is None or l.tol > max_tol_lim.tol:
          max_tol_lim = l
    return max_tol_lim

  def filter(self,  # type: ignore[override]
             dtype: Optional[DType] = None,
             device: Optional[str] = None,
             mode: Optional[str] = None) -> bool:
    return ((mode is None or mode in self.modes) and
            super().filter(device=device, dtype=dtype))


  @classmethod
  def limitations_for_harness(
      cls, harness: primitive_harness.Harness) -> Sequence["Jax2TfLimitation"]:
    group_method = getattr(cls, harness.group_name, None)
    if harness.group_name in cls.harness_groups_no_limitations:
      assert group_method is None, (
          f"Harness group {harness.group_name} is both in "
          f"'harness_groups_no_limitations' and has a custom "
          f"Jax2TfLimitation.classmethod defined (see module docstring)"
      )
      return []
    else:
      assert group_method is not None, (
          f"Harness group {harness.group_name} must be either part of "
          f"'harness_groups_no_limitations' or must have a custom "
          f"Jax2TfLimitation.classmethod defined (see module docstring)"
      )
      limitations = group_method(harness)
      assert isinstance(limitations, (list, tuple))
      return limitations


  # We keep here the explicit set of groups for which we don't have limitations
  harness_groups_no_limitations = {
      "abs", "and", "argmin", "argmax", "broadcast", "broadcast_in_dim", "ceil",
      "concatenate", "cos", "complex", "conj", "device_put", "dynamic_slice",
      "dynamic_update_slice", "exp", "eq", "floor", "log", "gather", "imag",
      "iota", "is_finite", "ne", "not", "or", "pad", "random_split",
      "reduce_and", "reduce_prod", "reduce_or", "reduce_sum", "real", "reshape",
      "select", "shift_left", "shift_right_logical", "shift_right_arithmetic",
      "sin", "slice", "sqrt", "squeeze", "stop_gradient", "tie_in", "transpose",
      "xor", "zeros_like"
  }


  @classmethod
  def helper_get_trig_custom_limitation(cls, np_inverse):

    def custom_assert(tst, result_jax, result_tf, *, args, tol):
      operand, = args
      tst.assertAllClose(operand, np_inverse(result_tf), atol=tol, rtol=tol)

    return custom_numeric(
        description="May return different but still correct results",
        dtypes=[np.complex64, np.complex128],
        custom_assert=custom_assert,
        modes=("eager", "graph"))

  @classmethod
  def acos(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[np.float16, dtypes.bfloat16, np.complex64],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        missing_tf_kernel(
            dtypes=[np.complex128],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=np.complex128, tol=1e-13),
        custom_numeric(dtypes=np.complex64, devices="tpu", tol=1e-3),
        custom_numeric(dtypes=np.complex64, devices=("cpu", "gpu"), tol=1e-4),
        cls.helper_get_trig_custom_limitation(np.cos),
    ]

  @classmethod
  def acosh(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16, np.float16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=np.complex64, devices=("cpu", "gpu"), tol=1e-3),
        custom_numeric(dtypes=np.complex128, devices=("cpu", "gpu"), tol=1e-12),
        cls.helper_get_trig_custom_limitation(np.cosh)
    ]

  @classmethod
  def add(cls, harness: primitive_harness.Harness):
    return []

  @classmethod
  # Also called add_jaxvals
  def add_any(cls, harness: primitive_harness.Harness):
    return []

  @classmethod
  def asin(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[np.float16, dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        missing_tf_kernel(dtypes=[np.complex64, np.complex128]),
        cls.helper_get_trig_custom_limitation(np.sin)
    ]

  @classmethod
  def asinh(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[np.float16, dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=np.complex64, devices=("cpu", "gpu"), tol=1e-3),
        custom_numeric(dtypes=np.complex128, devices=("cpu", "gpu"), tol=1e-12),
        cls.helper_get_trig_custom_limitation(np.sinh)
    ]

  @classmethod
  def atan(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[np.float16, dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        missing_tf_kernel(dtypes=[np.complex64, np.complex128]),
        cls.helper_get_trig_custom_limitation(np.tan)
    ]

  @classmethod
  def atanh(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[np.float16, dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=np.float64, tol=1e-14),
        custom_numeric(dtypes=np.complex64, tol=1e-3),
        custom_numeric(dtypes=np.complex128, devices=("cpu", "gpu"), tol=1e-12),
        cls.helper_get_trig_custom_limitation(np.tanh)
    ]

  @classmethod
  def atan2(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[np.float16, dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def bessel_i0e(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def bessel_i1e(cls, harness: primitive_harness.Harness):
    return cls.bessel_i0e(harness)

  @classmethod
  def bitcast_convert_type(cls, harness: primitive_harness.Harness):
    return [missing_tf_kernel(dtypes=[np.bool_])]

  @classmethod
  def cholesky(cls, harness: primitive_harness.Harness):

    def custom_assert(tst, result_jax, result_tf, *, tol, **_):
      # cholesky_p returns garbage in the strictly upper triangular part of the
      # result, so we can safely ignore that part.
      tst.assertAllClose(jnp.tril(result_jax), result_tf, atol=tol)

    return [
        # See https://github.com/google/jax/pull/3775#issuecomment-659407824;
        Jax2TfLimitation(
            "function not compilable",
            dtypes=[np.complex64, np.complex128],
            devices=("cpu", "gpu"),
            modes="compiled"),
        missing_tf_kernel(
            # Interesting: on TPU, complex64 works in eager
            # mode, but fails otherwise.
            dtypes=[np.complex64, np.complex128],
            devices="tpu",
            modes=("graph", "compiled")),
        # TODO(bchetioui): very high discrepancy in the float32/complex64 case
        custom_numeric(dtypes=[np.float32, np.complex64], tol=1e-2),
        custom_numeric(dtypes=[np.float64, np.complex128], tol=1e-6),
        custom_numeric(dtypes=[dtypes.bfloat16, np.float16], tol=5e-2),
        custom_numeric(
            custom_assert=custom_assert,
            description=(
                "May return different values in the strictly upper triangular "
                "part of the result. This does not matter for correctness, "
                "because this part of the matrix is not considered in the result."
            ))
    ]

  @classmethod
  def clamp(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(dtypes=[np.int8, np.uint16, np.uint32, np.uint64])
    ]

  @classmethod
  def convert_element_type(cls, harness: primitive_harness.Harness):
    return []

  @classmethod
  def conv_general_dilated(cls, harness: primitive_harness.Harness):
    return [
        Jax2TfLimitation(
            "jax2tf BUG: batch_group_count > 1 not yet converted",
            enabled=(harness.params["batch_group_count"] > 1)),
        missing_tf_kernel(dtypes=[np.complex64, np.complex128], devices="gpu"),
        custom_numeric(devices="gpu", tol=1e-4),
        custom_numeric(devices="tpu", tol=1e-3),
        # TODO(bchetioui): significant discrepancies in some float16 cases.
        custom_numeric(dtypes=np.float16, tol=1),
        # TODO(bchetioui): slight occasional discrepancy in float32 cases.
        custom_numeric(dtypes=np.float32, devices="tpu", tol=0.5),
        custom_numeric(dtypes=np.float32, devices="gpu", tol=1e-3),
        custom_numeric(dtypes=np.float32, devices="cpu", tol=1e-4),
        custom_numeric(dtypes=np.complex64, devices="tpu", tol=0.1),
        custom_numeric(dtypes=(np.complex64, np.complex128), devices=("cpu", "gpu"), tol=5e-4),
        # TODO(bchetioui): slight discrepancy when going through the path using
        # tf.nn.convolution.
        custom_numeric(dtypes=np.float64, devices="cpu", tol=1e-13),
    ]

  @classmethod
  def cosh(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[np.float16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def cummax(cls, harness):
    return [
        missing_tf_kernel(
            dtypes=[np.complex128],
            devices=("cpu", "gpu"),
        ),
        missing_tf_kernel(dtypes=[np.complex64], devices=("cpu", "gpu")),
        custom_numeric(dtypes=np.float16, tol=0.1),
        custom_numeric(dtypes=dtypes.bfloat16, tol=0.5)
    ]

  @classmethod
  def cummin(cls, harness):
    return [
        missing_tf_kernel(
            dtypes=[np.uint64, np.complex128],
            devices=("cpu", "gpu"),
        ),
        missing_tf_kernel(
            dtypes=[np.uint16, np.uint32, np.int8, np.complex64],),
        custom_numeric(dtypes=np.float16, tol=0.1),
        custom_numeric(dtypes=dtypes.bfloat16, tol=0.5),
    ]

  @classmethod
  def cumprod(cls, harness):
    return [
        custom_numeric(dtypes=np.float16, tol=0.1),
        custom_numeric(dtypes=dtypes.bfloat16, tol=0.5),
    ]

  @classmethod
  def cumsum(cls, harness):
    return [
        missing_tf_kernel(dtypes=[np.complex64], devices="tpu"),
        custom_numeric(dtypes=np.float16, tol=0.1),
        custom_numeric(dtypes=dtypes.bfloat16, tol=0.5),
    ]

  @classmethod
  def custom_linear_solve(cls, harness: primitive_harness.Harness):
    return [
        Jax2TfLimitation(
            "TODO: large numerical discrepancy",
            dtypes=np.float32,
            devices="tpu",
            expect_tf_error=False,
            skip_comparison=True),
        custom_numeric(dtypes=np.float32, devices="tpu", tol=0.01),
        custom_numeric(tol=1e-3),
    ]

  @classmethod
  def digamma(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype

    # In the bfloat16 case, TF and lax both return NaN in undefined cases.
    # digamma is not defined at 0 and -1
    def custom_assert(tst, result_jax, result_tf, *, args, tol):
      # lax.digamma returns NaN and tf.math.digamma returns inf
      arg, = args
      special_cases = (arg == 0.) | (arg == -1.)
      nr_special_cases = np.count_nonzero(special_cases)
      tst.assertAllClose(
          np.full((nr_special_cases,), dtype(np.nan)),
          result_jax[special_cases])
      tst.assertAllClose(
          np.full((nr_special_cases,), dtype(np.inf)), result_tf[special_cases])
      # non-special cases are equal
      tst.assertAllClose(
          result_jax[~special_cases],
          result_tf[~special_cases],
          atol=tol,
          rtol=tol)

    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=np.float64, tol=1e-13),
        custom_numeric(dtypes=np.float32, devices=["cpu", "gpu"], tol=1e-3),
        custom_numeric(
            dtypes=dtypes.bfloat16,
            custom_assert=custom_assert,
            description=(
                "May return different results at singularity points 0 and -1."
                "JAX returns nan and TF returns inf"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def div(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[
                np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16
            ],),
        Jax2TfLimitation(
            "TF integer division fails if divisor contains 0; JAX returns NaN",
            dtypes=[
                np.uint8, np.int8, np.uint16, np.uint32, np.uint64, np.int8,
                np.int16, np.int32, np.int64
            ],
            # Only the harnesses with "singularity" will have divide by 0
            enabled=("singularity" in harness.name))
    ]

  @classmethod
  def dot_general(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[
                np.bool_, np.uint8, np.uint16, np.uint32, np.uint64, np.int8,
                np.int16
            ],),
        missing_tf_kernel(
            dtypes=np.int64, devices=("cpu", "gpu"),
            modes="compiled",
            # Works for 2D matrices.
            enabled=(len(harness.params["lhs_shape"]) > 2)),
        custom_numeric(dtypes=dtypes.bfloat16, tol=0.3),
        custom_numeric(
            dtypes=[np.complex64, np.float32], devices=("cpu", "gpu"),
            tol=1e-5),
        custom_numeric(
            dtypes=[np.complex128, np.float64], devices=("cpu", "gpu"),
            tol=1e-12),
        custom_numeric(dtypes=np.float32, devices="tpu", tol=0.1),
        custom_numeric(dtypes=np.complex64, devices="tpu", tol=0.3),
        custom_numeric(dtypes=np.float16, devices=("gpu", "tpu"), tol=0.1),
        custom_numeric(dtypes=np.float16, devices="cpu", tol=0.01)
    ]

  @classmethod
  def eig(cls, harness: primitive_harness.Harness):
    compute_left_eigenvectors = harness.params["compute_left_eigenvectors"]
    compute_right_eigenvectors = harness.params["compute_right_eigenvectors"]
    dtype = harness.dtype

    def custom_assert(tst, result_jax, result_tf, *, args, tol):
      operand, = args
      inner_dimension = operand.shape[-1]

      # Test ported from tests.linlag_test.testEig
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
        tst.assertAllClose(
            closest_diff, np.array(0., closest_diff.dtype), atol=tol)

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

    return [
        # Eig does not work in JAX on gpu or tpu
        Jax2TfLimitation("function not compilable", modes="compiled",
                         devices="cpu"),
        Jax2TfLimitation(
            "TF Conversion of eig is not implemented when both compute_left_eigenvectors and compute_right_eigenvectors are set to True",
            enabled=(compute_left_eigenvectors and compute_right_eigenvectors)),
        custom_numeric(
            custom_assert=custom_assert,
            description=("May return the eigenvalues and eigenvectors in a "
                         "potentially different order. The eigenvectors may "
                         "also be different, but equally valid."),
            modes=("eager", "graph"))
    ]

  @classmethod
  def eigh(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype

    def custom_assert(tst, result_jax, result_tf, *, args, tol):
      operand, = args
      inner_dimension = operand.shape[-1]

      def check_right_eigenvectors(a, w, vr):
        tol = 1e-16
        # TODO(bchetioui): tolerance needs to be very high in compiled mode,
        # specifically for eigenvectors.
        if dtype == np.float64:
          tol = 2e-5
        elif dtype == np.float32:
          tol = 1e-2
        elif dtype in [dtypes.bfloat16, np.complex64]:
          tol = 1e-3
        elif dtype == np.complex128:
          tol = 2e-5
        tst.assertAllClose(
            np.matmul(a, vr) - w[..., None, :] * vr,
            np.zeros(a.shape, dtype=vr.dtype),
            atol=tol,
            # For bfloat16 the np.matmul returns float32 result.
            check_dtypes=False)

      def check_eigenvalue_is_in_array(eigenvalue, eigenvalues_array):
        tol = None
        if dtype in [dtypes.bfloat16, np.float32, np.complex64]:
          tol = 1e-3
        elif dtype in [np.float64, np.complex128]:
          tol = 1e-5
        closest_diff = min(abs(eigenvalues_array - eigenvalue))
        tst.assertAllClose(
            closest_diff, np.array(0., closest_diff.dtype), atol=tol)

      _, all_w_jax = result_jax
      all_vr_tf, all_w_tf = result_tf

      for idx in itertools.product(*map(range, operand.shape[:-2])):
        w_jax, w_tf = all_w_jax[idx], all_w_tf[idx]
        for i in range(inner_dimension):
          check_eigenvalue_is_in_array(w_jax[i], w_tf)
          check_eigenvalue_is_in_array(w_tf[i], w_jax)

      check_right_eigenvectors(operand, all_w_tf, all_vr_tf)

    return [
        # See https://github.com/google/jax/pull/3775#issuecomment-659407824;
        # TODO(b/181414529): enable after XLA/GPU bug is fixed.
        Jax2TfLimitation(
            "XLA lowering bug",
            dtypes=(np.complex64, np.complex128),
            devices=("gpu",),
            modes="compiled",
            skip_tf_run=True),
        missing_tf_kernel(
            dtypes=dtypes.bfloat16,
            devices="tpu",
            enabled=(harness.params["shape"] != (0, 0)),  # This actually works!
            ),
        Jax2TfLimitation(
            "function not yet compilable",
            dtypes=(np.complex64, np.complex128),
            modes="compiled",
            skip_tf_run=True),
        Jax2TfLimitation(
            "TODO: numeric discrepancies",
            dtypes=np.float16,
            devices="tpu",
            expect_tf_error=False,
            skip_comparison=True),
        custom_numeric(
            custom_assert=custom_assert,
            description=("May return the eigenvalues and eigenvectors in a "
                         "potentially different order. The eigenvectors may "
                         "also be different, but equally valid."))
    ]

  @classmethod
  def ge(cls, harness: primitive_harness.Harness):
    return [missing_tf_kernel(dtypes=[np.bool_])]

  @classmethod
  def gt(cls, harness: primitive_harness.Harness):
    return cls.ge(harness)

  @classmethod
  def erf(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def erfc(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def erf_inv(cls, harness: primitive_harness.Harness):
    # erf_inv is not defined for arg <= -1 or arg >= 1
    def custom_assert(tst, result_jax, result_tf, *, args, tol):  # noqa: F811
      arg, = args
      # for arg < -1 or arg > 1
      # lax.erf_inv returns NaN; tf.math.erf_inv return +/- inf
      special_cases = (arg < -1.) | (arg > 1.)
      # non-special cases are equal
      tst.assertAllClose(
          result_jax[~special_cases],
          result_tf[~special_cases],
          atol=tol,
          rtol=tol)

    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16, np.float16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=[np.float32, np.float64], tol=1e-4),
        custom_numeric(
            dtypes=[np.float32, np.float64],
            custom_assert=custom_assert,
            description=(
                "May return different results at undefined points (< -1 or > 1):"
                " JAX returns `NaN` and TF returns `+inf` or `-inf`."))
    ]

  @classmethod
  def expm1(cls, harness: primitive_harness.Harness):
    return [custom_numeric(dtypes=np.float64, tol=1e-5)]

  @classmethod
  def fft(cls, harness):
    return [
        Jax2TfLimitation(
            "TF function not compileable",
            devices=("cpu", "gpu"),
            dtypes=[np.float64, np.complex128],
            modes="compiled"),
        custom_numeric(tol=1e-3)
    ]

  @classmethod
  def _pow_test_util(cls, harness: primitive_harness.Harness):

    def custom_assert(tst, result_jax, result_tf, *, args, tol):
      # NaNs are mismatched, but assertAllClose will also behave weirdly for
      # complex numbers containing np.inf as one of their components. See
      # https://github.com/numpy/numpy/issues/15959 for more details.
      mask = (
          np.isnan(result_jax) + np.isnan(result_tf) + np.isinf(result_jax) +
          np.isinf(result_tf))
      tst.assertAllClose(result_jax[~mask], result_tf[~mask], rtol=tol)

    return [
        custom_numeric(
            dtypes=[np.float32, np.complex64], devices="tpu", tol=1e-2),
        custom_numeric(
            dtypes=[np.float32, np.complex64], devices=("cpu", "gpu"),
            tol=1e-3),
        custom_numeric(dtypes=[np.float64, np.complex128], tol=1e-12),
        custom_numeric(dtypes=np.float16, tol=1),
        # Values get really small for large negative powers.
        custom_numeric(dtypes=dtypes.bfloat16, tol=3),
        custom_numeric(
            dtypes=[np.complex64, np.complex128],
            custom_assert=custom_assert,
        )
    ]

  @classmethod
  def igamma(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype

    # igamma is not defined when the first argument is <=0
    def custom_assert(tst, result_jax, result_tf, *, args, tol):
      arg1, arg2 = args
      # lax.igamma returns NaN when arg1 == arg2 == 0; tf.math.igamma returns 0
      special_cases = (arg1 == 0.) & (arg2 == 0.)
      nr_special_cases = np.count_nonzero(special_cases)
      tst.assertAllClose(
          np.full((nr_special_cases,), np.nan, dtype=dtype),
          result_jax[special_cases])
      tst.assertAllClose(
          np.full((nr_special_cases,), 0., dtype=dtype),
          result_tf[special_cases])
      # non-special cases are equal
      tst.assertAllClose(result_jax[~special_cases], result_tf[~special_cases],
                         atol=tol, rtol=tol)

    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16, np.float16]),
        custom_numeric(
            custom_assert=custom_assert,
            description=(
                "May return different results at undefined points "
                "(both arguments 0). JAX returns `NaN` and TF returns 0 or "
                "JAX returns 1 and TF returns `NaN`"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def igammac(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype

    # igammac is not defined when the first argument is <=0
    def custom_assert(tst, result_jax, result_tf, *, args, tol):  # noqa: F811
      arg1, arg2 = args
      # lax.igammac returns 1. when arg1 <= 0; tf.math.igammac returns NaN
      special_cases = (arg1 <= 0.) | (arg2 <= 0)
      nr_special_cases = np.count_nonzero(special_cases)
      tst.assertAllClose(
          np.full((nr_special_cases,), 1., dtype=dtype),
          result_jax[special_cases])
      tst.assertAllClose(
          np.full((nr_special_cases,), np.nan, dtype=dtype),
          result_tf[special_cases])
      # non-special cases are equal
      tst.assertAllClose(
          result_jax[~special_cases],
          result_tf[~special_cases],
          atol=tol,
          rtol=tol)

    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16, np.float16]),
        custom_numeric(dtypes=np.float64, tol=1e-9),
        custom_numeric(devices="gpu", tol=1e-3),
        custom_numeric(
            custom_assert=custom_assert,
            devices=("cpu", "gpu"),
            modes=("eager", "graph"),
            description=(
                "May return different results at undefined points "
                "(both arguments less or equal 0). JAX returns `NaN` and TF returns 0 or "
                "JAX returns 1 and TF returns `NaN`")),
    ]

  @classmethod
  def integer_pow(cls, harness: primitive_harness.Harness):
    y = harness.params["y"]
    return [
        missing_tf_kernel(
            dtypes=[
                np.uint8, np.uint16, np.uint32, np.uint64
            ],),
        Jax2TfLimitation(
            "Different overflow behavior for large exponents. ",
            dtypes=[np.int8, np.int16, np.int32, np.int64, np.float32, np.complex64],
            enabled=(abs(y) > 10),
            expect_tf_error=False,
            skip_comparison=True)
    ] + list(cls._pow_test_util(harness))

  @classmethod
  def pow(cls, harness: primitive_harness.Harness):
    return cls._pow_test_util(harness)

  @classmethod
  def le(cls, harness: primitive_harness.Harness):
    return cls.ge(harness)

  @classmethod
  def lt(cls, harness: primitive_harness.Harness):
    return cls.ge(harness)

  @classmethod
  def lgamma(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=np.float64, tol=1e-11),
        custom_numeric(dtypes=np.float32, tol=1e-3)
    ]

  @classmethod
  def log1p(cls, harness: primitive_harness.Harness):
    return [
        custom_numeric(dtypes=np.float64, tol=1e-10),
        custom_numeric(dtypes=np.float32, tol=1e-3)
    ]

  @classmethod
  def lu(cls, harness: primitive_harness.Harness):
    dtype = harness.dtype

    def custom_assert(tst, result_jax, result_tf, *, args, tol):
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

      tst.assertArraysEqual(
          lax.linalg.lu_pivots_to_permutation(pivots, m), perm)
      tst.assertAllClose(
          jnp.matmul(p_mat, operand), jnp.matmul(l, u), atol=tol, rtol=tol)

    return [
        missing_tf_kernel(dtypes=[np.complex64], devices="tpu"),
        custom_numeric(
            dtypes=[np.float32, np.complex64], devices="tpu", tol=0.1),
        custom_numeric(
            dtypes=[np.float32, np.complex64], devices=("cpu", "gpu"),
            tol=1e-5),
        custom_numeric(dtypes=[np.float64, np.complex128], tol=1e-13),
        custom_numeric(
            custom_assert=custom_assert,
            description=("May return different, but also correct, results when "
                         "the decomposition is not unique")),
    ]

  @classmethod
  def max(cls, harness: primitive_harness.Harness):
    # TODO(bchetioui): discrepancies between TF & JAX when comparing with NaN;
    # JAX always returns NaN, while TF returns the value NaN is compared with.
    def custom_assert(tst, result_jax, result_tf, **_):
      mask = np.isnan(result_jax)
      tst.assertAllClose(result_jax[~mask], result_tf[~mask])

    return [
        missing_tf_kernel(
            dtypes=[np.int8, np.uint16, np.uint32, np.uint64],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"),
        ),
        missing_tf_kernel(
            dtypes=[np.bool_, np.complex64]),
        missing_tf_kernel(
            dtypes=[np.complex128],
            devices=("cpu", "gpu"),
        ),
        custom_numeric(
            custom_assert=custom_assert,
            description=(
                "May return different values when one of the values is NaN. "
                "JAX always returns NaN, while TF returns the value NaN is compared with."
            ))
    ]

  @classmethod
  def min(cls, harness: primitive_harness.Harness):
    # TODO(bchetioui): discrepancies between TF & JAX when comparing with NaN;
    # JAX always returns NaN, while TF returns the value NaN is compared with.
    def custom_assert(tst, result_jax, result_tf, **_):
      mask = np.isnan(result_jax)
      tst.assertAllClose(result_jax[~mask], result_tf[~mask])

    return [
        missing_tf_kernel(
            dtypes=[np.bool_, np.int8, np.uint16, np.uint32, np.uint64,
                    np.complex64]),
        missing_tf_kernel(
            dtypes=[np.complex128],
            devices=("cpu", "gpu"),
        ),
        custom_numeric(
            custom_assert=custom_assert,
            description=(
                "May return different values when one of the values is NaN. "
                "JAX always returns NaN, while TF returns the value NaN is compared with."
            ))
    ]

  @classmethod
  def mul(cls, harness: primitive_harness.Harness):
    return []

  @classmethod
  def neg(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(dtypes=[np.uint8, np.uint16, np.uint32, np.uint64],)
    ]

  @classmethod
  def nextafter(cls, harness: primitive_harness.Harness):
    return [missing_tf_kernel(dtypes=[np.float16, dtypes.bfloat16])]

  @classmethod
  def population_count(cls, harness: primitive_harness.Harness):
    return []

  @classmethod
  def qr(cls, harness: primitive_harness.Harness):
    # See https://github.com/google/jax/pull/3775#issuecomment-659407824;
    #     # jit_compile=True breaks for complex types.
    # TODO: see https://github.com/google/jax/pull/3775#issuecomment-659407824.
    # - for now, the performance of the HLO QR implementation called when
    #   compiling with TF is expected to have worse performance than the
    #   custom calls made in JAX.
    return [
        custom_numeric(tol=1e-4),
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices="tpu",
        )
    ]

  @classmethod
  def random_gamma(cls, harness: primitive_harness.Harness):
    return [custom_numeric(devices="tpu", tol=1e-3)]

  @classmethod
  def reduce_max(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(dtypes=[np.complex64]),
        missing_tf_kernel(dtypes=[np.complex128])
    ]

  @classmethod
  def reduce_min(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(dtypes=[np.complex64]),
        missing_tf_kernel(dtypes=[np.complex128])
    ]

  @classmethod
  def reduce_window_add(cls, harness):
    assert "add" == harness.params["computation"].__name__
    return [
        missing_tf_kernel(dtypes=[np.complex64], devices="tpu"),
    ]

  @classmethod
  def reduce_window_max(cls, harness):
    assert "max" == harness.params["computation"].__name__
    return [
        missing_tf_kernel(dtypes=[np.bool_, np.complex64]),
        missing_tf_kernel(
            dtypes=[np.complex128],
            devices=("cpu", "gpu"),
        )
    ]

  @classmethod
  def reduce_window_min(cls, harness):
    assert "min" == harness.params["computation"].__name__
    return [
        missing_tf_kernel(
            dtypes=[np.bool_, np.int8, np.uint16, np.uint32, np.uint64,
                    np.complex64, np.complex128],
        )
    ]

  @classmethod
  def reduce_window_mul(cls, harness):
    assert "mul" == harness.params["computation"].__name__
    return []

  @classmethod
  def regularized_incomplete_beta(cls, harness: primitive_harness.Harness):
    return [
        custom_numeric(dtypes=np.float64, tol=1e-14),
        missing_tf_kernel(dtypes=[np.float16, dtypes.bfloat16])
    ]

  @classmethod
  def rem(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[
                np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16
            ],),
        Jax2TfLimitation(
            "TF integer division fails if divisor contains 0; JAX returns NaN",
            dtypes=[
                np.uint8, np.int8, np.uint16, np.uint32, np.uint64, np.int8,
                np.int16, np.int32, np.int64
            ],
            # Only the harnesses with "singularity" will have divide by 0
            enabled=("singularity" in harness.name)),
    ]

  @classmethod
  def rev(cls, harness: primitive_harness.Harness):
    return [missing_tf_kernel(dtypes=[np.uint32, np.uint64])]

  @classmethod
  def round(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def rsqrt(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def scatter_add(cls, harness):
    return [
        missing_tf_kernel(dtypes=[np.bool_]),
        missing_tf_kernel(
            dtypes=[np.complex64],
            devices="tpu",
        )]

  @classmethod
  def scatter_max(cls, harness):
    return [
        missing_tf_kernel(
            dtypes=[np.bool_, np.complex64, np.complex128])
    ]

  @classmethod
  def scatter_min(cls, harness):
    return [
        missing_tf_kernel(
            dtypes=[
                np.int8, np.uint16, np.uint32, np.complex64, np.bool_,
                np.uint64, np.complex128
            ],)
    ]

  @classmethod
  def scatter_mul(cls, harness):
    return [
        missing_tf_kernel(dtypes=[np.bool_],),
        missing_tf_kernel(
            dtypes=[np.complex64],
            devices="tpu",
        ),
    ]

  @classmethod
  def select_and_gather_add(cls, harness):
    return [
        missing_tf_kernel(
            dtypes=[np.float32],
            devices="tpu",
            description=(
                "This JAX primitives is not not exposed directly in the JAX API "
                "but arises from JVP of `lax.reduce_window` for reducers "
                "`lax.max` or `lax.min`. It also arises from second-order "
                "VJP of the same. Implemented using XlaReduceWindow")),
        Jax2TfLimitation((
            "jax2tf unimplemented for 64-bit inputs because the current implementation "
            "relies on packing two values into a single value. This can be "
            "fixed by using a variadic XlaReduceWindow, when available"),
                         dtypes=[np.float64],
                         devices=("cpu", "gpu"))
    ]

  @classmethod
  def select_and_scatter_add(cls, harness):
    return []

  @classmethod
  def sign(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[
                np.uint32, np.uint16, np.int16, np.int8, np.uint8, np.uint64
            ],)
    ]

  @classmethod
  def sinh(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(
            dtypes=[np.float16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def sort(cls, harness: primitive_harness.Harness):
    return [
        Jax2TfLimitation(
            # I think that this is because TF is running on CPU even for GPU tests?
            "TODO: TF non-stable multiple-array sort",
            devices="gpu",
            enabled=(harness.params["num_arrays"] > 1 and
                     not harness.params["is_stable"]),
            expect_tf_error=False,
            skip_comparison=True),
        missing_tf_kernel(dtypes=[np.bool_],),
    ]

  @classmethod
  def sub(cls, harness):
    return []

  @classmethod
  def svd(cls, harness: primitive_harness.Harness):
    # TODO: slow test

    def custom_assert(tst, r_jax, r_tf, *, args, tol):

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
        tst.assertAllClose(
            r_jax_reconstructed, r_tf_reconstructed, atol=tol, rtol=tol)
      else:
        tst.assertAllClose(r_jax, r_tf, atol=tol, rtol=tol)

    return [
        # Works in JAX for complex due to custom calls on cpu and gpu
        Jax2TfLimitation(
            "function not compilable. Implemented using `tf.linalg.svd` and `tf.linalg.adjoint`",
            dtypes=[np.complex64, np.complex128],
            devices=("cpu", "gpu"),
            modes=("compiled",)),
        missing_tf_kernel(dtypes=[dtypes.bfloat16], devices="tpu"),
        custom_numeric(tol=1e-4),
        custom_numeric(custom_assert=custom_assert)
    ]

  @classmethod
  def tan(cls, harness):
    return [
        custom_numeric(dtypes=np.complex64, devices="tpu", tol=1e-4),
        custom_numeric(dtypes=np.complex64, devices=("cpu", "gpu"), tol=1e-3),
        custom_numeric(dtypes=np.complex128, devices=("cpu", "gpu"), tol=1e-12)]

  @classmethod
  def tanh(cls, harness):
    return [
        custom_numeric(dtypes=np.complex128, tol=1e-7),
        custom_numeric(dtypes=np.complex64, tol=1e-4)]

  @classmethod
  def top_k(cls, harness):

    def custom_assert(tst, result_jax, result_tf, **_):
      assert len(result_jax) == len(result_tf)
      # TODO: TF and JAX sort [inf, nan] differently.
      first_arr_jax, first_arr_tf = result_jax[0], result_tf[0]
      if np.all(first_arr_jax == first_arr_tf):
        for arr_jax, arr_tf in zip(result_jax, result_tf):
          tst.assertArraysEqual(arr_jax, arr_tf)
      else:
        mask_jax, mask_tf = np.isnan(first_arr_jax), np.isnan(first_arr_tf)
        tst.assertArraysEqual(first_arr_jax[~mask_jax], first_arr_tf[~mask_tf])

    return [
        missing_tf_kernel(
            dtypes=[np.uint64, np.int64],
            devices=("cpu", "gpu"),
            modes="compiled"),
        custom_numeric(
            dtypes=[np.float16, dtypes.bfloat16, np.float32, np.float64],
            custom_assert=custom_assert,
            description=(
               "Produces different results when the array contains `inf` and `NaN`"
               " (they are sorted differently in TF vs. XLA).")
        )]

  @classmethod
  def triangular_solve(cls, harness: primitive_harness.Harness):
    return [
        missing_tf_kernel(dtypes=[dtypes.bfloat16]),
        missing_tf_kernel(
            dtypes=[np.float16],
            devices=("gpu", "cpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=np.float32, tol=5e-3)
    ]


def custom_numeric(
    *,
    description="custom numeric comparison",
    dtypes=(),  # All
    modes=("eager", "graph", "compiled"),
    devices=("cpu", "gpu", "tpu"),
    custom_assert=None,
    tol=None) -> Jax2TfLimitation:

  return Jax2TfLimitation(
      description,
      expect_tf_error=False,
      dtypes=dtypes,
      devices=devices,
      modes=modes,
      custom_assert=custom_assert,
      tol=tol)


def missing_tf_kernel(
    *,
    description="op not defined for dtype",
    dtypes,
    modes=("eager", "graph", "compiled"),
    devices=("cpu", "gpu", "tpu"),
    enabled=True
) -> Jax2TfLimitation:

  return Jax2TfLimitation(
      description,
      dtypes=dtypes,
      devices=devices,
      modes=modes,
      enabled=enabled)
