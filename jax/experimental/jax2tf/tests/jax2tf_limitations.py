# Copyright 2021 The JAX Authors.
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
"""See primitives_test docstring for how the Jax2TfLimitations are used."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from jax._src.internal_test_util import test_harnesses
from jax._src import dtypes
import numpy as np

DType = Any


class Jax2TfLimitation(test_harnesses.Limitation):
  """Specific primitive limitations for jax2tf.

  See the primitive_test module docstring for details.
  """

  def __init__(
      self,
      description: str,
      *,
      devices: str | Sequence[str] = ("cpu", "gpu", "tpu"),
      dtypes: Sequence[DType] = (),
      enabled: bool = True,
      # jax2tf specific
      modes=("eager", "graph", "compiled"),
      skip_tf_run=False,
      expect_tf_error: bool = True,
      skip_comparison=False,
      custom_assert: Callable | None = None,
      tol=None):
    """See the test_harnesses.Limitation common arguments.

    Args :
      modes: one of "eager", "graph", "compiled"
      for_native_serialization: A bitmask with some of {FOR_NATIVE, FOR_NON_NATIVE}
        to specify how the limitation applies to native and non-native lowering.
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
        `custom_assert(tst, result_jax, result_tf, args=args, tol=tol, err_msg)`
        , where `tst` is the current TestCase instance, and args are the input
        arguments that the harness created. The `tol` is the maximum tolerance
        based on the applicable limitations. `err_msg` is passed to NumPy
        assert methods.
        `result_tf` is already converted to NumPy arrays.
    """
    super().__init__(
        description, devices=devices, dtypes=dtypes, enabled=enabled)
    if isinstance(modes, str):
      modes = (modes,)
    assert all(m in ["eager", "graph", "compiled"] for m in modes), "Invalid modes: {modes}"
    self.modes = modes
    self.expect_tf_error = expect_tf_error
    self.skip_tf_run = skip_tf_run
    self.custom_assert = custom_assert
    self.tol = tol
    self.skip_comparison = skip_comparison

  def get_max_tolerance_limitation(
      self, limitations: Sequence[Jax2TfLimitation]
  ) -> Jax2TfLimitation | None:
    """Pick the tolerance limitation that establishes the maximum tolerance."""
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

  @classmethod
  def limitations_for_harness(
      cls, harness: test_harnesses.Harness) -> Sequence[Jax2TfLimitation]:
    group_method = getattr(cls, harness.group_name, None)
    if group_method is not None:
      limitations = group_method(harness)
      assert isinstance(limitations, (list, tuple))
      return limitations
    else:
      return []

  @classmethod
  def asinh(cls, harness: test_harnesses.Harness):
    return [
        custom_numeric(dtypes=[np.complex128], devices=("cpu",),
                       modes=("eager", "compiled", "graph"),
                       tol=1e-13),
    ]

  @classmethod
  def cholesky(cls, harness: test_harnesses.Harness):
    return [
        custom_numeric(
            dtypes=[dtypes.bfloat16],
            tol=5e-5,
            # Error for GL
            devices=("tpu",),
            modes=("eager", "graph", "compiled")),
    ]

  @classmethod
  def conv_general_dilated(cls, harness: test_harnesses.Harness):
    return [
        # Even in compiled mode, for GPU we see a bit of discrepancy but
        # very minor.
        custom_numeric(dtypes=[np.float32], devices="cpu",
                       modes=("eager", "graph", "compiled"),
                       tol=1e-4),
    ]


  @classmethod
  def fft(cls, harness):
    return [
        custom_numeric(tol=1e-5, modes=("eager", "graph", "compiled"),
                       devices=("cpu",)),
    ]

  @classmethod
  def max(cls, harness: test_harnesses.Harness):
    # TODO(bchetioui): discrepancies between TF & JAX when comparing with NaN;
    # JAX always returns NaN, while TF returns the value NaN is compared with.
    def custom_assert(tst, result_jax, result_tf, err_msg, **_):
      mask = np.isnan(result_jax)
      tst.assertAllClose(result_jax[~mask], result_tf[~mask], err_msg=err_msg)

    return [
        # TODO(b/269996580)
        custom_numeric(
            custom_assert=custom_assert,
            devices="cpu",
            description=(
                "TF and JAX use different values of the compiler flag "
                "xla_cpu_enable_fast_min_max compiler flag and therefore have "
                "different behavior of NaN propagation through min/max."
            ),
            modes=("eager", "graph", "compiled"))
    ]

  @classmethod
  def min(cls, harness: test_harnesses.Harness):
    # TODO(bchetioui): discrepancies between TF & JAX when comparing with NaN;
    # JAX always returns NaN, while TF returns the value NaN is compared with.
    def custom_assert(tst, result_jax, result_tf, *, err_msg, **_):
      mask = np.isnan(result_jax)
      tst.assertAllClose(result_jax[~mask], result_tf[~mask], err_msg=err_msg)

    return [
        # TODO(b/269996580)
        custom_numeric(
            custom_assert=custom_assert,
            devices="cpu",
            description=(
                "TF and JAX use different values of the compiler flag "
                "xla_cpu_enable_fast_min_max compiler flag and therefore have "
                "different behavior of NaN propagation through min/max."
            ),
            modes=("eager", "graph", "compiled"),
        )
    ]


def custom_numeric(
    *,
    description="custom numeric comparison",
    dtypes=(),  # All
    modes=(
        "eager",
        "graph",
    ),  # By default we should not need tolerance for
    # "compiled"
    devices=("cpu", "gpu", "tpu"),
    custom_assert=None,
    enabled=True,
    tol=None) -> Jax2TfLimitation:

  return Jax2TfLimitation(
      description,
      expect_tf_error=False,
      dtypes=dtypes,
      devices=devices,
      modes=modes,
      custom_assert=custom_assert,
      enabled=enabled,
      tol=tol)
