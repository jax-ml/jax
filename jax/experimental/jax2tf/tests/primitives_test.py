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
"""Tests for JAX primitive coverage."""

import unittest

from absl.testing import absltest
from absl.testing import parameterized

from functools import partial
import itertools

import jax
from jax import dtypes
from jax import lax
from jax import numpy as jnp
from jax import test_util as jtu
from jax._src.lax import control_flow as lax_control_flow
from jax.config import config
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import tf_test_util
from jax.interpreters import xla

import numpy as np
import tensorflow as tf  # type: ignore[import]

config.parse_flags_with_absl()

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

INDEX = (
  jax.ops.index_add,
  jax.ops.index_max,
  jax.ops.index_min,
  jax.ops.index_mul,
  jax.ops.index_update,
)


class JaxPrimitiveTest(tf_test_util.JaxToTfTestCase):

  def test_primitive_coverage(self):
    """Fail if there are JAX primitives that are not implemented."""
    # Harvest primitives from XLA translation tables
    all_primitives = (set(xla.translations)
                      | set(xla.backend_specific_translations['cpu'])
                      | set(xla.backend_specific_translations['gpu'])
                      | set(xla.backend_specific_translations['tpu'])
                      | set(xla.initial_style_translations)
                      | set(xla.parallel_translations))

    tf_impl = set(jax.experimental.jax2tf.jax2tf.tf_impl) | set(jax.experimental.jax2tf.jax2tf.tf_impl_with_avals)
    tf_not_yet_impl = set(jax.experimental.jax2tf.jax2tf.tf_not_yet_impl)

    all_primitives = tuple(sorted(all_primitives, key=str))
    for p in all_primitives:
      # TODO: remove tie_in once omnistaging is on by default
      if p.name == "axis_index" or p.name == "tie_in":
        continue
      if p in tf_not_yet_impl:
        self.assertNotIn(p, tf_impl)  # Should not be in both tf_impl and tf_not_yet_impl
      else:
        self.assertIn(p, tf_impl)

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

  def test_concat(self):
    values = [np.array([1, 2], dtype=np.float32),
              np.array([1, 2], dtype=np.int32),
              np.array([1, 2], dtype=np.int8)]
    f_jax = jax.jit(lambda x: jnp.concatenate(x, axis=0))
    self.ConvertAndCompare(f_jax, values)

  @primitive_harness.parameterized(primitive_harness.lax_pad)
  def test_pad(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_select)
  def test_select(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_transpose)
  def test_transpose(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_control_flow_cumreduce)
  def test_cumreduce(self, harness: primitive_harness.Harness):
    f_jax, dtype = harness.params["f_jax"], harness.params["dtype"]
    dut = jtu.device_under_test()
    if (dtype == np.complex64 and
        f_jax in [lax_control_flow.cummin, lax_control_flow.cummax,
                  lax_control_flow.cumprod, lax_control_flow.cumsum] and
        dut == "tpu"):
      raise unittest.SkipTest("TODO(bchetioui): cum{min,max,prod,sum} fails "
                              "in JAX for complex64 on TPU")
    tol = None
    if f_jax == lax_control_flow.cumsum:
      tol = 0.1 if dtype == np.float16 else (0.5 if dtype == dtypes.bfloat16
                                             else tol)
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           atol=tol, rtol=tol)

  @primitive_harness.parameterized(primitive_harness.lax_top_k)
  def test_top_k(self, harness: primitive_harness.Harness):
    custom_assert = None
    k, dtype = harness.params["k"], harness.params["dtype"]
    if k > harness.params["shape"][-1] or k < 0:
      with self.assertRaisesRegex(ValueError, "k argument to top_k must be"):
        harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
      return
    if dtype in jtu.dtypes.complex:
      # TODO(necula): fix top_k complex bug on TPU
      if jtu.device_under_test() == "tpu":
        raise unittest.SkipTest("top_k complex on TPU raises different error")
      with self.assertRaisesRegex(RuntimeError,
                                  "Unimplemented: complex comparison"):
        harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
      return
    if dtype in jtu.dtypes.all_inexact:
      def custom_assert(result_jax, result_tf):
        assert len(result_jax) == len(result_tf)
        # TODO: TF and JAX sort [inf, nan] differently.
        first_arr_jax, first_arr_tf = result_jax[0], result_tf[0].numpy()
        if np.all(first_arr_jax == first_arr_tf):
          for arr_jax, arr_tf in zip(result_jax, result_tf):
            self.assertArraysEqual(arr_jax, arr_tf)
        else:
          mask_jax, mask_tf = np.isnan(first_arr_jax), np.isnan(first_arr_tf)
          self.assertArraysEqual(first_arr_jax[~ mask_jax],
                                 first_arr_tf[~ mask_tf])

    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           custom_assert=custom_assert)

  @primitive_harness.parameterized(primitive_harness.lax_sort)
  def test_sort(self, harness: primitive_harness.Harness):
    if (jtu.device_under_test() == "gpu" and
        len(harness.arg_descriptors) == 4 and
        not harness.params["is_stable"]):
      # TODO: fix the TF GPU test
      raise unittest.SkipTest("GPU tests are running TF on CPU")
    if jtu.device_under_test() == "tpu" and harness.params["dtype"] in jtu.dtypes.complex:
      raise unittest.SkipTest("JAX sort is not implemented on TPU for complex")
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_fft)
  def test_fft(self, harness: primitive_harness.Harness):
    if len(harness.params["fft_lengths"]) > 3:
      if jtu.device_under_test() == "gpu":
        with self.assertRaisesRegex(RuntimeError,
                                    "FFT only supports ranks 1-3"):
          harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
      else:
        raise unittest.SkipTest("TF does not support >3D FFTs.")
    elif (jtu.device_under_test() == "tpu" and
          len(harness.params["fft_lengths"]) > 1):
      # TODO(b/140351181): FFT is mostly unimplemented on TPU, even for JAX
      with self.assertRaisesRegex(RuntimeError,
                                  "only 1D FFT is currently supported."):
        harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
    else:
      tol = None if jtu.device_under_test() == "tpu" else 1e-3
      self.ConvertAndCompare(harness.dyn_fun,
                             *harness.dyn_args_maker(self.rng()),
                             atol=tol, rtol=tol)

  @primitive_harness.parameterized(primitive_harness.lax_linalg_cholesky)
  def test_cholesky(self, harness: primitive_harness.Harness):
    dtype = harness.params["dtype"]
    if dtype in [dtypes.bfloat16, np.float16]:
      raise unittest.SkipTest("Cholesky decomposition not supported for "
                              "(b)float16 in JAX.")
    operand = harness.dyn_args_maker(self.rng())[0]
    operand = np.matmul(operand, jnp.conj(np.swapaxes(operand, -1, -2)))
    tol = None
    # TODO(bchetioui): very high discrepancy in the float32/complex64 case
    if dtype in [np.float32, np.complex64]:
      tol = 1e-2
    # TODO(bchetioui): also high discrepancy in the float64/complex128 case
    elif dtype in [np.float64, np.complex128]:
      tol = 1e-11

    def custom_assert(result_jax, result_tf):
      # cholesky_p returns garbage in the strictly upper triangular part of the
      # result, so we can safely ignore that part.
      self.assertAllClose(jnp.tril(result_jax), result_tf, atol=tol)

    self.ConvertAndCompare(harness.dyn_fun, operand,
                           custom_assert=custom_assert,
                           always_custom_assert=True)

  @primitive_harness.parameterized(primitive_harness.lax_linalg_qr)
  def test_qr(self, harness: primitive_harness.Harness):
    # See jax.lib.lapack.geqrf for the list of compatible types

    dtype = harness.params["dtype"]
    dut = jtu.device_under_test()
    # These cases are not implemented in JAX
    if dtype in (jtu.dtypes.all_integer + [jnp.bfloat16]):
      unimplemented_jax = True
    elif dtype is np.complex64 and dut == "tpu":
      unimplemented_jax = True
    elif dtype is np.float16 and dut in ("cpu", "gpu"):
      unimplemented_jax = True
    else:
      unimplemented_jax = False

    if unimplemented_jax:
      raise unittest.SkipTest(f"QR not implemented in JAX for {dtype} on {dut}")

    # TODO: see https://github.com/google/jax/pull/3775#issuecomment-659407824.
    # - for now, the performance of the HLO QR implementation called when
    #   compiling with TF is expected to have worse performance than the
    #   custom calls made in JAX.
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           atol=1e-5, rtol=1e-5)

  @primitive_harness.parameterized(primitive_harness.lax_linalg_svd)
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_svd(self, harness: primitive_harness.Harness):
    if harness.params["dtype"] in [np.float16, dtypes.bfloat16]:
      if jtu.device_under_test() != "tpu":
        # Does not work in JAX
        with self.assertRaisesRegex(NotImplementedError, "Unsupported dtype"):
          harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
        return

    if harness.params["dtype"] in [np.complex64, np.complex128]:
      if jtu.device_under_test() == "tpu":
        # TODO: on JAX on TPU there is no SVD implementation for complex
        with self.assertRaisesRegex(RuntimeError,
                                    "Binary op compare with different element types"):
          harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
        return

    def _custom_assert(r_jax, r_tf, atol=1e-6, rtol=1e-6):
      def _reconstruct_operand(result, is_tf: bool):
        # Reconstructing operand as documented in numpy.linalg.svd (see
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
        s, u, v = result
        if is_tf:
          s = s.numpy()
          u = u.numpy()
          v = v.numpy()
        U = u[..., :s.shape[-1]]
        V = v[..., :s.shape[-1], :]
        S = s[..., None, :]
        return jnp.matmul(U * S, V), s.shape, u.shape, v.shape

      if harness.params["compute_uv"]:
        r_jax_reconstructed = _reconstruct_operand(r_jax, False)
        r_tf_reconstructed = _reconstruct_operand(r_tf, True)
        self.assertAllClose(r_jax_reconstructed, r_tf_reconstructed,
                            atol=atol, rtol=rtol)
      else:
        self.assertAllClose(r_jax, r_tf, atol=atol, rtol=rtol)

    tol = 1e-4
    custom_assert = partial(_custom_assert, atol=tol, rtol=tol)

    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           atol=tol, rtol=tol,
                           custom_assert=custom_assert,
                           always_custom_assert=True)

  @primitive_harness.parameterized(primitive_harness.lax_select_and_scatter_add)
  def test_select_and_scatter_add(self, harness: primitive_harness.Harness):
    if jtu.device_under_test() == "tpu" and not harness.params["run_on_tpu"]:
      raise unittest.SkipTest(
        "TODO: select_and_scatter on JAX on TPU only works when the parameters "
        "define 2 or more inactive dimensions"
      )
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_select_and_gather_add)
  @jtu.ignore_warning(category=UserWarning,
                      message="Using reduced precision for gradient.*")
  def test_select_and_gather_add(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_reduce_window)
  def test_reduce_window(self, harness: primitive_harness.Harness):
    dtype = harness.params['dtype']

    if (jtu.device_under_test() == 'tpu' and dtype is np.complex64):
      raise unittest.SkipTest(
          'TODO: JAX reduce_window on TPU does not handle complex64'
      )

    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_linalg_eig)
  def test_eig(self, harness: primitive_harness.Harness):
    operand = harness.dyn_args_maker(self.rng())[0]
    compute_left_eigenvectors = harness.params["compute_left_eigenvectors"]
    compute_right_eigenvectors = harness.params["compute_right_eigenvectors"]
    dtype = harness.params["dtype"]

    if jtu.device_under_test() != "cpu":
      raise unittest.SkipTest("eig only supported on CPU in JAX")

    if dtype in [np.float16, dtypes.bfloat16]:
      raise unittest.SkipTest("eig unsupported with (b)float16 in JAX")

    def custom_assert(result_jax, result_tf):
      result_tf = tuple(map(lambda e: e.numpy(), result_tf))
      inner_dimension = operand.shape[-1]
      # Test ported from tests.lax_test.testEig
      # Norm, adjusted for dimension and type.
      def norm(x):
        norm = np.linalg.norm(x, axis=(-2, -1))
        return norm / ((inner_dimension + 1) * jnp.finfo(dtype).eps)

      def check_right_eigenvectors(a, w, vr):
        self.assertTrue(
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
        self.assertAllClose(closest_diff, np.array(0., closest_diff.dtype),
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

    self.ConvertAndCompare(harness.dyn_fun, operand,
                           custom_assert=custom_assert)

  @primitive_harness.parameterized(primitive_harness.lax_linalg_eigh)
  def test_eigh(self, harness: primitive_harness.Harness):
    operand = harness.dyn_args_maker(self.rng())[0]
    lower = harness.params["lower"]
    # Make operand self-adjoint
    operand = (operand + np.conj(np.swapaxes(operand, -1, -2))) / 2
    # Make operand lower/upper triangular
    triangular_operand = np.tril(operand) if lower else np.triu(operand)
    dtype = harness.params["dtype"]

    if (dtype in [np.complex64, np.complex128] and
        jtu.device_under_test() == "tpu"):
      raise unittest.SkipTest("TODO: complex eigh not supported on TPU in JAX")

    def custom_assert(result_jax, result_tf):
      result_tf = tuple(map(lambda e: e.numpy(), result_tf))
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
        self.assertAllClose(np.matmul(a, vr) - w[..., None, :] * vr,
                            np.zeros(a.shape, dtype=vr.dtype),
                            atol=tol)

      def check_eigenvalue_is_in_array(eigenvalue, eigenvalues_array):
        tol = None
        if dtype in [dtypes.bfloat16, np.float32, np.complex64]:
          tol = 1e-3
        elif dtype in [np.float64, np.complex128]:
          tol = 1e-11
        closest_diff = min(abs(eigenvalues_array - eigenvalue))
        self.assertAllClose(closest_diff, np.array(0., closest_diff.dtype),
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

    self.ConvertAndCompare(harness.dyn_fun, triangular_operand,
                           custom_assert=custom_assert,
                           always_custom_assert=always_custom_assert)

  @primitive_harness.parameterized(primitive_harness.lax_linalg_lu)
  def test_lu(self, harness: primitive_harness.Harness):
    dtype = harness.params["dtype"]
    if dtype in [np.float16, dtypes.bfloat16]:
      raise unittest.SkipTest(
        f"LU is not implemented in JAX for dtype {dtype}.")
    tol = None
    if dtype in [np.float32, np.complex64]:
      if jtu.device_under_test() == "tpu":
        tol = 0.1
      else:
        tol = 1e-5
    if dtype in [np.float64, np.complex128]:
      tol = 1e-13
    operand, = harness.dyn_args_maker(self.rng())

    def custom_assert(result_jax, result_tf):
      lu, pivots, perm = tuple(map(lambda t: t.numpy(), result_tf))
      batch_dims = operand.shape[:-2]
      m, n = operand.shape[-2], operand.shape[-1]
      def _make_permutation_matrix(perm):
        result = []
        for idx in itertools.product(*map(range, operand.shape[:-1])):
          result += [0 if c != perm[idx] else 1 for c in range(m)]
        result = np.reshape(np.array(result, dtype=dtype), [*batch_dims, m, m])
        return result

      k = min(m, n)
      l = jnp.tril(lu, -1)[...,:, :k] + jnp.eye(m, k, dtype=dtype)
      u = jnp.triu(lu)[...,:k, :]
      p_mat = _make_permutation_matrix(perm)

      self.assertArraysEqual(lax.linalg.lu_pivots_to_permutation(pivots, m),
                             perm)
      self.assertAllClose(jnp.matmul(p_mat, operand), jnp.matmul(l, u),
                          atol=tol, rtol=tol)

    self.ConvertAndCompare(harness.dyn_fun, operand, atol=tol, rtol=tol,
                           custom_assert=custom_assert,
                           always_custom_assert=True)

  @primitive_harness.parameterized(
      primitive_harness.lax_linalg_triangular_solve)
  def test_triangular_solve(self, harness: primitive_harness.Harness):
    dtype = harness.params["dtype"]
    if dtype == np.float16 and jtu.device_under_test() == "gpu":
      raise unittest.SkipTest(
        f"Triangular solve is not implemented in JAX for dtype {dtype}")
    atol = rtol = None
    if dtype == np.float32:
      atol = rtol = 1e-5
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           atol=atol, rtol=rtol)

  @primitive_harness.parameterized(primitive_harness.lax_linear_solve)
  def test_linear_solve(self, harness: primitive_harness.Harness):
    a, b = harness.dyn_args_maker(self.rng())
    if harness.params["symmetric"]:
      a = a + a.T
    tol = None
    if (harness.params["dtype"] == np.float32 and
        jtu.device_under_test() == "tpu"):
      tol = 0.01

    self.ConvertAndCompare(harness.dyn_fun, a, b, atol=tol, rtol=tol)

  @primitive_harness.parameterized(primitive_harness.lax_unary_elementwise)
  def test_unary_elementwise(self, harness: primitive_harness.Harness):
    dtype = harness.params["dtype"]
    lax_name = harness.params["lax_name"]
    arg, = harness.dyn_args_maker(self.rng())
    custom_assert = None
    if lax_name == "digamma":
      # TODO(necula): fix bug with digamma/(f32|f16) on TPU
      if dtype in [np.float16, np.float32] and jtu.device_under_test() == "tpu":
        raise unittest.SkipTest("TODO: fix bug: nan vs not-nan")

      # In the bfloat16 case, TF and lax both return NaN in undefined cases.
      if not dtype is dtypes.bfloat16:
        # digamma is not defined at 0 and -1
        def custom_assert(result_jax, result_tf):
          # lax.digamma returns NaN and tf.math.digamma returns inf
          special_cases = (arg == 0.) | (arg == -1.)
          nr_special_cases = np.count_nonzero(special_cases)
          self.assertAllClose(np.full((nr_special_cases,), dtype(np.nan)),
                              result_jax[special_cases])
          self.assertAllClose(np.full((nr_special_cases,), dtype(np.inf)),
                              result_tf[special_cases])
          # non-special cases are equal
          self.assertAllClose(result_jax[~ special_cases],
                              result_tf[~ special_cases])
    if lax_name == "erf_inv":
      # TODO(necula): fix erf_inv bug on TPU
      if jtu.device_under_test() == "tpu":
        raise unittest.SkipTest("erf_inv bug on TPU: nan vs non-nan")
      # TODO: investigate: in the (b)float16 cases, TF and lax both return the
      # same result in undefined cases.
      if not dtype in [np.float16, dtypes.bfloat16]:
        # erf_inv is not defined for arg <= -1 or arg >= 1
        def custom_assert(result_jax, result_tf):  # noqa: F811
          # for arg < -1 or arg > 1
          # lax.erf_inv returns NaN; tf.math.erf_inv return +/- inf
          special_cases = (arg < -1.) | (arg > 1.)
          nr_special_cases = np.count_nonzero(special_cases)
          self.assertAllClose(np.full((nr_special_cases,), dtype(np.nan),
                                      dtype=dtype),
                              result_jax[special_cases])
          signs = np.where(arg[special_cases] < 0., -1., 1.)
          self.assertAllClose(np.full((nr_special_cases,),
                                      signs * dtype(np.inf), dtype=dtype),
                              result_tf[special_cases])
          # non-special cases are equal
          self.assertAllClose(result_jax[~ special_cases],
                              result_tf[~ special_cases])
    atol = None
    if jtu.device_under_test() == "gpu":
      # TODO(necula): revisit once we fix the GPU tests
      atol = 1e-3
    self.ConvertAndCompare(harness.dyn_fun, arg, custom_assert=custom_assert,
                           atol=atol)

  @primitive_harness.parameterized(primitive_harness.lax_comparators)
  def test_comparators(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_bitwise_not)
  def test_bitwise_not(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_zeros_like)
  def test_zeros_like(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_population_count)
  def test_population_count(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_argminmax)
  def test_argminmax(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_add_mul)
  def test_add_mul(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_min_max)
  def test_min_max(self, harness: primitive_harness.Harness):
    # TODO(bchetioui): discrepancies between TF & JAX when comparing with NaN;
    # JAX always returns NaN, while TF returns the value NaN is compared with.
    def custom_assert(result_jax, result_tf):
      mask = np.isnan(result_jax)
      self.assertAllClose(result_jax[~ mask], result_tf[~ mask])
    # TODO(bchetioui): figure out why we need always_custom_assert=True
    always_custom_assert = True
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           custom_assert=custom_assert,
                           always_custom_assert=always_custom_assert)

  @primitive_harness.parameterized(primitive_harness.lax_binary_elementwise)
  def test_binary_elementwise(self, harness):
    tol = None
    lax_name, dtype = harness.params["lax_name"], harness.params["dtype"]
    if lax_name in ("igamma", "igammac"):
      # TODO(necula): fix bug with igamma/f16
      if dtype in [np.float16, dtypes.bfloat16]:
        raise unittest.SkipTest("TODO: igamma(c) unsupported with (b)float16 in JAX")
      # TODO(necula): fix bug with igamma/f32 on TPU
      if dtype is np.float32 and jtu.device_under_test() == "tpu":
        raise unittest.SkipTest("TODO: fix bug: nan vs not-nan")
    arg1, arg2 = harness.dyn_args_maker(self.rng())
    custom_assert = None
    if lax_name == "igamma":
      # igamma is not defined when the first argument is <=0
      def custom_assert(result_jax, result_tf):
        # lax.igamma returns NaN when arg1 == arg2 == 0; tf.math.igamma returns 0
        special_cases = (arg1 == 0.) & (arg2 == 0.)
        nr_special_cases = np.count_nonzero(special_cases)
        self.assertAllClose(np.full((nr_special_cases,), np.nan, dtype=dtype),
                            result_jax[special_cases])
        self.assertAllClose(np.full((nr_special_cases,), 0., dtype=dtype),
                            result_tf[special_cases])
        # non-special cases are equal
        self.assertAllClose(result_jax[~ special_cases],
                            result_tf[~ special_cases])
    if lax_name == "igammac":
      # On GPU, tolerance also needs to be adjusted in compiled mode
      if dtype == np.float64 and jtu.device_under_test() == 'gpu':
        tol = 1e-14
      # igammac is not defined when the first argument is <=0
      def custom_assert(result_jax, result_tf):  # noqa: F811
        # lax.igammac returns 1. when arg1 <= 0; tf.math.igammac returns NaN
        special_cases = (arg1 <= 0.) | (arg2 <= 0)
        nr_special_cases = np.count_nonzero(special_cases)
        self.assertAllClose(np.full((nr_special_cases,), 1., dtype=dtype),
                            result_jax[special_cases])
        self.assertAllClose(np.full((nr_special_cases,), np.nan, dtype=dtype),
                            result_tf[special_cases])
        # On CPU, tolerance only needs to be adjusted in eager & graph modes
        tol = None
        if dtype == np.float64:
          tol = 1e-14

        # non-special cases are equal
        self.assertAllClose(result_jax[~ special_cases],
                            result_tf[~ special_cases], atol=tol, rtol=tol)
    self.ConvertAndCompare(harness.dyn_fun, arg1, arg2,
                           custom_assert=custom_assert, atol=tol, rtol=tol)

  @primitive_harness.parameterized(primitive_harness.lax_binary_elementwise_logical)
  def test_binary_elementwise_logical(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_broadcast_in_dim)
  def test_broadcast_in_dim(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_broadcast)
  def test_broadcast(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_betainc)
  def test_betainc(self, harness: primitive_harness.Harness):
    dtype = harness.params["dtype"]
    # TODO: https://www.tensorflow.org/api_docs/python/tf/math/betainc only
    # supports float32/64 tests.
    tol = None
    if dtype is np.float64:
      tol = 1e-14

    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           atol=tol, rtol=tol)

  # TODO(necula): combine tests that are identical except for the harness
  # wait until we get more experience with using harnesses.
  @primitive_harness.parameterized(primitive_harness.lax_shift_left)
  def test_shift_left(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_shift_right_logical)
  def test_shift_right_logical(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_shift_right_arithmetic)
  def test_shift_right_arithmetic(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_slice)
  def test_slice(self, harness):
    # JAX.slice rejects negative indices; check, and skip jax2tf
    if any(si < 0 or si >= sh or li < 0 or li > sh
           for sh, si, li in zip(harness.params["shape"],
                                 harness.params["start_indices"],
                                 harness.params["limit_indices"])):
      with self.assertRaisesRegex(TypeError, ""):
        harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
    else:
      self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_conj)
  def test_conj(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_dynamic_slice)
  def test_dynamic_slice(self, harness):
    # JAX.dynamic_slice rejects slice sizes too big; check this, and skip jax2tf
    args = harness.dyn_args_maker(self.rng())
    if any(li - si < 0 or li - si >= sh
           for sh, si, li in zip(harness.params["shape"],
                                 harness.params["start_indices"],
                                 harness.params["limit_indices"])):
      with self.assertRaisesRegex(TypeError, ""):
        harness.dyn_fun(*args)
      return

    self.ConvertAndCompare(harness.dyn_fun, *args)

  @primitive_harness.parameterized(primitive_harness.lax_dynamic_update_slice)
  def test_dynamic_update_slice(self, harness):
    # JAX.dynamic_update_slice rejects update slices too big; check, and skip jax2tf
    if any(ush > sh
           for sh, ush in zip(harness.params["shape"],
                              harness.params["update_shape"])):
      with self.assertRaisesRegex(TypeError, ""):
        harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
    else:
      self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_squeeze)
  def test_squeeze(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_dot_general)
  def test_dot_general(self, harness: primitive_harness.Harness):
    tol, dtype = None, harness.params["dtype"]
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
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           atol=tol, rtol=tol)

  @primitive_harness.parameterized(primitive_harness.lax_clamp)
  def test_clamp(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_concatenate)
  def test_concatenate(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_conv_general_dilated)
  def test_conv_general_dilated(self, harness: primitive_harness.Harness):
    dtype, device = harness.params["dtype"], jtu.device_under_test()
    if device == "gpu" and dtype in [np.complex64, np.complex128]:
      raise unittest.SkipTest("TODO: crash on GPU in TF")

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
    if harness.name == "_tf_conversion_path_3d_lhs=float32[1,4,28,28,1]_rhs=float32[2,3,3,1,16]_windowstrides=(1,1,1)_padding=VALID_lhsdilation=(1,1,1)_rhsdilation=(1,1,2)_dimensionnumbers=('NDHWC','DHWIO','NDHWC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=False":
      raise unittest.SkipTest("TODO: known but unidentified bug in compiled "
                              "mode")

    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
        atol=tol, rtol=tol, enable_xla=harness.params["enable_xla"])

  @primitive_harness.parameterized(primitive_harness.disable_xla)
  def test_disable_xla(self, harness: primitive_harness.Harness):
    with self.assertRaisesRegex(NotImplementedError,
                                "Call to pad can only be converted through "
                                "TFXLA, but XLA is disabled"):
      self.ConvertAndCompare(harness.dyn_fun,
          *harness.dyn_args_maker(self.rng()), enable_xla=False)

  @primitive_harness.parameterized(primitive_harness.lax_gather)
  def test_gather(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_scatter)
  def test_scatter(self, harness: primitive_harness.Harness):
    f_name = harness.params['f_lax'].__name__
    dtype = harness.params['dtype']

    if jtu.device_under_test() == 'tpu':
      if dtype is np.complex64 and f_name in ['scatter_min', 'scatter_max']:
          raise unittest.SkipTest(f"TODO: complex {f_name} on TPU fails in JAX")

    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

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
    for op in INDEX))
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

  @primitive_harness.parameterized(primitive_harness.random_gamma)
  def test_random_gamma(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           rtol=1e-5)

  @primitive_harness.parameterized(primitive_harness.random_split)
  def test_random_split(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  def test_stop_gradient(self):
    f = jax2tf.convert(lax.stop_gradient)
    self.assertEqual(f(tf.ones([])), 1.)

  # test_bfloat16_constant checks that https://github.com/google/jax/issues/3942 is
  # fixed
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

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
