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

See the back_compat_test_util module docstring for how to setup and update
these tests.
"""
import dataclasses
from functools import partial
import itertools
import math
from typing import Callable, Optional, Sequence

from absl.testing import absltest, parameterized

import numpy as np

import jax
from jax import config
from jax import lax
from jax.experimental import jax2tf
from jax.experimental.jax2tf import jax_export
from jax.experimental.jax2tf.tests import back_compat_test_util as bctu

from jax.experimental.jax2tf.tests.back_compat_testdata import cpu_ducc_fft
from jax.experimental.jax2tf.tests.back_compat_testdata import cuda_eigh_cusolver_syev
from jax.experimental.jax2tf.tests.back_compat_testdata import cpu_eigh_lapack_syev
from jax.experimental.jax2tf.tests.back_compat_testdata import cuda_qr_cusolver_geqrf
from jax.experimental.jax2tf.tests.back_compat_testdata import cpu_qr_lapack_geqrf
from jax.experimental.jax2tf.tests.back_compat_testdata import cuda_threefry2x32
from jax.experimental.jax2tf.tests.back_compat_testdata import tf_call_tf_function
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Eigh
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Lu
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_ApproxTopK
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Qr
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_Sharding
from jax.experimental.jax2tf.tests.back_compat_testdata import tpu_stablehlo_dynamic_reduce_window
from jax.experimental.jax2tf.tests.back_compat_testdata import stablehlo_dynamic_rng_bit_generator

from jax.experimental import pjit
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax._src.lib import xla_extension
from jax._src import test_util as jtu

config.parse_flags_with_absl()


class CompatTest(bctu.CompatTestBase):
  def test_dummy(self):
    # Tests the testing mechanism. Let this test run on all platforms
    dummy_data = self.load_testdata(bctu.dummy_data_dict)
    platform_dummy_data = dataclasses.replace(
        dummy_data, platform=self.default_jax_backend())
    self.run_one_test(jnp.sin, platform_dummy_data)

  def test_detect_different_output(self):
    # Test the detection mechanism. Let this test run on all platforms
    dummy_data = self.load_testdata(bctu.dummy_data_dict)
    platform_dummy_data = dataclasses.replace(
        dummy_data,
        platform=self.default_jax_backend(),
        expected_outputs=(np.array(2.0, dtype=np.float32),))
    with self.assertRaisesRegex(AssertionError, "Not equal to tolerance"):
      self.run_one_test(jnp.sin, platform_dummy_data)

  def test_detect_different_custom_calls(self):
    # Test the detection mechanism. Let this test run on all platforms
    dummy_data = self.load_testdata(bctu.dummy_data_dict)
    platform_dummy_data = dataclasses.replace(
        dummy_data,
        platform=self.default_jax_backend(),
        custom_call_targets=["missing"])
    with self.assertRaisesRegex(AssertionError, "Lists differ"):
      self.run_one_test(jnp.sin, platform_dummy_data)

  def test_custom_call_coverage(self):
    """Tests that the back compat tests cover all the targets declared stable."""
    targets_to_cover = set(jax_export._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE)
    # Add here all the testdatas that should cover the targets guaranteed
    # stable
    covering_testdatas = [
        cpu_ducc_fft.data_2023_03_17, cpu_ducc_fft.data_2023_06_14,
        cpu_eigh_lapack_syev.data_2023_03_17,
        cpu_qr_lapack_geqrf.data_2023_03_17, cuda_threefry2x32.data_2023_03_15,
        cuda_qr_cusolver_geqrf.data_2023_03_18, cuda_eigh_cusolver_syev.data_2023_03_17,
        tf_call_tf_function.data_2023_06_02,  # This is tested in back_compat_tf_test.py
        tpu_Eigh.data, tpu_Lu.data_2023_03_21, tpu_Qr.data_2023_03_17,
        tpu_Sharding.data_2023_03_16, tpu_ApproxTopK.data_2023_04_17,
        tpu_ApproxTopK.data_2023_05_16,
        tpu_stablehlo_dynamic_reduce_window.data_unary_2023_06_17,
        tpu_stablehlo_dynamic_reduce_window.data_variadic_2023_06_17,
        stablehlo_dynamic_rng_bit_generator.data_2023_06_17,]
    covering_testdatas = itertools.chain(
        *[self.load_testdata_nested(d) for d in covering_testdatas])
    covered_targets = set()
    for data in covering_testdatas:
      self.assertIsInstance(data, bctu.CompatTestData)
      covered_targets = covered_targets.union(data.custom_call_targets)

    covered_targets = covered_targets.union({
      # TODO(necula): add tests for eig on CPU
      "lapack_sgeev", "lapack_dgeev", "lapack_cgeev", "lapack_zgeev",
      # TODO(necula): add tests for qr on CPU in a separate change.
      "lapack_cpotrf", "lapack_dpotrf", "lapack_spotrf", "lapack_zpotrf",
      # TODO(necula): add tests for svd on CPU
      "lapack_sgesdd", "lapack_dsesdd", "lapack_cgesdd", "lapack_zgesdd",
    })
    not_covered = targets_to_cover.difference(covered_targets)
    self.assertEmpty(not_covered)

  def test_ducc_fft(self):
    def func(x):
      return lax.fft(x, fft_type="fft", fft_lengths=(4,))

    # An old lowering, with ducc_fft. We keep it for 6 months.
    data = self.load_testdata(cpu_ducc_fft.data_2023_03_17)
    # We have changed the lowering for fft, do not compare with current.
    self.run_one_test(func, data, compare_with_current=False)

    # A newer lowering, with dynamic_ducc_fft.
    data = self.load_testdata(cpu_ducc_fft.data_2023_06_14)
    self.run_one_test(func, data)

  @staticmethod
  def eigh_input(shape, dtype):
    # In order to keep inputs small, we construct the input programmatically
    operand = jnp.reshape(jnp.arange(math.prod(shape), dtype=dtype), shape)
    # Make operand self-adjoint
    operand = (operand + jnp.conj(jnp.swapaxes(operand, -1, -2))) / 2.
    return operand

  @staticmethod
  def eigh_harness(shape, dtype):
    operand = CompatTest.eigh_input(shape, dtype)
    return lax.linalg.eigh(jnp.tril(operand), lower=True, symmetrize_input=False)

  def check_eigh_results(self, operand, res_now, res_expected, *,
                         rtol, atol=None):
    v_now, w_now = res_now
    _, w_expected = res_expected
    n, m = operand.shape
    assert n == m
    assert v_now.shape == operand.shape
    assert w_now.shape == (n,)
    self.assertLessEqual(
        np.linalg.norm(np.eye(n) - np.matmul(np.conj(np.swapaxes(v_now, -1, -2)), v_now)),
        rtol)
    # w_now : f64[n] while v_now: c128[n, n]
    w_now_like_v = w_now[np.newaxis, :].astype(v_now.dtype)
    self.assertLessEqual(
        np.linalg.norm(np.matmul(operand, v_now) - w_now_like_v * v_now),
        rtol * np.linalg.norm(operand))
    self.assertAllClose(w_expected, w_now, rtol=rtol, atol=atol)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  def test_cpu_eigh_lapack_syevd(self, dtype_name="f32"):
    # For lax.linalg.eigh
    if not config.jax_enable_x64 and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    size = 8
    operand = CompatTest.eigh_input((size, size), dtype)
    func = lambda: CompatTest.eigh_harness((8, 8), dtype)
    data = self.load_testdata(cpu_eigh_lapack_syev.data_2023_03_17[dtype_name])
    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    atol = dict(f32=1e-4, f64=1e-12, c64=1e-4, c128=1e-12)[dtype_name]
    self.run_one_test(func, data, rtol=rtol, atol=atol,
                      check_results=partial(self.check_eigh_results, operand))

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}_{variant}",
           dtype_name=dtype_name, variant=variant)
      for dtype_name in ("f32", "f64")
      # We use different custom calls for sizes <= 32
      for variant in ["syevj", "syevd"])
  def test_cuda_eigh_cusolver_syev(self, dtype_name="f32", variant="syevj"):
    # For lax.linalg.eigh
    dtype = dict(f32=np.float32, f64=np.float64)[dtype_name]
    size = dict(syevj=8, syevd=36)[variant]
    rtol = dict(f32=1e-3, f64=1e-5)[dtype_name]
    atol = dict(f32=1e-2, f64=1e-10)[dtype_name]
    operand = CompatTest.eigh_input((size, size), dtype)
    func = lambda: CompatTest.eigh_harness((size, size), dtype)
    data = self.load_testdata(cuda_eigh_cusolver_syev.data_2023_03_17[f"{dtype_name}_{variant}"])
    self.run_one_test(func, data, rtol=rtol, atol=atol,
                      check_results=partial(self.check_eigh_results, operand))

  def test_tpu_Eigh(self):
    self.skipTest(
        "TODO(b/280668311): Change input matrix to not be ill-conditioned."
    )
    # For lax.linalg.eigh
    shape = (8, 8)
    dtype = np.float32
    operand = CompatTest.eigh_input(shape, dtype)
    func = lambda: CompatTest.eigh_harness(shape, dtype)
    data = self.load_testdata(tpu_Eigh.data)
    self.run_one_test(func, data, rtol=1e-3,
                      check_results=partial(self.check_eigh_results, operand))

  @staticmethod
  def qr_harness(shape, dtype):
    # In order to keep inputs small, we construct the input programmatically
    operand = jnp.reshape(jnp.arange(math.prod(shape), dtype=dtype), shape)
    return lax.linalg.qr(operand, full_matrices=True)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}", dtype_name=dtype_name)
      for dtype_name in ("f32", "f64", "c64", "c128"))
  def test_cpu_qr_lapack_geqrf(self, dtype_name="f32"):
    # For lax.linalg.qr
    if not config.jax_enable_x64 and dtype_name in ["f64", "c128"]:
      self.skipTest("Test disabled for x32 mode")

    dtype = dict(f32=np.float32, f64=np.float64,
                 c64=np.complex64, c128=np.complex128)[dtype_name]
    func = lambda: CompatTest.qr_harness((3, 3), dtype)
    data = self.load_testdata(cpu_qr_lapack_geqrf.data_2023_03_17[dtype_name])
    rtol = dict(f32=1e-3, f64=1e-5, c64=1e-3, c128=1e-5)[dtype_name]
    self.run_one_test(func, data, rtol=rtol)

  @parameterized.named_parameters(
      dict(testcase_name=f"_dtype={dtype_name}_{batched}",
           dtype_name=dtype_name, batched=batched)
      for dtype_name in ("f32",)
      # For batched qr we use cublas_geqrf_batched
      for batched in ("batched", "unbatched"))
  def test_cuda_qr_cusolver_geqrf(self, dtype_name="f32", batched="unbatched"):
    # For lax.linalg.qr
    dtype = dict(f32=np.float32, f64=np.float64)[dtype_name]
    rtol = dict(f32=1e-3, f64=1e-5)[dtype_name]
    shape = dict(batched=(2, 3, 3), unbatched=(3, 3))[batched]
    func = lambda: CompatTest.qr_harness(shape, dtype)
    data = self.load_testdata(cuda_qr_cusolver_geqrf.data_2023_03_18[batched])
    self.run_one_test(func, data, rtol=rtol)

  def test_tpu_Qr(self):
    # For lax.linalg.qr
    func = lambda: CompatTest.qr_harness((3, 3), np.float32)
    data = self.load_testdata(tpu_Qr.data_2023_03_17)
    self.run_one_test(func, data, rtol=1e-3)

  @staticmethod
  def lu_harness(shape, dtype):
    operand = jnp.reshape(jnp.arange(math.prod(shape), dtype=dtype), shape)
    return lax.linalg.lu(operand)

  def test_tpu_Lu(self):
    # For lax.linalg.lu
    func = lambda: CompatTest.lu_harness((3, 3), np.float32)
    data = self.load_testdata(tpu_Lu.data_2023_03_21)
    self.run_one_test(func, data, rtol=1e-3)

  def test_approx_top_k(self):
    def func():
      x = np.array([3.0, 1.0, 4.0, 2.0, 5.0, 6.0, 7.0])
      y = lax.approx_max_k(x, 3)
      z = lax.approx_max_k(x, 3)
      return y + z
    data = self.load_testdata(tpu_ApproxTopK.data_2023_05_16)
    self.run_one_test(func, data)

  def test_cu_threefry2x32(self):
    def func(x):
      return jax.random.uniform(x, (2, 4), dtype=np.float32)

    data = self.load_testdata(cuda_threefry2x32.data_2023_03_15)
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

    data = self.load_testdata(tpu_Sharding.data_2023_03_16)
    with mesh:
      self.run_one_test(func, data)

  def test_tpu_stablehlo_dynamic_reduce_window_unary(self):
    # stablehlo.dynamic_reduce_window is used temporarily on TPU for a
    # reduce window with dynamic shapes.
    # See https://github.com/openxla/stablehlo/issues/1258 for the long term.
    # The inputs are already in the test data, here only for readability.
    shape = (3, 4)
    _ = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    def func(x):
      return jnp.cumsum(x, axis=0)

    data = self.load_testdata(tpu_stablehlo_dynamic_reduce_window.data_unary_2023_06_17)
    self.run_one_test(
        func, data,
        polymorphic_shapes=("b, ...",))

  def test_tpu_stablehlo_dynamic_reduce_window_variadic(self):
    # stablehlo.dynamic_reduce_window is used temporarily on TPU for a
    # reduce window with dynamic shapes.
    # See https://github.com/openxla/stablehlo/issues/1258 for the long term.
    # The inputs are already in the test data, here only for readability.
    shape = (3, 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    y = 100 + np.arange(math.prod(shape), dtype=np.int32).reshape(shape)
    _ = (x, y)
    def func(x, y):  # x: f32[b, 2] y: i32[b, 2]
      return lax.reduce_window(
          (x, y), (np.array(1., np.float32), np.array(2, np.int32)),
          lambda xy0, xy1: (lax.add(xy0[0], xy1[0]),
                            lax.sub(xy0[1], xy1[1])),
          (2, x.shape[0]), (1, 1), "VALID")

    data = self.load_testdata(tpu_stablehlo_dynamic_reduce_window.data_variadic_2023_06_17)
    self.run_one_test(
        func, data,
        polymorphic_shapes=("b, ...", "b, ..."))

  def test_stablehlo_dynamic_rbg_bit_generator(self):
    # stablehlo.dynamic_rbg_bit_generator is used temporarily for a
    # rbg_bit_generator with dynamic shapes.
    # See https://github.com/openxla/stablehlo/issues/1344 for the long term.
    key = np.arange(42, 42+4, dtype=np.uint32)
    a_shape = (2, 3)
    a = np.arange(math.prod(a_shape), dtype=np.float32).reshape(a_shape)
    inputs = (key, a)
    del inputs  # already in the test data, here only for readability.

    def func(key, a):  # a is only used for its shape
      return jax.random.key_data(jax.random.split(key, a.shape[0] * a.shape[1]))

    # Note that the test currently checks that the generated sequence is the
    # same. According to the StableHLO spec: "The output is guaranteed to be
    # deterministic function of initial_state, but it is not guaranteed to be
    # deterministic between implementations"
    # See https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rng_bit_generator
    # This test will fail when the implementation changes. We expect this to
    # be rare, and most users may expect the RNG sequence to be the same
    # upon reloading of a saved model.
    # In case of an intended change in behavior we will have the option to
    # replace this strict check with something else.
    data = self.load_testdata(stablehlo_dynamic_rng_bit_generator.data_2023_06_17)

    prev_default_prng_impl = jax.config.jax_default_prng_impl
    try:
      jax.config.update("jax_default_prng_impl", "unsafe_rbg")

      self.run_one_test(func, data, polymorphic_shapes=(None, "b0, b1"))
    finally:
      jax.config.update("jax_default_prng_impl", prev_default_prng_impl)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
