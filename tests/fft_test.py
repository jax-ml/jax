# Copyright 2019 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import unittest

import numpy as onp

from absl.testing import absltest
from absl.testing import parameterized

from jax import lax
from jax import numpy as np
from jax import test_util as jtu

from jax.config import config
config.parse_flags_with_absl()


float_dtypes = [onp.float32, onp.float64]
# TODO(b/144573940): onp.complex128 isn't supported by XLA, and the JAX
# implementation casts to complex64.
complex_dtypes = [onp.complex64]
inexact_dtypes = float_dtypes + complex_dtypes
int_dtypes = [onp.int32, onp.int64]
bool_dtypes = [onp.bool_]
real_dtypes = float_dtypes + int_dtypes + bool_dtypes
all_dtypes = real_dtypes + complex_dtypes


def _get_fftn_test_axes(shape):
  axes = [[]]
  ndims = len(shape)
  # XLA's FFT op only supports up to 3 innermost dimensions.
  if ndims <= 3:
    axes.append(None)
  for naxes in range(1, min(ndims, 3) + 1):
    axes.extend(itertools.combinations(range(ndims), naxes))
  for index in range(1, ndims + 1):
    axes.append((-index,))
  return axes


def _get_fftn_func(module, inverse, real):
  if inverse:
    return _irfft_with_zeroed_inputs(module.irfftn) if real else module.ifftn
  else:
    return module.rfftn if real else module.fftn


def _irfft_with_zeroed_inputs(irfft_fun):
  # irfft isn't defined on the full domain of inputs, so in order to have a
  # well defined derivative on the whole domain of the function, we zero-out
  # the imaginary part of the first and possibly the last elements.
  def wrapper(z, axes):
    return irfft_fun(_zero_for_irfft(z, axes), axes=axes)
  return wrapper


def _zero_for_irfft(z, axes):
  if axes is not None and not axes:
    return z
  axis = z.ndim - 1 if axes is None else axes[-1]
  try:
    size = z.shape[axis]
  except IndexError:
    return z  # only if axis is invalid, as occurs in some tests
  if size % 2:
    parts = [lax.slice_in_dim(z.real, 0, 1, axis=axis).real,
             lax.slice_in_dim(z.real, 1, size - 1, axis=axis),
             lax.slice_in_dim(z.real, size - 1, size, axis=axis).real]
  else:
    parts = [lax.slice_in_dim(z.real, 0, 1, axis=axis).real,
             lax.slice_in_dim(z.real, 1, size, axis=axis)]
  return np.concatenate(parts, axis=axis)


class FftTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inverse={}_real={}_shape={}_axes={}".format(
          inverse, real, jtu.format_shape_dtype_string(shape, dtype), axes),
       "axes": axes, "shape": shape, "dtype": dtype,
       "rng_factory": rng_factory, "inverse": inverse, "real": real}
      for inverse in [False, True]
      for real in [False, True]
      for rng_factory in [jtu.rand_default]
      for dtype in (real_dtypes if real and not inverse else all_dtypes)
      for shape in [(10,), (10, 10), (9,), (2, 3, 4), (2, 3, 4, 5)]
      for axes in _get_fftn_test_axes(shape)))
  def testFftn(self, inverse, real, shape, dtype, axes, rng_factory):
    rng = rng_factory()
    args_maker = lambda: (rng(shape, dtype),)
    np_op = _get_fftn_func(np.fft, inverse, real)
    onp_op = _get_fftn_func(onp.fft, inverse, real)
    np_fn = lambda a: np_op(a, axes=axes)
    onp_fn = lambda a: onp_op(a, axes=axes) if axes is None or axes else a
    self._CheckAgainstNumpy(onp_fn, np_fn, args_maker, check_dtypes=True,
                            tol=1e-4)
    self._CompileAndCheck(np_fn, args_maker, check_dtypes=True)
    # Test gradient for differentiable types.
    if dtype in (float_dtypes if real and not inverse else inexact_dtypes):
      # TODO(skye): can we be more precise?
      tol = 1e-1
      jtu.check_grads(np_fn, args_maker(), order=2, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inverse={}_real={}".format(inverse, real),
       "inverse": inverse, "real": real}
      for inverse in [False, True]
      for real in [False, True]))
  def testFftnErrors(self, inverse, real):
    rng = jtu.rand_default()
    name = 'fftn'
    if real:
      name = 'r' + name
    if inverse:
      name = 'i' + name
    func = _get_fftn_func(np.fft, inverse, real)
    self.assertRaisesRegex(
        ValueError,
        "jax.np.fft.{} only supports 1D, 2D, and 3D FFTs. "
        "Got axes None with input rank 4.".format(name),
        lambda: func(rng([2, 3, 4, 5], dtype=onp.float64), axes=None))
    self.assertRaisesRegex(
        ValueError,
        "jax.np.fft.{} does not support repeated axes. Got axes \\[1, 1\\].".format(name),
        lambda: func(rng([2, 3], dtype=onp.float64), axes=[1, 1]))
    self.assertRaises(
        ValueError, lambda: func(rng([2, 3], dtype=onp.float64), axes=[2]))
    self.assertRaises(
        ValueError, lambda: func(rng([2, 3], dtype=onp.float64), axes=[-3]))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inverse={}_size={}_axis={}".format(
          inverse, jtu.format_shape_dtype_string([size], dtype), axis),
          "axis": axis, "size": size, "dtype": dtype, "rng_factory": rng_factory,
          "inverse": inverse}
      for inverse in [False, True]
      for rng_factory in [jtu.rand_default]
      for dtype in all_dtypes
      for size in [10]
      for axis in [-1, 0]))
  def testFft(self, inverse, size, dtype, axis, rng_factory):
    rng = rng_factory()
    args_maker = lambda: (rng([size], dtype),)
    np_op = np.fft.ifft if inverse else np.fft.fft
    onp_op = onp.fft.ifft if inverse else onp.fft.fft
    np_fn = lambda a: np_op(a, axis=axis)
    onp_fn = lambda a: onp_op(a, axis=axis)
    # Numpy promotes to complex128 aggressively.
    self._CheckAgainstNumpy(onp_fn, np_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(np_fn, args_maker, check_dtypes=True)
    # Test gradient for differentiable types.
    if dtype in inexact_dtypes:
        tol = 0.15  # TODO(skye): can we be more precise?
        jtu.check_grads(np_fn, args_maker(), order=1, atol=tol, rtol=tol)
        jtu.check_grads(np_fn, args_maker(), order=2, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inverse={}".format(inverse),
       "inverse": inverse}
      for inverse in [False, True]))
  def testFftErrors(self, inverse):
    rng = jtu.rand_default()
    name = 'ifft' if inverse else 'fft'
    func = np.fft.ifft if inverse else np.fft.fft

    self.assertRaisesRegex(
      ValueError,
      "jax.np.fft.{} does not support multiple axes. "
      "Please use jax.np.fft.{}n. "
      "Got axis \\[1, 1\\].".format(name, name),
      lambda: func(rng([2, 3], dtype=onp.float64), axis=[1, 1])
    )
    self.assertRaisesRegex(
      ValueError,
      "jax.np.fft.{} does not support multiple axes. "
      "Please use jax.np.fft.{}n. "
      "Got axis \\[1, 1\\].".format(name, name),
      lambda: func(rng([2, 3], dtype=onp.float64), axis=(1, 1))
    )
    self.assertRaises(
        ValueError, lambda: func(rng([2, 3], dtype=onp.float64), axis=[2]))
    self.assertRaises(
        ValueError, lambda: func(rng([2, 3], dtype=onp.float64), axis=[-3]))

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_inverse={}_shape={}_axis={}".format(
      inverse, jtu.format_shape_dtype_string(shape, dtype), axes),
      "axes": axes, "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
      "inverse": inverse}
    for inverse in [False, True]
    for rng_factory in [jtu.rand_default]
    for dtype in all_dtypes
    for shape in [(16, 8, 4, 8), (16, 8, 4, 8, 4)]
    for axes in [(-2, -1), (0, 1), (1, 3), (-1, 2)]))
  def testFft2(self, inverse, shape, dtype, axes, rng_factory):
    rng = rng_factory()
    args_maker = lambda: (rng(shape, dtype),)
    np_op = np.fft.ifft2 if inverse else np.fft.fft2
    onp_op = onp.fft.ifft2 if inverse else onp.fft.fft2
    np_fn = lambda a: np_op(a, axes=axes)
    onp_fn = lambda a: onp_op(a, axes=axes)
    # Numpy promotes to complex128 aggressively.
    self._CheckAgainstNumpy(onp_fn, np_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(np_fn, args_maker, check_dtypes=True)
    # Test gradient for differentiable types.
    if dtype in inexact_dtypes:
      tol = 0.15  # TODO(skye): can we be more precise?
      jtu.check_grads(np_fn, args_maker(), order=2, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_inverse={}".format(inverse),
     "inverse": inverse}
    for inverse in [False, True]))
  def testFft2Errors(self, inverse):
    rng = jtu.rand_default()
    name = 'ifft2' if inverse else 'fft2'
    func = np.fft.ifft2 if inverse else np.fft.fft2

    self.assertRaisesRegex(
      ValueError,
      "jax.np.fft.{} only supports 2 axes. "
      "Got axes = \\[0\\].".format(name, name),
      lambda: func(rng([2, 3], dtype=onp.float64), axes=[0])
    )
    self.assertRaisesRegex(
      ValueError,
      "jax.np.fft.{} only supports 2 axes. "
      "Got axes = \\[0, 1, 2\\].".format(name, name),
      lambda: func(rng([2, 3, 3], dtype=onp.float64), axes=(0, 1, 2))
    )
    self.assertRaises(
      ValueError, lambda: func(rng([2, 3], dtype=onp.float64), axes=[2, 3]))
    self.assertRaises(
      ValueError, lambda: func(rng([2, 3], dtype=onp.float64), axes=[-3, -4]))

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_size={}_d={}".format(
      jtu.format_shape_dtype_string([size], dtype), d),
      "dtype": dtype, "size": size, "rng_factory": rng_factory, "d": d}
    for rng_factory in [jtu.rand_default]
    for dtype in all_dtypes
    for size in [9, 10, 101, 102]
    for d in [0.1, 2.]))
  def testFftfreq(self, size, d, dtype, rng_factory):
    rng = rng_factory()
    args_maker = lambda: (rng([size], dtype),)
    np_op = np.fft.fftfreq
    onp_op = onp.fft.fftfreq
    np_fn = lambda a: np_op(size, d=d)
    onp_fn = lambda a: onp_op(size, d=d)
    # Numpy promotes to complex128 aggressively.
    self._CheckAgainstNumpy(onp_fn, np_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(np_fn, args_maker, check_dtypes=True)
    # Test gradient for differentiable types.
    if dtype in inexact_dtypes:
      tol = 0.15  # TODO(skye): can we be more precise?
      jtu.check_grads(np_fn, args_maker(), order=2, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_n={}".format(n),
     "n": n}
    for n in [[0,1,2]]))
  def testFftfreqErrors(self, n):
    name = 'fftfreq'
    func = np.fft.fftfreq
    self.assertRaisesRegex(
      ValueError,
      "The n argument of jax.np.fft.{} only takes an int. "
      "Got n = \\[0, 1, 2\\].".format(name),
      lambda: func(n=n)
    )
    self.assertRaisesRegex(
      ValueError,
      "The d argument of jax.np.fft.{} only takes a single value. "
      "Got d = \\[0, 1, 2\\].".format(name),
      lambda: func(n=10, d=n)
    )

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_size={}_d={}".format(
      jtu.format_shape_dtype_string([size], dtype), d),
      "dtype": dtype, "size": size, "rng_factory": rng_factory, "d": d}
    for rng_factory in [jtu.rand_default]
    for dtype in all_dtypes
    for size in [9, 10, 101, 102]
    for d in [0.1, 2.]))
  def testRfftfreq(self, size, d, dtype, rng_factory):
    rng = rng_factory()
    args_maker = lambda: (rng([size], dtype),)
    np_op = np.fft.rfftfreq
    onp_op = onp.fft.rfftfreq
    np_fn = lambda a: np_op(size, d=d)
    onp_fn = lambda a: onp_op(size, d=d)
    # Numpy promotes to complex128 aggressively.
    self._CheckAgainstNumpy(onp_fn, np_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(np_fn, args_maker, check_dtypes=True)
    # Test gradient for differentiable types.
    if dtype in inexact_dtypes:
      tol = 0.15  # TODO(skye): can we be more precise?
      jtu.check_grads(np_fn, args_maker(), order=2, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_n={}".format(n),
     "n": n}
    for n in [[0, 1, 2]]))
  def testRfftfreqErrors(self, n):
    name = 'rfftfreq'
    func = np.fft.rfftfreq
    self.assertRaisesRegex(
      ValueError,
      "The n argument of jax.np.fft.{} only takes an int. "
      "Got n = \\[0, 1, 2\\].".format(name),
      lambda: func(n=n)
    )
    self.assertRaisesRegex(
      ValueError,
      "The d argument of jax.np.fft.{} only takes a single value. "
      "Got d = \\[0, 1, 2\\].".format(name),
      lambda: func(n=10, d=n)
    )

if __name__ == "__main__":
  absltest.main()
