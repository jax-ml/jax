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
all_dtypes = float_dtypes + complex_dtypes + int_dtypes + bool_dtypes


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


class FftTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inverse={}_shape={}_axes={}".format(
          inverse, jtu.format_shape_dtype_string(shape, dtype), axes),
       "axes": axes, "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
       "inverse": inverse}
      for inverse in [False, True]
      for rng_factory in [jtu.rand_default]
      for dtype in all_dtypes
      for shape in [(10,), (10, 10), (2, 3, 4), (2, 3, 4, 5)]
      for axes in _get_fftn_test_axes(shape)))
  def testFftn(self, inverse, shape, dtype, axes, rng_factory):
    rng = rng_factory()
    args_maker = lambda: (rng(shape, dtype),)
    np_op = np.fft.ifftn if inverse else np.fft.fftn
    onp_op = onp.fft.ifftn if inverse else onp.fft.fftn
    np_fn = lambda a: np_op(a, axes=axes)
    onp_fn = lambda a: onp_op(a, axes=axes)
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
  def testFftnErrors(self, inverse):
    rng = jtu.rand_default()
    name = 'ifftn' if inverse else 'fftn'
    func = np.fft.ifftn if inverse else np.fft.fftn
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


if __name__ == "__main__":
  absltest.main()
