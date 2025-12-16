# Copyright 2019 The JAX Authors.
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


from functools import partial
import math

import numpy as np

from absl.testing import absltest
import scipy.ndimage as osp_ndimage

import jax
from jax import grad
from jax._src import test_util as jtu
from jax import dtypes
from jax.scipy import ndimage as lsp_ndimage

jax.config.parse_flags_with_absl()


float_dtypes = jtu.dtypes.floating
int_dtypes = jtu.dtypes.integer


def _fix_scipy_mode(mode):
  return {
    'constant': 'grid-constant', 'wrap': 'grid-wrap'
  }.get(mode, mode)


class NdimageTest(jtu.JaxTestCase):

  @jtu.sample_product(
    [dict(mode=mode, cval=cval)
     for mode in ['wrap', 'constant', 'nearest', 'mirror', 'reflect']
     for cval in ([0, -1] if mode == 'constant' else [0])
    ],
    [dict(order=order, prefilter=prefilter, impl=impl, rng_factory=rng_factory)
     for order in list(range(6))
     for prefilter in ([False, True] if order > 1 else [True])
     for impl, rng_factory in ([
       ("original", partial(jtu.rand_uniform, low=0, high=1)),
       ("fixed", partial(jtu.rand_uniform, low=-0.75, high=1.75)),
     ] if order < 2 else [
       ("fixed", partial(jtu.rand_uniform, low=-0.75, high=1.75))
      ])
    ],
    shape=[(5,), (3, 4), (3, 4, 5)],
    coords_shape=[(7,), (2, 3, 4)],
    dtype=float_dtypes + int_dtypes,
    coords_dtype=float_dtypes,
    round_=[True, False],
  )
  def testMapCoordinates(self, shape, dtype, coords_shape, coords_dtype, order,
                         mode, cval, prefilter, impl, round_, rng_factory):

    def args_maker():
      x = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
      coords = [(size - 1) * rng(coords_shape, coords_dtype) for size in shape]
      if round_:
        coords = [c.round().astype(int) for c in coords]
      return x, coords

    rng = rng_factory(self.rng())
    lsp_op = lambda x, c: lsp_ndimage.map_coordinates(
        x, c, order=order, mode=mode, cval=cval, prefilter=prefilter)

    scipy_mode = _fix_scipy_mode(mode) if impl == 'fixed' else mode
    osp_op = lambda x, c: osp_ndimage.map_coordinates(
      x, c, order=order, mode=scipy_mode, cval=cval, prefilter=prefilter)

    with jtu.strict_promotion_if_dtypes_match([dtype, int if round_ else coords_dtype]):
      if dtype in float_dtypes:
        epsilon = max(dtypes.finfo(dtypes.canonicalize_dtype(d)).eps
                      for d in [dtype, coords_dtype])
        self._CheckAgainstNumpy(osp_op, lsp_op, args_maker, tol=1e-3 if order > 3 else 100*epsilon)
      elif order > 1:
        # output often falls exactly on 1/2, susceptible to rounding errors
        self._CheckAgainstNumpy(osp_op, lsp_op, args_maker, atol=1, rtol=0)
      else:
        self._CheckAgainstNumpy(osp_op, lsp_op, args_maker, tol=0)

  def testMapCoordinatesErrors(self):
    x = np.arange(5.0)
    c = [np.linspace(0, 5, num=3)]
    with self.assertRaisesRegex(
      NotImplementedError, 'does not yet support order'):
      lsp_ndimage.map_coordinates(x, c, order=7, prefilter=False)
    with self.assertRaisesRegex(
        NotImplementedError, 'does not yet support mode'):
      lsp_ndimage.map_coordinates(x, c, order=1, mode='grid-wrap')
    with self.assertRaisesRegex(ValueError, 'sequence of length'):
      lsp_ndimage.map_coordinates(x, [c, c], order=1)

  @jtu.sample_product(
    dtype=float_dtypes + int_dtypes,
    order=[0, 1, 2, 3],
  )
  def testMapCoordinatesRoundHalf(self, dtype, order):
    x = np.arange(-3, 8, 2, dtype=dtype)
    c = np.array([[.5, 1.5, 2.5, 3.5]])
    def args_maker():
      return x, c

    lsp_op = lambda x, c: lsp_ndimage.map_coordinates(x, c, order=order, prefilter=False, mode='constant')
    osp_op = lambda x, c: osp_ndimage.map_coordinates(x, c, order=order, prefilter=False, mode='grid-constant')

    with jtu.strict_promotion_if_dtypes_match([dtype, c.dtype]):
      self._CheckAgainstNumpy(osp_op, lsp_op, args_maker)

  def testContinuousGradients(self):
    # regression test for https://github.com/jax-ml/jax/issues/3024

    def loss(delta):
      x = np.arange(100.0)
      border = 10
      indices = np.arange(x.size, dtype=x.dtype) + delta
      # linear interpolation of the linear function y=x should be exact
      shifted = lsp_ndimage.map_coordinates(x, [indices], order=1)
      return ((x - shifted) ** 2)[border:-border].mean()

    # analytical gradient of (x - (x - delta)) ** 2 is 2 * delta
    self.assertAllClose(grad(loss)(0.5), 1.0, check_dtypes=False)
    self.assertAllClose(grad(loss)(1.0), 2.0, check_dtypes=False)

  @jtu.sample_product(
    order=[3, 4, 5],
    mode=['reflect', 'wrap', 'mirror'],
    shape=[(5,), (3, 4), (3, 4, 5)],
    dtype=float_dtypes,
  )
  def testSplineFilter(self, order, mode, shape, dtype):
    rng = jtu.rand_uniform(self.rng(), low=-0.5, high=0.5)

    def args_maker():
      return (rng(shape, dtype=dtype),)

    lsp_op = lambda arr: lsp_ndimage.spline_filter(arr, order=order, mode=mode)
    osp_op = lambda arr: osp_ndimage.spline_filter(arr, order=order, output=dtype, mode=_fix_scipy_mode(mode))

    self._CheckAgainstNumpy(osp_op, lsp_op, args_maker, tol=1e-2)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
     for shape in [(5,), (3, 4), (3, 4, 5)]
     for axis in range(len(shape))
    ],
    order=[3, 4, 5],
    mode=['reflect', 'wrap', 'mirror'],
    dtype=float_dtypes,
  )
  def testSplineFilter1D(self, order, mode, shape, axis, dtype):
    rng = jtu.rand_uniform(self.rng(), low=-0.5, high=0.5)

    def args_maker():
      return (rng(shape, dtype=dtype),)

    lsp_op = lambda arr: lsp_ndimage.spline_filter1d(arr, order=order, axis=axis, mode=mode)
    osp_op = lambda arr: osp_ndimage.spline_filter1d(arr, order=order, axis=axis, output=dtype, mode=_fix_scipy_mode(mode))

    self._CheckAgainstNumpy(osp_op, lsp_op, args_maker, tol=1e-2)

  def testSplineFilterErrors(self):
    x = np.arange(5.0)
    with self.assertRaisesRegex(
      ValueError, r"Spline order '7' not supported for pre-filtering"):
      lsp_ndimage.spline_filter(x, order=7)
    with self.assertRaisesRegex(
        ValueError, r"Boundary mode 'fail' not supported for pre-filtering"):
      lsp_ndimage.spline_filter(x, order=3, mode='fail')

  def testSplineFilter1DErrors(self):
    x = np.arange(5.0)
    with self.assertRaisesRegex(
      ValueError, r"Spline order '7' not supported for pre-filtering"):
      lsp_ndimage.spline_filter1d(x, order=7)
    with self.assertRaisesRegex(
        ValueError, r"Boundary mode 'fail' not supported for pre-filtering"):
      lsp_ndimage.spline_filter1d(x, order=3, mode='fail')


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
