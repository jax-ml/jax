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


from functools import partial

from absl.testing import absltest, parameterized

import numpy as onp

from jax import lax
from jax import test_util as jtu
import jax.scipy.signal as jsp_signal
import scipy.signal as osp_signal

from jax.config import config
config.parse_flags_with_absl()

onedim_shapes = [(1,), (2,), (5,), (10,)]
twodim_shapes = [(1, 1), (2, 2), (2, 3), (3, 4), (4, 4)]


def supported_dtypes(dtypes):
  return [t for t in dtypes if t in jtu.supported_dtypes()]


float_dtypes = supported_dtypes([onp.float32, onp.float64])
int_dtypes = [onp.int32, onp.int64]
default_dtypes = float_dtypes + int_dtypes


class LaxBackedScipySignalTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.stats implementations"""

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "op={}_xshape=[{}]_yshape=[{}]_mode={}".format(
          op,
          jtu.format_shape_dtype_string(xshape, dtype),
          jtu.format_shape_dtype_string(yshape, dtype),
          mode),
       "xshape": xshape, "yshape": yshape, "dtype": dtype, "mode": mode,
       "rng_factory": jtu.rand_default,
       "jsp_op": getattr(jsp_signal, op),
       "osp_op": getattr(osp_signal, op)}
      for mode in ['full', 'same', 'valid']
      for op in ['convolve', 'correlate']
      for dtype in default_dtypes
      for xshape in onedim_shapes
      for yshape in onedim_shapes))
  def testConvolutions(self, xshape, yshape, dtype, mode, rng_factory, jsp_op, osp_op):
    rng = rng_factory()
    args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
    osp_fun = partial(osp_op, mode=mode)
    jsp_fun = partial(jsp_op, mode=mode, precision=lax.Precision.HIGHEST)
    tol = {onp.float16: 1e-2, onp.float32: 1e-5}
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jsp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "op={}_xshape=[{}]_yshape=[{}]_mode={}".format(
          op,
          jtu.format_shape_dtype_string(xshape, dtype),
          jtu.format_shape_dtype_string(yshape, dtype),
          mode),
       "xshape": xshape, "yshape": yshape, "dtype": dtype, "mode": mode,
       "rng_factory": jtu.rand_default,
       "jsp_op": getattr(jsp_signal, op),
       "osp_op": getattr(osp_signal, op)}
      for mode in ['full', 'same', 'valid']
      for op in ['convolve2d', 'correlate2d']
      for dtype in default_dtypes
      for xshape in twodim_shapes
      for yshape in twodim_shapes))
  def testConvolutions2D(self, xshape, yshape, dtype, mode, rng_factory, jsp_op, osp_op):
    rng = rng_factory()
    args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
    osp_fun = partial(osp_op, mode=mode)
    jsp_fun = partial(jsp_op, mode=mode, precision=lax.Precision.HIGHEST)
    tol = {onp.float16: 1e-2, onp.float32: 1e-5}
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jsp_fun, args_maker, check_dtypes=True)


if __name__ == "__main__":
    absltest.main()
