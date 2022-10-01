# Copyright 2018 The JAX Authors.
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


import collections
from functools import partial
import itertools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import numpy as jnp

from jax._src import dtypes
from jax._src import test_util as jtu

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

numpy_version = tuple(map(int, np.__version__.split('.')[:3]))

nonempty_nonscalar_array_shapes = [(4,), (3, 4), (3, 1), (1, 4), (2, 1, 4), (2, 3, 4)]
nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
one_dim_array_shapes = [(1,), (6,), (12,)]
empty_array_shapes = [(0,), (0, 4), (3, 0),]

scalar_shapes = [jtu.NUMPY_SCALAR_SHAPE, jtu.PYTHON_SCALAR_SHAPE]
array_shapes = nonempty_array_shapes + empty_array_shapes
nonzerodim_shapes = nonempty_nonscalar_array_shapes + empty_array_shapes
nonempty_shapes = scalar_shapes + nonempty_array_shapes
all_shapes = scalar_shapes + array_shapes

float_dtypes = jtu.dtypes.all_floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.all_integer
unsigned_dtypes = jtu.dtypes.all_unsigned
bool_dtypes = jtu.dtypes.boolean
default_dtypes = float_dtypes + int_dtypes
inexact_dtypes = float_dtypes + complex_dtypes
number_dtypes = float_dtypes + complex_dtypes + int_dtypes + unsigned_dtypes
all_dtypes = number_dtypes + bool_dtypes

python_scalar_dtypes = [jnp.bool_, jnp.int_, jnp.float_, jnp.complex_]

def _compatible_shapes(shape):
  if np.ndim(shape) == 0 or shape in scalar_shapes:
    return [shape]
  return (shape[n:] for n in range(len(shape) + 1))

def _get_y_shapes(y_dtype, shape, rowvar):
  # Helper function for testCov.
  if y_dtype is None:
    return [None]
  if len(shape) == 1:
    return [shape]
  elif rowvar or shape[0] == 1:
    return [(1, shape[-1]), (2, shape[-1]), (5, shape[-1])]
  return [(shape[0], 1), (shape[0], 2), (shape[0], 5)]


OpRecord = collections.namedtuple(
  "OpRecord",
  ["name", "nargs", "dtypes", "shapes", "rng_factory", "diff_modes",
   "test_name", "check_dtypes", "tolerance", "inexact", "kwargs"])

def op_record(name, nargs, dtypes, shapes, rng_factory, diff_modes,
              test_name=None, check_dtypes=True,
              tolerance=None, inexact=False, kwargs=None):
  test_name = test_name or name
  return OpRecord(name, nargs, dtypes, shapes, rng_factory, diff_modes,
                  test_name, check_dtypes, tolerance, inexact, kwargs)

JAX_REDUCER_RECORDS = [
    op_record("mean", 1, number_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record("sum", 1, all_dtypes, all_shapes, jtu.rand_default, []),
    op_record("nanmean", 1, inexact_dtypes, nonempty_shapes, jtu.rand_some_nan,
              [], inexact=True),
    op_record("nanprod", 1, all_dtypes, all_shapes, jtu.rand_some_nan, []),
    op_record("nansum", 1, number_dtypes, all_shapes, jtu.rand_some_nan, []),
]

JAX_REDUCER_INITIAL_RECORDS = [
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record("sum", 1, all_dtypes, all_shapes, jtu.rand_default, []),
    op_record("max", 1, all_dtypes, all_shapes, jtu.rand_default, []),
    op_record("min", 1, all_dtypes, all_shapes, jtu.rand_default, []),
]
if numpy_version >= (1, 22):  # initial & where keywords added in numpy 1.22
  JAX_REDUCER_INITIAL_RECORDS += [
      op_record("nanprod", 1, inexact_dtypes, all_shapes, jtu.rand_small_positive, []),
      op_record("nansum", 1, inexact_dtypes, all_shapes, jtu.rand_default, []),
      op_record("nanmax", 1, inexact_dtypes, all_shapes, jtu.rand_default, []),
      op_record("nanmin", 1, inexact_dtypes, all_shapes, jtu.rand_default, []),
  ]

JAX_REDUCER_WHERE_NO_INITIAL_RECORDS = [
    op_record("all", 1, bool_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("any", 1, bool_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("mean", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("var", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("std", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
]
if numpy_version >= (1, 22):  # where keyword added in numpy 1.22
  JAX_REDUCER_WHERE_NO_INITIAL_RECORDS += [
      op_record("nanmean", 1, inexact_dtypes, nonempty_shapes, jtu.rand_default, [],
                inexact=True),
      op_record("nanvar", 1, inexact_dtypes, nonempty_shapes, jtu.rand_default, [],
                inexact=True),
      op_record("nanstd", 1, inexact_dtypes, nonempty_shapes, jtu.rand_default, [],
                inexact=True),
  ]

JAX_REDUCER_NO_DTYPE_RECORDS = [
    op_record("all", 1, all_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("any", 1, all_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("max", 1, all_dtypes, nonempty_shapes, jtu.rand_default, []),
    op_record("min", 1, all_dtypes, nonempty_shapes, jtu.rand_default, []),
    op_record("var", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("std", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("nanmax", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
    op_record("nanmin", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
    op_record("nanvar", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan,
              [], inexact=True),
    op_record("nanstd", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan,
              [], inexact=True),
    op_record("ptp", 1, number_dtypes, nonempty_shapes, jtu.rand_default, []),
]

JAX_REDUCER_PROMOTE_INT_RECORDS = [
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record("sum", 1, all_dtypes, all_shapes, jtu.rand_default, []),
]


class JaxNumpyReducerTests(jtu.JaxTestCase):
  """Tests for LAX-backed Numpy reduction operations."""

  def _GetArgsMaker(self, rng, shapes, dtypes, np_arrays=True):
    def f():
      out = [rng(shape, dtype or jnp.float_)
             for shape, dtype in zip(shapes, dtypes)]
      if np_arrays:
        return out
      return [jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) else a
              for a in out]
    return f

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_inshape={}_axis={}_dtype={}_keepdims={}".format(
            rec.test_name.capitalize(),
            jtu.format_shape_dtype_string(shape, dtype), axis,
            "None" if out_dtype is None else np.dtype(out_dtype).name, keepdims),
         "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype, "out_dtype": out_dtype,
         "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
         "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
        for shape in rec.shapes for dtype in rec.dtypes
        for out_dtype in [None] + rec.dtypes if out_dtype not in unsigned_dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        for keepdims in [False, True]
        if jtu.is_valid_shape(shape, dtype))
      for rec in JAX_REDUCER_RECORDS))
  def testReducer(self, np_op, jnp_op, rng_factory, shape, dtype, out_dtype,
                  axis, keepdims, inexact):
    rng = rng_factory(self.rng())
    @jtu.ignore_warning(category=np.ComplexWarning)
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="mean of empty slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="overflow encountered.*")
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if dtype != jnp.bfloat16 else x.astype(np.float32)
      t = out_dtype if out_dtype != jnp.bfloat16 else np.float32
      return np_op(x_cast, axis, dtype=t, keepdims=keepdims)

    jnp_fun = lambda x: jnp_op(x, axis, dtype=out_dtype, keepdims=keepdims)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    tol_spec = {np.float16: 1e-2, np.int32: 1E-3, np.float32: 1e-3,
                np.complex64: 1e-3, np.float64: 1e-5, np.complex128: 1e-5}
    tol = jtu.tolerance(dtype, tol_spec)
    tol = max(tol, jtu.tolerance(out_dtype, tol_spec)) if out_dtype else tol
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=jnp.bfloat16 not in (dtype, out_dtype),
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol,
                          rtol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_inshape={}_axis={}_keepdims={}".format(
            rec.test_name.capitalize(),
            jtu.format_shape_dtype_string(shape, dtype), axis, keepdims),
        "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
        "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
        "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
        for shape in rec.shapes for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        for keepdims in [False, True]
        if jtu.is_valid_shape(shape, dtype))
      for rec in JAX_REDUCER_NO_DTYPE_RECORDS))
  def testReducerNoDtype(self, np_op, jnp_op, rng_factory, shape, dtype, axis,
                         keepdims, inexact):
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16 and rng_factory.__name__ == 'rand_some_nan'
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="All-NaN (slice|axis) encountered.*")
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {np.float16: 0.002}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, rtol=tol, atol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_inshape={}_axis={}_keepdims={}_initial={}".format(
            rec.test_name.capitalize(),
            jtu.format_shape_dtype_string(shape, dtype), axis, keepdims, initial),
        "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
        "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
        "initial": initial, "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
        for shape in rec.shapes for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        for initial in [0, 1] for keepdims in [False, True]
        if jtu.is_valid_shape(shape, dtype))
      for rec in JAX_REDUCER_INITIAL_RECORDS))
  def testReducerInitial(self, np_op, jnp_op, rng_factory, shape, dtype, axis,
                         keepdims, initial, inexact):
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16 and rng_factory.__name__ == 'rand_some_nan'
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=np.ComplexWarning)
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims, initial=initial)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, initial=initial)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {jnp.bfloat16: 3E-2}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, rtol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_inshape={}_axis={}_keepdims={}_initial={}_promote_integers={}".format(
            rec.test_name.capitalize(),
            jtu.format_shape_dtype_string(shape, dtype), axis, keepdims, initial, promote_integers),
        "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
        "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
        "initial": initial, "axis": axis, "keepdims": keepdims, "inexact": rec.inexact,
        "promote_integers": promote_integers}
        for shape in rec.shapes for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        for initial in [0, 1] for keepdims in [False, True]
        for promote_integers in [True, False]
        if jtu.is_valid_shape(shape, dtype))
      for rec in JAX_REDUCER_PROMOTE_INT_RECORDS))
  def testReducerPromoteInt(self, np_op, jnp_op, rng_factory, shape, dtype, axis,
                            keepdims, initial, inexact, promote_integers):
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16 and rng_factory.__name__ == 'rand_some_nan'
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=np.ComplexWarning)
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims, initial=initial)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      print(f"res.dtype = {res.dtype}")
      if not promote_integers and dtypes.issubdtype(res.dtype, np.integer):
        res = res.astype(dtypes.to_numeric_dtype(x.dtype))
      return res

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, initial=initial, promote_integers=promote_integers)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {jnp.bfloat16: 3E-2}
    print(jnp_fun(*args_maker()))
    print(np_fun(*args_maker()))
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, rtol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_inshape={}_axis={}_keepdims={}".format(
            rec.test_name.capitalize(),
            jtu.format_shape_dtype_string(shape, dtype), axis, keepdims),
        "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
        "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
        "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
        for shape in rec.shapes if np.prod(shape) == 0
        for dtype in rec.dtypes
        for keepdims in [False, True]
        for axis in range(-len(shape), len(shape)) if shape[axis] >= 1)
      for rec in JAX_REDUCER_INITIAL_RECORDS))
  def testReducerNoInitialZeroDims(self, np_op, jnp_op, rng_factory, shape, dtype, axis,
                                   keepdims, inexact):
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16 and rng_factory.__name__ == 'rand_some_nan'
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=np.ComplexWarning)
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {jnp.bfloat16: 3E-2}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, rtol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_inshape={}_axis={}_keepdims={}_initial={}_whereshape={}".format(
            rec.test_name.capitalize(),
            jtu.format_shape_dtype_string(shape, dtype), axis, keepdims, initial,
            jtu.format_shape_dtype_string(whereshape, bool)),
        "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
        "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name), "whereshape": whereshape,
        "initial": initial, "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
        for shape in rec.shapes for dtype in rec.dtypes
        for whereshape in _compatible_shapes(shape)
        for axis in list(range(-len(shape), len(shape))) + [None]
        for initial in [0, 1] for keepdims in [False, True]
        if jtu.is_valid_shape(shape, dtype))
      for rec in JAX_REDUCER_INITIAL_RECORDS))
  def testReducerWhere(self, np_op, jnp_op, rng_factory, shape, dtype, axis,
                       keepdims, initial, inexact, whereshape):
    if (shape in [()] + scalar_shapes and
        dtype in [jnp.int16, jnp.uint16] and
        jnp_op in [jnp.min, jnp.max]):
      self.skipTest("Known XLA failure; see https://github.com/google/jax/issues/4971.")
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16 and rng_factory.__name__ == 'rand_some_nan'
    # Do not pass where via args_maker as that is incompatible with _promote_like_jnp.
    where = jtu.rand_bool(self.rng())(whereshape, np.bool_)
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=np.ComplexWarning)
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims, initial=initial, where=where)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, initial=initial, where=where)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(itertools.chain.from_iterable(
    jtu.cases_from_list(
      {"testcase_name": "{}_inshape={}_axis={}_keepdims={}_whereshape={}".format(
        rec.test_name.capitalize(),
        jtu.format_shape_dtype_string(shape, dtype), axis, keepdims,
        jtu.format_shape_dtype_string(whereshape, bool)),
        "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
        "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name), "whereshape": whereshape,
        "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
      for shape in rec.shapes for dtype in rec.dtypes
      for whereshape in _compatible_shapes(shape)
      for axis in list(range(-len(shape), len(shape))) + [None]
      for keepdims in [False, True]
      if jtu.is_valid_shape(shape, dtype))
    for rec in JAX_REDUCER_WHERE_NO_INITIAL_RECORDS))
  def testReducerWhereNoInitial(self, np_op, jnp_op, rng_factory, shape, dtype, axis,
                                keepdims, inexact, whereshape):
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16
    # Do not pass where via args_maker as that is incompatible with _promote_like_jnp.
    where = jtu.rand_bool(self.rng())(whereshape, np.bool_)
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Mean of empty slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="invalid value encountered.*")
    @jtu.ignore_warning(category=np.ComplexWarning)
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims, where=where)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, where=where)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    if numpy_version >= (1, 20, 2) or np_op.__name__ in ("all", "any"):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testReductionOfOutOfBoundsAxis(self):  # Issue 888
    x = jnp.ones((3, 4))
    self.assertRaises(ValueError, lambda: jnp.sum(x, axis=2))

  def testReductionWithRepeatedAxisError(self):
    with self.assertRaisesRegex(ValueError, r"duplicate value in 'axis': \(0, 0\)"):
      jnp.sum(jnp.arange(3), (0, 0))

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name":
         "_shape={}_dtype={}_out_dtype={}_axis={}_ddof={}_keepdims={}"
         .format(shape, dtype.__name__, out_dtype.__name__, axis, ddof, keepdims),
         "shape": shape, "dtype": dtype, "out_dtype": out_dtype, "axis": axis,
         "ddof": ddof, "keepdims": keepdims}
        for shape in [(5,), (10, 5)]
        for dtype in all_dtypes
        for out_dtype in inexact_dtypes
        for axis in [None, 0, -1]
        for ddof in [0, 1, 2]
        for keepdims in [False, True]))
  def testVar(self, shape, dtype, out_dtype, axis, ddof, keepdims):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.")
    @jtu.ignore_warning(category=np.ComplexWarning)
    def np_fun(x):
      # Numpy fails with bfloat16 inputs
      out = np.var(x.astype(np.float32 if dtype == dtypes.bfloat16 else dtype),
                   dtype=np.float32 if out_dtype == dtypes.bfloat16 else out_dtype,
                   axis=axis, ddof=ddof, keepdims=keepdims)
      return out.astype(out_dtype)
    jnp_fun = partial(jnp.var, dtype=out_dtype, axis=axis, ddof=ddof, keepdims=keepdims)
    tol = jtu.tolerance(out_dtype, {np.float16: 1e-1, np.float32: 1e-3,
                                    np.float64: 1e-3, np.complex128: 1e-6})
    if (jnp.issubdtype(dtype, jnp.complexfloating) and
        not jnp.issubdtype(out_dtype, jnp.complexfloating)):
      self.assertRaises(ValueError, lambda: jnp_fun(*args_maker()))
    else:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                              tol=tol)
      self._CompileAndCheck(jnp_fun, args_maker, rtol=tol,
                            atol=tol)

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name":
         "_shape={}_dtype={}_out_dtype={}_axis={}_ddof={}_keepdims={}"
         .format(shape, dtype, out_dtype, axis, ddof, keepdims),
         "shape": shape, "dtype": dtype, "out_dtype": out_dtype, "axis": axis,
         "ddof": ddof, "keepdims": keepdims}
        for shape in [(5,), (10, 5)]
        for dtype in all_dtypes
        for out_dtype in inexact_dtypes
        for axis in [None, 0, -1]
        for ddof in [0, 1, 2]
        for keepdims in [False, True]))
  def testNanVar(self, shape, dtype, out_dtype, axis, ddof, keepdims):
    rng = jtu.rand_some_nan(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.")
    @jtu.ignore_warning(category=np.ComplexWarning)
    def np_fun(x):
      # Numpy fails with bfloat16 inputs
      out = np.nanvar(x.astype(np.float32 if dtype == dtypes.bfloat16 else dtype),
                      dtype=np.float32 if out_dtype == dtypes.bfloat16 else out_dtype,
                      axis=axis, ddof=ddof, keepdims=keepdims)
      return out.astype(out_dtype)
    jnp_fun = partial(jnp.nanvar, dtype=out_dtype, axis=axis, ddof=ddof, keepdims=keepdims)
    tol = jtu.tolerance(out_dtype, {np.float16: 1e-1, np.float32: 1e-3,
                                    np.float64: 1e-3, np.complex128: 1e-6})
    if (jnp.issubdtype(dtype, jnp.complexfloating) and
        not jnp.issubdtype(out_dtype, jnp.complexfloating)):
      self.assertRaises(ValueError, lambda: jnp_fun(*args_maker()))
    else:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                              tol=tol)
      self._CompileAndCheck(jnp_fun, args_maker, rtol=tol,
                            atol=tol)

  def testNanStdGrad(self):
    # Regression test for https://github.com/google/jax/issues/8128
    x = jnp.arange(5.0).at[0].set(jnp.nan)
    y = jax.grad(jnp.nanvar)(x)
    self.assertAllClose(y, jnp.array([0.0, -0.75, -0.25, 0.25, 0.75]))

    z = jax.grad(jnp.nanstd)(x)
    self.assertEqual(jnp.isnan(z).sum(), 0)

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name":
          "_shape={}_dtype={}_y_shape={}_y_dtype={}_rowvar={}_ddof={}_bias={}_fweights={}_aweights={}".format(
            shape, dtype, y_shape, y_dtype, rowvar, ddof, bias, fweights, aweights),
         "shape": shape, "y_shape": y_shape, "dtype": dtype, "y_dtype": y_dtype,"rowvar": rowvar, "ddof": ddof,
         "bias": bias, "fweights": fweights, "aweights": aweights}
        for shape in [(5,), (10, 5), (5, 10)]
        for dtype in all_dtypes
        for y_dtype in [None, dtype]
        for rowvar in [True, False]
        for y_shape in _get_y_shapes(y_dtype, shape, rowvar)
        for bias in [True, False]
        for ddof in [None, 2, 3]
        for fweights in [True, False]
        for aweights in [True, False]))
  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises mixed type promotion
  def testCov(self, shape, dtype, y_shape, y_dtype, rowvar, ddof, bias, fweights, aweights):
    rng = jtu.rand_default(self.rng())
    wrng = jtu.rand_positive(self.rng())
    wdtype = np.real(dtype(0)).dtype
    wshape = shape[-1:] if rowvar or shape[0] == 1 else shape[:1]

    args_maker = lambda: [rng(shape, dtype),
                          rng(y_shape, y_dtype) if y_dtype else None,
                          wrng(wshape, int) if fweights else None,
                          wrng(wshape, wdtype) if aweights else None]
    kwargs = dict(rowvar=rowvar, ddof=ddof, bias=bias)
    np_fun = lambda m, y, f, a: np.cov(m, y, fweights=f, aweights=a, **kwargs)
    jnp_fun = lambda m, y, f, a: jnp.cov(m, y, fweights=f, aweights=a, **kwargs)
    tol = {jnp.bfloat16: 5E-2, np.float16: 1E-2, np.float32: 1e-5,
           np.float64: 1e-13, np.complex64: 1e-5, np.complex128: 1e-13}
    tol = 7e-2 if jtu.device_under_test() == "tpu" else tol
    tol = jtu.join_tolerance(tol, jtu.tolerance(dtype))
    self._CheckAgainstNumpy(
        np_fun, jnp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol,
                          rtol=tol)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
