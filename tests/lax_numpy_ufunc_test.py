# Copyright 2022 Google LLC
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
import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
from jax import lax
from jax import tree_util
from jax._src import test_util as jtu
from jax._src.numpy.lax_numpy import _promote_dtypes, _promote_dtypes_inexact
import jax.numpy as jnp

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

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
exact_dtypes = bool_dtypes + int_dtypes + unsigned_dtypes
inexact_dtypes = float_dtypes + complex_dtypes
noncomplex_dtypes = exact_dtypes + float_dtypes
number_dtypes = float_dtypes + complex_dtypes + int_dtypes + unsigned_dtypes
all_dtypes = number_dtypes + bool_dtypes

python_scalar_dtypes = [jnp.bool_, jnp.int_, jnp.float_, jnp.complex_]

def _shapes_are_broadcast_compatible(shapes):
  try:
    lax.broadcast_shapes(*(() if s in scalar_shapes else s for s in shapes))
  except ValueError:
    return False
  else:
    return True

def _valid_dtypes_for_shape(shape, dtypes):
  # Not all (shape, dtype) pairs are valid. In particular, Python scalars only
  # have one type in each category (float, bool, etc.)
  if shape is jtu.PYTHON_SCALAR_SHAPE:
    return [t for t in dtypes if t in python_scalar_dtypes]
  return dtypes

def _promote_like_jnp(fun, inexact=False):
  """Decorator that promotes the arguments of `fun` to `jnp.result_type(*args)`.

  jnp and np have different type promotion semantics; this decorator allows
  tests make an np reference implementation act more like an jnp
  implementation.
  """
  _promote = _promote_dtypes_inexact if inexact else _promote_dtypes
  def wrapper(*args, **kw):
    flat_args, tree = tree_util.tree_flatten(args)
    args = tree_util.tree_unflatten(tree, _promote(*flat_args))
    return fun(*args, **kw)
  return wrapper

UfuncRecord = collections.namedtuple(
  "UfuncRecord",
  ["name", "dtypes", "shapes", "rng_factory", "check_dtypes", "inexact", "tol", "reduce_tol"])

def ufunc_record(name, dtypes, shapes, rng_factory, *, check_dtypes=True, inexact=False, tol=None, reduce_tol=None):
  return UfuncRecord(name, dtypes, shapes, rng_factory, check_dtypes, inexact, tol=tol, reduce_tol=reduce_tol)


UFUNC_RECORDS = [
    # Unary ufuncs:
    ufunc_record("bitwise_not", exact_dtypes, all_shapes, jtu.rand_default),
    ufunc_record("invert", exact_dtypes, all_shapes, jtu.rand_default),
    ufunc_record("fabs", noncomplex_dtypes, all_shapes, jtu.rand_default, inexact=True),
    ufunc_record("negative", number_dtypes, all_shapes, jtu.rand_default),
    ufunc_record("positive", number_dtypes, all_shapes, jtu.rand_default),
    ufunc_record("floor", noncomplex_dtypes, all_shapes, jtu.rand_default, inexact=True),
    ufunc_record("ceil", noncomplex_dtypes, all_shapes, jtu.rand_default, inexact=True),
    # Binary ufuncs:
    ufunc_record("add", all_dtypes, all_shapes, jtu.rand_default,
                 reduce_tol={np.float16: 5E-3, jnp.bfloat16: 5E-2}),
    ufunc_record("bitwise_and", exact_dtypes, all_shapes, jtu.rand_default),
    ufunc_record("bitwise_or", exact_dtypes, all_shapes, jtu.rand_default),
    ufunc_record("bitwise_xor", exact_dtypes, all_shapes, jtu.rand_default),
    ufunc_record("divide", all_dtypes, all_shapes, jtu.rand_nonzero, inexact=True),
    ufunc_record("floor_divide", default_dtypes + unsigned_dtypes, all_shapes, jtu.rand_nonzero),
    ufunc_record("maximum", all_dtypes, all_shapes, jtu.rand_default),
    ufunc_record("minimum", all_dtypes, all_shapes, jtu.rand_default),
    ufunc_record("multiply", all_dtypes, all_shapes, jtu.rand_default),
    ufunc_record("subtract", number_dtypes, all_shapes, jtu.rand_default,
                 reduce_tol={np.float16: 5E-3, jnp.bfloat16: 5E-2}),
    ufunc_record("true_divide", all_dtypes, all_shapes, jtu.rand_nonzero, inexact=True),
]


class JaxNumpyUfuncTests(jtu.JaxTestCase):
  def _GetArgsMaker(self, rng, shapes, dtypes, np_arrays=True):
    def f():
      out = [rng(shape, dtype or jnp.float_)
             for shape, dtype in zip(shapes, dtypes)]
      if np_arrays:
        return out
      return [jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) else a
              for a in out]
    return f

  @parameterized.named_parameters(
    {"testcase_name": f"_{rec.name}", "name": rec.name}
    for rec in UFUNC_RECORDS)
  def testUfuncAttributes(self, name):
    np_ufunc = getattr(np, name)
    jnp_ufunc = getattr(jnp, name)

    self.assertEqual(np_ufunc.identity, jnp_ufunc.identity)
    self.assertEqual(np_ufunc.nargs, jnp_ufunc.nargs + 1)  # numpy ufuncs accept `out` arg.
    self.assertEqual(np_ufunc.nin, jnp_ufunc.nin)
    self.assertEqual(np_ufunc.nout, jnp_ufunc.nout)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.name, shapes, dtypes),
         "rec": rec, "shapes": shapes, "dtypes": dtypes}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, getattr(jnp, rec.name).nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes)))
      for rec in itertools.chain(UFUNC_RECORDS)))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testCall(self, rec, shapes, dtypes):
    np_ufunc = getattr(np, rec.name)
    jnp_op = getattr(jnp, rec.name)

    np_op = _promote_like_jnp(np_ufunc, rec.inexact)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                               message="invalid value.*")(np_op)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                               message="divide by zero.*")(np_op)

    rng = rec.rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    tol = max(jtu.tolerance(dtype) for dtype in dtypes)
    tol = functools.reduce(jtu.join_tolerance, [tol, rec.tol, jtu.default_tolerance()])
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(np_op, jnp_op, args_maker, check_dtypes=rec.check_dtypes, tol=tol)
      self._CompileAndCheck(jnp_op, args_maker, check_dtypes=rec.check_dtypes, atol=tol, rtol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_axis={}_keepdims={}".format(
          jtu.format_test_name_suffix(rec.name, (shape,), (dtype,)), axis, keepdims),
         "rec": rec, "shape": shape, "dtype": dtype, "axis": axis, "keepdims": keepdims}
        for shape in nonempty_nonscalar_array_shapes
        for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape)))  # TODO(jakevdp) test None, but only for reorderable operations.
        for keepdims in [True, False])
      for rec in itertools.chain(UFUNC_RECORDS) if getattr(jnp, rec.name).nargs == 2))
  def testReduce(self, rec, shape, dtype, axis, keepdims):
    np_ufunc = getattr(np, rec.name)
    jnp_ufunc = getattr(jnp, rec.name)

    jnp_op = functools.partial(jnp_ufunc.reduce, axis=axis, keepdims=keepdims)
    np_op = _promote_like_jnp(functools.partial(np_ufunc.reduce, axis=axis, keepdims=keepdims), rec.inexact)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                               message="invalid value.*")(np_op)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                               message="divide by zero.*")(np_op)

    rng = rec.rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    tol = functools.reduce(jtu.join_tolerance, [jtu.tolerance(dtype), rec.reduce_tol, jtu.default_tolerance()])
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker, check_dtypes=rec.check_dtypes, tol=tol)
    self._CompileAndCheck(jnp_op, args_maker, check_dtypes=rec.check_dtypes, atol=tol, rtol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_axis={}".format(
          jtu.format_test_name_suffix(rec.name, (shape,), (dtype,)), axis),
         "rec": rec, "shape": shape, "dtype": dtype, "axis": axis}
        for shape in nonempty_nonscalar_array_shapes
        for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))))
      for rec in itertools.chain(UFUNC_RECORDS) if getattr(jnp, rec.name).nargs == 2))
  def testAccumulate(self, rec, shape, dtype, axis):
    np_ufunc = getattr(np, rec.name)
    jnp_ufunc = getattr(jnp, rec.name)

    @jtu.ignore_warning(category=RuntimeWarning, message="divide by zero.*")
    @jtu.ignore_warning(category=RuntimeWarning, message="invalid value.*")
    @functools.partial(_promote_like_jnp, inexact=rec.inexact)
    def np_op(a):
      result = np_ufunc.accumulate(a, axis=axis)
      if a.dtype != bool:
        result = result.astype(a.dtype)
      return result
    jnp_op = functools.partial(jnp_ufunc.accumulate, axis=axis)

    rng = rec.rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    tol = functools.reduce(jtu.join_tolerance, [jtu.tolerance(dtype), rec.reduce_tol, jtu.default_tolerance()])
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker, check_dtypes=rec.check_dtypes, tol=tol)
    self._CompileAndCheck(jnp_op, args_maker, check_dtypes=rec.check_dtypes, atol=tol, rtol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_idx={}".format(
          jtu.format_test_name_suffix(rec.name, (shape,), (dtype,)), idx),
         "rec": rec, "shape": shape, "dtype": dtype, "idx": idx}
        for shape in nonempty_nonscalar_array_shapes
        for dtype in rec.dtypes if dtype != np.bool_
        for idx in [0, slice(2)]) # TODO(jakevdp) add more index types.
      for rec in itertools.chain(UFUNC_RECORDS) if getattr(jnp, rec.name).nin == 2))
  def testAt(self, rec, shape, dtype, idx):
    np_ufunc = getattr(np, rec.name)
    jnp_ufunc = getattr(jnp, rec.name)

    extra_args = (1,) if jnp_ufunc.nin == 2 else ()

    @functools.partial(_promote_like_jnp, inexact=rec.inexact)
    def np_op(a):
      b = np.array(a, copy=True)
      np_ufunc.at(b, idx, *extra_args)
      return b
    def jnp_op(a):
      return jnp_ufunc.at(a, idx, *extra_args, inplace=False)

    rng = rec.rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    tol = functools.reduce(jtu.join_tolerance, [jtu.tolerance(dtype), rec.tol, jtu.default_tolerance()])
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker, check_dtypes=rec.check_dtypes, tol=tol)
    self._CompileAndCheck(jnp_op, args_maker, check_dtypes=rec.check_dtypes, atol=tol, rtol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_axis={}".format(
          jtu.format_test_name_suffix(rec.name, (shape,), (dtype,)), axis),
         "rec": rec, "shape": shape, "dtype": dtype, "axis": axis}
        for shape in one_dim_array_shapes # TODO(jakevdp) use nonempty_nonscalar_array_shapes
        for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))))
      for rec in itertools.chain(UFUNC_RECORDS) if getattr(jnp, rec.name).nin == 2))
  def testReduceat(self, rec, shape, dtype, axis):
    if rec.name in ["subtract", "divide", "true_divide", "floor_divide",
                    "bitwise_and", "bitwise_or", "bitwise_xor"]:
      self.skipTest(f"jnp.{rec.name}.reduceat() not supported")
    np_ufunc = getattr(np, rec.name)
    jnp_ufunc = getattr(jnp, rec.name)

    def np_op(a, indices):
      result = np_ufunc.reduceat(a, indices, axis=axis)
      if a.dtype != bool:
        result = result.astype(a.dtype)
      return result
    jnp_op = functools.partial(jnp_ufunc.reduceat, axis=axis)

    def _make_indices(rng):
      # Avoid repeated indices because of NumPy bug: https://github.com/numpy/numpy/issues/834
      size = shape[axis]
      return np.sort(rng.choice(size, size // 2, replace=False))

    rng = rec.rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype), _make_indices(self.rng())]
    tol = functools.reduce(jtu.join_tolerance, [jtu.tolerance(dtype), rec.reduce_tol, jtu.default_tolerance()])
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker, check_dtypes=rec.check_dtypes, tol=tol)
    self._CompileAndCheck(jnp_op, args_maker, check_dtypes=rec.check_dtypes, atol=tol, rtol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.name, (a_shape, b_shape), (dtype, dtype)),
         "rec": rec, "a_shape": a_shape, "b_shape": b_shape, "dtype": dtype}
        for a_shape in all_shapes
        for b_shape in all_shapes
        for dtype in rec.dtypes)
      for rec in itertools.chain(UFUNC_RECORDS) if getattr(jnp, rec.name).nargs == 2))
  def testOuter(self, rec, a_shape, b_shape, dtype):
    rng = rec.rng_factory(self.rng())
    np_op = _promote_like_jnp(getattr(np, rec.name).outer, rec.inexact)
    jnp_op = getattr(jnp, rec.name).outer
    args_maker = lambda: [rng(a_shape, dtype), rng(b_shape, dtype)]
    # TODO: set check_dtypes=rec.check_dtypes after fixing _promote_like_jnp to respect weak types.
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_op, args_maker, check_dtypes=rec.check_dtypes)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
