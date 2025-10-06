# Copyright 2025 The JAX Authors.
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

from functools import partial, wraps
import itertools
import math

from absl.testing import absltest

import numpy as np

import jax
from jax import lax
import jax.numpy as jnp
from jax._src import config
from jax._src import test_util as jtu

config.parse_flags_with_absl()


nonempty_array_shapes = [(), (4,), (3, 4), (3, 1), (1, 4), (2, 1, 4), (2, 3, 4)]
empty_array_shapes = [(0,), (0, 4), (3, 0),]

scalar_shapes = [jtu.NUMPY_SCALAR_SHAPE, jtu.PYTHON_SCALAR_SHAPE]
array_shapes = nonempty_array_shapes + empty_array_shapes
nonempty_shapes = scalar_shapes + nonempty_array_shapes
all_shapes = scalar_shapes + array_shapes

default_dtypes = jtu.dtypes.all_floating + jtu.dtypes.all_integer
inexact_dtypes = jtu.dtypes.all_floating + jtu.dtypes.complex
number_dtypes = default_dtypes + jtu.dtypes.complex + jtu.dtypes.all_unsigned


def np_unique_backport(ar, return_index=False, return_inverse=False, return_counts=False,
                       axis=None, **kwds):
  # Wrapper for np.unique, handling the change to inverse_indices in numpy 2.0
  result = np.unique(ar, return_index=return_index, return_inverse=return_inverse,
                     return_counts=return_counts, axis=axis, **kwds)
  if jtu.numpy_version() >= (2, 0, 1) or np.ndim(ar) == 1 or not return_inverse:
    return result

  idx = 2 if return_index else 1
  inverse_indices = result[idx]
  if axis is None:
    inverse_indices = inverse_indices.reshape(np.shape(ar))
  elif jtu.numpy_version() == (2, 0, 0):
    inverse_indices = inverse_indices.reshape(-1)
  return (*result[:idx], inverse_indices, *result[idx + 1:])

def arrays_with_overlapping_values(rng, shapes, dtypes, unique=False, overlap=0.5) -> list[jax.Array]:
  """Generate multiple arrays with some overlapping values.

  This is useful for tests of set-like operations.
  """
  assert 0 <= overlap <= 1
  sizes = [math.prod(jtu._dims_of_shape(shape)) for shape in shapes]
  total_size = int(sum(sizes) * (1 - overlap)) + max(sizes)  # non-strict upper-bound.
  if unique:
    vals = jtu.rand_unique_int(rng)((total_size,), 'int32')
  else:
    vals = jtu.rand_default(rng)((total_size,), 'int32')
  offsets = [int(sum(sizes[:i]) * (1 - overlap)) for i in range(len(sizes))]
  return [rng.permutation(vals[offset: offset + size]).reshape(shape).astype(dtype)
          for (offset, size, shape, dtype) in zip(offsets, sizes, shapes, dtypes)]

def with_size_argument(fun):
  @wraps(fun)
  def wrapped(*args, size=None, fill_value=None, **kwargs):
    result = fun(*args, **kwargs)
    if size is None or size == len(result):
      return result
    elif size < len(result):
      return result[:size]
    else:
      if fill_value is None:
        fill_value = result.min() if result.size else 0
      return np.pad(result, (0, size - len(result)), constant_values=fill_value)
  return wrapped


class LaxNumpySetopsTest(jtu.JaxTestCase):
  """Tests of set-like operations from jax.numpy."""

  @jtu.sample_product(
    element_shape=all_shapes,
    test_shape=all_shapes,
    dtype=default_dtypes,
    invert=[False, True],
    method=['auto', 'compare_all', 'binary_search', 'sort']
  )
  def testIsin(self, element_shape, test_shape, dtype, invert, method):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(element_shape, dtype), rng(test_shape, dtype)]
    jnp_fun = lambda e, t: jnp.isin(e, t, invert=invert, method=method)
    np_fun = lambda e, t: np.isin(e, t, invert=invert)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    dtype1=[s for s in default_dtypes if s != jnp.bfloat16],
    dtype2=[s for s in default_dtypes if s != jnp.bfloat16],
    shape1=all_shapes,
    shape2=all_shapes,
    overlap=[0.1, 0.5, 0.9],
  )
  def testSetdiff1d(self, shape1, shape2, dtype1, dtype2, overlap):
    args_maker = partial(arrays_with_overlapping_values, self.rng(),
                         shapes=[shape1, shape2], dtypes=[dtype1, dtype2],
                         overlap=overlap)
    with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
      self._CheckAgainstNumpy(np.setdiff1d, jnp.setdiff1d, args_maker)

  @jtu.sample_product(
    dtype1=[s for s in default_dtypes if s != jnp.bfloat16],
    dtype2=[s for s in default_dtypes if s != jnp.bfloat16],
    shape1=all_shapes,
    shape2=all_shapes,
    size=[1, 5, 10],
    fill_value=[None, -1],
    overlap=[0.1, 0.5, 0.9],
  )
  def testSetdiff1dSize(self, shape1, shape2, dtype1, dtype2, size, fill_value, overlap):
    args_maker = partial(arrays_with_overlapping_values, self.rng(),
                         shapes=[shape1, shape2], dtypes=[dtype1, dtype2],
                         overlap=overlap)
    def np_fun(arg1, arg2):
      result = np.setdiff1d(arg1, arg2)
      if size <= len(result):
        return result[:size]
      else:
        return np.pad(result, (0, size-len(result)), constant_values=fill_value or 0)
    def jnp_fun(arg1, arg2):
      return jnp.setdiff1d(arg1, arg2, size=size, fill_value=fill_value)
    with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
      self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      shape1=all_shapes,
      shape2=all_shapes,
  )
  def testSetdiff1dAssumeUnique(self, shape1, shape2):
    # regression test for https://github.com/jax-ml/jax/issues/32335
    args_maker = lambda: (jnp.arange(math.prod(shape1), dtype='int32').reshape(shape1),
                          jnp.arange(math.prod(shape2), dtype='int32').reshape(shape2))
    np_op = partial(np.setdiff1d, assume_unique=True)
    jnp_op = partial(jnp.setdiff1d, assume_unique=True)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)

  @jtu.sample_product(
    dtype1=[s for s in default_dtypes if s != jnp.bfloat16],
    dtype2=[s for s in default_dtypes if s != jnp.bfloat16],
    shape1=all_shapes,
    shape2=all_shapes,
    overlap=[0.1, 0.5, 0.9],
  )
  def testUnion1d(self, shape1, shape2, dtype1, dtype2, overlap):
    args_maker = partial(arrays_with_overlapping_values, self.rng(),
                         shapes=[shape1, shape2], dtypes=[dtype1, dtype2],
                         overlap=overlap)
    def np_fun(arg1, arg2):
      dtype = jnp.promote_types(arg1.dtype, arg2.dtype)
      return np.union1d(arg1, arg2).astype(dtype)
    with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
      self._CheckAgainstNumpy(np_fun, jnp.union1d, args_maker)

  @jtu.sample_product(
    dtype1=[s for s in default_dtypes if s != jnp.bfloat16],
    dtype2=[s for s in default_dtypes if s != jnp.bfloat16],
    shape1=nonempty_shapes,
    shape2=nonempty_shapes,
    size=[1, 5, 10],
    fill_value=[None, -1],
    overlap=[0.1, 0.5, 0.9],
  )
  def testUnion1dSize(self, shape1, shape2, dtype1, dtype2, size, fill_value, overlap):
    args_maker = partial(arrays_with_overlapping_values, self.rng(),
                         shapes=[shape1, shape2], dtypes=[dtype1, dtype2],
                         overlap=overlap)
    def np_fun(arg1, arg2):
      dtype = jnp.promote_types(arg1.dtype, arg2.dtype)
      result = np.union1d(arg1, arg2).astype(dtype)
      fv = result.min() if fill_value is None else fill_value
      if size <= len(result):
        return result[:size]
      else:
        return np.concatenate([result, np.full(size - len(result), fv, result.dtype)])
    def jnp_fun(arg1, arg2):
      return jnp.union1d(arg1, arg2, size=size, fill_value=fill_value)
    with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
      self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    dtype1=[s for s in default_dtypes if s != jnp.bfloat16],
    dtype2=[s for s in default_dtypes if s != jnp.bfloat16],
    shape1=all_shapes,
    shape2=all_shapes,
    assume_unique=[False, True],
    size=[None, 2, 5],
    fill_value=[None, 99],
    overlap=[0.1, 0.5, 0.9],
  )
  def testSetxor1d(self, shape1, dtype1, shape2, dtype2, assume_unique, size, fill_value, overlap):
    args_maker = partial(arrays_with_overlapping_values, self.rng(),
                         shapes=[shape1, shape2], dtypes=[dtype1, dtype2],
                         overlap=overlap)
    jnp_fun = lambda ar1, ar2: jnp.setxor1d(ar1, ar2, assume_unique=assume_unique,
                                            size=size, fill_value=fill_value)
    def np_fun(ar1, ar2):
      if assume_unique:
        # numpy requires 1D inputs when assume_unique is True.
        ar1 = np.ravel(ar1)
        ar2 = np.ravel(ar2)
      return with_size_argument(np.setxor1d)(ar1, ar2, assume_unique, size=size, fill_value=fill_value)
    with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

  @jtu.sample_product(
    dtype1=[s for s in default_dtypes if s != jnp.bfloat16],
    dtype2=[s for s in default_dtypes if s != jnp.bfloat16],
    shape1=nonempty_shapes,
    shape2=nonempty_shapes,
    assume_unique=[False, True],
    return_indices=[False, True],
    size=[None, 3, 5],
    fill_value=[None, -1],
    overlap=[0.1, 0.5, 0.9],
  )
  def testIntersect1d(self, shape1, dtype1, shape2, dtype2, assume_unique,
                      return_indices, size, fill_value, overlap):
    args_maker = partial(arrays_with_overlapping_values, self.rng(),
                         shapes=[shape1, shape2], dtypes=[dtype1, dtype2],
                         unique=assume_unique, overlap=overlap)

    def jnp_fun(ar1, ar2):
      return jnp.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices,
                             size=size, fill_value=fill_value)

    def np_fun(ar1, ar2):
      result = np.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
      def correct_size(x, fill_value):
        if size is None or size == len(x):
          return x
        elif size < len(x):
          return x[:size]
        else:
          if fill_value is None:
            fill_value = x.min()
          return np.pad(x, (0, size - len(x)), constant_values=fill_value)
      if return_indices:
        return tuple(correct_size(r, f) for r, f in zip(result, [fill_value, ar1.size, ar2.size]))
      else:
        return correct_size(result, fill_value)

    with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in all_shapes
      for axis in [None] + list(range(len(shape)))],
    dtype=number_dtypes,
    return_index=[False, True],
    return_inverse=[False, True],
    return_counts=[False, True],
  )
  def testUnique(self, shape, dtype, axis, return_index, return_inverse, return_counts):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    extra_args = (return_index, return_inverse, return_counts)
    use_defaults =  (False, *(True for arg in extra_args if arg)) if any(extra_args) else False
    np_fun = jtu.with_jax_dtype_defaults(lambda x: np_unique_backport(x, *extra_args, axis=axis), use_defaults)
    jnp_fun = lambda x: jnp.unique(x, *extra_args, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @jtu.sample_product(shape=all_shapes, dtype=number_dtypes)
  def testUniqueAll(self, shape, dtype):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(jnp.unique_all, np.unique_all, args_maker)

  @jtu.sample_product(shape=all_shapes, dtype=number_dtypes)
  def testUniqueCounts(self, shape, dtype):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(jnp.unique_counts, np.unique_counts, args_maker)

  @jtu.sample_product(shape=all_shapes, dtype=number_dtypes)
  def testUniqueInverse(self, shape, dtype):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(jnp.unique_inverse, np.unique_inverse, args_maker)

  @jtu.sample_product(shape=all_shapes, dtype=number_dtypes)
  def testUniqueValues(self, shape, dtype):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda *args: np.sort(np.unique_values(*args))
    self._CheckAgainstNumpy(jnp.unique_values, np_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonempty_array_shapes
      for axis in [None] + list(range(len(shape)))],
    dtype=number_dtypes,
    size=[1, 5, 10],
    fill_value=[None, 0, "slice"],
  )
  def testUniqueSize(self, shape, dtype, axis, size, fill_value):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    kwds = dict(axis=axis, return_index=True, return_inverse=True, return_counts=True)

    if fill_value == "slice":
      if axis is None:
        fill_value = rng((), dtype)
      else:
        fill_value = rng(shape[:axis] + shape[axis + 1:], dtype)
    elif fill_value is not None:
      fill_value = np.array(fill_value).astype(dtype)

    @partial(jtu.with_jax_dtype_defaults, use_defaults=(False, True, True, True))
    def np_fun(x, fill_value=fill_value):
      u, ind, inv, counts = np_unique_backport(x, **kwds)
      axis = kwds['axis']
      if axis is None:
        x = x.ravel()
        axis = 0

      n_unique = u.shape[axis]
      if size <= u.shape[axis]:
        slc = (slice(None),) * axis + (slice(size),)
        u, ind, counts = u[slc], ind[:size], counts[:size]
      else:
        extra = (0, size - n_unique)
        pads = [(0, 0)] * u.ndim
        pads[axis] = extra
        u = np.pad(u, pads, constant_values=0)
        slices = [slice(None)] * u.ndim
        slices[axis] = slice(1)
        if fill_value is None:
          fill_value = u[tuple(slices)]
        elif np.ndim(fill_value):
          fill_value = lax.expand_dims(fill_value, (axis,))
        slices[axis] = slice(n_unique, None)
        u[tuple(slices)] = fill_value
        ind = np.pad(ind, extra, constant_values=ind[0])
        counts = np.pad(counts, extra, constant_values=0)
      return u, ind, inv, counts

    jnp_fun = lambda x: jnp.unique(x, size=size, fill_value=fill_value, **kwds)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(dtype=inexact_dtypes)
  def testUniqueNans(self, dtype):
    def args_maker():
      x = [-0.0, 0.0, 1.0, 1.0, np.nan, -np.nan]
      if np.issubdtype(dtype, np.complexfloating):
        x = [complex(i, j) for i, j in itertools.product(x, repeat=2)]
      return [np.array(x, dtype=dtype)]

    kwds = dict(return_index=True, return_inverse=True, return_counts=True)
    jnp_fun = partial(jnp.unique, **kwds)
    def np_fun(x):
      dtype = x.dtype
      # numpy unique fails for bfloat16 NaNs, so we cast to float64
      if x.dtype == jnp.bfloat16:
        x = x.astype('float64')
      u, *rest = np.unique(x, **kwds)
      return (u.astype(dtype), *rest)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @jtu.sample_product(dtype=inexact_dtypes, equal_nan=[True, False])
  @jtu.ignore_warning(
      category=RuntimeWarning, message='invalid value encountered in cast'
  )
  def testUniqueEqualNan(self, dtype, equal_nan):
    shape = (20,)
    rng = jtu.rand_some_nan(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    def np_fun(x):
      dtype = x.dtype
      # numpy unique fails for bfloat16 NaNs, so we cast to float64
      if x.dtype == jnp.bfloat16:
        x = x.astype('float64')
      return np.unique(x, equal_nan=equal_nan).astype(dtype)
    jnp_fun = partial(jnp.unique, equal_nan=equal_nan)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
