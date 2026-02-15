# Copyright 2026 The JAX Authors.
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

"""Tests for experimental searchsorted primitive."""

import functools

from absl.testing import absltest
import jax
import numpy as np

from jax._src import config
from jax._src.numpy import hijax
from jax._src import test_util as jtu

config.parse_flags_with_absl()

_SIDES = ("left", "right")
_DTYPES = ("int32", "float32", "complex64")
_METHODS = ("compare_all", "scan", "scan_unrolled", "sort")
_QUERY_SHAPES = ((), (5,), (5, 7))
_SORTED_ARR_SHAPES = ((10,), (10, 11), (10, 11, 12))
_BATCH_SHAPES = ((), (3,), (3, 4))

searchsorted_jit = jax.jit(
    hijax.searchsorted,
    static_argnames=["side", "dimension", "batch_dims", "method"],
)


def searchsorted_reference(
    sorted_arr: np.ndarray,
    query: np.ndarray,
    *,
    side: str = "left",
    method: str | None = None,
    dimension: int = 0,
    batch_dims: int = 0,
    dtype: np.dtype = np.int32,
) -> np.ndarray:
  """Reference implementation of the searchsorted primitive in terms of numpy.

  Args:
    sorted_arr: N-dimensional array, which is assumed to be sorted in increasing
      order along ``dimension``.
    query: M-dimensional array of query values.
    side: {'left', 'right'}. If 'left', find the index of the first suitable
      location. If 'right', find the index of the last.
    method: unused.
    dimension: positive integer specifying the dimension along which to insert
      query values.
    batch_dims: integer specifying the number of leading dimensions of
      `sorted_arr` and `query` to treat as batch dimensions.

  Returns:
    An array specifying the insertion locations of `query` into `sorted_arr`.
  """
  del method  # unused in numpy.searchsorted.

  if batch_dims < 0 or batch_dims >= sorted_arr.ndim:
    raise ValueError(
        f"batch_dims={batch_dims} must be in range [0, {sorted_arr.ndim})"
    )

  if dimension < batch_dims or dimension >= sorted_arr.ndim:
    raise ValueError(
        f"dimension={dimension} must be in range [{batch_dims},"
        f" {sorted_arr.ndim})"
    )

  if sorted_arr.shape[:batch_dims] != query.shape[:batch_dims]:
    raise ValueError(
        "batch dimension sizes must match; got"
        f" {sorted_arr.shape[:batch_dims]} != {query.shape[:batch_dims]}"
    )

  # Recursive call to handle common batch dimensions.
  if batch_dims > 0:
    return np.array([
        searchsorted_reference(
            a, q, side=side, dimension=dimension - 1, batch_dims=batch_dims - 1
        )
        for a, q in zip(sorted_arr, query)
    ])

  # Recursive call to handle sorted array batch dimensions.
  if sorted_arr.ndim > 1:
    sorted_arr = np.moveaxis(sorted_arr, dimension, 1)
    return np.array(
        [searchsorted_reference(a, query, side=side) for a in sorted_arr]
    )

  # Base case: batched query handled by numpy.
  assert sorted_arr.ndim == 1
  return np.searchsorted(sorted_arr, query, side=side).astype(dtype)


class SearchsortedTest(jtu.JaxTestCase):

  @jtu.sample_product(
      dtype=_DTYPES,
      out_dtype=['int32', 'uint32']
  )
  def test_out_dtype(self, dtype, out_dtype):
    rand_dtype = jtu.rand_default(self.rng())
    sorted_arr = np.sort(rand_dtype((100,), dtype))
    query = rand_dtype((100,), dtype)
    result = hijax.searchsorted(sorted_arr, query, dtype=out_dtype)
    self.assertEqual(result.dtype, np.dtype(out_dtype))

  @jtu.sample_product(
      dtype=_DTYPES,
      side=_SIDES,
      method=_METHODS,
  )
  def test_1D_against_numpy(self, dtype, side, method):
    rand_dtype = jtu.rand_default(self.rng())
    sorted_arr = np.sort(rand_dtype((100,), dtype))
    query = rand_dtype((100,), dtype)
    expected = np.searchsorted(sorted_arr, query, side=side)

    with self.subTest(name="non-jit"):
      actual = hijax.searchsorted(
          sorted_arr, query, side=side, method=method
      )
      np.testing.assert_array_equal(expected, actual)

    with self.subTest(name="jit"):
      actual = searchsorted_jit(sorted_arr, query, side=side, method=method)
      np.testing.assert_array_equal(expected, actual)

  @jtu.sample_product(
      side=_SIDES,
      method=_METHODS,
  )
  def test_2D_single_query_against_numpy(self, side, method):
    sorted_arr = np.sort(
        self.rng().uniform(size=(10, 100), low=0, high=10).round(1), axis=1
    )
    query = self.rng().uniform(size=(50,), low=-0.5, high=10.5).round(1)
    expected = np.array(
        [np.searchsorted(arr, query, side=side) for arr in sorted_arr]
    )

    with self.subTest(name="non-jit"):
      actual = hijax.searchsorted(
          sorted_arr, query, dimension=1, side=side, method=method
      )
      np.testing.assert_array_equal(expected, actual)

    with self.subTest(name="jit"):
      actual = searchsorted_jit(
          sorted_arr, query, dimension=1, side=side, method=method
      )
      np.testing.assert_array_equal(expected, actual)

  @jtu.sample_product(
      side=_SIDES,
      method=_METHODS,
  )
  def test_2D_batched_query_against_numpy(self, side, method):
    sorted_arr = np.sort(
        self.rng().uniform(size=(10, 100), low=0, high=10).round(1), axis=1
    )
    query = self.rng().uniform(size=(10, 20, 30), low=-0.5, high=10.5).round(1)
    expected = np.array(
        [np.searchsorted(a, q, side=side) for a, q in zip(sorted_arr, query)]
    )

    with self.subTest(name="non-jit"):
      actual = hijax.searchsorted(
          sorted_arr, query, dimension=1, batch_dims=1, side=side, method=method
      )
      np.testing.assert_array_equal(expected, actual)

    with self.subTest(name="jit"):
      actual = searchsorted_jit(
          sorted_arr, query, dimension=1, batch_dims=1, side=side, method=method
      )
      np.testing.assert_array_equal(expected, actual)

  @jtu.sample_product(
      side=_SIDES,
      method=_METHODS,
      query_shape=_QUERY_SHAPES,
      sorted_arr_shape=_SORTED_ARR_SHAPES,
      batch_shape=_BATCH_SHAPES,
  )
  def test_all_shapes(
      self, side, method, query_shape, sorted_arr_shape, batch_shape
  ):
    rand_dtype = jtu.rand_default(self.rng())
    batch_dims = len(batch_shape)
    dimension = batch_dims + int(self.rng().randint(0, len(sorted_arr_shape)))
    sorted_arr = np.sort(
        rand_dtype((*batch_shape, *sorted_arr_shape), np.float32),
        axis=dimension,
    )
    query = rand_dtype((*batch_shape, *query_shape), np.float32)
    kwds = dict(
        side=side,
        dimension=dimension,
        batch_dims=batch_dims,
        method=method,
    )
    expected = searchsorted_reference(sorted_arr, query, **kwds)
    actual = hijax.searchsorted(sorted_arr, query, **kwds)
    actual_jit = searchsorted_jit(sorted_arr, query, **kwds)

    with self.subTest(name="non-jit"):
      np.testing.assert_array_equal(expected, actual)

    with self.subTest(name="jit"):
      np.testing.assert_array_equal(expected, actual_jit)

  def test_input_validation(self):
    with self.subTest(name="invalid_side"):
      sorted_arr = np.arange(10)
      query = np.arange(10)
      with self.assertRaisesWithLiteralMatch(
          ValueError, "invalid argument side='none', expected 'left' or 'right'"
      ):
        hijax.searchsorted(sorted_arr, query, side="none")

    with self.subTest(name="invalid_batch_dims"):
      sorted_arr = np.arange(10)
      query = np.arange(10)
      with self.assertRaisesWithLiteralMatch(
          ValueError, "batch_dims=-1 must be in range [0, 1)"
      ):
        hijax.searchsorted(sorted_arr, query, batch_dims=-1)

    with self.subTest(name="invalid_dimension"):
      sorted_arr = np.arange(10)
      query = np.arange(10)
      with self.assertRaisesWithLiteralMatch(
          ValueError, "dimension=2 must be in range [0, 1)"
      ):
        hijax.searchsorted(sorted_arr, query, dimension=2)

    with self.subTest(name="batch_dimension_size_mismatch"):
      sorted_arr = np.zeros((2, 5))
      query = np.zeros((3, 6))
      with self.assertRaisesWithLiteralMatch(
          ValueError, "batch dimension sizes must match; got (2,) != (3,)"
      ):
        hijax.searchsorted(sorted_arr, query, batch_dims=1, dimension=1)

    with self.subTest(name="dtype_mismatch"):
      sorted_arr = np.zeros(5, dtype="float32")
      query = np.zeros(5, dtype="int32")
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          "dtypes of sorted_arr and query must match; got float32 and int32",
      ):
        hijax.searchsorted(sorted_arr, query)

    with self.subTest(name="invalid_method"):
      sorted_arr = np.arange(5)
      query = 2
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          "invalid argument method='none', expected one of ['compare_all',"
          " 'scan', 'scan_unrolled', 'sort']",
      ):
        hijax.searchsorted(sorted_arr, query, method="none")

  @jtu.sample_product(
      side=_SIDES,
      method=_METHODS,
  )
  def test_vmap(self, side, method):
    with self.subTest(name="batched sorted_arr"):
      sorted_arr = np.sort(
          self.rng().uniform(size=(10, 100), low=0, high=10).round(1), axis=1
      )
      query = self.rng().uniform(size=50, low=-0.5, high=10.5).round(1)
      expected = np.array(
          [np.searchsorted(a, query, side=side) for a in sorted_arr]
      )
      actual = jax.vmap(
          functools.partial(
              hijax.searchsorted, side=side, method=method
          ),
          in_axes=(0, None),
          out_axes=0,
      )(sorted_arr, query)
      np.testing.assert_array_equal(expected, actual)

    with self.subTest(name="batched query"):
      sorted_arr = np.sort(self.rng().uniform(size=100, low=0, high=10).round(1))
      query = self.rng().uniform(size=(10, 50), low=-0.5, high=10.5).round(1)
      expected = np.array(
          [np.searchsorted(sorted_arr, q, side=side) for q in query]
      )
      actual = jax.vmap(
          functools.partial(
              hijax.searchsorted, side=side, method=method
          ),
          in_axes=(None, 0),
          out_axes=0,
      )(sorted_arr, query)
      np.testing.assert_array_equal(expected, actual)

    with self.subTest(name="batched sorted_arr and query"):
      sorted_arr = np.sort(
          self.rng().uniform(size=(10, 100), low=0, high=10).round(1), axis=1
      )
      query = self.rng().uniform(size=(10, 50), low=-0.5, high=10.5).round(1)
      expected = np.array(
          [np.searchsorted(a, q, side=side) for a, q in zip(sorted_arr, query)]
      )
      actual = jax.vmap(
          functools.partial(
              hijax.searchsorted,
              side=side,
              method=method,
          )
      )(sorted_arr, query)
      np.testing.assert_array_equal(expected, actual)

  def test_vmap_unbatched(self):
    """Test batched case where neither input is batched"""
    x = np.array([2, 1, 3], dtype='int32')
    y = np.array([2, 3, 4, 5])
    z = np.array([4, 3])
    def f(x, y, z):
      return x * hijax.searchsorted(y, z)
    result = jax.vmap(f, (0, None, None))(x, y, z)
    expected = x[:, None] * hijax.searchsorted(y, z)[None, :]
    self.assertArraysEqual(result, expected)

  def test_nested_vmap(self):
    x = np.arange(60).reshape(2, 3, 10)
    y = np.arange(5, 65, 10).reshape(2, 3)
    out = jax.vmap(jax.vmap(hijax.searchsorted))(x, y)
    expected = jax.numpy.array([
      [hijax.searchsorted(x, y) for x, y in zip(xrow, yrow)]
      for xrow, yrow in zip(x, y)
    ])
    self.assertArraysEqual(out, expected)

  @jtu.sample_product(
      side=_SIDES,
      method=_METHODS,
  )
  def test_autodiff(self, side, method, dtype="float32"):
    rand_dtype = jtu.rand_default(self.rng())
    sorted_arr = np.sort(rand_dtype((100,), dtype))
    query = rand_dtype((100,), dtype)
    func = functools.partial(
        hijax.searchsorted, side=side, method=method
    )
    primal_expected = func(sorted_arr, query)

    with self.subTest(name="forward mode"):
      primals = (sorted_arr, query)
      tangents = (np.ones_like(sorted_arr), np.ones_like(query))
      primal_out, tangent_out = jax.jvp(func, primals, tangents)
      np.testing.assert_array_equal(primal_expected, primal_out, strict=True)
      self.assertEqual(tangent_out.shape, primal_out.shape)
      self.assertEqual(tangent_out.dtype, jax.dtypes.float0)

    with self.subTest(name="reverse mode"):
      primal_out, f_vjp = jax.vjp(func, sorted_arr, query)
      tangent_out = np.ones_like(primal_out)
      tangents = f_vjp(tangent_out)
      np.testing.assert_array_equal(primal_expected, primal_out, strict=True)
      np.testing.assert_array_equal(
          tangents[0], np.zeros_like(sorted_arr), strict=True
      )
      np.testing.assert_array_equal(
          tangents[1], np.zeros_like(query), strict=True
      )

  def test_reference_batch_dims_out_of_range(self):
    sorted_arr = np.arange(10)
    query = np.arange(10)
    with self.assertRaisesWithLiteralMatch(
        ValueError, "batch_dims=2 must be in range [0, 1)"
    ):
      searchsorted_reference(sorted_arr, query, batch_dims=2)

  def test_reference_dimension_out_of_range(self):
    sorted_arr = np.arange(10)
    query = np.arange(10)
    with self.assertRaisesWithLiteralMatch(
        ValueError, "dimension=2 must be in range [0, 1)"
    ):
      searchsorted_reference(sorted_arr, query, dimension=2)

  def test_reference_batch_dimension_size_mismatch(self):
    sorted_arr = np.zeros((2, 5))
    query = np.zeros((3, 6))
    with self.assertRaisesWithLiteralMatch(
        ValueError, "batch dimension sizes must match; got (2,) != (3,)"
    ):
      searchsorted_reference(
          sorted_arr, query, batch_dims=1, dimension=1
      )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
