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

"""Tests for Pallas indexing logic and abstractions."""
from typing import Union
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import util
from jax._src.pallas import indexing
import numpy as np

try:
  import hypothesis as hp
except (ModuleNotFoundError, ImportError):
  raise unittest.SkipTest("tests depend on hypothesis library")

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as hps
hp.settings.register_profile(
    "deterministic", database=None, derandomize=True, deadline=None,
    max_examples=100, print_blob=True)
hp.settings.load_profile("deterministic")

Slice = indexing.Slice
NDIndexer = indexing.NDIndexer
ds = indexing.ds


def int_indexer_strategy(dim) -> hps.SearchStrategy[int]:
  return hps.integers(min_value=np.iinfo(np.int32).min, max_value=dim - 1)


@hps.composite
def slice_indexer_strategy(draw, dim) -> Union[Slice, slice]:
  start = draw(int_indexer_strategy(dim))
  size = draw(hps.integers(min_value=0, max_value=np.iinfo(np.int32).max))
  return draw(
      hps.one_of(
          hps.just(Slice(start, size)), hps.just(slice(start, start + size))
      )
  )


@hps.composite
def array_indexer_strategy(draw, shape) -> jax.Array:
  unbcast = [draw(hps.booleans()) for _ in shape]
  shape = tuple(1 if unb else s for unb, s in zip(unbcast, shape))
  return draw(hnp.arrays(dtype=np.dtype("int32"), shape=shape))


@hps.composite
def indexer_strategy(draw, dim, int_indexer_shape
                     ) -> Union[int, Slice, jax.Array]:
  return draw(hps.one_of(
      int_indexer_strategy(dim),
      slice_indexer_strategy(dim),
      array_indexer_strategy(int_indexer_shape),
  ))


@hps.composite
def nd_indexer_strategy(draw, shape) -> NDIndexer:
  num_indices = draw(hps.integers(min_value=0, max_value=len(shape)))
  int_indexer_shape = draw(hnp.array_shapes())
  indices = [draw(indexer_strategy(dim, int_indexer_shape)) for dim
             in shape[:num_indices]]
  return NDIndexer.from_indices_shape(indices, shape)


class IndexerTest(parameterized.TestCase):

  def test_simple_ndindexer(self):
    indices = (0, 0)
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), ())

  def test_invalid_ndindexer(self):
    indices = (0, 0, 0)
    shape = (5, 5)
    with self.assertRaises(ValueError):
      _ = NDIndexer.from_indices_shape(indices, shape)

  def test_ndindexer_with_padding(self):
    indices = ()
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), shape)

  def test_ndindexer_with_slices(self):
    indices = (slice(2, 3), slice(4, 7))
    shape = (5, 6)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (1, 2))

  def test_ndindexer_with_arrays(self):
    indices = (np.arange(10), np.arange(10))
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (10,))

    indices = (np.ones((10, 20)), np.ones((10, 20)))
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (10, 20))

  def test_ndindexer_with_arrays_and_broadcasting(self):
    indices = (np.arange(10)[None], np.arange(20)[:, None])
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (20, 10))

    indices = (np.arange(10)[:, None], np.arange(20)[None, :])
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (10, 20))

  def test_indexer_with_all_types(self):
    indices = (0, slice(10), np.arange(5))
    shape = (2, 3, 4)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (5, 3))

    indices = (0, slice(4, 10), np.arange(5))
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (5, 0))

    indices = (0, 5, np.arange(5))
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (5,))

    indices = (ds(2, 3), np.arange(5)[:, None], np.arange(4)[None])
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (5, 4, 3))

  @hp.given(hps.data())
  def test_ndindexer(self, data):
    shape = data.draw(hnp.array_shapes())
    indexer = data.draw(nd_indexer_strategy(shape))
    is_int_indexer = [not isinstance(idx, Slice) for idx in indexer.indices]
    rest_indexers, int_indexers = util.partition_list(
        is_int_indexer, indexer.indices
    )
    if int_indexers:
      expected_int_indexer_shape = int_indexers[0].shape
    else:
      expected_int_indexer_shape = ()
    self.assertTupleEqual(
        indexer.int_indexer_shape, expected_int_indexer_shape
    )
    for idx in rest_indexers:
      self.assertIsInstance(idx, (np.ndarray, Slice))
      if isinstance(idx, np.ndarray):
        self.assertTupleEqual(idx.shape, ())
        self.assertEqual(idx.dtype, np.dtype("int32"))
    rest_shape = tuple(
        r.size for r in rest_indexers if not isinstance(r, np.ndarray)
    )
    self.assertTupleEqual((*indexer.int_indexer_shape, *rest_shape),
                          indexer.get_indexer_shape())


if __name__ == "__main__":
  absltest.main()
