# Copyright 2018 Google LLC
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
import enum
from functools import partial
import itertools
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp

from jax import api
from jax import lax
from jax import numpy as jnp
from jax import ops
from jax import test_util as jtu
from jax import util

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

# We disable the whitespace continuation check in this file because otherwise it
# makes the test name formatting unwieldy.
# pylint: disable=bad-continuation


float_dtypes = [onp.float32, onp.float64]
int_dtypes = [onp.int32, onp.int64]
bool_types = [onp.bool_]
default_dtypes = float_dtypes + int_dtypes
all_dtypes = float_dtypes + int_dtypes + bool_types

IndexSpec = collections.namedtuple("IndexTest", ["shape", "indexer"])


suppress_deprecated_indexing_warnings = partial(
  jtu.ignore_warning, category=FutureWarning,
  message='Using a non-tuple sequence.*')


def check_grads(f, args, order, atol=None, rtol=None, eps=None):
  # TODO(mattjj,dougalm): add higher-order check
  default_tol = 1e-6 if FLAGS.jax_enable_x64 else 1e-2
  atol = atol or default_tol
  rtol = rtol or default_tol
  eps = eps or default_tol
  jtu.check_jvp(f, partial(api.jvp, f), args, atol, rtol, eps)
  jtu.check_vjp(f, partial(api.vjp, f), args, atol, rtol, eps)


STATIC_INDEXING_TESTS = [
    ("OneIntIndex", [
        IndexSpec(shape=(3,), indexer=1),
        IndexSpec(shape=(3, 3), indexer=0),
        IndexSpec(shape=(3, 4, 5), indexer=2),
        IndexSpec(shape=(3,), indexer=-1),
        IndexSpec(shape=(3,), indexer=-2),
    ]),
    ("TwoIntIndices", [
        IndexSpec(shape=(3, 3), indexer=(2, 1)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
        IndexSpec(shape=(3, 4, 5), indexer=(-1, 2)),
    ]),
    ("ThreeIntIndices", [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
    ("OneSliceIndex", [
        IndexSpec(shape=(10,), indexer=slice(1, 3)),
        IndexSpec(shape=(10,), indexer=slice(1, -1)),
        IndexSpec(shape=(10,), indexer=slice(None, -1)),
        IndexSpec(shape=(10,), indexer=slice(None, None, None)),
        IndexSpec(shape=(10, 8), indexer=slice(1, 3)),
        IndexSpec(shape=(10, 8), indexer=slice(1, None)),
        IndexSpec(shape=(10, 8), indexer=slice(None, 3)),
        IndexSpec(shape=(10, 8), indexer=slice(-3, None)),
    ]),
    ("OneSliceIndexNegativeStride", [
        IndexSpec(shape=(10,), indexer=slice(3, 1, -1)),
        IndexSpec(shape=(10,), indexer=slice(1, 8, -1)),  # empty result
        IndexSpec(shape=(10,), indexer=slice(None, 1, -2)),
        IndexSpec(shape=(10,), indexer=slice(None, None, -1)),
        IndexSpec(shape=(10, 8), indexer=slice(3, 1, -1)),
        IndexSpec(shape=(10, 8), indexer=slice(0, 8, -1)),  # empty result
        IndexSpec(shape=(10, 8), indexer=slice(None, None, -1)),
    ]),
    ("OneSliceIndexNonUnitStride", [
        IndexSpec(shape=(10,), indexer=slice(0, 8, 2)),
        IndexSpec(shape=(10,), indexer=slice(0, 8, 3)),
        IndexSpec(shape=(10,), indexer=slice(1, 3, 2)),
        IndexSpec(shape=(10,), indexer=slice(1, None, 2)),
        IndexSpec(shape=(10,), indexer=slice(None, 1, -2)),
        IndexSpec(shape=(10, 8), indexer=slice(1, 8, 3)),
        IndexSpec(shape=(10, 8), indexer=slice(None, None, 2)),
        IndexSpec(shape=(10, 8), indexer=slice(None, 1, -2)),
        IndexSpec(shape=(10, 8), indexer=slice(None, None, -2)),
    ]),
    ("TwoSliceIndices", [
        IndexSpec(shape=(10, 8), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(10, 8), indexer=(slice(1, None), slice(None, 2))),
        IndexSpec(
            shape=(10, 8), indexer=(slice(None, None, -1), slice(None, 2))),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, 3), slice(0, None))),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, None), slice(0, 2))),
    ]),
    ("OneColonIndex", [
        IndexSpec(shape=(3,), indexer=slice(None)),
        IndexSpec(shape=(3, 4), indexer=slice(None)),
    ]),
    ("MultipleColonIndices", [
        IndexSpec(shape=(3, 4), indexer=(slice(None), slice(None))),
        IndexSpec(shape=(3, 4, 5), indexer=(slice(None), slice(None))),
    ]),
    ("MixedSliceIndices", [
        IndexSpec(shape=(10, 4), indexer=(slice(None), slice(0, 2))),
        IndexSpec(shape=(10, 4), indexer=(1, slice(None))),
    ]),
    ("EllipsisIndex", [
        IndexSpec(shape=(3,), indexer=Ellipsis),
        IndexSpec(shape=(3, 4), indexer=Ellipsis),
        IndexSpec(shape=(3, 4, 5), indexer=(0, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis, 2, 3)),
    ]),
    ("NoneIndex", [
        IndexSpec(shape=(), indexer=None),
        IndexSpec(shape=(), indexer=(None, None)),
        IndexSpec(shape=(), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3,), indexer=None),
        IndexSpec(shape=(3, 4), indexer=None),
        IndexSpec(shape=(3, 4), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3, 4), indexer=(0, None, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, None, Ellipsis)),
    ]),
    ("EmptyIndex", [
        IndexSpec(shape=(), indexer=()),
        IndexSpec(shape=(3,), indexer=()),
        IndexSpec(shape=(3, 4), indexer=()),
    ]),
]

STATIC_INDEXING_GRAD_TESTS = [
    ("OneIntIndex", [
        IndexSpec(shape=(3,), indexer=1),
        IndexSpec(shape=(3, 3), indexer=0),
        IndexSpec(shape=(3, 4, 5), indexer=2),
        IndexSpec(shape=(3,), indexer=-1),
        IndexSpec(shape=(3,), indexer=-2),
    ]),
    ("TwoIntIndices", [
        IndexSpec(shape=(3, 3), indexer=(2, 1)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
        IndexSpec(shape=(3, 4, 5), indexer=(-1, 2)),
    ]),
    ("ThreeIntIndices", [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
    ("OneSliceIndex", [
        IndexSpec(shape=(5,), indexer=slice(1, 3)),
        IndexSpec(shape=(5,), indexer=slice(1, -1)),
        IndexSpec(shape=(5,), indexer=slice(None, -1)),
        IndexSpec(shape=(5,), indexer=slice(None, None, None)),
        IndexSpec(shape=(5, 4), indexer=slice(1, 3)),
        IndexSpec(shape=(5, 4), indexer=slice(1, None)),
        IndexSpec(shape=(5, 4), indexer=slice(None, 3)),
        IndexSpec(shape=(5, 4), indexer=slice(-3, None)),
    ]),
    ("TwoSliceIndices", [
        IndexSpec(shape=(5, 4), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(5, 4), indexer=(slice(1, None), slice(None, 2))),
        IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, None))),
        IndexSpec(shape=(5, 4, 3), indexer=(slice(1, None), slice(0, 2))),
    ]),
    ("OneColonIndex", [
        IndexSpec(shape=(3,), indexer=slice(None)),
        IndexSpec(shape=(3, 4), indexer=slice(None)),
    ]),
    ("MultipleColonIndices", [
        IndexSpec(shape=(3, 4), indexer=(slice(None), slice(None))),
        IndexSpec(shape=(3, 4, 5), indexer=(slice(None), slice(None))),
    ]),
    ("MixedSliceIndices", [
        IndexSpec(shape=(5, 4), indexer=(slice(None), slice(0, 2))),
        IndexSpec(shape=(5, 4), indexer=(1, slice(None))),
    ]),
    ("EllipsisIndex", [
        IndexSpec(shape=(3,), indexer=Ellipsis),
        IndexSpec(shape=(3, 4), indexer=Ellipsis),
        IndexSpec(shape=(3, 4, 5), indexer=(0, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis, 2, 3)),
    ]),
    ("NoneIndex", [
        IndexSpec(shape=(), indexer=None),
        IndexSpec(shape=(), indexer=(None, None)),
        IndexSpec(shape=(), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3,), indexer=None),
        IndexSpec(shape=(3, 4), indexer=None),
        IndexSpec(shape=(3, 4), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3, 4), indexer=(0, None, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, None, Ellipsis)),
    ]),
    # TODO(mattjj): these fail for uninteresting dtype reasons
    # ("EmptyIndex",
    #  [IndexSpec(shape=(), indexer=()),
    #   IndexSpec(shape=(3,), indexer=()),
    #   IndexSpec(shape=(3, 4), indexer=()),
    #   ]),
]

ADVANCED_INDEXING_TESTS = [
    ("One1DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([0, 1])),
     IndexSpec(shape=(3, 3), indexer=onp.array([1, 2, 1])),
     IndexSpec(shape=(3, 4, 5), indexer=onp.array([0, 2, 0, 1])),
     IndexSpec(shape=(3,), indexer=onp.array([-1, 1])),
     IndexSpec(shape=(3,), indexer=onp.array([-2, -1])),
     IndexSpec(shape=(0,), indexer=onp.array([], dtype=onp.int32)),
     ]),
    ("One2DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([[0, 0]])),
     IndexSpec(shape=(3, 3), indexer=onp.array([[1, 2, 1],
                                                [0, 1, -1]])),
     IndexSpec(shape=(3, 4, 5), indexer=onp.array([[0, 2, 0, 1],
                                                   [-1, -2, 1, 0]])),
     ]),
    ("Two1DIntArrayIndicesNoBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=[onp.array([0, 1]),
                                       onp.array([1, 2])]),
     IndexSpec(shape=(3, 4, 5), indexer=[onp.array([0, 2, 0, 1]),
                                         onp.array([-1, 0, -1, 2])]),
     ]),
    ("Two1DIntArrayIndicesWithBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=[onp.array([[0, 1]]),
                                       onp.array([1, 2])]),
     IndexSpec(shape=(3, 4, 5), indexer=[onp.array([[0, 2, 0, 1]]),
                                         onp.array([-1, 0, -1, 2])]),
     ]),
    ("ListOfPythonInts",
     [IndexSpec(shape=(3,), indexer=[0, 1, 0]),
     IndexSpec(shape=(3, 4, 5), indexer=[0, -1]),
     ]),
    ("ListOfListsOfPythonInts",
     [IndexSpec(shape=(3, 4, 5), indexer=[[0, 1]]),
     IndexSpec(shape=(3, 4, 5), indexer=[[[0], [-1]], [[2, 3, 0, 3]]]),
     ]),
    ("TupleOfListsOfPythonInts",
     [IndexSpec(shape=(3, 4, 5), indexer=([0, 1])),
     IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[2, 3, 0, 3]])),
     ]),
    ("ListOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=[0, onp.array([0, 1])]),
     IndexSpec(shape=(3, 4, 5), indexer=[0, 1,
                                         onp.array([[2, 3, 0, 3]])]),
     ]),
    ("ListOfListsOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=[[0, 1], onp.array([0])]),
     IndexSpec(shape=(3, 4, 5), indexer=[[[0], [-1]],
                                         onp.array([[2, 3, 0, 3]])]),
     ]),
]

ADVANCED_INDEXING_TESTS_NO_REPEATS = [
    ("One1DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([0, 1])),
      IndexSpec(shape=(3, 3), indexer=onp.array([1, 2, 0])),
      IndexSpec(shape=(3, 4, 5), indexer=onp.array([0, 2, 1])),
      IndexSpec(shape=(3,), indexer=onp.array([-1, 1])),
      IndexSpec(shape=(3,), indexer=onp.array([-2, -1])),
      IndexSpec(shape=(0,), indexer=onp.array([], dtype=onp.int32)),
     ]),
    ("One2DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([[0, 1]])),
      IndexSpec(shape=(6, 6), indexer=onp.array([[1, 2, 0],
                                                 [3, 4, -1]])),
     ]),
    ("Two1DIntArrayIndicesNoBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=[onp.array([0, 1]),
                                       onp.array([1, 2])]),
      IndexSpec(shape=(4, 5, 6), indexer=[onp.array([0, 2, 1, 3]),
                                          onp.array([-1, 0, -2, 1])]),
     ]),
    ("Two1DIntArrayIndicesWithBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=[onp.array([[0, 1]]),
                                       onp.array([1, 2])]),
      IndexSpec(shape=(4, 5, 6), indexer=[onp.array([[0, 2, -1, 1]]),
                                          onp.array([-1, 0, -2, 2])]),
     ]),
    ("ListOfPythonInts",
     [IndexSpec(shape=(3,), indexer=[0, 2, 1]),
      IndexSpec(shape=(3, 4, 5), indexer=[0, -1]),
     ]),
    ("ListOfListsOfPythonInts",
     [IndexSpec(shape=(3, 4, 5), indexer=[[0, 1]]),
      IndexSpec(shape=(3, 4, 5), indexer=[[[0], [-1]], [[2, 3, 0]]]),
     ]),
    ("TupleOfListsOfPythonInts",
     [IndexSpec(shape=(3, 4, 5), indexer=([0, 1])),
      IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[2, 3, 0]])),
     ]),
    ("ListOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=[0, onp.array([0, 1])]),
      IndexSpec(shape=(3, 4, 5), indexer=[0, 1,
                                          onp.array([[2, 3, 0]])]),
     ]),
    ("ListOfListsOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=[[0, 1], onp.array([0])]),
      IndexSpec(shape=(3, 4, 5), indexer=[[[0], [-1]],
                                          onp.array([[2, 3, 0]])]),
     ]),
]

MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS = [
    ("SlicesAndOneIntArrayIndex",
     [IndexSpec(shape=(2, 3), indexer=(onp.array([0, 1]), slice(1, 2))),
     IndexSpec(shape=(2, 3), indexer=(slice(0, 2),
                                      onp.array([0, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([0, 2]),
                                         slice(None))),
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([[0, 2], [1, 3]]),
                                         slice(None))),
     ]),
    ("SlicesAndTwoIntArrayIndices",
     [IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                          onp.array([0, 2]),
                                          onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         Ellipsis,
                                         onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         onp.array([-1, 2]),
                                         Ellipsis)),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         onp.array([-1, 2]),
                                         slice(1, 3))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         slice(1, 3),
                                         onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2, -2]),
                                         slice(None, None, 2),
                                         onp.array([-1, 2, 1]))),
     ]),
    ("NonesAndIntArrayIndices",
     [IndexSpec(shape=(3, 4, 5), indexer=[onp.array([0, 2]),
                                          None,
                                          onp.array([-1, 2])]),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         None,
                                         None,
                                         onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([0, 2]),
                                         None,
                                         None,
                                         onp.array([-1, 2]))),
     ]),
    ("IntArrayWithInt32Type",
     [IndexSpec(shape=(3, 4), indexer=(Ellipsis, onp.array(1, dtype=onp.int32)))
     ]),
]

MIXED_ADVANCED_INDEXING_TESTS = MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS + [
    ("SlicesAndOneIntArrayIndex",
     [
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([[0, 2], [1, 1]]),
                                         slice(None))),
     ]),
    ("SlicesAndTwoIntArrayIndices",
     [IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2, -2]),
                                         slice(None, None, 2),
                                         onp.array([-1, 2, -1]))),
      IndexSpec(shape=(3, 4, 5), indexer=(onp.array([[0, 2], [2, 0]]),
                                          Ellipsis,
                                          onp.array([[1, 0], [1, 0]]))),
     ]),]

class IndexingTest(jtu.JaxTestCase):
  """Tests for Numpy indexing translation rules."""

  @parameterized.named_parameters(jtu.cases_from_list({
      "testcase_name": "{}_inshape={}_indexer={}".format(
          name, jtu.format_shape_dtype_string( shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer
  } for name, index_specs in STATIC_INDEXING_TESTS
    for shape, indexer in index_specs
    for dtype in all_dtypes
    for rng_factory in [jtu.rand_default]))
  def testStaticIndexing(self, shape, dtype, rng_factory, indexer):
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype)]
    fun = lambda x: x[indexer]
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters({
      "testcase_name":
          "{}_inshape={}_indexer={}".format(name,
                                            jtu.format_shape_dtype_string(
                                                shape, dtype), indexer),
      "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer
  } for name, index_specs in STATIC_INDEXING_GRAD_TESTS
    for shape, indexer in index_specs
    for dtype in float_dtypes
    for rng_factory in [jtu.rand_default])
  def testStaticIndexingGrads(self, shape, dtype, rng_factory, indexer):
    rng = rng_factory()
    tol = 1e-2 if jnp.finfo(dtype).bits == 32 else None
    arg = rng(shape, dtype)
    fun = lambda x: x[indexer]**2
    check_grads(fun, (arg,), 2, tol, tol, tol)

  def _ReplaceSlicesWithTuples(self, idx):
    """Helper method to replace slices with tuples for dynamic indexing args."""
    if isinstance(idx, slice):
      triple = idx.start, idx.stop, idx.step
      isnone = [i for i, elt in enumerate(triple) if elt is None]
      zeros = itertools.repeat(0)
      nones = itertools.repeat(None)
      out = util.subvals(triple, zip(isnone, zeros))
      return out, lambda out: slice(*util.subvals(out, zip(isnone, nones)))
    elif isinstance(idx, (tuple, list)) and idx:
      t = type(idx)
      elts, packs = zip(*map(self._ReplaceSlicesWithTuples, idx))
      return elts, lambda elts: t((pack(i) for pack, i in zip(packs, elts)))
    else:
      return idx, lambda x: x

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer}
      for name, index_specs in [
          ("OneSliceIndex",
           [IndexSpec(shape=(5,), indexer=slice(1, 3)),
            IndexSpec(shape=(5, 4), indexer=slice(1, 3))]),
          ("TwoSliceIndices",
           [IndexSpec(shape=(5, 4), indexer=(slice(1, 3), slice(0, 2))),
            IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, 2)))]),
          ("NonUnitStrides", [
              IndexSpec(shape=(3,), indexer=slice(None, None, -1)),
              IndexSpec(shape=(3, 3), indexer=slice(0, 3, -2)),
              IndexSpec(shape=(3, 4, 5), indexer=slice(0, 4, 2))
          ]),
          ("OnlyStartOrStopDynamic", [
              IndexSpec(shape=(5, 4), indexer=(slice(None, 3), slice(0, 2))),
              IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, None)))
          ]),
      ]
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testDynamicIndexingWithSlicesErrors(self, shape, dtype, rng_factory, indexer):
    rng = rng_factory()
    unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

    @api.jit
    def fun(x, unpacked_indexer):
      indexer = pack_indexer(unpacked_indexer)
      return x[indexer]

    args_maker = lambda: [rng(shape, dtype), unpacked_indexer]
    self.assertRaises(IndexError, lambda: fun(*args_maker()))

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer}
      for name, index_specs in [
          ("OneIntIndex",
           [IndexSpec(shape=(3,), indexer=1),
            IndexSpec(shape=(3, 3), indexer=0),
            IndexSpec(shape=(3, 4, 5), indexer=2),
            IndexSpec(shape=(3,), indexer=-1),
            IndexSpec(shape=(3,), indexer=-2)]),
          ("TwoIntIndices",
           [IndexSpec(shape=(3, 3), indexer=(2, 1)),
            IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
            IndexSpec(shape=(3, 4, 5), indexer=(-1, 2))]),
          ("ThreeIntIndices",
           [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
      ]
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testDynamicIndexingWithIntegers(self, shape, dtype, rng_factory, indexer):
    rng = rng_factory()
    unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

    def fun(x, unpacked_indexer):
      indexer = pack_indexer(unpacked_indexer)
      return x[indexer]

    args_maker = lambda: [rng(shape, dtype), unpacked_indexer]
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer}
      for name, index_specs in [
          ("OneIntIndex",
           [IndexSpec(shape=(3,), indexer=1),
            IndexSpec(shape=(3, 3), indexer=0),
            IndexSpec(shape=(3, 4, 5), indexer=2),
            IndexSpec(shape=(3,), indexer=-1),
            IndexSpec(shape=(3,), indexer=-2),
            ]),
          ("TwoIntIndices",
           [IndexSpec(shape=(3, 3), indexer=(2, 1)),
            IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
            IndexSpec(shape=(3, 4, 5), indexer=(-1, 2)),
            ]),
          ("ThreeIntIndices",
           [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
      ]
      for shape, indexer in index_specs
      for dtype in float_dtypes
      for rng_factory in [jtu.rand_default])
  def testDynamicIndexingWithIntegersGrads(self, shape, dtype, rng_factory, indexer):
    rng = rng_factory()
    tol = 1e-2 if jnp.finfo(dtype).bits == 32 else None
    unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

    @api.jit
    def fun(unpacked_indexer, x):
      indexer = pack_indexer(unpacked_indexer)
      return x[indexer]

    arr = rng(shape, dtype)
    check_grads(partial(fun, unpacked_indexer), (arr,), 2, tol, tol, tol)

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer}
      for name, index_specs in ADVANCED_INDEXING_TESTS
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testAdvancedIntegerIndexing(self, shape, dtype, rng_factory, indexer):
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), indexer]
    fun = lambda x, idx: jnp.asarray(x)[idx]
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer}
      for name, index_specs in [
          ("One1DIntArrayIndex",
           [IndexSpec(shape=(3,), indexer=onp.array([0, 1])),
            IndexSpec(shape=(3, 3), indexer=onp.array([1, 2, 1])),
            IndexSpec(shape=(3, 4, 5), indexer=onp.array([0, 2, 0, 1])),
            IndexSpec(shape=(3,), indexer=onp.array([-1, 1])),
            IndexSpec(shape=(3,), indexer=onp.array([-2, -1])),
            ]),
          ("One2DIntArrayIndex",
           [IndexSpec(shape=(3,), indexer=onp.array([[0, 0]])),
            IndexSpec(shape=(3, 3), indexer=onp.array([[1, 2, 1],
                                                       [0, 1, -1]])),
            IndexSpec(shape=(3, 4, 5), indexer=onp.array([[0, 2, 0, 1],
                                                          [-1, -2, 1, 0]])),
            ]),
          ("Two1DIntArrayIndicesNoBroadcasting",
           [IndexSpec(shape=(3, 3), indexer=[onp.array([0, 1]),
                                             onp.array([1, 2])]),
            IndexSpec(shape=(3, 4, 5), indexer=[onp.array([0, 2, 0, 1]),
                                                onp.array([-1, 0, -1, 2])]),
            ]),
          ("Two1DIntArrayIndicesWithBroadcasting",
           [IndexSpec(shape=(3, 3), indexer=[onp.array([[0, 1]]),
                                             onp.array([1, 2])]),
            IndexSpec(shape=(3, 4, 5), indexer=[onp.array([[0, 2, 0, 1]]),
                                                onp.array([-1, 0, -1, 2])]),
            ]),
          ("ListOfPythonInts",
           [IndexSpec(shape=(3,), indexer=[0, 1, 0]),
            IndexSpec(shape=(3, 4, 5), indexer=[0, -1]),
            ]),
          ("ListOfListsOfPythonInts",
           [IndexSpec(shape=(3, 4, 5), indexer=[[0, 1]]),
            IndexSpec(shape=(3, 4, 5), indexer=[[[0], [-1]], [[2, 3, 0, 3]]]),
            ]),
          ("ListOfPythonIntsAndIntArrays",
           [IndexSpec(shape=(3, 4, 5), indexer=[0, onp.array([0, 1])]),
            IndexSpec(shape=(3, 4, 5), indexer=[0, 1,
                                                onp.array([[2, 3, 0, 3]])]),
            ]),
          ("ListOfListsOfPythonIntsAndIntArrays",
           [IndexSpec(shape=(3, 4, 5), indexer=[[0, 1], onp.array([0])]),
            IndexSpec(shape=(3, 4, 5), indexer=[[[0], [-1]],
                                                onp.array([[2, 3, 0, 3]])]),
            ]),
      ]
      for shape, indexer in index_specs
      for dtype in float_dtypes
      for rng_factory in [jtu.rand_default])
  def testAdvancedIntegerIndexingGrads(self, shape, dtype, rng_factory, indexer):
    rng = rng_factory()
    tol = 1e-2 if jnp.finfo(dtype).bits == 32 else None
    arg = rng(shape, dtype)
    fun = lambda x: jnp.asarray(x)[indexer]**2
    check_grads(fun, (arg,), 2, tol, tol, tol)

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer}
      for name, index_specs in MIXED_ADVANCED_INDEXING_TESTS
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testMixedAdvancedIntegerIndexing(self, shape, dtype, rng_factory, indexer):
    rng = rng_factory()
    indexer_with_dummies = [e if isinstance(e, onp.ndarray) else ()
                            for e in indexer]
    substitutes = [(i, e) for i, e in enumerate(indexer)
                   if not isinstance(e, onp.ndarray)]
    args_maker = lambda: [rng(shape, dtype), indexer_with_dummies]

    def fun(x, indexer_with_dummies):
      idx = type(indexer)(util.subvals(indexer_with_dummies, substitutes))
      return jnp.asarray(x)[idx]

    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  def testAdvancedIndexingManually(self):
    x = onp.random.RandomState(0).randn(3, 4, 5)
    index_array = onp.array([0, 2, -1, 0])

    op = lambda x, index_array: x[..., index_array, :]
    cop = api.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2, check_dtypes=True)

    op = lambda x, index_array: x[..., index_array, :, index_array, None]
    cop = api.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2, check_dtypes=True)

    op = lambda x, index_array: x[index_array, ..., index_array[:, None], None]
    cop = api.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2, check_dtypes=True)

  def testUnpacking(self):

    def foo(x):
      a, b, c = x
      return a + b + c

    cfoo = api.jit(foo)

    a1 = foo(onp.arange(3))
    a2 = cfoo(onp.arange(3))

    self.assertAllClose(a1, a2, check_dtypes=True)

  def testBooleanIndexingArray1D(self):
    idx = onp.array([True, True, False])
    x = api.device_put(onp.arange(3))
    ans = x[idx]
    expected = onp.arange(3)[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingList1D(self):
    idx = [True, True, False]
    x = api.device_put(onp.arange(3))
    ans = x[idx]
    expected = onp.arange(3)[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingArray2DBroadcast(self):
    idx = onp.array([True, True, False, True])
    x = onp.arange(8).reshape(4, 2)
    ans = api.device_put(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingList2DBroadcast(self):
    idx = [True, True, False, True]
    x = onp.arange(8).reshape(4, 2)
    ans = api.device_put(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingArray2D(self):
    idx = onp.array([[True, False],
                     [False, True],
                     [False, False],
                     [True, True]])
    x = onp.arange(8).reshape(4, 2)
    ans = api.device_put(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingDynamicShapeError(self):
    x = onp.zeros(3)
    i = onp.array([True, True, False])
    self.assertRaises(IndexError, lambda: api.jit(lambda x, i: x[i])(x, i))

  def testIssue187(self):
    x = jnp.ones((5, 5))
    x[[0, 2, 4], [0, 2, 4]]  # doesn't crash

    x = onp.arange(25).reshape((5, 5))
    ans = api.jit(lambda x: x[[0, 2, 4], [0, 2, 4]])(x)
    expected = x[[0, 2, 4], [0, 2, 4]]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testJVPOfGradOfIndexing(self):
    # Should return a value, even though we didn't pass a symbolic zero as the
    # index tangent.
    x = jnp.ones((3, 4), jnp.float32)
    i = jnp.ones((3,), jnp.int32)
    f = lambda x, i: jnp.sum(x[i])
    primals, tangents = api.jvp(api.grad(f), (x, i), (x, onp.zeros_like(i)))
    expected = onp.broadcast_to(
      onp.array([0, 3, 0], dtype=onp.float32)[:, None], (3, 4))
    self.assertAllClose(expected, primals, check_dtypes=True)
    self.assertAllClose(onp.zeros_like(x), tangents, check_dtypes=True)

  def testTrivialGatherIsntGenerated(self):
    # https://github.com/google/jax/issues/1621
    jaxpr = api.make_jaxpr(lambda x: x[:, None])(onp.arange(4))
    self.assertEqual(len(jaxpr.jaxpr.eqns), 1)
    self.assertNotIn('gather', str(jaxpr))

  def testIndexingEmptyDimension(self):
    # Issue 2671: XLA error when indexing into dimension of size 0
    x = jnp.ones((2, 0))
    # The following work, even on axis 1 of size 0
    _ = x[0, :] + x[0, None] + x[0, 1:] + x[0, 1:3:2]

    with self.assertRaisesRegex(IndexError,
                                "index .* is out of bounds for axis .* with size 0"):
      _ = onp.ones((2, 0))[0, 0]  # The numpy error
    with self.assertRaisesRegex(IndexError,
                                "index is out of bounds for axis .* with size 0"):
      _ = x[0, 0]  # JAX indexing
    with self.assertRaisesRegex(IndexError,
                                "index is out of bounds for axis .* with size 0"):
      api.jit(lambda i: x[0, i])(0)  # JAX indexing under jit

  def testBooleanIndexingWithEmptyResult(self):
    # based on a TensorFlow Probability test that started failing after #1622
    x = jnp.array([-1])
    mask = jnp.array([False])
    ans = x[mask]  # doesn't crash

    expected =  onp.array([-1])[onp.array([False])]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testFloatIndexingError(self):
    BAD_INDEX_TYPE_ERROR = "Indexer must have integer or boolean type, got indexer with type"
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      jnp.zeros(2)[0.]
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      jnp.zeros((2, 2))[(0, 0.)]
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      jnp.zeros((2, 2))[(0, 0.)]
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      api.jit(lambda idx: jnp.zeros((2, 2))[idx])((0, 0.))
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      ops.index_add(jnp.zeros(2), 0., 1.)
    with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
      ops.index_update(jnp.zeros(2), 0., 1.)


  def testIndexOutOfBounds(self):  # https://github.com/google/jax/issues/2245
    array = jnp.ones(5)
    self.assertAllClose(array, array[:10], check_dtypes=True)


def _broadcastable_shapes(shape):
  """Returns all shapes that broadcast to `shape`."""
  def f(rshape):
    yield []
    if rshape:
      for s in f(rshape[1:]):
        yield rshape[0:1] + s
      if rshape[0] != 1:
        for s in f(rshape[1:]):
          yield [1] + s
  for x in f(list(reversed(shape))):
    yield list(reversed(x))

@suppress_deprecated_indexing_warnings()
def _update_shape(shape, indexer):
  return onp.zeros(shape)[indexer].shape


class UpdateOps(enum.Enum):
  UPDATE = 0
  ADD = 1
  MUL = 2
  MIN = 3
  MAX = 4

  @suppress_deprecated_indexing_warnings()
  def onp_fn(op, indexer, x, y):
    x = x.copy()
    x[indexer] = {
      UpdateOps.UPDATE: lambda: y,
      UpdateOps.ADD: lambda: x[indexer] + y,
      UpdateOps.MUL: lambda: x[indexer] * y,
      UpdateOps.MIN: lambda: onp.minimum(x[indexer], y),
      UpdateOps.MAX: lambda: onp.maximum(x[indexer], y),
    }[op]()
    return x

  def jax_fn(op, indexer, x, y):
    return {
      UpdateOps.UPDATE: ops.index_update,
      UpdateOps.ADD: ops.index_add,
      UpdateOps.MUL: ops.index_mul,
      UpdateOps.MIN: ops.index_min,
      UpdateOps.MAX: ops.index_max,
    }[op](x, indexer, y)

  def sugar_fn(op, indexer, x, y):
    x = jnp.array(x)
    return {
      UpdateOps.UPDATE: x.at[indexer].set,
      UpdateOps.ADD: x.at[indexer].add,
      UpdateOps.MUL: x.at[indexer].mul,
      UpdateOps.MIN: x.at[indexer].min,
      UpdateOps.MAX: x.at[indexer].max,
    }[op](y)


class IndexedUpdateTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list({
      "testcase_name": "{}_inshape={}_indexer={}_update={}_sugared={}_op={}".format(
          name, jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), sugared, op.name),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer,
       "update_shape": update_shape, "update_dtype": update_dtype,
       "op": op, "sugared": sugared
  } for name, index_specs in STATIC_INDEXING_TESTS
    for shape, indexer in index_specs
    for op in UpdateOps
    for dtype in (all_dtypes if op == UpdateOps.UPDATE else default_dtypes)
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in ([dtype] if op == UpdateOps.ADD else all_dtypes)
    for sugared in [True, False]
    for rng_factory in [jtu.rand_default]))
  def testStaticIndexing(self, shape, dtype, update_shape, update_dtype,
                         rng_factory, indexer, sugared, op):
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    onp_fn = lambda x, y: UpdateOps.onp_fn(op, indexer, x, y)
    if sugared:
      jax_fn = lambda x, y: UpdateOps.sugar_fn(op, indexer, x, y)
    else:
      jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y)
    self._CheckAgainstNumpy(onp_fn, jax_fn, args_maker, check_dtypes=True)
    self._CompileAndCheck(jax_fn, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list({
      "testcase_name": "{}_inshape={}_indexer={}_update={}_sugared={}_op={}".format(
          name, jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), sugared, op.name),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer,
       "update_shape": update_shape, "update_dtype": update_dtype,
       "op": op, "sugared": sugared
  } for name, index_specs in ADVANCED_INDEXING_TESTS_NO_REPEATS
    for shape, indexer in index_specs
    for op in UpdateOps
    for dtype in (all_dtypes if op == UpdateOps.UPDATE else default_dtypes)
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in ([dtype] if op == UpdateOps.ADD else all_dtypes)
    for sugared in [True, False]
    for rng_factory in [jtu.rand_default]))
  def testAdvancedIndexing(self, shape, dtype, update_shape, update_dtype,
                           rng_factory, indexer, sugared, op):
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    onp_fn = lambda x, y: UpdateOps.onp_fn(op, indexer, x, y)
    if sugared:
      jax_fn = lambda x, y: UpdateOps.sugar_fn(op, indexer, x, y)
    else:
      jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y)
    self._CheckAgainstNumpy(onp_fn, jax_fn, args_maker, check_dtypes=True)
    self._CompileAndCheck(jax_fn, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list({
      "testcase_name": "{}_inshape={}_indexer={}_update={}_op={}".format(
          name, jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer,
       "update_shape": update_shape, "update_dtype": update_dtype,
       "op": op, "sugared": sugared
  } for name, index_specs in MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS
    for shape, indexer in index_specs
    for op in UpdateOps
    for dtype in (all_dtypes if op == UpdateOps.UPDATE else default_dtypes)
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in ([dtype] if op == UpdateOps.ADD else all_dtypes)
    for sugared in [True, False]
    for rng_factory in [jtu.rand_default]))
  def testMixedAdvancedIndexing(self, shape, dtype, update_shape, update_dtype,
                                rng_factory, indexer, sugared, op):
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    onp_fn = lambda x, y: UpdateOps.onp_fn(op, indexer, x, y)
    if sugared:
      jax_fn = lambda x, y: UpdateOps.sugar_fn(op, indexer, x, y)
    else:
      jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y)
    self._CheckAgainstNumpy(onp_fn, jax_fn, args_maker, check_dtypes=True)
    self._CompileAndCheck(jax_fn, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list({
      "testcase_name": "{}_inshape={}_indexer={}_update={}_op={}".format(
          name, jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer,
       "update_shape": update_shape, "update_dtype": update_dtype,
       "op": op
  } for name, index_specs in STATIC_INDEXING_TESTS
    for shape, indexer in index_specs
    for op in [UpdateOps.ADD, UpdateOps.MUL, UpdateOps.UPDATE]
    for dtype in float_dtypes
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in ([dtype] if op == UpdateOps.ADD else float_dtypes)
    for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("tpu")  # TODO(mattjj,phawkins): tpu issues
  def testStaticIndexingGrads(self, shape, dtype, update_shape, update_dtype,
                              rng_factory, indexer, op):
    rng = rng_factory()
    jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y)
    x = rng(shape, dtype)
    y = rng(update_shape, update_dtype)
    check_grads(jax_fn, (x, y), 2, rtol=1e-3, atol=1e-3, eps=1.)

  def testSegmentSumBehavior(self):
    # testAdvancedIndexing compares against NumPy, and as a result doesn't check
    # repeated indices. This test is just a simple manual check, based on
    # https://www.tensorflow.org/api_docs/python/tf/math/segment_sum
    data = onp.array([5, 1, 7, 2, 3, 4, 1, 3])
    segment_ids = onp.array([0, 0, 0, 1, 2, 2, 3, 3])

    ans = ops.index_add(onp.zeros(onp.max(segment_ids) + 1), segment_ids, data)
    expected = onp.array([13, 2, 7, 4])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testSegmentSum(self):
    data = onp.array([5, 1, 7, 2, 3, 4, 1, 3])
    segment_ids = onp.array([0, 0, 0, 1, 2, 2, 3, 3])

    # test with explicit num_segments
    ans = ops.segment_sum(data, segment_ids, num_segments=4)
    expected = onp.array([13, 2, 7, 4])
    self.assertAllClose(ans, expected, check_dtypes=False)

    # test without explicit num_segments
    ans = ops.segment_sum(data, segment_ids)
    expected = onp.array([13, 2, 7, 4])
    self.assertAllClose(ans, expected, check_dtypes=False)


if __name__ == "__main__":
  absltest.main()
