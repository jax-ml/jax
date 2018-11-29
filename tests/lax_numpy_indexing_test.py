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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from functools import partial
import itertools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp

from jax import api
from jax import lax
from jax import numpy as lnp
from jax import test_util as jtu
from jax.config import config

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


def check_grads(f, args, order, atol=None, rtol=None, eps=None):
  # TODO(mattjj,dougalm): add higher-order check
  default_tol = 1e-6 if FLAGS.jax_enable_x64 else 1e-2
  atol = atol or default_tol
  rtol = rtol or default_tol
  eps = eps or default_tol
  jtu.check_jvp(f, partial(api.jvp, f), args, atol, rtol, eps)
  jtu.check_vjp(f, partial(api.vjp, f), args, atol, rtol, eps)


class IndexingTest(jtu.JaxTestCase):
  """Tests for Numpy indexing translation rules."""

  @parameterized.named_parameters({
      "testcase_name":
          "{}_inshape={}_indexer={}".format(
              name, jtu.format_shape_dtype_string( shape, dtype), indexer),
      "shape": shape, "dtype": dtype, "rng": rng, "indexer": indexer
  } for name, index_specs in [
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
  ] for shape, indexer in index_specs for dtype in all_dtypes
                                  for rng in [jtu.rand_default()])
  @jtu.skip_on_devices("tpu")
  def testStaticIndexing(self, shape, dtype, rng, indexer):
    args_maker = lambda: [rng(shape, dtype)]
    fun = lambda x: x[indexer]
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters({
      "testcase_name":
          "{}_inshape={}_indexer={}".format(name,
                                            jtu.format_shape_dtype_string(
                                                shape, dtype), indexer),
      "shape": shape, "dtype": dtype, "rng": rng, "indexer": indexer
  } for name, index_specs in [
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
  ] for shape, indexer in index_specs for dtype in float_dtypes
                                  for rng in [jtu.rand_default()])
  @jtu.skip_on_devices("tpu")
  def testStaticIndexingGrads(self, shape, dtype, rng, indexer):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
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
      out = lax.subvals(triple, zip(isnone, zeros))
      return out, lambda out: slice(*lax.subvals(out, zip(isnone, nones)))
    elif isinstance(idx, (tuple, list)) and idx:
      t = type(idx)
      elts, packs = zip(*map(self._ReplaceSlicesWithTuples, idx))
      return elts, lambda elts: t((pack(i) for pack, i in zip(packs, elts)))
    else:
      return idx, lambda x: x

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng": rng, "indexer": indexer}
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
      for rng in [jtu.rand_default()])
  def testDynamicIndexingWithSlicesErrors(self, shape, dtype, rng, indexer):
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
       "shape": shape, "dtype": dtype, "rng": rng, "indexer": indexer}
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
      for rng in [jtu.rand_default()])
  def testDynamicIndexingWithIntegers(self, shape, dtype, rng, indexer):
    unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

    def fun(x, unpacked_indexer):
      indexer = pack_indexer(unpacked_indexer)
      return x[indexer]

    args_maker = lambda: [rng(shape, dtype), unpacked_indexer]
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng": rng, "indexer": indexer}
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
      for rng in [jtu.rand_default()])
  def DISABLED_testDynamicIndexingWithIntegersGrads(self, shape, dtype, rng, indexer):
    # TODO(mattjj): re-enable (test works but for grad-of-compile, in flux)
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
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
       "shape": shape, "dtype": dtype, "rng": rng, "indexer": indexer}
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
      for dtype in all_dtypes
      for rng in [jtu.rand_default()])
  def testAdvancedIntegerIndexing(self, shape, dtype, rng, indexer):
    args_maker = lambda: [rng(shape, dtype), indexer]
    fun = lambda x, idx: x[idx]
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng": rng, "indexer": indexer}
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
      for rng in [jtu.rand_default()])
  def testAdvancedIntegerIndexingGrads(self, shape, dtype, rng, indexer):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    arg = rng(shape, dtype)
    fun = lambda x: x[indexer]**2
    check_grads(fun, (arg,), 2, tol, tol, tol)

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng": rng, "indexer": indexer}
      for name, index_specs in [
          ("SlicesAndOneIntArrayIndex",
           [IndexSpec(shape=(2, 3), indexer=(onp.array([0, 1]), slice(1, 2))),
            IndexSpec(shape=(2, 3), indexer=(slice(0, 2),
                                             onp.array([0, 2]))),
            IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                                onp.array([0, 2]),
                                                slice(None))),
            IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                                onp.array([[0, 2], [1, 1]]),
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
                                                onp.array([-1, 2, -1]))),
            IndexSpec(shape=(3, 4, 5), indexer=(onp.array([[0, 2], [2, 0]]),
                                                Ellipsis,
                                                onp.array([[1, 0], [1, 0]]))),
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
      ]
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng in [jtu.rand_default()])
  def testMixedAdvancedIntegerIndexing(self, shape, dtype, rng, indexer):
    indexer_with_dummies = [e if isinstance(e, onp.ndarray) else ()
                            for e in indexer]
    substitutes = [(i, e) for i, e in enumerate(indexer)
                   if not isinstance(e, onp.ndarray)]
    args_maker = lambda: [rng(shape, dtype), indexer_with_dummies]

    def fun(x, indexer_with_dummies):
      idx = type(indexer)(lax.subvals(indexer_with_dummies, substitutes))
      return x[idx]

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


if __name__ == "__main__":
  config.config_with_absl()
  absltest.main()
