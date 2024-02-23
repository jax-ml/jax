import collections
import copy
from functools import partial
import inspect
import io
import itertools
import math
from typing import cast, Iterator, Optional, List, Tuple
import unittest
from unittest import SkipTest
import warnings

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
try:
  import numpy_dispatch
except ImportError:
  numpy_dispatch = None

import jax
import jax.ops
from jax import lax
from jax import numpy as jnp
from jax import tree_util
from jax.test_util import check_grads

from jax._src import core
from jax._src import array
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.lax import lax as lax_internal
from jax._src.numpy.util import _parse_numpydoc, ParsedDoc
import jax.util
from jax._src import array
from jax._src import lax_reference
from jax._src import config
config.parse_flags_with_absl()

numpy_version = jtu.numpy_version()

nonempty_nonscalar_array_shapes = [(4,), (3, 4), (3, 1), (1, 4), (2, 1, 4), (2, 3, 4)]
nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
one_dim_array_shapes = [(1,), (6,), (12,)]
empty_array_shapes = [(0,), (0, 4), (3, 0),]

scalar_shapes = [jtu.NUMPY_SCALAR_SHAPE, jtu.PYTHON_SCALAR_SHAPE]
array_shapes = nonempty_array_shapes
nonzerodim_shapes = nonempty_nonscalar_array_shapes
nonempty_shapes = scalar_shapes + nonempty_array_shapes
all_shapes = scalar_shapes + array_shapes

float_dtypes = [np.float16, np.float32]
int_dtypes = [np.int32]
unsigned_dtypes = [np.uint32]
bool_dtypes = jtu.dtypes.boolean
default_dtypes = float_dtypes + int_dtypes
inexact_dtypes = float_dtypes
number_dtypes = float_dtypes + int_dtypes + unsigned_dtypes
all_dtypes = number_dtypes + bool_dtypes


python_scalar_dtypes = [jnp.bool_, jnp.int_, jnp.float_]

# uint64 is problematic because with any uint type it promotes to float:
int_dtypes_no_uint64 = [d for d in int_dtypes + unsigned_dtypes if d != np.uint64]

def _indexer_with_default_outputs(indexer, use_defaults=True):
  """Like jtu.with_jax_dtype_defaults, but for __getitem__ APIs"""
  class Indexer:
    @partial(jtu.with_jax_dtype_defaults, use_defaults=use_defaults)
    def __getitem__(self, *args):
      return indexer.__getitem__(*args)
  return Indexer()

def _valid_dtypes_for_shape(shape, dtypes):
  # Not all (shape, dtype) pairs are valid. In particular, Python scalars only
  # have one type in each category (float, bool, etc.)
  if shape is jtu.PYTHON_SCALAR_SHAPE:
    return [t for t in dtypes if t in python_scalar_dtypes]
  return dtypes

def _shape_and_dtypes(shapes, dtypes):
  for shape in shapes:
    for dtype in _valid_dtypes_for_shape(shape, dtypes):
      yield (shape, dtype)

def _compatible_shapes(shape):
  if np.ndim(shape) == 0 or shape in scalar_shapes:
    return [shape]
  return (shape[n:] for n in range(len(shape) + 1))

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


JAX_ARGMINMAX_RECORDS = [
    op_record("argmin", 1, default_dtypes, nonempty_shapes, jtu.rand_some_equal, []),
    op_record("argmax", 1, default_dtypes, nonempty_shapes, jtu.rand_some_equal, []),
    op_record("nanargmin", 1, default_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
    op_record("nanargmax", 1, default_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
]

def _shapes_are_broadcast_compatible(shapes):
  try:
    lax.broadcast_shapes(*(() if s in scalar_shapes else s for s in shapes))
  except ValueError:
    return False
  else:
    return True

def _shapes_are_equal_length(shapes):
  return all(len(shape) == len(shapes[0]) for shape in shapes[1:])


class LaxBackedNumpyTests(jtu.JaxTestCase):
  """Tests for LAX-backed Numpy tests for metal implementation."""

  def _GetArgsMaker(self, rng, shapes, dtypes, np_arrays=True):
    def f():
      out = [rng(shape, dtype or jnp.float_)
             for shape, dtype in zip(shapes, dtypes)]
      if np_arrays:
        return out
      return [jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) else a
              for a in out]
    return f

  @parameterized.parameters(
    [dtype for dtype in [jnp.bool_, jnp.uint8, jnp.uint16, jnp.uint32,
                         jnp.uint64, jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                         jnp.float16, jnp.float32]
     if dtype == dtypes.canonicalize_dtype(dtype)])
  def testDtypeWrappers(self, dtype):
    arr = dtype(0)
    self.assertIsInstance(arr, jax.Array)
    self.assertEqual(arr.dtype, np.dtype(dtype))
    self.assertArraysEqual(arr, 0, check_dtypes=False)

    # No copy primitive is generated
    jaxpr = jax.make_jaxpr(dtype)(0)
    prims = [eqn.primitive for eqn in jaxpr.eqns]
    self.assertEqual(prims, [lax.convert_element_type_p])  # No copy generated.

  @jtu.sample_product(
      dtype=float_dtypes + [object],
      allow_pickle=[True, False],
  )
  def testLoad(self, dtype, allow_pickle):
    if dtype == object and not allow_pickle:
      self.skipTest("dtype=object requires allow_pickle=True")
    rng = jtu.rand_default(self.rng())
    arr = rng((10), dtype)
    with io.BytesIO() as f:
      jnp.save(f, arr)
      f.seek(0)
      arr_out = jnp.load(f, allow_pickle=allow_pickle)
    self.assertArraysEqual(arr, arr_out, allow_object_dtype=True)

  def testArrayEqualExamples(self):
    # examples from the array_equal() docstring.
    self.assertTrue(jnp.array_equal([1, 2], [1, 2]))
    self.assertTrue(jnp.array_equal(np.array([1, 2]), np.array([1, 2])))
    self.assertFalse(jnp.array_equal([1, 2], [1, 2, 3]))
    self.assertFalse(jnp.array_equal([1, 2], [1, 4]))

    a = np.array([1, np.nan])
    self.assertFalse(jnp.array_equal(a, a))
    self.assertTrue(jnp.array_equal(a, a, equal_nan=True))

  def testArrayEquivExamples(self):
    # examples from the array_equiv() docstring.
    self.assertTrue(jnp.array_equiv([1, 2], [1, 2]))
    self.assertFalse(jnp.array_equiv([1, 2], [1, 3]))
    with jax.numpy_rank_promotion('allow'):
      self.assertTrue(jnp.array_equiv([1, 2], [[1, 2], [1, 2]]))
      self.assertFalse(jnp.array_equiv([1, 2], [[1, 2, 1, 2], [1, 2, 1, 2]]))
      self.assertFalse(jnp.array_equiv([1, 2], [[1, 2], [1, 3]]))

  def testArrayModule(self):
    if numpy_dispatch is None:
      raise SkipTest('requires https://github.com/seberg/numpy-dispatch')

    jnp_array = jnp.array(1.0)
    np_array = np.array(1.0)

    module = numpy_dispatch.get_array_module(jnp_array)
    self.assertIs(module, jnp)

    module = numpy_dispatch.get_array_module(jnp_array, np_array)
    self.assertIs(module, jnp)

    def f(x):
      module = numpy_dispatch.get_array_module(x)
      self.assertIs(module, jnp)
      return x
    jax.jit(f)(jnp_array)
    jax.grad(f)(jnp_array)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in all_shapes
      for axis in list(range(-len(shape), len(shape)))],
    discont=[None, "pi", 2],
    period=["2pi", "pi"],
    dtype=default_dtypes,
  )

  # This currently seg-faults because of dynamic shape = (0, )
  def testUnwrap(self, shape, dtype, axis, discont, period):
    if (not np.all(shape)):
      self.skipTest("JAX metal does not support dynamic shape unwrap()")

    self.skipTest("JAX metal does not support remainder")
        
    if numpy_version < (1, 21) and period != "2pi":
      self.skipTest("numpy < 1.21 does not support the period argument to unwrap()")
    special_vals = {"pi": np.pi, "2pi": 2 * np.pi}
    period = special_vals.get(period, period)
    discont = special_vals.get(discont, discont)

    rng = jtu.rand_default(self.rng())

    def np_fun(x):
      dtype = None
      if x.dtype == dtypes.bfloat16:
        dtype = x.dtype
        x = x.astype(np.float32)
      if numpy_version < (1, 21):
        out = np.unwrap(x, axis=axis, discont=discont or np.pi)
      else:
        out = np.unwrap(x, axis=axis, discont=discont, period=period)
      return out if dtype is None else out.astype(dtype)

    jnp_fun = partial(jnp.unwrap, axis=axis, discont=discont, period=period)
    if not dtypes.issubdtype(dtype, np.inexact):
      # This case requires implicit dtype promotion
      jnp_fun = jax.numpy_dtype_promotion('standard')(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            atol={dtypes.bfloat16: 1e-1, np.float16: 1e-2})
    self._CompileAndCheck(jnp_fun, args_maker, atol={dtypes.bfloat16: 1e-1})

  @jtu.sample_product(
    [dict(shape=shape, indices=indices, update_shape=update_shape)
      for shape, indices, update_shape in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 3, 1)],
        [(1, 50, 8, 64), (0,20, 0, 0), (1,1,8,64)]
      ]
    ],
    dtype=default_dtypes,
  )
  def testDynamicUpdateSliceAgainstNumpy(self, shape, dtype, indices,
                                         update_shape):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      return [rng(shape, dtype), rng(update_shape, dtype), np.array(indices)]

    self._CheckAgainstNumpy(lax_reference.dynamic_update_slice,
                            lax.dynamic_update_slice, args_maker)

  @jtu.sample_product(
    [dict(init_val=init_val, op=op, dtype=dtype)
      for init_val, op, dtypes in [
          (0, lax.add, [np.float32]),
          (-np.inf, lax.max, [np.float32]),
          #(np.inf, lax.min, [np.float32]),
      ]
      for dtype in [np.float32]
    ],
    [dict(shape=shape, dims=dims, strides=strides, padding=padding,
          base_dilation=base_dilation, window_dilation=window_dilation)
      for shape, dims, strides, padding, base_dilation, window_dilation in (
        itertools.chain(
          itertools.product(
            [(3, 2, 4, 6)], [(1, 1, 2, 1), (2, 1, 2, 1)],
            [(1, 2, 2, 1), (1, 1, 1, 1)],
            ["VALID", "SAME", [(0, 1), (1, 0), (2, 3), (0, 2)]],
            [(1, 1, 1, 1)],
            [(1, 1, 1, 1), (1, 2, 2, 1)])))
    ],
  )
  def testReduceWindow(self, op, init_val, dtype, shape, dims, strides, padding,
                       base_dilation, window_dilation):
    rng = jtu.rand_small(self.rng())
    init_val = np.asarray(init_val, dtype=dtype)

    def fun(operand, init_val):
      return lax.reduce_window(operand, init_val, op, dims, strides, padding,
                               base_dilation, window_dilation)

    def reference_fun(operand, init_val):
      return lax_reference.reduce_window(operand, init_val, op, dims, strides,
                                         padding, base_dilation)

    args_maker = lambda: [rng(shape, dtype), init_val]
    self._CompileAndCheck(fun, args_maker)
    if all(d == 1 for d in window_dilation):
      self._CheckAgainstNumpy(reference_fun, fun, args_maker)

    # we separately test the version that uses a concrete init_val because it
    # can hit different code paths
    def fun(operand):
      return lax.reduce_window(operand, init_val, op, dims, strides, padding,
                               base_dilation, window_dilation)

    args_maker = lambda: [rng(shape, dtype)]
    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in all_shapes
      for axis in list(range(-len(shape), len(shape))) + [None]],
    dtype=all_dtypes,
  )
  def testCountNonzero(self, shape, dtype, axis):
    if (not np.all(shape)):
      self.skipTest("JAX metal does not support dynamic shape unwrap()")
    rng = jtu.rand_some_zero(self.rng())
    np_fun = lambda x: np.count_nonzero(x, axis)
    jnp_fun = lambda x: jnp.count_nonzero(x, axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(shape=all_shapes, dtype=all_dtypes)
  def testNonzero(self, shape, dtype):
    if (not np.all(shape)):
      self.skipTest("JAX metal does not support dynamic shape unwrap()")
    if (dtype == np.uint16):
      self.skipTest("JAX metal has a crash with NonZero uint16")
    self.skipTest("JAX metal has a bug with Pad")
    rng = jtu.rand_some_zero(self.rng())
    np_fun = lambda x: np.nonzero(x)
    np_fun = jtu.ignore_warning(
      category=DeprecationWarning,
      message="Calling nonzero on 0d arrays.*")(np_fun)
    jnp_fun = lambda x: jnp.nonzero(x)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

  @jtu.sample_product(
    [dict(shape=shape, fill_value=fill_value)
      for shape in nonempty_array_shapes
      for fill_value in [None, -1, shape or (1,)]
     ],
    dtype=all_dtypes,
    size=[1, 5, 10],
  )
  def testNonzeroSize(self, shape, dtype, size, fill_value):
    self.skipTest("JAX metal has a bug with Pad")
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    @jtu.ignore_warning(category=DeprecationWarning, message="Calling nonzero on 0d arrays.*")
    def np_fun(x):
      result = np.nonzero(x)
      if size <= len(result[0]):
        return tuple(arg[:size] for arg in result)
      else:
        fillvals = fill_value if np.ndim(fill_value) else len(result) * [fill_value or 0]
        return tuple(np.concatenate([arg, np.full(size - len(arg), fval, arg.dtype)])
                     for fval, arg in safe_zip(fillvals, result))
    jnp_fun = lambda x: jnp.nonzero(x, size=size, fill_value=fill_value)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(shape=all_shapes, dtype=[np.float32])
  def testArgWhere(self, shape, dtype):
    if (not np.all(shape)):
      self.skipTest("JAX metal does not support dynamic shape unwrap()")
    self.skipTest("JAX metal does not support ArgWhere")
    rng = jtu.rand_some_zero(self.rng())
    np_fun = jtu.ignore_warning(
      category=DeprecationWarning,
      message="Calling nonzero on 0d arrays.*")(np.argwhere)
    jnp_fun = jnp.argwhere
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

    # JIT compilation requires specifying a size statically. Full test of this
    # behavior is in testNonzeroSize().
    jnp_fun = lambda x: jnp.argwhere(x, size=np.size(x) // 2)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(np_op=getattr(np, rec.name), jnp_op=getattr(jnp, rec.name),
          shape=shape, dtype=dtype, axis=axis, rng_factory=rec.rng_factory)
      for rec in JAX_ARGMINMAX_RECORDS
      for shape, dtype in _shape_and_dtypes(rec.shapes, rec.dtypes)
      for axis in range(-len(shape), len(shape))],
    keepdims=[False, True],
  )
  def testArgMinMax(self, np_op, jnp_op, rng_factory, shape, dtype, axis, keepdims):
    rng = rng_factory(self.rng())
    if dtype == np.complex128 and jtu.device_under_test() == "gpu":
      raise unittest.SkipTest("complex128 reductions not supported on GPU")
    if "nan" in np_op.__name__ and dtype == jnp.bfloat16:
      raise unittest.SkipTest("NumPy doesn't correctly handle bfloat16 arrays")
    if numpy_version < (1, 22) and keepdims:
      raise unittest.SkipTest("NumPy < 1.22 does not support keepdims argument to argmin/argmax")
    kwds = {"keepdims": True} if keepdims else {}

    np_fun = jtu.with_jax_dtype_defaults(partial(np_op, axis=axis, **kwds))
    jnp_fun = partial(jnp_op, axis=axis, **kwds)

    args_maker = lambda: [rng(shape, dtype)]
    try:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    except ValueError as e:
      if str(e) == "All-NaN slice encountered":
        self.skipTest("JAX doesn't support checking for all-NaN slices")
      else:
        raise
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(name=rec.name, np_op=getattr(np, rec.name),
          jnp_op=getattr(jnp, rec.name))
      for rec in JAX_ARGMINMAX_RECORDS],
  )
  def testArgMinMaxEmpty(self, name, np_op, jnp_op):
    self.skipTest("JAX metal does crash")
    name = name[3:] if name.startswith("nan") else name
    msg = f"attempt to get {name} of an empty sequence"
    with self.assertRaises(ValueError, msg=msg):
      jnp_op(np.array([]))
    with self.assertRaises(ValueError, msg=msg):
      jnp_op(np.zeros((2, 0)), axis=1)
    np_fun = jtu.with_jax_dtype_defaults(partial(np_op, axis=0))
    jnp_fun = partial(jnp_op, axis=0)
    args_maker = lambda: [np.zeros((2, 0))]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, axes=axes)
      for lhs_shape, rhs_shape, axes in [
          [(2,), (2,), (-1, -1, -1, None)], # scalar output
          [(2, 4), (2, 4), (-1, -1, -1, 0)], # 2D vectors
          [(3, 4), (3, 4), (-1, -1, -1, 0)], # 3D vectors
          [(3, 4), (3, 6, 5, 4), (-1, -1, -1, 0)], # broadcasting
          [(4, 3), (3, 6, 5, 4), (1, 0, -1, None)], # different axes
          [(6, 1, 3), (5, 3), (-1, -1, -1, None)], # more broadcasting
          [(6, 1, 2), (5, 3), (-1, -1, -1, None)], # mixed 2D and 3D vectors
          [(10, 5, 2, 8), (1, 5, 1, 3), (-2, -1, -3, None)], # axes/broadcasting
          [(4, 5, 2), (4, 5, 2), (-1, -1, 0, None)], # axisc should do nothing
          [(4, 5, 2), (4, 5, 2), (-1, -1, -1, None)] # same as before
      ]],
    lhs_dtype=number_dtypes,
    rhs_dtype=number_dtypes,
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testCross(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    axisa, axisb, axisc, axis = axes
    jnp_fun = lambda a, b: jnp.cross(a, b, axisa, axisb, axisc, axis)
    def np_fun(a, b):
      a = a.astype(np.float32) if lhs_dtype == jnp.bfloat16 else a
      b = b.astype(np.float32) if rhs_dtype == jnp.bfloat16 else b
      out = np.cross(a, b, axisa, axisb, axisc, axis)
      return out.astype(jnp.promote_types(lhs_dtype, rhs_dtype))
    tol_spec = {dtypes.bfloat16: 3e-1, np.float16: 0.15}
    tol = max(jtu.tolerance(lhs_dtype, tol_spec),
              jtu.tolerance(rhs_dtype, tol_spec))
    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol)
      self._CompileAndCheck(jnp_fun, args_maker, atol=tol, rtol=tol)

  @jtu.sample_product(
    [dict(name=name, lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for name, lhs_shape, rhs_shape in [
          ("matrix-scalar", (3, 3), ()),
          ("scalar-matrix", (), (3, 3)),
          # ("matrix-vector", (4, 5), (5,)),
          ("vector-matrix", (6,), (6, 4)),
          ("matrix-matrix", (3, 4), (4, 5)),
          # ("tensor-vector", (4, 3, 2), (2,)),
          ("vector-tensor", (2,), (3, 2, 4)),
          ("tensor-matrix", (4, 3, 2), (2, 5)),
          ("matrix-tensor", (5, 2), (3, 2, 4)),
          ("tensor-tensor", (2, 3, 4), (5, 4, 1))]],
    lhs_dtype=float_dtypes,
    rhs_dtype=float_dtypes,
  )
  @jax.default_matmul_precision("float32")
  def testDot(self, name, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    tol = {np.float16: 1e-2, np.float32: 2e-5, np.float64: 1e-14,
           np.complex128: 1e-14}
    def np_dot(x, y):
      x = x.astype(np.float32) if lhs_dtype == jnp.bfloat16 else x
      y = y.astype(np.float32) if rhs_dtype == jnp.bfloat16 else y
      return np.dot(x, y).astype(jnp.promote_types(lhs_dtype, rhs_dtype))
    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstNumpy(np_dot, jnp.dot, args_maker, tol=tol)
      self._CompileAndCheck(jnp.dot, args_maker, atol=tol, rtol=tol)

  @jtu.sample_product(
    [dict(name=name, lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for name, lhs_shape, rhs_shape in [
          ("vector-vector", (3,), (3,)),
          ("matrix-vector", (3, 3), (3,)),
          ("vector-matrix", (3,), (3, 3)),
          ("matrix-matrix", (3, 3), (3, 3)),
          ("vector-tensor", (3,), (5, 3, 2)),
          ("tensor-vector", (5, 3, 2), (2,)),
          ("matrix-tensor", (5, 2), (3, 2, 4)),
          ("tensor-matrix", (5, 2, 3), (3, 2)),
          ("tensor-tensor", (5, 3, 4), (5, 4, 1)),
          ("tensor-tensor-broadcast", (3, 1, 3, 4), (5, 4, 1))]],
    lhs_dtype=float_dtypes,
    rhs_dtype=float_dtypes,
  )
  @jax.default_matmul_precision("float32")
  def testMatmul(self, name, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype):
    rng = jtu.rand_default(self.rng())
    def np_fun(x, y):
      dtype = jnp.promote_types(lhs_dtype, rhs_dtype)
      return np.matmul(x, y).astype(dtype)
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    tol = {np.float16: 1e-2, np.float32: 2e-2, np.float64: 1e-12,
           np.complex128: 1e-12}

    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstNumpy(np_fun, jnp.matmul, args_maker, tol=tol)
      self._CompileAndCheck(jnp.matmul, args_maker, atol=tol, rtol=tol)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, axes=axes)
      for lhs_shape, rhs_shape, axes in [
          [(3,), (), 0],
          [(2, 3, 4), (5, 6, 7), 0],  # from issue #740
          [(2, 3, 4), (3, 4, 5, 6), 2],
          [(2, 3, 4), (5, 4, 3, 6), [1, 2]],
          [(2, 3, 4), (5, 4, 3, 6), [[1, 2], [2, 1]]],
          [(1, 2, 3, 4), (4, 5, 3, 6), [[2, 3], [2, 0]]],
      ]],
    lhs_dtype=float_dtypes,
    rhs_dtype=float_dtypes,
  )
  @jax.default_matmul_precision("float32")
  def testTensordot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    jnp_fun = lambda a, b: jnp.tensordot(a, b, axes)
    def np_fun(a, b):
      a = a if lhs_dtype != jnp.bfloat16 else a.astype(np.float32)
      b = b if rhs_dtype != jnp.bfloat16 else b.astype(np.float32)
      dtype = jnp.promote_types(lhs_dtype, rhs_dtype)
      return np.tensordot(a, b, axes).astype(dtype)
    tol = {np.float16: 1e-1, np.float32: 1e-3, np.float64: 1e-12,
           np.complex64: 1e-3, np.complex128: 1e-12}

    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol)
      self._CompileAndCheck(jnp_fun, args_maker)

  def testTensordotErrors(self):
    a = self.rng().random((3, 2, 2))
    b = self.rng().random((2,))
    self.assertRaisesRegex(
      TypeError, "Number of tensordot axes.*exceeds input ranks.*",
      lambda: jnp.tensordot(a, b, axes=2))

    self.assertRaisesRegex(
      TypeError, "tensordot requires axes lists to have equal length.*",
      lambda: jnp.tensordot(a, b, axes=([0], [0, 1])))

    self.assertRaisesRegex(
      TypeError, "tensordot requires both axes lists to be either ints, tuples or lists.*",
      lambda: jnp.tensordot(a, b, axes=('bad', 'axes')))

    self.assertRaisesRegex(
      TypeError, "tensordot axes argument must be an int, a pair of ints, or a pair of lists.*",
      lambda: jnp.tensordot(a, b, axes='badaxes'))

  @jtu.sample_product(
    element_shape=all_shapes,
    test_shape=all_shapes,
    dtype=float_dtypes,  # TODO: int_dtypes are not working
    # dtype=default_dtypes,
    invert=[False, True],
  )
  def testIsin(self, element_shape, test_shape, dtype, invert):
    if ((not np.all(test_shape)) or (not np.all(element_shape))):
      self.skipTest("JAX metal does not support dynamic shape")
    if not ReportedIssuesTests.jax_metal_supported('0.0.6'):
      self.skipTest("JAx metal has a regression on the version")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(element_shape, dtype), rng(test_shape, dtype)]
    jnp_fun = lambda e, t: jnp.isin(e, t, invert=invert)
    np_fun = lambda e, t: np.isin(e, t, invert=invert)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonzerodim_shapes
      for axis in (None, *range(len(shape)))
    ],
    dtype=float_dtypes,
  )
  def testSort(self, dtype, shape, axis):
    if (axis != 0):
        return
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_fun = jnp.sort
    np_fun = np.sort
    if axis is not None:
      jnp_fun = partial(jnp_fun, axis=axis)
      np_fun = partial(np_fun, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
  
  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in [(2,3,4)]
      for axis in (-1, *range(len(shape)))
    ],
    dtype=float_dtypes,
  )
  def testSortKeyVal(self, dtype, shape, axis):
    #self.skipTest("JAX metal does not support sortkeyval.")
    rng = jtu.rand_default(self.rng())
    # This test relies on the property that wherever keys are tied, values are
    # too, since we don't guarantee the same ordering of values with equal keys.
    # To avoid that case, we generate unique keys (globally in the key array).
    def args_maker():
      flat_keys = np.arange(math.prod(shape), dtype=dtype)
      keys = self.rng().permutation(flat_keys).reshape(shape)
      values = rng(shape, dtype)
      return keys, values

    fun = lambda keys, values: lax.sort_key_val(keys, values, axis, is_stable=True)
    numpy_op = lambda ks, vs: lax_reference.sort_key_val(ks, vs, axis)
    self._CheckAgainstNumpy(fun, numpy_op, args_maker)

  @jtu.sample_product(
    [dict(shifts=shifts, axis=axis)
      for shifts, axis in [
        (3, None),
        (1, 1),
        # ((3,), (0,)),
        ((-2,), (-2,)),
        # ((1, 2), (0, -1)),
        ((4, 2, 5, 5, 2, 4), None),
        (100, None),
      ]
    ],
    dtype=all_dtypes,
    shape=[(3, 4), (3, 4, 5)],
  )
  def testRoll(self, shape, dtype, shifts, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), np.array(shifts)]
    jnp_op = partial(jnp.roll, axis=axis)
    np_op = partial(np.roll, axis=axis)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    dtype=[np.uint8, np.bool_],
    bitorder=['big', 'little'],
    shape=[(1, 2, 3, 4)],
    axis=[None, 0, 1, -2, -1],
  )
  def testPackbits(self, shape, dtype, axis, bitorder):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_op = partial(jnp.packbits, axis=axis, bitorder=bitorder)
    np_op = partial(np.packbits, axis=axis, bitorder=bitorder)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    dtype=[np.uint8],
    bitorder=['big', 'little'],
    shape=[(1, 2, 3, 4)],
    axis=[None, 0, 1, -2, -1],
    count=[None, 20],
  )
  def testUnpackbits(self, shape, dtype, axis, bitorder, count):
    rng = jtu.rand_int(self.rng(), 0, 256)
    args_maker = lambda: [rng(shape, dtype)]
    jnp_op = partial(jnp.unpackbits, axis=axis, bitorder=bitorder)
    np_op = partial(np.unpackbits, axis=axis, bitorder=bitorder)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)


  @jtu.sample_product(
    shape=array_shapes,
    dtype=all_dtypes,
  )
  def testNanToNum(self, shape, dtype):
    rng = jtu.rand_some_inf_and_nan(self.rng())
    dtype = np.dtype(dtypes.canonicalize_dtype(dtype)).type
    def np_fun(x):
      if dtype == jnp.bfloat16:
        x = np.where(np.isnan(x), dtype(0), x)
        x = np.where(np.isposinf(x), jnp.finfo(dtype).max, x)
        x = np.where(np.isneginf(x), jnp.finfo(dtype).min, x)
        return x
      else:
        return np.nan_to_num(x).astype(dtype)

    args_maker = lambda: [rng(shape, dtype)]
    check_dtypes = shape is not jtu.PYTHON_SCALAR_SHAPE
    self._CheckAgainstNumpy(np_fun, jnp.nan_to_num, args_maker,
                            check_dtypes=check_dtypes)
    self._CompileAndCheck(jnp.nan_to_num, args_maker,
                          check_dtypes=check_dtypes)
    # self._CompileAndCheck(jnp_op, args_maker)


  @jtu.sample_product(
    [dict(shapes=shapes, dtypes=dtypes)
        for shapes, dtypes in (
          ((), ()),
          (((7,),), (np.int32,)),
          (((3,), (4,)), (np.int32, np.int32)),
          (((3,), (1,), (4,)), (np.int32, np.int32, np.int32)),
        )
    ],
  )
  def testIx_(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)
                          for shape, dtype in zip(shapes, dtypes)]
    self._CheckAgainstNumpy(np.ix_, jnp.ix_, args_maker)
    self._CompileAndCheck(jnp.ix_, args_maker)

  @jtu.sample_product(
    dimensions=[(2,), (4, 5, 6)],
    dtype=number_dtypes,
    sparse=[True, False],
  )
  def testIndices(self, dimensions, dtype, sparse):
    if jtu.device_under_test() == "tpu" and dtype in (np.int16, np.uint16):
      raise unittest.SkipTest("Compilation failure on TPU ")
    def args_maker(): return []
    np_fun = partial(np.indices, dimensions=dimensions,
                     dtype=dtype, sparse=sparse)
    jnp_fun = partial(jnp.indices, dimensions=dimensions,
                      dtype=dtype, sparse=sparse)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(op=op, q_rng=q_rng)
      for (op, q_rng) in (
        ("percentile", partial(jtu.rand_uniform, low=0., high=100.)),
        ("quantile", partial(jtu.rand_uniform, low=0., high=1.)),
        ("nanpercentile", partial(jtu.rand_uniform, low=0., high=100.)),
        ("nanquantile", partial(jtu.rand_uniform, low=0., high=1.)),
      )
    ],
    [dict(a_shape=a_shape, axis=axis)
      for a_shape, axis in (
        ((7,), None),
        ((47, 7), 0),
        ((47, 7), ()),
        ((4, 101), 1),
        ((4, 47, 7), (1, 2)),
        ((4, 47, 7), (0, 2)),
        ((4, 47, 7), (1, 0, 2)),
      )
    ],
    a_dtype=default_dtypes,
    q_dtype=[np.float32],
    q_shape=scalar_shapes + [(1,), (4,)],
    keepdims=[False, True],
    method=['linear', 'lower', 'higher', 'nearest', 'midpoint'],
  )
  def testQuantile(self, op, q_rng, a_shape, a_dtype, q_shape, q_dtype,
                   axis, keepdims, method):
    raise unittest.SkipTest("JAX metal legalization error with Sort ")
    a_rng = jtu.rand_some_nan(self.rng())
    q_rng = q_rng(self.rng())
    if "median" in op:
      args_maker = lambda: [a_rng(a_shape, a_dtype)]
    else:
      args_maker = lambda: [a_rng(a_shape, a_dtype), q_rng(q_shape, q_dtype)]

    @jtu.ignore_warning(category=RuntimeWarning,
                        message="All-NaN slice encountered")
    def np_fun(*args):
      args = [x if jnp.result_type(x) != jnp.bfloat16 else
              np.asarray(x, np.float32) for x in args]
      if numpy_version <= (1, 22):
        return getattr(np, op)(*args, axis=axis, keepdims=keepdims,
                               interpolation=method)
      else:
        return getattr(np, op)(*args, axis=axis, keepdims=keepdims,
                               method=method)
    jnp_fun = partial(getattr(jnp, op), axis=axis, keepdims=keepdims,
                      method=method)

    # TODO(phawkins): we currently set dtype=False because we aren't as
    # aggressive about promoting to float64. It's not clear we want to mimic
    # Numpy here.
    tol_spec = {np.float16: 1E-2, np.float32: 2e-4, np.float64: 5e-6}
    tol = max(jtu.tolerance(a_dtype, tol_spec),
              jtu.tolerance(q_dtype, tol_spec))
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, rtol=tol)

  @jtu.sample_product(
    [dict(a_shape=a_shape, axis=axis)
      for a_shape, axis in (
        ((7,), None),
        ((47, 7), 0),
        ((4, 101), 1),
      )
    ],
    a_dtype=default_dtypes,
    keepdims=[False, True],
    op=["median", "nanmedian"],
  )
  def testMedian(self, op, a_shape, a_dtype, axis, keepdims):
    raise unittest.SkipTest("JAX metal legalization error with Sort ")
    if op == "median":
      a_rng = jtu.rand_default(self.rng())
    else:
      a_rng = jtu.rand_some_nan(self.rng())
    args_maker = lambda: [a_rng(a_shape, a_dtype)]
    def np_fun(*args):
      args = [x if jnp.result_type(x) != jnp.bfloat16 else
              np.asarray(x, np.float32) for x in args]
      return getattr(np, op)(*args, axis=axis, keepdims=keepdims)
    jnp_fun = partial(getattr(jnp, op), axis=axis, keepdims=keepdims)
    # TODO(phawkins): we currently set dtype=False because we aren't as
    # aggressive about promoting to float64. It's not clear we want to mimic
    # Numpy here.
    tol_spec = {np.float32: 2e-4, np.float64: 5e-6}
    tol = jtu.tolerance(a_dtype, tol_spec)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, rtol=tol)

  @jtu.sample_product(
    [dict(n=n, shapes=shapes)
      for n in range(1, 3)
      for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(all_shapes, 2 * n + 1))
    ],
    # To avoid forming the full product of shapes and dtypes we always sample
    # maximal set of dtypes.
    dtypes=itertools.combinations_with_replacement(all_dtypes, 3),
  )
  def testSelect(self, n, shapes, dtypes):
    dtypes = dtypes[:n+1]
    rng = jtu.rand_default(self.rng())
    n = len(dtypes) - 1
    def args_maker():
      condlist = [rng(shape, np.bool_) for shape in shapes[:n]]
      choicelist = [rng(shape, dtype)
                    for shape, dtype in zip(shapes[n:-1], dtypes[:n])]
      default = rng(shapes[-1], dtypes[-1])
      return condlist, choicelist, default
    # TODO(phawkins): float32/float64 type mismatches
    @jax.numpy_dtype_promotion('standard')
    def np_fun(condlist, choicelist, default):
      choicelist = [x if jnp.result_type(x) != jnp.bfloat16
                    else x.astype(np.float32) for x in choicelist]
      dtype = jnp.result_type(default, *choicelist)
      return np.select(condlist,
                        [np.asarray(x, dtype=dtype) for x in choicelist],
                        np.asarray(default, dtype=dtype))
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(np_fun, jnp.select, args_maker,
                              check_dtypes=False)
      self._CompileAndCheck(jnp.select, args_maker,
                            rtol={np.float64: 1e-7, np.complex128: 1e-7})

  # def testIssue330(self):
    # x = jnp.full((1, 1), jnp.array([1])[0])  # doesn't crash
    # self.assertEqual(x[0, 0], 1)

  def testScalarDtypePromotion(self):
    orig_numpy_result = (1 + np.eye(1, dtype=np.float32)).dtype
    jax_numpy_result = (1 + jnp.eye(1, dtype=jnp.float32)).dtype
    self.assertEqual(orig_numpy_result, jax_numpy_result)

  def testSymmetrizeDtypePromotion(self):
    x = np.eye(3, dtype=np.float32)
    orig_numpy_result = ((x + x.T) / 2).dtype

    x = jnp.eye(3, dtype=jnp.float32)
    jax_numpy_result = ((x + x.T) / 2).dtype
    self.assertEqual(orig_numpy_result, jax_numpy_result)

  def testIssue453(self):
    # https://github.com/google/jax/issues/453
    a = np.arange(6) + 1
    ans = jnp.reshape(a, (3, 2), order='F')
    expected = np.reshape(a, (3, 2), order='F')
    self.assertAllClose(ans, expected)

  @jtu.sample_product(
    input=[
      3,
      [3],
      [np.array(3)],
      [np.array([3])],
      [[np.array(3)]],
      [[np.array([3])]],
      [3, 4, 5],
      [
        [np.eye(2, dtype=np.int32) * 2, np.zeros((2, 3), dtype=np.int32)],
        [np.ones((3, 2), dtype=np.int32), np.eye(3, dtype=np.int32) * 3],
      ],
      [np.array([1, 2, 3]), np.array([2, 3, 4]), 10],
      [np.ones((2, 2), dtype=np.int32), np.zeros((2, 2), dtype=np.int32)],
      [[np.array([1, 2, 3])], [np.array([2, 3, 4])]],
    ],
  )
  def testBlock(self, input):
    args_maker = lambda: [input]
    self._CheckAgainstNumpy(np.block, jnp.block, args_maker)
    self._CompileAndCheck(jnp.block, args_maker)

  def testLongLong(self):
    self.assertAllClose(np.int64(7), jax.jit(lambda x: x)(np.longlong(7)))

  @jtu.sample_product(
    shape=[(5,), (10, 5), (4, 10)],
    dtype=number_dtypes,
    rowvar=[True, False],
  )
  @jax.default_matmul_precision("float32")
  def testCorrCoef(self, shape, dtype, rowvar):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      ok = False
      while not ok:
        x = rng(shape, dtype)
        ok = not np.any(np.isclose(np.std(x), 0.0))
      return (x,)
    np_fun = partial(np.corrcoef, rowvar=rowvar)
    np_fun = jtu.ignore_warning(
      category=RuntimeWarning, message="invalid value encountered.*")(np_fun)
    jnp_fun = partial(jnp.corrcoef, rowvar=rowvar)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(dtype=dtype, end_dtype=end_dtype, begin_dtype=begin_dtype,
          shape=shape, begin_shape=begin_shape, end_shape=end_shape)
      for dtype in number_dtypes
      for end_dtype in [None] + [dtype]
      for begin_dtype in [None] + [dtype]
      for shape in [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE]
      for begin_shape in (
        [None] if begin_dtype is None
        else [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE])
      for end_shape in (
        [None] if end_dtype is None
        else [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE])
    ],
  )
  def testEDiff1d(self, shape, dtype, end_shape, end_dtype, begin_shape,
                  begin_dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype),
            (None if end_dtype is None else rng(end_shape, end_dtype)),
            (None if begin_dtype is None else rng(begin_shape, begin_dtype))]
    np_fun = lambda x, to_end, to_begin: np.ediff1d(x, to_end, to_begin)
    jnp_fun = lambda x, to_end, to_begin: jnp.ediff1d(x, to_end, to_begin)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testEDiff1dWithDtypeCast(self):
    rng = jtu.rand_default(self.rng())
    shape = jtu.NUMPY_SCALAR_SHAPE
    dtype = jnp.float32
    end_dtype = jnp.int32
    args_maker = lambda: [rng(shape, dtype), rng(shape, end_dtype), rng(shape, dtype)]
    np_fun = lambda x, to_end, to_begin: np.ediff1d(x, to_end, to_begin)
    jnp_fun = lambda x, to_end, to_begin: jnp.ediff1d(x, to_end, to_begin)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shapes=[(), (5,), (5, 3)],
    dtype=number_dtypes,
    indexing=['xy', 'ij'],
    sparse=[True, False],
  )
  def testMeshGrid(self, shapes, dtype, indexing, sparse):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [(x,) for x in shapes],
                                    [dtype] * len(shapes))
    np_fun = partial(np.meshgrid, indexing=indexing, sparse=sparse)
    jnp_fun = partial(jnp.meshgrid, indexing=indexing, sparse=sparse)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testOgrid(self):
    # wrap indexer for appropriate dtype defaults.
    np_ogrid = _indexer_with_default_outputs(np.ogrid)
    def assertListOfArraysEqual(xs, ys):
      self.assertIsInstance(xs, list)
      self.assertIsInstance(ys, list)
      self.assertEqual(len(xs), len(ys))
      for x, y in zip(xs, ys):
        self.assertArraysEqual(x, y)

    self.assertArraysEqual(np_ogrid[1:7:2], jnp.ogrid[1:7:2])
    # List of arrays
    assertListOfArraysEqual(np_ogrid[:5,], jnp.ogrid[:5,])
    assertListOfArraysEqual(np_ogrid[0:5, 1:3], jnp.ogrid[0:5, 1:3])
    assertListOfArraysEqual(np_ogrid[1:3:2, 2:9:3], jnp.ogrid[1:3:2, 2:9:3])
    assertListOfArraysEqual(np_ogrid[:5, :9, :11], jnp.ogrid[:5, :9, :11])

  def testR_(self):
    a = np.arange(6).reshape((2,3))
    self.assertArraysEqual(np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])],
                           jnp.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])])
    self.assertArraysEqual(np.r_['-1', a, a], jnp.r_['-1', a, a])

    self.assertArraysEqual(np.r_['0,2', [1,2,3], [4,5,6]], jnp.r_['0,2', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.r_['0,2,0', [1,2,3], [4,5,6]], jnp.r_['0,2,0', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.r_['1,2,0', [1,2,3], [4,5,6]], jnp.r_['1,2,0', [1,2,3], [4,5,6]])
    # negative 1d axis start
    self.assertArraysEqual(np.r_['0,4,-1', [1,2,3], [4,5,6]], jnp.r_['0,4,-1', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.r_['0,4,-2', [1,2,3], [4,5,6]], jnp.r_['0,4,-2', [1,2,3], [4,5,6]])

    # matrix directives
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
      self.assertArraysEqual(np.r_['r',[1,2,3], [4,5,6]], jnp.r_['r',[1,2,3], [4,5,6]])
      self.assertArraysEqual(np.r_['c', [1, 2, 3], [4, 5, 6]], jnp.r_['c', [1, 2, 3], [4, 5, 6]])

    # bad directive
    with self.assertRaisesRegex(ValueError, "could not understand directive.*"):
      jnp.r_["asdfgh",[1,2,3]]
    # abstract tracer value for r_ slice
    with self.assertRaisesRegex(core.ConcretizationTypeError,
                                "slice start of jnp.r_"):
      jax.jit(lambda a, b: jnp.r_[a:b])(0, 2)

    # wrap indexer for appropriate dtype defaults.
    np_r_ = _indexer_with_default_outputs(np.r_)

    # Complex number steps
    atol = 1e-6
    rtol = 1e-6
    self.assertAllClose(np_r_[-1:1:6j],
                        jnp.r_[-1:1:6j],
                        atol=atol,
                        rtol=rtol)
    with jax.numpy_dtype_promotion('standard'):  # Requires dtype promotion.
      self.assertAllClose(np_r_[-1:1:6j, [0]*3, 5, 6],
                          jnp.r_[-1:1:6j, [0]*3, 5, 6],
                          atol=atol,
                          rtol=rtol)
    # Non-integer steps
    self.assertAllClose(np_r_[1.2:4.8:0.24],
                        jnp.r_[1.2:4.8:0.24],
                        atol=atol,
                        rtol=rtol)

  def testC_(self):
    a = np.arange(6).reshape((2, 3))
    self.assertArraysEqual(np.c_[np.array([1,2,3]), np.array([4,5,6])],
                           jnp.c_[np.array([1,2,3]), np.array([4,5,6])])
    self.assertArraysEqual(np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])],
                           jnp.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])])
    self.assertArraysEqual(np.c_['-1', a, a], jnp.c_['-1', a, a])

    self.assertArraysEqual(np.c_['0,2', [1,2,3], [4,5,6]], jnp.c_['0,2', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.c_['0,2,0', [1,2,3], [4,5,6]], jnp.c_['0,2,0', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.c_['1,2,0', [1,2,3], [4,5,6]], jnp.c_['1,2,0', [1,2,3], [4,5,6]])
    # negative 1d axis start
    self.assertArraysEqual(np.c_['0,4,-1', [1,2,3], [4,5,6]], jnp.c_['0,4,-1', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.c_['0,4,-2', [1,2,3], [4,5,6]], jnp.c_['0,4,-2', [1,2,3], [4,5,6]])
    # matrix directives, avoid numpy deprecation warning
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
      self.assertArraysEqual(np.c_['r',[1,2,3], [4,5,6]], jnp.c_['r',[1,2,3], [4,5,6]])
      self.assertArraysEqual(np.c_['c', [1, 2, 3], [4, 5, 6]], jnp.c_['c', [1, 2, 3], [4, 5, 6]])

    # bad directive
    with self.assertRaisesRegex(ValueError, "could not understand directive.*"):
      jnp.c_["asdfgh",[1,2,3]]
    # abstract tracer value for c_ slice
    with self.assertRaisesRegex(core.ConcretizationTypeError,
                                "slice start of jnp.c_"):
      jax.jit(lambda a, b: jnp.c_[a:b])(0, 2)

    # wrap indexer for appropriate dtype defaults.
    np_c_ = _indexer_with_default_outputs(np.c_)

    # Complex number steps
    atol = 1e-6
    rtol = 1e-6
    self.assertAllClose(np_c_[-1:1:6j],
                        jnp.c_[-1:1:6j],
                        atol=atol,
                        rtol=rtol)

    # Non-integer steps
    self.assertAllClose(np_c_[1.2:4.8:0.24],
                        jnp.c_[1.2:4.8:0.24],
                        atol=atol,
                        rtol=rtol)

  def testS_(self):
    self.assertEqual(np.s_[1:2:20],jnp.s_[1:2:20])

  def testIndex_exp(self):
    self.assertEqual(np.index_exp[5:3:2j],jnp.index_exp[5:3:2j])

  @jtu.sample_product(
    start_shape=[(2,), (2, 2)],
    stop_shape=[(2,), (2, 2)],
    num=[1, 2, 5, 20],
    endpoint=[True, False],
    retstep=[True, False],
    # floating-point compute between jitted platforms and non-jit + rounding
    # cause unavoidable variation in integer truncation for some inputs, so
    # we currently only test inexact 'dtype' arguments.
    dtype=inexact_dtypes + [None,],
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLinspace(self, start_shape, stop_shape, num, endpoint, retstep, dtype):
    rng = jtu.rand_default(self.rng())
    # relax default tolerances slightly
    tol = jtu.tolerance(dtype if dtype else np.float32) * 10
    args_maker = self._GetArgsMaker(rng,
                                    [start_shape, stop_shape],
                                    [dtype, dtype])
    start, stop = args_maker()
    ndim = len(np.shape(start + stop))
    for axis in range(-ndim, ndim):
      jnp_op = lambda start, stop: jnp.linspace(
        start, stop, num,
        endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis)
      np_op = lambda start, stop: np.linspace(
        start, stop, num,
        endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis)

      self._CheckAgainstNumpy(np_op, jnp_op, args_maker,
                              check_dtypes=False, tol=tol)
      self._CompileAndCheck(jnp_op, args_maker,
                            check_dtypes=False, atol=tol, rtol=tol)

  @jtu.sample_product(dtype=number_dtypes)
  def testLinspaceEndpoints(self, dtype):
    """Regression test for Issue #3014."""
    rng = jtu.rand_default(self.rng())
    endpoints = rng((2,), dtype)
    out = jnp.linspace(*endpoints, 10, dtype=dtype)
    self.assertAllClose(out[np.array([0, -1])], endpoints, rtol=0, atol=0)

  @jtu.sample_product(
    start_shape=[(2,), (2, 2)],
    stop_shape=[(2,), (2, 2)],
    num=[1, 2, 5, 20],
    endpoint=[True, False],
    base=[10.0, 2, np.e],
    # skip 16-bit floats due to insufficient precision for the test.
    dtype=[np.float32] + [None,],
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLogspace(self, start_shape, stop_shape, num,
                   endpoint, base, dtype):
    if (dtype in int_dtypes and
        jtu.device_under_test() in ("gpu", "tpu", "METAL") and
        not config.x64_enabled):
      raise unittest.SkipTest("GPUx32 truncated exponentiation"
                              " doesn't exactly match other platforms.")
    rng = jtu.rand_default(self.rng())
    # relax default tolerances slightly
    tol = {np.float32: 1e-2, np.float64: 1e-6, np.complex64: 1e-3, np.complex128: 1e-6}
    args_maker = self._GetArgsMaker(rng,
                                    [start_shape, stop_shape],
                                    [dtype, dtype])
    start, stop = args_maker()
    ndim = len(np.shape(start + stop))
    for axis in range(-ndim, ndim):
      jnp_op = lambda start, stop: jnp.logspace(
        start, stop, num, endpoint=endpoint, base=base, dtype=dtype, axis=axis)
      @jtu.ignore_warning(category=RuntimeWarning,
                          message="overflow encountered in power")
      def np_op(start, stop):
        return np.logspace(start, stop, num, endpoint=endpoint,
                           base=base, dtype=dtype, axis=axis)
      self._CheckAgainstNumpy(np_op, jnp_op, args_maker,
                              check_dtypes=False, tol=tol)
      if dtype in (inexact_dtypes + [None,]):
        # Why do compiled and op-by-op float16 np.power numbers differ
        # slightly more than expected?
        atol = {np.float16: 1e-2}
        self._CompileAndCheck(jnp_op, args_maker,
                              check_dtypes=False, atol=atol, rtol=tol)

  def testStackArrayArgument(self):
    # tests https://github.com/google/jax/issues/1271
    @jax.jit
    def foo(x):
      return jnp.stack(x)
    foo(np.zeros(2))  # doesn't crash

    @jax.jit
    def foo(x):
      return jnp.concatenate(x)
    foo(np.zeros((2, 2)))  # doesn't crash

  def testReluGradientConstants(self):
    # This is a regression test that verifies that constants associated with the
    # gradient of np.maximum (from lax._balanced_eq) aren't hoisted into the
    # outermost jaxpr. This was producing some large materialized constants for
    # every relu activation in a model.
    def body(i, xy):
      x, y = xy
      y = y + jax.grad(lambda z: jnp.sum(jnp.maximum(z, 0.)))(x)
      return x, y

    f = lambda y: lax.fori_loop(0, 5, body, (y, y))
    jaxpr = jax.make_jaxpr(f)(np.zeros((3, 4), np.float32))
    self.assertFalse(
      any(np.array_equal(x, np.full((3, 4), 2., dtype=np.float32))
          for x in jaxpr.consts))

  @jtu.sample_product(
    [dict(from_shape=from_shape, to_shape=to_shape)
      for from_shape, to_shape in [
          [(1, 3), (4, 3)],
          [(3,), (2, 1, 3)],
          [(3,), (3, 3)],
          [(1,), (3,)],
          [(1,), 3],
      ]
    ],
  )
  def testBroadcastTo(self, from_shape, to_shape):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [from_shape], [np.float32])
    np_op = lambda x: np.broadcast_to(x, to_shape)
    jnp_op = lambda x: jnp.broadcast_to(x, to_shape)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    [dict(shapes=shapes, broadcasted_shape=broadcasted_shape)
      for shapes, broadcasted_shape in [
        [[], ()],
        [[()], ()],
        [[(1, 3), (4, 3)], (4, 3)],
        [[(3,), (2, 1, 3)], (2, 1, 3)],
        [[(3,), (3, 3)], (3, 3)],
        [[(1,), (3,)], (3,)],
        [[(1,), 3], (3,)],
        [[(6, 7), (5, 6, 1), (7,), (5, 1, 7)], (5, 6, 7)],
        [[[1], [0, 1]], (0, 1)],
        [[(1,), np.array([0, 1])], (0, 1)],
      ]
    ],
  )
  def testBroadcastShapes(self, shapes, broadcasted_shape):
    # Test against np.broadcast_shapes once numpy 1.20 is minimum required version
    np.testing.assert_equal(jnp.broadcast_shapes(*shapes), broadcasted_shape)

  def testBroadcastToIssue1522(self):
    self.assertRaisesRegex(
        ValueError, "Incompatible shapes for broadcasting: .*",
        lambda: jnp.broadcast_to(np.ones((2, 3)), (1, 3)))

  def testBroadcastToIntIssue1548(self):
    self.assertAllClose(jnp.broadcast_to(1, (3, 2)), np.ones((3, 2)),
                        check_dtypes=False)

  def testBroadcastToOnScalar(self):
    self.assertIsInstance(jnp.broadcast_to(10.0, ()), jax.Array)
    self.assertIsInstance(np.broadcast_to(10.0, ()), np.ndarray)

  def testPrecision(self):

    ones_1d = np.ones((2,))
    ones_2d = np.ones((2, 2))
    ones_3d = np.ones((2, 2, 2))
    HIGHEST = lax.Precision.HIGHEST

    jtu.assert_dot_precision(None, jnp.dot, ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.dot, precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.dot, precision=HIGHEST),
        ones_3d, ones_3d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.matmul, precision=HIGHEST),
        ones_2d, ones_2d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.vdot, precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.tensordot, axes=2, precision=HIGHEST),
        ones_2d, ones_2d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.tensordot, axes=(0, 0), precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.tensordot, axes=((0,), (0,)), precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.einsum, 'i,i', precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.einsum, 'ij,ij', precision=HIGHEST),
        ones_2d, ones_2d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.inner, precision=HIGHEST),
        ones_1d, ones_1d)

  @jtu.sample_product(
    [dict(shape=shape, varargs=varargs, axis=axis)
        for shape in [(10,), (10, 15), (10, 15, 20)]
        for _num_axes in range(len(shape))
        for varargs in itertools.combinations(range(1, len(shape) + 1), _num_axes)
        for axis in itertools.combinations(range(len(shape)), _num_axes)
    ],
    dtype=inexact_dtypes,
  )
  def testGradient(self, shape, varargs, axis, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_fun = lambda y: jnp.gradient(y, *varargs, axis=axis)
    np_fun = lambda y: np.gradient(y, *varargs, axis=axis)
    self._CheckAgainstNumpy(
        np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testZerosShapeErrors(self):
    # see https://github.com/google/jax/issues/1822
    self.assertRaisesRegex(
        TypeError,
        "Shapes must be 1D sequences of concrete values of integer type.*",
        lambda: jnp.zeros(1.))
    self.assertRaisesRegex(
        TypeError,
        r"Shapes must be 1D sequences of concrete values of integer type.*\n"
        "If using `jit`, try using `static_argnums` or applying `jit` to "
        "smaller subfunctions.",
        lambda: jax.jit(jnp.zeros)(2))

  def testTraceMethod(self):
    x = self.rng().randn(3, 4).astype(jnp.float_)
    self.assertAllClose(x.trace(), jnp.array(x).trace())
    self.assertAllClose(x.trace(), jax.jit(lambda y: y.trace())(x))

  def testIntegerPowersArePrecise(self):
    # See https://github.com/google/jax/pull/3036
    # Checks if the squares of float32 integers have no numerical errors.
    # It should be satisfied with all integers less than sqrt(2**24).
    x = jnp.arange(-2**12, 2**12, dtype=jnp.int32)
    np.testing.assert_array_equal(jnp.square(x.astype(jnp.float32)), x * x)
    np.testing.assert_array_equal(x.astype(jnp.float32) ** 2, x * x)

    # Similarly for cubes.
    x = jnp.arange(-2**8, 2**8, dtype=jnp.int32)
    np.testing.assert_array_equal(x.astype(jnp.float32) ** 3, x * x * x)

    x = np.arange(10, dtype=np.float32)
    for i in range(10):
      self.assertAllClose(x.astype(jnp.float32) ** i, x ** i,
                          check_dtypes=False)

  def testToBytes(self):
    v = np.arange(12, dtype=np.int32).reshape(3, 4)
    for order in ['C', 'F']:
      self.assertEqual(jnp.asarray(v).tobytes(order), v.tobytes(order))

  def testToList(self):
    v = np.arange(12, dtype=np.int32).reshape(3, 4)
    self.assertEqual(jnp.asarray(v).tolist(), v.tolist())

  @jtu.skip_on_devices("METAL")
  def testArangeConcretizationError(self):
    msg = r"It arose in jax.numpy.arange argument `{}`".format
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg('stop')):
      jax.jit(jnp.arange)(3)

    with self.assertRaisesRegex(core.ConcretizationTypeError, msg('start')):
      jax.jit(lambda start: jnp.arange(start, 3))(0)

    with self.assertRaisesRegex(core.ConcretizationTypeError, msg('stop')):
      jax.jit(lambda stop: jnp.arange(0, stop))(3)

  @jtu.sample_product(dtype=[None] + float_dtypes)
  def testArange64Bit(self, dtype):
    # Test that jnp.arange uses 64-bit arithmetic to define its range, even if the
    # output has another dtype. The issue here is that if python scalar inputs to
    # jnp.arange are cast to float32 before the range is computed, it changes the
    # number of elements output by the range.  It's unclear whether this was deliberate
    # behavior in the initial implementation, but it's behavior that downstream users
    # have come to rely on.
    args = (1.2, 4.8, 0.24)

    # Ensure that this test case leads to differing lengths if cast to float32.
    self.assertLen(np.arange(*args), 15)
    self.assertLen(np.arange(*map(np.float32, args)), 16)

    jnp_fun = lambda: jnp.arange(*args, dtype=dtype)
    np_fun = jtu.with_jax_dtype_defaults(lambda: np.arange(*args, dtype=dtype), dtype is None)
    args_maker = lambda: []
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testIssue2347(self):
    # https://github.com/google/jax/issues/2347
    object_list = List[Tuple[jnp.array, float, float, jnp.array, bool]]
    self.assertRaises(TypeError, jnp.array, object_list)

    np_object_list = np.array(object_list)
    self.assertRaises(TypeError, jnp.array, np_object_list)

  def testDefaultDtypes(self):
    precision = config.default_dtype_bits.value
    assert precision in ['32', '64']
    self.assertEqual(jnp.bool_, np.bool_)
    self.assertEqual(jnp.int_, np.int32 if precision == '32' else np.int64)
    self.assertEqual(jnp.uint, np.uint32 if precision == '32' else np.uint64)
    self.assertEqual(jnp.float_, np.float32 if precision == '32' else np.float64)
    self.assertEqual(jnp.complex_, np.complex64 if precision == '32' else np.complex128)

  def testFromBuffer(self):
    buf = b'\x01\x02\x03'
    expected = np.frombuffer(buf, dtype='uint8')
    actual = jnp.frombuffer(buf, dtype='uint8')
    self.assertArraysEqual(expected, actual)

  def testFromFunction(self):
    def f(x, y, z):
      return x + 2 * y + 3 * z
    shape = (3, 4, 5)
    expected = np.fromfunction(f, shape=shape)
    actual = jnp.fromfunction(f, shape=shape)
    self.assertArraysEqual(expected, actual, check_dtypes=False)

  def testFromString(self):
    s = "1,2,3"
    expected = np.fromstring(s, sep=',', dtype=int)
    actual = jnp.fromstring(s, sep=',', dtype=int)
    self.assertArraysEqual(expected, actual)


# Most grad tests are at the lax level (see lax_test.py), but we add some here
# as needed for e.g. particular compound ops of interest.

GradTestSpec = collections.namedtuple(
    "GradTestSpec",
    ["op", "nargs", "order", "rng_factory", "dtypes", "name", "tol"])
def grad_test_spec(op, nargs, order, rng_factory, dtypes, name=None, tol=None):
  return GradTestSpec(
      op, nargs, order, rng_factory, dtypes, name or op.__name__, tol)

GRAD_TEST_RECORDS = [
    grad_test_spec(jnp.arcsinh, nargs=1, order=2,
                   rng_factory=jtu.rand_positive,
                   dtypes=[np.float64, np.complex64],
                   tol={np.complex64: 2e-2}),
    grad_test_spec(jnp.arccosh, nargs=1, order=1,
                   rng_factory=jtu.rand_positive,
                   dtypes=[np.float64, np.complex64],
                   tol={np.complex64: 2e-2}),
    grad_test_spec(jnp.arctanh, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=-0.9, high=0.9),
                   dtypes=[np.float64, np.complex64],
                   tol={np.complex64: 2e-2}),
    grad_test_spec(jnp.logaddexp, nargs=2, order=1,
                   rng_factory=partial(jtu.rand_uniform, low=-0.9, high=0.9),
                   dtypes=[np.float64], tol=1e-4),
    grad_test_spec(jnp.logaddexp2, nargs=2, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=-0.9, high=0.9),
                   dtypes=[np.float64], tol=1e-4),
]

GradSpecialValuesTestSpec = collections.namedtuple(
    "GradSpecialValuesTestSpec", ["op", "values", "order"])

GRAD_SPECIAL_VALUE_TEST_RECORDS = [
    GradSpecialValuesTestSpec(jnp.arcsinh, [0., 1000.], 2),
    GradSpecialValuesTestSpec(jnp.arccosh, [1000.], 2),
    GradSpecialValuesTestSpec(jnp.arctanh, [0.], 2),
    GradSpecialValuesTestSpec(jnp.sinc, [0.], 1),
]


from jaxlib import xla_client
from jax_plugins import metal_plugin
class ReportedIssuesTests(jtu.JaxTestCase):
  def dispatchOn(self, args, func, device=jax.devices('cpu')[0]):
    deviceArgs = []
    for arg in args:
       deviceArgs.append(jax.device_put(arg, device))
    return func(*deviceArgs)

  @staticmethod
  def compile_and_exec(module, args, run_on_cpu=False):
    backend = jax.lib.xla_bridge.get_backend('METAL')
    if (run_on_cpu):
      backend = jax.lib.xla_bridge.get_backend('cpu')
    executables = backend.compile(module)
    return xla_client.execute_with_python_values(executables, args, backend)

  @staticmethod
  def jax_metal_supported(target_ver):
    if metal_plugin is None or not hasattr(metal_plugin, 'version'):
      return False
    curr_ver = metal_plugin.version()
    if hasattr(jtu, 'parse_version'):
      return jtu.parse_version(curr_ver) >= jtu.parse_version(target_ver)
    return False

  
  #https://github.com/google/jax/issues/16420
  def test_broadcast_dim(self):
      x = jnp.arange(2)
      f = lambda x : jax.lax.broadcast_in_dim(x, (2, 2), (0,))
      res = f(x)
      print(res)
      res_cpu = self.dispatchOn([x],f)
      jtu.check_eq(res, res_cpu)
      f = lambda x : jax.lax.broadcast_in_dim(x, (2, 2), (1,))
      res = f(x)
      print(res)
      res_cpu = self.dispatchOn([x],f)
      jtu.check_eq(res, res_cpu)

  def test_identity(self):
      x = jnp.identity(4)
      jtu.check_eq(x, np.identity(4))
  
  def test_triu(self):
      x = np.ones((4,4))
      res = jnp.triu(x)
      jtu.check_eq(res, np.triu(x))
  
  #https://github.com/google/jax/issues/16471
  def test_matmul_1d(self):
      x = np.array(np.random.rand(3, 3))
      y = np.array(np.random.rand(3))
      z = np.array(np.random.rand(3))
      res = jnp.dot(y, z)
      self.assertArraysAllClose(res, np.dot(y,z))
      res = jnp.dot(x, y)
      self.assertArraysAllClose(res, np.dot(x,y))
  
  #https://github.com/google/jax/issues/17175
  def test_indexing(self):
      x = jnp.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=jnp.float32)
      @jax.vmap
      def f(i):
          return x[i]
      f = jax.jit(f)
      idx = jnp.array([1,1,2,2,0])
      res = f(idx)
      jtu.check_eq(res, np.array([[4., 5., 6.], [4., 5., 6.], [7., 8., 9.], [7., 8., 9.], [1., 2., 3.]]))

  #https://github.com/google/jax/issues/17344
  def test_take_along_axis(self):
    @jax.jit
    def f():
      idx = jnp.array([[0],[0],[0]])
      x = jnp.array([[0.3756883,  0.05820537, 0.7399422,  0.45242703],
                    [0.5848844,  0.18772626, 0.47942543, 0.20703673],
                    [0.1071583,  0.26139486, 0.25664794, 0.8109596]])
      return jnp.take_along_axis(x, idx, axis=1)
    jtu.check_eq(f(), self.dispatchOn([], f))
  
  #https://github.com/google/jax/issues/17590
  def test_in1d(self):
    a = np.array([123,2,4])
    b = np.array([123,1])
    res = jnp.isin(a,b)
    jtu.check_eq(res, np.isin(a, b))
  
  def test_indexing_update(self):
    x = jnp.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=jnp.float32)
    @jax.vmap
    def f(x):
      return x.at[0].set(1.0)
    f = jax.jit(f)
    res= f(x)
    jtu.check_eq(res, np.array([[1., 2., 3.], [1., 5., 6.,], [1., 8., 9.], [1., 11., 12.]]))

  #https://github.com/google/jax/issues/16326
  def test_indexing_update2(self):
    @jax.jit
    def f(x, r):
        x = x.at[:, 0].set(x[:, 0] / r)
        return x
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    fx = f(x, jnp.array([10.0]))
    jtu.check_eq(fx, np.array([[0.1, 2.0], [0.3, 4.]]))

  def test_gather_ir(self):
    ir = '''
#loc = loc(unknown)
module @jit_gather attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x2x3xf32> {mhlo.sharding = "{replicated}"} loc(unknown), %arg1: tensor<3x2xi32> {mhlo.sharding = "{replicated}"} loc(unknown)) -> tensor<3x2xf32> {
    %0 = "stablehlo.gather"(%arg0, %arg1) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 2], start_index_map = [0, 2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 2, 1]> : tensor<3xi64>} : (tensor<3x2x3xf32>, tensor<3x2xi32>) -> tensor<3x2xf32> loc(#loc2)
    return %0 : tensor<3x2xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/Users/shuhan/Code/jax-metal/tests/lax_numpy_indexing_test.py":1156:0)
#loc2 = loc("jit(gather)/jit(main)/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0, 2), start_index_map=(0, 2)) slice_sizes=(1, 2, 1) unique_indices=False indices_are_sorted=False mode=GatherScatterMode.CLIP fill_value=None]"(#loc1))
    '''
    data = np.array([[[0.6369617,  0.26978672, 0.04097353],
                      [0.01652764, 0.8132702,  0.91275555]],
                     [[0.60663575, 0.72949654, 0.543625  ],
                      [0.9350724,  0.81585354, 0.0027385 ]],
                      [[0.8574043,  0.03358557, 0.72965544],
                      [0.17565562, 0.8631789,  0.5414612 ]]], dtype=np.float32)
    index = np.array([[1, 0],[2, 1],[0, 2]], dtype=np.int32)
    res = ReportedIssuesTests.compile_and_exec(ir, [data, index])
    res_ref = ReportedIssuesTests.compile_and_exec(ir, [data, index], run_on_cpu = True)
    print(res)
    jtu.check_eq(res, res_ref)
  
  #https://github.com/google/jax/issues/16366
  def test_pad_interior_1(self):
    if not ReportedIssuesTests.jax_metal_supported('0.0.6'):
      raise unittest.SkipTest("jax-metal version doesn't support it.")
    ir = '''
    module @jit_gather attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
      func.func public @main(%arg0: tensor<128x7x7x64xf32> {mhlo.sharding = "{replicated}"} loc(unknown), %arg1: tensor<f32> {mhlo.sharding = "{replicated}"} loc(unknown)) -> tensor<128x15x15x64xf32> {
        %206 = "mhlo.pad"(%arg0, %arg1) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>, interior_padding = dense<[0, 1, 1, 0]> : tensor<4xi64>} : (tensor<128x7x7x64xf32>, tensor<f32>) -> tensor<128x15x15x64xf32>
        return %206 : tensor<128x15x15x64xf32>
      }
    }
    '''
    data = np.random.rand(128,7,7,64).astype(np.float32)
    padding = np.array(0.5, dtype=np.float32)
    res = ReportedIssuesTests.compile_and_exec(ir, [data, padding])
    res_ref = ReportedIssuesTests.compile_and_exec(ir, [data, padding], run_on_cpu = True)
    jtu.check_eq(res, res_ref)

  def test_pad_interior_2(self):
    if not ReportedIssuesTests.jax_metal_supported('0.0.6'):
      raise unittest.SkipTest("jax-metal version doesn't support it.")
    batch = 2
    seq_len = 8
    num_decode = 32

    seq = np.random.randint(size=(batch, seq_len, num_decode), low=0, high=256, dtype=np.uint8)
    res = jnp.cumsum(seq, axis=-1)
    res_ref = np.cumsum(seq, axis=-1, dtype=np.uint8)
    jtu.check_eq(res, res_ref)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
