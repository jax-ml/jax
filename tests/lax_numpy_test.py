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
import functools
from functools import partial
import inspect
import itertools
import operator
from typing import cast, Optional
import unittest
from unittest import SkipTest
import warnings

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
import jax.ops
from jax import api
from jax import lax
from jax import linear_util
from jax import numpy as jnp
from jax import test_util as jtu
from jax import dtypes
from jax import tree_util
from jax.interpreters import partial_eval, xla
from jax.test_util import check_grads

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
all_shapes =  scalar_shapes + array_shapes

float_dtypes = jtu.dtypes.all_floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.integer
unsigned_dtypes = jtu.dtypes.unsigned
bool_dtypes = jtu.dtypes.boolean
default_dtypes = float_dtypes + int_dtypes
inexact_dtypes = float_dtypes + complex_dtypes
number_dtypes = float_dtypes + complex_dtypes + int_dtypes
all_dtypes = number_dtypes + bool_dtypes


python_scalar_dtypes = [jnp.bool_, jnp.int_, jnp.float_, jnp.complex_]

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

OpRecord = collections.namedtuple(
  "OpRecord",
  ["name", "nargs", "dtypes", "shapes", "rng_factory", "diff_modes",
   "test_name", "check_dtypes", "tolerance", "inexact"])

def op_record(name, nargs, dtypes, shapes, rng_factory, diff_modes,
              test_name=None, check_dtypes=True,
              tolerance=None, inexact=False):
  test_name = test_name or name
  return OpRecord(name, nargs, dtypes, shapes, rng_factory, diff_modes,
                  test_name, check_dtypes, tolerance, inexact)

JAX_ONE_TO_ONE_OP_RECORDS = [
    op_record("abs", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("add", 2, all_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("ceil", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("conj", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal, []),
    op_record("exp", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("fabs", 1, float_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("float_power", 2, inexact_dtypes, all_shapes,
              partial(jtu.rand_default, scale=1), ["rev"],
              tolerance={jnp.bfloat16: 1e-2, np.float32: 1e-3,
                         np.float64: 1e-12, np.complex64: 2e-4,
                         np.complex128: 1e-12}, check_dtypes=False),
    op_record("floor", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("greater", 2, all_dtypes, all_shapes, jtu.rand_some_equal, []),
    op_record("greater_equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal, []),
    op_record("less", 2, all_dtypes, all_shapes, jtu.rand_some_equal, []),
    op_record("less_equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal, []),
    op_record("log", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("logical_and", 2, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("logical_not", 1, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("logical_or", 2, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("logical_xor", 2, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("maximum", 2, all_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("minimum", 2, all_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("multiply", 2, all_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("negative", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("nextafter", 2, [f for f in float_dtypes if f != jnp.bfloat16],
              all_shapes, jtu.rand_default, ["rev"], inexact=True, tolerance=0),
    op_record("not_equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal, ["rev"]),
    op_record("array_equal", 2, number_dtypes, all_shapes, jtu.rand_some_equal, ["rev"]),
    op_record("reciprocal", 1, inexact_dtypes, all_shapes, jtu.rand_default, []),
    op_record("subtract", 2, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("signbit", 1, default_dtypes + bool_dtypes, all_shapes,
              jtu.rand_some_inf_and_nan, ["rev"]),
    op_record("trunc", 1, float_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("sin", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("cos", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("tan", 1, number_dtypes, all_shapes,
              partial(jtu.rand_uniform, low=-1.5, high=1.5), ["rev"],
              inexact=True),
    op_record("sinh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("cosh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    # TODO(b/142975473): on CPU, tanh for complex128 is only accurate to
    # ~float32 precision.
    # TODO(b/143135720): on GPU, tanh has only ~float32 precision.
    op_record("tanh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              tolerance={np.float64: 1e-7, np.complex128: 1e-7},
              inexact=True),
    op_record("arcsin", 1, float_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arccos", 1, float_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arctan", 1, float_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arctan2", 2, float_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arcsinh", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("arccosh", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("arctanh", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True, tolerance={np.float64: 1e-9}),
]

JAX_COMPOUND_OP_RECORDS = [
    # angle has inconsistent 32/64-bit return types across numpy versions.
    op_record("angle", 1, number_dtypes, all_shapes, jtu.rand_default, [],
              check_dtypes=False, inexact=True),
    op_record("atleast_1d", 1, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("atleast_2d", 1, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("atleast_3d", 1, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("cbrt", 1, default_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("conjugate", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("deg2rad", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("divide", 2, number_dtypes, all_shapes, jtu.rand_nonzero, ["rev"],
              inexact=True),
    op_record("divmod", 2, int_dtypes + float_dtypes, all_shapes,
              jtu.rand_nonzero, []),
    op_record("exp2", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              tolerance={jnp.bfloat16: 4e-2, np.float16: 1e-2}, inexact=True),
    # TODO(b/142975473): on CPU, expm1 for float64 is only accurate to ~float32
    # precision.
    op_record("expm1", 1, number_dtypes, all_shapes, jtu.rand_positive, [],
              test_name="expm1_large", tolerance={np.float64: 1e-8}, inexact=True),
    op_record("expm1", 1, number_dtypes, all_shapes, jtu.rand_small_positive,
              [], tolerance={np.float64: 1e-8}, inexact=True),
    op_record("fix", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("floor_divide", 2, number_dtypes, all_shapes,
              jtu.rand_nonzero, ["rev"]),
    op_record("floor_divide", 2, unsigned_dtypes, all_shapes,
              jtu.rand_nonzero, ["rev"]),
    op_record("fmin", 2, number_dtypes, all_shapes, jtu.rand_some_nan, []),
    op_record("fmax", 2, number_dtypes, all_shapes, jtu.rand_some_nan, []),
    op_record("heaviside", 2, default_dtypes, all_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("hypot", 2, default_dtypes, all_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("kron", 2, number_dtypes, nonempty_shapes, jtu.rand_default, []),
    op_record("outer", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("imag", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("iscomplex", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("isfinite", 1, inexact_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isinf", 1, inexact_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isnan", 1, inexact_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isneginf", 1, float_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isposinf", 1, float_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isreal", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("isrealobj", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("log2", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("log10", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("log1p", 1, number_dtypes, all_shapes, jtu.rand_positive, [],
              test_name="log1p_large", tolerance={np.float64: 1e-12},
              inexact=True),
    op_record("log1p", 1, number_dtypes, all_shapes, jtu.rand_small_positive, [],
              tolerance={np.float64: 1e-12}, inexact=True),
    op_record("logaddexp", 2, float_dtypes, all_shapes,
              jtu.rand_some_inf_and_nan, ["rev"],
              tolerance={np.float64: 1e-12}, inexact=True),
    op_record("logaddexp2", 2, float_dtypes, all_shapes,
              jtu.rand_some_inf_and_nan, ["rev"],
              tolerance={np.float16: 1e-2, np.float64: 2e-14}, inexact=True),
    op_record("polyval", 2, number_dtypes, nonempty_nonscalar_array_shapes,
              jtu.rand_default, [], check_dtypes=False,
              tolerance={dtypes.bfloat16: 4e-2, np.float16: 1e-2,
                         np.float64: 1e-12}),
    op_record("positive", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("power", 2, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              tolerance={np.complex128: 1e-14}),
    op_record("rad2deg", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("ravel", 1, all_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("real", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("remainder", 2, default_dtypes, all_shapes, jtu.rand_nonzero, [],
              tolerance={np.float16: 1e-2}),
    op_record("mod", 2, default_dtypes, all_shapes, jtu.rand_nonzero, []),
    op_record("rint", 1, number_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_some_inf_and_nan, []),
    op_record("sign", 1, number_dtypes + unsigned_dtypes,
              all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record('copysign', 2, default_dtypes, all_shapes, jtu.rand_some_inf_and_nan, [],
              check_dtypes=False),
    op_record("sinc", 1, [t for t in number_dtypes if t != jnp.bfloat16],
              all_shapes, jtu.rand_default, ["rev"],
              tolerance={np.complex64: 1e-5}, inexact=True,
              check_dtypes=False),
    op_record("square", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("sqrt", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("transpose", 1, all_dtypes, all_shapes, jtu.rand_default, ["rev"],
              check_dtypes=False),
    op_record("true_divide", 2, all_dtypes, all_shapes, jtu.rand_nonzero,
              ["rev"], inexact=True),
    op_record("diff", 1, number_dtypes, nonzerodim_shapes, jtu.rand_default, ["rev"]),
    op_record("ediff1d", 3, [np.int32], all_shapes, jtu.rand_default, []),
    op_record("unwrap", 1, float_dtypes, nonempty_nonscalar_array_shapes,
              jtu.rand_default, ["rev"],
              # numpy.unwrap always returns float64
              check_dtypes=False,
              # numpy cumsum is inaccurate, see issue #3517
              tolerance={dtypes.bfloat16: 1e-1, np.float16: 1e-1})
]

JAX_BITWISE_OP_RECORDS = [
    op_record("bitwise_and", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool, []),
    op_record("bitwise_not", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool, []),
    op_record("bitwise_or", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool, []),
    op_record("bitwise_xor", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool, []),
]

JAX_REDUCER_RECORDS = [
    op_record("mean", 1, number_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record("sum", 1, all_dtypes, all_shapes, jtu.rand_default, []),
    op_record("nanmean", 1, inexact_dtypes, nonempty_shapes, jtu.rand_some_nan,
              [], inexact=True),
    op_record("nanprod", 1, inexact_dtypes, all_shapes, jtu.rand_some_nan, []),
    op_record("nansum", 1, number_dtypes, all_shapes, jtu.rand_some_nan, []),
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
    op_record("nanvar", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan,
              [], inexact=True),
    op_record("nanstd", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan,
              [], inexact=True),
]

JAX_ARGMINMAX_RECORDS = [
    op_record("argmin", 1, default_dtypes, nonempty_shapes, jtu.rand_some_equal, []),
    op_record("argmax", 1, default_dtypes, nonempty_shapes, jtu.rand_some_equal, []),
    op_record("nanargmin", 1, default_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
    op_record("nanargmax", 1, default_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
]

JAX_OPERATOR_OVERLOADS = [
    op_record("__add__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__sub__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__mul__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__eq__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__ne__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__lt__", 2, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__le__", 2, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__gt__", 2, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__ge__", 2, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__pos__", 1, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__neg__", 1, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__pow__", 2, inexact_dtypes, all_shapes, jtu.rand_positive, [],
              tolerance={np.float32: 2e-4, np.complex64: 2e-4, np.complex128: 1e-14}),
    op_record("__mod__", 2, default_dtypes, all_shapes, jtu.rand_nonzero, [],
              tolerance={np.float16: 1e-1}),
    op_record("__floordiv__", 2, default_dtypes, all_shapes,
              jtu.rand_nonzero, []),
    op_record("__truediv__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, [],
              inexact=True),
    op_record("__abs__", 1, number_dtypes, all_shapes, jtu.rand_default, []),
    # TODO(mattjj): __invert__ fails on bool dtypes because ~True == -2
    op_record("__invert__", 1, int_dtypes, all_shapes, jtu.rand_default, []),
    # TODO(mattjj): investigate these failures
    # op_record("__or__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__and__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    # op_record("__xor__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__divmod__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, []),
    # TODO(mattjj): lshift, rshift
]

JAX_RIGHT_OPERATOR_OVERLOADS = [
    op_record("__radd__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__rsub__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__rmul__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__rpow__", 2, inexact_dtypes, all_shapes, jtu.rand_positive, [],
              tolerance={np.float32: 2e-4, np.complex64: 1e-3}),
    op_record("__rmod__", 2, default_dtypes, all_shapes, jtu.rand_nonzero, [],
              tolerance={np.float16: 1e-1}),
    op_record("__rfloordiv__", 2, default_dtypes, all_shapes,
              jtu.rand_nonzero, []),
    op_record("__rtruediv__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, [],
              inexact=True),
    # op_record("__ror__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__rand__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    # op_record("__rxor__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__rdivmod__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, []),
]

class _OverrideEverything(object):
  pass

for rec in JAX_OPERATOR_OVERLOADS + JAX_RIGHT_OPERATOR_OVERLOADS:
  if rec.nargs == 2:
    setattr(_OverrideEverything, rec.name, lambda self, other: self)

class _OverrideNothing(object):
  pass

for rec in JAX_OPERATOR_OVERLOADS + JAX_RIGHT_OPERATOR_OVERLOADS:
  if rec.nargs == 2:
    setattr(_OverrideNothing, rec.name, lambda self, other: NotImplemented)


numpy_version = tuple(map(int, np.version.version.split('.')))
if numpy_version >= (1, 15):
  JAX_COMPOUND_OP_RECORDS += [
      op_record("isclose", 2, [t for t in all_dtypes if t != jnp.bfloat16],
                all_shapes, jtu.rand_small_positive, []),
      op_record("gcd", 2, int_dtypes, all_shapes, jtu.rand_default, []),
      op_record("lcm", 2, int_dtypes, all_shapes, jtu.rand_default, []),
  ]
  JAX_REDUCER_NO_DTYPE_RECORDS += [
      op_record("ptp", 1, number_dtypes, nonempty_shapes, jtu.rand_default, []),
  ]


def _dtypes_are_compatible_for_bitwise_ops(args):
  if len(args) <= 1:
    return True
  is_signed = lambda dtype: jnp.issubdtype(dtype, np.signedinteger)
  width = lambda dtype: jnp.iinfo(dtype).bits
  x, y = args
  if width(x) > width(y):
    x, y = y, x
  # The following condition seems a little ad hoc, but seems to capture what
  # numpy actually implements.
  return (
      is_signed(x) == is_signed(y)
      or (width(x) == 32 and width(y) == 32)
      or (width(x) == 32 and width(y) == 64 and is_signed(y)))

def _shapes_are_broadcast_compatible(shapes):
  accumulator = np.zeros([])
  for shape in shapes:
    try:
      accumulator = accumulator + np.zeros(shape)
    except ValueError:
      return False
  return True

def _shapes_are_equal_length(shapes):
  return all(len(shape) == len(shapes[0]) for shape in shapes[1:])


def _promote_like_jnp(fun, inexact=False):
  """Decorator that promotes the arguments of `fun` to `jnp.result_type(*args)`.

  jnp and np have different type promotion semantics; this decorator allows
  tests make an np reference implementation act more like an jnp
  implementation.
  """
  def wrapper(*args, **kw):
    flat_args = tree_util.tree_leaves(args)
    if inexact and not any(jnp.issubdtype(jnp.result_type(x), jnp.inexact)
                           for x in flat_args):
      dtype = jnp.result_type(jnp.float_, *flat_args)
    else:
      dtype = jnp.result_type(*flat_args)
    args = tree_util.tree_map(lambda a: np.asarray(a, dtype), args)
    return fun(*args, **kw)
  return wrapper


class LaxBackedNumpyTests(jtu.JaxTestCase):
  """Tests for LAX-backed Numpy implementation."""

  def _GetArgsMaker(self, rng, shapes, dtypes, np_arrays=True):
    def f():
      out = [rng(shape, dtype or jnp.float_)
             for shape, dtype in zip(shapes, dtypes)]
      if np_arrays:
        return out
      return [jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) else a
              for a in out]
    return f

  def testNotImplemented(self):
    for name in jnp._NOT_IMPLEMENTED:
      func = getattr(jnp, name)
      with self.assertRaises(NotImplementedError):
        func()

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.test_name, shapes,
                                                      dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes,
         "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
         "check_dtypes": rec.check_dtypes, "tolerance": rec.tolerance,
         "inexact": rec.inexact}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes)))
      for rec in itertools.chain(JAX_ONE_TO_ONE_OP_RECORDS,
                                 JAX_COMPOUND_OP_RECORDS)))
  def testOp(self, np_op, jnp_op, rng_factory, shapes, dtypes, check_dtypes,
             tolerance, inexact):
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                               message="invalid value.*")(np_op)

    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, np_arrays=False)
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in dtypes)
    tol = functools.reduce(jtu.join_tolerance,
                           [tolerance, tol, jtu.default_tolerance()])
    self._CheckAgainstNumpy(_promote_like_jnp(np_op, inexact), jnp_op,
                            args_maker, check_dtypes=check_dtypes, tol=tol)
    self._CompileAndCheck(jnp_op, args_maker, check_dtypes=check_dtypes,
                          atol=tol, rtol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.test_name, shapes,
                                                      dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes, "name": rec.name,
         "tol": rec.tolerance}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes)))
      for rec in JAX_OPERATOR_OVERLOADS))
  def testOperatorOverload(self, name, rng_factory, shapes, dtypes, tol):
    rng = rng_factory(self.rng())
    # np and jnp arrays have different type promotion rules; force the use of
    # jnp arrays.
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, np_arrays=False)
    fun = lambda *xs: getattr(operator, name.strip('_'))(*xs)
    self._CompileAndCheck(fun, args_maker, atol=tol, rtol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.test_name, shapes,
                                                      dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes, "name": rec.name,
         "op_tolerance": rec.tolerance}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes)))
      for rec in JAX_RIGHT_OPERATOR_OVERLOADS))
  def testRightOperatorOverload(self, name, rng_factory, shapes, dtypes,
                                op_tolerance):
    if shapes[1] is jtu.PYTHON_SCALAR_SHAPE:
      raise SkipTest("scalars not implemented")  # TODO(mattjj): clean up
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, np_arrays=False)
    fun = lambda fst, snd: getattr(snd, name)(fst)
    tol = max(jtu.tolerance(dtype, op_tolerance) for dtype in dtypes)
    self._CompileAndCheck( fun, args_maker, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": rec.test_name + "_{}".format(dtype),
       "rng_factory": rec.rng_factory,
       "op_name": rec.name, "dtype": dtype}
      for rec in JAX_OPERATOR_OVERLOADS if rec.nargs == 2
      for dtype in rec.dtypes))
  def testBinaryOperatorDefers(self, op_name, rng_factory, dtype):
    rng = rng_factory(self.rng())
    arg = jax.device_put(rng((), dtype))
    op = getattr(operator, op_name)

    other = _OverrideEverything()
    assert op(other, arg) is other
    assert op(arg, other) is other

    other = _OverrideNothing()
    if op_name == "__eq__":
      assert op(other, arg) is False
      assert op(arg, other) is False
    elif op_name == "__ne__":
      assert op(other, arg) is True
      assert op(arg, other) is True
    else:
      with self.assertRaises(TypeError):
        op(other, arg)
      with self.assertRaises(TypeError):
        op(arg, other)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.test_name, shapes, dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes,
         "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name)}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, rec.nargs))
        for dtypes in filter(
          _dtypes_are_compatible_for_bitwise_ops,
          itertools.combinations_with_replacement(rec.dtypes, rec.nargs)))
      for rec in JAX_BITWISE_OP_RECORDS))
  def testBitwiseOp(self, np_op, jnp_op, rng_factory, shapes, dtypes):
    rng = rng_factory(self.rng())
    if not FLAGS.jax_enable_x64 and any(
        jnp.iinfo(dtype).bits == 64 for dtype in dtypes):
      self.skipTest("x64 types are disabled by jax_enable_x64")
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker,
                            check_dtypes=jtu.PYTHON_SCALAR_SHAPE not in shapes)
    self._CompileAndCheck(jnp_op, args_maker)

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
        for out_dtype in [None] + rec.dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        for keepdims in [False, True])
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
      x_cast = x if dtype != jnp.bfloat16 else x.astype(np.float32)
      t = out_dtype if out_dtype != jnp.bfloat16 else np.float32
      return np_op(x_cast, axis, dtype=t, keepdims=keepdims)
    np_fun = _promote_like_jnp(np_fun, inexact)
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "{}_inshape={}_axis={}_keepdims={}".format(
          rec.test_name.capitalize(),
          jtu.format_shape_dtype_string(shape, dtype), axis, keepdims),
       "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
       "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
       "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
      for rec in JAX_REDUCER_NO_DTYPE_RECORDS
      for shape in rec.shapes for dtype in rec.dtypes
      for axis in list(range(-len(shape), len(shape))) + [None]
      for keepdims in [False, True]))
  def testReducerNoDtype(self, np_op, jnp_op, rng_factory, shape, dtype, axis,
                         keepdims, inexact):
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16 and rng_factory.__name__ == 'rand_some_nan'
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    def np_fun(x):
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res
    np_fun = _promote_like_jnp(np_fun, inexact)
    np_fun = jtu.ignore_warning(category=np.ComplexWarning)(np_fun)
    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for shape in all_shapes for dtype in all_dtypes
      for axis in list(range(-len(shape), len(shape))) + [None]))
  def testCountNonzero(self, shape, dtype, axis):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = lambda x: np.count_nonzero(x, axis)
    jnp_fun = lambda x: jnp.count_nonzero(x, axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes for dtype in all_dtypes))
  def testNonzero(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = lambda x: np.nonzero(x)
    np_fun = jtu.ignore_warning(
      category=DeprecationWarning,
      message="Calling nonzero on 0d arrays.*")(np_fun)
    jnp_fun = lambda x: jnp.nonzero(x)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes for dtype in all_dtypes))
  def testFlatNonzero(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = jtu.ignore_warning(
      category=DeprecationWarning,
      message="Calling nonzero on 0d arrays.*")(np.flatnonzero)
    jnp_fun = jnp.flatnonzero
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes for dtype in all_dtypes))
  def testArgWhere(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = jtu.ignore_warning(
      category=DeprecationWarning,
      message="Calling nonzero on 0d arrays.*")(np.argwhere)
    jnp_fun = jnp.argwhere
    args_maker = lambda: [rng(shape, dtype)]
    if shape in (scalar_shapes + [()]) and np.__version__ < "1.18":
      self.skipTest("np.argwhere() result for scalar input changed in numpy 1.18.")
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "{}_inshape={}_axis={}".format(
          rec.test_name.capitalize(),
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
       "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
       "axis": axis}
      for rec in JAX_ARGMINMAX_RECORDS
      for shape, dtype in _shape_and_dtypes(rec.shapes, rec.dtypes)
      for axis in range(-len(shape), len(shape))))
  def testArgMinMax(self, np_op, jnp_op, rng_factory, shape, dtype, axis):
    rng = rng_factory(self.rng())
    if dtype == np.complex128 and jtu.device_under_test() == "gpu":
      raise unittest.SkipTest("complex128 reductions not supported on GPU")
    if "nan" in np_op.__name__ and dtype == jnp.bfloat16:
      raise unittest.SkipTest("NumPy doesn't correctly handle bfloat16 arrays")

    def np_fun(array_to_reduce):
      return np_op(array_to_reduce, axis).astype(jnp.int_)

    def jnp_fun(array_to_reduce):
      return jnp_op(array_to_reduce, axis)

    args_maker = lambda: [rng(shape, dtype)]
    try:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    except ValueError as e:
      if str(e) == "All-NaN slice encountered":
        self.skipTest("JAX doesn't support checking for all-NaN slices")
      else:
        raise
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": rec.test_name.capitalize(), "name": rec.name,
       "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name)}
      for rec in JAX_ARGMINMAX_RECORDS))
  def testArgMinMaxEmpty(self, name, np_op, jnp_op):
    name = name[3:] if name.startswith("nan") else name
    msg = "attempt to get {} of an empty sequence".format(name)
    with self.assertRaises(ValueError, msg=msg):
      jnp_op(np.array([]))
    with self.assertRaises(ValueError, msg=msg):
      jnp_op(np.zeros((2, 0)), axis=1)
    np_fun = partial(np_op, axis=0)
    jnp_fun = partial(jnp_op, axis=0)
    args_maker = lambda: [np.zeros((2, 0))]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype),
          axes),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "axes": axes, "rng_factory": rng_factory}
      for rng_factory in [jtu.rand_default]
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
      ]
      for lhs_dtype, rhs_dtype in itertools.combinations_with_replacement(number_dtypes, 2)))
  def testCross(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes, rng_factory):
    rng = rng_factory(self.rng())
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
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol,
                          rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          name,
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "rng_factory": rng_factory}
      for rng_factory in [jtu.rand_default]
      for name, lhs_shape, rhs_shape in [
          ("matrix-scalar", (3, 3), ()),
          ("scalar-matrix", (), (3, 3)),
          ("matrix-vector", (4, 5), (5,)),
          ("vector-matrix", (6,), (6, 4)),
          ("matrix-matrix", (3, 4), (4, 5)),
          ("tensor-vector", (4, 3, 2), (2,)),
          ("vector-tensor", (2,), (3, 2, 4)),
          ("tensor-matrix", (4, 3, 2), (2, 5)),
          ("matrix-tensor", (5, 2), (3, 2, 4)),
          ("tensor-tensor", (2, 3, 4), (5, 4, 1))]
      for lhs_dtype, rhs_dtype in itertools.combinations_with_replacement(number_dtypes, 2)))
  def testDot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    tol = {np.float16: 1e-2, np.float32: 1e-5, np.float64: 1e-14,
           np.complex128: 1e-14}
    if jtu.device_under_test() == "tpu":
      tol[np.float32] = tol[np.complex64] = 2e-1
    def np_dot(x, y):
      x = x.astype(np.float32) if lhs_dtype == jnp.bfloat16 else x
      y = y.astype(np.float32) if rhs_dtype == jnp.bfloat16 else y
      return np.dot(x, y).astype(jnp.promote_types(lhs_dtype, rhs_dtype))
    self._CheckAgainstNumpy(np_dot, jnp.dot, args_maker,
                            tol=tol)
    self._CompileAndCheck(jnp.dot, args_maker, atol=tol,
                          rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          name,
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "rng_factory": rng_factory}
      for rng_factory in [jtu.rand_default]
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
          ("tensor-tensor-broadcast", (3, 1, 3, 4), (5, 4, 1))]
      for lhs_dtype, rhs_dtype in itertools.combinations_with_replacement(number_dtypes, 2)))
  def testMatmul(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, rng_factory):
    rng = rng_factory(self.rng())
    def np_fun(x, y):
      dtype = jnp.promote_types(lhs_dtype, rhs_dtype)
      return np.matmul(x, y).astype(dtype)
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    tol = {np.float16: 1e-2, np.float32: 2e-2, np.float64: 1e-12,
           np.complex128: 1e-12}
    if jtu.device_under_test() == "tpu":
      tol[np.float32] = tol[np.complex64] = 4e-2
    self._CheckAgainstNumpy(np_fun, jnp.matmul, args_maker, tol=tol)
    self._CompileAndCheck(jnp.matmul, args_maker, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype),
          axes),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "axes": axes, "rng_factory": rng_factory}
      for rng_factory in [jtu.rand_default]
      for lhs_shape, rhs_shape, axes in [
          [(3,), (), 0],
          [(2, 3, 4), (5, 6, 7), 0],  # from issue #740
          [(2, 3, 4), (3, 4, 5, 6), 2],
          [(2, 3, 4), (5, 4, 3, 6), [1, 2]],
          [(2, 3, 4), (5, 4, 3, 6), [[1, 2], [2, 1]]],
          [(1, 2, 3, 4), (4, 5, 3, 6), [[2, 3], [2, 0]]],
      ]
      for lhs_dtype, rhs_dtype in itertools.combinations_with_replacement(number_dtypes, 2)))
  def testTensordot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    jnp_fun = lambda a, b: jnp.tensordot(a, b, axes)
    def np_fun(a, b):
      a = a if lhs_dtype != jnp.bfloat16 else a.astype(np.float32)
      b = b if rhs_dtype != jnp.bfloat16 else b.astype(np.float32)
      dtype = jnp.promote_types(lhs_dtype, rhs_dtype)
      return np.tensordot(a, b, axes).astype(dtype)
    tol = {np.float16: 1e-1, np.float32: 1e-3, np.float64: 1e-12,
           np.complex64: 1e-3, np.complex128: 1e-12}
    if jtu.device_under_test() == "tpu":
      tol[np.float32] = tol[np.complex64] = 2e-1
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testTensordotErrors(self):
    a = np.random.random((3, 2, 2))
    b = np.random.random((2,))
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


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_invert={}".format(
          jtu.format_shape_dtype_string(element_shape, dtype),
          jtu.format_shape_dtype_string(test_shape, dtype), invert),
       "element_shape": element_shape, "test_shape": test_shape,
       "dtype": dtype, "invert": invert}
      for element_shape in all_shapes
      for test_shape in all_shapes
      for dtype in default_dtypes
      for invert in [True, False]))
  def testIsin(self, element_shape, test_shape, dtype, invert):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(element_shape, dtype), rng(test_shape, dtype)]
    jnp_fun = lambda e, t: jnp.isin(e, t, invert=invert)
    np_fun = lambda e, t: np.isin(e, t, invert=invert)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_invert={}".format(
          jtu.format_shape_dtype_string(element_shape, dtype),
          jtu.format_shape_dtype_string(test_shape, dtype), invert),
       "element_shape": element_shape, "test_shape": test_shape,
       "dtype": dtype, "invert": invert}
      for element_shape in all_shapes
      for test_shape in all_shapes
      for dtype in default_dtypes
      for invert in [True, False]))
  def testIn1d(self, element_shape, test_shape, dtype, invert):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(element_shape, dtype), rng(test_shape, dtype)]
    jnp_fun = lambda e, t: jnp.in1d(e, t, invert=invert)
    np_fun = lambda e, t: np.in1d(e, t, invert=invert)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_assume_unique={}_return_indices={}".format(
       jtu.format_shape_dtype_string(shape1, dtype1),
       jtu.format_shape_dtype_string(shape2, dtype2),
       assume_unique,
       return_indices),
       "shape1": shape1, "dtype1": dtype1, "shape2": shape2, "dtype2": dtype2,
       "assume_unique": assume_unique, "return_indices": return_indices}
      for dtype1 in [s for s in default_dtypes if s != jnp.bfloat16]
      for dtype2 in [s for s in default_dtypes if s != jnp.bfloat16]
      for shape1 in all_shapes
      for shape2 in all_shapes
      for assume_unique in [False, True]
      for return_indices in [False, True]))
  def testIntersect1d(self, shape1, dtype1, shape2, dtype2, assume_unique, return_indices):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape1, dtype1), rng(shape2, dtype2)]
    jnp_fun = lambda ar1, ar2: jnp.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
    np_fun = lambda ar1, ar2: np.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "rng_factory": jtu.rand_default}
      # TODO(phawkins): support integer dtypes too.
      for lhs_shape, lhs_dtype in _shape_and_dtypes(all_shapes, inexact_dtypes)
      for rhs_shape, rhs_dtype in _shape_and_dtypes(all_shapes, inexact_dtypes)
      if len(jtu._dims_of_shape(lhs_shape)) == 0
      or len(jtu._dims_of_shape(rhs_shape)) == 0
      or lhs_shape[-1] == rhs_shape[-1]))
  def testInner(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    def np_fun(lhs, rhs):
      lhs = lhs if lhs_dtype != jnp.bfloat16 else lhs.astype(np.float32)
      rhs = rhs if rhs_dtype != jnp.bfloat16 else rhs.astype(np.float32)
      dtype = jnp.promote_types(lhs_dtype, rhs_dtype)
      return np.inner(lhs, rhs).astype(dtype)
    jnp_fun = lambda lhs, rhs: jnp.inner(lhs, rhs)
    tol_spec = {np.float16: 1e-2, np.float32: 1e-5, np.float64: 1e-13,
                np.complex64: 1e-5}
    if jtu.device_under_test() == "tpu":
      tol_spec[np.float32] = tol_spec[np.complex64] = 2e-1
    tol = max(jtu.tolerance(lhs_dtype, tol_spec),
              jtu.tolerance(rhs_dtype, tol_spec))
    # TODO(phawkins): there are float32/float64 disagreements for some inputs.
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=False, atol=tol,
                          rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_amin={}_amax={}".format(
          jtu.format_shape_dtype_string(shape, dtype), a_min, a_max),
       "shape": shape, "dtype": dtype, "a_min": a_min, "a_max": a_max,
       "rng_factory": jtu.rand_default}
      for shape in all_shapes for dtype in number_dtypes
      for a_min, a_max in [(-1, None), (None, 1), (-1, 1),
                           (-np.ones(1), None),
                           (None, np.ones(1)),
                           (-np.ones(1), np.ones(1))]))
  def testClipStaticBounds(self, shape, dtype, a_min, a_max, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda x: np.clip(x, a_min=a_min, a_max=a_max)
    jnp_fun = lambda x: jnp.clip(x, a_min=a_min, a_max=a_max)
    args_maker = lambda: [rng(shape, dtype)]
    # TODO(phawkins): the promotion behavior changed in Numpy 1.17.
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testClipError(self):
    with self.assertRaisesRegex(ValueError, "At most one of a_min and a_max.*"):
      jnp.clip(jnp.zeros((3,)))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_decimals={}".format(
          jtu.format_shape_dtype_string(shape, dtype), decimals),
       "shape": shape, "dtype": dtype, "decimals": decimals,
       "rng_factory": jtu.rand_default}
      for shape, dtype in _shape_and_dtypes(all_shapes, number_dtypes)
      for decimals in [0, 1, -2]))
  def testRoundStaticDecimals(self, shape, dtype, decimals, rng_factory):
    rng = rng_factory(self.rng())
    if jnp.issubdtype(dtype, np.integer) and decimals < 0:
      self.skipTest("Integer rounding with decimals < 0 not implemented")
    np_fun = lambda x: np.round(x, decimals=decimals)
    jnp_fun = lambda x: jnp.round(x, decimals=decimals)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {jnp.bfloat16: 5e-2, np.float16: 1e-2}
    check_dtypes = shape is not jtu.PYTHON_SCALAR_SHAPE
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=check_dtypes, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=check_dtypes,
                          atol=tol, rtol=tol)

  def testOperatorRound(self):
    self.assertAllClose(round(np.float32(7.532), 1),
                        round(jnp.float32(7.5), 1))
    self.assertAllClose(round(np.float32(1.234), 2),
                        round(jnp.float32(1.234), 2))
    self.assertAllClose(round(np.float32(1.234)),
                        round(jnp.float32(1.234)), check_dtypes=False)
    self.assertAllClose(round(np.float32(7.532), 1),
                        round(jnp.array(7.5, jnp.float32), 1))
    self.assertAllClose(round(np.float32(1.234), 2),
                        round(jnp.array(1.234, jnp.float32), 2))
    self.assertAllClose(round(np.float32(1.234)),
                        round(jnp.array(1.234, jnp.float32)),
                        check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_mode={}_rpadwidth={}_rconstantvalues={}".format(
          jtu.format_shape_dtype_string(shape, dtype), mode, pad_width_rank,
          constant_values_rank),
       "shape": shape, "dtype": dtype, "mode": mode,
       "pad_width_rank": pad_width_rank,
       "constant_values_rank": constant_values_rank,
       "rng_factory": jtu.rand_default,
       "irng_factory": partial(jtu.rand_int, high=3)}
      for mode, constant_values_rank, shapes in [
        ('constant', 0, all_shapes),
        ('constant', 1, all_shapes),
        ('constant', 2, all_shapes),
        ('symmetric', None, nonempty_shapes),
        ('reflect', None, nonempty_shapes),
        ('wrap', None, nonempty_shapes),
        ('edge', None, nonempty_shapes),
      ]
      for shape, dtype in _shape_and_dtypes(shapes, all_dtypes)
      for pad_width_rank in range(3)))
  def testPad(self, shape, dtype, mode, pad_width_rank, constant_values_rank,
              rng_factory, irng_factory):
    rng = rng_factory(self.rng())
    irng = irng_factory(self.rng())
    pad_width = irng([len(shape), 2][2 - pad_width_rank:], np.int32)
    def np_fun(x, kwargs):
      if pad_width.size == 0:
        return x
      return np.pad(x, pad_width, mode=mode, **kwargs)
    def jnp_fun(x, kwargs):
      return jnp.pad(x, pad_width, mode=mode, **kwargs)

    def args_maker():
      kwargs = {}
      if constant_values_rank:
        kwargs["constant_values"] = rng(
          [len(shape), 2][2 - constant_values_rank:], dtype)
      return rng(shape, dtype), kwargs

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=shape is not jtu.PYTHON_SCALAR_SHAPE)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape=[{}]_reps={}".format(
          jtu.format_shape_dtype_string(shape, dtype), reps),
       "shape": shape, "dtype": dtype, "reps": reps,
       "rng_factory": jtu.rand_default}
      for reps in [(), (2,), (3, 4), (2, 3, 4)]
      for shape, dtype in _shape_and_dtypes(all_shapes, default_dtypes)
      ))
  def testTile(self, shape, dtype, reps, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda arg: np.tile(arg, reps)
    jnp_fun = lambda arg: jnp.tile(arg, reps)

    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=shape is not jtu.PYTHON_SCALAR_SHAPE)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes
      for dtype in all_dtypes))
  def testExtract(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, jnp.float32), rng(shape, dtype)]
    self._CheckAgainstNumpy(np.extract, jnp.extract, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "a_shape={} , b_shape={}".format(
          jtu.format_shape_dtype_string(a_shape, dtype),
          jtu.format_shape_dtype_string(b_shape, dtype)),
       "dtype": dtype, "a_shape": a_shape, "b_shape" : b_shape}
      for dtype in default_dtypes
      for a_shape in one_dim_array_shapes
      for b_shape in one_dim_array_shapes))
  def testPolyAdd(self, a_shape, b_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1, arg2: np.polyadd(arg1, arg2)
    jnp_fun = lambda arg1, arg2: jnp.polyadd(arg1, arg2)
    args_maker = lambda: [rng(a_shape, dtype), rng(b_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "a_shape={} , b_shape={}".format(
          jtu.format_shape_dtype_string(a_shape, dtype),
          jtu.format_shape_dtype_string(b_shape, dtype)),
       "dtype": dtype, "a_shape": a_shape, "b_shape" : b_shape}
      for dtype in default_dtypes
      for a_shape in one_dim_array_shapes
      for b_shape in one_dim_array_shapes))
  def testPolySub(self, a_shape, b_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1, arg2: np.polysub(arg1, arg2)
    jnp_fun = lambda arg1, arg2: jnp.polysub(arg1, arg2)
    args_maker = lambda: [rng(a_shape, dtype), rng(b_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_order={}".format(
          jtu.format_shape_dtype_string(a_shape, dtype),
          order),
       "dtype": dtype, "a_shape": a_shape, "order" : order}
      for dtype in default_dtypes
      for a_shape in one_dim_array_shapes
      for order in range(5)))
  def testPolyDer(self, a_shape, order, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1: np.polyder(arg1, m=order)
    jnp_fun = lambda arg1: jnp.polyder(arg1, m=order)
    args_maker = lambda: [rng(a_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for shape in all_shapes
      for dtype in all_dtypes
      for axis in [None] + list(range(len(shape)))))
  def testCompress(self, shape, dtype, axis):
    rng = jtu.rand_some_zero(self.rng())
    if shape in scalar_shapes or len(shape) == 0:
      cond_shape = (0,)
    elif axis is None:
      cond_shape = (np.prod(shape),)
    else:
      cond_shape = (shape[axis],)

    args_maker = lambda: [rng(cond_shape, jnp.float32), rng(shape, dtype)]

    np_fun = partial(np.compress, axis=axis)
    jnp_fun = partial(jnp.compress, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_condition=array[{}]_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), len(condition), axis),
       "shape": shape, "dtype": dtype, "condition": condition, "axis": axis}
      for shape in [(2, 3)]
      for dtype in int_dtypes
      # condition entries beyond axis size must be zero.
      for condition in [[1], [1, 0, 0, 0, 0, 0, 0]]
      for axis in [None, 0, 1]))
  def testCompressMismatchedShapes(self, shape, dtype, condition, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [np.array(condition), rng(shape, dtype)]
    np_fun = partial(np.compress, axis=axis)
    jnp_fun = partial(jnp.compress, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for shape in array_shapes
      for dtype in all_dtypes
      for axis in [None] + list(range(len(shape)))))
  def testCompressMethod(self, shape, dtype, axis):
    rng = jtu.rand_some_zero(self.rng())
    if shape in scalar_shapes or len(shape) == 0:
      cond_shape = (0,)
    elif axis is None:
      cond_shape = (np.prod(shape),)
    else:
      cond_shape = (shape[axis],)

    args_maker = lambda: [rng(cond_shape, jnp.float32), rng(shape, dtype)]

    np_fun = lambda condition, x: np.compress(condition, x, axis=axis)
    jnp_fun = lambda condition, x: x.compress(condition, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_axis={}_baseshape=[{}]_dtypes=[{}]".format(
          axis, ",".join(str(d) for d in base_shape),
          ",".join(np.dtype(dtype).name for dtype in arg_dtypes)),
       "axis": axis, "base_shape": base_shape, "arg_dtypes": arg_dtypes,
       "rng_factory": jtu.rand_default}
      for num_arrs in [3]
      for arg_dtypes in itertools.combinations_with_replacement(default_dtypes, num_arrs)
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for axis in range(-len(base_shape)+1, len(base_shape))))
  def testConcatenate(self, axis, base_shape, arg_dtypes, rng_factory):
    rng = rng_factory(self.rng())
    wrapped_axis = axis % len(base_shape)
    shapes = [base_shape[:wrapped_axis] + (size,) + base_shape[wrapped_axis+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), arg_dtypes)]
    def np_fun(*args):
      args = [x if x.dtype != jnp.bfloat16 else x.astype(np.float32)
              for x in args]
      dtype = functools.reduce(jnp.promote_types, arg_dtypes)
      return np.concatenate(args, axis=axis).astype(dtype)
    jnp_fun = lambda *args: jnp.concatenate(args, axis=axis)

    def args_maker():
      return [rng(shape, dtype) for shape, dtype in zip(shapes, arg_dtypes)]

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testConcatenateAxisNone(self):
    # https://github.com/google/jax/issues/3419
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[5]])
    jnp.concatenate((a, b), axis=None)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_axis={}_baseshape=[{}]_dtypes=[{}]".format(
          axis, ",".join(str(d) for d in base_shape),
          ",".join(np.dtype(dtype).name for dtype in arg_dtypes)),
       "axis": axis, "base_shape": base_shape, "arg_dtypes": arg_dtypes,
       "rng_factory": jtu.rand_default}
      for arg_dtypes in itertools.combinations_with_replacement(default_dtypes, 2)
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for axis in range(-len(base_shape)+1, len(base_shape))))
  def testAppend(self, axis, base_shape, arg_dtypes, rng_factory):
    rng = rng_factory(self.rng())
    wrapped_axis = axis % len(base_shape)
    shapes = [base_shape[:wrapped_axis] + (size,) + base_shape[wrapped_axis+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), arg_dtypes)]
    def np_fun(arr, values):
      arr = arr.astype(np.float32) if arr.dtype == jnp.bfloat16 else arr
      values = (values.astype(np.float32) if values.dtype == jnp.bfloat16
                else values)
      out = np.append(arr, values, axis=axis)
      return out.astype(jnp.promote_types(*arg_dtypes))
    jnp_fun = lambda arr, values: jnp.append(arr, values, axis=axis)

    def args_maker():
      return [rng(shape, dtype) for shape, dtype in zip(shapes, arg_dtypes)]

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)


  def _compute_total_repeat_length(self, shape, axis, repeats):
    # Calculate expected size of the repeated axis.
    if jnp.ndim(shape) == 0 :
      return repeats
    shape = jnp.array(shape)
    if shape.size == 0:
      return repeats
    if axis is None:
      axis = 0
      if jnp.ndim(shape) != 0:
        shape = jnp.array([jnp.product(shape)])
    # Broadcasting the repeats if a scalar value.
    expected_repeats = jnp.broadcast_to(jnp.ravel(repeats),
                                        [shape[axis]])
    # Total size will be num_repeats X axis length.
    total_repeat_length = jnp.sum(expected_repeats)
    return total_repeat_length


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape=[{}]_axis={}_repeats={}_fixed_size={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          axis, repeats, fixed_size),
       "axis": axis, "shape": shape, "dtype": dtype, "repeats": repeats,
       "rng_factory": jtu.rand_default, 'fixed_size': fixed_size}
      for repeats in [0, 1, 2]
      for shape, dtype in _shape_and_dtypes(all_shapes, default_dtypes)
      for axis in [None] + list(range(-len(shape), max(1, len(shape))))
      for fixed_size in [True, False]))
  def testRepeat(self, axis, shape, dtype, repeats, rng_factory, fixed_size):
    rng = rng_factory(self.rng())
    np_fun = lambda arg: np.repeat(arg, repeats=repeats, axis=axis)
    np_fun = _promote_like_jnp(np_fun)
    if fixed_size:
      total_repeat_length = self._compute_total_repeat_length(
          shape, axis, repeats)
      jnp_fun = lambda arg, rep: jnp.repeat(arg, repeats=rep, axis=axis,
                                     total_repeat_length=total_repeat_length)
      jnp_args_maker = lambda: [rng(shape, dtype), repeats]
      clo_fun = lambda arg: jnp.repeat(arg, repeats=repeats, axis=axis,
                                       total_repeat_length=total_repeat_length)
      clo_fun_args_maker = lambda: [rng(shape, dtype)]
      self._CompileAndCheck(jnp_fun, jnp_args_maker)
      self._CheckAgainstNumpy(np_fun, clo_fun, clo_fun_args_maker)
    else:
      # Now repeats is in a closure, so a constant.
      jnp_fun = lambda arg: jnp.repeat(arg, repeats=repeats, axis=axis)
      args_maker = lambda: [rng(shape, dtype)]
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
      self._CompileAndCheck(jnp_fun, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_ind={}_inv={}_count={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          return_index, return_inverse, return_counts),
       "shape": shape, "dtype": dtype,
       "return_index": return_index, "return_inverse": return_inverse,
       "return_counts": return_counts}
      for dtype in default_dtypes
      for shape in all_shapes
      for return_index in [False, True]
      for return_inverse in [False, True]
      for return_counts in [False, True]))
  def testUnique(self, shape, dtype, return_index, return_inverse, return_counts):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda x: np.unique(x, return_index, return_inverse, return_counts)
    jnp_fun = lambda x: jnp.unique(x, return_index, return_inverse, return_counts)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_fixed_size={}".format(fixed_size),
      "fixed_size": fixed_size}
      for fixed_size in [True, False]))
  def testNonScalarRepeats(self, fixed_size):
    '''
    Following numpy test suite from `test_repeat` at
    https://github.com/numpy/numpy/blob/master/numpy/core/tests/test_multiarray.py
    '''
    tol = 1e-5

    def test_single(m, args_maker, repeats, axis):
      lax_ans = jnp.repeat(m, repeats, axis)
      numpy_ans = np.repeat(m, repeats, axis)

      self.assertAllClose(lax_ans, numpy_ans, rtol=tol, atol=tol)
      if fixed_size:

        # Calculate expected size of the repeated axis.
        rep_length = self._compute_total_repeat_length(m.shape, axis, repeats)
        jnp_fun = lambda arg, rep: jnp.repeat(
            arg, repeats = rep, axis=axis, total_repeat_length=rep_length)
      else:
        jnp_fun = lambda arg: jnp.repeat(arg, repeats = repeats, axis=axis)
      self._CompileAndCheck(jnp_fun, args_maker)

    m = jnp.array([1,2,3,4,5,6])
    if fixed_size:
      args_maker = lambda: [m, repeats]
    else:
      args_maker = lambda: [m]

    for repeats in [2, [1,3,2,1,1,2], [1,3,0,1,1,2], [2], jnp.array([1,3,2,1,1,2]), jnp.array([2])]:
      test_single(m, args_maker, repeats, None)

    m_rect = m.reshape((2,3))
    if fixed_size:
      args_maker = lambda: [m_rect, repeats]
    else:
      args_maker = lambda: [m_rect]

    for repeats in [2, [2,1], [2], jnp.array([2,1]), jnp.array([2])]:
      test_single(m_rect, args_maker, repeats, axis=0)

    for repeats in [2, [1,3,2], [2], jnp.array([1,3,2]), jnp.array([2])]:
      test_single(m_rect, args_maker, repeats, axis=1)

  def testIssue2330(self):
    '''
    Make sure return value of jnp.concatenate is a jax.ndarray and is side-effect save
    '''
    def attempt_sideeffect(x):
      x = [x]
      x = jnp.concatenate(x)
      x -= 1.
      return x

    np_input = np.ones((1))
    jnp_input = jnp.ones((1))
    expected_np_input_after_call = np.ones((1))
    expected_jnp_input_after_call = jnp.ones((1))

    self.assertIs(type(jnp.concatenate([np_input])), jnp.DeviceArray)

    attempt_sideeffect(np_input)
    attempt_sideeffect(jnp_input)

    self.assertAllClose(np_input, expected_np_input_after_call)
    self.assertAllClose(jnp_input, expected_jnp_input_after_call)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "op={}_xshape=[{}]_yshape=[{}]_mode={}".format(
          op,
          jtu.format_shape_dtype_string(xshape, dtype),
          jtu.format_shape_dtype_string(yshape, dtype),
          mode),
       "xshape": xshape, "yshape": yshape, "dtype": dtype, "mode": mode,
       "rng_factory": jtu.rand_default,
       "jnp_op": getattr(jnp, op),
       "np_op": getattr(np, op)}
      for mode in ['full', 'same', 'valid']
      for op in ['convolve', 'correlate']
      for dtype in default_dtypes
      for xshape in one_dim_array_shapes
      for yshape in one_dim_array_shapes))
  def testConvolutions(self, xshape, yshape, dtype, mode, rng_factory, jnp_op, np_op):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
    precision = lax.Precision.HIGHEST if jtu.device_under_test() == "tpu" else None
    np_fun = partial(np_op, mode=mode)
    jnp_fun = partial(jnp_op, mode=mode, precision=precision)
    tol = {np.float16: 2e-1, np.float32: 1e-2, np.float64: 1e-14}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "op={}_shape=[{}]_axis={}_out_dtype={}".format(
          op, jtu.format_shape_dtype_string(shape, dtype), axis,
          out_dtype.__name__),
       "axis": axis, "shape": shape, "dtype": dtype, "out_dtype": out_dtype,
       "rng_factory": jtu.rand_default, "jnp_op": getattr(jnp, op),
       "np_op": getattr(np, op)}
      for op in ["cumsum", "cumprod"]
      for dtype in all_dtypes
      for out_dtype in default_dtypes
      for shape in all_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))))
  def testCumSumProd(self, axis, shape, dtype, out_dtype, np_op, jnp_op, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda arg: np_op(arg, axis=axis, dtype=out_dtype)
    np_fun = jtu.ignore_warning(category=np.ComplexWarning)(np_fun)
    jnp_fun = lambda arg: jnp_op(arg, axis=axis, dtype=out_dtype)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)

    args_maker = lambda: [rng(shape, dtype)]

    tol_thresholds = {dtypes.bfloat16: 4e-2}
    tol = max(jtu.tolerance(dtype, tol_thresholds),
              jtu.tolerance(out_dtype, tol_thresholds))
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "op={}_shape=[{}]_axis={}_out_dtype={}".format(
          op, jtu.format_shape_dtype_string(shape, dtype), axis,
          out_dtype.__name__),
       "axis": axis, "shape": shape, "dtype": dtype, "out_dtype": out_dtype,
       "jnp_op": getattr(jnp, op), "np_op": getattr(np, op)}
      for op in ["nancumsum", "nancumprod"]
      for dtype in all_dtypes
      for out_dtype in default_dtypes
      for shape in all_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))))
  def testNanCumSumProd(self, axis, shape, dtype, out_dtype, np_op, jnp_op):
    rng = jtu.rand_some_nan(self.rng())
    np_fun = partial(np_op, axis=axis, dtype=out_dtype)
    np_fun = jtu.ignore_warning(category=np.ComplexWarning)(np_fun)
    jnp_fun = partial(jnp_op, axis=axis, dtype=out_dtype)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)

    args_maker = lambda: [rng(shape, dtype)]

    tol_thresholds = {dtypes.bfloat16: 4e-2}
    tol = max(jtu.tolerance(dtype, tol_thresholds),
              jtu.tolerance(out_dtype, tol_thresholds))
    if dtype != jnp.bfloat16:
      # numpy functions do not properly handle bfloat16
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True,
                              tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_yshape={}_xshape={}_dx={}_axis={}".format(
        jtu.format_shape_dtype_string(yshape, dtype),
        jtu.format_shape_dtype_string(xshape, dtype) if xshape is not None else None,
        dx, axis),
        "yshape": yshape, "xshape": xshape, "dtype": dtype, "dx": dx, "axis": axis}
        for dtype in default_dtypes
        for yshape, xshape, dx, axis in [
          ((10,), None, 1.0, -1),
          ((3, 10), None, 2.0, -1),
          ((3, 10), None, 3.0, -0),
          ((10, 3), (10,), 1.0, -2),
          ((3, 10), (10,), 1.0, -1),
          ((3, 10), (3, 10), 1.0, -1),
          ((2, 3, 10), (3, 10), 1.0, -2),
        ]))
  @jtu.skip_on_devices("tpu")  # TODO(jakevdp): fix and reenable this test.
  def testTrapz(self, yshape, xshape, dtype, dx, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(yshape, dtype), rng(xshape, dtype) if xshape is not None else None]
    np_fun = partial(np.trapz, dx=dx, axis=axis)
    jnp_fun = partial(jnp.trapz, dx=dx, axis=axis)
    tol = jtu.tolerance(dtype, {np.float64: 1e-12})
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol,
                            check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol, rtol=tol,
                          check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}_m={}_n={}_k={}".format(
          np.dtype(dtype).name, m, n, k),
       "m": m, "n": n, "k": k, "dtype": dtype}
      for dtype in default_dtypes
      for n in [0, 4]
      for m in [None, 0, 1, 3, 4]
      for k in list(range(-4, 4))))
  def testTri(self, m, n, k, dtype):
    np_fun = lambda: np.tri(n, M=m, k=k, dtype=dtype)
    jnp_fun = lambda: jnp.tri(n, M=m, k=k, dtype=dtype)
    args_maker = lambda: []
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_shape={}_k={}".format(
          op, jtu.format_shape_dtype_string(shape, dtype), k),
       "dtype": dtype, "shape": shape, "op": op, "k": k,
       "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for op in ["tril", "triu"]
      for k in list(range(-3, 3))))
  def testTriLU(self, dtype, shape, op, k, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda arg: getattr(np, op)(arg, k=k)
    jnp_fun = lambda arg: getattr(jnp, op)(arg, k=k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "n={}_k={}_m={}".format(n, k, m),
      "n": n, "k": k, "m": m}
    for n in range(1, 5)
    for k in [-1, 0, 1]
    for m in range(1, 5)))
  def testTrilIndices(self, n, k, m):
    np_fun = lambda n, k, m: np.tril_indices(n, k=k, m=m)
    jnp_fun = lambda n, k, m: jnp.tril_indices(n, k=k, m=m)
    args_maker = lambda: [n, k, m]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "n={}_k={}_m={}".format(n, k, m),
      "n": n, "k": k, "m": m}
    for n in range(1, 5)
    for k in [-1, 0, 1]
    for m in range(1, 5)))
  def testTriuIndices(self, n, k, m):
    np_fun = lambda n, k, m: np.triu_indices(n, k=k, m=m)
    jnp_fun = lambda n, k, m: jnp.triu_indices(n, k=k, m=m)
    args_maker = lambda: [n, k, m]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_shape={}_k={}".format(
      jtu.format_shape_dtype_string(shape, dtype), k),
      "dtype": dtype, "shape": shape, "k": k, "rng_factory": jtu.rand_default}
    for dtype in default_dtypes
    for shape in [(1,1), (1,2), (2,2), (2,3), (3,2), (3,3), (4,4)]
    for k in [-1, 0, 1]))
  def testTriuIndicesFrom(self, shape, dtype, k, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda arr, k: np.triu_indices_from(arr, k=k)
    jnp_fun = lambda arr, k: jnp.triu_indices_from(arr, k=k)
    args_maker = lambda: [rng(shape, dtype), k]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_shape={}_k={}".format(
      jtu.format_shape_dtype_string(shape, dtype), k),
      "dtype": dtype, "shape": shape, "k": k, "rng_factory": jtu.rand_default}
    for dtype in default_dtypes
    for shape in [(1,1), (1,2), (2,2), (2,3), (3,2), (3,3), (4,4)]
    for k in [-1, 0, 1]))
  def testTrilIndicesFrom(self, shape, dtype, k, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda arr, k: np.tril_indices_from(arr, k=k)
    jnp_fun = lambda arr, k: jnp.tril_indices_from(arr, k=k)
    args_maker = lambda: [rng(shape, dtype), k]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_ndim={}_n={}".format(ndim, n),
       "ndim": ndim, "n": n}
      for ndim in [0, 1, 4]
      for n in [0, 1, 7]))
  def testDiagIndices(self, ndim, n):
    np.testing.assert_equal(np.diag_indices(n, ndim),
                             jnp.diag_indices(n, ndim))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "arr_shape={}".format(
        jtu.format_shape_dtype_string(shape, dtype)
      ),
       "dtype": dtype, "shape": shape}
      for dtype in default_dtypes
      for shape in [(1,1), (2,2), (3,3), (4,4), (5,5)]))
  def testDiagIndicesFrom(self, dtype, shape):
    rng = jtu.rand_default(self.rng())
    np_fun = np.diag_indices_from
    jnp_fun = jnp.diag_indices_from
    args_maker = lambda : [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_k={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k),
       "dtype": dtype, "shape": shape, "k": k, "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) in (1, 2)]
      for k in list(range(-4, 4))))
  def testDiag(self, shape, dtype, k, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda arg: np.diag(arg, k)
    jnp_fun = lambda arg: jnp.diag(arg, k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_k={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k),
       "dtype": dtype, "shape": shape, "k": k}
      for dtype in default_dtypes
      for shape in all_shapes
      for k in range(-4, 4)))
  def testDiagFlat(self, shape, dtype, k):
    rng = jtu.rand_default(self.rng())
    # numpy has inconsistencies for scalar values
    # https://github.com/numpy/numpy/issues/16477
    # jax differs in that it treats scalars values as length-1 arrays
    np_fun = lambda arg: np.diagflat(np.atleast_1d(arg), k)
    jnp_fun = lambda arg: jnp.diagflat(arg, k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_a1_shape={}_a2_shape2={}".format(
          jtu.format_shape_dtype_string(a1_shape, dtype),
          jtu.format_shape_dtype_string(a2_shape, dtype)),
       "dtype": dtype, "a1_shape": a1_shape, "a2_shape": a2_shape}
      for dtype in default_dtypes
      for a1_shape in one_dim_array_shapes
      for a2_shape in one_dim_array_shapes))
  def testPolyMul(self, a1_shape, a2_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1, arg2: np.polymul(arg1, arg2)
    jnp_fun_np = lambda arg1, arg2: jnp.polymul(arg1, arg2, trim_leading_zeros=True)
    jnp_fun_co = lambda arg1, arg2: jnp.polymul(arg1, arg2)
    args_maker = lambda: [rng(a1_shape, dtype), rng(a2_shape, dtype)]
    tol = {np.float16: 2e-1, np.float32: 5e-2, np.float64: 1e-14}
    self._CheckAgainstNumpy(np_fun, jnp_fun_np, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun_co, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_offset={}_axis1={}_axis2={}".format(
          jtu.format_shape_dtype_string(shape, dtype), offset, axis1, axis2),
       "dtype": dtype, "shape": shape, "offset": offset, "axis1": axis1,
       "axis2": axis2, "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for axis1 in range(-len(shape), len(shape))
      for axis2 in [a for a in range(-len(shape), len(shape))
                    if a % len(shape) != axis1 % len(shape)]
      for offset in list(range(-4, 4))))
  def testDiagonal(self, shape, dtype, offset, axis1, axis2, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda arg: np.diagonal(arg, offset, axis1, axis2)
    jnp_fun = lambda arg: jnp.diagonal(arg, offset, axis1, axis2)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_n={}".format(np.dtype(dtype).name, n),
       "dtype": dtype, "n": n}
      for dtype in default_dtypes
      for n in list(range(4))))
  def testIdentity(self, n, dtype):
    np_fun = lambda: np.identity(n, dtype)
    jnp_fun = lambda: jnp.identity(n, dtype)
    args_maker = lambda: []
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_x1={}_x2={}_x1_rng={}".format(
          jtu.format_shape_dtype_string(x1_shape, x1_dtype),
          jtu.format_shape_dtype_string(x2_shape, np.int32),
          x1_rng_factory_id),
       "x1_shape": x1_shape, "x1_dtype": x1_dtype,
       "x2_shape": x2_shape, "x1_rng_factory": x1_rng_factory,
       "x2_rng_factory": x2_rng_factory}
      for x1_rng_factory_id, x1_rng_factory in
        enumerate([jtu.rand_some_inf_and_nan, jtu.rand_some_zero])
      for x2_rng_factory in [partial(jtu.rand_int, low=-1075, high=1024)]
      for x1_shape, x2_shape in filter(_shapes_are_broadcast_compatible,
                                       itertools.combinations_with_replacement(array_shapes, 2))
      for x1_dtype in default_dtypes))
  @jtu.skip_on_devices("tpu")  # TODO(b/153053081)
  def testLdexp(self, x1_shape, x1_dtype, x2_shape, x1_rng_factory, x2_rng_factory):
    # integer types are converted to float64 in numpy's implementation
    if (x1_dtype not in [jnp.bfloat16, np.float16, np.float32]
        and not FLAGS.jax_enable_x64):
      self.skipTest("Only run float64 testcase when float64 is enabled.")
    x1_rng = x1_rng_factory(self.rng())
    x2_rng = x2_rng_factory(self.rng())
    np_fun = lambda x1, x2: np.ldexp(x1, x2)
    np_fun = jtu.ignore_warning(category=RuntimeWarning,
                                 message="overflow.*")(np_fun)
    jnp_fun = lambda x1, x2: jnp.ldexp(x1, x2)
    args_maker = lambda: [x1_rng(x1_shape, x1_dtype),
                          x2_rng(x2_shape, np.int32)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_x={}_rng_factory={}".format(
          jtu.format_shape_dtype_string(shape, dtype), rng_factory_id),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for rng_factory_id, rng_factory in enumerate([
          jtu.rand_some_inf_and_nan,
          jtu.rand_some_zero,
          partial(jtu.rand_not_small, offset=1e8),
      ])
      for shape in all_shapes
      for dtype in default_dtypes))
  @jtu.skip_on_devices("tpu")
  def testFrexp(self, shape, dtype, rng_factory):
    # integer types are converted to float64 in numpy's implementation
    if (dtype not in [jnp.bfloat16, np.float16, np.float32]
        and not FLAGS.jax_enable_x64):
      self.skipTest("Only run float64 testcase when float64 is enabled.")
    rng = rng_factory(self.rng())
    np_fun = lambda x: np.frexp(x)
    jnp_fun = lambda x: jnp.frexp(x)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dtype_{}_offset={}_axis1={}_axis2={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          out_dtype, offset, axis1, axis2),
       "dtype": dtype, "out_dtype": out_dtype, "shape": shape, "offset": offset,
       "axis1": axis1, "axis2": axis2, "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for out_dtype in [None] + number_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for axis1 in range(-len(shape), len(shape))
      for axis2 in range(-len(shape), len(shape))
      if (axis1 % len(shape)) != (axis2 % len(shape))
      for offset in list(range(-4, 4))))
  def testTrace(self, shape, dtype, out_dtype, offset, axis1, axis2, rng_factory):
    rng = rng_factory(self.rng())
    def np_fun(arg):
      if out_dtype == jnp.bfloat16:
        return np.trace(arg, offset, axis1, axis2, np.float32).astype(jnp.bfloat16)
      else:
        return np.trace(arg, offset, axis1, axis2, out_dtype)
    jnp_fun = lambda arg: jnp.trace(arg, offset, axis1, axis2, out_dtype)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_a={}_v={}_side={}".format(
      jtu.format_shape_dtype_string(ashape, dtype),
      jtu.format_shape_dtype_string(vshape, dtype),
      side), "ashape": ashape, "vshape": vshape, "side": side,
     "dtype": dtype, "rng_factory": rng_factory}
    for ashape in [(20,)]
    for vshape in [(), (5,), (5, 5)]
    for side in ['left', 'right']
    for dtype in default_dtypes
    for rng_factory in [jtu.rand_default]
  ))
  def testSearchsorted(self, ashape, vshape, side, dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [jnp.sort(rng(ashape, dtype)), rng(vshape, dtype)]
    np_fun = lambda a, v: np.searchsorted(a, v, side=side)
    jnp_fun = lambda a, v: jnp.searchsorted(a, v, side=side)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_x={}_bins={}_right={}_reverse={}".format(
      jtu.format_shape_dtype_string(xshape, dtype),
      jtu.format_shape_dtype_string(binshape, dtype),
      right, reverse), "xshape": xshape, "binshape": binshape,
      "right": right, "reverse": reverse, "dtype": dtype, "rng_factory": rng_factory}
    for xshape in [(20,), (5, 4)]
    for binshape in [(1,), (5,)]
    for right in [True, False]
    for reverse in [True, False]
    for dtype in default_dtypes
    for rng_factory in [jtu.rand_default]
  ))
  def testDigitize(self, xshape, binshape, right, reverse, dtype, rng_factory):
    order = jax.ops.index[::-1] if reverse else jax.ops.index[:]
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(xshape, dtype), jnp.sort(rng(binshape, dtype))[order]]
    np_fun = lambda x, bins: np.digitize(x, bins, right=right)
    jnp_fun = lambda x, bins: jnp.digitize(x, bins, right=right)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_test_name_suffix("", [shape] * len(dtypes), dtypes)),
       "shape": shape, "dtypes": dtypes}
      for dtypes in [
        [np.float32],
        [np.float32, np.float32],
        [np.float32, np.int32, np.float32],
        [np.float32, np.int64, np.float32],
        [np.float32, np.int32, np.float64],
      ]
      for shape in [(), (2,), (3, 4), (1, 5)]))
  def testColumnStack(self, shape, dtypes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]
    np_fun = _promote_like_jnp(np.column_stack)
    jnp_fun = jnp.column_stack
    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_test_name_suffix("", [shape] * len(dtypes), dtypes), axis),
       "shape": shape, "axis": axis, "dtypes": dtypes, "rng_factory": rng_factory}
      for dtypes in [
        [np.float32],
        [np.float32, np.float32],
        [np.float32, np.int32, np.float32],
        [np.float32, np.int64, np.float32],
        [np.float32, np.int32, np.float64],
      ]
      for shape in [(), (2,), (3, 4), (1, 100)]
      for axis in range(-len(shape), len(shape) + 1)
      for rng_factory in [jtu.rand_default]))
  def testStack(self, shape, axis, dtypes, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]
    np_fun = _promote_like_jnp(partial(np.stack, axis=axis))
    jnp_fun = partial(jnp.stack, axis=axis)
    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_{}".format(
          op, jtu.format_test_name_suffix("", [shape] * len(dtypes), dtypes)),
       "shape": shape, "op": op, "dtypes": dtypes, "rng_factory": rng_factory}
      for op in ["hstack", "vstack", "dstack"]
      for dtypes in [
        [np.float32],
        [np.float32, np.float32],
        [np.float32, np.int32, np.float32],
        [np.float32, np.int64, np.float32],
        [np.float32, np.int32, np.float64],
      ]
      for shape in [(), (2,), (3, 4), (1, 100), (2, 3, 4)]
      for rng_factory in [jtu.rand_default]))
  def testHVDStack(self, shape, op, dtypes, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]
    np_fun = _promote_like_jnp(getattr(np, op))
    jnp_fun = getattr(jnp, op)
    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outdtype={}".format(
          jtu.format_shape_dtype_string(shape, fill_value_dtype),
          np.dtype(out_dtype).name if out_dtype else "None"),
       "shape": shape, "fill_value_dtype": fill_value_dtype,
       "out_dtype": out_dtype, "rng_factory": jtu.rand_default}
      for shape in array_shapes + [3, np.array(7, dtype=np.int32)]
      for fill_value_dtype in default_dtypes
      for out_dtype in [None] + default_dtypes))
  def testFull(self, shape, fill_value_dtype, out_dtype, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda fill_value: np.full(shape, fill_value, dtype=out_dtype)
    jnp_fun = lambda fill_value: jnp.full(shape, fill_value, dtype=out_dtype)
    args_maker = lambda: [rng((), fill_value_dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(
    jtu.cases_from_list(
      {"testcase_name": ("_op={}_shape={}_dtype={}").format(op, shape, dtype),
       "np_op": getattr(np, op), "jnp_op": getattr(jnp, op),
       "shape": shape, "dtype": dtype}
      for op in ["zeros", "ones"]
      for shape in [2, (), (2,), (3, 0), np.array((4, 5, 6), dtype=np.int32),
                    np.array(4, dtype=np.int32)]
      for dtype in all_dtypes))
  def testZerosOnes(self, np_op, jnp_op, shape, dtype):
    args_maker = lambda: []
    np_op = partial(np_op, shape, dtype)
    jnp_op = partial(jnp_op, shape, dtype)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  def testOnesWithInvalidShape(self):
    with self.assertRaises(TypeError):
      jnp.ones((-1, 1))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_filldtype={}_outdtype={}".format(
          jtu.format_shape_dtype_string(shape, in_dtype),
          np.dtype(fill_value_dtype).name,
          np.dtype(out_dtype).name),
       "shape": shape, "in_dtype": in_dtype,
       "fill_value_dtype": fill_value_dtype, "out_dtype": out_dtype,
       "rng_factory": jtu.rand_default}
      for shape in array_shapes
      for in_dtype in default_dtypes
      for fill_value_dtype in default_dtypes
      for out_dtype in default_dtypes))
  def testFullLike(self, shape, in_dtype, fill_value_dtype, out_dtype, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda x, fill_value: np.full_like(x, fill_value, dtype=out_dtype)
    jnp_fun = lambda x, fill_value: jnp.full_like(x, fill_value, dtype=out_dtype)
    args_maker = lambda: [rng(shape, in_dtype), rng((), fill_value_dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_{}sections".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, num_sections),
       "shape": shape, "num_sections": num_sections, "axis": axis,
       "dtype": dtype, "rng_factory": jtu.rand_default}
      for shape, axis, num_sections in [
          ((3,), 0, 3), ((12,), 0, 3), ((12, 4), 0, 4), ((12, 4), 1, 2),
          ((2, 3, 4), -1, 2), ((2, 3, 4), -2, 3)]
      for dtype in default_dtypes))
  def testSplitStaticInt(self, shape, num_sections, axis, dtype, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda x: np.split(x, num_sections, axis=axis)
    jnp_fun = lambda x: jnp.split(x, num_sections, axis=axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testSplitTypeError(self):
    # If we pass an ndarray for indices_or_sections -> no error
    self.assertEqual(3, len(jnp.split(jnp.zeros(3), jnp.array([1, 2]))))

    CONCRETIZATION_MSG = "Abstract tracer value encountered where concrete value is expected."
    with self.assertRaisesRegex(TypeError, CONCRETIZATION_MSG):
      # An abstract tracer for idx
      api.jit(lambda idx: jnp.split(jnp.zeros((12, 2)), idx))(2.)
    with self.assertRaisesRegex(TypeError, CONCRETIZATION_MSG):
      # A list including an abstract tracer
      api.jit(lambda idx: jnp.split(jnp.zeros((12, 2)), [2, idx]))(2.)

    # A concrete tracer -> no error
    api.jvp(lambda idx: jnp.split(jnp.zeros((12, 2)), idx),
            (2.,), (1.,))
    # A tuple including a concrete tracer -> no error
    api.jvp(lambda idx: jnp.split(jnp.zeros((12, 2)), (1, idx)),
            (2,), (1,))

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_bins={}_range={}_weights={}".format(
      jtu.format_shape_dtype_string(shape, dtype), bins, range, weights),
      "shape": shape,
      "dtype": dtype,
      "bins": bins,
      "range": range,
      "weights": weights,
    }
    for shape in [(5,), (5, 5)]
    for dtype in number_dtypes
    for bins in [10, np.arange(-5, 6), [-5, 0, 3]]
    for range in [None, (0, 10)]
    for weights in [True, False]
  ))
  def testHistogramBinEdges(self, shape, dtype, bins, range, weights):
    rng = jtu.rand_default(self.rng())
    _weights = lambda w: abs(w) if weights else None
    np_fun = lambda a, w: np.histogram_bin_edges(a, bins=bins, range=range,
                                                   weights=_weights(w))
    jnp_fun = lambda a, w: jnp.histogram_bin_edges(a, bins=bins, range=range,
                                                   weights=_weights(w))
    args_maker = lambda: [rng(shape, dtype), rng(shape, dtype)]
    tol = {jnp.bfloat16: 2E-2, np.float16: 1E-2}
    # linspace() compares poorly to numpy when using bfloat16
    if dtype != jnp.bfloat16:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker,
                          atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_bins={}_density={}_weights={}".format(
      jtu.format_shape_dtype_string(shape, dtype), bins, density, weights),
      "shape": shape,
      "dtype": dtype,
      "bins": bins,
      "density": density,
      "weights": weights,
    }
    for shape in [(5,), (5, 5)]
    for dtype in default_dtypes
    # We only test explicit integer-valued bin edges beause in other cases
    # rounding errors lead to flaky tests.
    for bins in [np.arange(-5, 6), [-5, 0, 3]]
    for density in [True, False]
    for weights in [True, False]
  ))
  def testHistogram(self, shape, dtype, bins, density, weights):
    rng = jtu.rand_default(self.rng())
    _weights = lambda w: abs(w) if weights else None
    np_fun = lambda a, w: np.histogram(a, bins=bins, density=density,
                                         weights=_weights(w))
    jnp_fun = lambda a, w: jnp.histogram(a, bins=bins, density=density,
                                         weights=_weights(w))
    args_maker = lambda: [rng(shape, dtype), rng(shape, dtype)]
    tol = {jnp.bfloat16: 2E-2, np.float16: 1E-1}
    # np.searchsorted errors on bfloat16 with
    # "TypeError: invalid type promotion with custom data type"
    if dtype != jnp.bfloat16:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                              tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_{}sections".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, num_sections),
       "shape": shape, "num_sections": num_sections, "axis": axis,
       "dtype": dtype, "rng_factory": jtu.rand_default}
      for shape, axis, num_sections in [
          ((12, 4), 0, 4), ((12, 4), 1, 2),
          ((2, 3, 4), 2, 2), ((4, 3, 4), 0, 2)]
      for dtype in default_dtypes))
  def testHVDSplit(self, shape, num_sections, axis, dtype, rng_factory):
    rng = rng_factory(self.rng())
    def fn(module, axis):
      if axis == 0:
        return module.vsplit
      elif axis == 1:
        return module.hsplit
      else:
        assert axis == 2
        return module.dsplit

    np_fun = lambda x: fn(np, axis)(x, num_sections)
    jnp_fun = lambda x: fn(jnp, axis)(x, num_sections)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}_order={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype),
          order),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "order": order, "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for order in ["C", "F"]
      for arg_shape, out_shape in [
          (jtu.NUMPY_SCALAR_SHAPE, (1, 1, 1)),
          ((), (1, 1, 1)),
          ((7, 0), (0, 42, 101)),
          ((3, 4), 12),
          ((3, 4), (12,)),
          ((3, 4), -1),
          ((2, 1, 4), (-1,)),
          ((2, 2, 4), (2, 8))
      ]))
  def testReshape(self, arg_shape, out_shape, dtype, order, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda x: np.reshape(x, out_shape, order=order)
    jnp_fun = lambda x: jnp.reshape(x, out_shape, order=order)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype)),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for arg_shape, out_shape in [
          ((7, 0), (0, 42, 101)),
          ((2, 1, 4), (-1,)),
          ((2, 2, 4), (2, 8))
      ]))
  def testReshapeMethod(self, arg_shape, out_shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda x: np.reshape(x, out_shape)
    jnp_fun = lambda x: x.reshape(*out_shape)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_expanddim={!r}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), dim),
       "arg_shape": arg_shape, "dtype": dtype, "dim": dim,
       "rng_factory": jtu.rand_default}
      for arg_shape in [(), (3,), (3, 4)]
      for dtype in default_dtypes
      for dim in (list(range(-len(arg_shape)+1, len(arg_shape)))
                  + [np.array(0), np.array(-1), (0,), (np.array(0),),
                     (len(arg_shape), len(arg_shape) + 1)])))
  def testExpandDimsStaticDim(self, arg_shape, dtype, dim, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda x: np.expand_dims(x, dim)
    jnp_fun = lambda x: jnp.expand_dims(x, dim)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CompileAndCheck(jnp_fun, args_maker)

    if isinstance(dim, tuple) and numpy_version < (1, 18, 0):
      raise SkipTest("support for multiple axes added in NumPy 1.18.0")
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_axes=({},{})".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), ax1, ax2),
       "arg_shape": arg_shape, "dtype": dtype, "ax1": ax1, "ax2": ax2,
       "rng_factory": jtu.rand_default}
      for arg_shape, ax1, ax2 in [
          ((3, 4), 0, 1), ((3, 4), 1, 0), ((3, 4, 5), 1, 2),
          ((3, 4, 5), -1, -2), ((3, 4, 5), 0, 1)]
      for dtype in default_dtypes))
  def testSwapAxesStaticAxes(self, arg_shape, dtype, ax1, ax2, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda x: np.swapaxes(x, ax1, ax2)
    jnp_fun = lambda x: jnp.swapaxes(x, ax1, ax2)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_axis={!r}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), ax),
       "arg_shape": arg_shape, "dtype": dtype, "ax": ax,
       "rng_factory": jtu.rand_default}
      for arg_shape, ax in [
          ((3, 1), None),
          ((3, 1), 1),
          ((3, 1), -1),
          ((3, 1), np.array(1)),
          ((1, 3, 1), (0, 2)),
          ((1, 3, 1), (0,)),
          ((1, 4, 1), (np.array(0),))]
      for dtype in default_dtypes))
  def testSqueeze(self, arg_shape, dtype, ax, rng_factory):
    rng = rng_factory(self.rng())
    np_fun = lambda x: np.squeeze(x, ax)
    jnp_fun = lambda x: jnp.squeeze(x, ax)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}_weights={}_returned={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          axis,
          (None if weights_shape is None else jtu.format_shape_dtype_string(weights_shape, dtype)),
          returned),
       "rng_factory": jtu.rand_default, "shape": shape, "dtype": dtype, "axis": axis,
       "weights_shape": weights_shape, "returned": returned}
      for shape, dtype in _shape_and_dtypes(nonempty_shapes, number_dtypes)
      for axis in list(range(-len(shape), len(shape))) + [None]
      # `weights_shape` is either `None`, same as the averaged axis, or same as
      # that of the input
      for weights_shape in ([None, shape] if axis is None or len(shape) == 1
                            else [None, (shape[axis],), shape])
      for returned in [False, True]))
  def testAverage(self, shape, dtype, axis, weights_shape, returned, rng_factory):
    rng = rng_factory(self.rng())
    if weights_shape is None:
      np_fun = lambda x: np.average(x, axis, returned=returned)
      jnp_fun = lambda x: jnp.average(x, axis, returned=returned)
      args_maker = lambda: [rng(shape, dtype)]
    else:
      np_fun = lambda x, weights: np.average(x, axis, weights, returned)
      jnp_fun = lambda x, weights: jnp.average(x, axis, weights, returned)
      args_maker = lambda: [rng(shape, dtype), rng(weights_shape, dtype)]
    np_fun = _promote_like_jnp(np_fun, inexact=True)
    tol = {dtypes.bfloat16: 2e-1, np.float16: 1e-2, np.float32: 1e-6,
           np.float64: 1e-12, np.complex64: 1e-5}
    check_dtypes = shape is not jtu.PYTHON_SCALAR_SHAPE
    try:
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                                check_dtypes=check_dtypes, tol=tol)
    except ZeroDivisionError:
        self.skipTest("don't support checking for ZeroDivisionError")
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=check_dtypes,
                          rtol=tol, atol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       f"_arg{i}_ndmin={ndmin}_dtype={np.dtype(dtype) if dtype else None}",
       "arg": arg, "ndmin": ndmin, "dtype": dtype}
      for i, (arg, dtypes) in enumerate([
          ([True, False, True], all_dtypes),
          (3., all_dtypes),
          ([1, 2, 3], all_dtypes),
          (np.array([1, 2, 3], dtype=np.int64), all_dtypes),
          ([1., 2., 3.], all_dtypes),
          ([[1, 2], [3, 4], [5, 6]], all_dtypes),
          ([[1, 2.], [3, 4], [5, 6]], all_dtypes),
          ([[1., 2j], [3., 4.], [5., 6.]], complex_dtypes),
          ([[3, np.array(2, dtype=jnp.float_), 1],
           np.arange(3., dtype=jnp.float_)], all_dtypes),
      ])
      for dtype in [None] + dtypes
      for ndmin in [None, np.ndim(arg), np.ndim(arg) + 1, np.ndim(arg) + 2]))
  def testArray(self, arg, ndmin, dtype):
    args_maker = lambda: [arg]
    canonical_dtype = dtypes.canonicalize_dtype(dtype or np.array(arg).dtype)
    if ndmin is not None:
      np_fun = partial(np.array, ndmin=ndmin, dtype=canonical_dtype)
      jnp_fun = partial(jnp.array, ndmin=ndmin, dtype=dtype)
    else:
      np_fun = partial(np.array, dtype=canonical_dtype)
      jnp_fun = partial(jnp.array, dtype=dtype)

    # We are testing correct canonicalization behavior here, so we turn off the
    # permissive canonicalization logic in the test harness.
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            canonicalize_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testArrayUnsupportedDtypeError(self):
    with self.assertRaisesRegex(TypeError,
                                "JAX only supports number and bool dtypes.*"):
      jnp.array(3, [('a','<i4'),('b','<i4')])

  def testIssue121(self):
    assert not np.isscalar(jnp.array(3))

  def testArrayOutputsDeviceArrays(self):
    assert type(jnp.array([])) == jax.interpreters.xla.DeviceArray
    assert type(jnp.array(np.array([]))) == jax.interpreters.xla.DeviceArray

    class NDArrayLike:
        def __array__(self, dtype=None):
            return np.array([], dtype=dtype)
    assert type(jnp.array(NDArrayLike())) == jax.interpreters.xla.DeviceArray

    # NOTE(mattjj): disabled b/c __array__ must produce ndarrays
    # class DeviceArrayLike:
    #     def __array__(self, dtype=None):
    #         return jnp.array([], dtype=dtype)
    # assert type(jnp.array(DeviceArrayLike())) == jax.interpreters.xla.DeviceArray

  def testArrayMethod(self):
    class arraylike(object):
      dtype = np.float32
      def __array__(self, dtype=None):
        return np.array(3., dtype=dtype)
    a = arraylike()
    ans = jnp.array(a)
    assert ans == 3.

  def testMemoryView(self):
    ans = jnp.array(bytearray(b'\x2a'))
    self.assertAllClose(
        ans,
        np.array([0x2a], dtype=np.uint8))

  def testIsClose(self):
    c_isclose = api.jit(jnp.isclose)
    c_isclose_nan = api.jit(partial(jnp.isclose, equal_nan=True))
    n = 2

    rng = np.random.RandomState(0)
    x = rng.randn(n, 1)
    y = rng.randn(n, 1)
    inf = np.asarray(n * [np.inf]).reshape([n, 1])
    nan = np.asarray(n * [np.nan]).reshape([n, 1])
    args = [x, y, inf, -inf, nan]

    for arg0 in args:
      for arg1 in args:
        result_np = np.isclose(arg0, arg1)
        result_jax = jnp.isclose(arg0, arg1)
        result_jit = c_isclose(arg0, arg1)
        self.assertTrue(jnp.all(jnp.equal(result_np, result_jax)))
        self.assertTrue(jnp.all(jnp.equal(result_np, result_jit)))
        result_np = np.isclose(arg0, arg1, equal_nan=True)
        result_jax = jnp.isclose(arg0, arg1, equal_nan=True)
        result_jit = c_isclose_nan(arg0, arg1)
        self.assertTrue(jnp.all(jnp.equal(result_np, result_jax)))
        self.assertTrue(jnp.all(jnp.equal(result_np, result_jit)))

  def testAllClose(self):
    rng = np.random.RandomState(0)
    x = rng.randn(2, 2)
    y = rng.randn(2)

    def same(list1, list2):
      allclose = functools.partial(jnp.allclose, atol=1e-3, rtol=1e-3)
      elements_close = list(map(allclose, list1, list2))
      return jnp.all(jnp.array(elements_close))

    csame = api.jit(same)

    a1 = same((x, y), (x, y))
    a2 = csame((x, y), (x, y))
    a3 = csame((x, y), (x, 2 * y))

    self.assertTrue(a1)
    self.assertTrue(a2)
    self.assertFalse(a3)

  @jtu.skip_on_devices("tpu")  # TODO(mattjj): investigate this failure
  def testOnesBroadcastingConstantHandler(self):
    # TODO(mattjj): update this test for jax3
    self.skipTest("test needs jax3 update")

    def fun(x):
      ones = jnp.ones((3, 4))
      assert isinstance(ones, np.ndarray) and ones.strides == (0, 0)

      # To check that the constant handler generates a Broadcast for stride-zero
      # arrays, we monkey-patch the client instance.
      # TODO(mattjj): once we have better HLO dumping and inspecting facilities,
      # we can check the HLO more directly.
      c = x._node.c
      Broadcast = c.Broadcast  # pylint: disable=invalid-name
      was_called = []
      c.Broadcast = lambda *args: was_called.append(True) or Broadcast(*args)
      out = x + ones  # the ndarray constant handler should call Broadcast here
      assert was_called, "Broadcast was not called."

      return out

    fun = api.jit(fun)
    out_val = fun(jnp.ones(4))
    self.assertAllClose(out_val, np.full((3, 4), 2.), check_dtypes=False)

  def testZeroStridesConstantHandler(self):
    raw_const = np.random.RandomState(0).randn(1, 2, 1, 1, 5, 1)
    const = np.broadcast_to(raw_const, (3, 2, 3, 4, 5, 6))

    def fun(x):
      return x * const

    fun = api.jit(fun)
    out_val = fun(3.)
    self.assertAllClose(out_val, 3. * const, check_dtypes=False)

  def testIsInstanceNdarrayDuringTracing(self):
    arr = np.ones(3)

    @api.jit
    def f(x):
      self.assertIsInstance(x, jnp.ndarray)
      return jnp.sum(x)

    f(arr)

  def testNonArrayErrorMessage(self):
    x = [1., 2.]
    y = np.array([3., 4.])

    def g(x, y):
      return jnp.add(x, y)

    def f(x, y):
      return jnp.dot(x, y)

    self.assertRaises(TypeError, lambda: g(x, y))
    self.assertRaises(TypeError, lambda: f(x, y))
    self.assertRaises(TypeError, lambda: api.jit(g)(x, y))
    self.assertRaises(TypeError, lambda: api.jit(f)(x, y))

  def testAbstractionErrorMessage(self):

    @api.jit
    def f(x, n):
      for _ in range(n):
        x = x * x
      return x

    self.assertRaises(TypeError, lambda: f(3., 3))

    @api.jit
    def g(x):
      if x > 0.:
        return x * 2
      else:
        return x + 2

    self.assertRaises(TypeError, lambda: g(3.))

  def testTracingPrimitiveWithNoTranslationErrorMessage(self):
    # TODO(mattjj): update this for jax3
    self.skipTest("test needs jax3 update")
    foo = jnp._not_implemented(lambda x: x)

    # No error if there's no tracing.
    foo(np.arange(3))

    cfoo = api.jit(foo)
    self.assertRaises(NotImplementedError, lambda: cfoo(np.arange(3)))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "axis": axis}
      for shape in [(3,), (2, 3)]
      for dtype in default_dtypes
      for axis in list(range(-len(shape), len(shape))) + [None]  # Test negative axes
      for rng_factory in [jtu.rand_default]))
  def testFlip(self, shape, dtype, axis, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_op = lambda x: jnp.flip(x, axis)
    np_op = lambda x: np.flip(x, axis)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype}
      for shape in [(3,), (2, 3), (3, 2, 4)]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testFlipud(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_op = lambda x: jnp.flipud(x)
    np_op = lambda x: np.flipud(x)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype}
      for shape in [(3, 2), (2, 3), (3, 2, 4)]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testFliplr(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_op = lambda x: jnp.fliplr(x)
    np_op = lambda x: np.fliplr(x)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_k={}_axes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k, axes),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "k": k, "axes": axes}
      for shape, axes in [
          [(2, 3), (0, 1)],
          [(2, 3), (1, 0)],
          [(4, 3, 2), (0, 2)],
          [(4, 3, 2), (2, 1)],
      ]
      for k in range(-3, 4)
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testRot90(self, shape, dtype, k, axes, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_op = lambda x: jnp.rot90(x, k, axes)
    np_op = lambda x: np.rot90(x, k, axes)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  # TODO(mattjj): test infix operator overrides

  def testRavel(self):
    rng = np.random.RandomState(0)
    args_maker = lambda: [rng.randn(3, 4).astype("float32")]
    self._CompileAndCheck(lambda x: x.ravel(), args_maker)

  @parameterized.parameters(
    (0, (2, 1, 3)),
    (5, (2, 1, 3)),
    (0, ()),
    ([0, 1, 2], (2, 2)),
    ([[[0, 1], [2, 3]]], (2, 2)))
  def testUnravelIndex(self, flat_index, shape):
    args_maker = lambda: (flat_index, shape)
    self._CheckAgainstNumpy(np.unravel_index, jnp.unravel_index,
                            args_maker)
    self._CompileAndCheck(jnp.unravel_index, args_maker)

  def testUnravelIndexOOB(self):
    self.assertEqual(jnp.unravel_index(2, (2,)), (1,))
    self.assertEqual(jnp.unravel_index(-2, (2, 1, 3,)), (1, 0, 1))
    self.assertEqual(jnp.unravel_index(-3, (2,)), (0,))

  def testAstype(self):
    rng = np.random.RandomState(0)
    args_maker = lambda: [rng.randn(3, 4).astype("float32")]
    np_op = lambda x: np.asarray(x).astype(jnp.int32)
    jnp_op = lambda x: jnp.asarray(x).astype(jnp.int32)
    self._CheckAgainstNumpy(jnp_op, np_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_dtype={}".format(
          jtu.format_shape_dtype_string(shape, a_dtype), dtype),
      "shape": shape, "a_dtype": a_dtype, "dtype": dtype}
      for shape in [(8,), (3, 8)]  # last dim = 8 to ensure shape compatibility
      for a_dtype in (default_dtypes + unsigned_dtypes + bool_dtypes)
      for dtype in (default_dtypes + unsigned_dtypes + bool_dtypes)))
  def testView(self, shape, a_dtype, dtype):
    if jtu.device_under_test() == 'tpu':
      if jnp.dtype(a_dtype).itemsize in [1, 2] or jnp.dtype(dtype).itemsize in [1, 2]:
        self.skipTest("arr.view() not supported on TPU for 8- or 16-bit types.")
    if not FLAGS.jax_enable_x64:
      if jnp.dtype(a_dtype).itemsize == 8 or jnp.dtype(dtype).itemsize == 8:
        self.skipTest("x64 types are disabled by jax_enable_x64")
    rng = jtu.rand_fullrange(self.rng())
    args_maker = lambda: [rng(shape, a_dtype)]
    np_op = lambda x: np.asarray(x).view(dtype)
    jnp_op = lambda x: jnp.asarray(x).view(dtype)
    # Above may produce signaling nans; ignore warnings from invalid values.
    with np.errstate(invalid='ignore'):
      self._CheckAgainstNumpy(jnp_op, np_op, args_maker)
      self._CompileAndCheck(jnp_op, args_maker)

  def testPathologicalFloats(self):
    args_maker = lambda: [np.array([
      0b_0111_1111_1000_0000_0000_0000_0000_0000, # inf
      0b_1111_1111_1000_0000_0000_0000_0000_0000, # -inf
      0b_0111_1111_1100_0000_0000_0000_0000_0000, # qnan
      0b_1111_1111_1100_0000_0000_0000_0000_0000, # -qnan
      0b_0111_1111_1000_0000_0000_0000_0000_0001, # snan
      0b_1111_1111_1000_0000_0000_0000_0000_0001, # -snan
      0b_0111_1111_1000_0000_0000_1100_0000_0000, # nonstandard nan
      0b_1111_1111_1000_0000_0000_1100_0000_0000, # -nonstandard nan
      0b_0000_0000_0000_0000_0000_0000_0000_0000, # zero
      0b_1000_0000_0000_0000_0000_0000_0000_0000, # -zero
    ], dtype='uint32')]

    np_op = lambda x: np.asarray(x).view('float32').view('uint32')
    jnp_op = lambda x: jnp.asarray(x).view('float32').view('uint32')

    self._CheckAgainstNumpy(jnp_op, np_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  # TODO(mattjj): test other ndarray-like method overrides

  def testNpMean(self):
    # from https://github.com/google/jax/issues/125
    x = lax.add(jnp.eye(3, dtype=jnp.float_), 0.)
    ans = np.mean(x)
    self.assertAllClose(ans, np.array(1./3), check_dtypes=False)

  def testArangeOnFloats(self):
    # from https://github.com/google/jax/issues/145
    self.assertAllClose(np.arange(0.0, 1.0, 0.1, dtype=jnp.float_),
                        jnp.arange(0.0, 1.0, 0.1))
    # from https://github.com/google/jax/issues/3450
    self.assertAllClose(np.arange(2.5, dtype=jnp.float_),
                        jnp.arange(2.5))

  def testSortManually(self):
    # manual tests for sort are nice because we don't have to worry about ties.
    # lax.sort is tested combinatorially.
    ans = jnp.sort(np.array([16, 15, 23, 42, 8, 4]))
    expected = np.array([4, 8, 15, 16, 23, 42])
    self.assertAllClose(expected, ans)

    a = np.array([[1, 4], [3, 1]])
    ans = jnp.sort(a, axis=None)
    expected = np.array([1, 1, 3, 4])
    self.assertAllClose(expected, ans)

    a = np.array([[1, 4], [3, 1]])
    ans = jnp.sort(a)  # last axis
    expected = np.array([[1, 4], [1, 3]])
    self.assertAllClose(expected, ans)

    a = np.array([[1, 4], [3, 1]])
    ans = jnp.sort(a, axis=0)
    expected = np.array([[1, 1], [3, 4]])
    self.assertAllClose(expected, ans)

  def testArgsortManually(self):
    x = np.array([16, 15, 23, 42, 8, 4])
    ans = jnp.argsort(x)
    expected = np.argsort(x)
    self.assertAllClose(expected, ans, check_dtypes=False)

    x = np.array([[16, 15, 23], [42, 8, 4]])
    ans = jnp.argsort(x, axis=0)
    expected = np.argsort(x, axis=0)
    self.assertAllClose(expected, ans, check_dtypes=False)

    x = np.array([[16, 15, 23], [42, 8, 4]])
    ans = jnp.argsort(x, axis=1)
    expected = np.argsort(x, axis=1)
    self.assertAllClose(expected, ans, check_dtypes=False)

    x = np.array([[16, 15, 23], [42, 8, 4]])
    ans = jnp.argsort(x, axis=None)
    expected = np.argsort(x, axis=None)
    self.assertAllClose(expected, ans, check_dtypes=False)

    x = np.array([[16, 15, 23], [42, 8, 4]])
    ans = jnp.argsort(x)
    expected = np.argsort(x)
    self.assertAllClose(expected, ans, check_dtypes=False)

  def testMsortManually(self):
    args_maker = lambda: [np.random.randint(50, size=(5 ,5))]
    jnp_op = lambda x: jnp.msort(x)
    np_op = lambda x: np.msort(x)
    self._CheckAgainstNumpy(jnp_op, np_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_shifts={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          shifts, axis),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "shifts": shifts,
       "axis": axis}
      for dtype in all_dtypes
      for shape in [(3, 4), (3, 4, 5), (7, 4, 0)]
      for shifts, axis in [
        (3, None),
        (1, 1),
        ((3,), (0,)),
        ((-2,), (-2,)),
        ((1, 2), (0, -1)),
        ((4, 2, 5, 5, 2, 4), None),
        (100, None),
      ]
      for rng_factory in [jtu.rand_default]))
  def testRoll(self, shape, dtype, shifts, axis, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype), np.array(shifts)]
    jnp_op = partial(jnp.roll, axis=axis)
    np_op = partial(np.roll, axis=axis)
    self._CheckAgainstNumpy(jnp_op, np_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_start={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          axis, start),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "axis": axis,
       "start": start}
      for dtype in all_dtypes
      for shape in [(1, 2, 3, 4)]
      for axis in [-3, 0, 2, 3]
      for start in [-4, -1, 2, 4]
      for rng_factory in [jtu.rand_default]))
  def testRollaxis(self, shape, dtype, start, axis, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_op = partial(jnp.rollaxis, axis=axis, start=start)
    np_op = partial(np.rollaxis, axis=axis, start=start)
    self._CheckAgainstNumpy(jnp_op, np_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_bitorder={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, bitorder),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "axis": axis,
       "bitorder": bitorder}
      for dtype in [np.uint8, np.bool_]
      for bitorder in ['big', 'little']
      for shape in [(1, 2, 3, 4)]
      for axis in [None, 0, 1, -2, -1]
      for rng_factory in [jtu.rand_some_zero]))
  def testPackbits(self, shape, dtype, axis, bitorder, rng_factory):
    if numpy_version < (1, 17, 0):
      raise SkipTest("bitorder arg added in numpy 1.17.0")
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_op = partial(jnp.packbits, axis=axis, bitorder=bitorder)
    np_op = partial(np.packbits, axis=axis, bitorder=bitorder)
    self._CheckAgainstNumpy(jnp_op, np_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_bitorder={}_count={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, bitorder, count),
      "shape": shape, "dtype": dtype, "axis": axis, "bitorder": bitorder,
      "count": count}
      for dtype in [np.uint8]
      for bitorder in ['big', 'little']
      for shape in [(1, 2, 3, 4)]
      for axis in [None, 0, 1, -2, -1]
      for count in [None, 20]))
  def testUnpackbits(self, shape, dtype, axis, bitorder, count):
    if numpy_version < (1, 17, 0):
      raise SkipTest("bitorder arg added in numpy 1.17.0")
    rng = jtu.rand_int(self.rng(), 0, 256)
    args_maker = lambda: [rng(shape, dtype)]
    jnp_op = partial(jnp.unpackbits, axis=axis, bitorder=bitorder)
    np_op = partial(np.unpackbits, axis=axis, bitorder=bitorder)
    self._CheckAgainstNumpy(jnp_op, np_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_index={}_axis={}_mode={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          jtu.format_shape_dtype_string(index_shape, index_dtype),
          axis, mode),
       "shape": shape, "index_shape": index_shape, "dtype": dtype,
       "index_dtype": index_dtype, "axis": axis, "mode": mode}
      for shape in [(3,), (3, 4), (3, 4, 5)]
      for index_shape in scalar_shapes + [(3,), (2, 1, 3)]
      for axis in itertools.chain(range(-len(shape), len(shape)),
                                  [cast(Optional[int], None)])
      for dtype in all_dtypes
      for index_dtype in int_dtypes
      for mode in ['wrap', 'clip']))
  def testTake(self, shape, dtype, index_shape, index_dtype, axis, mode):
    def args_maker():
      x = rng(shape, dtype)
      i = rng_indices(index_shape, index_dtype)
      return x, i

    rng = jtu.rand_default(self.rng())
    rng_indices = jtu.rand_int(self.rng(), -5, 5)
    jnp_op = lambda x, i: jnp.take(x, i, axis=axis, mode=mode)
    np_op = lambda x, i: np.take(x, i, axis=axis, mode=mode)
    self._CheckAgainstNumpy(jnp_op, np_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_ishape={}_axis={}".format(
          jtu.format_shape_dtype_string(x_shape, dtype), i_shape, axis),
       "rng_factory": rng_factory, "x_shape": x_shape, "i_shape": i_shape, "dtype": dtype,
       "axis": axis}
      for x_shape, i_shape in filter(
        _shapes_are_equal_length,
        filter(_shapes_are_broadcast_compatible,
               itertools.combinations_with_replacement(nonempty_nonscalar_array_shapes, 2)))
      for axis in itertools.chain(range(len(x_shape)), [-1],
                                  [cast(Optional[int], None)])
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testTakeAlongAxis(self, x_shape, i_shape, dtype, axis, rng_factory):
    rng = rng_factory(self.rng())
    i_shape = np.array(i_shape)
    if axis is None:
      i_shape = [np.prod(i_shape, dtype=np.int64)]
    else:
      # Test the case where the size of the axis doesn't necessarily broadcast.
      i_shape[axis] *= 3
      i_shape = list(i_shape)
    def args_maker():
      x = rng(x_shape, dtype)
      n = np.prod(x_shape, dtype=np.int32) if axis is None else x_shape[axis]
      i = rng(i_shape, np.int32) % (2 * n - 1) - (n - 1)
      return x, i

    jnp_op = lambda x, i: jnp.take_along_axis(x, i, axis=axis)

    if hasattr(np, "take_along_axis"):
      np_op = lambda x, i: np.take_along_axis(x, i, axis=axis)
      self._CheckAgainstNumpy(jnp_op, np_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_n={}_increasing={}".format(
          jtu.format_shape_dtype_string([shape], dtype),
          n, increasing),
       "dtype": dtype, "shape": shape, "n": n, "increasing": increasing,
       "rng_factory": jtu.rand_default}
      for dtype in inexact_dtypes
      for shape in [0, 5]
      for n in [2, 4]
      for increasing in [False, True]))
  def testVander(self, shape, dtype, n, increasing, rng_factory):
    rng = rng_factory(self.rng())
    def np_fun(arg):
      arg = arg.astype(np.float32) if dtype == jnp.bfloat16 else arg
      return np.vander(arg, N=n, increasing=increasing)
    jnp_fun = lambda arg: jnp.vander(arg, N=n, increasing=increasing)
    args_maker = lambda: [rng([shape], dtype)]
    # np.vander seems to return float64 for all floating types. We could obey
    # those semantics, but they seem like a bug.
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            tol={np.float32: 1e-3})
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix("nan_to_num", [shape],
                                                      [dtype]),
         "rng_factory": jtu.rand_some_inf_and_nan, "shape": shape,
         "dtype": dtype}
        for shape in array_shapes
        for dtype in inexact_dtypes))
  def testNanToNum(self, rng_factory, shape, dtype):
    rng = rng_factory(self.rng())
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

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix("ix_", shapes, dtypes),
         "rng_factory": jtu.rand_default, "shapes": shapes, "dtypes": dtypes}
        for shapes, dtypes in (
          ((), ()),
          (((7,),), (np.int32,)),
          (((3,), (4,)), (np.int32, np.int32)),
          (((3,), (1,), (4,)), (np.int32, np.int32, np.int32)),
        )))
  def testIx_(self, rng_factory, shapes, dtypes):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)
                          for shape, dtype in zip(shapes, dtypes)]
    self._CheckAgainstNumpy(np.ix_, jnp.ix_, args_maker)
    self._CompileAndCheck(jnp.ix_, args_maker)

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": "_dimensions={}_dtype={}_sparse={}".format(
            dimensions, dtype, sparse),
         "dimensions": dimensions, "dtype": dtype, "sparse": sparse}
        for dimensions in [(), (2,), (3, 0), (4, 5, 6)]
        for dtype in number_dtypes
        for sparse in [True, False]))
  def testIndices(self, dimensions, dtype, sparse):
    def args_maker(): return []
    if np.__version__ < "1.17":
      if sparse:
        raise SkipTest("indices does not have sparse on numpy < 1.17")
      np_fun = partial(np.indices, dimensions=dimensions,
                        dtype=dtype)
    else:
      np_fun = partial(np.indices, dimensions=dimensions,
                        dtype=dtype, sparse=sparse)
    jnp_fun = partial(jnp.indices, dimensions=dimensions,
                      dtype=dtype, sparse=sparse)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name":
           "_op={}_a_shape={}_q_shape={}_axis={}_keepdims={}_interpolation={}".format(
             op,
             jtu.format_shape_dtype_string(a_shape, a_dtype),
             jtu.format_shape_dtype_string(q_shape, q_dtype),
             axis, keepdims, interpolation),
         "a_rng": jtu.rand_some_nan if 'nan' in op else jtu.rand_default,
         "q_rng": q_rng, "op": op,
         "a_shape": a_shape, "a_dtype": a_dtype,
         "q_shape": q_shape, "q_dtype": q_dtype, "axis": axis,
         "keepdims": keepdims,
         "interpolation": interpolation}
        for (op, q_rng) in (
          ("percentile", partial(jtu.rand_uniform, low=0., high=100.)),
          ("quantile", partial(jtu.rand_uniform, low=0., high=1.)),
          ("nanpercentile", partial(jtu.rand_uniform, low=0., high=100.)),
          ("nanquantile", partial(jtu.rand_uniform, low=0., high=1.)),
        )
        for a_dtype in default_dtypes
        for a_shape, axis in (
          ((7,), None),
          ((47, 7), 0),
          ((4, 101), 1),
        )
        for q_dtype in [np.float32]
        for q_shape in scalar_shapes + [(4,)]
        for keepdims in [False, True]
        for interpolation in ['linear', 'lower', 'higher', 'nearest',
                              'midpoint']))
  def testQuantile(self, op, a_rng, q_rng, a_shape, a_dtype, q_shape, q_dtype,
                   axis, keepdims, interpolation):
    if "quantile" in op and numpy_version < (1, 15):
      raise SkipTest("Numpy < 1.15 does not have np.quantile")
    a_rng = a_rng(self.rng())
    q_rng = q_rng(self.rng())
    if "median" in op:
        args_maker = lambda: [a_rng(a_shape, a_dtype)]
    else:
        args_maker = lambda: [a_rng(a_shape, a_dtype), q_rng(q_shape, q_dtype)]
    def np_fun(*args):
      args = [x if jnp.result_type(x) != jnp.bfloat16 else
              np.asarray(x, np.float32) for x in args]
      return getattr(np, op)(*args, axis=axis, keepdims=keepdims,
                     interpolation=interpolation)
    jnp_fun = partial(getattr(jnp, op), axis=axis, keepdims=keepdims,
                      interpolation=interpolation)

    # TODO(phawkins): we currently set dtype=False because we aren't as
    # aggressive about promoting to float64. It's not clear we want to mimic
    # Numpy here.
    tol_spec = {np.float32: 2e-4, np.float64: 5e-6}
    tol = max(jtu.tolerance(a_dtype, tol_spec),
              jtu.tolerance(q_dtype, tol_spec))
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, rtol=tol)


  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name":
           "_{}_a_shape={}_axis={}_keepdims={}".format(
             op, jtu.format_shape_dtype_string(a_shape, a_dtype),
             axis, keepdims),
         "op": op, "a_shape": a_shape, "a_dtype": a_dtype,
         "axis": axis,
         "keepdims": keepdims}
        for a_dtype in default_dtypes
        for a_shape, axis in (
          ((7,), None),
          ((47, 7), 0),
          ((4, 101), 1),
        )
        for keepdims in [False, True]
        for op in ["median", "nanmedian"]))
  def testMedian(self, op, a_shape, a_dtype, axis, keepdims):
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


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes for dtype in all_dtypes))
  def testWhereOneArgument(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = lambda x: np.where(x)
    np_fun = jtu.ignore_warning(
      category=DeprecationWarning,
      message="Calling nonzero on 0d arrays.*")(np_fun)
    jnp_fun = lambda x: jnp.where(x)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}".format("_".join(
        jtu.format_shape_dtype_string(shape, dtype)
        for shape, dtype in zip(shapes, dtypes))),
     "rng_factory": jtu.rand_default, "shapes": shapes, "dtypes": dtypes}
    for shapes in filter(_shapes_are_broadcast_compatible,
                         itertools.combinations_with_replacement(all_shapes, 3))
    for dtypes in itertools.combinations_with_replacement(all_dtypes, 3)))
  def testWhereThreeArgument(self, rng_factory, shapes, dtypes):
    args_maker = self._GetArgsMaker(rng_factory(self.rng()), shapes, dtypes)
    def np_fun(cond, x, y):
      return _promote_like_jnp(partial(np.where, cond))(x, y)
    self._CheckAgainstNumpy(np_fun, jnp.where, args_maker)
    self._CompileAndCheck(jnp.where, args_maker)

  def testWhereScalarPromotion(self):
    x = jnp.where(jnp.array([True, False]), 3,
                  jnp.ones((2,), dtype=jnp.float32))
    self.assertEqual(x.dtype, np.dtype(np.float32))

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix("", shapes,
                                                      (np.bool_,) * n + dtypes),
         "rng_factory": jtu.rand_default, "shapes": shapes, "dtypes": dtypes}
        for n in range(0, 3)
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(all_shapes, 2 * n + 1))
        for dtypes in itertools.combinations_with_replacement(all_dtypes, n + 1)))
  def testSelect(self, rng_factory, shapes, dtypes):
    rng = rng_factory(self.rng())
    n = len(dtypes) - 1
    def args_maker():
      condlist = [rng(shape, np.bool_) for shape in shapes[:n]]
      choicelist = [rng(shape, dtype)
                    for shape, dtype in zip(shapes[n:-1], dtypes[:n])]
      default = rng(shapes[-1], dtypes[-1])
      return condlist, choicelist, default
    # TODO(phawkins): float32/float64 type mismatches
    def np_fun(condlist, choicelist, default):
      choicelist = [x if jnp.result_type(x) != jnp.bfloat16
                    else x.astype(np.float32) for x in choicelist]
      dtype = jnp.result_type(default, *choicelist)
      return np.select(condlist,
                        [np.asarray(x, dtype=dtype) for x in choicelist],
                        np.asarray(default, dtype=dtype))
    self._CheckAgainstNumpy(np_fun, jnp.select, args_maker,
                            check_dtypes=False)
    self._CompileAndCheck(jnp.select, args_maker,
                          rtol={np.float64: 1e-7, np.complex128: 1e-7})


  def testIssue330(self):
    x = jnp.full((1, 1), jnp.array([1])[0])  # doesn't crash
    self.assertEqual(x[0, 0], 1)

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

  # NOTE(mattjj): I disabled this test when removing lax._safe_mul because
  # introducing the convention 0 * inf = 0 leads to silently wrong results in
  # some cases. See this comment for details:
  # https://github.com/google/jax/issues/1052#issuecomment-514083352
  # def testIssue347(self):
  #   # https://github.com/google/jax/issues/347
  #   def test_fail(x):
  #     x = jnp.sqrt(jnp.sum(x ** 2, axis=1))
  #     ones = jnp.ones_like(x)
  #     x = jnp.where(x > 0.5, x, ones)
  #     return jnp.sum(x)
  #   x = jnp.array([[1, 2], [3, 4], [0, 0]], dtype=jnp.float64)
  #   result = api.grad(test_fail)(x)
  #   assert not np.any(np.isnan(result))

  def testIssue453(self):
    # https://github.com/google/jax/issues/453
    a = np.arange(6) + 1
    ans = jnp.reshape(a, (3, 2), order='F')
    expected = np.reshape(a, (3, 2), order='F')
    self.assertAllClose(ans, expected)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_dtype={}".format(op, pytype.__name__),
       "pytype": pytype, "dtype": dtype, "op": op}
      for pytype, dtype in [(int, jnp.int_), (float, jnp.float_),
                            (bool, jnp.bool_), (complex, jnp.complex_)]
      for op in ["atleast_1d", "atleast_2d", "atleast_3d"]))
  def testAtLeastNdLiterals(self, pytype, dtype, op):
    # Fixes: https://github.com/google/jax/issues/634
    np_fun = lambda arg: getattr(np, op)(arg).astype(dtype)
    jnp_fun = lambda arg: getattr(jnp, op)(arg)
    args_maker = lambda: [pytype(2)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {
      "testcase_name": "_shape={}_dtype={}_weights={}_minlength={}_length={}".format(
        shape, dtype, weights, minlength, length
      ),
      "shape": shape,
      "dtype": dtype,
      "weights": weights,
      "minlength": minlength,
      "length": length,
      "rng_factory": rng_factory}
    for shape in [(5,), (10,)]
    for dtype in int_dtypes
    for weights in [True, False]
    for minlength in [0, 20]
    for length in [None, 10]
    for rng_factory in [jtu.rand_positive]
  ))
  def testBincount(self, shape, dtype, weights, minlength, length, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: (rng(shape, dtype), (rng(shape, 'float32') if weights else None))

    np_fun = partial(np.bincount, minlength=minlength)
    jnp_fun = partial(jnp.bincount, minlength=minlength, length=length)

    if length is not None:
      self._CompileAndCheck(jnp_fun, args_maker)
    if length is None:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

  def testBincountNegative(self):
    # Test that jnp.bincount ignores negative values.
    x_rng = jtu.rand_int(self.rng(), -100, 100)
    w_rng = jtu.rand_uniform(self.rng())
    shape = (1000,)
    x = x_rng(shape, 'int32')
    w = w_rng(shape, 'float32')

    xn = np.array(x)
    xn[xn < 0] = 0
    wn = np.array(w)
    np_result = np.bincount(xn[xn >= 0], wn[xn >= 0])
    jnp_result = jnp.bincount(x, w)
    self.assertAllClose(np_result, jnp_result, check_dtypes=False)


  @parameterized.named_parameters(*jtu.cases_from_list(
      {"testcase_name": "_case={}".format(i),
       "input": input}
       for i, input in enumerate([
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
       ])))
  def testBlock(self, input):
    args_maker = lambda: [input]
    self._CheckAgainstNumpy(np.block, jnp.block, args_maker)
    self._CompileAndCheck(jnp.block, args_maker)

  def testLongLong(self):
    self.assertAllClose(np.int64(7), api.jit(lambda x: x)(np.longlong(7)))

  def testArange(self):
    # test cases inspired by dask tests at
    # https://github.com/dask/dask/blob/master/dask/array/tests/test_creation.py#L92
    self.assertAllClose(jnp.arange(77),
                        np.arange(77, dtype=jnp.int_))
    self.assertAllClose(jnp.arange(2, 13),
                        np.arange(2, 13, dtype=jnp.int_))
    self.assertAllClose(jnp.arange(4, 21, 9),
                        np.arange(4, 21, 9, dtype=jnp.int_))
    self.assertAllClose(jnp.arange(53, 5, -3),
                        np.arange(53, 5, -3, dtype=jnp.int_))
    self.assertAllClose(jnp.arange(77, dtype=float),
                        np.arange(77, dtype=float))
    self.assertAllClose(jnp.arange(2, 13, dtype=int),
                        np.arange(2, 13, dtype=int))
    self.assertAllClose(jnp.arange(0, 1, -0.5),
                        np.arange(0, 1, -0.5, dtype=jnp.float_))

    self.assertRaises(TypeError, lambda: jnp.arange())

    # test that jnp.arange(N) doesn't instantiate an ndarray
    self.assertNotEqual(type(jnp.arange(77)), type(np.arange(77)))
    self.assertEqual(type(jnp.arange(77)), type(lax.iota(np.int32, 77)))

    # test that jnp.arange(N, dtype=int32) doesn't instantiate an ndarray
    self.assertNotEqual(type(jnp.arange(77, dtype=jnp.int32)),
                         type(np.arange(77, dtype=np.int32)))
    self.assertEqual(type(jnp.arange(77, dtype=jnp.int32)),
                      type(lax.iota(np.int32, 77)))

    # test laziness for int dtypes
    self.assertTrue(xla.is_device_constant(jnp.arange(77)))
    self.assertTrue(xla.is_device_constant(jnp.arange(77, dtype=jnp.int32)))

  def testIssue830(self):
    a = jnp.arange(4, dtype=jnp.complex64)
    self.assertEqual(a.dtype, jnp.complex64)

  def testIssue728(self):
    assert jnp.allclose(jnp.eye(5000), np.eye(5000))
    self.assertEqual(0, np.sum(jnp.eye(1050) - np.eye(1050)))

  def testIssue746(self):
    jnp.arange(12).reshape(3, 4)  # doesn't crash

  def testIssue764(self):
    x = jnp.linspace(190, 200, 4)
    f = api.grad(lambda x: jnp.sum(jnp.tanh(x)))
    # Expected values computed with autograd in float64 precision.
    expected = np.array([3.71669453e-165, 4.72999108e-168, 6.01954653e-171,
                          7.66067839e-174], np.float64)
    self.assertAllClose(f(x), expected, check_dtypes=False)

  def testIssue776(self):
    """Tests that the scatter-add transpose rule instantiates symbolic zeros."""
    def f(u):
      y = jax.ops.index_add(np.ones(10,), [2, 4, 5], u)
      # The transpose rule for lax.tie_in returns a symbolic zero for its first
      # argument.
      return lax.tie_in(y, 7.)

    self.assertAllClose(np.zeros(3,), api.grad(f)(np.ones(3,)))

  # NOTE(mattjj): I disabled this test when removing lax._safe_mul because this
  # is a numerical stability issue that should be solved with a custom jvp rule
  # of the sigmoid function being differentiated here, not by safe_mul.
  # def testIssue777(self):
  #   x = jnp.linspace(-200, 0, 4, dtype=np.float32)
  #   f = api.grad(lambda x: jnp.sum(1 / (1 + jnp.exp(-x))))
  #   self.assertAllClose(f(x), np.array([0., 0., 0., 0.25], dtype=np.float32))

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(op, [()], [dtype]),
         "dtype": dtype, "op": op}
      for dtype in float_dtypes
      for op in ("sqrt", "arccos", "arcsin", "arctan", "sin", "cos", "tan",
                 "sinh", "cosh", "tanh", "arccosh", "arcsinh", "arctanh", "exp",
                 "log", "expm1", "log1p")))
  def testMathSpecialFloatValues(self, op, dtype):
    np_op = getattr(np, op)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                                 message="invalid value.*")(np_op)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                                 message="divide by zero.*")(np_op)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                                 message="overflow.*")(np_op)

    jnp_op = getattr(jnp, op)
    dtype = np.dtype(dtypes.canonicalize_dtype(dtype)).type
    for x in (np.nan, -np.inf, -100., -2., -1., 0., 1., 2., 100., np.inf,
              jnp.finfo(dtype).max, np.sqrt(jnp.finfo(dtype).max),
              np.sqrt(jnp.finfo(dtype).max) * 2.):
      if (op in ("sin", "cos", "tan") and
          jtu.device_under_test() == "tpu"):
        continue  # TODO(b/132196789): fix and reenable.
      x = dtype(x)
      expected = np_op(x)
      actual = jnp_op(x)
      tol = jtu.tolerance(dtype, {np.float32: 1e-3, np.float64: 1e-7})
      self.assertAllClose(expected, actual, atol=tol,
                          rtol=tol)

  def testIssue883(self):
    # from https://github.com/google/jax/issues/883

    @partial(api.jit, static_argnums=(1,))
    def f(x, v):
      return x

    x = jnp.ones((10, 10))
    v = jnp.array([1, 2, 3])
    _ = f(x, v)
    _ = f(x, v)  # doesn't crash

  def testReductionOfOutOfBoundsAxis(self):  # Issue 888
    x = jnp.ones((3, 4))
    self.assertRaises(ValueError, lambda: jnp.sum(x, axis=2))

  def testIssue956(self):
    self.assertRaises(TypeError, lambda: jnp.ndarray((1, 1)))

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name":
         "_shape={}_dtype={}_out_dtype={}_axis={}_ddof={}_keepdims={}"
         .format(shape, dtype, out_dtype, axis, ddof, keepdims),
         "shape": shape, "dtype": dtype, "out_dtype": out_dtype, "axis": axis,
         "ddof": ddof, "keepdims": keepdims, "rng_factory": rng_factory}
        for shape in [(5,), (10, 5)]
        for dtype in all_dtypes
        for out_dtype in inexact_dtypes
        for axis in [None, 0, -1]
        for ddof in [0, 1, 2]
        for keepdims in [False, True]
        for rng_factory in [jtu.rand_default]))
  def testVar(self, shape, dtype, out_dtype, axis, ddof, keepdims, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.")
    def np_fun(x):
      out = np.var(x.astype(jnp.promote_types(np.float32, dtype)),
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
         "ddof": ddof, "keepdims": keepdims, "rng_factory": rng_factory}
        for shape in [(5,), (10, 5)]
        for dtype in all_dtypes
        for out_dtype in inexact_dtypes
        for axis in [None, 0, -1]
        for ddof in [0, 1, 2]
        for keepdims in [False, True]
        for rng_factory in [jtu.rand_some_nan]))
  def testNanVar(self, shape, dtype, out_dtype, axis, ddof, keepdims, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.")
    def np_fun(x):
      out = np.nanvar(x.astype(jnp.promote_types(np.float32, dtype)),
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

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": "_shape={}_dtype={}_rowvar={}_ddof={}_bias={}".format(
            shape, dtype, rowvar, ddof, bias),
         "shape": shape, "dtype": dtype, "rowvar": rowvar, "ddof": ddof,
         "bias": bias, "rng_factory": rng_factory}
        for shape in [(5,), (10, 5), (5, 10)]
        for dtype in all_dtypes
        for rowvar in [True, False]
        for bias in [True, False]
        for ddof in [None, 2, 3]
        for rng_factory in [jtu.rand_default]))
  def testCov(self, shape, dtype, rowvar, ddof, bias, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    np_fun = partial(np.cov, rowvar=rowvar, ddof=ddof, bias=bias)
    jnp_fun = partial(jnp.cov, rowvar=rowvar, ddof=ddof, bias=bias)
    tol = {np.float32: 1e-5, np.complex64: 1e-5,
           np.float64: 1e-13, np.complex128: 1e-13}
    tol = 7e-2 if jtu.device_under_test() == "tpu" else tol
    tol = jtu.join_tolerance(tol, jtu.tolerance(dtype))
    self._CheckAgainstNumpy(
        np_fun, jnp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol,
                          rtol=tol)

  def testIssue967(self):
    self.assertRaises(TypeError, lambda: jnp.zeros(1.5))

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": "_shape={}_dtype={}_rowvar={}".format(
            shape, dtype.__name__, rowvar),
         "shape": shape, "dtype": dtype, "rowvar": rowvar}
        for shape in [(5,), (10, 5), (3, 10)]
        for dtype in number_dtypes
        for rowvar in [True, False]))
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
    self._CheckAgainstNumpy(
        np_fun, jnp_fun, args_maker, check_dtypes=False,
        tol=1e-2 if jtu.device_under_test() == "tpu" else None)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(jtu.format_shape_dtype_string(shape, dtype),
       "None" if end_dtype is None else jtu.format_shape_dtype_string(end_shape, end_dtype),
       "None" if begin_dtype is None else jtu.format_shape_dtype_string(begin_shape, begin_dtype)),
       "shape": shape, "dtype": dtype, "end_shape": end_shape,
       "end_dtype": end_dtype, "begin_shape": begin_shape,
       "begin_dtype": begin_dtype}
      for dtype in number_dtypes
      for end_dtype in [None] + [dtype]
      for begin_dtype in [None] + [dtype]
      for shape in [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE]
      for begin_shape in (
        [None] if begin_dtype is None
        else [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE])
      for end_shape in (
        [None] if end_dtype is None
        else [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE])))
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

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": "_shapes={}_dtype={}_indexing={}_sparse={}".format(
            shapes, dtype, indexing, sparse),
         "shapes": shapes, "dtype": dtype, "indexing": indexing,
         "sparse": sparse, "rng_factory": rng_factory}
        for shapes in [(), (5,), (5, 3)]
        for dtype in number_dtypes
        for indexing in ['xy', 'ij']
        for sparse in [True, False]
        for rng_factory in [jtu.rand_default]))
  def testMeshGrid(self, shapes, dtype, indexing, sparse, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, [(x,) for x in shapes],
                                    [dtype] * len(shapes))
    np_fun = partial(np.meshgrid, indexing=indexing, sparse=sparse)
    jnp_fun = partial(jnp.meshgrid, indexing=indexing, sparse=sparse)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": ("_start_shape={}_stop_shape={}_num={}_endpoint={}"
                           "_retstep={}_dtype={}").format(
            start_shape, stop_shape, num, endpoint, retstep, dtype),
         "start_shape": start_shape, "stop_shape": stop_shape,
         "num": num, "endpoint": endpoint, "retstep": retstep,
         "dtype": dtype, "rng_factory": rng_factory}
        for start_shape in [(), (2,), (2, 2)]
        for stop_shape in [(), (2,), (2, 2)]
        for num in [0, 1, 2, 5, 20]
        for endpoint in [True, False]
        for retstep in [True, False]
        for dtype in number_dtypes + [None,]
        for rng_factory in [jtu.rand_default]))
  def testLinspace(self, start_shape, stop_shape, num, endpoint,
                   retstep, dtype, rng_factory):
    if num == 1 and not endpoint and numpy_version < (1, 17, 5):
      raise SkipTest("Numpy < 1.17.5 has a linspace bug.")
    rng = rng_factory(self.rng())
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
      # floating-point compute between jitted platforms and non-jit + rounding
      # cause unavoidable variation in integer truncation for some inputs.
      if dtype in (inexact_dtypes + [None,]):
        self._CompileAndCheck(jnp_op, args_maker,
                              check_dtypes=False, atol=tol, rtol=tol)

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": "_dtype={}".format(dtype),
         "dtype": dtype,
         "rng_factory": rng_factory}
        for dtype in number_dtypes
        for rng_factory in [jtu.rand_default]))
  def testLinspaceEndpoints(self, dtype, rng_factory):
    """Regression test for Issue #3014."""
    rng = rng_factory(self.rng())
    endpoints = rng((2,), dtype)
    out = jnp.linspace(*endpoints, 10, dtype=dtype)
    self.assertAllClose(out[[0, -1]], endpoints, rtol=0, atol=0)

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": ("_start_shape={}_stop_shape={}_num={}_endpoint={}"
                           "_base={}_dtype={}").format(
            start_shape, stop_shape, num, endpoint, base,
            dtype.__name__ if dtype else "None"),
         "start_shape": start_shape,
         "stop_shape": stop_shape,
         "num": num, "endpoint": endpoint, "base": base,
         "dtype": dtype, "rng_factory": rng_factory}
        for start_shape in [(), (2,), (2, 2)]
        for stop_shape in [(), (2,), (2, 2)]
        for num in [0, 1, 2, 5, 20]
        for endpoint in [True, False]
        for base in [10.0, 2, np.e]
        for dtype in inexact_dtypes + [None,]
        for rng_factory in [jtu.rand_default]))
  def testLogspace(self, start_shape, stop_shape, num,
                   endpoint, base, dtype, rng_factory):
    if (dtype in int_dtypes and
        jtu.device_under_test() in ("gpu", "tpu") and
        not FLAGS.jax_enable_x64):
      raise unittest.SkipTest("GPUx32 truncated exponentiation"
                              " doesn't exactly match other platforms.")
    rng = rng_factory(self.rng())
    # relax default tolerances slightly
    tol = {np.float16: 2e-2, np.float32: 1e-2, np.float64: 1e-6,
           np.complex64: 1e-3, np.complex128: 1e-6}
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

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": ("_start_shape={}_stop_shape={}_num={}_endpoint={}"
                           "_dtype={}").format(
            start_shape, stop_shape, num, endpoint, dtype),
         "start_shape": start_shape,
         "stop_shape": stop_shape,
         "num": num, "endpoint": endpoint,
         "dtype": dtype, "rng_factory": rng_factory}
        for start_shape in [(), (2,), (2, 2)]
        for stop_shape in [(), (2,), (2, 2)]
        for num in [0, 1, 2, 5, 20]
        for endpoint in [True, False]
        # NB: numpy's geomspace gives nonsense results on integer types
        for dtype in inexact_dtypes + [None,]
        for rng_factory in [jtu.rand_default]))
  def testGeomspace(self, start_shape, stop_shape, num,
                    endpoint, dtype, rng_factory):
    rng = rng_factory(self.rng())
    # relax default tolerances slightly
    tol = {np.float16: 4e-3, np.float32: 2e-3, np.float64: 1e-14,
           np.complex128: 1e-14}
    def args_maker():
      """Test the set of inputs np.geomspace is well-defined on."""
      start, stop = self._GetArgsMaker(rng,
                                [start_shape, stop_shape],
                                [dtype, dtype])()
      # np.geomspace can't handle differently ranked tensors
      # w. negative numbers!
      start, stop = jnp.broadcast_arrays(start, stop)
      if dtype in complex_dtypes:
        return start, stop
      # to avoid NaNs, non-complex start and stop cannot
      # differ in sign, elementwise
      start = start * jnp.sign(start) * jnp.sign(stop)
      return start, stop
    start, stop = args_maker()
    ndim = len(np.shape(start + stop))
    for axis in range(-ndim, ndim):
      def jnp_op(start, stop):
        return jnp.geomspace(start, stop, num, endpoint=endpoint, dtype=dtype,
                             axis=axis)
      def np_op(start, stop):
        start = start.astype(np.float32) if dtype == jnp.bfloat16 else start
        stop = stop.astype(np.float32) if dtype == jnp.bfloat16 else stop
        return np.geomspace(
          start, stop, num, endpoint=endpoint,
          dtype=dtype if dtype != jnp.bfloat16 else np.float32,
          axis=axis).astype(dtype)
      self._CheckAgainstNumpy(np_op, jnp_op, args_maker,
                              check_dtypes=False, tol=tol)
      if dtype in (inexact_dtypes + [None,]):
        self._CompileAndCheck(jnp_op, args_maker,
                              check_dtypes=False, atol=tol, rtol=tol)

  def testDisableNumpyRankPromotionBroadcasting(self):
    try:
      prev_flag = FLAGS.jax_numpy_rank_promotion
      FLAGS.jax_numpy_rank_promotion = "allow"
      jnp.ones(2) + jnp.ones((1, 2))  # works just fine
    finally:
      FLAGS.jax_numpy_rank_promotion = prev_flag

    try:
      prev_flag = FLAGS.jax_numpy_rank_promotion
      FLAGS.jax_numpy_rank_promotion = "raise"
      self.assertRaises(ValueError, lambda: jnp.ones(2) + jnp.ones((1, 2)))
    finally:
      FLAGS.jax_numpy_rank_promotion = prev_flag

    try:
      prev_flag = FLAGS.jax_numpy_rank_promotion
      FLAGS.jax_numpy_rank_promotion = "warn"
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        jnp.ones(2) + jnp.ones((1, 2))
        assert len(w) > 0
        msg = str(w[-1].message)
        expected_msg = ("Following NumPy automatic rank promotion for add on "
                        "shapes (2,) (1, 2).")
        self.assertEqual(msg[:len(expected_msg)], expected_msg)

        prev_len = len(w)
        jnp.ones(2) + 3
        self.assertEqual(len(w), prev_len)  # don't want to warn for scalars
    finally:
      FLAGS.jax_numpy_rank_promotion = prev_flag

  def testStackArrayArgument(self):
    # tests https://github.com/google/jax/issues/1271
    @api.jit
    def foo(x):
      return jnp.stack(x)
    foo(np.zeros(2))  # doesn't crash

    @api.jit
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
    wrapped = linear_util.wrap_init(f)
    pv = partial_eval.PartialVal.unknown(jax.ShapedArray((3, 4), np.float32))
    _, _, consts = partial_eval.trace_to_jaxpr(wrapped, [pv])
    self.assertFalse(
      any(np.array_equal(x, np.full((3, 4), 2., dtype=np.float32))
          for x in consts))

  @parameterized.named_parameters(
      {"testcase_name": "_from={}_to={}".format(from_shape, to_shape),
       "rng_factory": rng_factory, "from_shape": from_shape, "to_shape": to_shape}
      for from_shape, to_shape in [
          [(1, 3), (4, 3)],
          [(3,), (2, 1, 3)],
          [(3,), (3, 3)],
          [(1,), (3,)],
      ]
      for rng_factory in [jtu.rand_default])
  def testBroadcastTo(self, from_shape, to_shape, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, [from_shape], [np.float32])
    np_op = lambda x: np.broadcast_to(x, to_shape)
    jnp_op = lambda x: jnp.broadcast_to(x, to_shape)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  def testBroadcastToIssue1522(self):
    self.assertRaisesRegex(
        ValueError, "Incompatible shapes for broadcasting: .*",
        lambda: jnp.broadcast_to(np.ones((2, 3)), (1, 3)))

  def testBroadcastToIntIssue1548(self):
    self.assertAllClose(jnp.broadcast_to(1, (3, 2)), np.ones((3, 2)),
                        check_dtypes=False)

  def testBroadcastToOnScalar(self):
    self.assertIsInstance(jnp.broadcast_to(10.0, ()), jnp.ndarray)
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

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": ("_shape={}_varargs={} axis={}_dtype={}")
         .format(shape, varargs, axis, dtype),
         "shape": shape,
         "varargs": varargs,
         "axis": axis,
         "dtype": dtype, "rng_factory": rng_factory}
        for shape in [(10,), (10, 15), (10, 15, 20)]
        for _num_axes in range(len(shape))
        for varargs in itertools.combinations(range(1, len(shape) + 1), _num_axes)
        for axis in itertools.combinations(range(len(shape)), _num_axes)
        for dtype in inexact_dtypes
        for rng_factory in [jtu.rand_default]))
  def testGradient(self, shape, varargs, axis, dtype, rng_factory):
    rng = rng_factory(self.rng())
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
        "Shapes must be 1D sequences of concrete values of integer type.*\n"
        "If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.",
        lambda: api.jit(jnp.zeros)(2))

  def testTraceMethod(self):
    x = self.rng().randn(3, 4).astype(jnp.float_)
    self.assertAllClose(x.trace(), jnp.array(x).trace())
    self.assertAllClose(x.trace(), api.jit(lambda y: y.trace())(x))

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

  def testReductionWithRepeatedAxisError(self):
    with self.assertRaisesRegex(ValueError, r"duplicate value in 'axis': \(0, 0\)"):
      jnp.sum(jnp.arange(3), (0, 0))

  def testArangeConcretizationError(self):
    msg = r"Abstract tracer.*\(in jax.numpy.arange argument `{}`\).*".format
    with self.assertRaisesRegex(jax.core.ConcretizationTypeError, msg('stop')):
      jax.jit(jnp.arange)(3)

    with self.assertRaisesRegex(jax.core.ConcretizationTypeError, msg('start')):
      jax.jit(lambda start: jnp.arange(start, 3))(0)

    with self.assertRaisesRegex(jax.core.ConcretizationTypeError, msg('stop')):
      jax.jit(lambda stop: jnp.arange(0, stop))(3)


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
    grad_test_spec(jnp.arccosh, nargs=1, order=2,
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

class NumpyGradTests(jtu.JaxTestCase):
  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.name, shapes, itertools.repeat(dtype)),
         "op": rec.op, "rng_factory": rec.rng_factory, "shapes": shapes, "dtype": dtype,
         "order": rec.order, "tol": rec.tol}
        for shapes in itertools.combinations_with_replacement(nonempty_shapes, rec.nargs)
        for dtype in rec.dtypes)
      for rec in GRAD_TEST_RECORDS))
  def testOpGrad(self, op, rng_factory, shapes, dtype, order, tol):
    rng = rng_factory(self.rng())
    tol = jtu.join_tolerance(tol, {np.float32: 1e-1, np.float64: 1e-3,
                                   np.complex64: 1e-1, np.complex64: 1e-3})
    args = tuple(rng(shape, dtype) for shape in shapes)
    check_grads(op, args, order, ["fwd", "rev"], tol, tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
          {"testcase_name": "_{}_{}".format(rec.op.__name__, special_value),
           "op": rec.op, "special_value": special_value, "order": rec.order}
          for special_value in rec.values)
      for rec in GRAD_SPECIAL_VALUE_TEST_RECORDS))
  def testOpGradSpecialValue(self, op, special_value, order):
    check_grads(op, (special_value,), order, ["fwd", "rev"],
                atol={np.float32: 3e-3})

  def testTakeAlongAxisIssue1521(self):
    # https://github.com/google/jax/issues/1521
    idx = jnp.repeat(jnp.arange(3), 10).reshape((30, 1))

    def f(x):
      y = x * jnp.arange(3.).reshape((1, 3))
      return jnp.take_along_axis(y, idx, -1).sum()

    check_grads(f, (1.,), order=1)


  def testWrappedSignaturesMatch(self):
    """Test that jax.numpy function signatures match numpy."""
    jnp_funcs = {name: getattr(jnp, name) for name in dir(jnp)}
    func_pairs = {name: (fun, fun.__np_wrapped__) for name, fun in jnp_funcs.items()
                  if hasattr(fun, '__np_wrapped__')}
    assert len(func_pairs) > 0

    # TODO(jakevdp): fix some of the following signatures. Some are due to wrong argument names.
    unsupported_params = {
      'allclose': ['equal_nan'],
      'amax': ['initial', 'where'],
      'amin': ['initial', 'where'],
      'angle': ['deg'],
      'argmax': ['out'],
      'argmin': ['out'],
      'around': ['out'],
      'broadcast_to': ['subok', 'array'],
      'clip': ['out', 'kwargs'],
      'convolve': ['v', 'a'],
      'corrcoef': ['ddof', 'bias'],
      'correlate': ['v', 'a'],
      'cumprod': ['out'],
      'cumproduct': ['out'],
      'cumsum': ['out'],
      'diff': ['prepend', 'append'],
      'einsum_path': ['einsum_call', 'optimize'],
      'empty_like': ['shape', 'subok', 'a', 'order'],
      'eye': ['order'],
      'full': ['order'],
      'full_like': ['shape', 'subok', 'order'],
      'gradient': ['varargs', 'axis', 'f', 'edge_order'],
      'histogram': ['normed'],
      'isneginf': ['out'],
      'isposinf': ['out'],
      'isscalar': ['element'],
      'max': ['initial', 'where'],
      'min': ['initial', 'where'],
      'nancumprod': ['out'],
      'nancumsum': ['out'],
      'nanprod': ['dtype'],
      'nansum': ['dtype'],
      'ones': ['order'],
      'ones_like': ['shape', 'subok', 'a', 'order'],
      'pad': ['kwargs'],
      'polyadd': ['a1', 'a2'],
      'polyder': ['p'],
      'polysub': ['a1', 'a2'],
      'prod': ['initial', 'where'],
      'product': ['initial', 'where'],
      'round': ['out'],
      'stack': ['out'],
      'sum': ['initial', 'where'],
      'tile': ['A'],
      'zeros_like': ['shape', 'subok', 'a', 'order']
    }

    extra_params = {
      'all': ['dtype'],
      'alltrue': ['dtype'],
      'amax': ['dtype'],
      'amin': ['dtype'],
      'any': ['dtype'],
      'broadcast_to': ['arr'],
      'convolve': ['x', 'y'],
      'correlate': ['x', 'y'],
      'einsum_path': ['kwargs', 'subscripts'],
      'empty_like': ['x'],
      'gradient': ['kwargs', 'a', 'args'],
      'isneginf': ['infinity'],
      'isposinf': ['infinity'],
      'isscalar': ['num'],
      'max': ['dtype'],
      'min': ['dtype'],
      'nanmax': ['kwargs'],
      'nanmin': ['kwargs'],
      'nanprod': ['kwargs'],
      'nansum': ['kwargs'],
      'ones_like': ['x'],
      'pad': ['constant_values'],
      'polyadd': ['b', 'a'],
      'polyder': ['a'],
      'polysub': ['b', 'a'],
      'sometrue': ['dtype'],
      'tile': ['a'],
      'zeros_like': ['x']
    }

    mismatches = {}

    for name, (jnp_fun, np_fun) in func_pairs.items():
      # Some signatures have changed; skip for older numpy versions.
      if np.__version__ < "1.19" and name in ['einsum_path', 'gradient', 'isscalar']:
        continue
      # Note: can't use inspect.getfullargspec due to numpy issue
      # https://github.com/numpy/numpy/issues/12225
      try:
        np_params = inspect.signature(np_fun).parameters
      except ValueError:
        # Some functions cannot be inspected
        continue
      jnp_params = inspect.signature(jnp_fun).parameters
      extra = set(extra_params.get(name, []))
      unsupported = set(unsupported_params.get(name, []))

      # Checks to prevent tests from becoming out-of-date. If these fail,
      # it means that extra_params or unsupported_params need to be updated.
      assert extra.issubset(jnp_params), f"{name}: extra={extra} is not a subset of jnp_params={set(jnp_params)}."
      assert not unsupported.intersection(jnp_params), f"{name}: unsupported={unsupported} overlaps with jnp_params={set(jnp_params)}."

      # Skip functions that only have *args and **kwargs; we can't introspect these further.
      var_args = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
      if all(p.kind in var_args for p in jnp_params.values()):
        continue
      if all(p.kind in var_args for p in np_params.values()):
        continue

      # Remove known extra parameters.
      jnp_params = {a: p for a, p in jnp_params.items() if a not in extra}

      # Remove known unsupported parameters.
      np_params = {a: p for a, p in np_params.items() if a not in unsupported}

      # Older versions of numpy may have fewer parameters; to avoid extraneous errors on older numpy
      # versions, we allow for jnp to have more parameters.
      if list(jnp_params)[:len(np_params)] != list(np_params):
        mismatches[name] = {'np_params': list(np_params), 'jnp_params': list(jnp_params)}

    self.assertEqual(mismatches, {})


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
