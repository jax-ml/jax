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
"""Defines test inputs and invocations for JAX primitives.

Used to test various implementations of JAX primitives, e.g., against
NumPy (lax_reference) or TensorFlow.
"""

import operator
from typing import Any, Callable, Dict, Iterable, Optional, NamedTuple, Sequence, Tuple, Union

from functools import partial

from absl import testing
import jax
from jax import config
from jax import dtypes
from jax import ad_util
from jax import test_util as jtu
from jax import lax
from jax import numpy as jnp
from jax._src.lax import control_flow as lax_control_flow

from jaxlib import xla_client

import numpy as np

FLAGS = config.FLAGS

Rng = Any  # A random number generator

class RandArg(NamedTuple):
  """Descriptor for a randomly generated argument.

  See description of `Harness`.
  """
  shape: Tuple[int, ...]
  dtype: np.dtype

class StaticArg(NamedTuple):
  """Descriptor for a static argument.

  See description of `Harness`.
  """
  value: Any

class Harness:
  """Specifies inputs and callable for a primitive.

  A harness is conceptually a callable and a list of arguments, that together
  exercise a use case. The harness can optionally have additional parameters
  that can be used by the test.

  The arguments are specified through argument descriptors. An argument
  descriptor can be:
    * a numeric value or ndarray, or
    * an instance of ``RandArg(shape, dtype)`` to be used with a PRNG to generate
      random tensor of the given shape and type, or
    * an instance of ``StaticArg(value)``. These are values that specialize the
      callable, but are not exposed as external arguments.

  For example, a harness for ``lax.take(arr, indices, axis=None)`` may want
  to expose as external (dynamic) argument the array and the indices, and
  keep the axis as a static argument (technically specializing the `take` to
  a axis):

    Harness(f"take_axis={axis}",
            lax.take,
            [RandArg((2, 4), np.float32), np.array([-1, 0, 1]), StaticArg(axis)],
            axis=axis)
  """
  # Descriptive name of the harness, used as a testcase_name. Unique in a group.
  name: str
  # The function taking all arguments (static and dynamic).
  fun: Callable
  arg_descriptors: Sequence[Union[RandArg, StaticArg, Any]]
  rng_factory: Callable
  params: Dict[str, Any]

  def __init__(self, name, fun, arg_descriptors, *,
               rng_factory=jtu.rand_default, **params):
    self.name = name
    self.fun = fun
    self.arg_descriptors = arg_descriptors
    self.rng_factory = rng_factory
    self.params = params

  def __str__(self):
    return self.name

  def _arg_maker(self, arg_descriptor, rng: Rng):
    if isinstance(arg_descriptor, StaticArg):
      return arg_descriptor.value
    if isinstance(arg_descriptor, RandArg):
      return self.rng_factory(rng)(arg_descriptor.shape, arg_descriptor.dtype)
    return arg_descriptor

  def args_maker(self, rng: Rng) -> Sequence:
    """All-argument maker, including the static ones."""
    return [self._arg_maker(ad, rng) for ad in self.arg_descriptors]

  def dyn_args_maker(self, rng: Rng) -> Sequence:
    """A dynamic-argument maker, for use with `dyn_fun`."""
    return [self._arg_maker(ad, rng) for ad in self.arg_descriptors
            if not isinstance(ad, StaticArg)]

  def dyn_fun(self, *dyn_args):
    """Invokes `fun` given just the dynamic arguments."""
    all_args = self._args_from_dynargs(dyn_args)
    return self.fun(*all_args)

  def _args_from_dynargs(self, dyn_args: Sequence) -> Sequence:
    """All arguments, including the static ones."""
    next_dynamic_argnum = 0
    all_args = []
    for ad in self.arg_descriptors:
      if isinstance(ad, StaticArg):
        all_args.append(ad.value)
      else:
        all_args.append(dyn_args[next_dynamic_argnum])
        next_dynamic_argnum += 1
    return all_args


def parameterized(harness_group: Iterable[Harness],
                  one_containing : Optional[str] = None):
  """Decorator for tests.

  The tests receive a `harness` argument.

  The `one_containing` parameter is useful for debugging. If given, then
  picks only one harness whose name contains the string. The whole set of
  parameterized tests is reduced to one test, whose name is not decorated
  to make it easier to pick for running.
  """
  cases = tuple(
    dict(testcase_name=harness.name if one_containing is None else "",
         harness=harness)
    for harness in harness_group
    if one_containing is None or one_containing in harness.name)
  if one_containing is not None:
    if not cases:
      raise ValueError(f"Cannot find test case with name containing {one_containing}."
                       "Names are:"
                       "\n".join([harness.name for harness in harness_group]))
    cases = cases[0:1]
  return testing.parameterized.named_parameters(*cases)

### Harness definitions ###
###
_LAX_UNARY_ELEMENTWISE = (
  lax.abs, lax.acosh, lax.asinh, lax.atanh, lax.bessel_i0e, lax.bessel_i1e,
  lax.ceil, lax.cos, lax.cosh, lax.digamma, lax.erf, lax.erf_inv, lax.erfc,
  lax.exp, lax.expm1, lax.floor, lax.is_finite, lax.lgamma, lax.log,
  lax.log1p, lax.neg, lax.round, lax.rsqrt, lax.sign, lax.sin, lax.sinh,
  lax.sqrt, lax.tan, lax.tanh)

lax_unary_elementwise = tuple(
  Harness(f"{f_lax.__name__}_{jtu.dtype_str(dtype)}",
          f_lax,
          [arg],
          lax_name=f_lax.__name__,
          dtype=dtype)
  for f_lax in _LAX_UNARY_ELEMENTWISE
  for dtype in jtu.dtypes.all_floating
  for arg in [
    np.array([-1.6, -1.4, -1.0, 0.0, 0.1, 0.2, 1., 1.4, 1.6], dtype=dtype)
  ]
)

_LAX_COMPARATORS = (
  lax.eq, lax.ge, lax.gt, lax.le, lax.lt, lax.ne)

def _make_comparator_harness(name, *, dtype=np.float32, op=lax.eq, lhs_shape=(),
                             rhs_shape=()):
  return Harness(f"{name}_op={op.__name__}_lhs={jtu.format_shape_dtype_string(lhs_shape, dtype)}_rhs={jtu.format_shape_dtype_string(rhs_shape, dtype)}",
                 op,
                 [RandArg(lhs_shape, dtype), RandArg(rhs_shape, dtype)],
                 lhs_shape=lhs_shape,
                 rhs_shape=rhs_shape,
                 dtype=dtype)

lax_comparators = tuple( # Validate dtypes
  _make_comparator_harness("dtypes", dtype=dtype, op=op)
  for op in _LAX_COMPARATORS
  for dtype in (jtu.dtypes.all if op in [lax.eq, lax.ne] else
                set(jtu.dtypes.all) - set(jtu.dtypes.complex))
) + tuple( # Validate broadcasting behavior
  _make_comparator_harness("broadcasting", lhs_shape=lhs_shape,
                           rhs_shape=rhs_shape, op=op)
  for op in _LAX_COMPARATORS
  for lhs_shape, rhs_shape in [
    ((), (2, 3)),     # broadcast scalar
    ((1, 2), (3, 2)), # broadcast along specific axis
  ]
)

lax_bitwise_not = tuple(
  [Harness(f"{jtu.dtype_str(dtype)}",
          lax.bitwise_not,
          [arg],
          dtype=dtype)
  for dtype in jtu.dtypes.all_integer + jtu.dtypes.all_unsigned
  for arg in [
    np.array([-1, -3, -2, 0, 0, 2, 1, 3], dtype=dtype),
  ]] +

  [Harness("bool",
          f_lax,
          [arg],
          lax_name=f_lax.__name__,
          dtype=np.bool_)
  for f_lax in [lax.bitwise_not]
  for arg in [
    np.array([True, False])
  ]]
)

lax_zeros_like = tuple(
  Harness(f"shape={jtu.format_shape_dtype_string(shape, dtype)}",
          ad_util.zeros_like_p.bind,
          [RandArg(shape, dtype)],
          shape=shape,
          dtype=dtype)
  for shape in [(3, 4, 5)]
  for dtype in jtu.dtypes.all
)

lax_population_count = tuple(
  Harness(f"{jtu.dtype_str(dtype)}",
          lax.population_count,
          [arg],
          dtype=dtype)
  for dtype in jtu.dtypes.all_integer + jtu.dtypes.all_unsigned
  for arg in [
    np.array([-1, -2, 0, 1], dtype=dtype)
  ]
)

def _get_max_identity(dtype):
  if dtypes.issubdtype(dtype, np.inexact):
    return np.array(-np.inf, dtype)
  elif dtypes.issubdtype(dtype, np.integer):
    return np.array(dtypes.iinfo(dtype).min, dtype)
  elif dtypes.issubdtype(dtype, np.bool_):
    return np.array(False, np.bool_)

def _get_min_identity(dtype):
  if dtypes.issubdtype(dtype, np.inexact):
    return np.array(np.inf, dtype)
  elif dtypes.issubdtype(dtype, np.integer):
    return np.array(dtypes.iinfo(dtype).max, dtype)
  elif dtypes.issubdtype(dtype, np.bool_):
    return np.array(True, np.bool_)

def _make_argminmax_harness(name, *, shape=(15,), dtype=jnp.float32, axes=(0,),
                            index_dtype=np.int32, prim=lax.argmin_p,
                            arr=None):
  arr = arr if arr is not None else RandArg(shape, dtype)
  dtype, shape = arr.dtype, arr.shape
  index_dtype = dtypes.canonicalize_dtype(index_dtype)
  return Harness(f"{name}_prim={prim.name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_axes={axes}_indexdtype={index_dtype}",
                 lambda arg: prim.bind(arg, axes=axes, index_dtype=index_dtype),
                 [arr],
                 shape=shape,
                 dtype=dtype,
                 axes=axes,
                 index_dtype=index_dtype,
                 prim=prim)

lax_argminmax = tuple( # Validate dtypes for each primitive
  _make_argminmax_harness("dtypes", dtype=dtype, prim=prim)
  for dtype in set(jtu.dtypes.all) - set(jtu.dtypes.complex)
  for prim in [lax.argmin_p, lax.argmax_p]
) + tuple( # Validate axes for each primitive
  _make_argminmax_harness("axes", shape=shape, axes=axes, prim=prim)
  for shape, axes in [
    ((18, 12), (1,)), # non major axis
  ]
  for prim in [lax.argmin_p, lax.argmax_p]
) + tuple( # Validate index dtype for each primitive
  _make_argminmax_harness("index_dtype", index_dtype=index_dtype, prim=prim)
  for index_dtype in jtu.dtypes.all_integer + jtu.dtypes.all_unsigned
  for prim in [lax.argmin_p, lax.argmax_p]
)
# TODO(bchetioui): the below documents a limitation of argmin and argmax when a
# dimension of the input is too large. However, it is not categorizable as it
# seems that the converter fails before reaching the actual primitive call. This
# suggests that we may need to harden the converter to handle inputs this big.
# + tuple( # Document limitation in case of too large axis
#  _make_argminmax_harness("overflow_axis", prim=prim,
#                          arr=np.ones((2**31,), dtype=np.uint8))
#  for prim in [lax.argmin_p, lax.argmax_p]
#)

lax_add_mul = tuple(
  Harness(f"fun={f_jax.__name__}_{jtu.dtype_str(dtype)}",
          f_jax,
          [lhs, rhs],
          f_jax=f_jax,
          dtype=dtype)
  for f_jax in [lax.add, lax.mul]
  for dtype in filter(lambda t: t != np.bool_, jtu.dtypes.all)
  for lhs, rhs in [
    (np.array([1, 2], dtype=dtype), np.array([3, 4], dtype=dtype))
  ]
) + tuple(
  Harness(f"fun={f_jax.__name__}_bounds_{jtu.dtype_str(dtype)}",
          f_jax,
          [lhs, rhs],
          f_jax=f_jax,
          dtype=dtype)
  for f_jax in [lax.add, lax.mul]
  for dtype in filter(lambda t: t != np.bool_, jtu.dtypes.all)
  for lhs, rhs in [
    (np.array([3, 3], dtype=dtype),
     np.array([_get_max_identity(dtype), _get_min_identity(dtype)], dtype=dtype))
  ]
)

lax_min_max = tuple(
  Harness(f"fun={f_jax.__name__}_{jtu.dtype_str(dtype)}",
          f_jax,
          [lhs, rhs],
          f_jax=f_jax,
          dtype=dtype)
  for f_jax in [lax.min, lax.max]
  for dtype in jtu.dtypes.all
  for lhs, rhs in [
    (np.array([1, 2], dtype=dtype), np.array([3, 4], dtype=dtype))
  ]
) + tuple(
  Harness(f"fun={f_jax.__name__}_inf_nan_{jtu.dtype_str(dtype)}_{lhs[0]}_{rhs[0]}",
          f_jax,
          [lhs, rhs],
          f_jax=f_jax,
          dtype=dtype)
  for f_jax in [lax.min, lax.max]
  for dtype in jtu.dtypes.all_floating + jtu.dtypes.complex
  for lhs, rhs in [
    (np.array([np.inf, np.inf], dtype=dtype),
     np.array([np.nan, np.nan], dtype=dtype)),
    (np.array([-np.inf, -np.inf], dtype=dtype),
     np.array([np.nan, np.nan], dtype=dtype))
  ]
)

_LAX_BINARY_ELEMENTWISE = (
  lax.add, lax.atan2, lax.div, lax.igamma, lax.igammac, lax.max, lax.min,
  lax.nextafter, lax.rem, lax.sub)

lax_binary_elementwise = tuple(
  Harness(f"{f_lax.__name__}_{jtu.dtype_str(dtype)}",
          f_lax,
          [arg1, arg2],
          lax_name=f_lax.__name__,
          dtype=dtype
          )
  for f_lax in _LAX_BINARY_ELEMENTWISE
  for dtype in jtu.dtypes.all_floating
  for arg1, arg2 in [
    (np.array([-1.6, -1.4, -1.0, 0.0, 0.1, 0.2, 1., 1.4, 1.6], dtype=dtype),
     np.array([-1.6, 1.4, 1.0, 0.0, 0.1, 0.2, 1., 1.4, -1.6], dtype=dtype))
  ]
)

_LAX_BINARY_ELEMENTWISE_LOGICAL = (
    lax.bitwise_and, lax.bitwise_or, lax.bitwise_xor, lax.shift_left,
)

lax_binary_elementwise_logical = tuple(
  [Harness(f"{f_lax.__name__}_{jtu.dtype_str(dtype)}",
           f_lax,
           [arg1, arg2],
           lax_name=f_lax.__name__,
           dtype=dtype)
   for f_lax in _LAX_BINARY_ELEMENTWISE_LOGICAL
   for dtype in jtu.dtypes.all_integer + jtu.dtypes.all_unsigned
   for arg1, arg2 in [
     (np.array([1, 3, 2, 0, 0, 2, 1, 3], dtype=dtype),
      np.array([1, 2, 3, 0, 1, 0, 2, 3], dtype=dtype))
   ]
   ] +

  [Harness(f"{f_lax.__name__}_bool",
           f_lax,
           [arg1, arg2],
           lax_name=f_lax.__name__,
           dtype=np.bool_)
   for f_lax in [lax.bitwise_and, lax.bitwise_or, lax.bitwise_xor]
   for arg1, arg2 in [
     (np.array([True, True, False, False]),
      np.array([True, False, True, False])),
   ]
   ]
)

def _make_broadcast_in_dim_harness(name, *, dtype=np.float32,
                                   shape=(2,), outshape=(2,),
                                   broadcast_dimensions=(0,)):
  return Harness(f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_outshape={outshape}_broadcastdimensions={broadcast_dimensions}",
                 lambda operand: lax.broadcast_in_dim_p.bind(
                     operand, shape=outshape,
                     broadcast_dimensions=broadcast_dimensions),
                 [RandArg(shape, dtype)],
                 shape=shape,
                 dtype=dtype,
                 outshape=outshape,
                 broadcast_dimensions=broadcast_dimensions)

lax_broadcast_in_dim = tuple( # Validate dtypes
  _make_broadcast_in_dim_harness("dtypes", dtype=dtype)
  for dtype in jtu.dtypes.all
) + tuple( # Validate parameter combinations
  _make_broadcast_in_dim_harness("parameter_combinations", shape=shape,
                                 outshape=outshape,
                                 broadcast_dimensions=broadcast_dimensions)
  for shape, outshape, broadcast_dimensions in [
    [(2,), (3, 2), (1,)],        # add major dimension
    [(2,), (2, 3), (0,)],        # add inner dimension
    [(), (2, 3), ()],            # use scalar shape
    [(1, 2), (4, 3, 2), (0, 2)], # map size 1 dim to different output dim value
  ]
)

def _make_broadcast_harness(name, *, dtype=np.float32, shape=(2,), sizes=()):
  return Harness(f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_sizes={sizes}",
                 lambda operand: lax.broadcast_p.bind(operand, sizes=sizes),
                 [RandArg(shape, dtype)],
                 shape=shape,
                 dtype=dtype,
                 sizes=sizes)

lax_broadcast = tuple( # Validate dtypes
  _make_broadcast_harness("dtypes", dtype=dtype)
  for dtype in jtu.dtypes.all
) + tuple( # Validate sizes
  _make_broadcast_harness("sizes", sizes=sizes)
  for sizes in [
    (2,),      # broadcast 1 dim
    (1, 2, 3), # broadcast n > 1 dims
  ]
)

lax_betainc = tuple(
  Harness(f"_{jtu.dtype_str(dtype)}",
           lax.betainc,
           [arg1, arg2, arg3],
           dtype=dtype)
   for dtype in jtu.dtypes.all_floating
   for arg1, arg2, arg3 in [
     (np.array([-1.6, -1.4, -1.0, 0.0, 0.1, 0.3, 1, 1.4, 1.6], dtype=dtype),
      np.array([-1.6, 1.4, 1.0, 0.0, 0.2, 0.1, 1, 1.4, -1.6], dtype=dtype),
      np.array([1.0, -1.0, 2.0, 1.0, 0.3, 0.3, -1.0, 2.4, 1.6], dtype=dtype))
  ]
)


_gather_input = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
lax_gather = tuple(
  # Construct gather harnesses using take
  [Harness(f"from_take_indices_shape={indices.shape}_axis={axis}",
           lambda a, i, axis: jnp.take(a, i, axis=axis),
           [_gather_input,
            indices,
            StaticArg(axis)])
  for indices in [
    # Ensure each set of indices has a distinct shape
    np.array(2, dtype=np.int32),
    np.array([2], dtype=np.int32),
    np.array([2, 4], dtype=np.int32),
    np.array([[2, 4], [5, 6]], dtype=np.int32),
    np.array([0, 1, 10], dtype=np.int32),  # Index out of bounds
    np.array([0, 1, 2, -1], dtype=np.int32),  # Index out of bounds
  ]
  for axis in [0, 1, 2]] +

  # Directly from lax.gather in lax_test.py.
  [Harness(
    f"_shape={shape}_idxs_shape={idxs.shape}_dnums={dnums}_slice_sizes={slice_sizes}",
    lambda op, idxs, dnums, slice_sizes: lax.gather(op, idxs, dimension_numbers=dnums, slice_sizes=slice_sizes),
    [RandArg(shape, np.float32),
     idxs, StaticArg(dnums), StaticArg(slice_sizes)])
    for shape, idxs, dnums, slice_sizes in [
    ((5,), np.array([[0], [2]]), lax.GatherDimensionNumbers(
      offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
     (1,)),
    ((10,), np.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
      offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
     (2,)),
    ((10, 5,), np.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
      offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
     (1, 3)),
    ((10, 5), np.array([[0, 2], [1, 0]]), lax.GatherDimensionNumbers(
      offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)),
     (1, 3)),
    ]
  ]
)

def _make_scatter_harness(name, *, shape=(5,), f_lax=lax.scatter_min,
                          indices_are_sorted=False, unique_indices=False,
                          scatter_indices=np.array([[0], [2]]),
                          update_shape=(2,), dtype=np.float32,
                          dimension_numbers=((), (0,), (0,))):
  dimension_numbers = lax.ScatterDimensionNumbers(*dimension_numbers)
  return Harness(
    f"{name}_fun={f_lax.__name__}_shape={jtu.format_shape_dtype_string(shape, dtype)}_scatterindices={scatter_indices.tolist()}_updateshape={update_shape}_updatewindowdims={dimension_numbers.update_window_dims}_insertedwindowdims={dimension_numbers.inserted_window_dims}_scatterdimstooperanddims={dimension_numbers.scatter_dims_to_operand_dims}_indicesaresorted={indices_are_sorted}_uniqueindices={unique_indices}".replace(' ', ''),
    partial(f_lax, indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices),
    [RandArg(shape, dtype), StaticArg(scatter_indices),
     RandArg(update_shape, dtype), StaticArg(dimension_numbers)],
    f_lax=f_lax,
    shape=shape,
    dtype=dtype,
    scatter_indices=scatter_indices,
    update_shape=update_shape,
    dimension_numbers=dimension_numbers,
    indices_are_sorted=indices_are_sorted,
    unique_indices=unique_indices)

lax_scatter = tuple( # Validate dtypes
  _make_scatter_harness("dtypes", dtype=dtype)
  for dtype in jtu.dtypes.all
) + tuple( # Validate f_lax/update_jaxpr
  _make_scatter_harness("update_function", f_lax=f_lax)
  # We explicitly decide against testing lax.scatter, as its reduction function
  # is lambda x, y: y, which is not commutative and thus makes results
  # non-deterministic when an index into the operand is updated several times.
  for f_lax in [
    lax.scatter_add,
    lax.scatter_max,
    lax.scatter_mul
  ]
) + tuple( # Validate shapes, dimension numbers and scatter indices
  _make_scatter_harness("shapes_and_dimension_numbers", shape=shape,
                        update_shape=update_shape,
                        scatter_indices=np.array(scatter_indices),
                        dimension_numbers=dimension_numbers)
  for shape, scatter_indices, update_shape, dimension_numbers in [
    ((10,),   [[0], [0], [0]], (3, 2), ((1,), (), (0,))),
    ((10, 5), [[0], [2], [1]], (3, 3), ((1,), (0,), (0,)))
  ]
) + tuple ( # Validate sorted indices
  [_make_scatter_harness("indices_are_sorted", indices_are_sorted=True)]
) + tuple( # Validate unique_indices
  _make_scatter_harness("unique_indices", unique_indices=unique_indices)
  # `unique_indices` does not affect correctness, only performance, and thus
  # does not need to be tested here. If/when it will make sense to add a test
  # with `unique_indices` = True, particular care will have to be taken with
  # regards to the choice of parameters, as the results are only predictable
  # when all the indices to be updated are pairwise non-overlapping. Identifying
  # such cases is non-trivial.
  for unique_indices in [False]
)

disable_xla = tuple(
  Harness("_pad",
          lax.pad,
          [RandArg(shape, dtype), np.array(0, dtype), StaticArg(pads)],
          shape=shape,
          dtype=dtype,
          pads=pads)
  for shape in [(2, 3)]
  for dtype in [np.float32]
  for pads in [
    [(-1, 0, 0), (0, 0, 0)]
  ]
)

lax_pad = tuple(
  Harness(f"_inshape={jtu.format_shape_dtype_string(arg_shape, dtype)}_pads={pads}",
          lax.pad,
          [RandArg(arg_shape, dtype), np.array(0, dtype), StaticArg(pads)],
          rng_factory=jtu.rand_small,
          arg_shape=arg_shape, dtype=dtype, pads=pads)
  for arg_shape in [(2, 3)]
  for dtype in jtu.dtypes.all
  for pads in [
    [(0, 0, 0), (0, 0, 0)],  # no padding
    [(1, 1, 0), (2, 2, 0)],  # only positive edge padding
    [(1, 2, 1), (0, 1, 0)],  # edge padding and interior padding
    [(0, 0, 0), (-1, -1, 0)],  # negative padding
    [(0, 0, 0), (-2, -2, 4)],  # add big dilation then remove from edges
    [(0, 0, 0), (-2, -3, 1)],  # remove everything in one dimension
  ]
)

def _make_select_harness(name, *, shape_pred=(2, 3), shape_args=(2, 3),
                         dtype=np.float32):
  return Harness(f"{name}_shapepred={jtu.format_shape_dtype_string(shape_pred, np.bool_)}_shapeargs={jtu.format_shape_dtype_string(shape_args, dtype)}",
                 lax.select,
                 [RandArg(shape_pred, np.bool_), RandArg(shape_args, dtype),
                  RandArg(shape_args, dtype)],
                 shape_pred=shape_pred,
                 shape_args=shape_args,
                 dtype=dtype)

lax_select = tuple( # Validate dtypes
  _make_select_harness("dtypes", dtype=dtype)
  for dtype in jtu.dtypes.all
) + tuple( # Validate shapes
  _make_select_harness("shapes", shape_pred=shape_pred, shape_args=shape_args)
  for shape_pred, shape_args in [
    ((), (18,)), # scalar pred
  ]
)

def _make_transpose_harness(name, *, shape=(2, 3), permutation=(1, 0),
                            dtype=np.float32):
  return Harness(f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_permutation={permutation}".replace(' ', ''),
                 lambda x: lax.transpose_p.bind(x, permutation=permutation),
                 [RandArg(shape, dtype)],
                 shape=shape,
                 dtype=dtype,
                 permutation=permutation)

lax_transpose = tuple( # Validate dtypes
  _make_transpose_harness("dtypes", dtype=dtype)
  for dtype in jtu.dtypes.all
) + tuple( # Validate permutations
  _make_transpose_harness("permutations", shape=shape, permutation=permutation)
  for shape, permutation in [
    ((2, 3, 4), (0, 1, 2)), # identity
    ((2, 3, 4), (1, 2, 0)), # transposition
  ]
)

def _make_cumreduce_harness(name, *, f_jax=lax_control_flow.cummin,
                            shape=(8, 9), dtype=np.float32,
                            axis=0, reverse=False):
  return Harness(f"{name}_f={f_jax.__name__}_shape={jtu.format_shape_dtype_string(shape, dtype)}_axis={axis}_reverse={reverse}",
                 f_jax,
                 [RandArg(shape, dtype), StaticArg(axis), StaticArg(reverse)],
                 f_jax=f_jax,
                 shape=shape,
                 dtype=dtype,
                 axis=axis,
                 reverse=reverse)

lax_control_flow_cumreduce = tuple( # Validate dtypes for each function
  _make_cumreduce_harness("dtype_by_fun", dtype=dtype, f_jax=f_jax)
  for f_jax in [
    lax_control_flow.cummin,
    lax_control_flow.cummax,
    lax_control_flow.cumsum,
    lax_control_flow.cumprod
  ]
  for dtype in [dtype for dtype in jtu.dtypes.all if dtype != np.bool_]
) + tuple( # Validate axis for each function
  _make_cumreduce_harness("axis_by_fun", axis=axis, f_jax=f_jax, shape=shape)
  for shape in [(8, 9)]
  for f_jax in [
    lax_control_flow.cummin,
    lax_control_flow.cummax,
    lax_control_flow.cumsum,
    lax_control_flow.cumprod
  ]
  for axis in range(len(shape))
) + tuple( # Validate reverse for each function
  _make_cumreduce_harness("reverse", reverse=reverse, f_jax=f_jax)
  for f_jax in [
    lax_control_flow.cummin,
    lax_control_flow.cummax,
    lax_control_flow.cumsum,
    lax_control_flow.cumprod
  ]
  for reverse in [True]
)

def _make_top_k_harness(name, *, operand=None, shape=(5, 3), dtype=np.float32,
                        k=2):
  if operand is None:
    operand = RandArg(shape, dtype)
  return Harness(f"{name}_inshape={jtu.format_shape_dtype_string(operand.shape, operand.dtype)}_k={k}",
                 lax.top_k,
                 [operand, StaticArg(k)],
                 shape=operand.shape,
                 dtype=operand.dtype,
                 k=k)

lax_top_k = tuple( # Validate dtypes
  _make_top_k_harness("dtypes", dtype=dtype)
  for dtype in jtu.dtypes.all
) + tuple( # Validate k
  _make_top_k_harness("k", k=k)
  for k in [-2]
) + tuple( # Validate implicit properties of the sort
  _make_top_k_harness(name, operand=operand, k=k)
  for name, operand, k in [
      ("stability", np.array([5, 7, 5, 8, 8, 5], dtype=np.int32), 3),
      ("sort_inf_nan", np.array([+np.inf, np.nan, -np.nan, -np.inf, 3],
                                dtype=np.float32), 5)
  ]
)

def _make_sort_harness(name, *, operands=None, shape=(5, 7), dtype=np.float32,
                       dimension=0, is_stable=False, nb_arrays=1):
  if operands is None:
    operands = [RandArg(shape, dtype) for _ in range(nb_arrays)]
  return Harness(f"{name}_nbarrays={nb_arrays}_shape={jtu.format_shape_dtype_string(operands[0].shape, operands[0].dtype)}_axis={dimension}_isstable={is_stable}",
                 lambda *args: lax.sort_p.bind(*args[:-2], dimension=args[-2],
                                               is_stable=args[-1], num_keys=1),
                 [*operands, StaticArg(dimension), StaticArg(is_stable)],
                 shape=operands[0].shape,
                 dimension=dimension,
                 dtype=operands[0].dtype,
                 is_stable=is_stable,
                 nb_arrays=nb_arrays)

lax_sort = tuple( # Validate dtypes
  _make_sort_harness("dtypes", dtype=dtype)
  for dtype in jtu.dtypes.all
) + tuple( # Validate dimensions
  [_make_sort_harness("dimensions", dimension=1)]
) + tuple( # Validate stable sort
  [_make_sort_harness("is_stable", is_stable=True)]
) + tuple( # Potential edge cases
  _make_sort_harness("edge_cases", operands=operands, dimension=dimension)
  for operands, dimension in [
      ([np.array([+np.inf, np.nan, -np.nan, -np.inf, 2], dtype=np.float32)], -1)
  ]
) + tuple( # Validate multiple arrays
  _make_sort_harness("multiple_arrays", nb_arrays=nb_arrays, dtype=dtype)
  for nb_arrays, dtype in [
    (2, np.float32), # equivalent to sort_key_val
    (2, np.bool_),   # unsupported
    (3, np.float32), # unsupported
  ]
)

lax_linalg_cholesky = tuple(
  Harness(f"_shape={jtu.format_shape_dtype_string(shape, dtype)}",
          lambda *args: lax.linalg.cholesky_p.bind(*args),
          [RandArg(shape, dtype)],
          shape=shape,
          dtype=dtype)
  for dtype in jtu.dtypes.all_inexact
  for shape in [(1, 1), (4, 4), (2, 5, 5), (200, 200), (1000, 0, 0)]
)

lax_linalg_qr = tuple(
  Harness(f"multi_array_shape={jtu.format_shape_dtype_string(shape, dtype)}_fullmatrices={full_matrices}",
          lax.linalg.qr,
          [RandArg(shape, dtype), StaticArg(full_matrices)],
          shape=shape,
          dtype=dtype,
          full_matrices=full_matrices)
  for dtype in jtu.dtypes.all_floating  + jtu.dtypes.complex
  for shape in [(1, 1), (3, 3), (3, 4), (2, 10, 5), (2, 200, 100)]
  for full_matrices in [False, True]
)

def _make_fft_harness(name, *, shape=(14, 15, 16, 17), dtype=np.float32,
                      fft_type=xla_client.FftType.FFT, fft_lengths=(17,)):
  def _fft_rng_factory(dtype):
    _all_integers = (jtu.dtypes.all_integer + jtu.dtypes.all_unsigned +
        jtu.dtypes.boolean)
    # For integer types, use small values to keep the errors small
    if dtype in _all_integers:
      return jtu.rand_small
    else:
      return jtu.rand_default

  return Harness(f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_ffttype={fft_type}_fftlengths={fft_lengths}",
                 lambda *args: lax.fft_p.bind(args[0], fft_type=args[1],
                                              fft_lengths=args[2]),
                 [RandArg(shape, dtype), StaticArg(fft_type),
                  StaticArg(fft_lengths)],
                 rng_factory=_fft_rng_factory(dtype),
                 shape=shape,
                 dtype=dtype,
                 fft_type=fft_type,
                 fft_lengths=fft_lengths)

lax_fft = tuple( # Validate dtypes per FFT type
  _make_fft_harness("dtypes", shape=shape, dtype=dtype, fft_type=fft_type,
                    fft_lengths=fft_lengths)
  for shape in [(14, 15, 16, 17)]
  # FFT, IFFT, RFFT, IRFFT
  for fft_type in list(map(xla_client.FftType, [0, 1, 2, 3]))
  for dtype in (jtu.dtypes.floating if fft_type == xla_client.FftType.RFFT
                else jtu.dtypes.complex)
  for fft_lengths in [
    (shape[-1],) if fft_type != xla_client.FftType.IRFFT else
    ((shape[-1] - 1) * 2,)
  ]
) + tuple( # Validate dimensions per FFT type
  _make_fft_harness("dims", shape=shape, fft_type=fft_type,
                    fft_lengths=fft_lengths, dtype=dtype)
  for shape in [(14, 15, 16, 17)]
  for dims in [1, 2, 3]
  for fft_type in list(map(xla_client.FftType, [0, 1, 2, 3]))
  for dtype in [np.float32 if fft_type == xla_client.FftType.RFFT
                else np.complex64]
  for fft_lengths in [
    shape[-dims:] if fft_type != xla_client.FftType.IRFFT else
    shape[-dims:-1] + ((shape[-1] - 1) * 2,)
  ]
)

lax_linalg_svd = tuple(
  Harness(f"shape={jtu.format_shape_dtype_string(shape, dtype)}_fullmatrices={full_matrices}_computeuv={compute_uv}",
          lambda *args: lax.linalg.svd_p.bind(args[0], full_matrices=args[1],
                                              compute_uv=args[2]),
          [RandArg(shape, dtype), StaticArg(full_matrices), StaticArg(compute_uv)],
          shape=shape,
          dtype=dtype,
          full_matrices=full_matrices,
          compute_uv=compute_uv)
  for dtype in jtu.dtypes.all_floating + jtu.dtypes.complex
  for shape in [(2, 2), (2, 7), (29, 29), (2, 3, 53), (2, 3, 29, 7)]
  for full_matrices in [False, True]
  for compute_uv in [False, True]
)

lax_linalg_eig = tuple(
  Harness(f"_shape={jtu.format_shape_dtype_string(shape, dtype)}_computelefteigenvectors={compute_left_eigenvectors}_computerighteigenvectors={compute_right_eigenvectors}",
          lax.linalg.eig,
          [RandArg(shape, dtype), StaticArg(compute_left_eigenvectors),
           StaticArg(compute_right_eigenvectors)],
          shape=shape,
          dtype=dtype,
          compute_left_eigenvectors=compute_left_eigenvectors,
          compute_right_eigenvectors=compute_right_eigenvectors)
  for dtype in jtu.dtypes.all_inexact
  for shape in [(0, 0), (5, 5), (2, 6, 6)]
  for compute_left_eigenvectors in [False, True]
  for compute_right_eigenvectors in [False, True]
)

lax_linalg_eigh = tuple(
  Harness(f"_shape={jtu.format_shape_dtype_string(shape, dtype)}_lower={lower}",
          lax.linalg.eigh,
          [RandArg(shape, dtype), StaticArg(lower), StaticArg(False)],
          shape=shape,
          dtype=dtype,
          lower=lower)
  for dtype in jtu.dtypes.all_inexact
  for shape in [(0, 0), (50, 50), (2, 20, 20)]
  for lower in [False, True]
  # Filter out cases where implementation is missing in JAX
  if dtype != np.float16
)

lax_linalg_lu = tuple(
  Harness(f"_shape={jtu.format_shape_dtype_string(shape, dtype)}",
          lax.linalg.lu,
          [RandArg(shape, dtype)],
          shape=shape,
          dtype=dtype)
  for dtype in jtu.dtypes.all_inexact
  for shape in [
    (5, 5),    # square
    (3, 5, 5), # batched
    (3, 5),    # non-square
  ]
)

def _make_triangular_solve_harness(name, *, left_side=True, lower=False,
                                   ab_shapes=((4, 4), (4, 1)), dtype=np.float32,
                                   transpose_a=False, conjugate_a=False,
                                   unit_diagonal=False):
  a_shape, b_shape = ab_shapes
  f_lax = lambda a, b: (lax.linalg.triangular_solve_p.bind(
      a, b, left_side=left_side, lower=lower, transpose_a=transpose_a,
      conjugate_a=conjugate_a, unit_diagonal=unit_diagonal))

  return Harness(f"_{name}_a={jtu.format_shape_dtype_string(a_shape, dtype)}_b={jtu.format_shape_dtype_string(b_shape, dtype)}_leftside={left_side}_lower={lower}_transposea={transpose_a}_conjugatea={conjugate_a}_unitdiagonal={unit_diagonal}",
                 f_lax,
                 [RandArg(a_shape, dtype), RandArg(b_shape, dtype)],
                 dtype=dtype,
                 a_shape=a_shape,
                 b_shape=b_shape,
                 left_side=left_side,
                 lower=lower,
                 tranpose_a=transpose_a,
                 conjugate_a=conjugate_a,
                 unit_diagonal=unit_diagonal)

lax_linalg_triangular_solve = tuple( # Validate dtypes
  # This first harness runs the tests for all dtypes using default values for
  # all the other parameters, except unit_diagonal (to ensure that
  # tf.linalg.set_diag works reliably for all dtypes). Variations of other
  # parameters can thus safely skip testing their corresponding default value.
  # Note that this validates solving on the left.
  _make_triangular_solve_harness("dtypes", dtype=dtype,
                                 unit_diagonal=unit_diagonal)
  for dtype in jtu.dtypes.all_inexact
  for unit_diagonal in [False, True]
) + tuple( # Validate shapes when solving on the right
  _make_triangular_solve_harness("shapes_right", ab_shapes=ab_shapes,
                                 left_side=False)
  for ab_shapes in [
    ((4, 4), (1, 4)),        # standard
    ((2, 8, 8), (2, 10, 8)), # batched
  ]
) + tuple( # Validate transformations of a complex matrix
  _make_triangular_solve_harness("complex_transformations", dtype=np.complex64,
                                 lower=lower, transpose_a=transpose_a,
                                 conjugate_a=conjugate_a)
  for lower in [False, True]
  for transpose_a in [False, True]
  for conjugate_a in [False, True]
) + tuple( # Validate transformations of a real matrix
  _make_triangular_solve_harness("real_transformations", dtype=np.float32,
                                 lower=lower, transpose_a=transpose_a)
  for lower in [False, True]
  for transpose_a in [False, True]
  # conjugate_a is irrelevant for real dtypes, and is thus omitted
)

def _make_linear_solve_harnesses():
  def linear_solve(a, b, solve, transpose_solve=None, symmetric=False):
    matvec = partial(lax.dot, a, precision=lax.Precision.HIGHEST)
    return lax.custom_linear_solve(matvec, b, solve, transpose_solve, symmetric)

  def explicit_jacobian_solve(matvec, b):
    return lax.stop_gradient(jnp.linalg.solve(jax.api.jacobian(matvec)(b), b))

  def _make_harness(name, *, shape=(4, 4), dtype=np.float32, symmetric=False,
                    solvers=(explicit_jacobian_solve, explicit_jacobian_solve)):
    solve, transpose_solve = solvers
    transpose_solve_name = transpose_solve.__name__ if transpose_solve else None
    return Harness(f"_{name}_a={jtu.format_shape_dtype_string(shape, dtype)}_b={jtu.format_shape_dtype_string(shape[:-1], dtype)}_solve={solve.__name__}_transposesolve={transpose_solve_name}_symmetric={symmetric}",
                   linear_solve,
                   [RandArg(shape, dtype), RandArg(shape[:-1], dtype),
                    StaticArg(solve), StaticArg(transpose_solve),
                    StaticArg(symmetric)],
                   shape=shape,
                   dtype=dtype,
                   solve=solve,
                   transpose_solve=transpose_solve,
                   symmetric=symmetric)

  return tuple( # Validate dtypes
    _make_harness("dtypes", dtype=dtype)
    for dtype in
      jtu.dtypes.all_floating if not dtype in [np.float16, dtypes.bfloat16]
  ) + tuple( # Validate symmetricity
    [_make_harness("symmetric", symmetric=True)]
  ) + tuple( # Validate removing transpose_solve
    [_make_harness("transpose_solve", solvers=(explicit_jacobian_solve, None))]
  )

lax_linear_solve = _make_linear_solve_harnesses()

lax_slice = tuple(
  Harness(f"_shape={shape}_start_indices={start_indices}_limit_indices={limit_indices}_strides={strides}",  # type: ignore
          lax.slice,
          [RandArg(shape, dtype),  # type: ignore
           StaticArg(start_indices),  # type: ignore
           StaticArg(limit_indices),  # type: ignore
           StaticArg(strides)],  # type: ignore
          shape=shape,  # type: ignore
          start_indices=start_indices,  # type: ignore
          limit_indices=limit_indices)  # type: ignore
  for shape, start_indices, limit_indices, strides in [
    ((3,), (1,), (2,), None),
    ((7,), (4,), (7,), None),
    ((5,), (1,), (5,), (2,)),
    ((8,), (1,), (6,), (2,)),
    ((5, 3), (1, 1), (3, 2), None),
    ((5, 3), (1, 1), (3, 1), None),
    ((7, 5, 3), (4, 0, 1), (7, 1, 3), None),
    ((5, 3), (1, 1), (2, 1), (1, 1)),
    ((5, 3), (1, 1), (5, 3), (2, 1)),
    # out-of-bounds cases
    ((5,), (-1,), (0,), None),
    ((5,), (-1,), (1,), None),
    ((5,), (-4,), (-2,), None),
    ((5,), (-5,), (-2,), None),
    ((5,), (-6,), (-5,), None),
    ((5,), (-10,), (-9,), None),
    ((5,), (-100,), (-99,), None),
    ((5,), (5,), (6,), None),
    ((5,), (10,), (11,), None),
    ((5,), (0,), (100,), None),
    ((5,), (3,), (6,), None)
  ]
  for dtype in [np.float32]
)

def _make_conj_harness(name, *, shape=(3, 4), dtype=np.float32, **kwargs):
  return Harness(f"{name}_operand={jtu.format_shape_dtype_string(shape, dtype)}_kwargs={kwargs}".replace(" ", ""),
                 lambda x: lax.conj_p.bind(x, **kwargs),
                 [RandArg(shape, dtype)],
                 shape=shape,
                 dtype=dtype,
                 **kwargs)

lax_conj = tuple( # Validate dtypes
  _make_conj_harness("dtypes", dtype=dtype)
  for dtype in jtu.dtypes.floating + jtu.dtypes.complex
) + tuple( # Validate kwargs
  _make_conj_harness("kwargs", **kwargs)
  for kwargs in [
    { "_input_dtype": np.float32 },             # expected kwarg for ad
  ]
)

# Use lax_slice, but (a) make the start_indices dynamic arg, and (b) no strides.
lax_dynamic_slice = [
  Harness(harness.name,
          lax.dynamic_slice,
          [harness.arg_descriptors[0],
           np.array(list(start_indices)),
           StaticArg(tuple(map(operator.sub, limit_indices, start_indices)))],
          **harness.params)
  for harness in lax_slice
  for start_indices in [harness.params["start_indices"]]
  for limit_indices in [harness.params["limit_indices"]]
]

lax_dynamic_update_slice = tuple(
  Harness((f"_operand={jtu.format_shape_dtype_string(shape, dtype)}"  # type: ignore
           f"_update={jtu.format_shape_dtype_string(update_shape, update_dtype)}"
           f"_start_indices={start_indices}"),
          lax.dynamic_update_slice,
          [RandArg(shape, dtype),  # type: ignore
           RandArg(update_shape, update_dtype),  # type: ignore
           np.array(start_indices)],  # type: ignore
          shape=shape,  # type: ignore
          start_indices=start_indices,  # type: ignore
          update_shape=update_shape)  # type: ignore
  for shape, start_indices, update_shape in [
    ((3,), (1,), (1,)),
    ((5, 3), (1, 1), (3, 1)),
    ((7, 5, 3), (4, 1, 0), (2, 0, 1)),
    ((3,), (-1,), (1,)),  # out-of-bounds
    ((3,), (10,), (1,)),  # out-of-bounds
    ((3,), (10,), (4,)),  # out-of-bounds shape too big
    ((3,), (10,), (2,)),  # out-of-bounds
  ]
  for dtype, update_dtype in [
    (np.float32, np.float32),
    (np.float64, np.float64)
  ])

lax_squeeze = tuple(
  Harness(f"_inshape={jtu.format_shape_dtype_string(arg_shape, dtype)}_dimensions={dimensions}",  # type: ignore
          lax.squeeze,
          [RandArg(arg_shape, dtype), StaticArg(dimensions)],  # type: ignore[has-type]
          arg_shape=arg_shape, dtype=dtype, dimensions=dimensions)  # type: ignore[has-type]
  for arg_shape, dimensions in [
    ((1,), (0,)),
    ((1,), (-1,)),
    ((2, 1, 4), (1,)),
    ((2, 1, 4), (-2,)),
    ((2, 1, 3, 1), (1,)),
    ((2, 1, 3, 1), (1, 3)),
    ((2, 1, 3, 1), (3,)),
    ((2, 1, 3, 1), (1, -1)),
  ]
  for dtype in [np.float32]
)

shift_inputs = [
  (arg, dtype, shift_amount)
  for dtype in jtu.dtypes.all_unsigned + jtu.dtypes.all_integer
  for arg in [
    np.array([-250, -1, 0, 1, 250], dtype=dtype),
  ]
  for shift_amount in [-8, -1, 0, 1, 3, 7, 8, 16, 32, 64]
]

lax_shift_left = tuple(
  Harness(f"_dtype={dtype.__name__}_shift_amount={shift_amount}",  # type: ignore
          lax.shift_left,
          [arg, StaticArg(np.array([shift_amount], dtype=dtype))])
  for arg, dtype, shift_amount in shift_inputs
)

lax_shift_right_logical = tuple(
  Harness(f"_dtype={dtype.__name__}_shift_amount={shift_amount}",  # type: ignore
          lax.shift_right_logical,
          [arg, dtype(shift_amount)],
          dtype=dtype)
  for arg, dtype, shift_amount in shift_inputs
)

lax_shift_right_arithmetic = tuple(
  Harness(f"_dtype={dtype.__name__}_shift_amount={shift_amount}",  # type: ignore
          lax.shift_right_arithmetic,
          [arg, StaticArg(np.array([shift_amount], dtype=dtype))],
          dtype=dtype)
  for arg, dtype, shift_amount in shift_inputs
)

def _make_select_and_scatter_add_harness(
    name, *, shape=(2, 4, 6), dtype=np.float32, select_prim=lax.ge_p,
    window_dimensions=(2, 2, 2), window_strides=(1, 1, 1),
    padding=((0, 0), (0, 0), (0, 0)), nb_inactive_dims=0):
  ones = (1,) * len(shape)
  cotangent_shape = jax.api.eval_shape(
      lambda x: lax._select_and_gather_add(x, x, lax.ge_p, window_dimensions,
                                           window_strides, padding, ones, ones),
      np.ones(shape, dtype)).shape
  return Harness(f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_selectprim={select_prim}_windowdimensions={window_dimensions}_windowstrides={window_strides}_padding={padding}",
                 lax._select_and_scatter_add,
                 [RandArg(cotangent_shape, dtype), RandArg(shape, dtype),
                  StaticArg(select_prim), StaticArg(window_dimensions),
                  StaticArg(window_strides), StaticArg(padding)],
                 shape=shape,
                 dtype=dtype,
                 select_prim=select_prim,
                 window_dimensions=window_dimensions,
                 window_strides=window_strides,
                 padding=padding,
                 # JAX can only run select_and_scatter_add on TPU when 2
                 # or more dimensions are inactive
                 run_on_tpu=(nb_inactive_dims >= 2))

lax_select_and_scatter_add = tuple( # Validate dtypes
  _make_select_and_scatter_add_harness("dtypes", dtype=dtype)
  for dtype in set(jtu.dtypes.all) - set([np.complex64, np.complex128])
) + tuple( # Validate different reduction primitives
  _make_select_and_scatter_add_harness("select_prim", select_prim=select_prim)
  for select_prim in [lax.le_p]
) + tuple( # Validate padding
  _make_select_and_scatter_add_harness("padding", padding=padding)
  for padding in [
    # TODO(bchetioui): commented out the test based on
    # https://github.com/google/jax/issues/4690
    #((1, 2), (2, 3), (3, 4)) # non-zero padding
    ((1, 1), (1, 1), (1, 1)) # non-zero padding
  ]
) + tuple( # Validate window_dimensions
  _make_select_and_scatter_add_harness("window_dimensions",
                                       window_dimensions=window_dimensions)
  for window_dimensions in [
    (1, 2, 3) # uneven dimensions
  ]
) + tuple( # Validate window_strides
  _make_select_and_scatter_add_harness("window_strides",
                                       window_strides=window_strides)
  for window_strides in [
    (1, 2, 3) # smaller than/same as/bigger than corresponding window dimension
  ]
) + tuple( # Validate dtypes on TPU
  _make_select_and_scatter_add_harness("tpu_dtypes", dtype=dtype,
                                       nb_inactive_dims=nb_inactive_dims,
                                       window_strides=window_strides,
                                       window_dimensions=window_dimensions)
  for dtype in set(jtu.dtypes.all) - set([np.bool_, np.complex64, np.complex128,
                                          np.int8, np.uint8])
  for window_strides, window_dimensions, nb_inactive_dims in [
    ((1, 2, 1), (1, 3, 1), 2)
  ]
)

def _make_select_and_gather_add_harness(
    name, *, shape=(4, 6), dtype=np.float32, select_prim=lax.le_p,
    padding='VALID', window_dimensions=(2, 2), window_strides=(1, 1),
    base_dilation=(1, 1), window_dilation=(1, 1)):
  if isinstance(padding, str):
    padding = tuple(lax.padtype_to_pads(shape, window_dimensions,
                                        window_strides, padding))
  return Harness(f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_selectprim={select_prim}_windowdimensions={window_dimensions}_windowstrides={window_strides}_padding={padding}_basedilation={base_dilation}_windowdilation={window_dilation}",
                 lax._select_and_gather_add,
                 [RandArg(shape, dtype), RandArg(shape, dtype),
                  StaticArg(select_prim), StaticArg(window_dimensions),
                  StaticArg(window_strides), StaticArg(padding),
                  StaticArg(base_dilation), StaticArg(window_dilation)],
                 shape=shape,
                 dtype=dtype,
                 window_dimensions=window_dimensions,
                 window_strides=window_strides,
                 padding=padding,
                 base_dilation=base_dilation,
                 window_dilation=window_dilation)

lax_select_and_gather_add = tuple( # Validate dtypes
  _make_select_and_gather_add_harness("dtypes", dtype=dtype)
  for dtype in jtu.dtypes.all_floating
) + tuple( # Validate selection primitives
  [_make_select_and_gather_add_harness("select_prim", select_prim=lax.ge_p)]
) + tuple( # Validate window dimensions
  _make_select_and_gather_add_harness("window_dimensions",
                                      window_dimensions=window_dimensions)
  for window_dimensions in [(2, 3)]
) + tuple( # Validate window strides
  _make_select_and_gather_add_harness("window_strides",
                                      window_strides=window_strides)
  for window_strides in [(2, 3)]
) + tuple( # Validate padding
  _make_select_and_gather_add_harness("padding", padding=padding)
  for padding in ['SAME']
) + tuple( # Validate dilations
  _make_select_and_gather_add_harness("dilations", base_dilation=base_dilation,
                                      window_dilation=window_dilation)
  for base_dilation, window_dilation in [
    ((2, 3), (1, 1)), # base dilation, no window dilation
    ((1, 1), (2, 3)), # no base dilation, window dilation
    ((2, 3), (3, 2))  # base dilation, window dilation
  ]
)

def _make_reduce_window_harness(name, *, shape=(4, 6), base_dilation=(1, 1),
                                computation=lax.add, window_dimensions=(2, 2),
                                window_dilation=(1, 1), init_value=0,
                                window_strides=(1, 1), dtype=np.float32,
                                padding=((0, 0), (0, 0))):
  return Harness(f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_initvalue={init_value}_computation={computation.__name__}_windowdimensions={window_dimensions}_windowstrides={window_strides}_padding={padding}_basedilation={base_dilation}_windowdilation={window_dilation}".replace(' ', ''),
                 lax.reduce_window,
                 [RandArg(shape, dtype),
                  StaticArg(np.array(init_value, dtype=dtype)),  # Must be static to trigger the picking of the reducers
                  StaticArg(computation), StaticArg(window_dimensions),
                  StaticArg(window_strides), StaticArg(padding),
                  StaticArg(base_dilation), StaticArg(window_dilation)],
                 shape=shape,
                 dtype=dtype,
                 init_value=np.array(init_value, dtype=dtype),
                 computation=computation,
                 window_dimensions=window_dimensions,
                 window_strides=window_strides,
                 padding=padding,
                 base_dilation=base_dilation,
                 window_dilation=window_dilation)

lax_reduce_window = tuple( # Validate dtypes across all execution paths
  # This first harness runs the tests for all dtypes using default values for
  # the other parameters (outside of computation and its init_value), through
  # several execution paths. Variations of other parameters can thus safely
  # skip testing their corresponding default value.
  _make_reduce_window_harness("dtypes", dtype=dtype, computation=computation,
                              init_value=init_value)
  for dtype in jtu.dtypes.all
  for computation, init_value in [
    (lax.min, _get_min_identity(dtype)), # path through reduce_window_min
    (lax.max, _get_max_identity(dtype)), # path through TF reduce_window_max
    (lax.max, 1), # path through reduce_window
  ] + ([
    (lax.add, 0), # path_through reduce_window_sum
    (lax.mul, 1), # path through reduce_window
  ] if dtype != jnp.bool_ else [])
) + tuple( # Validate window_dimensions
  _make_reduce_window_harness("window_dimensions",
                              window_dimensions=window_dimensions)
  for window_dimensions in [(1, 1)]
) + tuple( # Validate window_strides
  _make_reduce_window_harness("window_strides", window_strides=window_strides)
  for window_strides in [(1, 2)]
) + tuple( # Validate padding
  [_make_reduce_window_harness("padding", padding=((1, 2), (0, 3)))]
) + tuple( # Validate base_dilation
  _make_reduce_window_harness("base_dilation", base_dilation=base_dilation)
  for base_dilation in [(1, 2)]
) + tuple( # Validate window_dilation
  _make_reduce_window_harness("window_dilation",
                              window_dilation=window_dilation)
  for window_dilation in [(1, 2)]
) + tuple( # Validate squeezing behavior and dimensions in tf.nn.max_pool
  _make_reduce_window_harness("squeeze_dim", computation=lax.max, shape=shape,
                              dtype=np.float32, init_value=-np.inf,
                              base_dilation=tuple([1] * len(shape)),
                              window_dilation=tuple([1] * len(shape)),
                              padding=tuple([(0, 0)] * len(shape)),
                              window_strides=tuple([1] * len(shape)),
                              window_dimensions=window_dimensions)
  for shape, window_dimensions in [
    ((2,), (2,)),           # 1 spatial dimension, left and right squeeze
    ((2, 1), (2, 1)),       # 1 spatial dimension, left squeeze
    ((1, 2), (1, 2)),       # 1 spatial dimension, right squeeze
    ((1, 2, 1), (1, 2, 1)), # 1 spatial dimension no squeeze
    ((2, 4), (2, 2)),       # 2 spatial dimensions, left and right squeeze
    ((2, 4, 3), (2, 2, 2)), # 3 spatial dimensions, left and right squeeze
    ((1, 4, 3, 2, 1), (1, 2, 2, 2, 1)) # 3 spatial dimensions, no squeeze
  ]
)

random_gamma = tuple(
  Harness(f"_shape={jtu.format_shape_dtype_string(shape, dtype)}",
          jax.jit(jax.random.gamma),
          [np.array([42, 43], dtype=np.uint32), RandArg(shape, dtype)])
  for shape in ((), (3,))
  for dtype in (np.float32, np.float64)
)

random_split = tuple(
  Harness(f"_i={key_i}",
          jax.jit(lambda key: jax.random.split(key, 2)),
          [key])
  for key_i, key in enumerate([np.array([0, 0], dtype=np.uint32),
                               np.array([42, 43], dtype=np.uint32),
                               np.array([0xFFFFFFFF, 0], dtype=np.uint32),
                               np.array([0, 0xFFFFFFFF], dtype=np.uint32),
                               np.array([0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32)])
)

def _make_clamp_harness(name, *, min_shape=(), operand_shape=(2, 3),
                        max_shape=(), dtype=np.float32, min_max=None):
  min_arr, max_arr = (min_max if min_max is not None else
                      [RandArg(min_shape, dtype), RandArg(max_shape, dtype)])
  return Harness(f"{name}_min={jtu.format_shape_dtype_string(min_arr.shape, min_arr.dtype)}_operand={jtu.format_shape_dtype_string(operand_shape, dtype)}_max={jtu.format_shape_dtype_string(max_arr.shape, max_arr.dtype)}",
                 lax.clamp,
                 [min_arr, RandArg(operand_shape, dtype), max_arr],
                 min_shape=min_arr.shape,
                 operand_shape=operand_shape,
                 max_shape=max_arr.shape,
                 dtype=dtype)

lax_clamp = tuple( # Validate dtypes
  _make_clamp_harness("dtypes", dtype=dtype)
  for dtype in set(jtu.dtypes.all) - set(jtu.dtypes.complex + [np.bool_])
) + tuple( # Validate broadcasting of min/max arrays
  _make_clamp_harness("broadcasting", min_shape=min_shape, max_shape=max_shape,
                      operand_shape=operand_shape)
  for min_shape, operand_shape, max_shape in [
    ((), (2, 3), (2, 3)),     # no broadcasting for max
    ((2, 3), (2, 3), ()),     # no broadcasting for min
    ((2, 3), (2, 3), (2, 3)), # no broadcasting
  ]
) + tuple( # Validate clamping when minval > maxval, and when minval < maxval
  _make_clamp_harness(f"order={is_ordered}", min_max=(min_arr, max_arr),
                      dtype=np.float32)
  for is_ordered, min_arr, max_arr in [
    (False, np.array(4., dtype=np.float32), np.array(1., dtype=np.float32)),
    (True, np.array(1., dtype=np.float32), np.array(4., dtype=np.float32))
  ]
)

def _make_dot_general_harness(
    name, *, lhs_shape=(3, 4), rhs_shape=(4, 2), dtype=np.float32,
    precision=None, dimension_numbers=(((1,), (0,)), ((), ()))):
  return Harness(f"_{name}_lhs={jtu.format_shape_dtype_string(lhs_shape, dtype)}_rhs={jtu.format_shape_dtype_string(rhs_shape, dtype)}_dimensionnumbers={dimension_numbers}_precision={precision}".replace(' ', ''),
                 lax.dot_general,
                 [RandArg(lhs_shape, dtype), RandArg(rhs_shape, dtype),
                  StaticArg(dimension_numbers), StaticArg(precision)],
                 dtype=dtype,
                 lhs_shape=lhs_shape,
                 rhs_shape=rhs_shape,
                 dimension_numbers=dimension_numbers,
                 precision=precision)

# There are two execution paths in the conversion of dot_general. The main path
# uses tf.einsum, while special cases use tf.linalg.matmul. For that reason,
# the below tests are designed to perform the same checks on both execution
# paths.
lax_dot_general = tuple( # Validate dtypes and precision
  # This first harness runs the tests for all dtypes and precisions using
  # default values for all the other parameters. Variations of other parameters
  # can thus safely skip testing their corresponding default value.
  _make_dot_general_harness("dtypes_and_precision", precision=precision,
                            lhs_shape=lhs_shape, rhs_shape=rhs_shape,
                            dimension_numbers=dimension_numbers, dtype=dtype)
  for dtype in jtu.dtypes.all
  for precision in [None, lax.Precision.DEFAULT, lax.Precision.HIGH,
                    lax.Precision.HIGHEST]
  for lhs_shape, rhs_shape, dimension_numbers in [
    ((3, 4), (4, 2), (((1,), (0,)), ((), ()))),
    ((1, 3, 4), (1, 4, 3), (((2, 1), (1, 2)), ((0,), (0,))))
  ]
) + tuple( # Validate batch dimensions
  _make_dot_general_harness("batch_dimensions", lhs_shape=lhs_shape,
                            rhs_shape=rhs_shape,
                            dimension_numbers=dimension_numbers)
  for lhs_shape, rhs_shape, dimension_numbers in [
    # Unique pattern that can go through tf.linalg.matmul
    ((4, 4, 3, 3, 4), (4, 4, 3, 4, 2), (((4,), (3,)), ((0, 1, 2), (0, 1, 2)))),
    # Main path with out of order batch dimensions
    ((8, 4, 3, 3, 4), (4, 8, 3, 4, 2), (((4, 3), (3, 2)), ((0, 1), (1, 0))))
  ]
) + tuple( # Validate squeezing behavior for matmul path
  _make_dot_general_harness("squeeze", lhs_shape=lhs_shape, rhs_shape=rhs_shape,
                            dimension_numbers=dimension_numbers)
  for lhs_shape, rhs_shape, dimension_numbers in [
    ((4,), (4, 4), (((0,), (0,)), ((), ()))), # (1, 4) -> (4,)
    ((4, 4), (4,), (((1,), (0,)), ((), ()))), # (4, 1) -> (4,)
    ((4,), (4,), (((0,), (0,)), ((), ()))),   # (1, 1) -> ()
  ]
)

def _make_concatenate_harness(name, *, shapes=[(2, 3), (2, 3)], dimension=0,
                              dtype=np.float32):
  shapes_str = '_'.join(jtu.format_shape_dtype_string(s, dtype) for s in shapes)
  return Harness(f"{name}_shapes={shapes_str}_dimension={dimension}",
                 lambda *args: lax.concatenate_p.bind(*args,
                                                      dimension=dimension),
                 [RandArg(shape, dtype) for shape in shapes],
                 shapes=shapes,
                 dtype=dtype,
                 dimension=dimension)

lax_concatenate = tuple( # Validate dtypes
  _make_concatenate_harness("dtypes", dtype=dtype)
  for dtype in jtu.dtypes.all
) + tuple( # Validate dimension
  _make_concatenate_harness("dimension", dimension=dimension)
  for dimension in [
    1, # non-major axis
  ]
) + tuple( # Validate > 2 operands
  _make_concatenate_harness("nb_operands", shapes=shapes)
  for shapes in [
    [(2, 3, 4), (3, 3, 4), (4, 3, 4)], # 3 operands
  ]
)

def _make_conv_harness(name, *, lhs_shape=(2, 3, 9, 10), rhs_shape=(3, 3, 4, 5),
                       dtype=np.float32, window_strides=(1, 1), precision=None,
                       padding=((0, 0), (0, 0)), lhs_dilation=(1, 1),
                       rhs_dilation=(1, 1), feature_group_count=1,
                       dimension_numbers=("NCHW", "OIHW", "NCHW"),
                       batch_group_count=1, enable_xla=True):

  return Harness(f"_{name}_lhs={jtu.format_shape_dtype_string(lhs_shape, dtype)}_rhs={jtu.format_shape_dtype_string(rhs_shape, dtype)}_windowstrides={window_strides}_padding={padding}_lhsdilation={lhs_dilation}_rhsdilation={rhs_dilation}_dimensionnumbers={dimension_numbers}_featuregroupcount={feature_group_count}_batchgroupcount={batch_group_count}_precision={precision}_enablexla={enable_xla}".replace(' ', ''),
                 lax.conv_general_dilated,
                 [RandArg(lhs_shape, dtype), RandArg(rhs_shape, dtype),
                  StaticArg(window_strides), StaticArg(padding),
                  StaticArg(lhs_dilation), StaticArg(rhs_dilation),
                  StaticArg(dimension_numbers), StaticArg(feature_group_count),
                  StaticArg(batch_group_count), StaticArg(precision)],
                 lhs_shape=lhs_shape,
                 rhs_shape=rhs_shape,
                 dtype=dtype,
                 window_strides=window_strides,
                 padding=padding,
                 lhs_dilation=lhs_dilation,
                 rhs_dilation=rhs_dilation,
                 dimension_numbers=dimension_numbers,
                 feature_group_count=feature_group_count,
                 batch_group_count=batch_group_count,
                 precision=precision,
                 enable_xla=enable_xla)

lax_conv_general_dilated = tuple( # Validate dtypes and precision
  # This first harness runs the tests for all dtypes and precisions using
  # default values for all the other parameters. Variations of other parameters
  # can thus safely skip testing their corresponding default value.
  _make_conv_harness("dtype_precision", dtype=dtype, precision=precision)
  for dtype in jtu.dtypes.all_inexact
  for precision in [None, lax.Precision.DEFAULT, lax.Precision.HIGH,
                    lax.Precision.HIGHEST]
) + tuple( # Validate variations of feature_group_count and batch_group_count
  _make_conv_harness("group_counts", lhs_shape=lhs_shape, rhs_shape=rhs_shape,
                     feature_group_count=feature_group_count,
                     batch_group_count=batch_group_count)
  for batch_group_count, feature_group_count in [
      (1, 2), # feature_group_count != 1
      (2, 1), # batch_group_count != 1
  ]
  for lhs_shape, rhs_shape in [
      ((2 * batch_group_count, 3 * feature_group_count, 9, 10),
       (3 * feature_group_count * batch_group_count, 3, 4, 5))
  ]
) + tuple( # Validate variations of window_strides
  _make_conv_harness("window_strides", window_strides=window_strides)
  for window_strides in [
      (2, 3)  # custom window
  ]
) + tuple( # Validate variations of padding
  _make_conv_harness("padding", padding=padding)
  for padding in [
      ((1, 2), (0, 0)), # padding only one spatial axis
      ((1, 2), (2, 1))  # padding on both spatial axes
  ]
) + tuple( # Validate variations of dilations
  _make_conv_harness("dilations", lhs_dilation=lhs_dilation,
                     rhs_dilation=rhs_dilation)
  for lhs_dilation, rhs_dilation in [
      ((2, 2), (1, 1)), # dilation only on LHS (transposed)
      ((1, 1), (2, 3)), # dilation only on RHS (atrous)
      ((2, 3), (3, 2))  # dilation on both LHS and RHS (transposed & atrous)
  ]
) + tuple(
  _make_conv_harness("dimension_numbers", lhs_shape=lhs_shape,
                     rhs_shape=rhs_shape, dimension_numbers=dimension_numbers)
  # Dimension numbers and corresponding permutation
  for dimension_numbers, lhs_shape, rhs_shape in [
      (("NHWC", "HWIO", "NHWC"), (2, 9, 10, 3), (4, 5, 3, 3)), # TF default
      (("NCHW", "HWIO", "NHWC"), (2, 3, 9, 10), (4, 5, 3, 3)), # custom
  ]
) + tuple(
  _make_conv_harness("tf_conversion_path_1d", lhs_shape=lhs_shape,
                     padding=padding, rhs_shape=rhs_shape,
                     dimension_numbers=dimension_numbers, window_strides=(1,),
                     lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
                     enable_xla=enable_xla)
  for padding, lhs_dilation, rhs_dilation in [
    ("VALID", (1,), (1,)), # no dilation with "VALID" padding
    ("SAME",  (1,), (1,)), # no dilation with "SAME" padding
    ("VALID", (1,), (2,)), # dilation only on RHS with "VALID" padding
    ("SAME",  (1,), (2,)), # dilation only on RHS with "SAME" padding
    # TODO(bchetioui): LHS dilation with string padding can never be done using
    # TF convolution functions for now.
  ]
  for dimension_numbers, lhs_shape, rhs_shape in [
    (("NWC", "WIO", "NWC"), (1, 28, 1), (3, 1, 16)), # TF default
    # TODO(bchetioui): the NCW data format is not supported on CPU for TF
    # for now. That path is thus disabled to allow the code to use XLA instead.
  ]
  for enable_xla in [False, True]
) + tuple(
  _make_conv_harness("tf_conversion_path_2d", lhs_shape=lhs_shape,
                     padding=padding, rhs_shape=rhs_shape,
                     dimension_numbers=dimension_numbers, window_strides=(1, 1),
                     lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
                     enable_xla=enable_xla)
  for padding, lhs_dilation, rhs_dilation in [
    ("VALID", (1, 1), (1, 1)), # no dilation with "VALID" padding
    ("SAME",  (1, 1), (1, 1)), # no dilation with "SAME" padding
    ("VALID", (1, 1), (1, 2)), # dilation only on RHS with "VALID" padding
    ("SAME",  (1, 1), (1, 2)), # dilation only on RHS with "SAME" padding
    # TODO(bchetioui): LHS dilation with string padding can never be done using
    # TF convolution functions for now.
  ]
  for dimension_numbers, lhs_shape, rhs_shape in [
    (("NHWC", "HWIO", "NHWC"), (1, 28, 28, 1), (3, 3, 1, 16)), # TF default
    # TODO(bchetioui): the NCHW data format is not supported on CPU for TF
    # for now. That path is thus disabled to allow the code to use XLA instead.
  ]
  for enable_xla in [False, True]
) + tuple(
  _make_conv_harness("tf_conversion_path_3d", lhs_shape=lhs_shape,
                     padding=padding, rhs_shape=rhs_shape,
                     dimension_numbers=dimension_numbers,
                     window_strides=(1, 1, 1), lhs_dilation=lhs_dilation,
                     rhs_dilation=rhs_dilation, enable_xla=enable_xla)
  for padding, lhs_dilation, rhs_dilation in [
    ("VALID", (1, 1, 1), (1, 1, 1)), # no dilation with "VALID" padding
    ("SAME",  (1, 1, 1), (1, 1, 1)), # no dilation with "SAME" padding
    ("VALID", (1, 1, 1), (1, 1, 2)), # dilation only on RHS with "VALID" padding
    ("SAME",  (1, 1, 1), (1, 1, 2)), # dilation only on RHS with "SAME" padding
    # TODO(bchetioui): LHS dilation with string padding can never be done using
    # TF convolution functions for now.
  ]
  for dimension_numbers, lhs_shape, rhs_shape in [
    # TF default
    (("NDHWC", "DHWIO", "NDHWC"), (1, 4, 28, 28, 1), (2, 3, 3, 1, 16)),
    # TODO(bchetioui): the NCDHW data format is not supported on CPU for TF
    # for now. That path is thus disabled to allow the code to use XLA instead.
  ]
  for enable_xla in [False, True]
)
