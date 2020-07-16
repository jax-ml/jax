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

from absl import testing
from jax import config
from jax import test_util as jtu
from jax import lax
from jax import lax_linalg
from jax import numpy as jnp

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


lax_betainc = tuple(
  Harness(f"_{jtu.dtype_str(dtype)}",
           lax.betainc,
           [arg1, arg2, arg3],
           dtype=dtype)
   for dtype in jtu.dtypes.all_floating
   for arg1, arg2, arg3 in [
     (np.array([-1.6, -1.4, -1.0, 0.0, 0.1, 0.3, 1, 1.4, 1.6], dtype=dtype),
      np.array([-1.6, 1.4, 1.0, 0.0, 0.2, 0.1, 1, 1.4, -1.6], dtype=dtype),
      np.array([1.0, -1.0, 2.0, 1.0, 0.3, 0.3, -1.0, 2.4, 1.6],
               dtype=np.float32))
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


lax_pad = tuple(
  Harness(f"_inshape={jtu.format_shape_dtype_string(arg_shape, dtype)}_pads={pads}",
          lax.pad,
          [RandArg(arg_shape, dtype), np.array(0, dtype), StaticArg(pads)],
          rng_factory=jtu.rand_small,
          arg_shape=arg_shape, dtype=dtype, pads=pads)
  for arg_shape in [(2, 3)]
  for dtype in jtu.dtypes.all_floating + jtu.dtypes.all_integer
  for pads in [
    [(0, 0, 0), (0, 0, 0)],  # no padding
    [(1, 1, 0), (2, 2, 0)],  # only positive edge padding
    [(1, 2, 1), (0, 1, 0)],  # edge padding and interior padding
    [(0, 0, 0), (-1, -1, 0)],  # negative padding
    [(0, 0, 0), (-2, -2, 4)],  # add big dilation then remove from edges
    [(0, 0, 0), (-2, -3, 1)],  # remove everything in one dimension
  ]
)


lax_sort = tuple( # one array, random data, all axes, all dtypes
  Harness(f"one_array_shape={jtu.format_shape_dtype_string(shape, dtype)}_axis={dimension}_isstable={is_stable}",
          lax.sort,
          [RandArg(shape, dtype), StaticArg(dimension), StaticArg(is_stable)],
          shape=shape,
          dimension=dimension,
          dtype=dtype,
          is_stable=is_stable)
  for dtype in jtu.dtypes.all
  for shape in [(5,), (5, 7)]
  for dimension in range(len(shape))
  for is_stable in [False, True]
) + tuple( # one array, potential edge cases
  Harness(f"one_special_array_shape={jtu.format_shape_dtype_string(arr.shape, arr.dtype)}_axis={dimension}_isstable={is_stable}",
          lax.sort,
          [arr, StaticArg(dimension), StaticArg(is_stable)],
          shape=arr.shape,
          dimension=dimension,
          dtype=arr.dtype,
          is_stable=is_stable)
  for arr, dimension in [
      [np.array([+np.inf, np.nan, -np.nan, -np.inf, 2, 4, 189], dtype=np.float32), -1]
  ]
  for is_stable in [False, True]
) + tuple( # several arrays, random data, all axes, all dtypes
  Harness(f"multi_array_shape={jtu.format_shape_dtype_string(shape, dtype)}_axis={dimension}_isstable={is_stable}",
          lambda *args: lax.sort_p.bind(*args[:-2], dimension=args[-2], is_stable=args[-1], num_keys=1),
          [RandArg(shape, dtype), RandArg(shape, dtype), StaticArg(dimension), StaticArg(is_stable)],
          shape=shape,
          dimension=dimension,
          dtype=dtype,
          is_stable=is_stable)
  for dtype in jtu.dtypes.all
  for shape in [(5,), (5, 7)]
  for dimension in range(len(shape))
  for is_stable in [False, True]
)

lax_linalg_qr = tuple(
  Harness(f"multi_array_shape={jtu.format_shape_dtype_string(shape, dtype)}_fullmatrices={full_matrices}",
          lax_linalg.qr,
          [RandArg(shape, dtype), StaticArg(full_matrices)],
          shape=shape,
          dtype=dtype,
          full_matrices=full_matrices)
  for dtype in jtu.dtypes.all
  for shape in [(1, 1), (3, 3), (3, 4), (2, 10, 5), (2, 200, 100)]
  for full_matrices in [False, True]
)

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
    [(3,), (1,), (2,), None],
    [(7,), (4,), (7,), None],
    [(5,), (1,), (5,), (2,)],
    [(8,), (1,), (6,), (2,)],
    [(5, 3), (1, 1), (3, 2), None],
    [(5, 3), (1, 1), (3, 1), None],
    [(7, 5, 3), (4, 0, 1), (7, 1, 3), None],
    [(5, 3), (1, 1), (2, 1), (1, 1)],
    [(5, 3), (1, 1), (5, 3), (2, 1)],
    # out-of-bounds cases
    [(5,), (-1,), (0,), None],
    [(5,), (-1,), (1,), None],
    [(5,), (-4,), (-2,), None],
    [(5,), (-5,), (-2,), None],
    [(5,), (-6,), (-5,), None],
    [(5,), (-10,), (-9,), None],
    [(5,), (-100,), (-99,), None],
    [(5,), (5,), (6,), None],
    [(5,), (10,), (11,), None],
    [(5,), (0,), (100,), None],
  ]
  for dtype in [np.float32]
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
    [(3,), (1,), (1,)],
    [(5, 3), (1, 1), (3, 1)],
    [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
    [(3,), (-1,), (1,)],  # out-of-bounds
    [(3,), (10,), (1,)],  # out-of-bounds
    [(3,), (10,), (4,)],  # out-of-bounds shape too big
    [(3,), (10,), (2,)],  # out-of-bounds
  ]
  for dtype, update_dtype in [
    (np.float32, np.float32),
    (np.float64, np.float32)
  ])

lax_squeeze = tuple(
  Harness(f"_inshape={jtu.format_shape_dtype_string(arg_shape, dtype)}_dimensions={dimensions}",  # type: ignore
          lax.squeeze,
          [RandArg(arg_shape, dtype), StaticArg(dimensions)],  # type: ignore[has-type]
          arg_shape=arg_shape, dtype=dtype, dimensions=dimensions)  # type: ignore[has-type]
  for arg_shape, dimensions in [
    [(1,), (0,)],
    [(1,), (-1,)],
    [(2, 1, 4), (1,)],
    [(2, 1, 4), (-2,)],
    [(2, 1, 3, 1), (1,)],
    [(2, 1, 3, 1), (1, 3)],
    [(2, 1, 3, 1), (3,)],
    [(2, 1, 3, 1), (1, -1)],
  ]
  for dtype in [np.float32]
)

shift_inputs = [
  (arg, dtype, shift_amount)
  for dtype in jtu.dtypes.all_unsigned + jtu.dtypes.all_integer
  for arg in [
    np.array([-250, -1, 0, 1, 250], dtype=dtype),
  ]
  for shift_amount in [0, 1, 2, 3, 7]
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
          [arg, StaticArg(np.array([shift_amount], dtype=dtype))])
  for arg, dtype, shift_amount in shift_inputs
)

lax_shift_right_arithmetic = tuple(
  Harness(f"_dtype={dtype.__name__}_shift_amount={shift_amount}",  # type: ignore
          lax.shift_right_arithmetic,
          [arg, StaticArg(np.array([shift_amount], dtype=dtype))])
  for arg, dtype, shift_amount in shift_inputs
)
