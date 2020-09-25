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
from jax import test_util as jtu
from jax import lax
from jax import lax_linalg
from jax import numpy as jnp

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
          [StaticArg(lhs), StaticArg(rhs)],
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
          [StaticArg(lhs), StaticArg(rhs)],
          f_jax=f_jax,
          dtype=dtype)
  for f_jax in [lax.min, lax.max]
  for dtype in jtu.dtypes.all_floating + jtu.dtypes.complex
  for lhs, rhs in [
    (np.array([np.inf], dtype=dtype), np.array([np.nan], dtype=dtype)),
    (np.array([-np.inf], dtype=dtype), np.array([np.nan], dtype=dtype))
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

lax_scatter = tuple(
  # Directly from lax.scatter in tests/lax_test.py
  Harness(
    f"fun={f_lax.__name__}_shape={jtu.format_shape_dtype_string(shape, dtype)}_scatterindices={scatter_indices.tolist()}_updateshape={update_shape}_updatewindowdims={dimension_numbers.update_window_dims}_insertedwindowdims={dimension_numbers.inserted_window_dims}_scatterdimstooperanddims={dimension_numbers.scatter_dims_to_operand_dims}_indicesaresorted={indices_are_sorted}_uniqueindices={unique_indices}".replace(' ', ''),
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
  # We explicitly decide against testing lax.scatter, as its reduction function
  # is lambda x, y: y, which is not commutative and thus makes results
  # non-deterministic when an index into the operand is updated several times.
  for f_lax in [lax.scatter_min, lax.scatter_max, lax.scatter_mul,
                lax.scatter_add]
  for dtype in { lax.scatter_min: jtu.dtypes.all
               , lax.scatter_max: jtu.dtypes.all
                 # lax.scatter_mul and lax.scatter_add are not compatible with
                 # np.bool_ operands.
               , lax.scatter_mul: filter(lambda t: t != np.bool_, jtu.dtypes.all)
               , lax.scatter_add: filter(lambda t: t != np.bool_, jtu.dtypes.all)
               }[f_lax]
  for shape, scatter_indices, update_shape, dimension_numbers in [
      ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,))),
      ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(),
        scatter_dims_to_operand_dims=(0,))),
      ((10, 5,), np.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,))),
  ]
  for indices_are_sorted in [False, True]
  # `unique_indices` does not affect correctness, only performance, and thus
  # does not need to be tested here. If/when it will make sense to add a test
  # with `unique_indices` = True, particular care will have to be taken with
  # regards to the choice of parameters, as the results are only predictable
  # when all the indices to be updated are pairwise non-overlapping. Identifying
  # such cases is non-trivial.
  for unique_indices in [False]
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

lax_top_k = tuple( # random testing
  Harness(f"_inshape={jtu.format_shape_dtype_string(shape, dtype)}_k={k}",
          lax.top_k,
          [RandArg(shape, dtype), StaticArg(k)],
          shape=shape,
          dtype=dtype,
          k=k)
  for dtype in jtu.dtypes.all
  for shape in [(3,), (5, 3)]
  for k in [-1, 1, 3, 4]
  for rng_factory in [jtu.rand_default]
) + tuple( # stability test
  Harness(f"stability_inshape={jtu.format_shape_dtype_string(arr.shape, arr.dtype)}_k={k}",
          lax.top_k,
          [arr, StaticArg(k)],
          shape=arr.shape,
          dtype=arr.dtype,
          k=k)
  for arr in [
      np.array([5, 7, 5, 8, 8, 5], dtype=np.int32)
  ]
  for k in [1, 3, 6]
) + tuple( # nan/inf sorting test
  Harness(f"nan_inshape={jtu.format_shape_dtype_string(arr.shape, arr.dtype)}_k={k}",
          lax.top_k,
          [arr, StaticArg(k)],
          shape=arr.shape,
          dtype=arr.dtype,
          k=k)
  for arr in [
      np.array([+np.inf, np.nan, -np.nan, np.nan, -np.inf, 3], dtype=np.float32)
  ]
  for k in [1, 3, 6]
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
) + tuple( # 2 arrays, random data, all axes, all dtypes
  Harness(f"two_arrays_shape={jtu.format_shape_dtype_string(shape, dtype)}_axis={dimension}_isstable={is_stable}",
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
) + tuple( # 3 arrays, random data, all axes, all dtypes
  Harness(f"three_arrays_shape={jtu.format_shape_dtype_string(shape, dtype)}_axis={dimension}_isstable={is_stable}",
          lambda *args: lax.sort_p.bind(*args[:-2], dimension=args[-2], is_stable=args[-1], num_keys=1),
          [RandArg(shape, dtype), RandArg(shape, dtype), RandArg(shape, dtype),
           StaticArg(dimension), StaticArg(is_stable)],
          shape=shape,
          dimension=dimension,
          dtype=dtype,
          is_stable=is_stable)
  for dtype in jtu.dtypes.all
  for shape in [(5,)]
  for dimension in (0,)
  for is_stable in [False, True]
)

lax_linalg_cholesky = tuple(
  Harness(f"_shape={jtu.format_shape_dtype_string(shape, dtype)}",
          lambda *args: lax_linalg.cholesky_p.bind(*args),
          [RandArg(shape, dtype)],
          shape=shape,
          dtype=dtype)
  for dtype in jtu.dtypes.all_inexact
  for shape in [(1, 1), (4, 4), (2, 5, 5), (200, 200), (1000, 0, 0)]
)

lax_linalg_qr = tuple(
  Harness(f"multi_array_shape={jtu.format_shape_dtype_string(shape, dtype)}_fullmatrices={full_matrices}",
          lax_linalg.qr,
          [RandArg(shape, dtype), StaticArg(full_matrices)],
          shape=shape,
          dtype=dtype,
          full_matrices=full_matrices)
  for dtype in jtu.dtypes.all_floating  + jtu.dtypes.complex
  for shape in [(1, 1), (3, 3), (3, 4), (2, 10, 5), (2, 200, 100)]
  for full_matrices in [False, True]
)

def _fft_harness_gen(nb_axes):
  def _fft_rng_factory(dtype):
    _all_integers = jtu.dtypes.all_integer + jtu.dtypes.all_unsigned + jtu.dtypes.boolean
    # For integer types, use small values to keep the errors small
    if dtype in _all_integers:
      return jtu.rand_small
    else:
      return jtu.rand_default

  return tuple(
    Harness(f"{nb_axes}d_shape={jtu.format_shape_dtype_string(shape, dtype)}_ffttype={fft_type}_fftlengths={fft_lengths}",
            lax.lax_fft.fft,
            [RandArg(shape, dtype), StaticArg(fft_type), StaticArg(fft_lengths)],
            rng_factory=_fft_rng_factory(dtype),
            shape=shape,
            dtype=dtype,
            fft_type=fft_type,
            fft_lengths=fft_lengths)
    for dtype in jtu.dtypes.all
    for shape in filter(lambda x: len(x) >= nb_axes,
                        [(10,), (12, 13), (14, 15, 16), (14, 15, 16, 17)])
    for fft_type, fft_lengths in [(xla_client.FftType.FFT, shape[-nb_axes:]),
                                  (xla_client.FftType.IFFT, shape[-nb_axes:]),
                                  (xla_client.FftType.RFFT, shape[-nb_axes:]),
                                  (xla_client.FftType.IRFFT,
                                   shape[-nb_axes:-1] + ((shape[-1] - 1) * 2,))]
    if not (dtype in jtu.dtypes.complex and fft_type == xla_client.FftType.RFFT)
  )

lax_fft = tuple(_fft_harness_gen(1) + _fft_harness_gen(2) + _fft_harness_gen(3) +
                _fft_harness_gen(4))

lax_linalg_svd = tuple(
  Harness(f"shape={jtu.format_shape_dtype_string(shape, dtype)}_fullmatrices={full_matrices}_computeuv={compute_uv}",
          lambda *args: lax_linalg.svd_p.bind(args[0], full_matrices=args[1],
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
          lax_linalg.eig,
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
          lax_linalg.eigh,
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
    [(5,), (3,), (6,), None]
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
    (np.float64, np.float64)
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
          [arg, StaticArg(np.array([shift_amount], dtype=dtype))],
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

lax_select_and_gather_add = tuple(
  # Tests with 2d shapes (see tests.lax_autodiff_test.testReduceWindowGrad)
  Harness(f"2d_shape={jtu.format_shape_dtype_string(shape, dtype)}_selectprim={select_prim}_windowdimensions={window_dimensions}_windowstrides={window_strides}_padding={padding}_basedilation={base_dilation}_windowdilation={window_dilation}",
          lax._select_and_gather_add,
          [RandArg(shape, dtype), RandArg(shape, dtype), StaticArg(select_prim),
           StaticArg(window_dimensions), StaticArg(window_strides),
           StaticArg(padding), StaticArg(base_dilation),
           StaticArg(window_dilation)],
          shape=shape,
          dtype=dtype,
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          padding=padding,
          base_dilation=base_dilation,
          window_dilation=window_dilation)
  for dtype in jtu.dtypes.all_floating
  for shape in [(4, 6)]
  for select_prim in [lax.le_p, lax.ge_p]
  for window_dimensions in [(2, 1), (1, 2)]
  for window_strides in [(1, 1), (2, 1), (1, 2)]
  for padding in tuple(set([tuple(lax.padtype_to_pads(shape, window_dimensions,
                                                      window_strides, p))
                            for p in ['VALID', 'SAME']] +
                           [((0, 3), (1, 2))]))
  for base_dilation in [(1, 1)]
  for window_dilation in [(1, 1)]
) + tuple(
  # Tests with 4d shapes (see tests.lax_autodiff_test.testReduceWindowGrad)
  Harness(f"4d_shape={jtu.format_shape_dtype_string(shape, dtype)}_selectprim={select_prim}_windowdimensions={window_dimensions}_windowstrides={window_strides}_padding={padding}_basedilation={base_dilation}_windowdilation={window_dilation}",
          lax._select_and_gather_add,
          [RandArg(shape, dtype), RandArg(shape, dtype), StaticArg(select_prim),
           StaticArg(window_dimensions), StaticArg(window_strides),
           StaticArg(padding), StaticArg(base_dilation),
           StaticArg(window_dilation)],
          shape=shape,
          dtype=dtype,
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          padding=padding,
          base_dilation=base_dilation,
          window_dilation=window_dilation)
  for dtype in jtu.dtypes.all_floating
  for shape in [(3, 2, 4, 6)]
  for select_prim in [lax.le_p, lax.ge_p]
  for window_dimensions in [(1, 1, 2, 1), (2, 1, 2, 1)]
  for window_strides in [(1, 2, 2, 1), (1, 1, 1, 1)]
  for padding in tuple(set([tuple(lax.padtype_to_pads(shape, window_dimensions,
                                                      window_strides, p))
                            for p in ['VALID', 'SAME']] +
                           [((0, 1), (1, 0), (2, 3), (0, 2))]))
  for base_dilation in [(1, 1, 1, 1)]
  for window_dilation in [(1, 1, 1, 1)]
)

lax_reduce_window = tuple(
  # Tests with 2d shapes (see tests.lax_test.testReduceWindow)
  Harness(f"2d_shape={jtu.format_shape_dtype_string(shape, dtype)}_initvalue={init_value}_computation={computation.__name__}_windowdimensions={window_dimensions}_windowstrides={window_strides}_padding={padding}_basedilation={base_dilation}_windowdilation={window_dilation}".replace(' ', ''),
          lax.reduce_window,
          [RandArg(shape, dtype), StaticArg(init_value), StaticArg(computation),
           StaticArg(window_dimensions), StaticArg(window_strides),
           StaticArg(padding), StaticArg(base_dilation), StaticArg(window_dilation)],
          shape=shape,
          dtype=dtype,
          init_value=init_value,
          computation=computation,
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          padding=padding,
          base_dilation=base_dilation,
          window_dilation=window_dilation)
  for computation in [lax.add, lax.max, lax.min, lax.mul]
  for dtype in { lax.add: filter(lambda t: t != np.bool_, jtu.dtypes.all)
               , lax.mul: filter(lambda t: t != np.bool_, jtu.dtypes.all)
               , lax.max: jtu.dtypes.all
               , lax.min: jtu.dtypes.all
               }[computation]
  for init_value in map(
      dtype,
      (lambda ts: ts[0] if not dtype in jtu.dtypes.all_floating else ts[1])(
          { lax.add: ([0, 1], [0, 1])
          , lax.mul: ([1], [1])
          , lax.max: ([1], [-np.inf, 1])
          , lax.min: ([0], [np.inf, 0])
          }[computation]
      )
  )
  for shape in [(4, 6)]
  for window_dimensions in [(1, 2)]
  for window_strides in [(2, 1)]
  for padding in tuple(set([tuple(lax.padtype_to_pads(shape, window_dimensions,
                                                      window_strides, p))
                            for p in ['VALID', 'SAME']] +
                           [((0, 3), (1, 2))]))
  for base_dilation in [(2, 3)]
  for window_dilation in [(1, 2)]
) + tuple(
  # Tests with 4d shapes (see tests.lax_test.testReduceWindow)
  Harness(f"4d_shape={jtu.format_shape_dtype_string(shape, dtype)}_initvalue={init_value}_computation={computation.__name__}_windowdimensions={window_dimensions}_windowstrides={window_strides}_padding={padding}_basedilation={base_dilation}_windowdilation={window_dilation}".replace(' ', ''),
          lax.reduce_window,
          [RandArg(shape, dtype), StaticArg(init_value), StaticArg(computation),
           StaticArg(window_dimensions), StaticArg(window_strides),
           StaticArg(padding), StaticArg(base_dilation), StaticArg(window_dilation)],
          shape=shape,
          dtype=dtype,
          init_value=init_value,
          computation=computation,
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          padding=padding,
          base_dilation=base_dilation,
          window_dilation=window_dilation)
  for computation in [lax.add, lax.max, lax.min, lax.mul]
  for dtype in { lax.add: filter(lambda t: t != np.bool_, jtu.dtypes.all)
               , lax.mul: filter(lambda t: t != np.bool_, jtu.dtypes.all)
               , lax.max: jtu.dtypes.all
               , lax.min: jtu.dtypes.all
               }[computation]
  for init_value in map(
      dtype,
      (lambda ts: ts[0] if not dtype in jtu.dtypes.all_floating else ts[1])(
          { lax.add: ([0, 1], [0, 1])
          , lax.mul: ([1], [1])
          , lax.max: ([1], [-np.inf, 1])
          , lax.min: ([0], [np.inf, 0])
          }[computation]
      )
  )
  for shape in [(3, 2, 4, 6)]
  for window_dimensions in [(1, 1, 2, 1)]
  for window_strides in [(1, 2, 2, 1)]
  for padding in tuple(set([tuple(lax.padtype_to_pads(shape, window_dimensions,
                                                      window_strides, p))
                            for p in ['VALID', 'SAME']] +
                           [((0, 1), (1, 0), (2, 3), (0, 2))]))
  for base_dilation in [(2, 1, 3, 2)]
  for window_dilation in [(1, 2, 2, 1)]
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

def _make_conv_harness(name, *, lhs_shape=(2, 3, 9, 10), rhs_shape=(3, 3, 4, 5),
                       dtype=np.float32, window_strides=(1, 1), precision=None,
                       padding=((0, 0), (0, 0)), lhs_dilation=(1, 1),
                       rhs_dilation=(1, 1), feature_group_count=1,
                       dimension_numbers=("NCHW", "OIHW", "NCHW"),
                       batch_group_count=1):
  return Harness(f"_{name}_lhs={jtu.format_shape_dtype_string(lhs_shape, dtype)}_rhs={jtu.format_shape_dtype_string(rhs_shape, dtype)}_windowstrides={window_strides}_padding={padding}_lhsdilation={lhs_dilation}_rhsdilation={rhs_dilation}_dimensionnumbers={dimension_numbers}_featuregroupcount={feature_group_count}_batchgroupcount={batch_group_count}_precision={precision}".replace(' ', ''),
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
                 precision=precision)

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
)
