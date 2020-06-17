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
from jax import dtypes
from jax import lax

import numpy as np

FLAGS = config.FLAGS

# TODO: these are copied from tests/lax_test.py (make this source of truth)
# Do not run int64 tests unless FLAGS.jax_enable_x64, otherwise we get a
# mix of int32 and int64 operations.
def supported_dtypes(dtypes):
  return [t for t in dtypes if
          t in jtu.supported_dtypes() and
          (FLAGS.jax_enable_x64 or np.dtype(t).itemsize != 8)]

float_dtypes = supported_dtypes([dtypes.bfloat16, np.float16, np.float32,
                                 np.float64])
complex_elem_dtypes = supported_dtypes([np.float32, np.float64])
complex_dtypes = supported_dtypes([np.complex64, np.complex128])
inexact_dtypes = float_dtypes + complex_dtypes
int_dtypes = supported_dtypes([np.int8, np.int16, np.int32, np.int64])
uint_dtypes = supported_dtypes([np.uint8, np.uint16, np.uint32, np.uint64])
bool_dtypes = [np.bool_]
default_dtypes = float_dtypes + int_dtypes
all_dtypes = float_dtypes + complex_dtypes + int_dtypes + bool_dtypes

Rng = Any  # A random number generator

class RandArg(NamedTuple):
  """Descriptor for a randomly generated argument."""
  shape: Tuple[int, ...]
  dtype: np.dtype

class StaticArg(NamedTuple):
  """Descriptor for a static argument."""
  value: Any

class Harness:
  """Specifies inputs and callable for a primitive.

  A primitive can take dynamic and static arguments. The dynamic arguments can
  be generated using a RNG, are numeric (and appropriate for JIT).
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
    cases = cases[0:1]
  return testing.parameterized.named_parameters(*cases)


lax_pad = jtu.cases_from_list(
  Harness(f"_inshape={jtu.format_shape_dtype_string(arg_shape, dtype)}_pads={pads}",
          lax.pad,
          [RandArg(arg_shape, dtype), np.array(0, dtype), StaticArg(pads)],
          rng_factory=jtu.rand_small,
          arg_shape=arg_shape, dtype=dtype, pads=pads)
  for arg_shape in [(2, 3)]
  for dtype in default_dtypes
  for pads in [
    [(0, 0, 0), (0, 0, 0)],  # no padding
    [(1, 1, 0), (2, 2, 0)],  # only positive edge padding
    [(1, 2, 1), (0, 1, 0)],  # edge padding and interior padding
    [(0, 0, 0), (-1, -1, 0)],  # negative padding
    [(0, 0, 0), (-2, -2, 4)],  # add big dilation then remove from edges
    [(0, 0, 0), (-2, -3, 1)],  # remove everything in one dimension
  ]
)

lax_slice = jtu.cases_from_list(
  Harness(f"_shape={shape}_start_indices={start_indices}_limit_indices={limit_indices}_strides={strides}",  # type: ignore
          lax.slice,
          [RandArg(shape, dtype),  # type: ignore
           StaticArg(start_indices),  # type: ignore
           StaticArg(limit_indices),  # type: ignore
           StaticArg(strides)],  # type: ignore
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
  ]
  for dtype in [np.float32]
)

# Like lax_slice, but (a) make the start_indices dynamic arg, and no strides.
lax_dynamic_slice = [
  Harness(harness.name,
          lax.dynamic_slice,
          [harness.arg_descriptors[0],
           np.array(list(start_indices)),
           StaticArg(tuple(map(operator.sub, limit_indices, start_indices)))])
  for harness in lax_slice
  for start_indices in [harness.params["start_indices"]]
  for limit_indices in [harness.params["limit_indices"]]
]

lax_dynamic_update_slice = jtu.cases_from_list(
  Harness((f"_operand={jtu.format_shape_dtype_string(shape, dtype)}"  # type: ignore
           f"_update={jtu.format_shape_dtype_string(update_shape, update_dtype)}"
           f"_start_indices={start_indices}"),
          lax.dynamic_update_slice,
          [RandArg(shape, dtype),  # type: ignore
           RandArg(update_shape, update_dtype),  # type: ignore
           np.array(start_indices)])  # type: ignore
  for shape, start_indices, update_shape in [
    [(3,), (1,), (1,)],
    [(5, 3), (1, 1), (3, 1)],
    [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
  ]
  for dtype, update_dtype in [
    (np.float32, np.float32),
    (np.float64, np.float32)
  ])

lax_squeeze = jtu.cases_from_list(
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
  for dtype in supported_dtypes(uint_dtypes + int_dtypes)
  for arg in [
    np.array([-250, -1, 0, 1, 250], dtype=dtype),
  ]
  for shift_amount in [0, 1, 2, 3, 7]
]

lax_shift_left = jtu.cases_from_list(
  Harness(f"_dtype={dtype.__name__}_shift_amount={shift_amount}",  # type: ignore
          lax.shift_left,
          [arg, StaticArg(np.array([shift_amount], dtype=dtype))])
  for arg, dtype, shift_amount in shift_inputs
)

lax_shift_right_logical = jtu.cases_from_list(
  Harness(f"_dtype={dtype.__name__}_shift_amount={shift_amount}",  # type: ignore
          lax.shift_right_logical,
          [arg, StaticArg(np.array([shift_amount], dtype=dtype))])
  for arg, dtype, shift_amount in shift_inputs
)

lax_shift_right_arithmetic = jtu.cases_from_list(
  Harness(f"_dtype={dtype.__name__}_shift_amount={shift_amount}",  # type: ignore
          lax.shift_right_arithmetic,
          [arg, StaticArg(np.array([shift_amount], dtype=dtype))])
  for arg, dtype, shift_amount in shift_inputs
)
