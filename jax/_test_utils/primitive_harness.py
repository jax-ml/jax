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


from typing import Any, Callable, Dict, Iterable, Optional, NamedTuple, Sequence, Tuple, Union, cast

from absl import testing
from jax import api
from jax import test_util as jtu
from jax import dtypes
from jax import lax

import numpy as onp

# TODO: these are copied from tests/lax_test.py (make this source of truth)
def supported_dtypes(dtypes):
  return [t for t in dtypes if t in jtu.supported_dtypes()]

float_dtypes = supported_dtypes([dtypes.bfloat16, onp.float16, onp.float32,
                                 onp.float64])
complex_elem_dtypes = supported_dtypes([onp.float32, onp.float64])
complex_dtypes = supported_dtypes([onp.complex64, onp.complex128])
inexact_dtypes = float_dtypes + complex_dtypes
int_dtypes = supported_dtypes([onp.int32, onp.int64])
uint_dtypes = supported_dtypes([onp.uint32, onp.uint64])
bool_dtypes = [onp.bool_]
default_dtypes = float_dtypes + int_dtypes
all_dtypes = float_dtypes + complex_dtypes + int_dtypes + bool_dtypes

Rng = Any  # A random number generator

class RandArg(NamedTuple):
  """Descriptor for a randomly generated argument."""
  shape: Tuple[int, ...]
  dtype: onp.dtype

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

  def _arg_maker(self, arg_descriptor, rng: Rng):
    if type(arg_descriptor) is StaticArg:
      return arg_descriptor.value
    if type(arg_descriptor) is RandArg:
      return self.rng_factory(rng)(arg_descriptor.shape, arg_descriptor.dtype)
    return arg_descriptor

  def args_maker(self, rng: Rng) -> Sequence:
    """All-argument maker, including the static ones."""
    return [self._arg_maker(ad, rng) for ad in self.arg_descriptors]

  def dyn_args_maker(self, rng: Rng) -> Sequence:
    """A dynamic-argument maker, for use with `dyn_fun`."""
    return [self._arg_maker(ad, rng) for ad in self.arg_descriptors
            if type(ad) is not StaticArg]

  def dyn_fun(self, *dyn_args):
    """Invokes `fun` given just the dynamic arguments."""
    all_args = self._args_from_dynargs(dyn_args)
    return self.fun(*all_args)

  def _args_from_dynargs(self, dyn_args: Sequence) -> Sequence:
    """All arguments, including the static ones."""
    next_dynamic_argnum = 0
    all_args = []
    for ad in self.arg_descriptors:
      if type(ad) is StaticArg:
        all_args.append(ad.value)
      else:
        all_args.append(dyn_args[next_dynamic_argnum])
        next_dynamic_argnum += 1
    return all_args


def parameterized(harness_group: Iterable[Harness]):
  return testing.parameterized.named_parameters(
    dict(testcase_name=harness.name, harness=harness)
    for harness in harness_group)


lax_pad = jtu.cases_from_list(
  Harness(f"_inshape={jtu.format_shape_dtype_string(shape, dtype)}_pads={pads}",
          lax.pad,
          [RandArg(shape, dtype),
           onp.array(0, dtype),
           StaticArg(pads)],
          rng_factory=jtu.rand_small)
  for shape in [(2, 3)]
  for dtype in default_dtypes
  for pads in [[(1, 2, 1), (0, 1, 0)]]
)

lax_squeeze = jtu.cases_from_list(
  Harness(f"_inshape={jtu.format_shape_dtype_string(arg_shape, dtype)}_dimensions={dimensions}",
          lax.squeeze,
          [RandArg(arg_shape, dtype),
           StaticArg(dimensions)],
          arg_shape=arg_shape, dtype=dtype)
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
  for dtype in [onp.float32]
)
