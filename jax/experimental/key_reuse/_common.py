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

from __future__ import annotations

from typing import NamedTuple
from jax import core
from jax.interpreters import batching, mlir
import numpy as np


class Sink(NamedTuple):
  idx: int
  mask: bool | np.ndarray = True


class Source(NamedTuple):
  idx: int
  mask: bool | np.ndarray = True


class KeyReuseSignature(NamedTuple):
  sinks: list[Sink]
  sources: list[Source]


class KeyReuseError(RuntimeError):
  pass

consume_p = core.Primitive("consume")
consume_p.def_impl(lambda x: x)
consume_p.def_abstract_eval(lambda x: x)
batching.defvectorized(consume_p)
mlir.register_lowering(
    consume_p,
    mlir.lower_fun(lambda x: x, multiple_results=False))

def consume(key):
  """Consume the key and return a consumed copy."""
  return consume_p.bind(key)

unconsumed_copy_p = core.Primitive("unconsumed_copy")
unconsumed_copy_p.def_impl(lambda x: x)
unconsumed_copy_p.def_abstract_eval(lambda x: x)
batching.defvectorized(unconsumed_copy_p)
mlir.register_lowering(
    unconsumed_copy_p,
    mlir.lower_fun(lambda x: x, multiple_results=False))

def unconsumed_copy(key):
  """Return a copy of key marked as unconsumed."""
  return unconsumed_copy_p.bind(key)

assert_consumed_value_p = core.Primitive("assert_consumed_value")
assert_consumed_value_p.def_impl(lambda x, *, value: x)
assert_consumed_value_p.def_abstract_eval(lambda x, *, value: x)
batching.defvectorized(assert_consumed_value_p)
mlir.register_lowering(
    assert_consumed_value_p,
    mlir.lower_fun(lambda x, *, value: x, multiple_results=False))

def assert_unconsumed(key):
  """Assert that a key is unconsumed"""
  assert_consumed_value_p.bind(key, value=False)

def assert_consumed(key, value=True):
  """Assert that a key is consumed"""
  assert_consumed_value_p.bind(key, value=value)
