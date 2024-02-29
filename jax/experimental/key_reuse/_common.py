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
from jax._src import prng
import numpy as np


class Sink(NamedTuple):
  idx: int
  mask: bool | np.ndarray = True

  def __repr__(self):
    if isinstance(self.mask, bool) and self.mask:
      return f"Sink({self.idx})"
    else:
      return f"Sink({self.idx}, mask={self.mask})"


class Source(NamedTuple):
  idx: int
  mask: bool | np.ndarray = True

  def __repr__(self):
    if isinstance(self.mask, bool) and self.mask:
      return f"Source({self.idx})"
    else:
      return f"Source({self.idx}, mask={self.mask})"

class Forward(NamedTuple):
  in_idx: int
  out_idx: int


class KeyReuseSignature(NamedTuple):
  sinks: list[Sink]
  sources: list[Source]
  forwards: list[Forward] = []

  def check_signature(self, *args, jaxpr=None):
    for sink in self.sinks:
      if not isinstance(args[sink.idx], prng.PRNGKeyArray):
        continue
      if np.any(args[sink.idx]._consumed & sink.mask):
        msg = f"Previously-consumed key at index {sink.idx} passed to function"
        if jaxpr:
          msg += f"\n{jaxpr=}"
        raise KeyReuseError(msg)

  def update_consumption(self, args_in, args_out):
    for sink in self.sinks:
      arg = args_in[sink.idx]
      if isinstance(arg, prng.PRNGKeyArray):
        arg._consumed = arg._consumed | sink.mask
    for arg in args_out:
      if isinstance(arg, prng.PRNGKeyArray):
        arg._consumed = True
    for source in self.sources:
      if isinstance(args_out[source.idx], prng.PRNGKeyArray):
        args_out[source.idx]._consumed = ~np.asarray(source.mask)
    for forward in self.forwards:
      arg_in = args_in[forward.in_idx]
      arg_out = args_out[forward.out_idx]
      if isinstance(arg_in, prng.PRNGKeyArray) and isinstance(arg_out, prng.PRNGKeyArray):
        arg_out._consumed = arg_in._consumed


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
