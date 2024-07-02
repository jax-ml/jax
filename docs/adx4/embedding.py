# ---
# Copyright 2024 The JAX Authors.
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

from typing import TypeAlias, Union, Sequence, Optional, Any, Callable, TypeVar
from contextlib import contextmanager
from dataclasses import dataclass

from util import *
from core import *

# === embedding ===

# We keep a "current emitter" as a globally-readable context. This is purely to
# reduce clutter in user-facing code. Internally we pass around emitters
# explicitly so it's easier to follow the flow of data.
@dataclass
class CurrentEmitter:
  emitter : Emitter

@contextmanager
def set_current_emitter(emitter):
  prev = current_emitter.emitter
  current_emitter.emitter = emitter
  try:
    yield
  finally:
    current_emitter.emitter = prev

def top_level_emitter():
  return FrontendLoweringEmitter(eval_emitter)

current_emitter = CurrentEmitter(top_level_emitter())

def emit_primitive(p, args, funargs=()):
  emitter = current_emitter.emitter
  args_canonical = [canonicalize_pyval(arg) for arg in args]
  arg_tys = [arg.ty for arg in args_canonical]
  if isinstance(p, CallableHof):
    result_ty = None
  else:
    fun_tys = [f.ty for f in funargs]
    result_ty = p.result_type(*(tuple(arg_tys) + tuple(fun_tys)))
  return emitter.emit_primitive(p, result_ty, args_canonical, funargs)

# This turns a function that reads the implicit "current_emitter" context into
# one that takes the emitter explicitly, conforming to the `OpStream` API
@dataclass
class WithExplicitEmitter:
  f : Callable
  def __call__(self, emitter, *args):
    with set_current_emitter(emitter):
      return self.f(*args)

# `emit` requires each argument to be a `TraceVal`, which is either a `JaxVal`
# or a `Tracer`. A PyVal could be a tuples of tracers, or a Python float
# representing a rank-0 array. We canonicalize these to a `TraceVal` before
# calling `emit`.

PyVal : TypeAlias = Any
pyval_canonicalizers = {}

def register_canonicalizer(t, f):
  pyval_canonicalizers[t] = f

# may use current emitter, for example to build a FancyTuple from a python
# tuple.
def canonicalize_pyval(x: PyVal) -> TraceVal:
  if isinstance(x, JaxVal):
    return x
  elif isinstance(x, Tracer):
    return x
  elif type(x) in pyval_canonicalizers:
    return pyval_canonicalizers[type(x)](x)
  else:
    raise TypeError(f'Unrecognized JAX type: {type(x)}')

def trace_to_jaxpr(f, arg_types:list[JaxType]) -> Jaxpr:
  builder = BuilderEmitter(arg_types)
  with set_current_emitter(builder):
    result = canonicalize_pyval(f(*builder.args))
    return builder.build(result)
