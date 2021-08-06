# Copyright 2021 Google LLC
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

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import contextlib
import threading

from jax import core
from jax import linear_util as lu
from jax._src import api

from .ir_builder import Builder, FunctionBuilder
from .mlir_imports import *
from .primitives import HLO_HANDLERS, PrimitiveInvocation


class _ThreadLocalState(threading.local):

  def __init__(self):
    self.function_builder: Optional[FunctionBuilder] = None


_thread_local_state = _ThreadLocalState()


def _current_function_builder() -> FunctionBuilder:
  fb = _thread_local_state.function_builder
  assert fb is not None, "No current function builder"
  return fb


@contextlib.contextmanager
def new_function_scope(fb: FunctionBuilder):
  # TODO: Support stack
  assert _thread_local_state.function_builder is None
  _thread_local_state.function_builder = fb
  try:
    yield
  finally:
    _thread_local_state.function_builder = None


def trace_flat_function(
    fun: Callable,
    *,
    builder: Builder,
    in_avals: Sequence[core.AbstractValue],
    exported_name: Optional[str] = None) -> Sequence[core.AbstractValue]:
  api._check_callable(fun)
  if exported_name is None:
    exported_name = getattr(fun, "__name__", "unknown")
  if not core.trace_state_clean():
    raise ValueError(
        "convert must be used outside all JAX transformations." +
        f"Trace state: {core.thread_local_state.trace_state.trace_stack}")

  input_types = _convert_avals_to_ir_types(builder, in_avals)
  fb = builder.create_function(exported_name, input_types, [])
  in_vals = list(fb.func_op.entry_block.arguments)

  # Interpret the function.
  wrapped_fun = lu.wrap_init(fun)
  with core.new_base_main(IreeTrace) as main:
    with new_function_scope(fb):
      fun = interpret_function_ir(wrapped_fun, main, in_avals)
      out_vals = fun.call_wrapped(*in_vals)

  # Peel out the IR value.
  returns = [ir_value for ir_value, _ in out_vals]
  out_avals = [aval for _, aval in out_vals]

  # Remove me.
  fb.emit_return(returns)
  return out_avals


@lu.transformation
def interpret_function_ir(main: core.MainTrace,
                          in_avals: Sequence[core.ShapedArray],
                          *in_vals: ir.Value):
  trace = IreeTrace(main, core.cur_sublevel())
  in_tracers = tuple(
      IreeTracer(trace, val, aval) for val, aval in zip(in_vals, in_avals))
  # The outs may be core.unit, see comment in TensorFlowTrace.pure.
  outs = yield in_tracers, {}  # type: Sequence[Union[Any, core.Unit]]
  out_tracers: Iterable[IreeTracer] = (map(trace.full_raise,
                                           outs))  # type: ignore
  out_vals_with_avals: Sequence[Tuple[Any, core.ShapedArray]] = (tuple(
      (t.val, t.aval) for t in out_tracers))
  yield out_vals_with_avals


class IreeTracer(core.Tracer):
  val: ir.Value
  _aval: core.AbstractValue
  __slots__ = ["val", "_aval"]

  def __init__(self, trace: "IreeTrace", val: ir.Value,
               aval: core.AbstractValue):
    self._trace = trace
    self._aval = aval
    self.val = val

  @property
  def aval(self):
    return self._aval

  def full_lower(self):
    return self


class IreeTrace(core.Trace):
  """A trace of an MLIR fragment."""
  HANDLER_TABLES = [HLO_HANDLERS]

  def process_primitive(self, primitive: core.Primitive,
                        tracers: Sequence[IreeTracer], params) -> IreeTracer:
    args_avals: Sequence[core.ShapedArray] = tuple(t.aval for t in tracers)
    args_vals: Sequence[ir.Value] = tuple(t.val for t in tracers)
    out_aval = primitive.abstract_eval(*args_avals, **params)
    inv = PrimitiveInvocation(primitive, args_vals, args_avals, out_aval,
                              params)
    fb = _current_function_builder()
    result = None
    for table in self.HANDLER_TABLES:
      handler = table.get(primitive)
      if handler is None:
        continue
      with fb.b.loc, fb.ip:
        result = handler(fb, inv)
      if result is NotImplemented:
        continue

    if result is None:
      with fb.b.loc, fb.ip:
        result = inv.emit_fallback(fb)

    out = IreeTracer(self, result, out_aval)
    return out


def _convert_avals_to_ir_types(builder: Builder,
                               avals: Sequence[core.AbstractValue]):

  def convert(aval: core.AbstractValue):
    # TODO: Better way to do this?
    if isinstance(aval, core.ShapedArray):
      element_type = builder.convert_dtype_to_ir_type(aval.dtype)
      # TODO: Handle symbolic shape dims?
      return ir.RankedTensorType.get(aval.shape, element_type)
    elif isinstance(aval, core.UnshapedArray):
      element_type = builder.convert_dtype_to_ir_type(aval.dtype)
      return ir.UnrankedTensorType.get(element_type)

  with builder.loc:
    return [convert(aval) for aval in avals]
