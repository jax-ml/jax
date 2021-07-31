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

import logging

import numpy as np

from jax import core, tree_util
from jax import linear_util as lu
from jax.core import DimSize, Shape
from jax._src import api
from jax._src import util

from .ir_builder import Builder


def trace_function(fun: Callable,
                   *,
                   builder: Builder,
                   shapes_and_dtypes,
                   exported_name: Optional[str] = None):
  # TODO: shapes_and_dtypes is hoaky. Formalize.
  api._check_callable(fun)
  if exported_name is None:
    exported_name = getattr(fun, "__name__", "unknown")
  if not core.trace_state_clean():
    raise ValueError(
        "convert must be used outside all JAX transformations." +
        f"Trace state: {core.thread_local_state.trace_state.trace_stack}")

  with builder.loc:
    input_types = _convert_shapes_to_types(builder, shapes_and_dtypes)
  fb = builder.create_function(exported_name, input_types, [])

  # Create a list of arrays for tracing.
  def _create_array_val_from_type(t):
    if builder.ir.RankedTensorType.isinstance(t):
      t = builder.ir.RankedTensorType(t)
      shape = _convert_shaped_type_dims_to_list(t)
      dtype = _convert_ir_type_to_dtype(builder, t.element_type)
      return core.ShapedArray(shape, dtype)
    elif builder.ir.UnrankedTensorType.isinstance(t):
      t = builder.ir.UnrankedTensorType(t)
      dtype = _convert_ir_type_to_dtype(builder, t.element_type)
      return core.UnshapedArray(dtype)
    raise ValueError(f"IR type cannot be mapped to JAX type: {t}")

  in_avals = [_create_array_val_from_type(t) for t in input_types]

  # Interpret the function.
  wrapped_fun = lu.wrap_init(fun)
  with core.new_base_main(IreeTrace) as main:
    fun = _interpret_subtrace(wrapped_fun, main, in_avals)
    out_vals = fun.call_wrapped(*in_avals)

  # Remove me.
  fb.emit_return([])


@lu.transformation
def _interpret_subtrace(main: core.MainTrace,
                        in_avals: Sequence[core.ShapedArray], *in_vals):
  trace = IreeTrace(main, core.cur_sublevel())
  in_tracers = tuple(
      IreeTracer(trace, val, aval)
      for val, aval in zip(in_vals, in_avals))
  # The outs may be core.unit, see comment in TensorFlowTrace.pure.
  outs = yield in_tracers, {}  # type: Sequence[Union[Any, core.Unit]]
  out_tracers: Iterable[IreeTracer] = (
      map(trace.full_raise, outs))  # type: ignore
  out_vals_with_avals: Sequence[Tuple[Any, core.ShapedArray]] = (
      tuple((t.val, t.aval) for t in out_tracers))
  yield out_vals_with_avals


class IreeTracer(core.Tracer):
  # val: TfVal
  # _aval: core.ShapedArray
  __slots__ = ["val", "_aval"]

  def __init__(self, trace: "IreeTrace", val,
               aval: core.AbstractValue):
    self._trace = trace
    self._aval = aval
    self.val = val

  @property
  def aval(self):
    return self._aval


class IreeTrace(core.Trace):
  def process_primitive(self, primitive: core.Primitive,
                        tracers: Sequence[IreeTracer],
                        params) -> IreeTracer:
    print(f"PRIMTIVE: {primitive}({tracers})")
    return tracers[0]  # TODO: It's quitting time.


def _convert_shapes_to_types(builder: Builder, shapes_and_dtypes):
  # TODO: Flatten, etc. And burn with fire.
  types = []
  for dtype, shape in shapes_and_dtypes:
    element_type = _convert_dtype_to_ir_type(builder, dtype)
    if shape is None:
      t = builder.ir.UnrankedTensorType.get(element_type)
    else:
      t = builder.ir.RankedTensorType.get(shape, element_type)
    types.append(t)
  return types


def _convert_dtype_to_ir_type(builder: Builder, dtype):
  # TODO: Terrible.
  return builder.ir.F32Type.get()


def _convert_ir_type_to_dtype(builder: Builder, element_type):
  # TODO: Terrible.
  return np.float32


def _convert_shaped_type_dims_to_list(t):
  # TODO: Ugh. Has anyone tried to use this before?
  def get_dim(index):
    if t.is_dynamic_dim(index):
      return None
    return t.get_dim_size(index)

  return [get_dim(i) for i in range(t.rank)]
