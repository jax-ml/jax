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

from jax import core
from typing import Any, Callable, Dict, Sequence, Union

from jax._src.lax import lax

from .ir_builder import FunctionBuilder
from .iree_imports import *


class PrimitiveInvocation:
  __slots__ = [
      "primitive",
      "vals",
      "avals",
      "out_aval",
      "params",
  ]

  def __init__(self, primitive: core.Primitive, vals: Sequence[ir.Value],
               avals: Sequence[core.AbstractValue],
               out_aval: Union[core.AbstractValue,
                               Sequence[core.AbstractValue]], params: dict):
    self.primitive = primitive
    self.vals = vals
    self.avals = avals
    self.out_aval = out_aval
    self.params = params

  def emit_fallback(self, fb: FunctionBuilder):
    # TODO: This is super fragile: just to see what is going on.
    op_name = f"jax_unrecognized.{self.primitive.name}"
    # TODO: Handle multi-result.
    result_types = [fb.b.convert_aval_to_ir_type(self.out_aval)]
    attributes = {}
    for k, v in self.params.items():
      if isinstance(v, tuple):
        items = []
        for item in v:
          if isinstance(item, int):
            items.append(
                ir.IntegerAttr.get(ir.IntegerType.get_signless(64), item))
        attributes[k] = ir.ArrayAttr.get(items)
    op = ir.Operation.create(op_name,
                             result_types,
                             self.vals,
                             attributes=attributes)
    return op.result


PrimitiveHandler = Callable[[FunctionBuilder, PrimitiveInvocation], Any]
PrimitiveHandlerTable = Dict[core.Primitive, PrimitiveHandler]


def _add_handler(table: PrimitiveHandlerTable, primitive: core.Primitive):

  def decorator(f):
    table[primitive] = f
    return f

  return decorator


HLO_HANDLERS: PrimitiveHandlerTable = {}


@_add_handler(HLO_HANDLERS, lax.add_p)
def hlo_add(fb: FunctionBuilder, inv: PrimitiveInvocation):
  # TODO: Most of this can be generically meta-programmed.
  b = fb.b
  rt = b.convert_aval_to_ir_type(inv.out_aval)
  op = chlo.BroadcastAddOp(rt,
                           inv.vals[0],
                           inv.vals[1],
                           broadcast_dimensions=None)
  return op.result


@_add_handler(HLO_HANDLERS, lax.abs_p)
def hlo_abs(fb: FunctionBuilder, inv: PrimitiveInvocation):
  # TODO: Most of this can be generically meta-programmed.
  b = fb.b
  rt = b.convert_aval_to_ir_type(inv.out_aval)
  op = mhlo.AbsOp(rt, inv.vals[0])
  return op.result
