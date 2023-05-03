# Copyright 2022 The JAX Authors.
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

# Helpers for building MLIR operators
from typing import Dict, Optional, Sequence, Union
import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.stablehlo as hlo
import numpy as np


def custom_call(
    call_target_name: Union[str, bytes],
    out_types: Sequence[ir.Type],
    operands: Sequence[ir.Value],
    operand_layouts: Optional[Sequence[Sequence[int]]] = None,
    result_layouts: Optional[Sequence[Sequence[int]]] = None,
    backend_config: str = "",
    has_side_effect: bool = False,
    api_version: int = 2,
    operand_output_aliases: Dict[int, int] = {},
    result_shapes: Optional[Sequence[ir.Value]] = None,
) -> Sequence[ir.Value]:
  """Wraps a hlo.CustomCall

  Args:
  ...
  operand_output_alias: a dictionary mapping input operand index -> output
    index that must alias.
  result_shapes: 1D integer tensors that represent the result shapes, to be
      used when the results have dynamic shapes. Its length must
      match the number of the results. They are appended to the list
      of operands.
  """
  i32_type = ir.IntegerType.get_signless(32)
  attributes = dict(
      call_target_name=ir.StringAttr.get(call_target_name),
      has_side_effect=ir.BoolAttr.get(has_side_effect),
      backend_config=ir.StringAttr.get(backend_config),
      api_version=ir.IntegerAttr.get(i32_type, api_version),
      called_computations=ir.ArrayAttr.get([]),
      output_operand_aliases=ir.ArrayAttr.get([
          hlo.OutputOperandAlias.get(
              # if len(out_types) == 1 then the aliasing refers implicitly to
              # the only output.
              output_tuple_indices=[output_idx] if len(out_types) > 1 else [],
              operand_index=input_idx,
              operand_tuple_indices=[])
          for input_idx, output_idx in operand_output_aliases.items()
      ])
  )
  if result_shapes is not None:
    # We add the result_shapes at the end of the operands, and must pass
    # the indices_of_output_operands attribute.
    assert len(result_shapes) == len(out_types), (result_shapes, out_types)
    # We will add the result_shapes at the end of the operands
    attributes["indices_of_shape_operands"] = ir.DenseIntElementsAttr.get(
        np.asarray(list(range(len(operands), len(operands) + len(result_shapes))),
                   dtype=np.int64))
    if operand_layouts is not None:
      assert len(operand_layouts) == len(operands), (operand_layouts, operands)
      operand_layouts = list(operand_layouts) + [(0,)] * len(result_shapes)
    operands = list(operands) + list(result_shapes)

  if operand_layouts is not None:
    assert result_layouts is not None
    assert len(result_layouts) == len(out_types), (result_layouts, out_types)
    attributes["operand_layouts"] = ir.ArrayAttr.get([
        ir.DenseIntElementsAttr.get(
            np.atleast_1d(np.asarray(l, dtype=np.int64)),
            type=ir.IndexType.get()) for l in operand_layouts
    ])
    attributes["result_layouts"] = ir.ArrayAttr.get([
        ir.DenseIntElementsAttr.get(
            np.atleast_1d(np.asarray(l, dtype=np.int64)),
            type=ir.IndexType.get()) for l in result_layouts
    ])

  # TODO(necula): CustomCall constructor does not yet support
  # indices_of_shape_operands, so we use the generic builder

  # The generic builder is pickier about the type of the operands, and some
  # of the callers did not call .result
  operands = [opnd if isinstance(opnd, ir.Value) else opnd.result
              for opnd in operands]
  out = hlo.CustomCallOp.build_generic(results=out_types,
                                       operands=operands, attributes=attributes)
  return out.results
