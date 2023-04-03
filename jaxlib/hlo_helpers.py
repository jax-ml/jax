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
    operand_layouts: Sequence[Sequence[int]],
    result_layouts: Sequence[Sequence[int]],
    backend_config: Optional[str] = None,
    has_side_effect: bool = False,
    api_version: int = 2,
    operand_output_aliases: Dict[int, int] = {},
    indices_of_shape_operands: Sequence[int] = (),
) -> Union[ir.Value, Sequence[ir.Value]]:
  """Less-verbose helper for building a CustomCallOp.

  Once https://github.com/llvm/llvm-project/issues/54932 is fixed, this helper
  may be able to go away.

  Args:
  ...
  operand_output_alias: a dictionary mapping input numbers -> output numbers
    that must alias.
  indices_of_shape_operands: in presence of dynamic shapes, must pass in the
    output shapes as some of the operands. These are the indices of those
    operands.
  """
  i32_type = ir.IntegerType.get_signless(32)
  results = (out_types
             if len(out_types) == 1 else [ir.TupleType.get_tuple(out_types)])
  attributes = dict(
      call_target_name=ir.StringAttr.get(call_target_name),
      has_side_effect=ir.BoolAttr.get(has_side_effect),
      backend_config=ir.StringAttr.get(
          "" if backend_config is None else backend_config),
      api_version=ir.IntegerAttr.get(i32_type, api_version),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(
              np.atleast_1d(np.asarray(l, dtype=np.int64)),
              type=ir.IndexType.get()) for l in operand_layouts
      ]),
      result_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(
              np.atleast_1d(np.asarray(l, dtype=np.int64)),
              type=ir.IndexType.get()) for l in result_layouts
      ]),
      output_operand_aliases=ir.ArrayAttr.get([
          hlo.OutputOperandAlias.get(
              output_tuple_indices=[] if len(out_types) == 1 else [output],
              operand_index=input,
              operand_tuple_indices=[])
          for input, output in operand_output_aliases.items()
      ])
  )
  if indices_of_shape_operands:
    attributes["indices_of_shape_operands"] = ir.DenseIntElementsAttr.get(
        np.asarray(indices_of_shape_operands, dtype=np.int64))

  # TODO(necula): CustomCall constructor does not yet support
  # indices_of_shape_operands, so we use the generic builder

  # The generic builder is pickier about the type of the operands, and some
  # of the callers did not call .result
  operands = [opnd if isinstance(opnd, ir.Value) else opnd.result
              for opnd in operands]
  out = hlo.CustomCallOp.build_generic(results=results, operands=operands, attributes=attributes)
  if len(out_types) == 1:
    return out.result
  else:
    return [
        hlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
        for i in range(len(out_types))
    ]
