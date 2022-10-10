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

# Helpers for building MHLO operators

from typing import Dict, Optional, Sequence, Union
import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.mhlo as mhlo
import numpy as np


def custom_call(
    call_target_name: str,
    out_types: Sequence[ir.Type],
    operands: Sequence[ir.Value],
    operand_layouts: Sequence[Sequence[int]],
    result_layouts: Sequence[Sequence[int]],
    backend_config: Optional[str] = None,
    has_side_effect: bool = False,
    api_version: int = 2,
    operand_output_aliases: Dict[int, int] = {},
) -> Union[ir.Value, Sequence[ir.Value]]:
  """Less-verbose helper for building an MHLO custom call op.

  Once https://github.com/llvm/llvm-project/issues/54932 is fixed, this helper
  may be able to go away.

  Args:
  ...
  operand_output_alias: a dictionary mapping input numbers -> output numbers
    that must alias.
  """
  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      (out_types
       if len(out_types) == 1 else [ir.TupleType.get_tuple(out_types)]),
      operands,
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
          mhlo.OutputOperandAlias.get(
              output_tuple_indices=[] if len(out_types) == 1 else [output],
              operand_index=input,
              operand_tuple_indices=[])
          for input, output in operand_output_aliases.items()
      ]))
  if len(out_types) == 1:
    return out.result
  else:
    return [
        mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
        for i in range(len(out_types))
    ]
