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

"""A small libary of helpers for use in jaxlib to build MLIR operations."""

from collections.abc import Sequence
from functools import partial
from typing import Callable, Optional, Union

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.stablehlo as hlo
import numpy as np


_dtype_to_ir_type_factory : dict[np.dtype, Callable[[], ir.Type]] = {
  np.dtype(np.bool_): partial(ir.IntegerType.get_signless, 1),
  np.dtype(np.int8): partial(ir.IntegerType.get_signless, 8),
  np.dtype(np.int16): partial(ir.IntegerType.get_signless, 16),
  np.dtype(np.int32): partial(ir.IntegerType.get_signless, 32),
  np.dtype(np.int64): partial(ir.IntegerType.get_signless, 64),
  np.dtype(np.uint8): partial(ir.IntegerType.get_unsigned, 8),
  np.dtype(np.uint16): partial(ir.IntegerType.get_unsigned, 16),
  np.dtype(np.uint32): partial(ir.IntegerType.get_unsigned, 32),
  np.dtype(np.uint64): partial(ir.IntegerType.get_unsigned, 64),
  np.dtype(np.float16): ir.F16Type.get,
  np.dtype(np.float32): ir.F32Type.get,
  np.dtype(np.float64): ir.F64Type.get,
  np.dtype(np.complex64): lambda: ir.ComplexType.get(ir.F32Type.get()),
  np.dtype(np.complex128): lambda: ir.ComplexType.get(ir.F64Type.get()),
}
def dtype_to_ir_type(dtype) -> ir.Type:
  return _dtype_to_ir_type_factory[np.dtype(dtype)]()


def shape_dtype_to_ir_type(shape: Sequence[int], dtype) -> ir.Type:
  return ir.RankedTensorType.get(shape, dtype_to_ir_type(dtype))


# When we generate custom calls with dynamic shapes we have to pass
# both the result_types, with ir.ShapedType.get_dynamic_size in place of
# the dynamic dimensions, and also result_shapes, which are ir.Value
# representing 1D int32 tensors. If all the shapes are static we can use
# result_shapes=None. We first construct for each result a pair with the shape
# and element type, the shape containing either integer or ir.Value.
DimensionSize = Union[int, ir.Value]  # an ir.Value if not static dimension
ShapeTypePair = tuple[Sequence[DimensionSize], ir.Type]

def mk_result_types_and_shapes(
    shape_type_pairs: Sequence[ShapeTypePair]
) -> tuple[list[ir.Type], Optional[list[ir.Value]]]:
  result_types: list[ir.Type] = []
  result_shapes: list[ir.Value] = []
  has_dynamic_shapes = any(
      any(not isinstance(d, int) for d in rshape)
      for rshape, _ in shape_type_pairs)
  for (rshape, rtype) in shape_type_pairs:
    if has_dynamic_shapes:
      result_shapes.append(shape_tensor(rshape))
    result_types.append(
        ir.RankedTensorType.get(
            [d if isinstance(d, int) else ir.ShapedType.get_dynamic_size()
             for d in rshape],
            rtype))
  return (result_types,
          result_shapes if has_dynamic_shapes else None)

# TODO(necula): share this with mlir.shape_tensor
def shape_tensor(sizes: Sequence[Union[int, ir.Value]]) -> ir.Value:
  int1d = shape_dtype_to_ir_type((1,), np.int32)
  i32_type = shape_dtype_to_ir_type((), np.int32)
  def dim_to_i32x1(d):
    if type(d) is int:
      return hlo_const(np.array([d], dtype=np.int32))
    else:
      if d.type != i32_type:
        d = hlo.ConvertOp(i32_type, d).result
      return hlo.ReshapeOp(int1d, d).result
  ds = [dim_to_i32x1(sz) for sz in sizes]
  if not ds:
    return hlo_const(np.array([], np.int32))
  elif len(ds) == 1:
    return ds[0]
  else:
    return hlo.ConcatenateOp(
        ds, ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)).result

def hlo_const(x: np.ndarray) -> ir.Value:
  assert isinstance(x, np.ndarray)
  return hlo.ConstantOp(
      ir.DenseElementsAttr.get(x, type=dtype_to_ir_type(x.dtype))).result

def hlo_u8(x: int):
  return hlo_const(np.array(x, dtype=np.uint8))
def hlo_s32(x: int):
  return hlo_const(np.array(x, dtype=np.int32))

def ensure_hlo_s32(x: DimensionSize):
  return hlo_s32(x) if isinstance(x, int) else x


def hlo_min(x: DimensionSize, y: DimensionSize) -> DimensionSize:
  if type(x) is int:
    if type(y) is int:
      return min(x, y)
    x = hlo_s32(x)
  if type(y) is int:
    y = hlo_s32(y)
  return hlo.MinOp(x, y).result


def hlo_add(x: DimensionSize, y: DimensionSize) -> DimensionSize:
  if type(x) is int:
    if type(y) is int:
      return x + y
    x = hlo_s32(x)
  if type(y) is int:
    y = hlo_s32(y)
  return hlo.AddOp(x, y).result


# TODO(necula): share this with mlir.custom_call
def custom_call(
    call_target_name: Union[str, bytes],
    out_types: Sequence[ir.Type],
    operands: Sequence[ir.Value],
    operand_layouts: Optional[Sequence[Sequence[int]]] = None,
    result_layouts: Optional[Sequence[Sequence[int]]] = None,
    backend_config: str = "",
    has_side_effect: bool = False,
    api_version: int = 2,
    operand_output_aliases: dict[int, int] = {},
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
  attributes = dict(
      call_target_name=ir.StringAttr.get(call_target_name),
      has_side_effect=ir.BoolAttr.get(has_side_effect),
      backend_config=ir.StringAttr.get(backend_config),
      api_version=ir.IntegerAttr.get(
          ir.IntegerType.get_signless(32), api_version),
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
