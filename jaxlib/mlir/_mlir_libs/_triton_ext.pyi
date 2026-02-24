# Copyright 2025 The JAX Authors.
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

from jaxlib.mlir import ir

def register_dialect(context: ir.Context, load: bool = ...) -> None: ...

class PointerType(ir.Type):
  def __init__(self, cast_from_type: ir.Type) -> None: ...

  static_typeid: ir.TypeID = ...
  """static_typeid(/) -> TypeID"""

  @property
  def typeid(self) -> ir.TypeID: ...
  def __repr__(self) -> str: ...
  @staticmethod
  def get(pointee_type: ir.Type, address_space: int) -> PointerType:
    """Creates a PointerType type."""

  @property
  def pointee_type(self) -> ir.Type: ...
  @property
  def address_space(self) -> int: ...

def infer_reduce_op_encoding(
    arg0: ir.Attribute, arg1: int, /
) -> ir.Attribute | None: ...
