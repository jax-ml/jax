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

from mlir import ir

def register_dialect(context: ir.Context, load: bool = ...) -> None: ...
def private_has_communication(arg: ir.Operation, /) -> tuple[bool, bool]: ...
def private_set_arg_attr(
    arg0: ir.Operation, arg1: int, arg2: str, arg3: ir.Attribute, /
) -> None: ...

class Float8EXMYType(ir.Type):
  @staticmethod
  def isinstance(other_type: ir.Type) -> bool: ...
  def __repr__(self) -> str: ...
  @staticmethod
  def get(
      exmy_type: ir.Type | None = None, ctx: ir.Context | None = None
  ) -> Float8EXMYType: ...
  @property
  def underlying_type(self) -> ir.Type: ...
