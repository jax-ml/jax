# ---
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
#
# ---

import core
from core import Op, InterpreterVal, Ty, ArrayTy

# === Mutable arrays ===

class MutableArray:
  def __init__(self, val):
    self.val = val

class MutableArrayTy(Ty):
  def __init__(self, array_ty: ArrayTy):
    assert isinstance(array_ty, ArrayTy)
    self.array_ty = array_ty

  def _setitem_(self, ref, idx, val):
    val = cast_if_scalar(val, self.array_ty)
    assert typeof(val) == self.array_ty
    assert idx == Ellipsis
    emit_op(Set(), (), (ref, val))

  def _getitem_(self, ref, idx):
    assert idx == Ellipsis, breakpoint()
    return emit_op(Get(), self.array_ty, (ref,))

  def __str__(self):
    return f"Ref[{self.array_ty}]"

class Set(Op):
  def eval(self, _, ref, val):
    ref.val = val
  def __str__(self): return "set"

class Get(Op):
  def eval(self, _, ref): return ref.val
  def __str__(self): return "get"
