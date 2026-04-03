# Copyright 2026 The JAX Authors.
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

from collections.abc import Sequence
import types
from typing import overload

class TritonKernel:
  def __init__(
      self,
      arg0: str,
      arg1: int,
      arg2: int,
      arg3: int,
      arg4: str,
      arg5: str,
      arg6: int,
      /,
  ) -> None: ...

class TritonParameter:
  pass

def create_array_parameter(arg0: int, arg1: int, /) -> TritonParameter: ...
@overload
def create_scalar_parameter(arg0: bool, arg1: str, /) -> TritonParameter: ...
@overload
def create_scalar_parameter(arg0: int, arg1: str, /) -> TritonParameter: ...
@overload
def create_scalar_parameter(arg0: float, arg1: str, /) -> TritonParameter: ...

class TritonKernelCall:
  def __init__(
      self,
      arg0: TritonKernel,
      arg1: int,
      arg2: int,
      arg3: int,
      arg4: Sequence[TritonParameter],
      /,
  ) -> None: ...
  def to_proto(self, arg0: str, arg1: bytes, /) -> bytes: ...

class TritonAutotunedKernelCall:
  def __init__(
      self,
      arg0: str,
      arg1: Sequence[tuple[TritonKernelCall, str]],
      arg2: Sequence[tuple[int, int, int]],
      /,
  ) -> None: ...
  def to_proto(self, arg0: str, arg1: bytes, /) -> bytes: ...

def get_custom_call() -> types.CapsuleType: ...
def get_compute_capability(arg: int, /) -> int: ...
def get_arch_details(arg: int, /) -> str: ...
def get_serialized_metadata(arg: bytes, /) -> bytes: ...
