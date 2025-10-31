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

from typing import Any, Generic, TypeVar

unset: object = ...

_T = TypeVar("_T")

class Config(Generic[_T]):
  def __init__(
      self,
      name: str,
      value: _T,
      *,
      include_in_jit_key: bool = ...,
      include_in_trace_context: bool = ...,
  ) -> None: ...
  @property
  def value(self) -> _T: ...
  @property
  def name(self) -> str: ...
  def get_local(self) -> Any: ...
  def get_global(self) -> _T: ...
  def set_local(self, value: Any | None) -> None: ...
  def swap_local(self, value: Any | None) -> Any: ...
  def set_global(self, value: Any | None) -> None: ...

def trace_context() -> tuple: ...
