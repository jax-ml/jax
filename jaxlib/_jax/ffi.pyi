# Copyright 2025 The JAX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import enum
from typing import Any

class Buffer:
  @property
  def dtype(self) -> Any: ...
  @property
  def ndim(self) -> int: ...
  @property
  def shape(self) -> tuple[int, ...]: ...
  @property
  def writeable(self) -> bool: ...
  def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any: ...
  def __cuda_array_interface__(self) -> Any: ...
  def __dlpack__(
      self,
      stream: Any = None,
      max_version: Any = None,
      dl_device: Any = None,
      copy: Any = None,
  ) -> Any: ...
  def __dlpack_device__(self) -> tuple[int, int]: ...

class ExecutionStage(enum.IntEnum):
  INSTANTIATE = ...
  PREPARE = ...
  INITIALIZE = ...
  EXECUTE = ...

class ExecutionContext:
  def stage(self) -> ExecutionStage: ...
  def stream(self) -> int: ...
