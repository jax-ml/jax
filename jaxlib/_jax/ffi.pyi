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

"""Python bindings for the XLA FFI."""

import enum
import numpy
import typing_extensions

class Buffer:
  @property
  def dtype(self) -> numpy.dtype: ...
  @property
  def ndim(self) -> int: ...
  @property
  def shape(self) -> tuple: ...
  @property
  def writeable(self) -> bool: ...
  def __array__(
      self, dtype: object | None = ..., copy: object | None = ...
  ) -> numpy.ndarray: ...
  @property
  def __cuda_array_interface__(self) -> dict: ...
  def __dlpack__(
      self,
      stream: object | None = ...,
      max_version: object | None = ...,
      dl_device: object | None = ...,
      copy: object | None = ...,
  ) -> typing_extensions.CapsuleType: ...
  def __dlpack_device__(self) -> tuple: ...

class ExecutionStage(enum.Enum):
  INSTANTIATE = 0

  PREPARE = 1

  INITIALIZE = 2

  EXECUTE = 3

class ExecutionContext:
  @property
  def stage(self) -> ExecutionStage: ...
  @property
  def stream(self) -> int: ...
