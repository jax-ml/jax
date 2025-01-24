# Copyright 2024 The JAX Authors.
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

import threading
from typing import Protocol

from jaxlib.triton import dialect  # noqa: F401  # pytype: disable=import-error


class CompilationResult(Protocol):
  asm: bytes
  smem_bytes: int
  cluster_dim_x: int
  cluster_dim_y: int
  cluster_dim_z: int


class CompilationHandler(Protocol):

  def __call__(
      self,
      module: bytes,
      arch_name: str,
      num_warps: int,
      num_ctas: int,
      num_stages: int,
  ) -> CompilationResult:
    ...


_compilation_handlers: dict[str, CompilationHandler] = {}
_compilation_handlers_lock = threading.Lock()


def register_compilation_handler(
    platform: str, handler: CompilationHandler
) -> None:
  with _compilation_handlers_lock:
    if existing_handler := _compilation_handlers.get(platform):
      raise RuntimeError(
          f'Platform {platform} already has a Triton compilation handler:'
          f' {existing_handler}'
      )
    _compilation_handlers[platform] = handler


def compile(
    platform: str,
    module: bytes,
    arch_name: str,
    *,
    num_warps: int,
    num_ctas: int,
    num_stages: int,
) -> CompilationResult:
  with _compilation_handlers_lock:
    handler = _compilation_handlers.get(platform)
  if handler is None:
    raise RuntimeError(
        f'Platform {platform} does not have a Triton compilation handler'
    )
  return handler(module, arch_name, num_warps, num_ctas, num_stages)
