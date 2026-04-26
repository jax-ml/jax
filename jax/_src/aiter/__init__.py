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
"""
Public exports for the `jax_aiter.mha` sub-package.

Uses unified AITER entry point (aiter::mha_fwd / aiter::mha_bwd) for both
batch and variable-length attention. CK vs ASM v3 dispatch handled internally
by AITER.

Public API:
    flash_attn_func: Batch flash attention with custom_vjp
    flash_attn_varlen: Variable-length flash attention with custom_vjp
"""

try:
  from .aiter_mha import (
    flash_attn_func,
    flash_attn_varlen,
  )
except (ImportError, RuntimeError, OSError):
  flash_attn_func = None  # type: ignore[assignment]
  flash_attn_varlen = None  # type: ignore[assignment]

__all__ = [
  "flash_attn_func",
  "flash_attn_varlen",
]
