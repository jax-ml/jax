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

"""Compiler params for `lax.fusion` and `lax.pipeline`.

Each primitive takes an optional `compiler_params` frozen dataclass matching
the primitive's params class or protocol. Platform lowering rules read
`compiler_params` directly and may dispatch to handlers registered per params
class.

This module holds what the primitives share: the base protocol, argument
validation, and helpers for lowering state.

Register a lowering that reads the nesting state of the primitive modules, for
example whether a `lax.pipeline` body is being lowered, with
`mlir.register_lowering(..., cacheable=False)`, for both the default and the
platform rule: `mlir.jaxpr_subcomp` caches lowerings by primitive, equation
context, avals, and params.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Protocol, runtime_checkable


@runtime_checkable
class CompilerParams(Protocol):
  """Base protocol for the compiler params of `lax.fusion` and `lax.pipeline`.

  Implementations target one platform and are frozen dataclasses, so they are
  hashable primitive parameters.
  """

  __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


def check_compiler_params(
    params: Any,
    api_name: str,
    expected: type[Any] = CompilerParams,
) -> Any:
  """Validates the `compiler_params` argument of `api_name`.

  Args:
    params: `None` or a frozen dataclass instance for `api_name`.
    api_name: Public API name used in error messages.
    expected: Params class or protocol expected by the primitive.

  Returns:
    `params` unchanged (`None` or the validated frozen dataclass).

  Raises:
    TypeError: If `params` is not `None` and is not a frozen dataclass
      implementing `expected`.
  """
  if params is None:
    return None
  if not isinstance(params, expected):
    raise TypeError(
        f"{api_name}: compiler_params must be None or an object implementing"
        f" {expected.__name__}, got {type(params).__name__}: {params!r}."
    )
  dataclass_params = getattr(type(params), "__dataclass_params__", None)
  if not dataclasses.is_dataclass(params) or not getattr(
      dataclass_params, "frozen", False
  ):
    raise TypeError(
        f"{api_name}: compiler_params must be a frozen dataclass, got"
        f" {type(params).__name__}."
    )
  return params


# Key of the `module_state` dict in `ModuleContext.cached_primitive_lowerings`.
_MODULE_STATE_KEY = object()


def module_state(mod_ctx: Any) -> dict[str, Any]:
  """Returns per module lowering state preserved across isolated contexts.

  Args:
    mod_ctx: MLIR `ModuleContext` for the module being lowered.

  Returns:
    A mutable dict shared across `mod_ctx` and any isolated contexts derived
    from it, for example the counter for `lax.fusion` ids.
  """
  cache = mod_ctx.cached_primitive_lowerings
  if _MODULE_STATE_KEY not in cache:
    cache[_MODULE_STATE_KEY] = {}
  return cache[_MODULE_STATE_KEY]


def isolated_lowering_context(mod_ctx: Any) -> Any:
  """Returns `mod_ctx` with empty lowering caches and the same module state.

  Lowers each `lax.fusion` or `lax.pipeline` body into callees of its own. The
  lowering caches key on the jaxpr, while a body's lowering depends on its
  nesting and params, and XLA writes a call's fusion group into its callee.

  Args:
    mod_ctx: Parent MLIR `ModuleContext`.

  Returns:
    A copy of `mod_ctx` with fresh lowering caches that shares `module_state`.
  """
  return mod_ctx.replace(
      lowering_cache={},
      cached_primitive_lowerings={_MODULE_STATE_KEY: module_state(mod_ctx)},
  )
