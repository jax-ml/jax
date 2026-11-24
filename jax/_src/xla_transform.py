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

"""Internal implementation for jax.extend.xla compiler pass registration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import enum
import inspect
from typing import Protocol

from jax._src import xla_bridge
from jax._src.lib import _jax
from jax._src.lib import _xla


class PipelineStage(enum.Enum):
  PRE_SCHEDULER = 0
  POST_SCHEDULER = 1


class _HloTransformCallback(Protocol):
  """A transform callback that receives only the serialized HloModuleProto."""

  def __call__(self, hlo_module: bytes, /) -> bytes | None:
    ...


class _HloTransformCallbackWithConfig(Protocol):
  """A transform callback that also receives the serialized module config."""

  def __call__(
      self, hlo_module: bytes, /, *, module_config: bytes
  ) -> bytes | None:
    ...


def _normalize_platforms(
    platforms: Sequence[str] | str | None,
) -> list[str]:
  """Canonicalizes platforms to a list of platform strings."""
  if platforms is None:
    return list(xla_bridge.backends().keys())
  if isinstance(platforms, str):
    return [platforms]
  return list(set(platforms))


def _accepts_module_config(callback: Callable[..., bytes | None]) -> bool:
  """Whether `callback` can be called as ``callback(hlo_module, module_config=...)``.

  Raises:
    TypeError: if `callback` cannot take the module config by keyword but
      requires more than one positional argument, so it would fail when XLA
      invokes it during compilation.
  """
  try:
    params = inspect.signature(callback).parameters
  except (TypeError, ValueError):  # ValueError: no introspectable signature.
    return False
  param = params.get("module_config")
  if param is not None:
    by_keyword = param.kind in (
        inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
  else:
    by_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
  required_positional = [
      p for p in params.values()
      if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD)
      and p.default is inspect.Parameter.empty
      and not (by_keyword and p is param)
  ]
  if len(required_positional) > 1:
    raise TypeError(
        f"{callback!r} requires {len(required_positional)} positional "
        "arguments; an XLA transform callback takes the serialized HLO module "
        "positionally and, optionally, the module config as a keyword "
        "argument named `module_config`.")
  return by_keyword


def register_hlo_module_transformation(
    callback: _HloTransformCallback | _HloTransformCallbackWithConfig,
    *,
    name: str,
    stage: PipelineStage = PipelineStage.PRE_SCHEDULER,
    platforms: Sequence[str] | str | None = None,
) -> None:
  """Register a custom compiler pass that transforms HLO modules.

  The registered pass will be called during XLA compilation at the specified
  pipeline stage. The callback receives a serialized ``HloModuleProto`` as
  bytes, optionally followed by the serialized ``HloModuleConfigProto`` the
  module is being compiled with, and should return either:

  - Modified serialized ``HloModuleProto`` bytes if the module was changed.
  - ``None`` if no changes were made.

  Multiple registration calls at the same stage (with different callbacks) will
  be added to a queue, and be invoked in the order they were registered.

  Args:
    callback: A function ``(hlo_module: bytes) -> bytes | None`` or
      ``(hlo_module: bytes, *, module_config: bytes) -> bytes | None``. It
      receives a serialized HloModuleProto and optionally returns a modified
      one. If it declares a ``module_config`` parameter (or ``**kwargs``), that
      is passed by keyword and is the serialized
      ``HloModuleConfigProto`` the module is compiled with, which carries the
      FDO profile (``fdo_profile``, for PGLE a ``ProfiledInstructionsProto``),
      ``debug_options``, ``static_device_assignment``, etc. Plugins built before
      the module config was added to the PJRT transform extension always pass
      empty bytes.
    name: A name for the compiler pass.
    stage: The pipeline stage at which the pass runs. Must be a
      ``PipelineStage`` enum.
    platforms: The list of platforms to register the pass for (e.g. ``"cpu"``,
      ``"tpu"``). If ``None``, the pass is registered for all known backends by
      default. Can be a single platform string or a sequence of strings.
  """
  wants_module_config = _accepts_module_config(callback)

  # The default covers a jaxlib that predates the module config argument.
  def native_callback(hlo_module: bytes, module_config: bytes = b""):
    if wants_module_config:
      return callback(hlo_module, module_config=module_config)  # type: ignore[call-arg]
    return callback(hlo_module)  # type: ignore[call-arg]

  # Unconditionally trigger backend initialization so that all PJRT plugins are loaded
  # before we try to access them.
  xla_bridge.backends()

  for platform in _normalize_platforms(platforms):
    if platform == "cpu":
      # Register CPU directly as it's the only non-PJRT C API backend.
      _xla.register_xla_transform(name, stage.value, native_callback)
      continue
    c_api = _jax.get_pjrt_plugin(platform)
    _xla.register_xla_transform_c_api(
        c_api, name, stage.value, native_callback)


def clear_hlo_module_transformation(
    name: str,
    stage: PipelineStage = PipelineStage.PRE_SCHEDULER,
    platforms: Sequence[str] | str | None = None,
) -> bool:
  """Clear a registered custom compiler pass.

  Args:
    name: The name of the compiler pass to clear.
    stage: The pipeline stage of the pass. Must be a ``PipelineStage`` enum.
    platforms: The list of platforms to clear the pass for. If ``None``, the
      pass is cleared for all known backends.

  Returns:
    True if the pass was found and cleared, False otherwise.
  """
  # Unconditionally trigger backend initialization so that all PJRT plugins are loaded
  # before we try to access them.
  xla_bridge.backends()

  cleared = False
  for platform in _normalize_platforms(platforms):
    if platform == "cpu":
      # Clear CPU directly as it's the only non-PJRT C API backend.
      cleared |= _xla.clear_xla_transform(name, stage.value)
      continue
    c_api = _jax.get_pjrt_plugin(platform)
    cleared |= _xla.clear_xla_transform_c_api(c_api, name, stage.value)

  return cleared
