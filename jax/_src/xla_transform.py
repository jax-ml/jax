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
import logging

from jax._src import xla_bridge
from jax._src.lib import _xla, xla_client

logger = logging.getLogger(__name__)


class PipelineStage(enum.Enum):
  PRE_SCHEDULER = 0
  POST_SCHEDULER = 1


def register_hlo_module_transformation(
    callback: Callable[[bytes], bytes | None],
    *,
    name: str,
    stage: PipelineStage = PipelineStage.PRE_SCHEDULER,
    platforms: Sequence[str] | str | None = None,
) -> None:
  """Register a custom compiler pass that transforms HLO modules.

  The registered pass will be called during XLA compilation at the specified
  pipeline stage. The callback receives a serialized ``HloModuleProto`` as
  bytes and should return either:

  - Modified serialized ``HloModuleProto`` bytes if the module was changed.
  - ``None`` if no changes were made.

  Multiple registration calls at the same stage (with different callbacks) will
  be added to a queue, and be invoked in the order they were registered.

  Args:
    callback: A function ``(bytes) -> bytes | None`` that receives a serialized
      HloModuleProto and optionally returns a modified one.
    name: A name for the compiler pass.
    stage: The pipeline stage at which the pass runs. Must be a
      ``PipelineStage`` enum.
    platforms: The list of platforms to register the pass for (e.g. ``"cpu"``,
      ``"tpu"``). If ``None``, the pass is registered for all known backends by
      default. Can be a single platform string or a sequence of strings.
  """
  if not isinstance(stage, PipelineStage):
    raise TypeError(f"stage must be a PipelineStage enum, got {type(stage)}")
  stage_int = stage.value

  # Canonicalize platforms to a list of strings early.
  if platforms is None:
    platforms_list = ["cpu"] + list(xla_bridge._backend_factories.keys())
    platforms_list = list(dict.fromkeys(platforms_list))
  elif isinstance(platforms, str):
    platforms_list = [platforms]
  else:
    platforms_list = list(dict.fromkeys(platforms))

  # Register directly for in-process backends (e.g., CPU).
  if "cpu" in platforms_list:
    _xla.register_xla_transform(name, stage_int, callback)

  # Register via the PJRT C API extension on each plugin (e.g. TPU, GPU). The
  # extension is keyed on PJRT_Api*, not a client instance, so we register by
  # plugin name — this works under AOT compilation (CompileOnlyPyClient has no
  # underlying PjRtCApiClient) as well as on real hardware.
  def register_on_plugin(platform: str) -> None:
    try:
      _xla.register_xla_transform_c_api_by_plugin(
          platform, name, stage_int, callback)
    except RuntimeError:
      logger.debug(
          "Could not register XLA transform via C API for platform %s",
          platform,
      )

  for platform in platforms_list:
    if platform == "cpu":
      continue
    if xla_client.pjrt_plugin_loaded(platform):
      register_on_plugin(platform)
    else:
      # Plugin not loaded yet — register once it is. The callback fires for
      # every loaded plugin, so guard so we only register on this platform
      # once (after its plugin has loaded).
      done = []
      def _on_load(c_api, p=platform, done=done):
        del c_api
        if not done and xla_client.pjrt_plugin_loaded(p):
          register_on_plugin(p)
          done.append(True)
      xla_bridge.register_plugin_callbacks(_on_load)


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
  if not isinstance(stage, PipelineStage):
    raise TypeError(f"stage must be a PipelineStage enum, got {type(stage)}")
  stage_int = stage.value

  if platforms is None:
    platforms_list = ["cpu"] + list(xla_bridge._backend_factories.keys())
    platforms_list = list(dict.fromkeys(platforms_list))
  elif isinstance(platforms, str):
    platforms_list = [platforms]
  else:
    platforms_list = list(dict.fromkeys(platforms))

  cleared = False
  if "cpu" in platforms_list:
    cleared |= _xla.clear_xla_transform(name, stage_int)

  # Also clear on initialized plugin clients.
  for platform in platforms_list:
    if platform != "cpu":
      try:
        initialized_backends = xla_bridge.backends()
        if platform in initialized_backends:
          client = initialized_backends[platform]
          cleared |= _xla.clear_xla_transform_c_api(client, name, stage_int)
      except RuntimeError:
        pass
      except ValueError:
        pass

  return cleared
