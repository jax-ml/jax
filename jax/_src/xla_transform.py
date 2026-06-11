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
from jax._src.lib import _xla

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
  if _xla is None:
    raise NotImplementedError(
        "register_hlo_module_transformation requires jaxlib version >= 0.10.2"
    )
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

  # Also register via the PJRT C API on all plugin clients when they are
  # initialized (e.g., TPU, GPU).
  def register_on_client(client):
    if client.platform in platforms_list:
      try:
        _xla.register_xla_transform_c_api(client, name, stage_int, callback)
      except RuntimeError:
        logger.debug(
            "Could not register XLA transform via C API for client platform %s",
            client.platform,
        )

  # Only register the initialization hook if we need to register on some plugin backends.
  if any(p != "cpu" for p in platforms_list):
    xla_bridge.register_backend_initialization_hook(register_on_client)
