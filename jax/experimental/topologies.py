# Copyright 2023 The JAX Authors.
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

from __future__ import annotations

import logging
from collections.abc import Sequence

import jax
from jax._src import config
from jax._src import xla_bridge as xb
from jax._src.lib import _jax
from jax.experimental import mesh_utils

Device = _jax.Device

logger = logging.getLogger(__name__)


class TopologyDescription:
  def __init__(self, devices: list[Device]):
    self.devices: list[Device] = devices


def get_attached_topology(platform=None) -> TopologyDescription:
  return TopologyDescription(jax.devices(backend=platform))


def _get_autotuning_client():
  """Returns a real backend client for kernel autotuning during cross-compilation."""
  backend_name = config.cross_compile_autotuning_backend.value
  if backend_name is None:
    return None
  try:
    return xb.get_backend(backend_name)
  except Exception:
    logger.warning(
        "Could not get autotuning backend %r; cross-compiling without "
        "autotuning.", backend_name)
    return None


def get_topology_desc(
    topology_name: str = "", platform: str | None = None, **kwargs
) -> TopologyDescription:
  autotuning_client = _get_autotuning_client()
  if platform == "tpu" or platform is None:
    return TopologyDescription(
        xb.make_pjrt_tpu_topology(
            topology_name, **kwargs
        )._make_compile_only_devices(autotuning_client=autotuning_client)
    )
  try:
    topology = xb.make_pjrt_topology(platform, topology_name, **kwargs)
    return TopologyDescription(topology._make_compile_only_devices(autotuning_client=autotuning_client))  # pytype: disable=attribute-error
  except _jax.JaxRuntimeError as e:
    msg, *_ = e.args
    if msg.startswith("UNIMPLEMENTED"):
      raise NotImplementedError(msg) from e
    else:
      raise


# -- future mesh_utils --


def make_mesh(
    topo: TopologyDescription,
    mesh_shape: Sequence[int],
    axis_names: tuple[str, ...],
    *,
    contiguous_submeshes: bool = False
) -> jax.sharding.Mesh:
  devices = mesh_utils.create_device_mesh(
      mesh_shape, list(topo.devices), contiguous_submeshes=contiguous_submeshes)
  return jax.sharding.Mesh(devices, axis_names)
