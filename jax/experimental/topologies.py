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

import abc
from typing import (Sequence, List, Tuple, Optional, Mapping, Dict, Set,
                    FrozenSet, Union, cast)

import numpy as np

import jax
from jax.experimental import mesh_utils
from jax._src.lib import xla_client as xc

Device = xc.Device


class Topology(abc.ABC):
  def __init__(self, devices: List[Device]):
    self.devices: List[Device] = devices


def get_attached_topology(platform=None) -> Topology:
  return Topology(jax.devices(backend=platform))


# -- future mesh_utils --

def make_mesh(topo: Topology, mesh_shape: Sequence[int],
              axis_names: Tuple[str, ...],
              *, contiguous_submeshes: bool = False
              ) -> jax.sharding.Mesh:
  devices = mesh_utils.create_device_mesh(
      mesh_shape, list(topo.devices), contiguous_submeshes=contiguous_submeshes)
  return jax.sharding.Mesh(devices, axis_names)
