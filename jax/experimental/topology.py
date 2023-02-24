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
from jax._src.lib import xla_client as xc
from typing import (Sequence, List, Tuple, Optional, Mapping, Dict, Set,
                    FrozenSet, Union, cast)

Device = xc.Device

class Topology(metaclass=abc.ABCMeta):
  """Abstract `Topology` interface which exposes the connectivity of physical
  hardware to jax users.
  """

  @abc.abstractmethod
  def devices(self) -> List[Device]:
    """A full list of all devices in this `Topology`."""
    raise NotImplementedError('Subclasses should implement this method.')


class DeviceListTopology(Topology):
  """Wraps a list of devices as a `Topology`."""

  def __init__(self, devices: List[Device]):
    self._devices = devices

  def devices(self) -> List[Device]:
    return self._devices


class _LoweringOnlyDevice:

  def __init__(self, client, i, platform):
    self.client = client
    self.id = i
    self.process_index = 0
    self.platform = platform


class _LoweringOnlyClient:

  def __init__(self, n, platform):
    self.platform = platform
    self.devices = [_LoweringOnlyDevice(self, i, platform) for i in range(n)]

  def process_index(self):
    return 0

  def device_count(self):
    return len(self.devices)


def lowering_topology(n, platform):
  """Constructs a topology usable in lowering for the specified platform."""
  return DeviceListTopology(_LoweringOnlyClient(n, platform).devices)
