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

import enum

class TransferGuardLevel(enum.Enum):
  ALLOW = 0

  LOG = 1

  DISALLOW = 2

  LOG_EXPLICIT = 3

  DISALLOW_EXPLICIT = 4

class GarbageCollectionGuardLevel(enum.Enum):
  ALLOW = 0

  LOG = 1

  FATAL = 2

class GuardState:
  @property
  def host_to_device(self) -> TransferGuardLevel | None: ...
  @host_to_device.setter
  def host_to_device(self, arg: TransferGuardLevel | None) -> None: ...
  @property
  def device_to_device(self) -> TransferGuardLevel | None: ...
  @device_to_device.setter
  def device_to_device(self, arg: TransferGuardLevel | None) -> None: ...
  @property
  def device_to_host(self) -> TransferGuardLevel | None: ...
  @device_to_host.setter
  def device_to_host(self, arg: TransferGuardLevel | None) -> None: ...
  @property
  def explicit_device_put(self) -> bool: ...
  @explicit_device_put.setter
  def explicit_device_put(self, arg: bool, /) -> None: ...
  @property
  def explicit_device_get(self) -> bool: ...
  @explicit_device_get.setter
  def explicit_device_get(self, arg: bool, /) -> None: ...
  @property
  def garbage_collect_array(self) -> GarbageCollectionGuardLevel | None: ...
  @garbage_collect_array.setter
  def garbage_collect_array(
      self, arg: GarbageCollectionGuardLevel | None
  ) -> None: ...

def global_state() -> GuardState: ...
def thread_local_state() -> GuardState: ...
def update_thread_guard_global_state(set_thread_id: bool | None) -> None: ...
