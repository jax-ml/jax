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

from collections.abc import Callable, Mapping

from jax.jaxlib._jax import Client

class ClientConnectionOptions:
  def __init__(self) -> None: ...
  @property
  def on_disconnect(self) -> Callable[[str], None] | None: ...
  @on_disconnect.setter
  def on_disconnect(self, arg: Callable[[str], None] | None) -> None: ...
  @property
  def on_connection_update(self) -> Callable[[str], None] | None: ...
  @on_connection_update.setter
  def on_connection_update(self, arg: Callable[[str], None] | None) -> None: ...
  @property
  def connection_timeout_in_seconds(self) -> int | None: ...
  @connection_timeout_in_seconds.setter
  def connection_timeout_in_seconds(self, arg: int | None) -> None: ...
  @property
  def initialization_data(self) -> dict[str, bytes | bool | int] | None: ...
  @initialization_data.setter
  def initialization_data(
      self, arg: Mapping[str, bytes | bool | int] | None
  ) -> None: ...

def get_client(
    proxy_server_address: str, options: ClientConnectionOptions
) -> Client: ...
