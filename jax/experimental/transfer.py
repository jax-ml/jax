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
"""`jax.experimental.transfer`: DCN cross slice transfer."""

import jax
from typing import Any, TYPE_CHECKING
from jax._src.lib import xla_client as _xc
from jax._src.lib import xla_extension_version
from jax._src.util import use_cpp_class, use_cpp_method

class TransferConnection:
  """Represents a connection to exactly one peer."""

  @use_cpp_method()
  def _pull_flat(self, uuid, backend, xs_flat):
    raise NotImplementedError()

  def pull(self, uuid: int, xs: Any) -> Any:
    """Fetches a pytree of arrays from a remote device.

    Args:
       uuid: identifier for the request
       xs: A pytree of ShapeDtypeStruct.
    Returns:
       A pytree of arrays.
    """
    xs_flat, tree = jax.tree.flatten(xs)
    if not xs_flat:
      return xs
    backend = next(iter(xs_flat[0].sharding.device_set)).client
    return tree.unflatten(self._pull_flat(uuid, backend, xs_flat))


if not TYPE_CHECKING and xla_extension_version >= 305:
  TransferConnection = use_cpp_class(_xc._xla.TransferConnection)(TransferConnection)


class TransferServer:

  @use_cpp_method()
  def address(self) -> str:
    """Returns the address that this server can be connected to with."""
    raise NotImplementedError()

  @use_cpp_method()
  def _await_pull_flat(self, uuid, args: list[jax.Array]):
    raise NotImplementedError()

  @use_cpp_method()
  def connect(self, address: str) -> TransferConnection:
    """Creates a connection to a remote server."""
    raise NotImplementedError()

  def await_pull(self, uuid: int, arrays: Any) -> Any:
    """Schedules a pytree of arrays to be fetched by a remote device."""
    self._await_pull_flat(uuid, jax.tree.flatten(arrays)[0])


if not TYPE_CHECKING and xla_extension_version >= 305:
  TransferServer = use_cpp_class(_xc._xla.TransferServer)(TransferServer)

if xla_extension_version >= 305:
  start_transfer_server = _xc._xla.start_transfer_server
