# Copyright 2020 The JAX Authors.
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
"""Backwards compatibility shim for the deprecated host_callback APIs.

.. warning::
  The host_callback APIs are deprecated as of March 20, 2024.
  The functionality is subsumed by the
  `new JAX external callbacks <https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html>`_
  See https://github.com/jax-ml/jax/issues/20385.

"""

from __future__ import annotations

from collections.abc import Callable
import logging
import warnings

import jax
from jax.experimental import io_callback


logger = logging.getLogger(__name__)


# We keep a shim for host_callback.call because it is still used in a few
# places in google.
def call(callback_func: Callable,
         arg,
         *,
         result_shape=None,
         call_with_device=False,
         device_index=0,
         callback_flavor=None):
  """Make a call to the host, and expect a result.

  .. warning::
    The host_callback APIs are deprecated as of March 20, 2024.
    The functionality is subsumed by the
    `new JAX external callbacks <https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html>`_
    See https://github.com/jax-ml/jax/issues/20385.
  """
  warnings.warn("""The host_callback APIs are deprecated as of March 20, 2024.
    The functionality is subsumed by the
    new JAX external callbacks (https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html).
    See https://github.com/jax-ml/jax/issues/20385
  """, DeprecationWarning, stacklevel=2)
  if callback_flavor is not None:
    raise NotImplementedError(
        "host_callback.call is only supported with the IO_CALLBACK flavor.")
  if call_with_device:
    raise NotImplementedError(
        "host_callback.call is only supported with the call_with_device=False.")
  callback_device = jax.local_devices()[device_index]
  sharding = jax.sharding.SingleDeviceSharding(callback_device)
  return io_callback(callback_func, result_shape, arg,
                     sharding=sharding,
                     ordered=True)

import typing
if typing.TYPE_CHECKING:
  def id_tap(tap_func,
            arg,
            *,
            result=None,
            tap_with_device=False,
            device_index=0,
            callback_flavor=None,
            **kwargs):
    raise NotImplementedError(
        "host_callback.id_tap is no longer supported. "
        "See https://github.com/jax-ml/jax/issues/20385"
    )

del typing
