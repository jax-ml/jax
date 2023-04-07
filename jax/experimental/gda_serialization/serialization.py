# Copyright 2021 The JAX Authors.
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
"""GlobalDeviceArray serialization and deserialization."""

from jax.experimental.array_serialization.serialization import (
    get_tensorstore_spec as _deprecated_get_tensorstore_spec,
    async_serialize as _deprecated_async_serialize,
    run_serialization as _deprecated_run_serialization,
    async_deserialize as _deprecated_async_deserialize,
    run_deserialization as _deprecated_run_deserialization,
    GlobalAsyncCheckpointManagerBase as _deprecated_GlobalAsyncCheckpointManagerBase,
    GlobalAsyncCheckpointManager as _deprecated_GlobalAsyncCheckpointManager,
    AsyncManager as _deprecated_AsyncManager,
    _LimitInFlightBytes as _deprecated_LimitInFlightBytes,
    _get_metadata as _deprecated_get_metadata,
    TS_CONTEXT as _deprecated_TS_CONTEXT,
)


_deprecations = {
    "get_tensorstore_spec": (
        (
            "jax.experimental.gda_serialization.get_tensorstore_spec is"
            " deprecated. Use"
            " jax.experimental.array_serialization.get_tensorstore_spec"
        ),
        _deprecated_get_tensorstore_spec,
    ),
    "async_serialize": (
        (
            "jax.experimental.gda_serialization.async_serialize is deprecated. "
            "Use jax.experimental.array_serialization.async_serialize"
        ),
        _deprecated_async_serialize,
    ),
    "run_serialization": (
        (
            "jax.experimental.gda_serialization.run_serialization is deprecated."
            " Use jax.experimental.array_serialization.run_serialization"
        ),
        _deprecated_run_serialization,
    ),
    "async_deserialize": (
        (
            "jax.experimental.gda_serialization.async_deserialize is deprecated."
            " Use jax.experimental.array_serialization.async_deserialize"
        ),
        _deprecated_async_deserialize,
    ),
    "run_deserialization": (
        (
            "jax.experimental.gda_serialization.run_deserialization is"
            " deprecated. Use"
            " jax.experimental.array_serialization.run_deserialization"
        ),
        _deprecated_run_deserialization,
    ),
    "GlobalAsyncCheckpointManagerBase": (
        (
            "jax.experimental.gda_serialization.GlobalAsyncCheckpointManagerBase"
            " is deprecated. "
            "Use jax.experimental.array_serialization.GlobalAsyncCheckpointManagerBase"
        ),
        _deprecated_GlobalAsyncCheckpointManagerBase,
    ),
    "GlobalAsyncCheckpointManager": (
        (
            "jax.experimental.gda_serialization.GlobalAsyncCheckpointManager is"
            " deprecated. Use"
            " jax.experimental.array_serialization.GlobalAsyncCheckpointManager"
        ),
        _deprecated_GlobalAsyncCheckpointManager,
    ),
    "AsyncManager": (
        (
            "jax.experimental.gda_serialization.AsyncManager is deprecated. "
            "Use jax.experimental.array_serialization.AsyncManager"
        ),
        _deprecated_AsyncManager,
    ),
    "_LimitInFlightBytes": (
        (
            "jax.experimental.gda_serialization._LimitInFlightBytes is deprecated. "
            "Use jax.experimental.array_serialization._LimitInFlightBytes"
        ),
        _deprecated_LimitInFlightBytes,
    ),
    "_get_metadata": (
        (
            "jax.experimental.gda_serialization._get_metadata is deprecated. "
            "Use jax.experimental.array_serialization._get_metadata"
        ),
        _deprecated_get_metadata,
    ),
    "TS_CONTEXT": (
        (
            "jax.experimental.gda_serialization.TS_CONTEXT is deprecated. "
            "Use jax.experimental.array_serialization.TS_CONTEXT"
        ),
        _deprecated_TS_CONTEXT,
    ),
}

import typing
if typing.TYPE_CHECKING:
  from jax.experimental.array_serialization.serialization import (
      get_tensorstore_spec as get_tensorstore_spec,
      async_serialize as async_serialize,
      run_serialization as run_serialization,
      async_deserialize as async_deserialize,
      run_deserialization as run_deserialization,
      GlobalAsyncCheckpointManagerBase as GlobalAsyncCheckpointManagerBase,
      GlobalAsyncCheckpointManager as GlobalAsyncCheckpointManager,
      AsyncManager as AsyncManager,
      _LimitInFlightBytes as _LimitInFlightBytes,
      _get_metadata as _get_metadata,
      TS_CONTEXT as TS_CONTEXT,
  )
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
