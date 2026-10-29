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

"""Traces the Pathways backend of JAX's `proxy` backend in profiler sessions."""

from __future__ import annotations

import json

from jax._src.lib import _pathways
from jax._src.lib import _profiler
from jax._src.lib import ifrt_version
from jax._src.lib import jaxlib_extension_version
from jax._src.lib import xla_client

# Keys of `ProfileOptions.advanced_configuration` that configure how profiler
# sessions trace Pathways backends. They are not forwarded to Pathways hosts.
# The maximum number of Pathways hosts to trace (int, defaults to 1).
_PATHWAYS_MAX_NUM_HOSTS = "pathways_max_num_hosts"
# A JSON-encoded Pathways profile request (str), i.e. `{"profileRequest": ...}`,
# to send instead of the request built from the options. Values of other types
# fail the trace.
_PATHWAYS_PROFILE_REQUEST = "pathways_profile_request"
_ADVANCED_CONFIG_VALUE_KEYS = {
    bool: "boolValue",
    int: "int64Value",
    str: "stringValue",
}


def _profile_request(options: _profiler.ProfileOptions) -> str | None:
  """Returns the request that traces the Pathways backend in a session.

  The request is a JSON-encoded Pathways profile request that starts the trace
  when it is executed as a PluginProgram and stops it when it is executed again.
  Each traced Pathways host writes its trace to
  `<log_dir>/plugins/profile/<session_id>`, so this returns None if the session
  with `options` has no log directory.
  """
  if not options.repository_path:
    return None
  config = dict(options.advanced_configuration)
  if _PATHWAYS_PROFILE_REQUEST in config:
    return config[_PATHWAYS_PROFILE_REQUEST]
  max_num_hosts = config.pop(_PATHWAYS_MAX_NUM_HOSTS, 1)
  return json.dumps({
      "profileRequest": {
          "traceLocation": options.repository_path,
          "maxNumHosts": max_num_hosts,
          "includeResourceManagers": True,
          "maxDurationSecs": options.duration_ms / 1000,
          "xprofTraceOptions": {
              "traceDirectory": options.repository_path,
              "traceSessionName": options.session_id,
              "hostTraceLevel": options.host_tracer_level,
              "pwTraceOptions": {
                  "enablePythonTracer": options.python_tracer_level > 0,
                  "advancedConfiguration": {
                      key: {_ADVANCED_CONFIG_VALUE_KEYS[type(value)]: value}
                      for key, value in config.items()
                  },
              },
          },
      }
  })


if jaxlib_extension_version >= 515 and ifrt_version >= 82:

  def register(client: xla_client.Client) -> None:
    """Makes profiler sessions also trace the Pathways backend of `client`.

    This includes captures through the server started by
    `jax.profiler.start_server`. The backend is traced while `client` is alive.
    JAX calls this for its `proxy` backend. Objects that aren't clients, e.g.,
    fakes in tests, are ignored.
    """
    if not isinstance(client, xla_client.Client):
      return
    _pathways.register_profiler(client, _profile_request)
