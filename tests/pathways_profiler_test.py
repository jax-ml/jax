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

import json
import unittest

from absl.testing import absltest
from jax._src import pathways_profiler
from jax._src import test_util as jtu
import jax.profiler


class PathwaysProfilerTest(jtu.JaxTestCase):

  def testProfileRequest(self):
    options = jax.profiler.ProfileOptions()
    options.repository_path = "gs://bucket/logs"
    options.session_id = "my_session"
    options.host_tracer_level = 0
    options.python_tracer_level = 0
    options.duration_ms = 2500
    options.advanced_configuration = {
        "pathways_max_num_hosts": 4,
        "tpu_trace_mode": "TRACE_ONLY_HOST",
        "enable_fw_throttle_event": True,
        "num_buffers": 2,
    }

    request = json.loads(pathways_profiler._profile_request(options))

    self.assertEqual(
        request,
        {
            "profileRequest": {
                "traceLocation": "gs://bucket/logs",
                "maxNumHosts": 4,
                "includeResourceManagers": True,
                "maxDurationSecs": 2.5,
                "xprofTraceOptions": {
                    "traceDirectory": "gs://bucket/logs",
                    "traceSessionName": "my_session",
                    "hostTraceLevel": 0,
                    "pwTraceOptions": {
                        "enablePythonTracer": False,
                        "advancedConfiguration": {
                            "tpu_trace_mode": {
                                "stringValue": "TRACE_ONLY_HOST"
                            },
                            "enable_fw_throttle_event": {"boolValue": True},
                            "num_buffers": {"int64Value": 2},
                        },
                    },
                },
            }
        },
    )

  def testProfileRequestWithoutLogDirOrFromOptions(self):
    options = jax.profiler.ProfileOptions()
    self.assertIsNone(pathways_profiler._profile_request(options))

    options.repository_path = "gs://bucket/logs"
    options.advanced_configuration = {
        "pathways_profile_request": '{"profileRequest": {}}'
    }
    self.assertEqual(
        pathways_profiler._profile_request(options), '{"profileRequest": {}}'
    )

  @unittest.skipIf(
      not hasattr(pathways_profiler, "register"), "Requires newer jaxlib"
  )
  def testRegisterIgnoresNonClients(self):
    # Would raise a TypeError if the object was passed to jaxlib.
    pathways_profiler.register(object())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
