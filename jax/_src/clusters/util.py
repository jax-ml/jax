# Copyright 2022 The JAX Authors.
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

import logging
import socket
import time


logger = logging.getLogger(__name__)


def wait_for_host(
  host_address, timeout_secs, retry_secs=5, retry_exp=1, retry_max=None
):
  # wait for a host to come online
  host_found = False
  max_time = time.time() + timeout_secs
  while not host_found and time.time() < max_time:
    try:
      socket.gethostbyname(host_address)
      host_found = True
      logger.debug("Found host with address %s", host_address)
    except socket.gaierror:
      logger.debug(
          "Failed to recognize host address %s"
          " retrying...", host_address
      )
      time.sleep(retry_secs)
      retry_secs = min(retry_secs * retry_exp, retry_max or timeout_secs)
  if not host_found:
    raise RuntimeError(f"Failed to recognize host address {host_address}")
