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

from __future__ import annotations

import os
import re

from jax._src import clusters


_PRTE_LAUNCHED = "PRTE_LAUNCHED"
_PMIX_NAMESPACE = "PMIX_NAMESPACE"
_PMIX_SERVER_URIS = (
    "PMIX_SERVER_URI41",
    "PMIX_SERVER_URI4",
    "PMIX_SERVER_URI3",
    "PMIX_SERVER_URI21",
    "PMIX_SERVER_URI2",
)
_PROCESS_COUNT = "OMPI_COMM_WORLD_SIZE"
_PROCESS_ID = "OMPI_COMM_WORLD_RANK"
_LOCAL_PROCESS_ID = "OMPI_COMM_WORLD_LOCAL_RANK"


class PrrteCluster(clusters.ClusterEnv):

  name: str = "prrte"

  @classmethod
  def is_env_present(cls) -> bool:
    return os.environ.get(_PRTE_LAUNCHED) == "1"

  @classmethod
  def get_coordinator_address(
      cls,
      timeout_secs: int | None,
      override_coordinator_port: str | None,
  ) -> str:
    del timeout_secs
    pmix_uri = next(
        (os.environ[var] for var in _PMIX_SERVER_URIS if var in os.environ),
        None,
    )
    if pmix_uri is None:
      raise RuntimeError(
          "Could not find a PMIX_SERVER_URI in the PRRTE environment."
      )

    ipv4_match = re.search(r"tcp4://([^,:;]+):", pmix_uri)
    ipv6_match = re.search(r"tcp6://\[([^,\]]+)\]", pmix_uri)
    if ipv4_match is not None:
      coordinator_host = ipv4_match.group(1)
    elif ipv6_match is not None:
      coordinator_host = f"[{ipv6_match.group(1)}]"
    else:
      raise RuntimeError(
          "Could not parse coordinator IP address from PRRTE environment."
      )

    if override_coordinator_port:
      port = override_coordinator_port
    else:
      namespace = os.environ.get(_PMIX_NAMESPACE, "")
      job_id_match = re.search(r"-(\d+)@", namespace)
      if job_id_match is None:
        raise RuntimeError(
            "Could not parse job ID from PMIX_NAMESPACE in PRRTE environment."
        )
      job_id = int(job_id_match.group(1))
      port = str(job_id % 2**12 + (65535 - 2**12 + 1))
    return f"{coordinator_host}:{port}"

  @classmethod
  def get_process_count(cls) -> int:
    return int(os.environ[_PROCESS_COUNT])

  @classmethod
  def get_process_id(cls) -> int:
    return int(os.environ[_PROCESS_ID])

  @classmethod
  def get_local_process_id(cls) -> int | None:
    return int(os.environ[_LOCAL_PROCESS_ID])
