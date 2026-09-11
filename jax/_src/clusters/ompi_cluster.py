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

from __future__ import annotations

import os
import re
from jax._src import clusters

# OMPI_MCA_orte_hnp_uri exists only when processes are launched via mpirun or mpiexec
# in ORTE-based Open MPI (<5). PRRTE-based Open MPI 5 programs are routed to
# Mpi4pyCluster detection
_ORTE_URI = 'OMPI_MCA_orte_hnp_uri'
_OMPI_VERSION = 'OMPI_VERSION'
_PROCESS_COUNT = 'OMPI_COMM_WORLD_SIZE'
_PROCESS_ID = 'OMPI_COMM_WORLD_RANK'
_LOCAL_PROCESS_ID = 'OMPI_COMM_WORLD_LOCAL_RANK'

class OmpiCluster(clusters.ClusterEnv):

  name: str = "ompi"

  @classmethod
  def _is_ompi_5(cls) -> bool:
    # Use OMPI_VERSION when available; otherwise distinguish ORTE from PRRTE
    # by the presence of the legacy ORTE URI.
    if not all(
        variable in os.environ
        for variable in (_PROCESS_COUNT, _PROCESS_ID, _LOCAL_PROCESS_ID)
    ):
      return False

    try:
      major_version = int(
          os.environ.get(_OMPI_VERSION, '').split('.', maxsplit=1)[0]
      )
    except ValueError:
      # OMPI_VERSION is not guaranteed. The legacy launcher always provides
      # its ORTE URI, whereas the PRRTE launcher does not.
      return _ORTE_URI not in os.environ
    return major_version >= 5

  @classmethod
  def _mpi4py_cluster(cls) -> type[clusters.ClusterEnv]:
    from jax._src.clusters.mpi4py_cluster import Mpi4pyCluster
    if not Mpi4pyCluster.is_env_present():
      raise RuntimeError(
          "Open MPI 5 or newer requires mpi4py for distributed "
          "initialization."
      )
    return Mpi4pyCluster

  @classmethod
  def is_env_present(cls) -> bool:
    return _ORTE_URI in os.environ or cls._is_ompi_5()

  @classmethod
  def get_coordinator_address(cls, timeout_secs: int | None, override_coordinator_port: str | None) -> str:
    if cls._is_ompi_5():
      return cls._mpi4py_cluster().get_coordinator_address(
          timeout_secs=timeout_secs,
          override_coordinator_port=override_coordinator_port,
      )

    # Examples of orte_uri:
    # 1531576320.0;tcp://10.96.0.1,10.148.0.1,10.108.0.1:34911
    # 1314521088.0;tcp6://[fe80::b9b:ac5d:9cf0:b858,2620:10d:c083:150e::3000:2]:43370
    orte_uri = os.environ[_ORTE_URI]
    if override_coordinator_port:
        port = override_coordinator_port
    else:
        job_id_str = orte_uri.split('.', maxsplit=1)[0]
        # The jobid is always a multiple of 2^12, let's divide it by 2^12
        # to reduce likelihood of port conflict between jobs
        job_id = int(job_id_str) // 2**12
        # Pick port in ephemeral range [(65535 - 2^12 + 1), 65535]
        port = str(job_id % 2**12 + (65535 - 2**12 + 1))
    launcher_ip_match = re.search(r"tcp://(.+?)[,:]|tcp6://\[(.+?)[,\]]", orte_uri)
    if launcher_ip_match is None:
        raise RuntimeError('Could not parse coordinator IP address from Open MPI environment.')
    launcher_ip = next(i for i in launcher_ip_match.groups() if i is not None)
    return f'{launcher_ip}:{port}'

  @classmethod
  def get_process_count(cls) -> int:
    if cls._is_ompi_5():
      return cls._mpi4py_cluster().get_process_count()
    return int(os.environ[_PROCESS_COUNT])

  @classmethod
  def get_process_id(cls) -> int:
    if cls._is_ompi_5():
      return cls._mpi4py_cluster().get_process_id()
    return int(os.environ[_PROCESS_ID])

  @classmethod
  def get_local_process_id(cls) -> int | None:
    if cls._is_ompi_5():
      return cls._mpi4py_cluster().get_local_process_id()
    return int(os.environ[_LOCAL_PROCESS_ID])
