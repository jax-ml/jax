# Copyright 2022 Google LLC
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

import os
import unittest

from absl.testing import absltest

import jax
import jax._src.lib
from jax._src import test_util as jtu
import jax.numpy as jnp

@unittest.skipIf(
    os.environ.get("SLURM_JOB_NUM_NODES", None) != "2",
    "Slurm environment with at least two nodes needed!")
class MultiNodeGpuTest(jtu.JaxTestCase):

  def test_gpu_multi_node_initialize_and_psum(self):

    # Hookup the ENV vars expected to be set already in the SLURM environment
    nodelist = os.environ.get("SLURM_STEP_NODELIST", None)
    if nodelist is not None:
      coordinator_address = nodelist.split('[')[0] + \
                            nodelist.split('[')[1].split(',')[0]
    num_tasks = os.environ.get("SLURM_NPROCS", None)
    taskid = os.environ.get("SLURM_PROCID", None)
    localid = os.environ.get("SLURM_LOCALID", None)

    # fixing port since it needs to be the same for all the processes
    port = "54321"

    print(f"coord addr:port : {coordinator_address}:{port}\nTotal tasks: "
          f"{num_tasks}\ntask id: {taskid}\nlocal id: {localid}")

    self.assertEqual(
        coordinator_address is None or num_tasks is None or taskid is None,
        False)

    jax.distributed.initialize(coordinator_address=f'{coordinator_address}:{port}',
                               num_processes=int(num_tasks),
                               process_id=int(taskid))

    print(f"Total devices: {jax.device_count()}, Total tasks: {int(num_tasks)}, "
          f"Devices per task: {jax.local_device_count()}")

    self.assertEqual(jax.device_count(),
                     int(num_tasks) * jax.local_device_count())

    x = jnp.ones(jax.local_device_count())
    y = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(x)
    self.assertEqual(y[0], jax.device_count())
    print(y)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
