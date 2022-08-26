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
import subprocess
import sys
import threading
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax.config import config
from jax._src import distributed
import jax.numpy as jnp
from jax._src.lib import xla_extension_version
from jax._src import test_util as jtu

try:
  import portpicker
except ImportError:
  portpicker = None

config.parse_flags_with_absl()


@unittest.skipIf(not portpicker, "Test requires portpicker")
class DistributedTest(jtu.JaxTestCase):

  # TODO(phawkins): Enable after https://github.com/google/jax/issues/11222
  # is fixed.
  @unittest.SkipTest
  def testInitializeAndShutdown(self):
    # Tests the public APIs. Since they use global state, we cannot use
    # concurrency to simulate multiple tasks.
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(coordinator_address=f"localhost:{port}",
                               num_processes=1,
                               process_id=0)
    jax.distributed.shutdown()


  @parameterized.parameters([1, 2, 4])
  def testConcurrentInitializeAndShutdown(self, n):
    port = portpicker.pick_unused_port()
    def task(i):
      # We can't call the public APIs directly because they use global state.
      state = distributed.State()
      state.initialize(coordinator_address=f"localhost:{port}",
                       num_processes=n,
                       process_id=i)
      state.shutdown()

    threads = [threading.Thread(target=task, args=(i,)) for i in range(n)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()


@unittest.skipIf(not portpicker, "Test requires portpicker")
class MultiProcessGpuTest(jtu.JaxTestCase):

  def test_gpu_distributed_initialize(self):
    if jax.devices()[0].platform != 'gpu':
      raise unittest.SkipTest('Tests only for GPU.')

    port = portpicker.pick_unused_port()
    num_gpus = 4
    num_gpus_per_task = 1
    num_tasks = num_gpus // num_gpus_per_task

    os.environ["JAX_PORT"] = str(port)
    os.environ["NUM_TASKS"] = str(num_tasks)

    subprocesses = []
    for task in range(num_tasks):
      env = os.environ.copy()
      env["TASK"] = str(task)
      env["CUDA_VISIBLE_DEVICES"] = ",".join(
          str((task * num_gpus_per_task) + i) for i in range(num_gpus_per_task))
      args = [
          sys.executable,
          "-c",
          ('import jax, os; '
           'jax.distributed.initialize('
               'f\'localhost:{os.environ["JAX_PORT"]}\', '
               'int(os.environ["NUM_TASKS"]), int(os.environ["TASK"])); '
           'print(f\'{jax.local_device_count()},{jax.device_count()}\', end="")'
          )
      ]
      subprocesses.append(subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True))

    for proc in subprocesses:
      out, _ = proc.communicate()
      self.assertEqual(proc.returncode, 0)
      self.assertEqual(out, f'{num_gpus_per_task},{num_gpus}')

  @unittest.skipIf(xla_extension_version < 88,
                   "Test requires jaxlib 0.3.17 or newer")
  def test_distributed_jax_cuda_visible_devices(self):
    """Test jax_cuda_visible_devices works in distributed settings."""
    if jax.devices()[0].platform != 'gpu':
      raise unittest.SkipTest('Tests only for GPU.')

    port = portpicker.pick_unused_port()
    num_gpus = 4
    num_gpus_per_task = 1
    num_tasks = num_gpus // num_gpus_per_task

    os.environ["JAX_PORT"] = str(port)
    os.environ["NUM_TASKS"] = str(num_tasks)

    subprocesses = []
    for task in range(num_tasks):
      env = os.environ.copy()
      env["TASK"] = str(task)
      visible_devices = ",".join(
          str((task * num_gpus_per_task) + i) for i in range(num_gpus_per_task))
      program = (
        'import jax, os; '
        f'jax.config.update("jax_cuda_visible_devices", "{visible_devices}"); '
        'jax.distributed.initialize('
        'f\'localhost:{os.environ["JAX_PORT"]}\', '
        'int(os.environ["NUM_TASKS"]), int(os.environ["TASK"])); '
        'print(f\'{jax.local_device_count()},{jax.device_count()}\', end="")'
      )
      args = [sys.executable, "-c", program]
      subprocesses.append(subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True))

    for proc in subprocesses:
      out, _ = proc.communicate()
      self.assertEqual(proc.returncode, 0)
      self.assertEqual(out, f'{num_gpus_per_task},{num_gpus}')

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
