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

"""Tests for host_callback on multi-host setup."""

import unittest

import jax
from jax import lax
from jax import numpy as jnp
from jax._src import pjit
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax.experimental import io_callback
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
import numpy as np


class _CollectCallbacks:
  """Collect the callback arguments."""

  def __init__(self, test_method_name):
    self.collected = []
    self.test_method_name = test_method_name

  def collect(self, what) -> None:
    print(f"collect[{self.test_method_name}]: {what}")
    self.collected.append(what)


callback_collector = None

NR_PROCESSES = 4
NR_LOCAL_DEVICES = 2
NR_DEVICES = NR_PROCESSES * NR_LOCAL_DEVICES


def sorted_devices():
  devices = sorted(
      jax.devices(),
      key=lambda d: (d.process_index, getattr(d, "core_on_chip", 0)),
  )
  if len(devices) != NR_DEVICES:
    raise unittest.SkipTest("Test assumes that it runs on 8 devices.")
  if jax.process_count() != NR_PROCESSES:
    raise unittest.SkipTest(f"Test assumes we have {NR_PROCESSES} processes.")
  return devices


class IoCallbackMultiProcessTest(jtu.JaxTestCase,
                                 jt_multiprocess.MultiProcessTest):

  def setUp(self):
    super(jtu.JaxTestCase, self).setUp()
    global callback_collector
    callback_collector = _CollectCallbacks(self._testMethodName)

  def tearDown(self):
    super(jtu.JaxTestCase, self).tearDown()
    jax.effects_barrier()

  def test_pure_callback_pmap(self):
    # x_global: i32[D, 2] = [[0, 1], [10, 11], [20, 21], ...]
    # x_local: i32[L, 2]
    x_global = np.arange(100, dtype=np.int32).reshape((10, 10))[:NR_DEVICES, :2]

    process_idx = jax.process_index()
    local_device_idx = process_idx * NR_LOCAL_DEVICES
    x_local = x_global[local_device_idx:local_device_idx + NR_LOCAL_DEVICES]

    def func(x):  # Runs on each device.
      sum_global = jax.lax.psum(x, "d")
      return jax.pure_callback(callback_func,
                               x,  # result_shapes_dtype
                               lax.axis_index("d"), x, sum_global)

    def callback_func(axis_index, x, sum_global):
      callback_collector.collect((axis_index, x, sum_global))
      return x * np.array(3, np.int32) + sum_global

    pmap_func = jax.pmap(func, axis_name="d", devices=sorted_devices())
    res = pmap_func(x_local)
    expected_sum_global = np.sum(x_global, axis=0, dtype=np.int32)
    # On each host we only get the local result.
    self.assertAllClose(x_local * np.array(3, np.int32) + expected_sum_global,
                        res)

    jax.effects_barrier()

    # Each process gets only the callbacks for its local devices.
    self.assertAllClose(
        sorted(callback_collector.collected, key=lambda x: x[0]),
        [(np.array(process_idx * NR_LOCAL_DEVICES, dtype=np.int32),
          np.array([10 * local_device_idx,
                    10 * local_device_idx + 1], dtype=np.int32),
          expected_sum_global),
         (np.array(process_idx * NR_LOCAL_DEVICES + 1, dtype=np.int32),
          np.array([10 * local_device_idx + 10,
                    10 * local_device_idx + 11], dtype=np.int32),
          expected_sum_global)])

  def test_io_callback_pjit(self):
    devices = np.array(sorted_devices()).reshape(
        (NR_PROCESSES, NR_LOCAL_DEVICES))
    mesh = jax.sharding.Mesh(devices, ["p", "l"])

    # x_global: i32[P, L, 3] = [[[0, 1, 2], [10, 11, 12]],
    #                           [[100, 101, 102], [110, 111, 112]],
    #                           ...]
    # x_local: i32[1, L, 3]
    # y: i32[3, 5]
    x_global = jnp.arange(
        1000, dtype=jnp.int32).reshape(
            (10, 10, 10))[:NR_PROCESSES, :NR_LOCAL_DEVICES, :3]
    process_id = jax.process_index()
    x_local = x_global[process_id:process_id + 1]

    def callback_times5_func(x):
      callback_collector.collect(x)
      return x * np.array(5, np.int32)

    def fun(x_local):
      return io_callback(callback_times5_func,
                         x_local,  # result shape dtypes
                         x_local)

    expected_res = x_local * np.array(5, np.int32)
    pjit_fun = pjit.pjit(fun,
                         in_shardings=P("p", "l"),
                         out_shardings=P("p", "l"))

    with mesh:
      gx = multihost_utils.host_local_array_to_global_array(
          x_local, mesh, P("p", "l"))
      global_res = pjit_fun(gx)
      res = multihost_utils.global_array_to_host_local_array(
          global_res, mesh, P("p", "l"))

    self.assertAllClose(expected_res, res)
    jax.effects_barrier()

    if jax.process_index() == 0:
      # All calls are on the process 0; the 100s digit specifies the device
      self.assertAllClose(callback_collector.collected,
                          [np.array([[[0, 1, 2],
                                      [10, 11, 12]],
                                     [[100, 101, 102],
                                      [110, 111, 112]],
                                     [[200, 201, 202],
                                      [210, 211, 212]],
                                     [[300, 301, 302],
                                      [310, 311, 312]]], dtype=np.int32)])
    else:
      self.assertAllClose(callback_collector.collected, [])


if __name__ == "__main__":
  jt_multiprocess.main()
