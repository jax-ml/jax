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

"""Multihost tests for jax.Array."""

import jax
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax.experimental import colocated_python
import numpy as np


class ColocatedPythonTestMultiHost(jt_multiprocess.MultiProcessTest):

  def setUp(self):
    super().setUp()
    jtu.request_cpu_devices(jax.local_device_count())

  def test_colocated_cpu_devices(self):
    if jax.device_count() % 2 == 0:
      mesh_shape = (2, jax.device_count() // 2)
    else:
      mesh_shape = (1, jax.device_count())
    mesh = jax.make_mesh(mesh_shape, ("x", "y"),
                         axis_types=(jax.sharding.AxisType.Explicit,) * 2)
    cpu_mesh1 = colocated_python.colocated_cpu_devices(mesh)

    cpu_devices = colocated_python.colocated_cpu_devices(mesh.devices.flat)
    cpu_mesh2 = jax.make_mesh(mesh_shape, ("x", "y"),
                              axis_types=(jax.sharding.AxisType.Explicit,) * 2,
                              devices=cpu_devices)
    self.assertEqual(cpu_mesh1, cpu_mesh2)

  def test_simple_function(self):
    @colocated_python.colocated_python
    def add_one(x):
      return jax.make_array_from_single_device_arrays(
          x.shape, x.sharding, [s.data + 1 for s in x.addressable_shards])

    mesh = jax.make_mesh((jax.device_count(),), ("x",),
                         axis_types=(jax.sharding.AxisType.Explicit,))
    cpu_mesh = colocated_python.colocated_cpu_devices(mesh)
    cpu_sharding = jax.NamedSharding(cpu_mesh, jax.P("x"))

    x = np.arange(cpu_mesh.size)
    x = jax.device_put(x, cpu_sharding)

    out = add_one(x)

    out = jax.jit(lambda x: x,
                  out_shardings=jax.NamedSharding(cpu_mesh, jax.P()))(out)
    out = jax.device_get(out)

    np.testing.assert_equal(out, np.arange(cpu_mesh.size) + 1)


if __name__ == "__main__":
  jt_multiprocess.main()
