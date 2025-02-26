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
"""A basic educational example."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax.experimental import _private_mm as mm
from jax.experimental._private_mm.examples import launch_utils


def step():
    devices = jax.devices()
    mesh1 = Mesh(devices[:4], ('data',))
    mesh2 = Mesh(devices[4:], ('data',))

    sharding1 = NamedSharding(mesh1, P('data'))
    sharding2 = NamedSharding(mesh2, P('data'))

    shape = (512, 2**20)

    @partial(mm.jit, in_shardings=sharding1, out_shardings=sharding1)
    def stage1(x):
        return x + 1

    @partial(mm.jit, in_shardings=sharding2, out_shardings=sharding2)
    def stage2(x):
        return x * 2

    a0: mm.MpmdArray = mm.device_put(jnp.zeros(shape), sharding1)
    b0: mm.MpmdArray = mm.device_put(jnp.ones(shape), sharding1)

    # Enqueue all work on [a]
    a1 = stage1(a0)
    a1 = mm.device_put(a1, sharding2)
    a2 = stage2(a1)

    # Enqueue all work on [b]
    b1 = stage1(b0)
    b1 = mm.device_put(b1, sharding2)
    b2 = stage2(b1)

    # Only print if a2/b2 resident (i.e., we belong to the last stage):
    if not a2.is_fully_remote:
        assert not b2.is_fully_remote
        print(a2.jax_array)
        print(b2.jax_array)


def example_basic(num_processes, process_id):
    assert jax.device_count() == 8
    # FIXME: Support stages spread across multiple processes.
    assert 2 % num_processes == 0

    for i in range(3):
        step()


if __name__ == '__main__':
    import sys
    num_processes = 2
    if len(sys.argv) >= 2:
        num_processes = int(sys.argv[1])
    success = launch_utils.launch_example(num_processes, example_basic)
    sys.exit(0 if success else 1)
