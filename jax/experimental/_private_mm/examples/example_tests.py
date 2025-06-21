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
"""Runs some simple mm operations on varying numbers of processes."""

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax.experimental import _private_mm as mm
from jax.experimental._private_mm.examples import launch_utils


def make_two_meshes():
    devices = jax.devices()
    num_devices = len(devices)
    assert num_devices % 2 == 0
    mesh1 = Mesh(devices[:num_devices//2], ('data',))
    mesh2 = Mesh(devices[num_devices//2:], ('data',))
    sharding1 = NamedSharding(mesh1, P('data'))
    sharding2 = NamedSharding(mesh2, P('data'))
    return mesh1, mesh2, sharding1, sharding2


def test_device_put_uncommitted(_num_processes, process_id):
    _, _, sharding1, sharding2 = make_two_meshes()
    x = mm.device_put(jnp.ones((16,16)), sharding1)
    x.block_until_ready()


def test_device_put_across_meshes(_num_processes, process_id):
    _, _, sharding1, sharding2 = make_two_meshes()
    x = mm.device_put(jnp.ones((16,16)), sharding1)
    y = mm.device_put(x, sharding2)
    if y.is_fully_remote:
        y.block_until_ready()
    else:
        np.testing.assert_array_equal(y.jax_array, jnp.ones((16,16)))


def test_jit_and_transfer(_num_processes, process_id):
    _, _, sharding1, sharding2 = make_two_meshes()
    x1 = mm.device_put(jnp.ones((16,16)), sharding1)
    x2 = mm.jit(lambda x: x + 1, out_shardings=sharding1)(x1)
    y1 = mm.device_put(x2, sharding2)
    y2 = mm.jit(lambda x: x * 2, out_shardings=sharding2)(y1)
    if y2.is_fully_remote:
        y2.block_until_ready()
    else:
        np.testing.assert_array_equal(y2.jax_array, jnp.full((16,16), 4))


def run_test(num_processes, test_fun, name):
    print(f' - {name} ... ', end='', flush=True)
    success = launch_utils.launch_example(num_processes, test_fun)
    if success:
        print('OK')
    else:
        print('FAIL')
    return success


def run_tests():
    # For 1 process mm.device_puts simply reduce to jax.device_puts.
    # For 2 processes and tests involving two meshes we require NCCL comms,
    # but all devices of a mesh are managed by the same process.
    # For 4 processes and tests involving two meshes we additionally have to
    # deal with devices of a mesh being managed by multipel processes.
    # (The latter currently doesn't work.)
    NUM_PROCESSESS = (1, 2, 4)
    TESTS = [
        ('device_put_uncommitted', test_device_put_uncommitted),
        ('device_put_across_meshes', test_device_put_across_meshes),
        ('jit_and_transfer', test_jit_and_transfer),
    ]
    num_failures = 0
    for num_processes in NUM_PROCESSESS:
        print(f'=== {num_processes=} ===')
        for test_name, test_fun in TESTS:
            success = run_test(num_processes, test_fun, test_name)
            if not success:
                num_failures += 1
    if num_failures == 0:
        print('All tests succeeded!')
        return True
    else:
        print(f'{num_failures} tests failed!')
        return False


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
