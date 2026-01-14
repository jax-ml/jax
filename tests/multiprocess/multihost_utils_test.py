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

"""Multihost tests for pjit."""

import math
import unittest

from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
import numpy as np


class MultiHostUtilsTest(jt_multiprocess.MultiProcessTest):

  def test_process_allgather_stacked(self):
    elems_per_host = 4

    num_processes = jax.process_count()
    x = jnp.ones((4,)).reshape((2, 2))
    out = multihost_utils.process_allgather(x, tiled=False)
    self.assertEqual(out.shape, (num_processes, 2, 2))
    np.testing.assert_array_equal(out, np.stack([x] * num_processes))

    x = jnp.ones((64,)).reshape((8, 4, 2))
    out = multihost_utils.process_allgather(x, tiled=False)
    self.assertEqual(out.shape, (num_processes, 8, 4, 2))
    np.testing.assert_array_equal(out, np.stack([x] * num_processes))

    x = np.arange(elems_per_host) + jax.process_index() * elems_per_host
    out = multihost_utils.process_allgather(x, tiled=False)
    self.assertEqual(out.shape, (num_processes, 4))
    np.testing.assert_array_equal(
        out,
        np.arange(elems_per_host * jax.process_count()).reshape(
            num_processes, elems_per_host
        ),
    )

    x = np.array(0) + jax.process_index() * elems_per_host
    out = multihost_utils.process_allgather(x, tiled=False)
    self.assertEqual(out.shape, (num_processes,))
    np.testing.assert_array_equal(
        out, np.arange(num_processes) * elems_per_host
    )

  def test_process_allgather_concatenated(self):
    elems_per_host = 4

    num_processes = jax.process_count()
    x = jnp.ones((4,)).reshape((2, 2))
    out = multihost_utils.process_allgather(x, tiled=True)
    self.assertEqual(out.shape, (2 * num_processes, 2))
    np.testing.assert_array_equal(out, np.concatenate([x] * num_processes))

    x = jnp.ones((64,)).reshape((8, 4, 2))
    out = multihost_utils.process_allgather(x, tiled=True)
    self.assertEqual(out.shape, (8 * num_processes, 4, 2))
    np.testing.assert_array_equal(out, np.concatenate([x] * num_processes))

    x = np.arange(elems_per_host) + jax.process_index() * elems_per_host
    out = multihost_utils.process_allgather(x, tiled=True)
    self.assertEqual(out.shape, (elems_per_host * num_processes,))
    np.testing.assert_array_equal(
        out, np.arange(elems_per_host * jax.process_count())
    )

    x = np.array(0) + jax.process_index() * elems_per_host
    out = multihost_utils.process_allgather(x, tiled=True)
    self.assertEqual(out.shape, (num_processes,))
    np.testing.assert_array_equal(
        out, np.arange(num_processes) * elems_per_host
    )

  def test_process_allgather_set_mesh(self):
    devices = jax.devices()[1:] + [jax.devices()[0]]
    user_mesh = jax.sharding.Mesh(
        np.array(devices).reshape(jax.device_count(), 1, 1),
        ('x', 'y', 'z'),
    )
    x = jnp.ones((4,)).reshape((2, 2))
    # process_allgather should not be impacted by any global mesh context.
    with jax.set_mesh(user_mesh):
      num_processes = jax.process_count()
      out = multihost_utils.process_allgather(x, tiled=True)
      self.assertEqual(out.shape, (2 * num_processes, 2))
      np.testing.assert_array_equal(out, np.concatenate([x] * num_processes))

  def test_broadcast_one_to_all(self):
    elems_per_host = 4

    x = np.arange(elems_per_host) + jax.process_index() * elems_per_host
    out = multihost_utils.broadcast_one_to_all((x, x))
    jax.tree.map(
        lambda x: np.testing.assert_array_equal(  # pylint: disable=g-long-lambda
            x, np.arange(elems_per_host)
        ),
        out,
    )

    x = np.array(0) + jax.process_index() * elems_per_host
    out = multihost_utils.broadcast_one_to_all(x)
    np.testing.assert_array_equal(out, np.array(0))

  def test_broadcast_one_to_all_set_mesh(self):
    devices = jax.devices()[1:] + [jax.devices()[0]]
    user_mesh = jax.sharding.Mesh(
        np.array(devices).reshape(jax.device_count(), 1, 1),
        ('x', 'y', 'z'),
    )
    # broadcast_one_to_all should not be impacted by any global mesh context.
    with jax.set_mesh(user_mesh):
      elems_per_host = 4

      x = np.arange(elems_per_host) + jax.process_index() * elems_per_host
      out = multihost_utils.broadcast_one_to_all((x, x))
      jax.tree.map(
          lambda x: np.testing.assert_array_equal(  # pylint: disable=g-long-lambda
              x, np.arange(elems_per_host)
          ),
          out,
      )

      x = np.array(0) + jax.process_index() * elems_per_host
      out = multihost_utils.broadcast_one_to_all(x)
      np.testing.assert_array_equal(out, np.array(0))

  def test_broadcast_one_to_all_uint8(self):
    elems_per_host = 4

    x = (np.arange(elems_per_host, dtype=jnp.uint8) +
         jax.process_index() * elems_per_host)
    out = multihost_utils.broadcast_one_to_all((x, x))
    jax.tree.map(
        lambda x: np.testing.assert_array_equal(  # pylint: disable=g-long-lambda
            x, np.arange(elems_per_host, dtype=jnp.uint8)
        ),
        out,
    )
    jax.tree.map(lambda o: self.assertEqual(o.dtype, jnp.uint8), out)

    x = np.array(0, dtype=jnp.uint8) + jax.process_index() * elems_per_host
    out = multihost_utils.broadcast_one_to_all(x)
    self.assertEqual(out.dtype, jnp.uint8)
    np.testing.assert_array_equal(out, np.array(0, dtype=jnp.uint8))

  def test_sync_global_devices(self):
    multihost_utils.sync_global_devices('test sync global devices')

  def test_sync_global_devices_error(self):
    # All processes should raise.
    with self.assertRaises(AssertionError):
      if jax.process_index() == 0:
        multihost_utils.sync_global_devices('test message')
      else:
        multihost_utils.sync_global_devices('test message2')

  def test_sync_global_devices_mesh_context_manager(self):
    global_mesh = jtu.create_mesh((2, 2), ('x', 'y'), iota_order=True)
    with global_mesh:
      multihost_utils.sync_global_devices('test sync global devices')

  def test_assert_equal_global(self):
    mesh = jtu.create_mesh((8,), 'x')
    shape = (8, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    inp = jax.make_array_from_callback(
        shape, jax.NamedSharding(mesh, P()), lambda idx: np_inp[idx])
    multihost_utils.assert_equal(inp)

  def test_process_allgather_cache_hit(self):
    x = jnp.ones((4,)).reshape(2, 2)
    y = jnp.arange(4.0).reshape(2, 2)

    num_processes = jax.process_count()
    with jtu.count_pjit_cpp_cache_miss() as count:
      out = multihost_utils.process_allgather(x, tiled=False)
      out2 = multihost_utils.process_allgather(y, tiled=False)

    # Cpp cache hit.
    self.assertEqual(count(), 1)

    self.assertEqual(out.shape, (num_processes, 2, 2))
    np.testing.assert_array_equal(out, np.stack([x] * num_processes))
    self.assertEqual(out2.shape, (num_processes, 2, 2))
    np.testing.assert_array_equal(out2, np.stack([y] * num_processes))

  def test_reshard(self):
    mesh1 = jtu.create_mesh((8,), 'x')
    mesh2 = jax.sharding.Mesh(
        np.asarray(jax.devices()[::-1]).reshape(4, 2), ('x', 'y')
    )

    shape = (8, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    inp = jax.make_array_from_callback(
        shape,
        jax.sharding.NamedSharding(mesh1, P('x')),
        lambda idx: np_inp[idx],
    )

    out = jax.device_put(inp, jax.sharding.NamedSharding(mesh2, P('x', 'y')))
    self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
    for s in out.addressable_shards:
      np.testing.assert_array_equal(s.data, np_inp[s.index])

  @parameterized.named_parameters(
      ('inp_replicated', P(), P('x', 'y')),
      ('target_replicated', P('x'), P()),
      ('both_replicated', P(), P()),
  )
  def test_reshard_replicated_sharding(self, inp_spec, target_spec):
    mesh1 = jtu.create_mesh((8,), 'x')
    mesh2 = jax.sharding.Mesh(
        np.asarray(jax.devices()[::-1]).reshape(4, 2), ('x', 'y')
    )

    shape = (8, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    inp = jax.make_array_from_callback(
        shape,
        jax.sharding.NamedSharding(mesh1, inp_spec),
        lambda idx: np_inp[idx],
    )

    out = jax.device_put(inp, jax.sharding.NamedSharding(mesh2, target_spec))
    self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
    for s in out.addressable_shards:
      np.testing.assert_array_equal(s.data, np_inp[s.index])

  def test_reshard_same_device_assignment(self):
    mesh1 = jtu.create_mesh((4, 2), ('x', 'y'))
    mesh2 = jtu.create_mesh((2, 4), ('x', 'y'))

    shape = (8, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    inp = jax.make_array_from_callback(
        shape,
        jax.sharding.NamedSharding(mesh1, P('x', 'y')),
        lambda idx: np_inp[idx],
    )

    out = jax.device_put(inp, jax.sharding.NamedSharding(mesh2, P('y')))
    self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
    for s in out.addressable_shards:
      np.testing.assert_array_equal(s.data, np_inp[s.index])

  def test_reshard_pytree(self):
    mesh1 = jtu.create_mesh((8,), 'x')

    dev = jax.devices()
    if len(dev) < 8:
      raise unittest.SkipTest('Test requires 8 devices')
    dev_list = [dev[0], dev[7], dev[6], dev[2], dev[4], dev[3], dev[5], dev[1]]
    mesh2 = jax.sharding.Mesh(
        np.asarray(dev_list).reshape(2, 2, 2), ('x', 'y', 'z')
    )

    shape = (8, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    inp = jax.make_array_from_callback(
        shape,
        jax.sharding.NamedSharding(mesh1, P('x')),
        lambda idx: np_inp[idx],
    )

    out1, out2 = jax.device_put(
        (inp, inp), jax.sharding.NamedSharding(mesh2, P('x', 'y'))
    )

    for out in (out1, out2):
      self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
      for s in out.addressable_shards:
        np.testing.assert_array_equal(s.data, np_inp[s.index])

  def test_reshard_different_devices(self):
    if jtu.is_device_tpu('5', 'e'):
      raise unittest.SkipTest('Test fails on v5e')
    dev = jax.devices()
    if len(dev) < 8:
      raise unittest.SkipTest('Test requires 8 devices')
    mesh1 = jax.sharding.Mesh([dev[0], dev[2], dev[4], dev[6]], 'x')
    mesh2 = jax.sharding.Mesh(jax.devices(), 'x')

    shape = (8, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    inp = jax.make_array_from_callback(
        shape,
        jax.sharding.NamedSharding(mesh1, P('x')),
        lambda idx: np_inp[idx],
    )

    with self.assertRaisesRegex(
        ValueError,
        'input and target sharding should have the same set of devices',
    ):
      jax.device_put(inp, jax.sharding.NamedSharding(mesh2, P('x')))

  def test_process_allgather_array_not_fully_addressable(self):
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    global_input_data = np.arange(math.prod(global_input_shape)).reshape(
        global_input_shape
    )

    arr = jax.make_array_from_callback(
        global_input_shape,
        jax.sharding.NamedSharding(global_mesh, P('x', 'y')),
        lambda idx: global_input_data[idx],
    )

    out = multihost_utils.process_allgather(arr, tiled=True)
    np.testing.assert_array_equal(out, global_input_data)

    with self.assertRaisesRegex(
        ValueError,
        'Gathering global non-fully-addressable arrays only supports'
        ' tiled=True'):
      multihost_utils.process_allgather(arr, tiled=False)

  @jtu.ignore_warning(
      category=DeprecationWarning,
      message='jax.sharding.PmapSharding is deprecated',
  )
  def test_host_local_array_to_global_array_already_global(self):
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    global_input_data = np.arange(math.prod(global_input_shape)).reshape(
        global_input_shape
    )

    arr = jax.make_array_from_callback(
        global_input_shape,
        jax.sharding.NamedSharding(global_mesh, P('x', 'y')),
        lambda idx: global_input_data[idx],
    )

    out = multihost_utils.host_local_array_to_global_array(
        arr, global_mesh, P('x', 'y')
    )

    self.assertEqual(id(arr), id(out))

  def test_host_local_array_to_global_array_same_sharding_array(self):
    if jtu.is_device_tpu('5', 'e'):
      raise unittest.SkipTest('Test fails on v5e')
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'), iota_order=True)
    local_input_shape = (2, 2)

    elems_per_host = 4
    local_input_data = (
        jnp.arange(elems_per_host) + jax.process_index() * elems_per_host
    ).reshape(local_input_shape)

    arr = jax.make_array_from_callback(
        local_input_shape,
        jax.sharding.NamedSharding(global_mesh.local_mesh, P('x', 'y')),
        lambda idx: local_input_data[idx],
    )

    out = multihost_utils.host_local_array_to_global_array(
        arr, global_mesh, P('x', 'y')
    )

    expected_global_shape = (8, 2)
    self.assertEqual(out.shape, expected_global_shape)

    global_data = np.arange(math.prod(expected_global_shape)).reshape(
        expected_global_shape
    )
    for a, o in zip(arr.addressable_shards, out.addressable_shards):
      self.assertEqual(
          a.data.unsafe_buffer_pointer(), o.data.unsafe_buffer_pointer()
      )
      np.testing.assert_array_equal(o.data, global_data[o.index])

  def test_host_local_to_global_reshard_committed_single_device_array(self):
    if jtu.is_device_tpu('5', 'e'):
      raise unittest.SkipTest('Test fails on v5e')
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'), iota_order=True)
    local_input_shape = (2, 2)

    elems_per_host = 4
    local_input_data = (
        jnp.arange(elems_per_host) + jax.process_index() * elems_per_host
    ).reshape(local_input_shape)

    arr = jax.make_array_from_callback(
        local_input_shape,
        jax.sharding.NamedSharding(global_mesh.local_mesh, P('x', 'y')),
        lambda idx: local_input_data[idx],
    )

    out = multihost_utils.host_local_array_to_global_array(
        arr, global_mesh, P('x', 'y')
    )

    expected_global_shape = (8, 2)
    self.assertEqual(out.shape, expected_global_shape)

    global_data = np.arange(math.prod(expected_global_shape)).reshape(
        expected_global_shape
    )
    for a, o in zip(arr.addressable_shards, out.addressable_shards):
      self.assertEqual(
          a.data.unsafe_buffer_pointer(), o.data.unsafe_buffer_pointer()
      )
      np.testing.assert_array_equal(o.data, global_data[o.index])

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_host_local_to_global_replicated(self):
    num_local_devices = jax.local_device_count()
    global_mesh = jax.sharding.Mesh(jax.devices(), axis_names=['x'])
    local_input_shape = (2, 2)
    local_input_data = jnp.arange(4).reshape(local_input_shape)

    out = multihost_utils.host_local_array_to_global_array(
        local_input_data, global_mesh, P()
    )

    expected_global_shape = (2, 2)
    self.assertEqual(out.shape, expected_global_shape)
    self.assertLen(out.addressable_shards, num_local_devices)
    # Array is accessible on every host.
    np.testing.assert_array_equal(out, local_input_data)

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_host_local_to_global_locally_replicated(self):
    # Make an array which is locally replicated but sharded across hosts.
    num_processes = jax.process_count()
    num_local_devices = jax.local_device_count()
    global_mesh = jtu.create_mesh(
        (num_processes, num_local_devices), ('host', 'dev'), iota_order=True)
    local_input_shape = (2, 2)
    host_id = jax.process_index()
    local_input_data = jnp.arange(4).reshape(local_input_shape) * host_id

    out = multihost_utils.host_local_array_to_global_array(
        local_input_data, global_mesh, P('host', None))
    global_data = np.concatenate([jnp.arange(4).reshape(local_input_shape) * i
                                  for i in range(num_processes)])
    expected_global_shape = global_data.shape
    self.assertEqual(out.shape, expected_global_shape)
    self.assertLen(out.addressable_shards, num_local_devices)
    for o in out.addressable_shards:
      # Each shard has the same shape matching local_input_shape and smae
      # global index.
      self.assertEqual(o.data.shape, local_input_shape)
      self.assertEqual(o.index, out.addressable_shards[0].index)
      np.testing.assert_array_equal(o.data, global_data[o.index])

  def test_global_array_to_host_local_array(self):
    if jtu.is_device_tpu('5', 'e'):
      raise unittest.SkipTest('Test fails on v5e')
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'), iota_order=True)
    global_shape = (8, 2)
    global_data = np.arange(math.prod(global_shape)).reshape(global_shape)

    arr = jax.make_array_from_callback(
        global_shape,
        jax.sharding.NamedSharding(global_mesh, P('x', 'y')),
        lambda idx: global_data[idx],
    )

    out = multihost_utils.global_array_to_host_local_array(
        arr, global_mesh, P('x')
    )

    self.assertEqual(out.shape, (2, 2))
    self.assertEqual(
        out.sharding, jax.sharding.NamedSharding(global_mesh.local_mesh, P('x'))
    )

    local_input_data = (np.arange(4) + jax.process_index() * 4).reshape(
        out.shape
    )
    for s in out.addressable_shards:
      np.testing.assert_array_equal(s.data, local_input_data)

  def test_host_local_array_to_global_array_none_error(self):
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    global_shape = (8, 2)
    data = np.arange(math.prod(global_shape)).reshape(global_shape)

    with self.assertRaisesRegex(
        ValueError, '`None` is not a valid input to the pspecs argument'
    ):
      multihost_utils.host_local_array_to_global_array(data, global_mesh, None)

    with self.assertRaisesRegex(
        ValueError, '`None` is not a valid input to the pspecs argument'
    ):
      multihost_utils.global_array_to_host_local_array(data, global_mesh, None)

  def test_live_devices(self):
    with multihost_utils.live_devices(jax.devices()) as live:
      self.assertEqual(set(live), set(jax.devices()))


if __name__ == '__main__':
  jt_multiprocess.main()
