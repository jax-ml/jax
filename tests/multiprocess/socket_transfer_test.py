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

"""Tests for socket transfer."""

import jax
from jax._src import test_multiprocess as jt_multiprocess
from jax.sharding import PartitionSpec as P
import numpy as np

try:
  import portpicker  # pytype: disable=import-error
except ImportError:
  portpicker = None


class SocketTransferTest(jt_multiprocess.MultiProcessTest):

  def test_cross_host_transfer_single_device_sharding(self):
    x = np.arange(64).reshape(8, 8)
    src_pid = 0
    dst_pid = 1
    src_sharding = jax.sharding.SingleDeviceSharding(
        jax.local_devices(process_index=src_pid)[0])
    dst_sharding = jax.sharding.SingleDeviceSharding(
        jax.local_devices(process_index=dst_pid)[0])
    y = jax.device_put(x, src_sharding)
    z = jax.device_put(y, dst_sharding)
    z.block_until_ready()
    if jax.process_index() == dst_pid:
      self.assertLen(z.addressable_shards, 1)
      np.testing.assert_array_equal(z.addressable_shards[0].data, x)
    else:
      self.assertEmpty(z.addressable_shards)

  def test_cross_host_transfer_named_sharding(self):
    x = np.arange(64).reshape(8, 8)
    n_local = jax.local_device_count()
    src_pid = 0
    dst_pid = 1
    src_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((n_local,), ("x",),
                      devices=jax.local_devices(process_index=src_pid),
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P("x"))
    dst_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((n_local,), ("x",),
                      devices=jax.local_devices(process_index=dst_pid),
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P("x"))
    y = jax.device_put(x, src_sharding)
    z = jax.device_put(y, dst_sharding)
    z.block_until_ready()
    if jax.process_index() == dst_pid:
      self.assertLen(z.addressable_shards, n_local)
      for shard in z.addressable_shards:
        np.testing.assert_array_equal(shard.data, x[shard.index])
    else:
      self.assertEmpty(z.addressable_shards)

  def test_cross_host_transfer_named_sharding_replicated(self):
    x = np.arange(64).reshape(8, 8)
    n_dev = jax.device_count() // 2
    src_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((n_dev,), ("x",), devices=jax.devices()[:n_dev],
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P()
    )
    dst_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((n_dev,), ("x",), devices=jax.devices()[n_dev:],
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P()
    )
    y = jax.device_put(x, src_sharding)
    z = jax.device_put(y, dst_sharding)
    z.block_until_ready()
    for shard in z.addressable_shards:
      np.testing.assert_array_equal(shard.data, x[shard.index])


if __name__ == "__main__":
  if portpicker is None:
    socket_port = 12345
  else:
    socket_port = portpicker.pick_unused_port()
  jax.config.update("jax_force_dcn_cross_host_transfers", True)
  jax.config.update(
      "jax_cross_host_transfer_socket_address", f"127.0.0.1:{socket_port}")
  # Too small for good performance, but set to avoid oom in msan tests.
  jax.config.update(
      "jax_cross_host_transfer_transfer_size",
      64 * 1024,
  )
  jt_multiprocess.main()
