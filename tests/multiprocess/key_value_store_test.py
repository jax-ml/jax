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

"""Distributed key value store test."""

import jax
from jax._src import distributed
from jax._src import test_multiprocess as jt_multiprocess


class KeyValueStoreTest(jt_multiprocess.MultiProcessTest):
  def testBlockingKeyValueGet(self):
    client = distributed.global_state.client
    key = 'test_key'
    expected_value = 'JAX is great!'
    timeout_in_ms = 1000

    if jax.process_index() == 0:
      client.key_value_set(key, expected_value)
    actual_value = client.blocking_key_value_get(key, timeout_in_ms)

    self.assertEqual(expected_value, actual_value)

  def testBlockingKeyValueSetTwice(self):
    client = distributed.global_state.client
    key = 'test_key_' + str(jax.process_index())
    expected_value = 'JAX is great!'

    with self.assertRaisesRegex(
        jax.errors.JaxRuntimeError,
        r'ALREADY_EXISTS: key .* already exists.'
    ):
      client.key_value_set(key, expected_value)
      client.key_value_set(key, expected_value)

  def testBlockingKeyValueSetTwice_Overwrite(self):
    client = distributed.global_state.client
    key = 'test_key_overwrite_' + str(jax.process_index())
    initial_value = 'JAX is okay!'
    overwritten_value = 'JAX is great!'
    timeout_in_ms = 1000

    client.key_value_set(key, initial_value)
    client.key_value_set(key, overwritten_value, allow_overwrite=True)
    actual_value = client.blocking_key_value_get(key, timeout_in_ms)

    self.assertEqual(overwritten_value, actual_value)

  def testBlockingKeyValueGetBytes(self):
    client = distributed.global_state.client
    key = 'test_key2'
    expected_value = b'JAX is great!'
    timeout_in_ms = 1000

    if jax.process_index() == 0:
      client.key_value_set_bytes(key, expected_value)
    actual_value = client.blocking_key_value_get_bytes(key, timeout_in_ms)

    self.assertEqual(expected_value, actual_value)

  def testKeyValueTryGet(self):
    client = distributed.global_state.client
    key = 'test_key_try_get'
    expected_value = 'JAX is great!'
    if jax.process_index() == 0:
      client.key_value_set(key, expected_value)
    client.wait_at_barrier('kv_try_get_barrier', 1000)  # 1 second.

    actual_value = client.key_value_try_get(key)

    self.assertEqual(expected_value, actual_value)

  def testKeyValueTryGet_NotFound(self):
    client = distributed.global_state.client
    key = 'test_key_not_found'
    with self.assertRaisesRegex(
        jax.errors.JaxRuntimeError,
        r'NOT_FOUND: Config key .* not found.'
    ):
      client.key_value_try_get(key)

  def testKeyValueTryGetBytes(self):
    client = distributed.global_state.client
    key = 'test_key_try_get_bytes'
    expected_value = b'JAX is great!'
    if jax.process_index() == 0:
      client.key_value_set_bytes(key, expected_value)
    client.wait_at_barrier('kv_try_get_bytes_barrier', 1000)  # 1 second.

    actual_value = client.key_value_try_get_bytes(key)

    self.assertEqual(expected_value, actual_value)

  def testKeyValueDirGet(self):
    client = distributed.global_state.client
    kvs = [('dir/key0', 'value0'), ('dir/key2', 'value2'),
           ('dir/nested/key3', 'value3')]
    timeout_in_ms = 1000

    if jax.process_index() == 0:
      for kv in kvs:
        client.key_value_set(kv[0], kv[1])
    client.wait_at_barrier('wait_for_kv_set1', timeout_in_ms)
    actual_kvs = client.key_value_dir_get('dir/')
    self.assertSameElements(kvs, actual_kvs)

  def testKeyValueDirGetBytes(self):
    client = distributed.global_state.client
    kvs = [('dir2/key0', b'value0'), ('dir2/key2', b'avalue2'),
           ('dir2/nested/key3', b'avalue3')]
    timeout_in_ms = 1000

    if jax.process_index() == 0:
      for kv in kvs:
        client.key_value_set_bytes(kv[0], kv[1])
    client.wait_at_barrier('wait_for_kv_set2', timeout_in_ms)
    actual_kvs = client.key_value_dir_get_bytes('dir2/')
    self.assertSameElements(kvs, actual_kvs)

  def testLargeKeyValueDirGet(self):
    client = distributed.global_state.client
    value_size = 1024 * 1024  # bytes
    num_keys = 10
    kvs = [(f'dir3/key{i}', 'x' * value_size) for i in range(num_keys)]
    timeout_in_ms = 30 * 1000

    if jax.process_index() == 0:
      for kv in kvs:
        client.key_value_set(kv[0], kv[1])
    client.wait_at_barrier('wait_for_kv_set3', timeout_in_ms)
    actual_kvs = client.key_value_dir_get('dir3/')
    self.assertSameElements(kvs, actual_kvs)

if __name__ == '__main__':
  jt_multiprocess.main()
