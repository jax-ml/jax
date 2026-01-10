# Copyright 2021 The JAX Authors.
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

# pylint: disable=g-importing-member
import asyncio
from dataclasses import dataclass
from functools import partial
import json
import logging
import math
import os
import pathlib
import pickle
import tempfile
import threading
import time
import tracemalloc  as tm
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import array
from jax._src import config
from jax._src import test_util as jtu
from jax._src.export._export import (
    deserialization_registry as node_deserialization_registry)
from jax._src.export._export import (
    serialization_registry as node_serialization_registry)
from jax._src.layout import Layout
from jax._src.layout import Format
from jax.experimental.array_serialization import pytree_serialization
from jax.experimental.array_serialization import serialization
from jax.experimental.array_serialization import tensorstore_impl as ts_impl

from jax.experimental.array_serialization.pytree_serialization_utils import (
    register_pytree_node_serialization)

import jax.numpy as jnp

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import SingleDeviceSharding
import numpy as np
import tensorstore as ts
# pylint: enable=g-importing-member

jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


_DEFAULT_SHARDING = None  # to be overridden by tests with SingleDeviceSharding


def tree_load(*args, **kw):
  return pytree_serialization.load(*args, shardings=_DEFAULT_SHARDING, **kw)

tree_save = pytree_serialization.save
tree_load_pytreedef = pytree_serialization.load_pytreedef


def _get_replicated_sharding(devices):
  return NamedSharding(jax.sharding.Mesh(devices, 'x'), P())


class CheckpointTest(jtu.JaxTestCase):

  def _on_commit_callback(self, temp_ckpt_dir, final_ckpt_dir):
    os.rename(temp_ckpt_dir, final_ckpt_dir)

  def test_deserialize_on_array_list(self):
    global_mesh = jtu.create_mesh((2, 4), ('x', 'y'))
    inp_shape = (16, 64)
    pspec = P('x', 'y')
    sharding = NamedSharding(global_mesh, pspec)
    inputs = []
    lambda_fn = lambda idx: src[idx]
    num_arrays = 5
    for _ in range(num_arrays):
      src = jax.random.normal(jax.random.key(0), inp_shape)
      inp = array.make_array_from_callback(inp_shape, sharding, lambda_fn)
      inputs.append(inp)
    ckpt_dir = pathlib.Path(self.create_tempdir().full_path)
    tspecs = [
        serialization.get_tensorstore_spec(f'{ckpt_dir}/array_{i}')
        for i in range(num_arrays)
    ]
    inputs = tuple(inputs)
    tspecs = tuple(tspecs)
    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        inputs,
        tspecs,
        on_commit_callback=partial(
            self._on_commit_callback, ckpt_dir, ckpt_dir
        ),
    )
    manager.wait_until_finished()
    shardings = tuple([sharding] * num_arrays)
    restored_arrays = manager.deserialize(shardings, tspecs)
    self.assertLen(restored_arrays, num_arrays)
    for inp, deserialized_array in zip(inputs, restored_arrays):
      self.assertArraysEqual(deserialized_array, inp)

  @jtu.skip_on_devices('cpu')
  @jtu.thread_unsafe_test()
  def test_memory_consumption(self):
    global_mesh = jtu.create_mesh((2, 4), ('x', 'y'))
    inp_shape = (2_048, 4_096)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)
    sharding = NamedSharding(global_mesh, pspec)
    src = jnp.arange(num, dtype=np.int32).reshape(inp_shape)  # 8e6 elements
    inp = array.make_array_from_callback(
        inp_shape, sharding,
        lambda idx: src[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir(
        'memprof-deserialize').full_path)
    tspec = serialization.get_tensorstore_spec(str(ckpt_dir))

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [inp], [tspec],
        on_commit_callback=partial(
            self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    async def deserialize_with_byte_limit():
      r = await serialization.async_deserialize(
          sharding, tspec, inp_shape,
          byte_limiter=serialization._LimitInFlightBytes(4_200_000))
      r.block_until_ready()

    tm.start()
    asyncio.run(deserialize_with_byte_limit())
    unused_current, peak = tm.get_traced_memory()
    # NB: some padding + tensorstore overhead. It should always be
    # less than array size (2048 * 4096 * 4 = 32M)
    self.assertLess(peak, 13_000_000)
    deserialize_wo_limit = serialization.async_deserialize(
        sharding, tspec, inp_shape)
    tm.clear_traces()
    # NB: call block_until_ready() is important here and above
    # because otherwise this leads to racing condition and segfault with
    # tensorstore attempting to dealloc using tracemalloc which is already
    # destroyed.
    asyncio.run(deserialize_wo_limit).block_until_ready()

    unused_current, peak = tm.get_traced_memory()
    # We load entire array in memory here.
    self.assertGreater(peak, 30_000_000)
    tm.stop()

  @jtu.thread_unsafe_test()
  def test_memory_consumption_for_save(self):
    global_mesh = jtu.create_mesh((1, 1), ('x', 'y'))
    inp_shape = (16 * 1024, 16 * 1024)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)
    sharding = NamedSharding(global_mesh, pspec)
    src = jnp.arange(num, dtype=np.int32).reshape(inp_shape)
    inp = array.make_array_from_callback(
        inp_shape, sharding, lambda idx: src[idx]
    )
    ckpt_dir = pathlib.Path(self.create_tempdir(
        'memprofsave-serialize').full_path)
    tspec = ts_impl.get_tensorstore_spec(str(ckpt_dir), ocdbt=False,
                                         driver='zarr3')
    tspec['metadata'] = {
        'shape': inp.shape,
        'data_type': jnp.dtype(inp.dtype).name,
        'chunk_grid': {
            'name': 'regular',
            'configuration': {'chunk_shape': np.array(np.maximum(1, inp.shape))}
        }
    }
    is_cpu = jtu.test_device_matches(['cpu'])
    tm.start()
    try:
      manager = serialization.GlobalAsyncCheckpointManager()
      manager.serialize([inp], [tspec], on_commit_callback=partial(
          self._on_commit_callback, ckpt_dir, ckpt_dir))
      manager.wait_until_finished()
      unused_current, peak = tm.get_traced_memory()
      self.assertLess(peak, src.nbytes * (1 * (not is_cpu) + 0.5))
    finally:
      tm.stop()

  def test_checkpointing_with_path_variant(self):
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    inp_shape = (8, 2)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)

    # First Array
    global_input_data1 = np.arange(num, dtype=np.int32).reshape(inp_shape)
    a1 = array.make_array_from_callback(
        inp_shape, NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data1[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir('ckpt_variant').full_path)
    ckpt_path1 = pathlib.Path(
        self.create_tempdir(f'{ckpt_dir}/first').full_path)

    ckpt_paths = [str(ckpt_path1)]
    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize_with_paths(
        [a1], ckpt_paths,
        on_commit_callback=partial(
            self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    m1, = manager.deserialize_with_paths(
        [NamedSharding(global_mesh, pspec)], ckpt_paths)
    self.assertIsInstance(m1, array.ArrayImpl)
    self.assertArraysEqual(np.asarray(m1.addressable_shards[0].data),
                           np.array([[0], [2]], dtype=np.int32))
    self.assertArraysEqual(np.asarray(m1.addressable_shards[1].data),
                           np.array([[1], [3]], dtype=np.int32))
    self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
    self.assertEqual(m1.dtype, np.int32)

  def test_checkpointing_jax_array(self):
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    inp_shape = (8, 2)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)

    # First Array
    global_input_data1 = np.arange(num, dtype=np.int32).reshape(inp_shape)
    a1 = array.make_array_from_callback(
        inp_shape, NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data1[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir('ckpt').full_path)
    ckpt_path1 = pathlib.Path(
        self.create_tempdir(f'{ckpt_dir}/first').full_path)

    # Second Array
    global_input_data2 = np.arange(
        num, num + num, dtype=np.int32).reshape(inp_shape)
    a2 = array.make_array_from_callback(
        inp_shape, NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data2[idx])
    ckpt_path2 = pathlib.Path(
        self.create_tempdir(f'{ckpt_dir}/second').full_path)

    # Third Array
    def cb3(_):
      return np.array([], dtype=np.float32)
    global_mesh1d = jtu.create_mesh((8,), ('x',))
    a3 = array.make_array_from_callback(
        (0,), NamedSharding(global_mesh1d, P(None)), cb3)
    ckpt_path3 = pathlib.Path(
        self.create_tempdir(f'{ckpt_dir}/third').full_path)

    ckpt_paths = [str(ckpt_path1), str(ckpt_path2), str(ckpt_path3)]
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [a1, a2, a3], tspecs,
        on_commit_callback=partial(
            self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    m1, m2, m3 = serialization.run_deserialization(
        [NamedSharding(global_mesh, pspec),
         NamedSharding(global_mesh, P('x')),
         NamedSharding(global_mesh1d, P(None))],
        tspecs)

    self.assertIsInstance(m1, array.ArrayImpl)
    self.assertArraysEqual(np.asarray(m1.addressable_shards[0].data),
                           np.array([[0], [2]], dtype=np.int32))
    self.assertArraysEqual(np.asarray(m1.addressable_shards[1].data),
                           np.array([[1], [3]], dtype=np.int32))
    self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
    self.assertEqual(m1.dtype, np.int32)

    self.assertIsInstance(m2, array.ArrayImpl)
    self.assertArraysEqual(np.asarray(m2.addressable_shards[0].data),
                           np.array([[16, 17], [18, 19]], dtype=np.int32))
    self.assertArraysEqual(np.asarray(m2.addressable_shards[1].data),
                           np.array([[16, 17], [18, 19]], dtype=np.int32))
    self.assertEqual(m2.addressable_shards[0].data.shape, (2, 2))
    self.assertEqual(m2.dtype, np.int32)

    self.assertIsInstance(m3, array.ArrayImpl)
    for i, s in enumerate(m3.addressable_shards):
      self.assertEqual(s.index, (slice(None),))
      self.assertEqual(s.replica_id, i)
      self.assertArraysEqual(np.asarray(s.data), np.array([], dtype=np.float32))
    self.assertEqual(m3.dtype, np.float32)

  @jtu.thread_unsafe_test()
  def test_deserialization_does_not_hang_when_concurrent_gb_exceeded(self):
    TIMEOUT_SEC = 5
    mngr = serialization.GlobalAsyncCheckpointManager(timeout_secs=TIMEOUT_SEC)
    a = jnp.ones((1024, 1024))
    path = str(pathlib.Path(self.create_tempdir('small').full_path) / 'array')
    mngr.serialize_with_paths([a], [path])
    mngr.wait_until_finished()

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(mngr.deserialize_with_paths, [a.sharding],
                             [path], concurrent_gb=1e-9)  # 1 byte
    try:
      future.result(timeout=TIMEOUT_SEC)
    except TimeoutError:
      future.cancel()
      self.fail('Deserialization times out if size exceeds concurrent_gb.')

  def test_checkpointing_ocdbt_transaction(self):
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    inp_shape = (8, 2)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)

    # First Array
    global_input_data1 = np.arange(num, dtype=np.int32).reshape(inp_shape)
    a1 = array.make_array_from_callback(
        inp_shape,
        NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data1[idx],
    )
    ckpt_dir = pathlib.Path(self.create_tempdir('ckpt').full_path)
    ckpt_path1 = ckpt_dir / 'first'

    # Second Array
    global_input_data2 = np.arange(num, num + num, dtype=np.int32).reshape(
        inp_shape
    )
    a2 = array.make_array_from_callback(
        inp_shape,
        NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data2[idx],
    )
    ckpt_path2 = ckpt_dir / 'second'

    # Third Array
    def cb3(_):
      return np.array([], dtype=np.float32)

    global_mesh1d = jtu.create_mesh((8,), ('x',))
    a3 = array.make_array_from_callback(
        (0,), NamedSharding(global_mesh1d, P(None)), cb3
    )
    ckpt_path3 = ckpt_dir / 'third'

    ckpt_paths = [str(ckpt_path1), str(ckpt_path2), str(ckpt_path3)]
    tspecs = jax.tree.map(partial(ts_impl.get_tensorstore_spec, ocdbt=True),
                          ckpt_paths)

    manager = serialization.GlobalAsyncCheckpointManager()
    with ts.Transaction(atomic=True) as transaction:
      manager.serialize(
          [a1, a2, a3],
          tspecs,
          on_commit_callback=partial(
              self._on_commit_callback, ckpt_dir, ckpt_dir
          ),
          transaction=transaction,
      )
    manager.wait_until_finished()

    m1, m2, m3 = serialization.run_deserialization(
        [NamedSharding(global_mesh, pspec), NamedSharding(global_mesh, P('x')),
         NamedSharding(global_mesh1d, P(None))], tspecs)

    self.assertIsInstance(m1, array.ArrayImpl)
    self.assertArraysEqual(
        np.asarray(m1.addressable_shards[0].data),
        np.array([[0], [2]], dtype=np.int32),
    )
    self.assertArraysEqual(
        np.asarray(m1.addressable_shards[1].data),
        np.array([[1], [3]], dtype=np.int32),
    )
    self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
    self.assertEqual(m1.dtype, np.int32)

    self.assertIsInstance(m2, array.ArrayImpl)
    self.assertArraysEqual(
        np.asarray(m2.addressable_shards[0].data),
        np.array([[16, 17], [18, 19]], dtype=np.int32),
    )
    self.assertArraysEqual(
        np.asarray(m2.addressable_shards[1].data),
        np.array([[16, 17], [18, 19]], dtype=np.int32),
    )
    self.assertEqual(m2.addressable_shards[0].data.shape, (2, 2))
    self.assertEqual(m2.dtype, np.int32)

    self.assertIsInstance(m3, array.ArrayImpl)
    for i, s in enumerate(m3.addressable_shards):
      self.assertEqual(s.index, (slice(None),))
      self.assertEqual(s.replica_id, i)
      self.assertArraysEqual(np.asarray(s.data), np.array([], dtype=np.float32))
    self.assertEqual(m3.dtype, np.float32)

  @parameterized.product(input_dtype=[np.int32, jnp.bfloat16])
  def test_checkpointing_with_bigger_shape_jax_array(self, input_dtype):
    global_mesh = jtu.create_mesh((2, 2), ('x', 'y'), iota_order=True)
    global_input_shape = (8, 2)
    num = math.prod(global_input_shape)

    global_input_data1 = np.arange(num, dtype=input_dtype).reshape(
        global_input_shape
    )
    def cb1(index):
      return global_input_data1[index]
    arr = array.make_array_from_callback(
        global_input_shape, NamedSharding(global_mesh, P('x', 'y')), cb1)
    ckpt_dir = pathlib.Path(self.create_tempdir('first').full_path)

    ckpt_paths = [str(ckpt_dir)]
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [arr], tspecs,
        on_commit_callback=partial(
            self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    ds = NamedSharding(jtu.create_mesh((4, 2), ('x', 'y'), iota_order=True),
                       P('x', 'y'))

    m1, = serialization.run_deserialization([ds], tspecs, [(12, 2)],
                                            [np.float32])

    expected_data = {
        0: np.array([[0], [2], [4]], dtype=np.float32),
        1: np.array([[1], [3], [5]], dtype=np.float32),
        2: np.array([[6], [8], [10]], dtype=np.float32),
        3: np.array([[7], [9], [11]], dtype=np.float32),
        4: np.array([[12], [14], [0]], dtype=np.float32),
        5: np.array([[13], [15], [0]], dtype=np.float32),
        6: np.array([[0], [0], [0]], dtype=np.float32),
        7: np.array([[0], [0], [0]], dtype=np.float32),
    }

    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), expected_data[l.device.id])

    new_ds = _get_replicated_sharding(list(global_mesh.devices.flat))
    m2, = serialization.run_deserialization([new_ds], tspecs, [(8, 2)],
                                            [np.float32])
    for l in m2.addressable_shards:
      self.assertArraysEqual(l.data, global_input_data1.astype('float32'))

  @parameterized.product(input_dtype=[jnp.int4, jnp.int8])
  def test_checkpointing_with_int4(self, input_dtype):
    if config.use_shardy_partitioner.value:
      self.skipTest('TODO(b/376077396): Fix JaxRuntimeError: INVALID_ARGUMENT')
    global_mesh = jtu.create_mesh((2, 2), ('x', 'y'), iota_order=True)
    global_input_shape = (8, 2)
    num = math.prod(global_input_shape)

    global_input_data = np.arange(num, dtype=input_dtype).reshape(
        global_input_shape
    )
    def cb(index):
      return global_input_data[index]
    arr = array.make_array_from_callback(
        global_input_shape, NamedSharding(global_mesh, P('x', 'y')), cb)
    ckpt_dir = pathlib.Path(self.create_tempdir('first').full_path)

    ckpt_paths = [str(ckpt_dir)]
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [arr], tspecs,
        on_commit_callback=partial(
            self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    ds = NamedSharding(jtu.create_mesh((4, 2), ('x', 'y'), iota_order=True),
                       P('x', 'y'))

    target_dtype = jnp.dtype('int4')
    m1, = serialization.run_deserialization([ds], tspecs, [(12, 2)],
                                            [target_dtype])

    # values bigger than 7 are converted properly.
    expected_data = {
        0: jnp.array([[0], [2], [4]], dtype=target_dtype),
        1: jnp.array([[1], [3], [5]], dtype=target_dtype),
        2: jnp.array([[6], [8], [10]], dtype=target_dtype),
        3: jnp.array([[7], [9], [11]], dtype=target_dtype),
        4: jnp.array([[12], [14], [0]], dtype=target_dtype),
        5: jnp.array([[13], [15], [0]], dtype=target_dtype),
        6: jnp.array([[0], [0], [0]], dtype=target_dtype),
        7: jnp.array([[0], [0], [0]], dtype=target_dtype),
    }

    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), expected_data[l.device.id])

    new_ds = _get_replicated_sharding(list(global_mesh.devices.flat))
    m2, = serialization.run_deserialization([new_ds], tspecs, [(8, 2)],
                                            [target_dtype])
    for l in m2.addressable_shards:
      self.assertArraysEqual(l.data, global_input_data.astype(target_dtype))

  def test_checkpointing_scalar_jax_array(self):
    global_mesh = jtu.create_mesh((2,), ('x'))
    global_input_shape = ()
    data = np.array(4)
    s = NamedSharding(global_mesh, P(None))
    array1 = array.make_array_from_callback(
        global_input_shape, s, lambda idx: data[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir('first').full_path)

    ckpt_paths = [str(ckpt_dir)]
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, ckpt_paths)
    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [array1], tspecs,
        on_commit_callback=partial(
            self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    ds = NamedSharding(jtu.create_mesh((2,), ('x')), P(None))

    m1, = serialization.run_deserialization([ds], tspecs, [()], [np.float32])

    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), data.astype(np.float32))

  def test_deserialize_tensorstore_array_jax_array(self):
    global_mesh = jtu.create_mesh((2,), ('x'))
    data = np.arange(1024)
    tspec = ts.array(data).spec()
    m1, = serialization.run_deserialization(
        [NamedSharding(global_mesh, P(None))], [tspec])
    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), data)

  def test_spec_has_metadata(self):
    spec = {
        'a': {
            'b': 1,
            'c': 2,
        },
        'd': 3,
        'e': {
            'a': 2,
            'metadata': 3
        },
        'f': 4
    }
    self.assertTrue(ts_impl._spec_has_metadata(spec))
    self.assertTrue(
        ts_impl._spec_has_metadata({
            'driver': 'zarr',
            'kvstore': 'gfile',
            'metadata': {
                'chunks': 4,
                'shape': (32, 64)
            },
            'one_more': 'thing'
        }))

  def test_spec_has_no_metadata(self):
    spec = {
        'a': {
            'b': 1,
            'c': 2,
        },
        'd': 3,
        'e': {
            'a': 2,
        },
        'f': 4
    }
    self.assertFalse(ts_impl._spec_has_metadata(spec))

  def test_empty_spec_has_no_metadata(self):
    spec = {}
    self.assertFalse(ts_impl._spec_has_metadata(spec))

  @parameterized.named_parameters(
      ('gcs', 'gs://my/ckpt/dir/path'),
      ('file', '/my/ckpt/dir/path')
  )
  def test_get_tensorstore_spec_ocdbt(self, path):
    spec = ts_impl.get_tensorstore_spec(path, ocdbt=True)
    is_gcs_path = path.startswith('gs://')
    # for OCDBT the last part of the path is the key in the kvstore
    expected_path = os.path.split(path)[0]
    if is_gcs_path:
      self.assertEqual(spec['kvstore']['base']['driver'], 'gcs')
      self.assertTrue(expected_path.endswith(spec['kvstore']['base']['path']))
    else:
      self.assertEqual(spec['kvstore']['base']['path'], expected_path)

  def test_get_tensorstore_spec_not_absolute_path(self):
    path = 'my/ckpt/path'
    with self.assertRaisesRegex(ValueError,
                                'Checkpoint path should be absolute'):
      ts_impl.get_tensorstore_spec(path, ocdbt=True)

  def test_maybe_cloud_storage(self):
    gs_path = 'gs://some-buck/path/array_name'
    gs_spec = ts_impl.get_tensorstore_spec(gs_path, ocdbt=True)
    self.assertTrue(serialization.is_remote_storage(gs_spec))

    local_path = '/tmp/checkpoint/array_name'
    local_spec = ts_impl.get_tensorstore_spec(local_path, ocdbt=True)
    self.assertFalse(serialization.is_remote_storage(local_spec))

    nested_tspec = {
        'driver': 'cast',
        'dtype': 'int32',
        'base': {
            'driver': 'zarr',
            'kvstore': {'driver': 'ocdbt',
                        'base': 's3://some-bucket/path/array_name'},
        },
    }
    self.assertTrue(serialization.is_remote_storage(nested_tspec))

  @parameterized.named_parameters(('4_devices', 4), ('1_device', 1))
  def test_load_with_layout(self, device_count):
    if not jtu.test_device_matches(['tpu']):
      self.skipTest('Layouts are only supported on TPUs')

    mesh = jtu.create_mesh((2, 2) if device_count == 4 else (1, 1), ('x', 'y'))
    np_inp = np.arange(device_count * 128 * 128 * 2).reshape((-1, 2 * 128))
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    transpose_layout = Layout(
        major_to_minor=arr.format.layout.major_to_minor[::-1],
        tiling=arr.format.layout.tiling)
    out_format = Format(transpose_layout, arr.sharding)
    out_ref = jax.device_put(arr.T, out_format)

    ckpt_dir = pathlib.Path(self.create_tempdir('ckpt').full_path)
    ckpt_path = pathlib.Path(self.create_tempdir(f'{ckpt_dir}/first').full_path)
    tspecs = jax.tree.map(ts_impl.get_tensorstore_spec, [ckpt_path])

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [out_ref], tspecs,
        on_commit_callback=partial(
            self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    out_specs = [out_format, jax.ShapeDtypeStruct(
        out_ref.shape, out_ref.dtype, sharding=out_format)]

    for out_spec, name in zip(out_specs, ['format', 'ShapeDtypeStruct']):
      with self.subTest(f'deserialization_with_{name}'):
        out, = serialization.run_deserialization([out_spec], tspecs)
        self.assertEqual(out.format, out_format)
        self.assertIsInstance(out, array.ArrayImpl)
        self.assertArraysEqual(out, out_ref)
        for s, s_ref in zip(out.addressable_shards, out_ref.addressable_shards):
          self.assertArraysEqual(s.data, s_ref.data)

  def test_deserialization_with_int4(self):
    if config.use_shardy_partitioner.value:
      self.skipTest('TODO(b/376077396): Fix JaxRuntimeError: INVALID_ARGUMENT')
    if jtu.test_device_matches(['gpu']):
      self.skipTest("Fails on GPU. Enable after it's fixed")
    dtype = jnp.int4
    shape = (8, 2)
    arr = jnp.arange(np.prod(shape)).reshape(shape).astype(dtype)

    ckpt_dir = pathlib.Path(self.create_tempdir('test_ckpt').full_path)

    # Run serialization.
    sharding = _get_replicated_sharding(list(jax.devices()))
    tspecs = jax.tree.map(serialization.get_tensorstore_spec, [ckpt_dir])
    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [arr],
        tspecs,
        on_commit_callback=lambda: None,
    )
    manager.wait_until_finished()

    # Run deserialization.
    deserialized_arr, = serialization.run_deserialization(
        shardings=[sharding], tensorstore_specs=tspecs, global_shapes=[shape],
        dtypes=[dtype])

    out = deserialized_arr.astype(jnp.int8)  # doesn't crash
    self.assertEqual(out.dtype, jnp.int8)
    self.assertArraysEqual(out + out, out * 2)


class TransferShardTest(jtu.JaxTestCase):

  @jtu.skip_on_devices('cpu')
  def test_transfer_shard_to_host(self):
    np_inp = np.arange(16).reshape((4, 4))
    sharding = SingleDeviceSharding(jax.devices()[0], memory_kind='device')
    arr = jax.device_put(np_inp, sharding)
    shard = arr.addressable_shards[0]

    np_out = asyncio.run(ts_impl._transfer_shard_to_host(shard))

    self.assertArraysEqual(np_out, np_inp)


def _remove_from_serialization_registry(t: Any):
  if t in node_serialization_registry:
    serialized_name = node_serialization_registry[t][0]
    del node_serialization_registry[t]
    del node_deserialization_registry[serialized_name]


class UserAPITestCase(jtu.JaxTestCase):
  name: str | None
  path: pathlib.Path | None

  def setUp(self):
    super().setUp()
    tmpdir = tempfile.TemporaryDirectory()
    self.enter_context(tmpdir)
    self.name = tmpdir.name
    self.path = pathlib.Path(self.name)

  def tearDown(self):
    self.path = None
    self.name = None
    super().tearDown()

  def generate_random_fp32(self, shape, dtype=jnp.float32):
    seed = round(time.time() * 1e6) % (2 ** 31)
    key = jax.random.key(seed)
    return jax.random.normal(key, shape=shape).astype(dtype)

  def generate_clean_tree(self, dtype=jnp.float32):
    r1 = self.generate_random_fp32((), dtype=dtype)
    r2 = self.generate_random_fp32((4,), dtype=dtype)
    r3 = self.generate_random_fp32((2, 3), dtype=dtype)
    return (r1, {'a': r2, 'rs': [r1, r2, r3], 'c': {'d': {'e': (r2,)}}})

  def _is_equal(self, el1, el2):
    if not isinstance(el1, type(el2)) or not isinstance(el2, type(el1)):
      return False
    if isinstance(el1, (np.ndarray, jax.Array)):
      return (el1.dtype == el2.dtype and el1.shape == el2.shape
              and jnp.allclose(el1, el2))
    else:
      return el1 == el2

  def assertPyTreeEqual(self, p1, p2, is_leaf=None):
    leaves1, struct1 = jax.tree.flatten(p1, is_leaf=is_leaf)
    leaves2, struct2 = jax.tree.flatten(p2, is_leaf=is_leaf)
    self.assertEqual(struct1, struct2)
    self.assertTrue(all(self._is_equal(el1, el2)
                        for (el1, el2) in zip(leaves1, leaves2)))

_DTYPES_LIST = [
    jnp.uint8,
    jnp.uint16,
    jnp.uint32,
    jnp.int8,
    jnp.int16,
    jnp.int32,
    jnp.float8_e4m3fn,
    jnp.float8_e4m3fnuz,
    jnp.float8_e5m2,
    jnp.float8_e5m2fnuz,
    jnp.float8_e4m3b11fnuz,
    jnp.bfloat16,
    jnp.float16,
    jnp.float32,
    jnp.complex64,
]

_X64_DTYPES_LIST = [
    jnp.uint64,
    jnp.int64,
    jnp.float64,
    jnp.complex128,
]

if jax.config.x64_enabled:
  _DTYPES_LIST.extend(_X64_DTYPES_LIST)


@jax.tree_util.register_pytree_node_class
class CustomNode:
  def __init__(self, a):
    self.a = a

  def tree_flatten(self):
    return (self.a,), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


@partial(jax.tree_util.register_dataclass, data_fields=['a', 'd'],
                   meta_fields=['c'])
@dataclass
class CustomDataclass:
  a: int
  c: str
  d: int


@jax.tree_util.register_static
class CustomStatic:
  def __init__(self, a):
    self.a = a

# we're testing custom type registration which modifies the global registry
# so need to ensure we're not running multiple custom types tests in parallel
custom_types_threading_lock = threading.Lock()


class UserPytreeAPITest(UserAPITestCase):
  def setUp(self):
    super().setUp()
    global _DEFAULT_SHARDING
    _DEFAULT_SHARDING = SingleDeviceSharding(jax.devices()[0])
    self.tempdirs = []

  def tearDown(self):
    for tempdir in self.tempdirs:
      tempdir.cleanup()
    super().tearDown()

  def create_tempdir(self):
    tempdir = tempfile.TemporaryDirectory()
    self.tempdirs.append(tempdir)
    return pathlib.Path(tempdir.name).resolve()

  @parameterized.product(tree=[{'a': 1}, [1, 2, 3], (1, 2, 3), 1, 2, 3])
  def test_save_then_load(self, tree):  # pylint: disable=redefined-outer-name
    path = self.create_tempdir()
    tree = jax.tree.map(jnp.array, tree)
    tree_save(tree, path)
    tree2 = tree_load(path)
    self.assertPyTreeEqual(tree, tree2)

  @parameterized.product(dtype=_DTYPES_LIST)
  def test_saving_dtype(self, dtype):
    if dtype in _X64_DTYPES_LIST and jtu.test_device_matches(['tpu']):
      self.skipTest('Don\'t test x64 dtypes on TPUs')
    path = self.create_tempdir()
    test_tree = self.generate_clean_tree(dtype=dtype)
    tree_save(test_tree, path)
    new_tree = tree_load(path)
    self.assertPyTreeEqual(test_tree, new_tree)

  def test_do_not_overwrite_noncheckpoint_directories(self):
    path = self.create_tempdir()
    path.mkdir(exist_ok=True)
    (path / 'hello.txt').write_text('Hello World')
    with self.assertRaisesRegex(RuntimeError, 'Refusing to work on a directory'
                                ' that is not a previous checkpoint.'):
      tree_save({'a': jnp.ones(1)}, path)

  def test_checkpoint_exists(self):
    path = self.create_tempdir()
    tree_save({'a': jnp.ones(1)}, path)
    with self.assertRaises(ValueError):
      tree_save({'a': jnp.ones(1)}, path, overwrite=False)

  @parameterized.product(test_load_fail=[True, False])
  def test_custom_types(self, test_load_fail):
    path = self.create_tempdir()
    with custom_types_threading_lock:
      magic_value = jnp.ones(()) * 37
      n = CustomNode(magic_value)
      d = CustomDataclass(magic_value, 'hello', magic_value + 1)
      s = CustomStatic(magic_value - 1)
      tree_to_save = [n, (d, s)]

      register_pytree_node_serialization(CustomNode,
                                        serialized_name='CustomNode',
                                        serialize_auxdata=pickle.dumps,
                                        deserialize_auxdata=pickle.loads)
      register_pytree_node_serialization(CustomStatic,
                                        serialized_name='CustomStatic',
                                        serialize_auxdata=pickle.dumps,
                                        deserialize_auxdata=pickle.loads)
      register_pytree_node_serialization(CustomDataclass,
                                        serialized_name='CustomDataclass',
                                        serialize_auxdata=pickle.dumps,
                                        deserialize_auxdata=pickle.loads)
      tree_save(tree_to_save, path)
      if test_load_fail:
        _ = [_remove_from_serialization_registry(cls)
            for cls in [CustomStatic, CustomNode, CustomDataclass]]
        with self.assertRaises(ValueError):
          _ = tree_load(path)
      else:
        tree2 = tree_load(path)
        self.assertEqual(tree2[0].a, magic_value)
        self.assertEqual(tree2[1][0].a, magic_value)
        self.assertEqual(tree2[1][0].c, 'hello')
        self.assertEqual(tree2[1][0].d, magic_value + 1)
        self.assertEqual(tree2[1][1].a, magic_value - 1)
        _ = [_remove_from_serialization_registry(cls)
            for cls in [CustomStatic, CustomNode, CustomDataclass]]

  def test_flax_frozen_dict(self):
    path = self.create_tempdir()
    try:
      # pylint: disable=g-import-not-at-top
      # pylint: disable=g-importing-member
      from flax.core.frozen_dict import FrozenDict
      # pylint: enable=g-importing-member
      # pylint: enable=g-import-not-at-top
    except ImportError:
      logging.warning('Skipping Flax FrozenDict tests as flax is not installed')
      return

    try:
      register_pytree_node_serialization(FrozenDict,
                                         serialized_name='FrozenDict',
                                         serialize_auxdata=pickle.dumps,
                                         deserialize_auxdata=pickle.loads)
      tree_save(FrozenDict(a=1, b=self.generate_clean_tree()), path)
      tree_load(path)
    finally:
      _remove_from_serialization_registry(FrozenDict)

  def test_register_as_decorator(self):
    @partial(register_pytree_node_serialization,
                       serialized_name='CustomDNode',
                       serialize_auxdata=json.dumps,
                       deserialize_auxdata=json.loads)
    @partial(jax.tree_util.register_dataclass, data_fields=['a', 'b'],
                      meta_fields=[])
    @dataclass
    class CustomDNode:
      a: int
      b: int

    # test whether the object can be created (is visible in this scope)
    _ = CustomDNode(1, 2)

  def test_custom_node_registration(self):
    path = self.create_tempdir()

    @jax.tree_util.register_static
    @dataclass
    class P:
      a: int = 2

    @partial(jax.tree_util.register_dataclass, data_fields=['a', 'b'],
                       meta_fields=['op'])
    @dataclass
    class D:
      a: Any
      b: Any
      op: str

    def serialize_D(data):
      return json.dumps(jax.tree.map(lambda x: np.array(x).tolist(), data)
                        ).encode('utf-8')

    def deserialize_D(data):
      return jnp.array(json.loads(data))

    data = [jnp.ones(1), {'world': [jnp.zeros(3), (jnp.ones(1), jnp.ones(2))]},
            7 * jnp.ones(()), P()]

    serialize_fn = lambda p: json.dumps(int(p.a)).encode('utf-8')
    deserialize_fn = lambda data: P(json.loads(data))

    with self.assertRaises(ValueError):
      tree_save(data, path)

    register_pytree_node_serialization(P,
                                       serialized_name='P',
                                       serialize_auxdata=serialize_fn,
                                       deserialize_auxdata=deserialize_fn)
    magic_value = -171
    data[-1].a = jnp.array(magic_value)
    tree_save(data, path)
    ret = tree_load(path)
    self.assertLen(ret, len(data))
    self.assertEqual(ret[-1].a, magic_value)

    magic_val = 17 * jnp.ones(2)
    data.append(D(jnp.ones(1), jax.numpy.zeros(2), magic_val))
    with self.assertRaises(ValueError):
      tree_save(data, path)

    register_pytree_node_serialization(D,
                                       serialized_name='D',
                                       serialize_auxdata=serialize_D,
                                       deserialize_auxdata=deserialize_D)
    tree_save(data, path)
    ret = tree_load(path)
    self.assertLen(ret, len(data))
    self.assertLess(jnp.linalg.norm(ret[-1].op - magic_val), 1e-5)

    jax.tree.flatten(data)

  def test_masked_reading(self):
    path = self.create_tempdir()
    data = [jnp.ones(1), {'world': [jnp.zeros(3), (jnp.ones(1), jnp.ones(2))]},
            7 * jnp.ones(())]
    tree_save(data, path)
    for mask in [False, True]:
      ret = tree_load(path, mask=mask)
      expected = jax.tree.map(lambda x: None if not mask else x, data)
      self.assertPyTreeEqual(ret, expected, is_leaf=lambda x: x is None)

    mask = [True, False, False]
    expected = data[:1] + jax.tree.map(lambda x: None, data[1:])
    ret = tree_load(path, mask=mask)
    self.assertPyTreeEqual(ret, expected, is_leaf=lambda x: x is None)

    mask = [True, True, False]
    expected = data[:2] + jax.tree.map(lambda x: None, data[2:])
    ret = tree_load(path, mask=mask)
    self.assertPyTreeEqual(ret, expected, is_leaf=lambda x: x is None)

    mask = [True, {'world': [True, (False, True)]}, False]
    data[1]['world'][1] = (None, data[1]['world'][1][1])
    ret = tree_load(path, mask=mask)
    self.assertPyTreeEqual(ret, expected, is_leaf=lambda x: x is None)

  # TODO(rdyro): Remove when serialization supports non-arrays
  @parameterized.product(obj=[b'hello', 'hello', 1, 1.0, 1j])
  def test_serialization_works_for_arrays_only(self, obj):
    path = self.create_tempdir()
    data = [{'world': [jnp.zeros(3), (jnp.ones(1), jnp.ones(2))]}, obj]
    msg = ('For serialization, all leaves must be either None or'
           ' jax.Array-like objects.')
    with self.assertRaisesRegex(ValueError, msg):
      tree_save(data, path)

  def test_load_pytreedef(self):
    path = self.create_tempdir()
    data = [jnp.ones(1), {'world': [jnp.zeros(3), (jnp.ones(1), jnp.ones(2))]},
            7 * jnp.ones(())]
    tree_save(data, path)
    pytreedef = tree_load_pytreedef(path)
    expected_pytreedef = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), data)
    self.assertPyTreeEqual(pytreedef, expected_pytreedef)

  @parameterized.product(data=[
      None, [None], [None, np.ones(())],
      [None, {'world': [None, (np.ones(1), np.ones(2))]}, np.ones(())],
      [None, {'world': [np.zeros(3), (None, np.ones(2))]}, None]])
  def test_save_and_load_null_leaves(self, data):
    path = self.create_tempdir()
    # TPUs might not have X64 enabled, so we need to convert to float32
    data = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float32), data)
    tree_save(data, path)
    pytreedef = tree_load_pytreedef(path)
    is_leaf = lambda x: x is None
    expected_pytreedef = jax.tree.map(lambda x: jax.ShapeDtypeStruct(
        x.shape, x.dtype) if x is not None else x, data, is_leaf=is_leaf)
    self.assertPyTreeEqual(pytreedef, expected_pytreedef)
    load_data = tree_load(path)
    load_leaves, load_struct = jax.tree.flatten(load_data, is_leaf=is_leaf)
    expected_leaves, expected_struct = jax.tree.flatten(data, is_leaf=is_leaf)
    self.assertEqual(load_struct, expected_struct)
    self.assertLen(load_leaves, len(expected_leaves))
    for (l1, l2) in zip(load_leaves, expected_leaves):
      if l1 is None:
        self.assertIsNone(l2)
      else:
        self.assertArraysEqual(l1, l2)

  @parameterized.product(manually_broadcast_ts_specs=[True, False])
  def test_custom_ts_specs(self, manually_broadcast_ts_specs):
    if ts_impl._TS_ARRAY_DRIVER == 'zarr':
      self.skipTest('Skipping since this test assumes zarr is NOT the default')
    path = self.create_tempdir()
    data = [jnp.ones(()), (jnp.zeros(()), jnp.ones(())), None]
    ts_spec = {'driver': 'zarr', 'metadata': {'shape': ()}}
    if manually_broadcast_ts_specs:
      ts_specs = [ts_spec, (ts_spec, None), None]  # None ts_spec allowed
    else:
      ts_specs = ts_spec
    tree_save(data, path, ts_specs=ts_specs)
    load_data = tree_load(path, ts_specs=ts_specs)
    self.assertPyTreeEqual(data, load_data)
    with self.assertRaisesRegex(ValueError,
                                'NOT_FOUND: Error opening "zarr3" driver:'):
      _ = tree_load(path)  # default attempts to open with zarr3 and fails

  def test_save_load_future_printable(self):
    path = self.create_tempdir()
    data = [jnp.ones(())]
    save_fut = pytree_serialization.nonblocking_save(data, path)
    str(save_fut)
    save_fut.result()
    load_fut = pytree_serialization.nonblocking_load(
        path, shardings=_DEFAULT_SHARDING)
    str(load_fut)
    load_fut.result()

  def test_format_alone_not_supported(self):
    # passing a format for a dtype not matching the dtype on disk will cause an
    # XLA error (since formats can be dtype/bit-width specific), hence allow
    # format only if dtype is also specified
    path = self.create_tempdir()
    data = jnp.arange(16 * 16, dtype=jnp.bfloat16).reshape((16, 16))
    sharding = NamedSharding(jtu.create_mesh((1, 1), ('x', 'y')), P('x', None))
    data: jax.Array = jax.device_put(data, sharding)
    tree_save(data, path)
    with self.assertRaisesRegex(NotImplementedError,
                                'Deserialization with `Format` instead of'
                                ' `Sharding` is not currently supported.'):
      pytree_serialization.load(path, shardings=data.format)

  def test_formats_support(self):
    path = self.create_tempdir()
    data = jnp.arange(16 * 16, dtype=jnp.float32).reshape((16, 16))
    data_bf16_format = jnp.arange(16 * 16, dtype=jnp.bfloat16).reshape(
        (16, 16)).format
    sharding = NamedSharding(jtu.create_mesh((1, 1), ('x', 'y')), P('x', None))
    data: jax.Array = jax.device_put(data, sharding)
    tree_save(data, path)
    pytree_serialization.load(path, shardings=jax.ShapeDtypeStruct(
        data.shape, jnp.bfloat16, sharding=data_bf16_format))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
