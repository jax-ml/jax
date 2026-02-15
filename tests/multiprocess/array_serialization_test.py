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

import asyncio
import functools
import math
import os
import pathlib
import shutil
import time

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from etils.epath import Path
import jax
from jax import numpy as jnp
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import pytree_serialization
from jax.experimental.array_serialization import serialization
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

# pylint: enable=g-importing-member
# pylint: enable=g-multiple-import

_SERIALIZE_DESERIALIZE_REPEATS = 20


def create_array(global_shape, global_mesh, mesh_axes, global_data=None):
  if global_data is None:
    global_data = np.arange(
        math.prod(global_shape), dtype=np.float32).reshape(global_shape)

  return jax.make_array_from_callback(
      global_shape, NamedSharding(global_mesh, mesh_axes),
      lambda idx: global_data[idx], dtype=global_data.dtype)


class ArraySerialization(jt_multiprocess.MultiProcessTest):
  def _on_commit_callback(self, temp_dir, final_dir):
    if temp_dir == final_dir:
      temp_file = self.create_tempfile(
          os.path.join(final_dir, "commit_success.txt"))
      with temp_file.open_text("w") as f:
        f.write(f"Checkpoint commit was successful to {final_dir}")
    else:
      logging.info("Renaming %s to %s", temp_dir, final_dir)
      if os.path.exists(final_dir):
        raise ValueError(f"Final dir {final_dir} already exists.")
      os.rename(temp_dir, final_dir)
      logging.info("Finished saving checkpoint to `%s`.", final_dir)

  def get_tempdir(self, name, create=True, process_local=False):
    """Get temporary directory.

    Args:
      name: name of directory.
      create: make a new temp directory.
      process_local: access a directory specific to its own jax.process_index()
    Returns:
      temp directory path
    """
    dir_name = f"{name}_{jax.process_index()}" if process_local else name
    path = os.path.join(absltest.get_default_test_tmpdir(), dir_name)
    assert not Path(path).exists() or not list(Path(path).iterdir())

    multihost_utils.sync_global_devices("get_tempdir")
    # Now it's safe to create the directory again, if needed.
    if create:
      os.makedirs(path, exist_ok=True)
      self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
      multihost_utils.sync_global_devices("create_tempdir")

    return path

  # self.create_tempdir deletes the directory before re-creating it again which
  # is a recipe for race conditions when used without proper attention.
  def create_tempdir(self, *args, **kwargs):
    raise AssertionError(
        "DO NOT USE self.create_tempdir IN THIS TEST. Use self.get_tempdir "
        "instead.")

  def test_async_checkpointing(self):
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    global_input_shape = (8, 2)
    mesh_axes = P("x", "y")
    # race conditions trigger only on repeated runs
    for i in range(_SERIALIZE_DESERIALIZE_REPEATS):
      array1 = create_array(global_input_shape, global_mesh, mesh_axes)

      temp_ckpt_dir = pathlib.Path(self.get_tempdir(f"temp_first_{i}",
                                                    process_local=False))
      ckpt_dir = self.get_tempdir(f"first_{i}", create=False,
                                  process_local=False)

      s_tspecs = jax.tree.map(serialization.get_tensorstore_spec,
                              [str(temp_ckpt_dir)])

      manager = serialization.GlobalAsyncCheckpointManager(timeout_secs=10)
      manager.serialize([array1],
                        s_tspecs,
                        on_commit_callback=functools.partial(
                            self._on_commit_callback, temp_ckpt_dir, ckpt_dir))
      manager.wait_until_finished()
      d_tspecs = jax.tree.map(serialization.get_tensorstore_spec,
                              [str(ckpt_dir)])
      (m1,) = manager.deserialize([NamedSharding(global_mesh, mesh_axes)],
                                  d_tspecs)
      self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
      self.assertEqual(m1.dtype, np.float32)
      for shard_expected, shard_restored in zip(
          array1.addressable_shards, m1.addressable_shards
      ):
        np.testing.assert_array_equal(shard_expected.data, shard_restored.data)
    multihost_utils.sync_global_devices("end_of_looping")

  def test_async_checkpointing_same_final_temp_ckpt_dir(self):
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    global_input_shape = (8, 2)
    mesh_axes = P("x", "y")
    def test(i):
      array1 = create_array(global_input_shape, global_mesh, mesh_axes)
      ckpt_dir = self.get_tempdir(f"same_final_temp_checkpoint_dir_{i}")

      s_tspecs = jax.tree.map(serialization.get_tensorstore_spec,
                              [str(ckpt_dir)])

      manager = serialization.GlobalAsyncCheckpointManager(timeout_secs=5)
      manager.serialize([array1],
                        s_tspecs,
                        on_commit_callback=functools.partial(
                            self._on_commit_callback, ckpt_dir, ckpt_dir))
      manager.wait_until_finished()

      assert (pathlib.Path(ckpt_dir) / "commit_success.txt").exists()

      (m1,) = manager.deserialize([
          NamedSharding(global_mesh, mesh_axes)], s_tspecs)
      self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
      self.assertEqual(m1.dtype, np.float32)
      for shard_expected, shard_restored in zip(
          array1.addressable_shards, m1.addressable_shards
      ):
        np.testing.assert_array_equal(shard_expected.data, shard_restored.data)

    # back-to-back saving test to catch process filesystem race conditions
    for i in range(_SERIALIZE_DESERIALIZE_REPEATS):
      test(i)

  def test_async_checkpointing_error(self):
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    global_input_shape = (8, 2)
    mesh_axes = P("x", "y")
    array1 = create_array(global_input_shape, global_mesh, mesh_axes)

    temp_ckpt_dir = pathlib.Path(self.get_tempdir("temp_second",
                                                  process_local=True))
    ckpt_dir = pathlib.Path(self.get_tempdir("second", process_local=True))

    manager = serialization.GlobalAsyncCheckpointManager(timeout_secs=5)
    s_tspecs = jax.tree.map(serialization.get_tensorstore_spec,
                            [str(temp_ckpt_dir)])
    manager.serialize([array1],
                      s_tspecs,
                      on_commit_callback=functools.partial(
                          self._on_commit_callback, temp_ckpt_dir, ckpt_dir))
    # All processes won't raise the same error.
    with self.assertRaises(Exception):
      manager.wait_until_finished()

  @parameterized.product(driver=["zarr", "zarr3"])
  def test_async_checkpointing_metric(self, driver):
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    global_input_shape = (8, 2)
    mesh_axes = P("x", "y")
    array1 = create_array(global_input_shape, global_mesh, mesh_axes)
    temp_ckpt_dir = pathlib.Path(self.get_tempdir("temp_first",
                                                  process_local=True))
    ckpt_dir = self.get_tempdir(f"first_async_checkpoint_driver={driver}",
                                create=False, process_local=True)
    s_tspecs = jax.tree.map(functools.partial(
        serialization.get_tensorstore_spec, driver=driver),
                            [str(temp_ckpt_dir)])
    durations = {}  # Map metric name to time duration.
    def append_metric_duration(metric, duration, **kwargs):
      del kwargs
      if metric not in durations:
        durations[metric] = 0.
      durations[metric] += duration
    jax.monitoring.register_event_duration_secs_listener(append_metric_duration)

    manager = serialization.GlobalAsyncCheckpointManager(timeout_secs=5)
    manager.serialize([array1],
                      s_tspecs,
                      on_commit_callback=functools.partial(
                          self._on_commit_callback, temp_ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    self.assertGreater(
        durations["/jax/checkpoint/write/async/thread_duration_sec"], 0)

  def test_fully_addressable_error(self):
    array1 = jnp.zeros((8, 2))

    temp_ckpt_dir = pathlib.Path(self.get_tempdir("temp_first",
                                                  process_local=True))
    ckpt_dir = self.get_tempdir("first_fully_addressable", create=False,
                                process_local=True)

    s_tspecs = jax.tree.map(serialization.get_tensorstore_spec,
                            [str(temp_ckpt_dir)])

    manager = serialization.GlobalAsyncCheckpointManager(timeout_secs=5)
    with self.assertRaisesRegex(
        ValueError,
        "Passing fully addressable arrays to a multiprocess serialization is "
        "not allowed, as this may lead to a race condition between processes. "
        "Serialization have failed for the array with the path "
        ".*/temp_first.*"):
      manager.serialize([array1],
                        s_tspecs,
                        on_commit_callback=functools.partial(
                            self._on_commit_callback, temp_ckpt_dir, ckpt_dir))

  @parameterized.product(driver=["zarr", "zarr3"])
  def test_async_serialization_local_fs(self, driver):
    """Test serialize and deserialize using independent local storage from indiviual process."""
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    global_input_shape = (8, 2)
    mesh_axes = P("x", "y")

    def _test(i):
      test_array = create_array(global_input_shape, global_mesh, mesh_axes)

      # using a local temp directory for test,
      # this will be different per jax process
      # ckpt_dir = self.get_tempdir(f"chk_pid_{i}", process_local=True)
      ckpt_dir = self.get_tempdir(f"chk_{i}", process_local=True)

      ts_specs = serialization.get_tensorstore_spec(ckpt_dir, ocdbt=False,
                                                    driver=driver)
      asyncio.run(ts_impl.async_serialize(test_array, ts_specs,
                                          primary_host=None))
      multihost_utils.sync_global_devices(f"serialize_{i}")
      restored = asyncio.run(ts_impl.async_deserialize(
          NamedSharding(global_mesh, mesh_axes), ts_specs))
      multihost_utils.sync_global_devices(f"deserialize_{i}")

      self.assertEqual(restored.addressable_shards[0].data.shape, (2, 1))
      self.assertEqual(restored.dtype, np.float32)

      for shard_expected, shard_restored in zip(
          test_array.addressable_shards, restored.addressable_shards
      ):
        np.testing.assert_array_equal(shard_expected.data, shard_restored.data)
      multihost_utils.sync_global_devices("loop_end")

    # back-to-back saving test to catch process filesystem race conditions
    for i in range(_SERIALIZE_DESERIALIZE_REPEATS):
      _test(i)
    multihost_utils.sync_global_devices("end_of_looping")

  @parameterized.product(driver=["zarr", "zarr3"])
  def test_async_serialization_local_fs_with_replica(self, driver):
    """Test serialize and deserialize using independent local storage from indiviual process for a Jax array with replicas."""
    global_mesh = jtu.create_mesh((1, 8), ("x", "y"))
    global_input_shape = (8,)
    mesh_axes = P("x", None)
    test_array = create_array(global_input_shape, global_mesh, mesh_axes)

    # using a local temp directory for test,
    # this will be different for each jax process
    ckpt_dir = self.get_tempdir("chk_pid", process_local=False)

    ts_specs = serialization.get_tensorstore_spec(ckpt_dir, driver=driver)

    # for each process, we use the replica_id from its first replica
    replica_id = test_array.addressable_shards[0].replica_id

    asyncio.run(ts_impl.async_serialize(test_array, ts_specs, primary_host=None,
                                        replica_id=replica_id))
    restored = asyncio.run(ts_impl.async_deserialize(
        NamedSharding(global_mesh, mesh_axes), ts_specs))

    self.assertEqual(restored.addressable_shards[0].data.shape, (8,))
    self.assertEqual(restored.dtype, np.float32)

    for shard_expected, shard_restored in zip(
        test_array.addressable_shards, restored.addressable_shards
    ):
      np.testing.assert_array_equal(shard_expected.data, shard_restored.data)

  # DO NOT SUBMIT Revert deletion., do detect remote path instead, see code


class PyTreeArraySerialization(jt_multiprocess.MultiProcessTest):

  def tearDown(self):
    # Make sure everyone is done before we run cleanups.
    multihost_utils.sync_global_devices("end_test")
    super().tearDown()

  def get_tempdir(self, name, create=True, process_local=False,
                  overwrite=False):
    """Get temporary directory.

    Args:
      name: name of directory.
      create: make a new temp directory.
      process_local: access a directory specific to its own jax.process_index()
      overwrite: if True, delete the directory if it exists.
    Returns:
      temp directory path
    """
    dir_name = f"{name}_{jax.process_index()}" if process_local else name
    path = os.path.join(absltest.get_default_test_tmpdir(), dir_name)
    assert not (overwrite and not process_local)
    if overwrite:
      # Make sure the directory does not exist yet.
      shutil.rmtree(path, ignore_errors=True)
    else:
      assert not Path(path).exists() or not list(Path(path).iterdir())
    multihost_utils.sync_global_devices("get_tempdir")
    # Now it's safe to create the directory again, if needed.
    if create:
      os.makedirs(path, exist_ok=True)
      self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
      multihost_utils.sync_global_devices("create_tempdir")

    return path

  # self.create_tempdir deletes the directory before re-creating it again which
  # is a recipe for race conditions when used without proper attention.
  def create_tempdir(self, *args, **kwargs):
    raise AssertionError(
        "DO NOT USE self.create_tempdir IN THIS TEST. Use self.get_tempdir "
        "instead.")

  def test_sync_checkpointing(self):
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    global_input_shape = (8, 2)
    mesh_axes = P("x", "y")
    array1 = create_array(global_input_shape, global_mesh, mesh_axes)

    temp_ckpt_dir = pathlib.Path(self.get_tempdir("temp_first"))

    pytree_serialization.save(array1, temp_ckpt_dir)
    m1 = pytree_serialization.load(
        temp_ckpt_dir, NamedSharding(global_mesh, mesh_axes))
    self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
    self.assertEqual(m1.dtype, np.float32)

  def test_submesh_checkpointing(self):
    device_num = jax.device_count()
    submesh1 = jax.make_mesh((device_num // 2, 1), P("x", "y"),
                             devices=jax.devices()[:device_num // 2])
    submesh2 = jax.make_mesh((device_num // 2, 1), P("x", "y"),
                             devices=jax.devices()[device_num // 2:])
    global_input_shape = (8, 2)
    global_data = np.arange(math.prod(global_input_shape),
                            dtype=np.float32).reshape(global_input_shape)
    mesh_axes = P("x", "y")
    array1 = create_array(global_input_shape, submesh1, mesh_axes,
                          global_data=global_data)
    array_ref = create_array(global_input_shape, submesh2, mesh_axes,
                             global_data=global_data)

    temp_ckpt_dir = pathlib.Path(self.get_tempdir("temp_submesh"))

    pytree_serialization.save(array1, temp_ckpt_dir)
    m1 = pytree_serialization.load(
        temp_ckpt_dir, NamedSharding(submesh2, mesh_axes))
    if m1.addressable_shards:
      self.assertEqual(m1.addressable_shards[0].data.shape,
                       (8 // (device_num // 2), 2))
    self.assertEqual(m1.dtype, np.float32)
    for i, shard in enumerate(m1.addressable_shards):
      np.testing.assert_array_equal(array_ref.addressable_shards[i].data,
                                    shard.data)

  def test_fully_addressable_error(self):
    array1 = jnp.zeros((8, 2))

    ckpt_dir = self.get_tempdir("temp_third", create=False)
    with self.assertRaisesRegex(
        ValueError,
        "Passing fully addressable arrays to a multiprocess serialization is "
        "not allowed, as this may lead to a race condition between processes. "
        "Serialization have failed for the array with the path "
        ".*/temp_third.*"):
      pytree_serialization.save(array1, ckpt_dir)

  def test_sync_serialization_process_local(self):
    """Test serialize and deserialize using independent local storage from indiviual process."""

    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    global_input_shape = (8, 2)
    mesh_axes = P("x", "y")
    test_array = create_array(global_input_shape, global_mesh, mesh_axes)

    # using a local temp directory for test, this will be forced to be the
    # same for each jax process because of the synchronization of UUID in the
    # pytree of the data structure
    ckpt_dir = self.get_tempdir("chk_sync", process_local=True)

    msg = ("Saving to different locations on different hosts is not supported, "
           "because it is extremely fragile. Consider using a single location.")
    with self.assertRaisesRegex(ValueError, msg):
      pytree_serialization.save(test_array, ckpt_dir)

  def test_nonblocking_serialization_process_local(self):
    """Test serialize and deserialize using independent local storage from indiviual process."""

    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    global_input_shape = (8, 2)
    mesh_axes = P("x", "y")
    test_array = create_array(global_input_shape, global_mesh, mesh_axes)

    # using a local temp directory for test, this will be forced to be the
    # same for each jax process because of the synchronization of UUID in the
    # pytree of the data structure
    ckpt_dir = self.get_tempdir("chk_sync", process_local=True)

    fut = pytree_serialization.nonblocking_save(test_array, ckpt_dir)
    self.assertEqual(fut.pytree,
                     jax.ShapeDtypeStruct(shape=(8, 2), dtype=np.float32))
    while not fut.done():
      time.sleep(1e-5)
    msg = ("Saving to different locations on different hosts is not supported, "
           "because it is extremely fragile. Consider using a single location.")
    with self.assertRaisesRegex(ValueError, msg):
      fut.result()

  def test_nonblocking_checkpointing(self):
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    global_input_shape = (8, 2)
    mesh_axes = P("x", "y")
    # back-to-back saving test to catch process filesystem race conditions
    for i in range(_SERIALIZE_DESERIALIZE_REPEATS):
      array1 = create_array(global_input_shape, global_mesh, mesh_axes)

      temp_ckpt_dir = pathlib.Path(self.get_tempdir(f"temp_second_{i}"))

      fut = pytree_serialization.nonblocking_save(array1, temp_ckpt_dir)
      self.assertEqual(fut.pytree,
                       jax.ShapeDtypeStruct(shape=(8, 2), dtype=np.float32))
      while not fut.done():
        time.sleep(1e-5)
      fut = pytree_serialization.nonblocking_load(
          temp_ckpt_dir,
          shardings=NamedSharding(global_mesh, mesh_axes))
      self.assertEqual(fut.pytree,
                       jax.ShapeDtypeStruct(shape=(8, 2), dtype=np.float32))
      while not fut.done():
        time.sleep(1e-5)
      m1 = fut.result()
      self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
      self.assertEqual(m1.dtype, np.float32)

      for shard_expected, shard_restored in zip(
          array1.addressable_shards, m1.addressable_shards
      ):
        np.testing.assert_array_equal(shard_expected.data, shard_restored.data)

  @parameterized.product(nonblocking_save=[True, False],
                         nonblocking_load=[True, False])
  def test_async_await_on_future(self, nonblocking_save, nonblocking_load):
    """Test the async/await API and saving and loading in async context."""

    async def _async_fn(i):
      global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
      global_input_shape = (8, 2)
      mesh_axes = P("x", "y")
      test_array = create_array(global_input_shape, global_mesh, mesh_axes)

      ckpt_dir = Path(self.get_tempdir(f"temp_last_{i}"))
      if nonblocking_save:
        await pytree_serialization.nonblocking_save(test_array, ckpt_dir,
                                                    overwrite=True)
      else:
        pytree_serialization.save(test_array, ckpt_dir, overwrite=True)
      if nonblocking_load:
        restored = await pytree_serialization.nonblocking_load(
            ckpt_dir, shardings=NamedSharding(global_mesh, mesh_axes))
      else:
        restored = pytree_serialization.load(
            ckpt_dir, shardings=NamedSharding(global_mesh, mesh_axes))
      self.assertEqual(restored.addressable_shards[0].data.shape, (2, 1))
      self.assertEqual(restored.dtype, np.float32)

      for shard_expected, shard_restored in zip(
          test_array.addressable_shards, restored.addressable_shards
      ):
        np.testing.assert_array_equal(shard_expected.data, shard_restored.data)

    # back-to-back saving test to catch process filesystem race conditions
    for i in range(_SERIALIZE_DESERIALIZE_REPEATS):
      asyncio.run(_async_fn(i))


if __name__ == "__main__":
  jt_multiprocess.main()
