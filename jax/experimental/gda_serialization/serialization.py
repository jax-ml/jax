# Copyright 2021 Google LLC
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
"""GlobalDeviceArray serialization and deserialization."""

import asyncio
import os
import re
import threading
import time
from typing import Callable
from absl import logging

import jax
from jax._src.util import prod
from jax.experimental import global_device_array as gda
from jax.experimental.maps import Mesh
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
import tensorflow.compat.v2 as tf
# internal import


TS_CONTEXT = ts.Context({'file_io_concurrency': {'limit': 128}})


async def create_async_gda_from_callback(
    global_shape: gda.Shape,
    global_mesh: Mesh,
    mesh_axes: gda.MeshAxes,
    data_callback: Callable[[gda.Index], asyncio.Future],
):
  global_idx_rid = gda.get_shard_indices_replica_ids(
      global_shape, global_mesh, mesh_axes)
  local_devices = global_mesh.local_devices
  future_arrays = [data_callback(global_idx_rid[d][0])
                   for d in local_devices]
  # Pause here and come back to `from_async_callback()` when future_arrays are
  # ready. device_put cannot happen with future_arrays.
  local_arrays = await asyncio.gather(*future_arrays)

  dbs = [jax.device_put(array, device)
         for array, device in zip(local_arrays, local_devices)]
  return gda.GlobalDeviceArray(global_shape, global_mesh, mesh_axes, dbs,
                               gda._GdaFastPathArgs(global_idx_rid, local_devices))


def _get_metadata(gda):
  if gda.dtype == jnp.bfloat16:
    # Tensorstore uses 'bfloat16', not '<V2'.
    dtype = 'bfloat16'
  else:
    dtype = np.dtype(gda.dtype).str

  return {
      'compressor': {
          'id': 'gzip'
      },
      'shape': gda.shape,
      'chunks': np.array(np.maximum(1, gda.local_data(0).shape)),
      'dtype': dtype,
  }


def _spec_has_metadata(tree):
  if not isinstance(tree, dict):
    return False
  return 'metadata' in tree or any(
      _spec_has_metadata(subtree) for _, subtree in tree.items())


def get_tensorstore_spec(ckpt_path: str):
  spec = {'driver': 'zarr', 'kvstore': {}}

  if ckpt_path.startswith('gs://'):
    m = re.fullmatch('^gs://([^/]*)/(.*)$', ckpt_path, re.DOTALL)
    if m is None:
      raise ValueError('The ckpt_path should contain the bucket name and the '
                       f'file path inside the bucket. Got: {ckpt_path}')
    gcs_bucket = m.group(1)
    path_without_bucket = m.group(2)
    spec['kvstore'] = {'driver': 'gcs', 'bucket': gcs_bucket,
                       'path': path_without_bucket}
  else:
    spec['kvstore'] = {'driver': 'file', 'path': ckpt_path}
  return spec


async def async_serialize(gda_inp: gda.GlobalDeviceArray, tensorstore_spec,
                          commit_future=None):
  # 'metadata' may not be present at the top level (for example, if we are using
  # a 'cast' driver).
  if not _spec_has_metadata(tensorstore_spec):
    tensorstore_spec['metadata'] = _get_metadata(gda_inp)

  t = await ts.open(
      ts.Spec(tensorstore_spec), create=True, open=True, context=TS_CONTEXT)

  async def _write_array(shard):
    if shard.replica_id == 0:
      write_future = t[shard.index].write(shard.data)
      if commit_future is not None:
        assert isinstance(commit_future, list)
        commit_future.append(write_future.commit)
        await write_future.copy
      else:
        await write_future.commit

  future_write_state = jax.tree_util.tree_map(_write_array,
                                              gda_inp.local_shards)
  return await asyncio.gather(*future_write_state)


def run_serialization(gdas, tensorstore_specs):
  async def _run_serializer():
    future_writer = jax.tree_map(async_serialize, gdas, tensorstore_specs)
    return await asyncio.gather(*future_writer)
  asyncio.run(_run_serializer())


async def async_deserialize(mesh, mesh_axes, tensorstore_spec,
                            global_shape=None, dtype=None):
  t = ts.open(ts.Spec(tensorstore_spec), open=True, context=TS_CONTEXT).result()
  shape = t.shape if global_shape is None else global_shape
  requires_padding = prod(shape) > prod(t.shape)

  if requires_padding:
    new_shard_shape = gda.get_shard_shape(shape, mesh, mesh_axes)

  async def cb(index):
    if requires_padding:
      # This is needed because the shape the array was saved with is smaller
      # than the requested shape of the array in which it will be reloaded. So
      # the extra values will be filled with 0s.
      out = np.zeros(new_shard_shape, dtype=t.dtype.numpy_dtype)
      requested_domain = ts.IndexTransform(input_shape=shape)[index].domain
      restricted_domain = t.domain.intersect(requested_domain)
      await ts.array(out)[ts.d[:].translate_to[requested_domain.origin]][restricted_domain].write(t[restricted_domain])
    else:
      out = await t[index].read()

    if dtype is not None:
      # Cast while reloading on process to avoid 2 copies on device if the
      # casting is done on device.
      return out.astype(dtype)
    return out

  return await create_async_gda_from_callback(shape, mesh, mesh_axes, cb)


def run_deserialization(global_meshes, mesh_axes, tensorstore_specs,
                        global_shapes=None, dtypes=None):
  async def _run_deserializer():
    future_gdas = jax.tree_map(
        async_deserialize, global_meshes, mesh_axes, tensorstore_specs,
        [None] * len(tensorstore_specs) if global_shapes is None else global_shapes,
        [None] * len(tensorstore_specs) if dtypes is None else dtypes)
    return await asyncio.gather(*future_gdas)
  return asyncio.run(_run_deserializer())


no_lockfiles_exists = lambda paths: all(not tf.io.gfile.exists(f) for f in paths)


class _RetryWithTimeout:
  def __init__(self, secs):
    self.secs = secs

  def __enter__(self):
    self.timeout_after = time.time() + self.secs
    return self

  def __exit__(self, type, value, traceback):
    pass

  @property
  def timed_out(self):
    return time.time() > self.timeout_after


class GlobalAsyncCheckpointManager:
  """Responsible for serializing GDAs via TensorStore.

  This class manages the state of an ongoing asynchronous checkpoint.

  For example, say a checkpoint happens on every step. If you checkpoint on
  step 1 and after some computation the model is on checkpoint 2. But step 1's
  checkpoint hasn't finished committing to the storage layer yet. So until that
  is finished, checkpoint for step 2 will need to be blocked. Maintaining a
  class allows to maintain that state.

  Example:

  Below is a simplified training loop:

  ```
  manager = GlobalAsyncCheckpointManager()

  # Restore checkpoint if available or initialize the train_state from
  # init_fn().
  train_state = manager.deserialize(...)

  while ...:
    if step % num_steps_between_checkpoints == 0:
      manager.serialize(train_state)
      train_state = train_step(train_state, input)
      # This is a non-blocking call.
      manager.check_for_errors()

  manager.serialize(train_state)
  # Wait before the end of the program for the checkpoint to finish. This is a
  # blocking call.
  manager.wait_until_finished()
  ```
  """

  def __init__(self, temp_checkpoint_dir, final_checkpoint_dir, timeout_secs=300):
    self.temp_checkpoint_dir = temp_checkpoint_dir
    self.final_checkpoint_dir = final_checkpoint_dir
    self._timeout_secs = timeout_secs

    self._commit_futures = None
    self._thread = None
    self._exception = None

  def __del__(self):
    if self._thread is not None and self._thread.is_alive():
      logging.warning('Please add `.wait_until_finished()` in the main thread '
                      'before your program finishes because there is a '
                      'possibility of losing errors raised if the '
                      'GlobalAsyncCheckpointManager is deleted before '
                      'serialization is completed.')

  def _thread_func(self):
    try:
      for future in self._commit_futures:
        for f in future:
          f.result()
      logging.info('Commit to storage layer has completed.')

      current_process = jax.process_index()
      lockfiles_dir = os.path.join(self.temp_checkpoint_dir, 'lockfiles')
      all_lockfile_paths = [
          os.path.join(lockfiles_dir, f'lockfile_{p}')
          for p in range(jax.process_count())
      ]
      current_process_lockfile = os.path.join(lockfiles_dir,
                                              f'lockfile_{current_process}')

      with _RetryWithTimeout(self._timeout_secs) as t:
        while not tf.io.gfile.exists(current_process_lockfile):
          if t.timed_out:
            raise RuntimeError('Terminating after waiting for '
                               f'{self._timeout_secs} secs for lockfile to appear')
          logging.info('Waiting for current process %s lockfile to appear.',
                       current_process)
          time.sleep(60)

      tf.io.gfile.remove(current_process_lockfile)
      logging.info('Lockfile removed for process %s', current_process)

      # This while loop will not trigger until all commits have finished.
      if current_process == 0:
        with _RetryWithTimeout(self._timeout_secs) as t:
          while True:
            if t.timed_out:
              raise RuntimeError('Terminating after waiting for '
                                 f'{self._timeout_secs} secs for '
                                 'finishing the serialization.')
            # Mark as done when no lockfiles exist.
            if no_lockfiles_exists(all_lockfile_paths):
              tf.io.gfile.remove(lockfiles_dir)
              logging.info('Lockfiles directory removed.')

              logging.info('Renaming %s to %s', self.temp_checkpoint_dir, self.final_checkpoint_dir)
              tf.io.gfile.rename(self.temp_checkpoint_dir, self.final_checkpoint_dir)
              logging.info('Finished saving GDA checkpoint to `%s`.', self.final_checkpoint_dir)
              break
            else:
              logging.info('Thread sleeping for 60 seconds.')
              time.sleep(60)

    except Exception as e:
      self._exception = e

  def _start_commit_thread(self):
    self._thread = threading.Thread(target=self._thread_func)
    self._thread.start()

  def _write_lockfiles(self):
    lockfiles_dir = os.path.join(self.temp_checkpoint_dir, 'lockfiles')
    tf.io.gfile.mkdir(lockfiles_dir)

    for p in range(jax.process_count()):
      with tf.io.gfile.GFile(
          os.path.join(lockfiles_dir, f'lockfile_{p}'), mode='w') as f:
        f.write('File to track if all chunks have been written.')

    if len(tf.io.gfile.listdir(lockfiles_dir)) != jax.process_count():
      raise ValueError("Process 0 couldn't write all the lockfiles.")

    logging.info('Lock files for all processes have been written by process 0.')

  def check_for_errors(self):
    if self._exception is not None:
      # Clears self._exception so it is only raised once.
      exception = self._exception
      self._exception = None
      raise exception  # pylint: disable=raising-bad-type

  def wait_until_finished(self):
    if self._thread is not None:
      self._thread.join()
      self._thread = None

    self.check_for_errors()

  def serialize(self, gdas, tensorstore_specs):
    """Serializes GlobalDeviceArrays via TensorStore asynchronously.

    TensorStore writes to a storage layer in 2 steps:
    *  Reading/copying from the source after which the source can be modified.
         * Returns a copy future.
    *  Writing/committing to the storage layer.
         * Returns a commit future.

    In asynchronous mode, the serialization waits for the commit future to
    finish in a separate thread allowing other computation to proceed.

    Args:
      gdas: GlobalDeviceArrays that should be serialized.
      tensorstore_specs: TensorStore specs that are used to serialize GDAs.
    """
    logging.info('Waiting for thread to finish serialization.')
    self.wait_until_finished()

    # Process 0 writes lock files for all processes.
    if jax.process_index() == 0:
      self._write_lockfiles()

    self._commit_futures = [[] for _ in range(len(tensorstore_specs))]

    async def _run_serializer():
      future_writer = jax.tree_map(async_serialize, gdas,
                                   tensorstore_specs, self._commit_futures)
      return await asyncio.gather(*future_writer)
    asyncio.run(_run_serializer())

    self._start_commit_thread()

  def deserialize(self, global_meshes, mesh_axes, tensorstore_specs,
                  global_shapes=None, dtypes=None):
    return run_deserialization(global_meshes, mesh_axes, tensorstore_specs,
                               global_shapes, dtypes)
