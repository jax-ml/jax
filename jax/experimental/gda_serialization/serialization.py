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

import abc
import asyncio
from functools import partial
import re
import threading
from typing import Callable, Sequence, Optional
from absl import logging

import jax
from jax._src import distributed
from jax._src.util import prod
from jax.experimental import global_device_array as gda
from jax.experimental.maps import Mesh
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from etils import epath


TS_CONTEXT = ts.Context({'file_io_concurrency': {'limit': 128}})
_REMOVED_VALUE = 'Value removed'
_CHECKPOINT_SUCCESS = 'checkpoint_write_success'


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


# Lifted from T5X.
class _LimitInFlightBytes:
  """Limits in-flight bytes when reading/writing checkpoints per process."""

  def __init__(self, num_bytes):
    self._max_bytes = num_bytes
    self._available_bytes = num_bytes
    self._cv = asyncio.Condition(lock=asyncio.Lock())

  async def wait_for_bytes(self, requested_bytes):
    async with self._cv:
      await self._cv.wait_for(lambda: self._available_bytes > requested_bytes)
      self._available_bytes -= requested_bytes
      assert self._available_bytes >= 0

  async def release_bytes(self, requested_bytes):
    async with self._cv:
      self._available_bytes += requested_bytes
      assert self._available_bytes <= self._max_bytes
      self._cv.notify_all()


async def async_serialize(gda_inp: gda.GlobalDeviceArray, tensorstore_spec,
                          commit_future=None):
  # 'metadata' may not be present at the top level (for example, if we are using
  # a 'cast' driver).
  if not _spec_has_metadata(tensorstore_spec):
    tensorstore_spec['metadata'] = _get_metadata(gda_inp)

  if jax.process_index() == 0:
    open_future = ts.open(
        ts.Spec(tensorstore_spec), create=True, open=True, context=TS_CONTEXT)
    # Asynchronous case.
    if commit_future is not None:
      assert isinstance(commit_future, list)
      commit_future.append(open_future)
    else:
      await open_future

  # `ts.open` runs twice for process 0 because for the first time, we just get
  # the future to be awaited upon in the background thread. The second one runs
  # with `assume_metadata=True` which does no I/O operation and returns the
  # tensorstore object.
  # For every process other than `0`, we open with `assume_metadata=True`.
  t = await ts.open(
      ts.Spec(tensorstore_spec), open=True, assume_metadata=True, context=TS_CONTEXT)

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
    future_writer = jax.tree_util.tree_map(async_serialize, gdas, tensorstore_specs)
    return await asyncio.gather(*future_writer)
  asyncio.run(_run_serializer())


def estimate_read_memory_footprint(t: ts.TensorStore) -> int:
  rank = t.rank
  num_bytes = t.dtype.numpy_dtype.itemsize
  chunk_template = t.chunk_layout.read_chunk_template
  origin = t.domain.origin
  shape = t.domain.shape
  chunk_origin = chunk_template.origin
  chunk_shape = chunk_template.shape

  for i in range(rank):
    origin_value = origin[i]
    chunk_origin_value = chunk_origin[i]
    chunk_size = chunk_shape[i]
    lower = origin_value - chunk_origin_value
    upper = origin_value + shape[i] - chunk_origin_value
    lower_aligned = lower // chunk_size * chunk_size
    upper_aligned = -(-upper // chunk_size) * chunk_size
    num_bytes *= (upper_aligned - lower_aligned)
  return num_bytes


async def async_deserialize(mesh, mesh_axes, tensorstore_spec,
                            global_shape=None, dtype=None,
                            byte_limiter: Optional[_LimitInFlightBytes] = None):
  t = await ts.open(ts.Spec(tensorstore_spec), open=True, context=TS_CONTEXT)
  shape = t.shape if global_shape is None else global_shape
  new_shard_shape = gda.get_shard_shape(tuple(shape), mesh, mesh_axes)

  async def cb(index):
    # This maybe needed because the shape the array was saved with is smaller
    # than the requested shape of the array in which it will be reloaded. So
    # the extra values will be filled with 0s.
    out = np.zeros(new_shard_shape, dtype=t.dtype.numpy_dtype)
    requested_domain = ts.IndexTransform(input_shape=shape)[index].domain
    restricted_domain = t.domain.intersect(requested_domain)

    requested_bytes = estimate_read_memory_footprint(t[restricted_domain])

    # Limit the bytes read for every shard.
    if byte_limiter is not None:
      await byte_limiter.wait_for_bytes(requested_bytes)

    await ts.array(out)[ts.d[:].translate_to[requested_domain.origin]][restricted_domain].write(
        t[restricted_domain])

    if dtype is not None:
      # Cast while reloading on process to avoid 2 copies on device if the
      # casting is done on device.
      return out.astype(dtype)

    if byte_limiter is not None:
      await byte_limiter.release_bytes(requested_bytes)
    return out

  return await create_async_gda_from_callback(tuple(shape), mesh, mesh_axes, cb)


def run_deserialization(global_meshes, mesh_axes, tensorstore_specs,
                        global_shapes=None, dtypes=None, concurrent_gb=32):
  concurrent_bytes = concurrent_gb * 10**9

  async def _run_deserializer():
    # Object should be created once per process.
    byte_limiter = _LimitInFlightBytes(concurrent_bytes)

    future_gdas = jax.tree_util.tree_map(
        partial(async_deserialize, byte_limiter=byte_limiter),
        global_meshes, mesh_axes, tensorstore_specs,
        [None] * len(tensorstore_specs) if global_shapes is None else global_shapes,
        [None] * len(tensorstore_specs) if dtypes is None else dtypes)
    return await asyncio.gather(*future_gdas)
  return asyncio.run(_run_deserializer())


def _get_key(key: str):
  return f'checkpoint_{key}'


class GlobalAsyncCheckpointManagerBase(metaclass=abc.ABCMeta):
  """Interface for checkpointing GDAs asynchronously.

  This class manages the state of an ongoing asynchronous checkpoint.

  For example, say a checkpoint happens on every step. If you checkpoint on
  step 1 and after some computation the model is on checkpoint 2. But step 1's
  checkpoint hasn't finished committing to the storage layer yet. So until that
  is finished, checkpoint for step 2 will need to be blocked. Maintaining a
  class allows to maintain that state.

  Example:

  Below is a simplified training loop:

  ```
  # Call this at the start of your program.
  jax.distributed.initialize()

  manager = GlobalAsyncCheckpointManager()

  # Restore checkpoint if available or initialize the train_state from
  # init_fn().
  train_state = manager.deserialize(...)

  while ...:
    if step % num_steps_between_checkpoints == 0:
      manager.serialize(train_state, temp_checkpoint_dir=...,
                        final_checkpoint_dir=...)
      train_state = train_step(train_state, input)
      # This is a non-blocking call.
      manager.check_for_errors()

  manager.serialize(train_state, temp_checkpoint_dir=...,
                    final_checkpoint_dir=...)
  # Wait before the end of the program for the checkpoint to finish. This is a
  # blocking call.
  manager.wait_until_finished()
  ```
  """

  @abc.abstractmethod
  def check_for_errors(self):
    """Checks if any errors have been raised in the child thread.

    This is a non-blocking call that can be called in the main thread.
    """

  @abc.abstractmethod
  def wait_until_finished(self):
    """Blocks until serialization has finished."""

  @abc.abstractmethod
  # TODO(b/233793426): Try removing temp_checkpoint_dir and final_checkpoint_dir
  # from the API and use a callback instead. This will affect how async
  # mechanism works.
  def serialize(self, gdas, tensorstore_specs, *, temp_checkpoint_dir,
                final_checkpoint_dir):
    """Serializes GDAs to TensorStore."""

  @abc.abstractmethod
  def deserialize(self, global_meshes, mesh_axes, tensorstore_specs,
                  global_shapes=None, dtypes=None):
    """Deserializes GDAs from TensorStore."""


class AsyncManager:

  def __init__(self, timeout_secs=300):
    self._timeout_secs = timeout_secs
    self._timeout_in_ms = self._timeout_secs * 1000

    self._commit_futures = None
    self._thread = None
    self._exception = None

    if distributed.global_state.client is None:
      raise ValueError('Please initialize the distributed system via '
                       '`jax.distributed.initialize()` at the start of your '
                       'program.')
    self._client = distributed.global_state.client
    self._final_checkpoint_dir = None

  def __del__(self):
    if self._thread is not None and self._thread.is_alive():
      logging.warning('Please add `.wait_until_finished()` in the main thread '
                      'before your program finishes because there is a '
                      'possibility of losing errors raised if the '
                      'this class is deleted before writing is completed.')

  def _thread_func(self, temp_checkpoint_dir, final_checkpoint_dir):
    try:
      current_process = jax.process_index()
      logging.info('Starting commit to storage layer by process: %s',
                   current_process)
      for future in self._commit_futures:
        future.result()
      logging.info('Finished committing to storage layer by process: %s',
                   current_process)

      # All processes will wait at the barrier. When all processes are at the
      # barrier, the barrier will be satisfied. If not, then it will timeout.
      self._client.wait_at_barrier(self._final_checkpoint_dir,
                                   self._timeout_in_ms)
      logging.info('Finished waiting at barrier for process %s',
                   current_process)

      if current_process == 0:
        logging.info('Renaming %s to %s', temp_checkpoint_dir,
                     final_checkpoint_dir)
        epath.Path(temp_checkpoint_dir).rename(final_checkpoint_dir)
        logging.info('Finished saving checkpoint to `%s`.',
                     final_checkpoint_dir)
        self._client.key_value_set(
            _get_key(self._final_checkpoint_dir), _CHECKPOINT_SUCCESS)
    except Exception as e:
      self._exception = e

  def _start_async_commit(self, temp_checkpoint_dir, final_checkpoint_dir):
    self._final_checkpoint_dir = final_checkpoint_dir
    self._thread = threading.Thread(
        target=self._thread_func,
        args=(temp_checkpoint_dir, final_checkpoint_dir))
    self._thread.start()

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

    if self._final_checkpoint_dir is not None:
      # Block until process 0 writes success value to the key value store.
      # If it fails to write it, then `blocking_key_value_get` will time out.
      self._client.blocking_key_value_get(
          _get_key(self._final_checkpoint_dir), self._timeout_in_ms)

  def _add_futures(self, futures: Sequence[asyncio.Future]):
    self._commit_futures = futures


class GlobalAsyncCheckpointManager(AsyncManager, GlobalAsyncCheckpointManagerBase):
  """Responsible for serializing GDAs via TensorStore."""

  def serialize(self, gdas, tensorstore_specs, *, temp_checkpoint_dir,
                final_checkpoint_dir):
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
      temp_checkpoint_dir: Temporary checkpoint directory where the checkpoints
        will be written.
      final_checkpoint_dir: Final checkpoint directory where the checkpoints
        will be moved from `temp_checkpoint_dir`.
    """
    logging.info('Waiting for previous serialization to finish.')
    self.wait_until_finished()

    commit_futures = [[] for _ in range(len(tensorstore_specs))]

    async def _run_serializer():
      future_writer = jax.tree_util.tree_map(async_serialize, gdas, tensorstore_specs,
                                   commit_futures)
      return await asyncio.gather(*future_writer)

    asyncio.run(_run_serializer())

    self._add_futures(jax.tree_util.tree_flatten(commit_futures)[0])

    # Used in wait_until_finished to check on process != 0, if the checkpoint
    # has finished writing.
    self._start_async_commit(temp_checkpoint_dir, final_checkpoint_dir)

  def deserialize(self, global_meshes, mesh_axes, tensorstore_specs,
                  global_shapes=None, dtypes=None):
    return run_deserialization(global_meshes, mesh_axes, tensorstore_specs,
                               global_shapes, dtypes)
