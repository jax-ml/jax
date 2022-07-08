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
"""Utilities for synchronizing and communication across multiple hosts."""

import functools
from typing import Optional
import zlib

import jax
from jax.tree_util import PyTreeDef
from jax.experimental import maps
from jax.experimental.pjit import pjit, FROM_GDA
from jax.interpreters.pxla import PartitionSpec as P
from jax.experimental.global_device_array import GlobalDeviceArray
from jax._src import distributed
import numpy as np


# This needs to be top-level for the jax compilation cache.
@functools.partial(jax.pmap, axis_name='hosts')
def _psum(x: PyTreeDef) -> PyTreeDef:
  return jax.lax.psum(x, 'hosts')


def broadcast_one_to_all(in_tree: PyTreeDef,
                         is_source: Optional[bool] = None) -> PyTreeDef:
  """Broadcast data from a source host (host 0 by default) to all other hosts.

  Args:
    in_tree: pytree of arrays - each array *must* have the same shape across the
      hosts.
    is_source: optional bool denoting whether the caller is the source. Only
      'source host' will contribute the data for the broadcast. If None, then
      host 0 is used.

  Returns:
    A pytree matching in_tree where the leaves now all contain the data from the
    first host.
  """
  if is_source is None:
    is_source = jax.process_index() == 0

  def pre_pmap(x):
    if isinstance(x, GlobalDeviceArray):
      raise ValueError('GDAs cannot be broadcasted from source host to other '
                       'hosts.')
    if is_source:
      return np.concatenate([
          x[None, ...],
          np.repeat([np.zeros_like(x)],
                    jax.local_device_count() - 1, 0)
      ])
    else:
      return np.repeat([np.zeros_like(x)], jax.local_device_count(), 0)

  def post_pmap(x):
    return jax.device_get(x)[0]

  in_tree = jax.tree_util.tree_map(pre_pmap, in_tree)
  in_tree = jax.device_get(_psum(in_tree))
  return jax.tree_util.tree_map(post_pmap, in_tree)


def sync_global_devices(name: str):
  """Creates a barrier across all hosts/devices."""
  h = np.int32(zlib.crc32(name.encode()))
  assert_equal(h, f"sync_global_devices name mismatch ('{name}')")


def process_allgather(in_tree: PyTreeDef, tiled: bool = False) -> PyTreeDef:
  """Gather data from across processes.

  Args:
    in_tree: pytree of arrays - each array _must_ have the same shape across the
      hosts.
    tiled: Whether to stack or concat the output. Defaults to False i.e. stack
      into a new positional axis at index 0.
      This does not affect GDA inputs as the GDA output will always be
      concatenated.
      Scalar inputs will always be stacked.

  Returns:
    Pytress of arrays where the data is gathered from all hosts.
      * If the input is a GDA, then the data is fully replicated.
      * If the input is non-GDA, then the output shape is dependent on the
        `tiled` argument. If its False, then the output will be stacked else
        concatenated.
      * If the input is non-GDA and scalar, then the output will be stacked.
  """

  def _pjit(inp):
    if isinstance(inp, GlobalDeviceArray):
      if inp.is_fully_replicated:
        return inp.local_data(0).to_py()
      global_mesh = inp.mesh
      in_axis_resources = FROM_GDA
    else:
      # DA/SDA/np.array will be sharded based on global_mesh.local_mesh.
      # Shape of local_mesh will always be (1, local_device_count())
      devices = np.array(jax.devices()).reshape(jax.process_count(),
                                                jax.local_device_count())
      global_mesh = maps.Mesh(devices, ('processes', 'local_devices'))
      in_axis_resources = P('processes')
      if inp.ndim == 0 or not tiled:
        inp = np.expand_dims(inp, axis=0)

    with maps.Mesh(global_mesh.devices, global_mesh.axis_names):
      out = pjit(lambda x: x, in_axis_resources=in_axis_resources,
                 out_axis_resources=None)(inp)
    return out.local_data(0).to_py()

  with jax._src.config.parallel_functions_output_gda(True):
    return jax.tree_util.tree_map(_pjit, in_tree)


def assert_equal(in_tree, fail_message: str = ''):
  """Verifies that all the hosts have the same tree of values."""
  expected = broadcast_one_to_all(in_tree)
  if not jax.tree_util.tree_all(
      jax.tree_util.tree_map(lambda *x: np.all(np.equal(*x)), in_tree, expected)):
    raise AssertionError(
        f'{fail_message} Expected: {expected}; got: {in_tree}.')


def reached_preemption_sync_point(step_id: int) -> bool:
  """Determine whether all hosts have reached a preemption sync step.

  When any host receive a preemption notice, the notice will be propagated to
  all hosts and trigger a synchronization protocol in background. The
  synchronization protocol calculates the maximum step ids from all hosts, and
  uses the next step id (i.e., max + 1) as the safe step to save a checkpoint.
  All hosts should continue training more steps until this method returns True,
  indicating that the `step_id` is equal to the safe step and the hosts should
  start saving a checkpoint. This feature requires enabling
  `jax.config.jax_coordination_service`.

  To use this API, all hosts must start training from the same step and call at
  every training step. Example usage:

  ```
  def should_save(step_id: int) -> bool:

    # Should save an on-demand checkpoint for preemption
    if multihost_utils.reached_preemption_sync_point(step_id):
      return True

    # Should save a regular checkpoint
    return step_id - last_saved_checkpoint_step >= save_interval_steps
  ```

  Preemption notice is provided by the cluster scheduler to notify the
  application in advance before it gets evicted. By default, we use SIGTERM as
  the signal for preemption notice.

  TODO(b/230630494): Add instructions for customized preemption notice.

  Returns:
    A boolean indicating whether all hosts have reached a synchronization step
    after some hosts are preempted.

  Raises:
    RuntimeError: if preemption sync manager has not been inititialized.
  """
  if distributed.global_state.client is None:
    return False
  sync_manager = distributed.global_state.preemption_sync_manager
  if sync_manager is None:
    raise RuntimeError("Preemption sync manager has not been initialized.")
  return sync_manager.reached_sync_point(step_id)
