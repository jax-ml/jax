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
"""Utilities for synchronizing and communication across multiple hosts."""

import functools
import itertools as it
from typing import Optional
import zlib

from typing import Any
import jax
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from jax._src import dispatch
from jax._src import array
from jax._src import sharding
from jax.tree_util import PyTreeDef
from jax._src.interpreters import pxla
from jax.interpreters import xla
from jax.experimental import pjit as pjit_lib
from jax.experimental.pjit import pjit, FROM_GDA
from jax.sharding import PartitionSpec as P
from jax.experimental.global_device_array import GlobalDeviceArray
from jax._src import distributed
from jax._src import config as config_internal
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


# Identity function is at the top level so that `process_allgather` doesn't
# recompile on every invocation.
def _identity_fn(x):
  return x


def _handle_array_process_allgather(inp, tiled):
  if isinstance(inp, array.ArrayImpl) and not inp.is_fully_addressable:
    reps = sharding.OpShardingSharding(inp.sharding._device_assignment,
                                       sharding._get_replicated_op_sharding())
    out = pjit(_identity_fn, out_axis_resources=reps)(inp)
  else:
    # All inputs here will be fully addressable.
    if jax.process_count() == 1:
      return np.asarray(inp)

    devices = np.array(jax.devices()).reshape(jax.process_count(),
                                              jax.local_device_count())
    global_mesh = jax.sharding.Mesh(devices, ('processes', 'local_devices'))
    pspec = P('processes')
    s = jax.sharding.NamedSharding(global_mesh, pspec)

    host_np_arr = np.asarray(inp)
    if host_np_arr.ndim == 0 or not tiled:
      host_np_arr = np.expand_dims(host_np_arr, axis=0)

    aval = jax.core.ShapedArray(host_np_arr.shape, host_np_arr.dtype)
    global_aval = global_mesh._local_to_global(
        pxla.get_array_mapping(pspec), aval)

    bufs = [jax.device_put(host_np_arr, d) for d in jax.local_devices()]
    global_arr = array.make_array_from_single_device_arrays(
        global_aval.shape, s, bufs)
    with global_mesh:
      out = pjit(_identity_fn, out_axis_resources=None)(global_arr)

  return np.asarray(out.addressable_data(0))


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
    if jax.config.jax_array:
      return _handle_array_process_allgather(inp, tiled)
    else:
      if isinstance(inp, GlobalDeviceArray):
        if inp.is_fully_replicated:
          return np.asarray(inp.addressable_data(0))
        global_mesh = inp.mesh
        in_axis_resources = FROM_GDA
      else:
        # DA/SDA/np.array will be sharded based on global_mesh.local_mesh.
        # Shape of local_mesh will always be (1, local_device_count())
        devices = np.array(jax.devices()).reshape(jax.process_count(),
                                                  jax.local_device_count())
        global_mesh = jax.sharding.Mesh(devices, ('processes', 'local_devices'))
        in_axis_resources = P('processes')
        if inp.ndim == 0 or not tiled:
          inp = np.expand_dims(inp, axis=0)

      with global_mesh:
        out = pjit(_identity_fn, in_axis_resources=in_axis_resources,
                   out_axis_resources=None)(inp)
      return np.asarray(out.addressable_data(0))

  if jax.config.jax_array:
    return jax.tree_map(_pjit, in_tree)  # array route
  else:
    with config_internal.parallel_functions_output_gda(True):
      return jax.tree_map(_pjit, in_tree)  # gda route


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


@functools.lru_cache()
def _flatten_pspecs(name, in_tree, pspecs_thunk):
  return pjit_lib.flatten_axis_resources(
      name, in_tree, pspecs_thunk(), tupled_args=True)

@functools.lru_cache()
def _local_to_global_aval(local_aval, mesh, pspec):
  return mesh._local_to_global(pxla.get_array_mapping(pspec), local_aval)

@functools.lru_cache()
def _global_to_local_aval(global_aval, mesh, pspec):
  return mesh._global_to_local(
      pxla.get_array_mapping(pspec), global_aval)

def _device_put(x, device):
  try:
    return dispatch.device_put_handlers[type(x)](x, device)
  except KeyError as err:
    raise TypeError(f"No device_put handler for type: {type(x)}") from err


def host_local_array_to_global_array(local_inputs: Any,
                                     global_mesh: jax.sharding.Mesh,
                                     pspecs: Any):
  """Converts a host local value to a globally sharded `jax.Array`.

  You can use this function to transition to `jax.Array`. Using `jax.Array` with
  `pjit` has the same semantics of using GDA with pjit i.e. all `jax.Array`
  inputs to pjit should be globally shaped.

  If you are currently passing host local values to pjit, you can use this
  function to convert your host local values to global Arrays and then pass that
  to pjit.

  Example usage:

  ```
  from jax.experimental import multihost_utils

  global_inputs = multihost_utils.host_local_array_to_global_array(
    host_local_inputs, global_mesh, in_pspecs)

  with mesh:
    global_out = pjitted_fun(global_inputs)

  host_local_output = multihost_utils.global_array_to_host_local_array(
    global_out, mesh, out_pspecs)
  ```

  Args:
    local_inputs: A Pytree of host local values.
    global_mesh: A ``jax.sharding.Mesh`` object.
    pspecs: A Pytree of ``jax.sharding.PartitionSpec``s.

  Raises:
    RuntimeError: If `jax.config.jax_array` not previously enabled.
  """
  if not jax.config.jax_array:
    raise RuntimeError(
        "Please enable `jax_array` to use `host_local_array_to_global_array`. "
        "You can use jax.config.update('jax_array', True) or set the "
        "environment variable  JAX_ARRAY=1 , or set the `jax_array` boolean "
        "flag to something true-like.")

  def _convert(arr, pspec):
    # If the Array is not fully addressable i.e. not host local, return it.
    if isinstance(arr, array.ArrayImpl) and not arr.is_fully_addressable:
      return arr
    if isinstance(arr, array.ArrayImpl) and isinstance(
        arr.sharding, jax.sharding.PmapSharding):
      arr = np.array(arr)

    local_sharding = jax.sharding.NamedSharding(global_mesh.local_mesh, pspec)

    # If the input is a concrete jax.Array and the input array sharding
    # matches the `local_sharding`, then there's no need to reshard and create
    # copies.
    if (isinstance(arr, array.ArrayImpl) and
        arr.sharding._device_assignment == local_sharding._device_assignment and
        pxla.are_op_shardings_equal(
            arr.sharding._to_xla_op_sharding(arr.ndim),
            local_sharding._to_xla_op_sharding(arr.ndim))):
      arrays = arr._arrays
    else:
      arr = xla.canonicalize_dtype(arr)
      arrays = list(it.chain.from_iterable(
          _device_put(arr[index], d)
          for d, index in local_sharding.devices_indices_map(arr.shape).items()
      ))

    global_aval = _local_to_global_aval(
        jax.core.ShapedArray(arr.shape, arrays[0].dtype), global_mesh, pspec)

    return array.ArrayImpl(
        global_aval, jax.sharding.NamedSharding(global_mesh, pspec),
        arrays, committed=True, _skip_checks=True)

  flattened_inps, in_tree = tree_flatten(local_inputs)
  in_pspecs = _flatten_pspecs('input pspecs', in_tree,
                              pjit_lib.hashable_pytree(pspecs))
  out = tree_map(_convert, tuple(flattened_inps), in_pspecs)
  return tree_unflatten(in_tree, out)


def global_array_to_host_local_array(global_inputs: Any,
                                     global_mesh: jax.sharding.Mesh,
                                     pspecs: Any):
  """Converts a global `jax.Array` to a host local `jax.Array`.

  You can use this function to transition to `jax.Array`. Using `jax.Array` with
  `pjit` has the same semantics of using GDA with pjit i.e. all `jax.Array`
  inputs to pjit should be globally shaped and the output from `pjit` will also
  be globally shaped `jax.Array`s

  You can use this function to convert the globally shaped `jax.Array` output
  from pjit to host local values again so that the transition to jax.Array can
  be a mechanical change.

  Example usage:

  ```
  from jax.experimental import multihost_utils

  global_inputs = multihost_utils.host_local_array_to_global_array(
    host_local_inputs, global_mesh, in_pspecs)

  with mesh:
    global_out = pjitted_fun(global_inputs)

  host_local_output = multihost_utils.global_array_to_host_local_array(
    global_out, mesh, out_pspecs)
  ```

  Args:
    global_inputs: A Pytree of global `jax.Array`s.
    global_mesh: A ``jax.sharding.Mesh`` object.
    pspecs: A Pytree of ``jax.sharding.PartitionSpec``s.

  Raises:
    RuntimeError: If `jax.config.jax_array` not previously enabled.
  """
  if not jax.config.jax_array:
    raise RuntimeError(
        "Please enable `jax_array` to use `global_array_to_host_local_array`. "
        "You can use jax.config.update('jax_array', True) or set the "
        "environment variable  JAX_ARRAY=1 , or set the `jax_array` boolean "
        "flag to something true-like.")

  def _convert(arr, pspec):
    # If the Array is already fully addressable i.e. host local, return it.
    if isinstance(arr, array.ArrayImpl) and arr.is_fully_addressable:
      return arr
    local_aval = _global_to_local_aval(arr.aval, global_mesh, pspec)
    return array.ArrayImpl(
        local_aval, jax.sharding.NamedSharding(global_mesh.local_mesh, pspec),
        arr._arrays, committed=True)

  flattened_inps, out_tree = tree_flatten(global_inputs)
  out_pspecs = _flatten_pspecs('output pspecs', out_tree,
                               pjit_lib.hashable_pytree(pspecs))
  out = tree_map(_convert, tuple(flattened_inps), out_pspecs)
  return tree_unflatten(out_tree, out)
