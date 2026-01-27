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
from __future__ import annotations

from functools import lru_cache, partial
from typing import Any

from jax._src import api
from jax._src.api_util import (
    argnums_partial, donation_vector, flatten_fun,
    flat_out_axes, fun_signature, fun_sourceinfo)
from jax._src import array
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import prng
from jax._src import sharding_impls
from jax._src import stages
from jax._src import traceback_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.interpreters import pxla
from jax._src.lax import lax
from jax._src.mesh import Mesh
from jax._src.shard_map import _axes_to_pspec, _shard_map
from jax._src.tree_util import (
    broadcast_flattened_prefix_with_treedef, prefix_errors, tree_flatten, tree_map, tree_unflatten)
import numpy as np

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip
traceback_util.register_exclusion(__file__)


# Implementing pmap in terms of shard_map

def pmap(f, axis_name=None, *, in_axes=0, out_axes=0,
         static_broadcasted_argnums=(), devices=None, backend=None,
         axis_size=None, donate_argnums=()):
  if devices is not None:
    if not devices:
      raise ValueError("'devices' argument to pmap must be non-empty, or None.")
    devices = tuple(devices)
  # TODO(vanderplas): move these definitions into jax._src and avoid local import.
  import jax.experimental.multihost_utils as mhu  # pytype: disable=import-error
  axis_name, static_broadcasted_tuple, donate_tuple = api._shared_code_pmap(
      f, axis_name, static_broadcasted_argnums, donate_argnums, in_axes, out_axes)
  if isinstance(axis_name, core._TempAxisName):  # pylint: disable=protected-access
    axis_name = repr(axis_name)
  wrapped_fun = _pmap_wrap_init(f, static_broadcasted_tuple)

  def infer_params(*args, **kwargs):
    fun, dyn_argnums, dyn_args = _get_dyn_args(
        wrapped_fun, static_broadcasted_tuple, args)
    dyn_args_flat, dyn_args_tree = tree_flatten((dyn_args, kwargs))
    in_axes_flat = _get_in_axes_flat(
        in_axes, dyn_argnums, dyn_args, kwargs, len(dyn_args_flat),
        dyn_args_tree)
    local_axis_size = _mapped_axis_size(dyn_args_flat, in_axes_flat)
    donated_invars = _get_donated_invars(
        donate_tuple, dyn_args_tree, len(dyn_args_flat))
    fun, out_axes_thunk = flat_out_axes(fun, out_axes)
    flat_fun, out_tree = flatten_fun(fun, dyn_args_tree)
    global_axis_size = _get_global_axis_size(local_axis_size, devices,
                                             backend, axis_size)
    trace_state_clean = core.trace_state_clean()
    mesh = Mesh(
        _get_devices(devices, local_axis_size, global_axis_size, backend),
        (axis_name,))
    _pmapped, in_specs, out_specs = _cached_shard_map(
        flat_fun, mesh, in_axes_flat, out_axes_thunk, axis_name)
    jitted_f = api.jit(
        _pmapped,
        donate_argnums=[i for i, val in enumerate(donated_invars) if val])
    if xb.process_count() > 1:
      if trace_state_clean:
        flat_global_args = [
            host_local_array_to_global_array(arr, global_mesh=mesh, pspec=spec)
            for arr, spec in zip(dyn_args_flat, in_specs)]
      else:
        flat_global_args = mhu.host_local_array_to_global_array(
            dyn_args_flat, mesh, list(in_specs))
    else:
      flat_global_args = dyn_args_flat
    return jitted_f, flat_global_args, dyn_args_tree, out_tree, mesh, out_specs, donate_tuple

  @util.wraps(f)
  def wrapped(*args, **kwargs):
    jitted_f, flat_global_args, _, out_tree, mesh, out_specs, _ = infer_params(
        *args, **kwargs)
    outs = jitted_f(*flat_global_args)
    if xb.process_count() > 1:
      if core.trace_state_clean():
        outs = [
            global_array_to_host_local_array(out, global_mesh=mesh, pspec=spec)
            for out, spec in zip(outs, out_specs())
        ]
      else:
        outs = mhu.global_array_to_host_local_array(outs, mesh, out_specs())
    return tree_unflatten(out_tree(), outs)

  def lower(*args, **kwargs):
    jitted_f, flat_global_args, in_tree, out_tree, _, _, donate_tuple = infer_params(
        *args, **kwargs)
    abstract_args = list(map(core.shaped_abstractify, flat_global_args))
    args_info = stages.make_args_info(in_tree, abstract_args, donate_tuple)
    lowered = jitted_f.trace(*flat_global_args).lower()
    lowered = stages.Lowered(lowered._lowering, args_info, out_tree(),
                             no_kwargs=lowered._no_kwargs)
    return lowered
  wrapped.lower = lower
  return wrapped


@lu.cache
def _cached_shard_map(flat_fun, mesh, in_axes_flat, out_axes_thunk, axis_name):
  f_transformed = flat_fun.f_transformed
  def reset_stores_f_transformed(*args, **kwargs):
    for store in flat_fun.stores:
      if store is not None:
        store.reset()
    return f_transformed(*args, **kwargs)
  flat_fun.f_transformed = reset_stores_f_transformed
  in_specs = tuple(map(partial(_axes_to_pspec, axis_name), in_axes_flat))
  out_specs = lambda: map(partial(_axes_to_pspec, axis_name), out_axes_thunk())
  fun = _handle_reshapes(flat_fun, in_axes_flat, out_axes_thunk)
  return (_shard_map(fun.call_wrapped, mesh=mesh, in_specs=in_specs,
                     out_specs=out_specs, check_vma=False,
                     axis_names=set(mesh.axis_names)),
          in_specs, out_specs)

@lu.transformation2
def _handle_reshapes(f, in_axes, out_axes_thunk, *args, **kwargs):
  args = tree_map(lambda x, ax: x if ax is None else lax.squeeze(x, [ax]),
                  list(args), list(in_axes))
  out = f(*args)
  return tree_map(lambda x, ax: x if ax is None else lax.expand_dims(x, [ax]),
                  list(out), list(out_axes_thunk()))

def _get_devices(devices, local_axis_size, global_axis_size, backend):
  if backend is not None and devices is None:
    devs = xb.devices(backend=backend)
  else:
    devs = xb.devices() if devices is None else devices
  if xb.process_count() > 1:
    return devs[:global_axis_size]
  return devs[:local_axis_size]


def _ensure_index_tuple(x) -> tuple[int, ...]:
  try:
    return (int(x),)
  except TypeError:
    return tuple(int(i) for i in x)


def _mapped_axis_size(args, in_axes):
  """Infer axis size from the first mapped argument.

  shard_map already does a check on all arguments, so just look at first arg.

  Args:
    args: Flat list of arguments.
    in_axes: Flat tuple of axis indices (int or None for each arg).

  Returns:
    The size of the mapped axis.

  Raises:
    ValueError: If no args have a mapped axis.
  """
  if args and in_axes:
    # Fast path: check first arg/axis (most common case).
    if in_axes[0] is not None and hasattr(args[0], "shape"):
      return int(args[0].shape[in_axes[0]])
    # Slow path: scan for first mapped arg.
    if isinstance(in_axes, tuple):
      for arg, ax in zip(args, in_axes):
        if ax is not None and hasattr(arg, "shape"):
          return int(arg.shape[ax])
  raise ValueError("pmap requires at least one argument with a mapped axis.")


def _get_global_axis_size(local_axis_size: int, in_devices, backend_name: str,
                          global_axis_size: int | None):
  if (xb.process_count() == 1 and global_axis_size is not None and
      global_axis_size != local_axis_size):
    raise ValueError(
        f"Specified axis_size {global_axis_size} doesn't match received "
        f"axis_size {local_axis_size}.")

  if in_devices is not None and backend_name is None:
    backend = xb.get_device_backend(in_devices[0])
  else:
    backend = xb.get_backend(backend_name)

  if global_axis_size is None:
    if xb.process_count(backend) == 1:
      global_axis_size = local_axis_size
    elif in_devices is not None:
      global_axis_size = len(in_devices)
    else:
      global_axis_size = local_axis_size * xb.process_count(backend)
      assert all(
          len(xb.local_devices(pi, backend)) == xb.local_device_count(backend)
          for pi in range(xb.process_count(backend)))
  return global_axis_size


def _pmap_wrap_init(f, static_broadcasted_tuple):
  """Create a wrapped function with DebugInfo for pmap.

  Args:
    f: The function to wrap.
    static_broadcasted_tuple: Tuple of static argument indices.

  Returns:
    A lu.WrappedFun ready for pmap.
  """
  # Compute arg_names from signature, excluding static argnums
  if (signature := fun_signature(f)) is not None:
    static_set = frozenset(static_broadcasted_tuple)
    arg_names = tuple(
        name
        for i, name in enumerate(signature.parameters.keys())
        if i not in static_set
    )
  else:
    arg_names = None
  dbg = lu.DebugInfo("pmap", fun_sourceinfo(f), arg_names, None)
  return lu.wrap_init(f, debug_info=dbg)


def _get_dyn_args(wrapped_f, static_broadcasted_tuple, args):
  """Extract dynamic args and argnums after handling static args.

  Args:
    wrapped_f: The wrapped function.
    static_broadcasted_tuple: Tuple of static argument indices.
    args: Positional arguments.

  Returns:
    dyn_f: function with static args bound
    dyn_argnums: list of dynamic arg indices (or None if no static args)
    dyn_args: dynamic positional arguments (after static removed)

  Raises:
    ValueError: If static_broadcasted_argnums exceeds number of args.
  """

  if static_broadcasted_tuple:
    if max(static_broadcasted_tuple) >= len(args):
      raise ValueError(
          "pmapped function has"
          f" static_broadcasted_argnums={static_broadcasted_tuple} but was"
          f" called with only {len(args)} positional"
          f" argument{'s' if len(args) > 1 else ''}. All static broadcasted"
          " arguments must be passed positionally."
      )
    dyn_argnums = [
        i for i in range(len(args)) if i not in static_broadcasted_tuple
    ]
    wrapped_f, dyn_args = argnums_partial(wrapped_f, dyn_argnums, args)
  else:
    dyn_argnums = None
    dyn_args = args
  return wrapped_f, dyn_argnums, dyn_args


def _get_in_axes_flat(
    in_axes, dyn_argnums, dyn_args, kwargs, num_flat_args, in_tree
):
  """Compute flat in_axes tuple from in_axes prefix and args structure.

  Args:
    in_axes: The original in_axes specification.
    dyn_argnums: The indices of dynamic (non-static) positional args, or None if
      no static args.
    dyn_args: The dynamic positional args (after static args removed).
    kwargs: The keyword arguments.
    num_flat_args: Total number of flat args.
    in_tree: The PyTreeDef of (dyn_args, kwargs).

  Returns:
    Flat tuple of axis indices (int or None for each flat arg).

  Raises:
    ValueError: If in_axes is not a valid prefix of the args structure.
  """
  # Compute dyn_in_axes from in_axes and dyn_argnums
  if dyn_argnums is not None and isinstance(in_axes, tuple):
    dyn_in_axes = tuple(in_axes[i] for i in dyn_argnums)
  else:
    dyn_in_axes = in_axes

  # Fast path: avoid broadcast_prefix for common simple cases
  in_axes_flat = None

  if isinstance(dyn_in_axes, int):
    if dyn_in_axes == 0:
      # Most common case: all args mapped on axis 0, including kwargs (which also get 0)
      in_axes_flat = (0,) * num_flat_args
    elif not kwargs:
      # No kwargs: broadcast single in_axes to all positional leaves
      in_axes_flat = (dyn_in_axes,) * num_flat_args
  elif dyn_in_axes is None and not kwargs:
    # Unusual case: no mapping, no kwargs
    in_axes_flat = (None,) * num_flat_args
  elif (
      not kwargs
      and isinstance(dyn_in_axes, tuple)
      and all(isinstance(ax, int) or ax is None for ax in dyn_in_axes)
  ):
    # No kwargs: check if it's a simple flat tuple matching positional args
    if len(dyn_in_axes) == len(dyn_args) and num_flat_args == len(dyn_args):
      # Each positional arg is a leaf (no nested structure)
      in_axes_flat = dyn_in_axes

  # Slow path: use broadcast_flattened_prefix_with_treedef for complex cases
  if in_axes_flat is None:
    try:
      # Flatten in_axes prefix tree (treating None as leaf)
      flat_in_axes_prefix, in_axes_tree = tree_flatten(
          (dyn_in_axes, 0), is_leaf=lambda x: x is None
      )
      in_axes_flat = tuple(
          broadcast_flattened_prefix_with_treedef(
              flat_in_axes_prefix, in_axes_tree, in_tree
          )
      )
    except ValueError:
      e, *_ = prefix_errors((dyn_in_axes, 0), (dyn_args, kwargs))
      ex = e("pmap in_axes")
      (msg,) = ex.args
      msg += (
          "\n\nThe 'full pytree' here is the tuple of arguments passed "
          "positionally to the pmapped function, and the value of `in_axes` "
          "must be a tree prefix of that tuple. But it was not a prefix."
      )
      if kwargs:
        msg += (
            "\n\nWhen some arguments are passed by keyword to the pmapped "
            "function, they are not included in the comparison to `in_axes`. "
            "Instead, each argument passed by keyword is mapped over its "
            "leading axis. See the description of `in_axes` in the `pmap` "
            "docstring: "
            "https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html#jax.pmap"
        )
      msg += (
          "\n\nCheck that the value of the `in_axes` argument to `pmap` "
          "is a tree prefix of the tuple of arguments passed positionally to "
          "the pmapped function."
      )
      raise ValueError(msg) from None

  return in_axes_flat


def _get_donated_invars(donate_tuple, in_tree, num_flat_args):
  """Compute donation vector for arguments.

  Args:
    donate_tuple: Tuple of donated argument indices.
    in_tree: PyTreeDef of input structure.
    num_flat_args: Number of flat arguments.

  Returns:
    Tuple of bools indicating which flat args are donated.
  """

  if donate_tuple and not config.debug_nans.value:
    return donation_vector(donate_tuple, (), in_tree)
  else:
    return (False,) * num_flat_args


@lru_cache
def _local_to_global_aval(local_aval, mesh, pspec):
  pspec = sharding_impls.prepare_axis_resources(pspec, 'pspec to array_mapping')
  return pxla.mesh_local_to_global(
      mesh, sharding_impls.get_array_mapping(pspec), local_aval)

@lru_cache
def _global_to_local_aval(global_aval, mesh, pspec):
  pspec = sharding_impls.prepare_axis_resources(pspec, 'pspec to array_mapping')
  return pxla.mesh_global_to_local(
      mesh, sharding_impls.get_array_mapping(pspec), global_aval)


def host_local_array_to_global_array(
    arr: Any, *, global_mesh: Mesh, pspec: Any):
  if pspec is None:
    raise ValueError(
        '`None` is not a valid input to the pspecs argument. Please use '
        'jax.sharding.PartitionSpec() if you wanted to replicate your input.')
  if isinstance(arr, array.ArrayImpl) and not arr.is_fully_addressable:
    return arr
  if (isinstance(arr, array.ArrayImpl) and isinstance(
      arr.sharding, sharding_impls.PmapSharding)) or not hasattr(arr, 'shape'):
    arr = np.array(arr)
  if arr.dtype == dtypes.float0:
    arr = np.zeros(arr.shape, dtype=np.dtype(bool))
  dtype = arr.dtype
  if is_prng_key_array := isinstance(arr, prng.PRNGKeyArray):
    arr = arr._base_array

  local_sharding = sharding_impls.NamedSharding(global_mesh.local_mesh, pspec)

  if (isinstance(arr, array.ArrayImpl) and
      arr.sharding.is_equivalent_to(local_sharding, arr.ndim)):
    arrays = [x.data for x in arr.addressable_shards]
  else:
    arr = dtypes.canonicalize_value(arr)
    arrays = [
        arr[i] for i in local_sharding.devices_indices_map(arr.shape).values()
    ]

  global_aval = _local_to_global_aval(
      core.ShapedArray(arr.shape, arr.dtype), global_mesh, pspec)

  out = pxla.batched_device_put(
      global_aval, sharding_impls.NamedSharding(global_mesh, pspec),
      arrays, list(global_mesh.local_mesh.devices.flat))
  if is_prng_key_array:
    return prng.PRNGKeyArray(dtype._impl, out)
  return out


def global_array_to_host_local_array(
    arr: Any, *, global_mesh: Mesh, pspec: Any):
  if pspec is None:
    raise ValueError(
        '`None` is not a valid input to the pspecs argument. Please use '
        'jax.sharding.PartitionSpec() if you wanted to replicate your input.')
  if isinstance(arr, array.ArrayImpl) and arr.is_fully_addressable:
    return arr
  if not hasattr(arr, 'shape'):
    arr = np.array(arr)
  if arr.dtype == dtypes.float0:
    arr = np.zeros(arr.shape, dtype=np.dtype(bool))
  dtype = arr.dtype
  if is_prng_key_array := isinstance(arr, prng.PRNGKeyArray):
    arr = arr._base_array

  global_sharding = sharding_impls.NamedSharding(global_mesh, pspec)
  local_sharding = sharding_impls.NamedSharding(global_mesh.local_mesh, pspec)
  local_aval = _global_to_local_aval(
      core.ShapedArray(arr.shape, arr.dtype), global_mesh, pspec)

  if isinstance(arr, array.ArrayImpl):
    if arr.sharding.is_equivalent_to(global_sharding, arr.ndim):
      arrays = arr._arrays
    else:
      resharded_array = dispatch.device_put_p.bind(arr, devices=None, srcs=None, device=global_sharding)
      arrays = resharded_array._arrays
    out = array.ArrayImpl(local_aval, local_sharding, arrays, committed=True)
    if is_prng_key_array:
      return prng.PRNGKeyArray(dtype._impl, out)
    return out
  else:
    arr = dtypes.canonicalize_value(arr)
    arrays = [
        arr[i] for i in local_sharding.devices_indices_map(arr.shape).values()
    ]
  return pxla.batched_device_put(
      local_aval, local_sharding, arrays,
      list(global_mesh.local_mesh.devices.flat))
