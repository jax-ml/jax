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
from typing import Any, Callable, NamedTuple
import warnings

from jax._src import api
from jax._src.api_util import (
    argnums_partial, donation_vector, fun_signature, fun_sourceinfo)
from jax._src import array
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import pjit as pjit_lib
from jax._src import prng
from jax._src import sharding_impls
from jax._src import stages
from jax._src import traceback_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.lib import jaxlib_extension_version
from jax._src.interpreters import pxla
from jax._src.lax import lax
from jax._src.lib import xla_client as xc
from jax._src.mesh import Mesh
from jax._src.shard_map import _axes_to_pspec, shard_map
from jax._src.tree_util import (
    broadcast_flattened_prefix_with_treedef, broadcast_prefix,
    prefix_errors, tree_flatten, tree_map, tree_unflatten)
import numpy as np

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

# jaxlib extension version >= 400 supports _rewrap_with_aval_and_sharding
_SUPPORTS_REWRAP = jaxlib_extension_version >= 400
traceback_util.register_exclusion(__file__)


# Implementing pmap in terms of shard_map

def pmap(f, axis_name=None, *, in_axes=0, out_axes=0,
         static_broadcasted_argnums=(), devices=None, backend=None,
         axis_size=None, donate_argnums=()):
  if devices is not None:
    if not devices:
      raise ValueError("'devices' argument to pmap must be non-empty, or None.")
    devices = tuple(devices)
  axis_name, static_broadcasted_tuple, donate_tuple = api._shared_code_pmap(  # pylint: disable=protected-access
      f, axis_name, static_broadcasted_argnums, donate_argnums, in_axes, out_axes)
  if isinstance(axis_name, core._TempAxisName):  # pylint: disable=protected-access
    axis_name = repr(axis_name)
  wrapped_fun = _pmap_wrap_init(f, static_broadcasted_tuple)
  out_axes_flat, out_axes_tree = tree_flatten(out_axes)
  out_axes_flat = tuple(out_axes_flat)

  def infer_params(*args, **kwargs):
    process_count = xb.process_count(backend)
    trace_state_clean = core.trace_state_clean()
    dyn_f, dyn_argnums, dyn_args = _get_dyn_args(
        wrapped_fun, static_broadcasted_tuple, args)
    dyn_args_flat, dyn_args_tree = tree_flatten((dyn_args, kwargs))
    in_axes_flat = _get_in_axes_flat(
        in_axes, dyn_argnums, dyn_args, kwargs, len(dyn_args_flat),
        dyn_args_tree)
    local_axis_size = _mapped_axis_size(dyn_args_flat, in_axes_flat)
    donated_invars = _get_donated_invars(
        donate_tuple, dyn_args_tree, len(dyn_args_flat))
    mesh_devices = _get_mesh_devices(
        devices, backend, local_axis_size, axis_size, trace_state_clean)
    cached = _cached_shard_map(
        dyn_f, dyn_args_tree, in_axes_flat, out_axes_flat, out_axes_tree,
        donated_invars, mesh_devices, axis_name)
    jitted_f = (cached.jitted_f_with_shardings if trace_state_clean
                else cached.jitted_f)
    if process_count > 1:
      dyn_args_flat = host_local_array_to_global_array(
          dyn_args_flat, cached, trace_state_clean, donated_invars
      )
    return (cached, jitted_f, dyn_args_flat, dyn_args_tree, donate_tuple,
            process_count, trace_state_clean)

  @util.wraps(f)
  def wrapped(*args, **kwargs):
    cached, jitted_f, dyn_args_flat, _, _, process_count, trace_state_clean = (
        infer_params(*args, **kwargs))
    out = jitted_f(*dyn_args_flat)
    if process_count > 1:
      out = global_array_to_host_local_array(out, cached, trace_state_clean)
    return out

  def lower(*args, **kwargs):
    _, jitted_f, args_flat, in_tree, donated_tuple, _, _ = infer_params(
        *args, **kwargs
    )
    abstract_args = list(map(core.shaped_abstractify, args_flat))
    args_info = stages.make_args_info(in_tree, abstract_args, donated_tuple)
    lowered = jitted_f.trace(*args_flat).lower()
    # NOTE(dsuo): Calling .compile()(*inputs) will fail because our jitted function
    # has no notion of host-local <> global conversion.
    return stages.Lowered(
        lowered._lowering,  # pylint: disable=protected-access
        args_info,
        lowered.out_tree,
        no_kwargs=lowered._no_kwargs,  # pylint: disable=protected-access
    )

  wrapped.lower = lower
  return wrapped


class CachedShardMap(NamedTuple):
  """Core cached pmap result.

  Attributes:
    pmapped: The shard_map-transformed function.
    in_specs_flat: Flattened input PartitionSpecs for array conversion.
    local_devices: List of devices in the local mesh.
    in_local_shardings: NamedSharding for each input using local mesh.
    in_global_shardings: NamedSharding for each input using global mesh.
    mesh: The global Mesh for this pmap invocation.
    out_specs: Output PartitionSpecs as a pytree prefix.
    out_local_shardings_thunk: Cached thunk returning (local, global) sharding
      pairs for output pspecs.
    donate_argnums: Indices of donated arguments.
    out_global_shardings: Output NamedShardings as a pytree.
    jitted_f: Pre-cached jit wrapper without explicit shardings.
    jitted_f_with_shardings: Pre-cached jit wrapper with in/out shardings.
  """

  pmapped: Callable[..., Any]
  in_specs_flat: tuple[sharding_impls.PartitionSpec, ...]
  local_devices: list[xc.Device]
  in_local_shardings: list[sharding_impls.NamedSharding]
  in_global_shardings: list[sharding_impls.NamedSharding]
  mesh: Mesh
  out_specs: Any  # pytree of PartitionSpecs
  out_local_shardings_thunk: Callable[
      [sharding_impls.PartitionSpec],
      tuple[sharding_impls.NamedSharding, sharding_impls.NamedSharding],
  ]
  donate_argnums: list[int]
  out_global_shardings: Any  # pytree of NamedShardings
  jitted_f: Any
  jitted_f_with_shardings: Any


@lu.cache
def _cached_shard_map(fun, in_tree, in_axes_flat, out_axes_flat, out_axes_tree,
                      donated_invars, mesh_devices, axis_name):
  mesh = Mesh(mesh_devices, (axis_name,))
  out_axes = tree_unflatten(out_axes_tree, list(out_axes_flat))
  in_specs = tuple(map(partial(_axes_to_pspec, axis_name), in_axes_flat))
  out_specs = tree_map(
      partial(_axes_to_pspec, axis_name), out_axes, is_leaf=lambda x: x is None
  )
  def _fun(*flat_args):
    args = tree_map(
        lambda x, ax: x if ax is None else lax.squeeze(x, [ax]),
        flat_args,
        in_axes_flat,
    )
    args, kwargs = tree_unflatten(in_tree, args)
    out = fun.call_wrapped(*args, **kwargs)
    out_flat, out_tree = tree_flatten(out)
    out_axes_flat = broadcast_prefix(out_axes, out, is_leaf=lambda x: x is None)
    out_flat = tree_map(
        lambda x, ax: x if ax is None else lax.expand_dims(x, [ax]),
        out_flat,
        out_axes_flat,
    )
    return tree_unflatten(out_tree, out_flat)
  _pmapped = shard_map(_fun, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                       check_vma=False, axis_names=set(mesh.axis_names))
  # Donation is now safe in multi-host mode because host_local_array_to_global_array
  # copies donated arrays instead of rewrapping them (which would share buffers).
  donate_argnums = [i for i, val in enumerate(donated_invars) if val]

  # out_specs is a pytree, so use tree_map to convert to shardings
  get_sharding = (
      lambda spec: sharding_impls.NamedSharding(mesh, spec)
      if spec is not None else spec)
  out_global_shardings = tree_map(
      get_sharding, out_specs, is_leaf=lambda x: x is None)

  @lru_cache
  def out_local_shardings_thunk(pspec):
    return (
        sharding_impls.NamedSharding(mesh.local_mesh, pspec),
        sharding_impls.NamedSharding(mesh, pspec),
    )

  local_devices = list(mesh.local_mesh.devices.flat)
  in_local_shardings = [
      sharding_impls.NamedSharding(mesh.local_mesh, p) for p in in_specs]
  in_global_shardings = [
      sharding_impls.NamedSharding(mesh, p) for p in in_specs]
  jitted_f = api.jit(_pmapped, donate_argnums=donate_argnums)
  jitted_f_with_shardings = api.jit(
      _pmapped,
      donate_argnums=donate_argnums,
      in_shardings=tuple(in_global_shardings),
      out_shardings=out_global_shardings,
  )
  return CachedShardMap(
      pmapped=_pmapped,
      in_specs_flat=in_specs,
      local_devices=local_devices,
      in_local_shardings=in_local_shardings,
      in_global_shardings=in_global_shardings,
      mesh=mesh,
      out_specs=out_specs,
      out_local_shardings_thunk=out_local_shardings_thunk,
      donate_argnums=donate_argnums,
      out_global_shardings=out_global_shardings,
      jitted_f=jitted_f,
      jitted_f_with_shardings=jitted_f_with_shardings,
  )


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
def _get_mesh_devices(devices, backend, local_axis_size, axis_size,
                      trace_state_clean):
  """Compute effective mesh devices based on context.

  Args:
    devices: The mesh devices tuple.
    backend: The backend to use.
    local_axis_size: The local axis size (per-process).
    axis_size: User-specified global axis size (optional).
    trace_state_clean: True if in execution mode (not tracing).

  Returns:
    Tuple of effective mesh devices sliced appropriately.

  Raises:
    ValueError: If axis_size doesn't match inferred size in single-process.
  """
  process_count = xb.process_count(backend)

  # Validate explicit axis_size in single-process mode
  if (process_count == 1 and axis_size is not None and
      axis_size != local_axis_size):
    raise ValueError(
        f"Specified axis_size {axis_size} doesn't match received "
        f"axis_size {local_axis_size}.")

  # Compute global_axis_size
  if axis_size is not None:
    global_axis_size = axis_size
  elif process_count > 1:
    global_axis_size = local_axis_size * process_count
    # Validate all processes have the same number of local devices
    assert all(
        len(xb.local_devices(pi, backend)) == xb.local_device_count(backend)
        for pi in range(process_count))
  else:
    global_axis_size = local_axis_size

  # Determine mesh devices
  if devices is not None:
    mesh_devices = devices
  elif process_count > 1:
    # Multi-process: group devices by process (host) for optimal collective
    # performance. This matches the old pmap's device ordering which uses
    # local_devices(process_index) in a nested loop, ensuring devices from
    # the same host are contiguous in the mesh.
    # pylint: disable=g-complex-comprehension
    mesh_devices = tuple(
        d
        for process_index in range(process_count)
        for d in xb.local_devices(process_index, backend)
    )
    # pylint: enable=g-complex-comprehension
  elif backend is not None:
    mesh_devices = tuple(xb.devices(backend=backend))
  else:
    mesh_devices = tuple(xb.devices())

  if not trace_state_clean and process_count > 1:
    # Tracing in multihost: use local devices
    return tuple(xb.local_devices(backend=backend)[:local_axis_size])
  else:
    return mesh_devices[:global_axis_size]


@lru_cache
def _local_to_global_aval(shape, dtype, sharding):
  """Compute global aval from local shape."""
  pspec_prepared = sharding_impls.prepare_axis_resources(sharding.spec, "pspec")
  local_aval = core.ShapedArray(shape, dtype)
  return pxla.mesh_local_to_global(
      sharding.mesh,
      sharding_impls.get_array_mapping(pspec_prepared),
      local_aval,
  )


@lru_cache
def _global_to_local_aval(shape, dtype, sharding):
  """Compute local aval from global shape."""
  pspec_prepared = sharding_impls.prepare_axis_resources(sharding.spec, "pspec")
  global_aval = core.ShapedArray(shape, dtype)
  return pxla.mesh_global_to_local(
      sharding.mesh,
      sharding_impls.get_array_mapping(pspec_prepared),
      global_aval,
  )


@lru_cache
def _local_device_indices(local_sharding, shape):
  """Cached device indices for slicing arrays."""
  return tuple(local_sharding.devices_indices_map(shape).values())


@lru_cache
def _is_sharding_equivalent(sharding_a, sharding_b, ndim):
  """Check if sharding is equivalent to NamedSharding(mesh.local_mesh, pspec)."""
  return sharding_a.is_equivalent_to(sharding_b, ndim)


@lru_cache
def _get_out_shardings(out_tree, pspecs, out_shardings_thunk):
  """Get flattened output shardings, combining pspec flattening and sharding lookup."""
  out_pspecs_flat = pjit_lib.flatten_axis_resources(
      "output pspecs", out_tree, pspecs, tupled_args=True
  )
  return tuple(zip(*[out_shardings_thunk(p) for p in out_pspecs_flat]))


def host_local_array_to_global_array(
    dyn_args_flat, cached, trace_state_clean, donated_invars
):
  """Convert host-local arrays to global arrays for multihost pmap.

  Args:
    dyn_args_flat: Flat list of input arrays.
    cached: CachedPmap tuple with mesh and sharding info.
    trace_state_clean: True if in execution mode (not tracing).
    donated_invars: Tuple of bools indicating which args are donated. For
      donated args that require the slow path, we delete the original to free
      memory.

  Returns:
    Converted global arrays.
  """
  if not trace_state_clean:
    import jax.experimental.multihost_utils as mhu  # pytype: disable=import-error

    return list(
        mhu.host_local_array_to_global_array(
            tuple(dyn_args_flat), cached.mesh, cached.in_specs_flat
        )
    )

  in_local_shardings = cached.in_local_shardings
  in_global_shardings = cached.in_global_shardings

  if dyn_args_flat and isinstance(
      dyn_args_flat[0], (core.Tracer, core.AbstractValue)
  ):
    return dyn_args_flat

  for i, arr in enumerate(dyn_args_flat):
    local_sharding = in_local_shardings[i]
    global_sharding = in_global_shardings[i]
    donated = donated_invars[i]
    prng_impl = None
    typ = type(arr)
    if typ is array.ArrayImpl and not arr.is_fully_addressable:
      continue
    if typ is not array.ArrayImpl:
      if typ is prng.PRNGKeyArray:
        prng_impl = arr.dtype._impl  # pylint: disable=protected-access
        arr = arr._base_array  # pylint: disable=protected-access
      dtype = arr.dtype
      if dtype == dtypes.float0:
        arr = np.zeros(arr.shape, dtype=bool)
      arr = np.asarray(arr)
      if dtype != dtypes.canonicalize_dtype(dtype):
        arr = dtypes.canonicalize_value(arr)
    shape, dtype = arr.shape, arr.dtype
    typ = type(arr)

    global_aval = _local_to_global_aval(shape, dtype, global_sharding)
    if typ == array.ArrayImpl and _is_sharding_equivalent(
        arr.sharding, local_sharding, len(arr.shape)
    ):
      # Fast path: rewrap without copy (shares buffers with original).
      # For donated args, jit's donation will invalidate the shared buffers,
      # which is the expected behavior - original arrays become invalid.
      if _SUPPORTS_REWRAP:
        dyn_args_flat[i] = arr._rewrap_with_aval_and_sharding(  # pylint: disable=protected-access
            global_aval, global_sharding
        )
      else:
        # Fallback for older jaxlib: use batched_device_put with shard data.
        arrays = [x.data for x in arr.addressable_shards]
        dyn_args_flat[i] = pxla.batched_device_put(
            global_aval,
            global_sharding,
            arrays,
            list(local_sharding._device_assignment),
        )  # pylint: disable=protected-access
    else:
      # Slow path: slice and device_put (creates new buffers).
      # For donated args, we must explicitly delete the original to free memory.
      arrays = [
          arr[idx] for idx in _local_device_indices(local_sharding, shape)
      ]
      dyn_args_flat[i] = pxla.batched_device_put(
          global_aval,
          global_sharding,
          arrays,
          list(local_sharding._device_assignment),
      )  # pylint: disable=protected-access
      if donated and typ is array.ArrayImpl:
        warnings.warn(
            "Donated pmap argument required resharding. This causes a brief "
            "2x memory spike before the original is freed. For optimal "
            "donation, ensure inputs are correctly sharded before pmap.",
            stacklevel=4,
        )
        arr.delete()
    if prng_impl is not None:
      dyn_args_flat[i] = prng.PRNGKeyArray(prng_impl, dyn_args_flat[i])

  return dyn_args_flat


def global_array_to_host_local_array(out, cached, trace_state_clean):
  """Convert global arrays to host-local arrays for multihost pmap output.

  Args:
    out: The output pytree from jitted function.
    cached: CachedPmap tuple with mesh and sharding info.
    trace_state_clean: True if in execution mode (not tracing).

  Returns:
    Host-local output pytree.
  """
  if not trace_state_clean:
    import jax.experimental.multihost_utils as mhu  # pytype: disable=import-error

    return mhu.global_array_to_host_local_array(
        out, cached.mesh, cached.out_specs
    )

  out_flat, out_tree = tree_flatten(out)
  out_local_shardings, out_global_shardings = _get_out_shardings(
      out_tree, cached.out_specs, cached.out_local_shardings_thunk
  )

  if out_flat and isinstance(out_flat[0], (core.Tracer, core.AbstractValue)):
    return out

  for i, arr in enumerate(out_flat):
    local_sharding = out_local_shardings[i]
    global_sharding = out_global_shardings[i]
    prng_impl = None
    typ = type(arr)
    if typ is array.ArrayImpl and arr.is_fully_addressable:
      continue
    if typ is not array.ArrayImpl:
      if typ is prng.PRNGKeyArray:
        prng_impl = arr.dtype._impl  # pylint: disable=protected-access
        arr = arr._base_array  # pylint: disable=protected-access
      try:
        _ = arr.shape
      except AttributeError:
        arr = np.array(arr)
      dtype = arr.dtype
      if dtype == dtypes.float0:
        arr = np.zeros(arr.shape, dtype=bool)
      if dtype != dtypes.canonicalize_dtype(dtype):
        arr = dtypes.canonicalize_value(arr)
    shape, dtype = arr.shape, arr.dtype
    typ = type(arr)

    local_aval = _global_to_local_aval(shape, dtype, global_sharding)
    if typ == array.ArrayImpl:
      if not _is_sharding_equivalent(arr.sharding, global_sharding, len(shape)):
        arr = api.device_put(arr, global_sharding)
      if _SUPPORTS_REWRAP:
        out_flat[i] = arr._rewrap_with_aval_and_sharding(  # pylint: disable=protected-access
            local_aval, local_sharding
        )
      else:
        # Fallback for older jaxlib: construct ArrayImpl directly.
        out_flat[i] = array.ArrayImpl(
            local_aval, local_sharding, arr._arrays, committed=True
        )  # pylint: disable=protected-access
    else:
      arrays = [
          arr[idx] for idx in _local_device_indices(local_sharding, shape)
      ]
      out_flat[i] = pxla.batched_device_put(
          local_aval,
          local_sharding,
          arrays,
          list(local_sharding._device_assignment),
      )  # pylint: disable=protected-access
    if prng_impl is not None:
      out_flat[i] = prng.PRNGKeyArray(prng_impl, out_flat[i])

  return tree_unflatten(out_tree, out_flat)
