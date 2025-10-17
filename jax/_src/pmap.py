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

from functools import partial

from jax._src import core
from jax._src import dispatch
from jax._src import linear_util as lu
from jax._src import stages
from jax._src import traceback_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.shard_map import _shard_map, _axes_to_pspec
from jax._src.api import _shared_code_pmap, _prepare_pmap, jit
from jax._src.mesh import Mesh
from jax._src.lax import lax
from jax._src.tree_util import tree_map, tree_unflatten

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip
traceback_util.register_exclusion(__file__)

# Implementing pmap in terms of shard_map

def pmap(f, axis_name=None, *, in_axes=0, out_axes=0,
         static_broadcasted_argnums=(), devices=None, backend=None,
         axis_size=None, donate_argnums=(), global_arg_shapes=None):
  del global_arg_shapes
  # TODO(vanderplas): move these definitions into jax._src and avoid local import.
  import jax.experimental.multihost_utils as mhu  # pytype: disable=import-error
  devices = tuple(devices) if devices is not None else devices
  axis_name, static_broadcasted_tuple, donate_tuple = _shared_code_pmap(
      f, axis_name, static_broadcasted_argnums, donate_argnums, in_axes, out_axes)
  if isinstance(axis_name, core._TempAxisName):
    axis_name = repr(axis_name)

  def infer_params(*args, __check=True, **kwargs):
    p = _prepare_pmap(f, in_axes, out_axes, static_broadcasted_tuple,
                      donate_tuple, devices, backend, axis_size, args, kwargs)
    if __check:
      for arg in p.flat_args:
        dispatch.check_arg(arg)
    mesh = Mesh(_get_devices(p, backend), (axis_name,))
    _pmapped, in_specs, out_specs = _cached_shard_map(
        p.flat_fun, mesh, p.in_axes_flat, p.out_axes_thunk, axis_name)
    jitted_f = jit(
        _pmapped,
        donate_argnums=[i for i, val in enumerate(p.donated_invars) if val])
    if __check and xb.process_count() > 1:
      flat_global_args = mhu.host_local_array_to_global_array(
          p.flat_args, mesh, list(in_specs))
    else:
      flat_global_args = p.flat_args
    return jitted_f, flat_global_args, p, mesh, out_specs, donate_tuple

  @util.wraps(f)
  def wrapped(*args, **kwargs):
    jitted_f, flat_global_args, p, mesh, out_specs, _ = infer_params(
        *args, **kwargs)
    outs = jitted_f(*flat_global_args)
    if xb.process_count() > 1:
      outs = mhu.global_array_to_host_local_array(outs, mesh, out_specs())
    return tree_unflatten(p.out_tree(), outs)

  def lower(*args, **kwargs):
    jitted_f, flat_global_args, p, _, _, donate_tuple = infer_params(
        *args, __check=False, **kwargs
    )
    abstract_args = list(map(core.shaped_abstractify, flat_global_args))
    args_info = stages.make_args_info(p.in_tree, abstract_args, donate_tuple)
    lowered = jitted_f.trace(*flat_global_args).lower()
    lowered = stages.Lowered(lowered._lowering, args_info, p.out_tree(),
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

def _get_devices(p, backend):
  if backend is not None and p.devices is None:
    devs = xb.devices(backend=backend)
  else:
    devs = xb.devices() if p.devices is None else p.devices
  if xb.process_count() > 1:
    return devs[:p.global_axis_size]
  return devs[:p.local_axis_size]
