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

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence, Iterable
import contextlib
import dataclasses
from functools import partial
import inspect
import itertools as it
import logging
import weakref
from typing import NamedTuple, Any, Union, cast
import warnings

import numpy as np

from jax._src import api
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import mesh as mesh_lib
from jax._src import op_shardings
from jax._src import profiler
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import stages
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.core import typeof, cur_qdd
from jax._src.api_util import (
  argnums_partial_except, flatten_axes, flatten_fun, flatten_fun_nokwargs,
  donation_vector, check_callable, resolve_argnums,
  argnames_partial_except, debug_info, _check_no_aliased_ref_args,
  _check_no_aliased_closed_over_refs)
from jax._src.interpreters import partial_eval as pe
from jax._src.partition_spec import PartitionSpec
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib import jax_jit
from jax._src.lib import xla_client as xc
from jax._src.lib import jaxlib_extension_version
from jax._src.mesh import AbstractMesh
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (
    NamedSharding, GSPMDSharding,
    SingleDeviceSharding, PmapSharding, AUTO, UNSPECIFIED, UnspecifiedValue,
    prepare_axis_resources, parse_flatten_op_sharding, canonicalize_sharding,
    flatten_spec, _internal_use_concrete_mesh)
from jax._src.layout import Format, Layout, AutoLayout, get_layout_for_vmap
from jax._src.state.types import RefEffect
from jax._src.traceback_util import api_boundary
from jax._src.tree_util import (
    tree_flatten, tree_unflatten, treedef_is_leaf, tree_structure,
    treedef_children, broadcast_prefix, all_leaves, prefix_errors, keystr,
    PyTreeDef, none_leaf_registry as none_lr, tree_map)
from jax._src.typing import ArrayLike
from jax._src.util import (
    HashableFunction, safe_map, safe_zip, wraps, distributed_debug_log,
    split_list, weakref_lru_cache, merge_lists, subs_list, fun_name,
    fun_qual_name)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

traceback_util.register_exclusion(__file__)

PjitSharding = Union[GSPMDSharding, UnspecifiedValue, AUTO]
PjitShardingMinusUnspecified = Union[GSPMDSharding, AUTO]
MeshSharding = Union[NamedSharding, UnspecifiedValue, AUTO]
MeshShardingMinusUnspecified = Union[NamedSharding, AUTO]

logger = logging.getLogger(__name__)


class PjitInfo(NamedTuple):
  """Things that we know about a jit instance before it is called.

  In other words, this structure contains arguments to jit()/pjit(),
  preprocessed and validated.
  """
  fun_sourceinfo: str
  fun_signature: inspect.Signature | None
  # Shardings, as specified by the user. These can either be UNSPECIFIED or they
  # can be a tree (prefix) of shardings or None.
  user_specified_in_shardings: bool
  in_shardings_treedef: PyTreeDef
  in_shardings_leaves: tuple[Any, ...]
  out_shardings_treedef: PyTreeDef
  out_shardings_leaves: tuple[Any, ...]
  in_layouts_treedef: PyTreeDef
  in_layouts_leaves: tuple[Any, ...]
  out_layouts_treedef: PyTreeDef
  out_layouts_leaves: tuple[Any, ...]
  static_argnums: tuple[int, ...]
  static_argnames: tuple[str, ...]
  donate_argnums: tuple[int, ...]
  donate_argnames: tuple[str, ...]
  device: xc.Device | None
  backend: str | None
  keep_unused: bool
  inline: bool
  abstracted_axes: Any | None
  use_resource_env: bool  # False for jit, True for pjit
  compiler_options_kvs: tuple[tuple[str, Any], ...]

  # Hash and compare PjitInfo by identity when used as a cache key.
  def __hash__(self):
    return id(self)

  def __eq__(self, other):
    return self is other


def _python_pjit_helper(fun: Callable, jit_info: PjitInfo, *args, **kwargs):
  p, args_flat = _infer_params(fun, jit_info, args, kwargs)

  for arg in args_flat:
    dispatch.check_arg(arg)

  try:
    if (core.trace_state_clean() and not config.debug_key_reuse.value
        and not p.params['jaxpr'].jaxpr.is_high):
      args_flat = map(core.full_lower, args_flat)
      core.check_eval_args(args_flat)
      out_flat, compiled, profiler, const_args = _pjit_call_impl_python(
          *args_flat, **p.params)
    else:
      out_flat = jit_p.bind(*args_flat, **p.params)
      compiled = None
      profiler = None
      const_args = []
  except stages.DeviceAssignmentMismatchError as e:
    fails, = e.args
    fun_name = getattr(fun, '__qualname__', getattr(fun, '__name__', str(fun)))
    msg = stages._device_assignment_mismatch_error(
        fun_name, fails, args_flat, 'jit', p.arg_names)
    raise ValueError(msg) from None
  except dtypes.InvalidInputException as e:
    arg_names = [''] * len(args_flat) if p.arg_names is None else p.arg_names
    # Run canonicalization again to figure out which arg failed.
    if p.params['jaxpr'].consts:
      raise TypeError(e.args[0]) from e
    else:
      for arg, name, aval in zip(args_flat, arg_names, p.in_avals):
        try:
          dtypes.canonicalize_value(arg)
        except dtypes.InvalidInputException as _:
          # Reraise as TypeError with the new message.
          raise TypeError(
              f"Argument '{name}' of shape {aval.str_short()} of type"
              f' {type(arg)} is not a valid JAX type.') from e
      raise AssertionError("Unreachable") from e
  except api_util.InternalFloatingPointError as e:
    if getattr(fun, '_apply_primitive', False):
      raise FloatingPointError(f"invalid value ({e.ty}) encountered in {fun.__qualname__}") from None
    api_util.maybe_recursive_nan_check(e, fun, args, kwargs)

  outs = tree_unflatten(p.out_tree, out_flat)
  return (outs, out_flat, p.out_tree, args_flat,
          p.params['jaxpr'], compiled, profiler, const_args)


def _need_to_rebuild_with_fdo(pgle_profiler):
  return (pgle_profiler is not None and pgle_profiler.is_enabled()
          and not pgle_profiler.is_fdo_consumed())

def _get_fastpath_data(
    executable, out_tree, args_flat, out_flat, effects, consts_for_constvars,
    abstracted_axes, pgle_profiler, const_args: Sequence[ArrayLike]
    ) -> pxla.MeshExecutableFastpathData | None:
  out_reflattened, out_tree = pxla.reflatten_outputs_for_dispatch(out_tree, out_flat)

  use_fastpath = (
      executable is not None
      and isinstance(executable, pxla.MeshExecutable)
      and isinstance(executable.unsafe_call, pxla.ExecuteReplicated)
      # No effects in computation
      and not executable.unsafe_call.ordered_effects
      and not executable.unsafe_call.has_unordered_effects
      and not executable.unsafe_call.has_host_callbacks
      and all(isinstance(x, xc.ArrayImpl) for x in out_reflattened)
      and abstracted_axes is None
      # no ref state effects
      and not any(isinstance(e, RefEffect) for e in effects)
      # no prng reuse checking
      and not (config.debug_key_reuse.value and any(
        hasattr(arg, 'dtype') and dtypes.issubdtype(arg.dtype, dtypes.prng_key)
        for arg in (*args_flat, *out_flat, *consts_for_constvars)))
      and not _need_to_rebuild_with_fdo(pgle_profiler)
      )
  if jaxlib_extension_version < 366:
    use_fastpath = use_fastpath and not const_args

  if use_fastpath:
    out_avals = [o.aval for o in out_reflattened]
    out_committed = [o._committed for o in out_reflattened]
    kept_var_bitvec = [i in executable._kept_var_idx
                       for i in range(len(const_args) + len(args_flat))]
    in_shardings = [
        sharding_impls.physical_sharding(a, s)
        if a is not core.abstract_token and dtypes.issubdtype(a.dtype, dtypes.extended)
        else s
        for s, a in zip(executable._in_shardings, executable.in_avals)
    ]
    fastpath_data = pxla.MeshExecutableFastpathData(
        executable.xla_executable, out_tree, in_shardings,
        executable._out_shardings, out_avals, out_committed, kept_var_bitvec,
        executable._dispatch_in_layouts, const_args)
  else:
    fastpath_data = None
  return fastpath_data


# The entries are doubled here from the default 4096 because _pjit_call_impl
# also has a cpp dispatch path and that would double the number of entries in
# the global shared cache.
# This cache is only used for jit's with only fun. For example: jax.jit(f)
_cpp_pjit_cache_fun_only = xc._xla.PjitFunctionCache(capacity=8192)

# This cache is used for jit where extra arguments are defined other than the
# fun. For example: jax.jit(f, donate_argnums=...) OR
# jax.jit(f, out_shardings=...), etc. We don't use the same cache because the
# capacity might get full very fast because of all the jitted function in JAX
# which might evict train_step for example.
_cpp_pjit_cache_explicit_attributes = xc._xla.PjitFunctionCache(capacity=8192)


def _get_cpp_global_cache(contains_explicit_attributes: bool):
  if contains_explicit_attributes:
    return _cpp_pjit_cache_explicit_attributes
  else:
    return _cpp_pjit_cache_fun_only


def _cpp_pjit(fun: Callable, jit_info: PjitInfo):

  @api_boundary
  def cache_miss(*args, **kwargs):
    # args do not include the const args
    # See https://docs.jax.dev/en/latest/internals/constants.html.
    if config.no_tracing.value:
      raise RuntimeError(f"re-tracing function {jit_info.fun_sourceinfo} for "
                         "`jit`, but 'no_tracing' is set")

    (outs, out_flat, out_tree, args_flat, jaxpr,
     executable, pgle_profiler, const_args) = _python_pjit_helper(
         fun, jit_info, *args, **kwargs)

    maybe_fastpath_data = _get_fastpath_data(
        executable, out_tree, args_flat, out_flat, jaxpr.effects, jaxpr.consts,
        jit_info.abstracted_axes, pgle_profiler,
        const_args)

    return outs, maybe_fastpath_data, _need_to_rebuild_with_fdo(pgle_profiler)

  cache_key = pxla.JitGlobalCppCacheKeys(
      donate_argnums=jit_info.donate_argnums,
      donate_argnames=jit_info.donate_argnames,
      device=jit_info.device, backend=jit_info.backend,
      in_shardings_treedef=jit_info.in_shardings_treedef,
      in_shardings_leaves=jit_info.in_shardings_leaves,
      out_shardings_treedef=jit_info.out_shardings_treedef,
      out_shardings_leaves=jit_info.out_shardings_leaves,
      in_layouts_treedef=jit_info.in_layouts_treedef,
      in_layouts_leaves=jit_info.in_layouts_leaves,
      out_layouts_treedef=jit_info.out_layouts_treedef,
      out_layouts_leaves=jit_info.out_layouts_leaves,
      compiler_options_kvs=jit_info.compiler_options_kvs)
  cpp_pjit_f = xc._xla.pjit(
      fun_name(fun), fun, cache_miss, jit_info.static_argnums,
      jit_info.static_argnames, cache_key, tree_util.dispatch_registry,
      pxla.cc_shard_arg,
      _get_cpp_global_cache(cache_key.contains_explicit_attributes))

  cpp_pjitted_f = wraps(fun)(cpp_pjit_f)
  cpp_pjitted_f._fun = fun
  cpp_pjitted_f._jit_info = jit_info
  cpp_jitted_f_class = type(cpp_pjitted_f)
  # TODO(necula): make clear_cache private, no need to have it part of the API
  cpp_jitted_f_class.clear_cache = jit_evict_fn
  cpp_jitted_f_class.lower = jit_lower
  cpp_jitted_f_class.trace = jit_trace
  cpp_jitted_f_class.eval_shape = jit_eval_shape
  return cpp_pjitted_f

@api_boundary
def jit_trace(jit_func, *args, **kwargs) -> stages.Traced:
  p, args_flat = _infer_params(jit_func._fun, jit_func._jit_info, args, kwargs)
  donate_argnums = tuple(i for i, d in enumerate(p.params['donated_invars']) if d)
  args_info = stages.make_args_info(p.in_tree, p.in_avals, donate_argnums)
  lower_callable = partial(_resolve_and_lower, args_flat, **p.params,
                           pgle_profiler=None)
  return stages.Traced(
      p.params['jaxpr'], args_info, p.params["name"], p.out_tree,
      lower_callable, args_flat, p.arg_names, len(p.consts),
      p.params['out_shardings'])


@api_boundary
def jit_lower(jit_func, *args, **kwargs):
  return jit_trace(jit_func, *args, **kwargs).lower()

@api_boundary
def jit_eval_shape(jit_func, *args, **kwargs):
  return jit_trace(jit_func, *args, **kwargs).out_info

def jit_evict_fn(self):
  self._clear_cache()
  _create_pjit_jaxpr.evict_function(self._fun)  # pytype: disable=attribute-error
  _infer_params_cached.cache_clear()


def _split_layout_and_sharding(entries):
  entries_flat, treedef = tree_flatten(entries, is_leaf=lambda x: x is None)
  layouts, shardings = [], []

  for e in entries_flat:
    if isinstance(e, Format):
      layouts.append(e.layout)
      shardings.append(e.sharding)
    elif isinstance(e, (Layout, AutoLayout)):
      raise ValueError(
          '`jax.jit` does not accept device-local layouts directly. Create '
          'a `Format` instance wrapping this device-local layout and pass '
          f'that to `jit` instead. Got {e}')
    else:
      layouts.append(None)
      shardings.append(e)

  assert len(layouts) == len(shardings)
  return tree_unflatten(treedef, layouts), tree_unflatten(treedef, shardings)


def _parse_jit_arguments(fun: Callable, *, in_shardings: Any,
                         out_shardings: Any,
                         static_argnums: int | Sequence[int] | None,
                         static_argnames: str | Iterable[str] | None,
                         donate_argnums: int | Sequence[int] | None,
                         donate_argnames: str | Iterable[str] | None,
                         keep_unused: bool, device: xc.Device | None,
                         backend: str | None, inline: bool,
                         abstracted_axes: Any | None,
                         compiler_options: dict[str, Any] | None,
                         use_resource_env: bool) -> PjitInfo:
  """Parses the arguments to jit/pjit.

  Performs any preprocessing and validation of the arguments that we can do
  ahead of time before the jit()-ed function is invoked.
  """
  if abstracted_axes and not config.dynamic_shapes.value:
    raise ValueError("abstracted_axes must be used with --jax_dynamic_shapes")

  check_callable(fun)

  if backend is not None or device is not None:
    warnings.warn(
        'backend and device argument on jit is deprecated. You can use'
        ' `jax.device_put(..., jax.local_devices(backend="cpu")[0])` on the'
        ' inputs to the jitted function to get the same behavior.',
        DeprecationWarning,
    )
    if device is not None and backend is not None:
      raise ValueError("can't specify both a device and a backend for jit, "
                       f"got {device=} and {backend=}")
    if in_shardings is not None and not isinstance(in_shardings, UnspecifiedValue):
      raise ValueError('If backend or device is specified on jit, then '
                       'in_shardings should not be specified.')
    if out_shardings is not None and not isinstance(out_shardings, UnspecifiedValue):
      raise ValueError('If backend or device is specified on jit, then '
                       'out_shardings should not be specified.')

  if isinstance(in_shardings, list):
    # To be a tree prefix of the positional args tuple, in_axes can never be a
    # list: if in_axes is not a leaf, it must be a tuple of trees. However,
    # in cases like these users expect tuples and lists to be treated
    # essentially interchangeably, so we canonicalize lists to tuples here
    # rather than raising an error. https://github.com/jax-ml/jax/issues/2367
    in_shardings = tuple(in_shardings)

  in_layouts, in_shardings = _split_layout_and_sharding(in_shardings)
  out_layouts, out_shardings = _split_layout_and_sharding(out_shardings)

  in_shardings = prepare_axis_resources(in_shardings, 'in_shardings')
  out_shardings = prepare_axis_resources(out_shardings, 'out_shardings',
                                         allow_unconstrained_dims=True)

  user_specified_in_shardings = (in_shardings is not None and
                                 not isinstance(in_shardings, UnspecifiedValue))

  in_shardings_leaves, in_shardings_treedef = none_lr.flatten(in_shardings)
  out_shardings_leaves, out_shardings_treedef = none_lr.flatten(out_shardings)
  in_layouts_leaves, in_layouts_treedef = none_lr.flatten(in_layouts)
  out_layouts_leaves, out_layouts_treedef = none_lr.flatten(out_layouts)

  fun_sourceinfo = api_util.fun_sourceinfo(fun)
  fun_signature = api_util.fun_signature(fun)

  donate_argnums, donate_argnames, static_argnums, static_argnames = resolve_argnums(
      fun, fun_signature, donate_argnums, donate_argnames, static_argnums,
      static_argnames)

  compiler_options_kvs = (() if compiler_options is None else
                          tuple(compiler_options.items()))
  return PjitInfo(
        fun_sourceinfo=fun_sourceinfo,
        fun_signature=fun_signature,
        user_specified_in_shardings=user_specified_in_shardings,
        in_shardings_treedef=in_shardings_treedef,
        in_shardings_leaves=tuple(in_shardings_leaves),
        out_shardings_treedef=out_shardings_treedef,
        out_shardings_leaves=tuple(out_shardings_leaves),
        in_layouts_treedef=in_layouts_treedef,
        in_layouts_leaves=tuple(in_layouts_leaves),
        out_layouts_treedef=out_layouts_treedef,
        out_layouts_leaves=tuple(out_layouts_leaves),
        static_argnums=static_argnums,
        static_argnames=static_argnames, donate_argnums=donate_argnums,
        donate_argnames=donate_argnames, device=device, backend=backend,
        keep_unused=keep_unused, inline=inline,
        abstracted_axes=abstracted_axes,
        use_resource_env=use_resource_env,
        compiler_options_kvs=compiler_options_kvs)

def make_jit(fun: Callable,
             *,
             in_shardings: Any,
             out_shardings: Any,
             static_argnums: int | Sequence[int] | None,
             static_argnames: str | Iterable[str] | None,
             donate_argnums: int | Sequence[int] | None,
             donate_argnames: str | Iterable[str] | None,
             keep_unused: bool,
             device: xc.Device | None,
             backend: str | None,
             inline: bool,
             abstracted_axes: Any | None,
             compiler_options: dict[str, Any] | None,
             use_resource_env: bool) -> Any:
  """jit() and pjit() are thin wrappers around this function."""
  jit_info = _parse_jit_arguments(
        fun, in_shardings=in_shardings, out_shardings=out_shardings,
        static_argnums=static_argnums, static_argnames=static_argnames,
        donate_argnums=donate_argnums, donate_argnames=donate_argnames,
        keep_unused=keep_unused, device=device, backend=backend, inline=inline,
        abstracted_axes=abstracted_axes, compiler_options=compiler_options,
        use_resource_env=use_resource_env)
  return _cpp_pjit(fun, jit_info)


class PjitParams(NamedTuple):
  # Only jaxpr constants, we can't keep other arguments alive. These go as
  # first arguments for `params['jaxpr']`.
  consts: list[ArrayLike]  # Corresponding to jaxpr.constvars
  # Everything we need to trace, lower, and compile the jit function; passed
  # to `pjit_call_impl_python`, along with the `args_flat`
  params: dict[str, Any]
  in_avals: tuple[core.AbstractValue, ...]  # Not including the const_args
  in_tree: PyTreeDef  # Not including the const_args
  out_tree: PyTreeDef
  arg_names: tuple[str, ...]  # Not including the const_args


def _infer_params_impl(
    fun: Callable,
    ji: PjitInfo,
    ctx_mesh: mesh_lib.Mesh,
    dbg: core.DebugInfo,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    in_avals: tuple[core.AbstractValue, ...] | None,
) -> tuple[PjitParams, list[Any]]:
  util.test_event("pjit._infer_params_impl", fun)
  have_kwargs = bool(kwargs)
  if have_kwargs and ji.user_specified_in_shardings:
    raise ValueError(
        "pjit does not support kwargs when in_shardings is specified.")

  if not ctx_mesh.empty and (ji.backend or ji.device):
    raise ValueError(
        "Mesh context manager should not be used with jit when backend or "
        "device is also specified as an argument to jit.")

  axes_specs = _flat_axes_specs(ji.abstracted_axes, *args, **kwargs)

  f = lu.wrap_init(fun, debug_info=dbg)
  f, dyn_args = argnums_partial_except(f, ji.static_argnums, args, allow_invalid=True)
  del args
  f, dyn_kwargs = argnames_partial_except(f, ji.static_argnames, kwargs)
  del kwargs

  explicit_args, in_tree = tree_flatten((dyn_args, dyn_kwargs))
  flat_fun, out_tree = flatten_fun(f, in_tree)

  if (ji.donate_argnums or ji.donate_argnames) and not config.debug_nans.value:
    donated_invars = donation_vector(ji.donate_argnums, ji.donate_argnames, in_tree)
  else:
    donated_invars = (False,) * len(explicit_args)

  # If backend or device is set as an arg on jit, then resolve them to
  # in_shardings and out_shardings as if user passed in in_shardings
  # and out_shardings.
  device_or_backend_set = bool(ji.backend or ji.device)
  if device_or_backend_set:
    sharding = _create_sharding_with_device_backend(ji.device, ji.backend)
    leaves, treedef = tree_flatten(sharding)
    in_shardings_leaves = out_shardings_leaves = tuple(leaves)
    in_shardings_treedef = out_shardings_treedef = treedef
  else:
    api_name = 'pjit' if ji.use_resource_env else 'jit'
    in_shardings_leaves = tuple(
        _create_sharding_for_array(ctx_mesh, x, 'in_shardings', api_name)
        for x in ji.in_shardings_leaves)
    out_shardings_leaves = tuple(
        _create_sharding_for_array(ctx_mesh, x, 'out_shardings', api_name)
        for x in ji.out_shardings_leaves)
    in_shardings_treedef = ji.in_shardings_treedef
    out_shardings_treedef = ji.out_shardings_treedef

  assert None not in in_shardings_leaves
  assert None not in out_shardings_leaves

  in_type: core.InputType | tuple[core.AbstractValue, ...]
  if config.dynamic_shapes.value:
    assert in_avals is None
    in_type = pe.infer_lambda_input_type(axes_specs, explicit_args)
    in_avals = tuple(a for a, e in in_type if e)
  else:
    in_type = in_avals  # type: ignore
    in_type = tuple(core.AvalQDD(a, cur_qdd(x)) if a.has_qdd  # type: ignore
                    else a for a, x in zip(in_type, explicit_args))
  assert in_avals is not None

  in_shardings_flat, in_layouts_flat = _process_in_axis_resources(
      in_shardings_treedef, in_shardings_leaves,
      ji.in_layouts_treedef, ji.in_layouts_leaves,
      in_avals, in_tree, flat_fun.debug_info, device_or_backend_set, have_kwargs)

  qdd_token = _qdd_cache_index(flat_fun, in_type)

  jaxpr, consts, out_avals = _create_pjit_jaxpr(
      flat_fun, in_type, qdd_token, IgnoreKey(ji.inline))
  if config.mutable_array_checks.value:
    _check_no_aliased_closed_over_refs(dbg, (*jaxpr.consts, *consts), explicit_args)
  _qdd_cache_update(flat_fun, in_type, qdd_token, consts,
                    jaxpr.in_aval_qdds[:len(consts)])

  out_shardings_flat, out_layouts_flat = _check_and_canonicalize_out_shardings(
      out_shardings_treedef, out_shardings_leaves, ji.out_layouts_treedef,
      ji.out_layouts_leaves, HashableFunction(out_tree, closure=()),
      tuple(out_avals), jaxpr.jaxpr._debug_info, device_or_backend_set)

  assert len(explicit_args) == len(in_shardings_flat) == len(in_layouts_flat)

  if config.dynamic_shapes.value:
    implicit_args = _extract_implicit_args(
        cast(core.InputType, in_type), explicit_args)
  else:
    implicit_args = []
  args_flat = [*implicit_args, *explicit_args]

  num_extra_args = len(implicit_args) + len(consts)
  in_shardings_flat = (UNSPECIFIED,) * num_extra_args + in_shardings_flat
  in_layouts_flat = (None,) * num_extra_args + in_layouts_flat
  donated_invars = (False,) * num_extra_args + donated_invars
  assert (len(in_shardings_flat) == len(in_layouts_flat) ==
          len(donated_invars) == len(consts) + len(args_flat))

  params = dict(
      jaxpr=jaxpr,
      in_shardings=in_shardings_flat,
      out_shardings=out_shardings_flat,
      in_layouts=in_layouts_flat,
      out_layouts=out_layouts_flat,
      donated_invars=donated_invars,
      ctx_mesh=ctx_mesh,
      name=fun_qual_name(flat_fun),
      keep_unused=ji.keep_unused,
      inline=ji.inline,
      compiler_options_kvs=ji.compiler_options_kvs,
  )
  return (PjitParams(consts, params, in_avals,
                     in_tree, out_tree(), dbg.arg_names),
          args_flat)


class InferParamsCacheEntry:
  """Mutable value object for _infer_params_cached."""
  __slots__ = ['pjit_params']
  pjit_params: PjitParams | None
  def __init__(self):
    self.pjit_params = None


# We use an outer cache that is keyed on the signature of the arguments, but
# when populating a cache entry using _infer_params_impl, we need to provide
# actual arguments. In principle, we could refactor _infer_params_impl to look
# only at an argument signature instead of args/kwargs in those cases that we
# cache, but this was a more minimal change.
@util.weakref_lru_cache
def _infer_params_cached(
    fun: Callable,
    jit_info: PjitInfo,
    signature: jax_jit.ArgumentSignature,
    in_avals: tuple[core.AbstractValue, ...],
    ctx_mesh: mesh_lib.Mesh,
) -> InferParamsCacheEntry:
  return InferParamsCacheEntry()


def _infer_params(
    fun: Callable, ji: PjitInfo, args: tuple[Any, ...], kwargs: dict[str, Any]
  ) -> tuple[PjitParams, list[core.Value]]:
  if ji.use_resource_env:  # pjit
    phys_mesh = mesh_lib.thread_resources.env.physical_mesh
    with (_internal_use_concrete_mesh(phys_mesh),
          mesh_lib.use_abstract_mesh(phys_mesh.abstract_mesh)):
      return _infer_params_internal(fun, ji, args, kwargs)
  else:
    return _infer_params_internal(fun, ji, args, kwargs)


def _infer_params_internal(
    fun: Callable, ji: PjitInfo, args: tuple[Any, ...], kwargs: dict[str, Any]
  ) -> tuple[PjitParams, list[Any]]:
  ctx_mesh = mesh_lib.get_concrete_mesh()
  dbg = debug_info(
      'jit', fun, args, kwargs, static_argnums=ji.static_argnums,
      static_argnames=ji.static_argnames, sourceinfo=ji.fun_sourceinfo,
      signature=ji.fun_signature)

  if config.dynamic_shapes.value:  # don't use the cache
    p, args_flat = _infer_params_impl(fun, ji, ctx_mesh, dbg,
                                      args, kwargs, in_avals=None)
    return p, p.consts + args_flat

  signature, dynargs = jax_jit.parse_arguments(
      args, tuple(kwargs.values()), tuple(kwargs.keys()), ji.static_argnums,
      ji.static_argnames, tree_util.default_registry)
  avals = _infer_input_type(fun, dbg, dynargs)
  entry = _infer_params_cached(fun, ji, signature, avals, ctx_mesh)

  if entry.pjit_params is None:
    p, args_flat = _infer_params_impl(
        fun, ji, ctx_mesh, dbg, args, kwargs, in_avals=avals)
    if p.params['jaxpr'].jaxpr.is_high:
      return p, p.consts + args_flat
    entry.pjit_params = p
  return entry.pjit_params, entry.pjit_params.consts + dynargs

def _infer_input_type(fun: Callable, dbg: core.DebugInfo,
                      explicit_args) -> tuple[core.AbstractValue, ...]:
  avals = []
  try:
    for i, x in enumerate(explicit_args):
      avals.append(core.shaped_abstractify(x))
  except OverflowError:
    arg_path = f"argument path is {dbg.arg_names[i]}"  # pytype: disable=name-error
    raise OverflowError(
      "An overflow was encountered while parsing an argument to a jitted "
      f"computation, whose {arg_path}."
    ) from None
  except TypeError:
    arg_description = f"path {dbg.arg_names[i]}"  # pytype: disable=name-error
    raise TypeError(
      f"Error interpreting argument to {fun} as an abstract array."
      f" The problematic value is of type {type(x)} and was passed to"  # pytype: disable=name-error
      f" the function at {arg_description}.\n"
      "This typically means that a jit-wrapped function was called with a non-array"
      " argument, and this argument was not marked as static using the"
      " static_argnums or static_argnames parameters of jax.jit."
    ) from None
  if config.mutable_array_checks.value:
    _check_no_aliased_ref_args(dbg, avals, explicit_args)
  return tuple(avals)

def _extract_implicit_args(
  in_type: Sequence[tuple[core.AbstractValue, bool]],
  explicit_args: Sequence[Any]
) -> Sequence[core.Tracer]:
  """
  Given an input type and explicitly-passed arguments (per the user-facing API
  calling convention), extract implicit axis size arguments from shapes of
  explicit arguments (for the trace-time / jaxpr-level calling convention).
  """
  # First, using `in_type` construct a list to represent the full argument list,
  # leaving the implicit arguments as None placeholders for now.
  explicit_args_ = iter(explicit_args)
  args = [next(explicit_args_) if expl else None for _, expl in in_type]
  assert next(explicit_args_, None) is None
  del explicit_args, explicit_args_

  # Next, populate the implicit arguments using the DBIdxs in `in_type`.
  for i, (aval, explicit) in enumerate(in_type):
    if not explicit or not isinstance(aval, core.DShapedArray):
      continue  # can't populate an implicit argument
    arg = args[i]
    assert arg is not None
    for d1, d2 in zip(aval.shape, arg.aval.shape):
      if isinstance(d1, core.DBIdx):
        if args[d1.val] is None:
          args[d1.val] = d2
        assert core.same_referent(args[d1.val], d2)
  assert all(x is not None for x in args)
  return [x for x, (_, e) in zip(args, in_type) if not e]  # type: ignore

def _flat_axes_specs(abstracted_axes, *args, **kwargs
                     ) -> list[pe.AbstractedAxesSpec] | None:
  if abstracted_axes is None: return None
  if kwargs: raise NotImplementedError
  def ax_leaf(l):
    return (isinstance(l, dict) and all_leaves(l.values()) or
            isinstance(l, tuple) and all_leaves(l, lambda x: x is None))
  return broadcast_prefix(abstracted_axes, args, ax_leaf)


class JitWrapped(stages.Wrapped):

  def eval_shape(self, *args, **kwargs):
    """See ``jax.eval_shape``."""
    raise NotImplementedError

  def trace(self, *args, **kwargs) -> stages.Traced:
    raise NotImplementedError


# in_shardings and out_shardings can't be None as the default value
# because `None` means that the input is fully replicated.
def pjit(
    fun: Callable,
    in_shardings: Any = UNSPECIFIED,
    out_shardings: Any = UNSPECIFIED,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: xc.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any | None = None,
    compiler_options: dict[str, Any] | None = None,
) -> JitWrapped:
  """Makes ``fun`` compiled and automatically partitioned across multiple devices.

  NOTE: This function is now equivalent to jax.jit please use that instead.
  The returned function has semantics equivalent to those of ``fun``, but is
  compiled to an XLA computation that runs across multiple devices
  (e.g. multiple GPUs or multiple TPU cores). This can be useful if the jitted
  version of ``fun`` would not fit in a single device's memory, or to speed up
  ``fun`` by running each operation in parallel across multiple devices.

  The partitioning over devices happens automatically based on the
  propagation of the input partitioning specified in ``in_shardings`` and
  the output partitioning specified in ``out_shardings``. The resources
  specified in those two arguments must refer to mesh axes, as defined by
  the :py:func:`jax.sharding.Mesh` context manager. Note that the mesh
  definition at :func:`~pjit` application time is ignored, and the returned function
  will use the mesh definition available at each call site.

  Inputs to a :func:`~pjit`'d function will be automatically partitioned across devices
  if they're not already correctly partitioned based on ``in_shardings``.
  In some scenarios, ensuring that the inputs are already correctly pre-partitioned
  can increase performance. For example, if passing the output of one
  :func:`~pjit`'d function to another :func:`~pjit`’d function (or the same
  :func:`~pjit`’d function in a loop), make sure the relevant
  ``out_shardings`` match the corresponding ``in_shardings``.

  .. note::
    **Multi-process platforms:** On multi-process platforms such as TPU pods,
    :func:`~pjit` can be used to run computations across all available devices across
    processes. To achieve this, :func:`~pjit` is designed to be used in SPMD Python
    programs, where every process is running the same Python code such that all
    processes run the same :func:`~pjit`'d function in the same order.

    When running in this configuration, the mesh should contain devices across
    all processes. All inputs arguments must be globally shaped.
    ``fun`` will still be executed across *all* devices in the mesh,
    including those from other processes, and will be given a global view of the
    data spread across multiple processes as a single array.

    The SPMD model also requires that the same multi-process :func:`~pjit`'d
    functions must be run in the same order on all processes, but they can be
    interspersed with arbitrary operations running in a single process.

  Args:
    fun: Function to be compiled. Should be a pure function, as side-effects may
      only be executed once. Its arguments and return value should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      Positional arguments indicated by ``static_argnums`` can be anything at
      all, provided they are hashable and have an equality operation defined.
      Static arguments are included as part of a compilation cache key, which is
      why hash and equality operators must be defined.
    in_shardings: Pytree of structure matching that of arguments to ``fun``,
      with all actual arguments replaced by resource assignment specifications.
      It is also valid to specify a pytree prefix (e.g. one value in place of a
      whole subtree), in which case the leaves get broadcast to all values in
      that subtree.

      The ``in_shardings`` argument is optional. JAX will infer the shardings
      from the input :py:class:`jax.Array`'s, and defaults to replicating the input
      if the sharding cannot be inferred.

      The valid resource assignment specifications are:

      - :py:class:`Sharding`, which will decide how the value
        will be partitioned. With this, using a mesh context manager is not
        required.
      - :py:obj:`None` is a special case whose semantics are:
          - if the mesh context manager is *not* provided, JAX has the freedom to
            choose whatever sharding it wants.
            For in_shardings, JAX will mark is as replicated but this behavior
            can change in the future.
            For out_shardings, we will rely on the XLA GSPMD partitioner to
            determine the output shardings.
          - If the mesh context manager is provided, None will imply that the
            value will be replicated on all devices of the mesh.
      - For backwards compatibility, in_shardings still supports ingesting
        :py:class:`PartitionSpec`. This option can *only* be used with the
        mesh context manager.

        - :py:class:`PartitionSpec`, a tuple of length at most equal to the rank
          of the partitioned value. Each element can be a :py:obj:`None`, a mesh
          axis or a tuple of mesh axes, and specifies the set of resources assigned
          to partition the value's dimension matching its position in the spec.

      The size of every dimension has to be a multiple of the total number of
      resources assigned to it.
    out_shardings: Like ``in_shardings``, but specifies resource
      assignment for function outputs.
      The ``out_shardings`` argument is optional. If not specified, :py:func:`jax.jit`
      will use GSPMD's sharding propagation to determine how to shard the outputs.
    static_argnums: An optional int or collection of ints that specify which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded in
      Python (during tracing), and so the corresponding argument values can be
      any Python object.

      Static arguments should be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and immutable. Calling the jitted function
      with different values for these constants will trigger recompilation.
      Arguments that are not arrays or containers thereof must be marked as
      static.

      If ``static_argnums`` is not provided, no arguments are treated as static.
    static_argnames: An optional string or collection of strings specifying
      which named arguments to treat as static (compile-time constant). See the
      comment on ``static_argnums`` for details. If not
      provided but ``static_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    donate_argnums: Specify which positional argument buffers are "donated" to
      the computation. It is safe to donate argument buffers if you no longer
      need them once the computation has finished. In some cases XLA can make
      use of donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation, JAX
      will raise an error if you try to. By default, no argument buffers are
      donated.

      If neither ``donate_argnums`` nor ``donate_argnames`` is provided, no
      arguments are donated. If ``donate_argnums`` is not provided but
      ``donate_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``donate_argnames``
      (or vice versa). If both ``donate_argnums`` and ``donate_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``donate_argnums`` or ``donate_argnames`` will
      be donated.

      For more details on buffer donation see the
      `FAQ <https://docs.jax.dev/en/latest/faq.html#buffer-donation>`_.
    donate_argnames: An optional string or collection of strings specifying
      which named arguments are donated to the computation. See the
      comment on ``donate_argnums`` for details. If not
      provided but ``donate_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    keep_unused: If `False` (the default), arguments that JAX determines to be
      unused by `fun` *may* be dropped from resulting compiled XLA executables.
      Such arguments will not be transferred to the device nor provided to the
      underlying executable. If `True`, unused arguments will not be pruned.
    device: This argument is deprecated. Please put your arguments on the
      device you want before passing them to jit.
      Optional, the Device the jitted function will run on. (Available devices
      can be retrieved via :py:func:`jax.devices`.) The default is inherited
      from XLA's DeviceAssignment logic and is usually to use
      ``jax.devices()[0]``.
    backend: This argument is deprecated. Please put your arguments on the
      backend you want before passing them to jit.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation and
    automatically partitioned by the mesh available at each call site.

  For example, a convolution operator can be automatically partitioned over
  an arbitrary set of devices by a single :func:`~pjit` application:

  >>> import jax
  >>> import jax.numpy as jnp
  >>> import numpy as np
  >>> from jax.sharding import Mesh, PartitionSpec
  >>> from jax.experimental.pjit import pjit
  >>>
  >>> x = jnp.arange(8, dtype=jnp.float32)
  >>> f = pjit(lambda x: jax.numpy.convolve(x, jnp.asarray([0.5, 1.0, 0.5]), 'same'),
  ...         in_shardings=None, out_shardings=PartitionSpec('devices'))
  >>> with Mesh(np.array(jax.devices()), ('devices',)):
  ...   print(f(x))  # doctest: +SKIP
  [ 0.5  2.   4.   6.   8.  10.  12.  10. ]
  """
  return make_jit(
      fun, in_shardings=in_shardings, out_shardings=out_shardings,
      static_argnums=static_argnums, static_argnames=static_argnames,
      donate_argnums=donate_argnums, donate_argnames=donate_argnames,
      keep_unused=keep_unused, device=device, backend=backend, inline=inline,
      abstracted_axes=abstracted_axes, compiler_options=compiler_options,
      use_resource_env=True)


def hashable_pytree(pytree):
  vals, treedef = tree_flatten(pytree)
  vals = tuple(vals)
  return HashableFunction(lambda: tree_unflatten(treedef, vals),
                          closure=(treedef, vals))


def _create_sharding_for_array(mesh, x, name, api_name):
  if x is None:
    if api_name == 'jit' or mesh.empty:
      return UNSPECIFIED
    return sharding_impls.cached_named_sharding(mesh, PartitionSpec())
  if isinstance(x, (AUTO, UnspecifiedValue, Sharding)):
    return x
  if mesh.empty:
    raise RuntimeError(
        f'{api_name} requires a non-empty mesh in context if you are passing'
        f' `PartitionSpec`s to {name}. You can define a context mesh via'
        ' `jax.set_mesh(mesh)`. Alternatively, provide `Sharding`s to'
        f' {name} and then the mesh context manager is not required.')
  assert isinstance(x, PartitionSpec), x
  return sharding_impls.cached_named_sharding(mesh, x)


def _create_sharding_with_device_backend(device, backend):
  if device is not None:
    assert backend is None
    out = SingleDeviceSharding(device)
  elif backend is not None:
    assert device is None
    out = SingleDeviceSharding(xb.get_backend(backend).local_devices()[0])
  else:
    raise AssertionError('Unreachable!')
  out._device_backend = True
  return out


def flatten_axis_resources(what, tree, shardings, tupled_args):
  try:
    return tuple(flatten_axes(what, tree, shardings, tupled_args=tupled_args))
  except ValueError:
    pass  # Raise a tree prefix error below

  # Tree leaves are always valid prefixes, so if there was a prefix error as
  # assumed here, axis_resources must not be a leaf.
  assert not treedef_is_leaf(tree_structure(shardings))

  # Check the type directly rather than using isinstance because of namedtuples.
  if tupled_args and (type(shardings) is not tuple or
                      len(shardings) != len(tree.children())):
    # We know axis_resources is meant to be a tuple corresponding to the args
    # tuple, but while it is a non-leaf pytree, either it wasn't a tuple or it
    # wasn't the right length.
    msg = (f"{what} specification must be a tree prefix of the positional "
           f"arguments tuple. In particular, {what} must either be a Sharding, "
           "a PartitionSpec, or a tuple of length equal to the number of "
           "positional arguments.")
    # If `tree` represents an args tuple, then `axis_resources` must be a tuple.
    # TODO(mattjj,apaszke): disable implicit list casts, remove 'or list' below
    if type(shardings) is not tuple:
      msg += f" But {what} is not a tuple: got {type(shardings)} instead."
    elif len(shardings) != len(tree.children()):
      msg += (f" But {what} is the wrong length: got a tuple or list of length "
              f"{len(shardings)} for an args tuple of length "
              f"{len(tree.children())}.")

    # As an extra hint, let's check if the user just forgot to wrap
    # shardings in a singleton tuple.
    if len(tree.children()) == 1:
      try: flatten_axes(what, tree, (shardings,))
      except ValueError: pass  # That's not the issue.
      else:
        msg += (f" Given the corresponding argument being "
                f"passed, it looks like {what} might need to be wrapped in "
                f"a singleton tuple.")

    raise ValueError(msg)

  axis_tree = shardings

  # Because we only have the `tree` treedef and not the full pytree here,
  # we construct a dummy tree to compare against. Revise this in callers?
  dummy_tree = tree_unflatten(tree, [PytreeLeaf()] * tree.num_leaves)
  errors = prefix_errors(axis_tree, dummy_tree)
  if errors:
    e = errors[0]  # Only show information about the first disagreement found.
    raise e(what)

  # At this point we've failed to find a tree prefix error.
  assert False, "Please open a bug report!"  # This should be unreachable.

class PytreeLeaf:
  def __repr__(self): return "pytree leaf"


@util.cache(max_size=4096, trace_context_in_key=False)
def _process_in_axis_resources(in_shardings_treedef, in_shardings_leaves,
                               in_layouts_treedef, in_layouts_leaves,
                               in_avals, in_tree, debug_info: core.DebugInfo,
                               device_or_backend_set, kws):
  if not kws:
    in_tree, _ = treedef_children(in_tree)

  orig_in_shardings = tree_unflatten(in_shardings_treedef, in_shardings_leaves)
  # Only do this if original in_shardings are unspecified. If it is AUTO, go
  # via flatten_axis_resources.
  if isinstance(orig_in_shardings, UnspecifiedValue):
    in_shardings_flat = (orig_in_shardings,) * len(in_avals)
  else:
    in_shardings_flat = flatten_axis_resources(
        "pjit in_shardings", in_tree, orig_in_shardings, tupled_args=True)

  in_layouts = tree_unflatten(in_layouts_treedef, in_layouts_leaves)
  if in_layouts is None:
    in_layouts_flat = (in_layouts,) * len(in_avals)
  else:
    in_layouts_flat = flatten_axis_resources(
        "pjit in_layouts", in_tree, in_layouts, tupled_args=True)

  if not config.dynamic_shapes.value:
    pjit_check_aval_sharding(in_shardings_flat, in_avals,
                             debug_info.safe_arg_names(len(in_avals)),
                             "pjit arguments", allow_uneven_sharding=False)
    check_aval_layout_compatibility(
        in_layouts_flat, in_avals,
        debug_info.safe_arg_names(len(in_avals)), "jit arguments")  # type: ignore[arg-type]
  return in_shardings_flat, in_layouts_flat

callsites_with_tracing_cache_miss: set[str] = set()

def diff_tracing_cache_keys(
    k: tuple, oldk: tuple, debug_info: lu.DebugInfo) -> tuple[Sequence[str], int]:
  """Explanations of differences between the cache keys, along with diff sizes.

  Result: a pair of a list of explanations for differences, and the total size
    of the differences. The sizes are used to pick the old key with the smallest
    different size for the explanation that is shown to the user.
  """
  (fun_transforms_k, fun_params_k, fun_in_type_k,
   (arg_in_type_k, _, arg_inline_k), ctx_k) = k
  (fun_transforms_ok, fun_params_ok, fun_in_type_ok,
   (arg_in_type_ok, _, arg_inline_ok), ctx_ok) = oldk

  diffs: list[tuple[str, int]] = []  # each difference with its size
  def unavailable(key_field: str, what_k, what_ok):
    diffs.append(
        (f"different {key_field}:\n    now: {what_k}\n != before: {what_ok}.\n"
         "explanation unavailable! "
         "please open an issue at https://github.com/jax-ml/jax.",
         10))

  def list_diff_size(s1: Sequence, s2: Sequence) -> int:
    min_len = min(len(s1), len(s2))
    diff_size = max(len(s1), len(s2)) - min_len
    diff_size += sum(e1 != e2 for e1, e2 in zip(s1[:min_len],
                                                s2[:min_len]))
    return diff_size

  different_leaf_count = False

  def explain_transform_argnums_partial(param_k: tuple, param_ok: tuple):
    dyn_argnums_k, static_args_k = param_k
    dyn_argnums_ok, static_args_ok = param_ok
    if dyn_argnums_k != dyn_argnums_ok:
      diffs.append(
          ("different static_argnums:\n"
           f"  dynamic argnums now {dyn_argnums_k} and before {dyn_argnums_ok}",
          1))
    if static_args_k != static_args_ok:
      diffs.append(
          ("different value of static args:\n"
           f"  now {', '.join(repr(a.val) for a in static_args_k)}"
           f" and before {', '.join(repr(a.val) for a in static_args_ok)}",
          list_diff_size(static_args_k, static_args_ok)))

  def explain_transform_argnames_partial(param_k: tuple, param_ok: tuple):
    static_kwargs_k, = param_k
    static_kwargs_ok, = param_ok
    static_kwargs_k = [(k, v.val) for k, v in
                       sorted(static_kwargs_k.val.items())]
    static_kwargs_ok = [(k, v.val) for k, v in
                        sorted(static_kwargs_ok.val.items())]
    if static_kwargs_k != static_kwargs_ok:
      diffs.append(
          ("different value of static kwargs:\n"
           f"  now {{{', '.join(f'{k}: {repr(v)}' for k, v in static_kwargs_k)}}}"
           f" and before {{{', '.join(f'{k}: {repr(v)}' for k, v in static_kwargs_ok)}}}",
          list_diff_size(static_kwargs_k, static_kwargs_ok)))

  def explain_in_tree_diff(in_tree_k: PyTreeDef, in_tree_ok: PyTreeDef):
    nonlocal different_leaf_count
    different_leaf_count = (in_tree_k.num_leaves != in_tree_ok.num_leaves)
    if not different_leaf_count:
      # Look for the special case of passing positional args as kwargs or
      # vice-versa; the common prefix of positional args match.
      args_tree_k, kwargs_tree_k = treedef_children(in_tree_k)
      nr_args_k = len(treedef_children(args_tree_k))
      args_tree_ok, kwargs_tree_ok = treedef_children(in_tree_ok)
      nr_args_ok = len(treedef_children(args_tree_k))
      if (treedef_children(args_tree_k)[:min(nr_args_k, nr_args_ok)] ==
          treedef_children(args_tree_ok)[:min(nr_args_k, nr_args_ok)]):
        keys_k = kwargs_tree_k.node_data()[1]  # type: ignore[index]
        keys_ok = kwargs_tree_ok.node_data()[1]  # type: ignore[index]
        diffs.append(
            (("different number of args and kwargs, but same total number.\n"
              f"  now {nr_args_k} args and kwargs "
              f"with keys {keys_k}\n"
              f"  before {nr_args_ok} args and kwargs "
              f"with keys {keys_ok}"),
             abs(nr_args_ok - nr_args_k)))
        return

    in_tree_k_str = str(in_tree_k)
    in_tree_k_str = (in_tree_k_str if len(in_tree_k_str) < 73
                     else in_tree_k_str[:73] + "...")
    in_tree_ok_str = str(in_tree_ok)
    in_tree_ok_str = (in_tree_ok_str if len(in_tree_ok_str) < 73
                     else in_tree_ok_str[:73] + "...")
    diff = [f"different input pytree:\n  now: {in_tree_k_str}\n"
            f"  before: {in_tree_ok_str}"]

    errs = list(tree_util.equality_errors_pytreedef(in_tree_k, in_tree_ok))
    for path, thing1, thing2, explanation in errs:
      fst, *path = path  # type: ignore
      base = ["args", "kwargs"][fst.idx]
      diff.append(
          f"  * at {base}{keystr(tuple(path))}, now {thing1} and before {thing2},"
          f" so {explanation}")
    diffs.append(("\n".join(diff), len(errs)))

  def explain_args_type_diff(args_k: tuple[core.AbstractValue],
                             args_ok: tuple[core.AbstractValue]):
    diff_size = 0
    arg_names = debug_info.safe_arg_names(len(args_k))
    def arg_type_to_str(at):
      if hasattr(at, "str_short"):
        return at.str_short(short_dtypes=True)
      else:
        return str(at)
    args_k_str = ", ".join(f"{an}: {arg_type_to_str(at)}"
                           for an, at in zip(arg_names, args_k))
    args_k_str = args_k_str if len(args_k_str) < 73 else args_k_str[:73] + "..."
    diff = [f"different input types:\n  types now: {args_k_str}"]
    add_weak_type_hint = False

    for name, arg_t_k, arg_t_ok in zip(arg_names, args_k, args_ok):
      if arg_t_k == arg_t_ok: continue
      this_arg_diff_size = 0
      if type(arg_t_k) == type(arg_t_ok) == core.ShapedArray:
        s1, s2 = arg_type_to_str(arg_t_k), arg_type_to_str(arg_t_ok)
        this_arg_diff_size += list_diff_size(arg_t_k.shape, arg_t_ok.shape)  # type: ignore

        if arg_t_k.weak_type != arg_t_ok.weak_type:  # type: ignore
          s1 += f"{{weak_type={arg_t_k.weak_type}}}"  # type: ignore
          s2 += f"{{weak_type={arg_t_ok.weak_type}}}"  # type: ignore
          add_weak_type_hint = True
          this_arg_diff_size += 1
        elif arg_t_k.sharding != arg_t_ok.sharding:  # type: ignore
          s1 = arg_t_k.str_short(short_dtypes=True, mesh_axis_types=True)  # type: ignore
          s2 = arg_t_ok.str_short(short_dtypes=True, mesh_axis_types=True)  # type: ignore
          this_arg_diff_size += 1
      else:
        s1, s2 = str(arg_t_k), str(arg_t_ok)
      diff_size += max(1, this_arg_diff_size)
      diff.append(f"    * at {name}, now {s1} and before {s2}")

    if add_weak_type_hint:
      diff.append(
          "where weak_type=True often means a Python builtin numeric value, and \n"
          "weak_type=False means a jax.Array.\n"
          "See https://docs.jax.dev/en/latest/type_promotion.html#weak-types.")
    diffs.append(("\n".join(diff), diff_size))

  if fun_transforms_k != fun_transforms_ok:
    if len(fun_transforms_k) != len(fun_transforms_ok):
      different_leaf_count = True  # Skip other more precise checks
      unavailable("fun_transforms length",
                  fun_transforms_k, fun_transforms_ok)
    else:
      for i, (t, ot) in enumerate(zip(fun_transforms_k, fun_transforms_ok)):
        t_name = t[0].__name__
        if t == ot: continue

        if t[0] != ot[0]:
          unavailable(f"fun_transforms[{i}] transform", t, ot)
          continue
        if t_name == "flatten_fun":
          explain_in_tree_diff(t[1][0], ot[1][0])
          continue
        if t_name == "_argnums_partial":
          explain_transform_argnums_partial(t[1], ot[1])
          continue
        if t_name == "_argnames_partial":
          explain_transform_argnames_partial(t[1], ot[1])
          continue
        unavailable(f"fun_transforms.{t_name} params", t[1:], ot[1:])
        continue

  # If we had different leaf counts, we can discard the _argnums_partial
  # difference. That transform sometimes occurs before the flatten_fun
  if different_leaf_count:
    diffs = [d for d in diffs if "fun_transforms._argnums_partial" not in d[0]]
  if fun_params_k != fun_params_ok:
    unavailable("fun_params", fun_params_k, fun_params_ok)
  if fun_in_type_k != fun_in_type_ok:
    unavailable("fun_in_type", fun_params_k, fun_params_ok)
  if arg_in_type_k != arg_in_type_ok and not different_leaf_count:
    explain_args_type_diff(arg_in_type_k, arg_in_type_ok)
  if arg_inline_k != arg_inline_ok:
    unavailable("arg_inline", arg_inline_k, arg_inline_ok)
  if ctx_k != ctx_ok:
    assert len(ctx_k) == len(ctx_ok)
    idxs = [f"  [{i}]: now {c_k} and before {c_ok}"
            for i, (c_k, c_ok) in enumerate(zip(ctx_k, ctx_ok)) if c_k != c_ok]
    diffs.append(
        ("different tracing context, e.g. due to config or context manager.\n"
         "found differences at positions\n" +
         ", and\n".join(idxs) +
         "\ncompare to tuple returned by "
         "config.trace_context() in jax/_src/config.py.",
         len(idxs)))
  if not diffs:  # Should never happen, but let's not crash
    unavailable("something (unexpected empty diffs)", k, oldk)
  diffs_and_sizes = util.unzip2(sorted(diffs, key=lambda d: d[1]))
  return (diffs_and_sizes[0], sum(diffs_and_sizes[1]))


def explain_tracing_cache_miss(
    fun: lu.WrappedFun, unseen_f: bool, cache: dict,
    key: tuple, elapsed_sec: float):
  if config.check_tracer_leaks.value: return  # TODO(mattjj): can remove this
  if key[3][2].val: return  # No explanations for "inline" functions

  debug_info = fun.debug_info
  func_filename = debug_info.func_filename
  if func_filename and not source_info_util.is_user_filename(func_filename):
   return

  msg: list[str] = []
  p = msg.append
  done = lambda: logger.log(logging.WARNING, "\n".join(msg))

  callsite = source_info_util.summarize(source_info_util.current())
  p(f"TRACING CACHE MISS at {callsite} costing {elapsed_sec * 1e3:.3f} ms because:")

  # have we seen this function before at all?
  src_info = ""
  if func_filename:
    src_info += f" defined at {func_filename}"
  if func_lineno := debug_info.func_lineno:
    src_info += f":{func_lineno}"
  func_name = debug_info.func_name
  if unseen_f or not cache:
    p(f"  never seen function:\n    {func_name} id={id(fun.f)}{src_info}")
    if callsite in callsites_with_tracing_cache_miss:
      p("  but seen another function defined on the same line; maybe the function is\n"
        "  being re-defined repeatedly, preventing caching?")
    else:
      callsites_with_tracing_cache_miss.add(callsite)
    return done()

  p(f"  for {func_name}{src_info}")

  diffs = [diff_tracing_cache_keys(key, ok, debug_info)
           for ok in cache.keys() if key != ok]
  assert diffs, "we must find some diffs if key differs from all cache keys"
  min_diff = min(diffs, key=lambda v: v[1])
  smallest_diffs: Sequence[Sequence[str]]  # the diffs for the closest keys
  smallest_diffs = [d[0] for d in diffs if d[1] == min_diff[1]]
  def indent_subsequent_lines(indent: int, msg: str) -> str:
    return msg.replace("\n", "\n" + " " * indent)
  def p_one_diff(diff: Sequence[str]):
    for d in diff:
      p("  * key with " + indent_subsequent_lines(4, d))

  if len(smallest_diffs) == 1:
    p("  all previously seen cache keys are different. Closest previous key:")
    p_one_diff(smallest_diffs[0])
  else:
    p("  all previously seen cache keys are different. "
      "Several previous keys are closest:")
    for d in smallest_diffs:
      p_one_diff(d)

  done()


@partial(lu.cache, explain=explain_tracing_cache_miss)
def _create_pjit_jaxpr(
    fun: lu.WrappedFun,
    in_type: core.InputType | Sequence[core.AbstractValue],
    qdd_token: int,
    ignored_inline: IgnoreKey
) -> tuple[core.ClosedJaxpr, list[core.Value], list[core.AbstractValue]]:
  util.test_event("create_pjit_jaxpr")
  del qdd_token  # just part of the cache key
  del ignored_inline  # just for explain_cache_miss
  if config.no_tracing.value:
    raise RuntimeError(f"re-tracing function {fun.f} for `jit`, but "
                       "'no_tracing' is set")
  with dispatch.log_elapsed_time(
      "Finished tracing + transforming {fun_name} for pjit in {elapsed_time:.9f} sec",
      fun_name=fun.__name__, event=dispatch.JAXPR_TRACE_EVENT):
    if config.dynamic_shapes.value:
      jaxpr, global_out_avals, consts = pe.trace_to_jaxpr_dynamic2(
          lu.annotate(fun, cast(core.InputType, in_type)))
    else:
      jaxpr, global_out_avals, consts = pe.trace_to_jaxpr_dynamic(fun, in_type)

  if config.debug_key_reuse.value:
    # Import here to avoid circular imports
    from jax.experimental.key_reuse._core import check_key_reuse_jaxpr  # pytype: disable=import-error
    check_key_reuse_jaxpr(jaxpr)

  # TODO(mattjj,yashkatariya): if we take the 'true' path then we *must* fall
  # off the C++ dispatch fast path for correctness. Ensure that happens.
  if any(isinstance(c, core.Tracer) or core.typeof(c).has_qdd for c in consts):
    closed_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))
    final_consts = consts
  else:
    closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
    final_consts = []
  return closed_jaxpr, final_consts, global_out_avals


@util.cache(max_size=4096, trace_context_in_key=False)
def _check_and_canonicalize_out_shardings(
    out_shardings_treedef, out_shardings_leaves, out_layouts_treedef,
    out_layouts_leaves, out_tree, out_avals,
    debug_info: core.DebugInfo,
    device_or_backend_set):
  orig_out_shardings = tree_unflatten(out_shardings_treedef, out_shardings_leaves)
  if isinstance(orig_out_shardings, (UnspecifiedValue, Sharding)):
    out_shardings_flat = (orig_out_shardings,) * len(out_avals)
  else:
    out_shardings_flat = flatten_axis_resources(
        "pjit out_shardings", out_tree(), orig_out_shardings,
        tupled_args=False)

  out_layouts = tree_unflatten(out_layouts_treedef, out_layouts_leaves)
  if out_layouts is None:
    out_layouts_flat = (out_layouts,) * len(out_avals)
  else:
    out_layouts_flat = flatten_axis_resources(
        "pjit out_layouts", out_tree(), out_layouts, tupled_args=False)

  if not config.dynamic_shapes.value:
    pjit_check_aval_sharding(
        out_shardings_flat, out_avals,
        debug_info.safe_result_paths(len(out_avals)),
        "pjit outputs", allow_uneven_sharding=False)
    check_aval_layout_compatibility(
        out_layouts_flat, out_avals,
        debug_info.safe_result_paths(len(out_avals)),
        "jit outputs")
  return out_shardings_flat, out_layouts_flat

_seen_qdds = weakref.WeakKeyDictionary()  # type: ignore

def _seen_qdds_get(fun, in_type) -> list:
  assert fun.in_type is None or fun.in_type == in_type
  cache = _seen_qdds.setdefault(fun.f, defaultdict(list))
  return cache[(fun.transforms, fun.params, in_type)]

def _qdd_cache_index(fun, in_type) -> int:
  cases = _seen_qdds_get(fun, in_type)
  for i, records in enumerate(cases):
    for obj, qdd in records:
      if core.cur_qdd(obj) != qdd: break
    else:
      return i
  return len(cases)

def _qdd_cache_update(fun, in_type, i, consts, aval_qdds):
  cases = _seen_qdds_get(fun, in_type)
  if i == len(cases):
    cases.append([(c, aval_qdd.qdd) for c, aval_qdd in zip(consts, aval_qdds)
                  if aval_qdd.has_qdd])


@dataclasses.dataclass(frozen=True)
class IgnoreKey:
  val: Any
  def __hash__(self):
    return hash(self.__class__)
  def __eq__(self, other):
    return isinstance(other, IgnoreKey)  # ignore self.val!


def pjit_check_aval_sharding(
    shardings, flat_avals, names: Sequence[str],
    what_aval: str, allow_uneven_sharding: bool):
  for aval, s, name in zip(flat_avals, shardings, names):
    if isinstance(s, (UnspecifiedValue, AUTO)):
      continue
    name_str = f' with pytree key path {name}' if name else ''
    shape = aval.shape
    try:
      # Sharding interfaces can implement `check_compatible_aval` as an optional
      # method to raise a more meaningful error.
      if hasattr(s, 'check_compatible_aval'):
        s.check_compatible_aval(shape)
      else:
        s._to_xla_hlo_sharding(len(shape))
    except ValueError as e:
      raise ValueError(
          f'One of {what_aval}{name_str} is incompatible with its sharding '
          f'annotation {s}: {e}')
    # Use the `OpSharding` proto to find out how many ways each dimension of
    # the aval is sharded. This approach will work across all
    # Sharding.
    hlo_sharding = s._to_xla_hlo_sharding(len(shape))
    assert hlo_sharding is not None
    num_ways_dim_sharded, _ = op_shardings.get_num_ways_dim_sharded(hlo_sharding)
    for i, size in enumerate(num_ways_dim_sharded):
      if not allow_uneven_sharding and shape[i] % size != 0:
        raise ValueError(f"One of {what_aval}{name_str} was given the sharding "
                         f"of {s}, which implies that "
                         f"the global size of its dimension {i} should be "
                         f"divisible by {size}, but it is equal to {shape[i]} "
                         f"(full shape: {shape})")


def check_aval_layout_compatibility(
    layouts, flat_avals, names: Sequence[str], what_aval: str):
  for aval, l, name in zip(flat_avals, layouts, names):
    if l is None or isinstance(l, AutoLayout):
      continue
    name_str = f' with pytree key path {name}' if name else ''
    try:
      l.check_compatible_aval(aval.shape)
    except ValueError as e:
      raise ValueError(
          f'One of {what_aval}{name_str} is incompatible with its layout '
          f'annotation {l}: {e}')


# -------------------- pjit rules --------------------

jit_p = core.Primitive("jit")
jit_p.is_effectful = lambda params: bool(params['jaxpr'].effects)  # type: ignore
jit_p.multiple_results = True
jit_p.skip_canonicalization = True

def _is_high(jaxpr, **_) -> bool:
  return jaxpr.jaxpr.is_high
jit_p.is_high = _is_high  # type: ignore

def _to_lojax(*hi_args, jaxpr, **params):
  # convert closed-over boxes to explicit args
  jaxpr, closed_over_himutables = pe.convert_const_himutables(jaxpr)
  hi_args = [*closed_over_himutables, *hi_args]
  params = _converted_mutables_add_params(len(closed_over_himutables), **params)

  # expand pjit params that must match number of lo inputs/outputs
  lo_nums_in = [len(aval.lo_ty()) for aval in jaxpr.in_aval_qdds]
  lo_nums_out = [len(t.lo_ty()) for t in jaxpr.out_avals]
  lo_muts_out = sum(len(aval.lo_ty()) for aval in jaxpr.final_aval_qdds if aval.has_qdd)
  params = _lojax_expand_params(lo_nums_in, lo_nums_out, lo_muts_out, **params)

  # collect lo input values
  lo_args = [lo_val for aval, x in zip(jaxpr.in_aval_qdds, hi_args)
             for lo_val in (aval.read_loval(x) if aval.has_qdd
                            else aval.lower_val(x))]

  # lower the jaxpr and bind it using lo input values
  lo_jaxpr = pe.lower_jaxpr(jaxpr)
  all_outs = jit_p.bind(*lo_args, jaxpr=lo_jaxpr, **params)
  out_mut, lo_outs = split_list(all_outs, [lo_muts_out])

  # collect and apply mutations
  out_mut_ = iter(out_mut)
  in_idx = {v: i for i, v in enumerate(jaxpr.jaxpr.invars)}
  for v in jaxpr.jaxpr.invars:
    if v.final_qdd is not None:
      qdd = v.final_qdd
      lo_vals = it.islice(out_mut_, len(v.aval.lo_ty_qdd(qdd)))
      v.aval.update_from_loval(qdd, hi_args[in_idx[v]], *lo_vals)
  assert next(out_mut_, None) is None

  # collect output values into hi types
  lo_outs_ = iter(lo_outs)
  hi_outs = [t.raise_val(*it.islice(lo_outs_, len(t.lo_ty())))
             for t in jaxpr.out_avals]
  assert next(lo_outs_, None) is None

  return hi_outs
jit_p.to_lojax = _to_lojax

def _converted_mutables_add_params(
    n, *, donated_invars, in_shardings, in_layouts, **params):
  donated_invars = (False,) * n + donated_invars
  in_shardings = (UNSPECIFIED,) * n + in_shardings
  in_layouts = (None,) * n + in_layouts
  return dict(params, donated_invars=donated_invars, in_shardings=in_shardings,
              in_layouts=in_layouts)


def _lojax_expand_params(
    nums_in, nums_out, muts_out, *, donated_invars, in_shardings, in_layouts,
    out_shardings, out_layouts, **params):
  # some pjit params match the length of hi_jaxpr.invars/outvars, so when
  # lowering we must expand them to match their number of lojax types
  def expand(ns, xs):
    return tuple(y for n, x in zip(ns, xs) for y in (x,) * n)
  donated_invars = expand(nums_in , donated_invars)
  in_shardings   = expand(nums_in , in_shardings  )
  in_layouts     = expand(nums_in , in_layouts    )
  out_shardings  = expand(nums_out, out_shardings )
  out_layouts    = expand(nums_out, out_layouts   )

  # also, the lo_jaxpr has pure outputs corresponding to mutable hi_jaxpr types
  out_shardings = (UNSPECIFIED,) * muts_out + out_shardings
  out_layouts = (None,) * muts_out + out_layouts

  new_params = dict(params, donated_invars=donated_invars,
                    in_shardings=in_shardings, in_layouts=in_layouts,
                    out_shardings=out_shardings, out_layouts=out_layouts)
  return new_params

def _resolve_in_layouts(args, jit_in_layouts, resolved_in_shardings,
                        in_avals) -> Sequence[Layout | AutoLayout | None]:
  # If device or backend is set, return the default layout. This is because you
  # can pass arrays on cpu (with untiled layouts) to jit with backend='tpu'
  # which causes error checks to fail. Returning the default layout allows
  # this to exist. It's the same for handling shardings.
  if pxla.check_device_backend_on_shardings(resolved_in_shardings):
    return (None,) * len(jit_in_layouts)

  resolved_in_layouts: list[Layout | AutoLayout | None] = []
  for arg, jit_in_l, rs, aval in safe_zip(
      args, jit_in_layouts, resolved_in_shardings, in_avals):
    committed = getattr(arg, '_committed', True)
    # `arg_layout` is only used for checking purposes in the `else` branch
    # below. We cannot replace default layout with None to raise nicer errors.
    # `dispatch_arg_layout` replaces default layouts with `None` to simplify
    # dispatch and lowering logic downstream.
    if hasattr(arg, 'format'):
      arg_layout = arg.format.layout
      dispatch_arg_layout = (None if pxla.is_default_layout(arg_layout, rs, aval)
                             else arg_layout)
    else:
      arg_layout, dispatch_arg_layout = None, None
    # Sharding can be unspecified when array is committed if it's a PmapSharding.
    is_pmap_sharding = (isinstance(rs, UnspecifiedValue) or
                        isinstance(getattr(arg, 'sharding', None), PmapSharding))
    if jit_in_l is None:
      if committed:
        if is_pmap_sharding:
          resolved_in_layouts.append(None)
        else:
          resolved_in_layouts.append(dispatch_arg_layout)
      else:
        resolved_in_layouts.append(None)
    else:
      # arg_layout can be None because some backends don't implement the
      # required layout methods. Hence `arr.format` can return
      # `Format(None, sharding)`
      if (committed
          and not is_pmap_sharding
          and arg_layout is not None
          and not pxla.is_user_xla_layout_equal(jit_in_l, arg_layout)):
        extra_msg = ''
        if isinstance(jit_in_l, AutoLayout):
          extra_msg = (
              ' The layout given to `jax.jit` is `Layout.AUTO` but'
              ' the corresponding argument passed is a `jax.Array` with a'
              ' concrete layout. Consider passing a `jax.ShapeDtypeStruct`'
              ' instead of `jax.Array` as an argument to the jitted function '
              ' when using `Layout.AUTO`.'
          )
        raise ValueError('Layout passed to jit does not match the layout '
                          'on the respective arg. '
                          f'Got pjit layout: {jit_in_l},\n'
                          f'arg layout: {arg_layout} for '
                          f'arg shape: {core.shaped_abstractify(arg).str_short()}.'
                          f'{extra_msg}')
      jit_in_l = (None if isinstance(jit_in_l, Layout) and
                  pxla.is_default_layout(jit_in_l, rs, aval) else jit_in_l)
      resolved_in_layouts.append(jit_in_l)
  return tuple(resolved_in_layouts)

def _resolve_out_layouts(out_layouts, out_shardings, out_avals):
  new_out_layouts = []
  for out_l, out_s, out_aval in safe_zip(out_layouts, out_shardings, out_avals):
    if out_l is None:
      new_out_layouts.append(None)
    elif (isinstance(out_l, Layout) and
          pxla.is_default_layout(out_l, out_s, out_aval)):
      new_out_layouts.append(None)
    else:
      new_out_layouts.append(out_l)
  return tuple(new_out_layouts)

def finalize_arg_sharding(arg_s, committed):
  if isinstance(arg_s, UnspecifiedValue):
    return arg_s
  else:
    if committed:
      # If the arg has a PmapSharding, then reshard it unconditionally.
      return UNSPECIFIED if isinstance(arg_s, PmapSharding) else arg_s
    else:
      assert isinstance(arg_s, Sharding)
      if dispatch.is_single_device_sharding(arg_s):
        return UNSPECIFIED
      raise NotImplementedError('Having uncommitted Array sharded on '
                                'multiple devices is not supported.')

def _resolve_in_shardings(args, pjit_in_shardings: Sequence[PjitSharding]
                          ) -> Sequence[PjitSharding]:
  # If True, means that device or backend is set by the user on pjit and it
  # has the same semantics as device_put i.e. doesn't matter which device the
  # arg is on, reshard it to the device mentioned. So don't do any of the
  # checks and just return the pjit_in_shardings directly. `shard_args` will
  # handle the resharding.
  if pxla.check_device_backend_on_shardings(pjit_in_shardings):
    return pjit_in_shardings

  committed_arg_shardings = []
  for a in args:
    arg_s = getattr(a, 'sharding', None)
    # arg sharding can be None in case of ShapeDtypeStruct. jax.Array does
    # not allow None as the sharding.
    if arg_s is None:
      continue
    # Don't consider PmapSharding inputs as committed. They will get resharded
    # unconditionally.
    if isinstance(arg_s, PmapSharding):
      continue
    if getattr(a, '_committed', True):
      committed_arg_shardings.append((arg_s, stages.MismatchType.ARG_SHARDING, None))

  resolved_in_shardings: list[PjitSharding] = []
  for arg, pjit_in_s in zip(args, pjit_in_shardings):
    # arg sharding can be None in case of ShapeDtypeStruct. jax.Array does
    # not allow None as the sharding.
    arg_s, committed = ((arg.sharding, getattr(arg, '_committed', True))
                        if hasattr(arg, 'sharding') and arg.sharding is not None
                        else (UNSPECIFIED, False))
    if isinstance(arg_s, NamedSharding) and arg_s.mesh.empty:
      arg_s, committed = UNSPECIFIED, False
    if isinstance(pjit_in_s, UnspecifiedValue):
      resolved_in_shardings.append(finalize_arg_sharding(arg_s, committed))
    else:
      if (isinstance(arg, np.ndarray) and
          not pjit_in_s.is_fully_replicated and  # type: ignore[union-attr]
          xb.process_count() > 1):
        raise ValueError(
            'Passing non-trivial shardings for numpy '
            'inputs is not allowed. To fix this error, either specify a '
            'replicated sharding explicitly or use '
            '`jax.make_array_from_process_local_data(...)` '
            'to convert your host local numpy inputs to a jax.Array which you '
            'can pass to jit. '
            'If the numpy input is the same on each process, then you can use '
            '`jax.make_array_from_callback(...) to create a `jax.Array` which '
            'you can pass to jit. '
            f'Got arg shape: {arg.shape}, arg value: {arg}')
      if not isinstance(arg_s, UnspecifiedValue) and arg_s._is_concrete:
        # jax.jit does not allow resharding across different memory kinds even
        # if the argument is uncommitted. Use jax.device_put for those cases,
        # either outside or inside jax.jit.
        if pjit_in_s.memory_kind != arg_s.memory_kind:  # type: ignore[union-attr]
          raise ValueError(
              'Memory kinds passed to jax.jit does not match memory kind on the'
              f' respective arg. Got pjit memory kind: {pjit_in_s.memory_kind}, '  # type: ignore[union-attr]
              f'arg memory kind: {arg_s.memory_kind} for '
              f'arg shape: {core.shaped_abstractify(arg).str_short()}')
        if (committed and
            not isinstance(arg_s, PmapSharding) and
            not op_shardings.are_op_shardings_equal(
                pjit_in_s._to_xla_hlo_sharding(arg.ndim),  # type: ignore[union-attr]
                arg_s._to_xla_hlo_sharding(arg.ndim))):
          raise ValueError('Sharding passed to pjit does not match the sharding '
                           'on the respective arg. '
                           f'Got pjit sharding: {pjit_in_s},\n'
                           f'arg sharding: {arg_s} for '
                           f'arg shape: {core.shaped_abstractify(arg).str_short()}')
      resolved_in_shardings.append(pjit_in_s)

  return tuple(resolved_in_shardings)


def _resolve_and_lower(
    args, jaxpr: core.ClosedJaxpr, in_shardings, out_shardings, in_layouts,
    out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline,
    lowering_platforms, lowering_parameters, pgle_profiler,
    compiler_options_kvs) -> pxla.MeshComputation:
  in_shardings = _resolve_in_shardings(args, in_shardings)
  in_layouts = _resolve_in_layouts(args, in_layouts, in_shardings,
                                   jaxpr.in_avals)
  out_layouts = _resolve_out_layouts(out_layouts, out_shardings, jaxpr.out_avals)
  return _pjit_lower(
      jaxpr, in_shardings, out_shardings, in_layouts, out_layouts,
      donated_invars, ctx_mesh, name, keep_unused, inline, compiler_options_kvs,
      lowering_platforms=lowering_platforms,
      lowering_parameters=lowering_parameters,
      pgle_profiler=pgle_profiler)

_pgle_profiler_dict = weakref.WeakKeyDictionary()  # type: ignore

def _pjit_call_impl_python(
    *args,
    jaxpr: core.ClosedJaxpr,
    in_shardings, out_shardings, in_layouts, out_layouts,
    donated_invars, ctx_mesh, name, keep_unused, inline,
    compiler_options_kvs):
  util.test_event("jit_cpp_cache_miss")
  pgle_compile_options, pgle_profiler = {}, None
  if config.enable_pgle.value and config.pgle_profiling_runs.value > 0:
    compilation_target_key = jaxpr
    pgle_profiler = _pgle_profiler_dict.get(compilation_target_key)
    if pgle_profiler is None:
      pgle_profiler = profiler.PGLEProfiler(
          config.pgle_profiling_runs.value,
          config.pgle_aggregation_percentile.value)
      _pgle_profiler_dict[compilation_target_key] = pgle_profiler

    # The method below will return FDO profile when module was profiled
    # config.jax_pgle_profiling_runs amount of times, otherwise the result will
    # be None.
    fdo_profile = pgle_profiler.consume_fdo_profile()
    if fdo_profile is not None:
      pgle_compile_options['fdo_profile'] = fdo_profile

  compiler_options_kvs = compiler_options_kvs + tuple(pgle_compile_options.items())
  # Passing mutable PGLE profile here since it should be extracted by JAXPR to
  # initialize the fdo_profile compile option.
  computation = _resolve_and_lower(
      args, jaxpr=jaxpr, in_shardings=in_shardings,
      out_shardings=out_shardings, in_layouts=in_layouts,
      out_layouts=out_layouts, donated_invars=donated_invars,
      ctx_mesh=ctx_mesh, name=name, keep_unused=keep_unused,
      inline=inline, lowering_platforms=None,
      lowering_parameters=mlir.LoweringParameters(),
      pgle_profiler=pgle_profiler,
      compiler_options_kvs=compiler_options_kvs,
  )
  compiled = computation.compile()

  # This check is expensive so only do it if enable_checks is on.
  if compiled._auto_spmd_lowering and config.enable_checks.value:
    pxla.check_array_xla_sharding_layout_match(
        args, compiled._in_shardings, compiled._in_layouts,  # type: ignore
        jaxpr.jaxpr._debug_info, compiled._kept_var_idx)
  if config.distributed_debug.value:
    # Defensively only perform fingerprint logic if debug logging is enabled
    # NOTE(skyewm): I didn't benchmark this
    fingerprint = None
    if hasattr(compiled.runtime_executable(), "fingerprint"):
      fingerprint = compiled.runtime_executable().fingerprint
    if fingerprint is not None:
      fingerprint = fingerprint.hex()
    distributed_debug_log(("Running pjit'd function", name),
                          ("in_shardings", in_shardings),
                          ("out_shardings", out_shardings),
                          ("in_layouts", in_layouts),
                          ("out_layouts", out_layouts),
                          ("abstract args", map(core.abstractify, args)),
                          ("fingerprint", fingerprint))
  return (compiled.unsafe_call(*computation.const_args, *args),
          compiled, pgle_profiler, computation.const_args)

@weakref_lru_cache
def _get_jaxpr_as_fun(jaxpr, in_shardings, out_shardings, in_layouts,
                      out_layouts, donated_invars, ctx_mesh, name,
                      keep_unused, inline, compiler_options_kvs):
  # The input jaxpr to `_get_jaxpr_as_fun` is under a weakref_lru_cache so
  # returning `core.jaxpr_as_fun(jaxpr)` directly creates a strong reference to
  # the jaxpr defeating the purpose of weakref_lru_cache. So return a function
  # that closes over a weakrefed jaxpr and gets called inside that function.
  # This way there won't be a strong reference to the jaxpr from the output
  # function.
  jaxpr = weakref.ref(jaxpr)
  return lambda *args: core.jaxpr_as_fun(jaxpr())(*args)  # pylint: disable=unnecessary-lambda


def _pjit_call_impl(*args, jaxpr: core.ClosedJaxpr,
                    in_shardings, out_shardings, in_layouts, out_layouts,
                    donated_invars, ctx_mesh, name, keep_unused, inline,
                    compiler_options_kvs):
  def call_impl_cache_miss(*args_, **kwargs_):
    # args_ do not include the const args
    # See https://docs.jax.dev/en/latest/internals/constants.html.
    # TODO(necula): remove num_const_args when fixing the C++ path
    out_flat, compiled, pgle_profiler, const_args = _pjit_call_impl_python(
        *args, jaxpr=jaxpr, in_shardings=in_shardings,
        out_shardings=out_shardings, in_layouts=in_layouts,
        out_layouts=out_layouts, donated_invars=donated_invars,
        ctx_mesh=ctx_mesh, name=name, keep_unused=keep_unused,
        inline=inline, compiler_options_kvs=compiler_options_kvs)
    fastpath_data = _get_fastpath_data(
        compiled, tree_structure(out_flat), args, out_flat,
        jaxpr.effects, jaxpr.consts, None, pgle_profiler,
        const_args)
    return out_flat, fastpath_data, _need_to_rebuild_with_fdo(pgle_profiler)

  f = _get_jaxpr_as_fun(
      jaxpr, in_shardings, out_shardings, in_layouts, out_layouts,
      donated_invars, ctx_mesh, name, keep_unused, inline,
      compiler_options_kvs)
  donated_argnums = tuple(i for i, d in enumerate(donated_invars) if d)
  cache_key = pxla.JitGlobalCppCacheKeys(
      donate_argnums=donated_argnums, donate_argnames=None,
      device=None, backend=None,
      in_shardings_treedef=None, in_shardings_leaves=in_shardings,
      out_shardings_treedef=None, out_shardings_leaves=out_shardings,
      in_layouts_treedef=None, in_layouts_leaves=in_layouts,
      out_layouts_treedef=None, out_layouts_leaves=out_layouts)
  return xc._xla.pjit(
      name, f, call_impl_cache_miss, [], [], cache_key,
      tree_util.dispatch_registry, pxla.cc_shard_arg,
      _get_cpp_global_cache(cache_key.contains_explicit_attributes))(*args)

jit_p.def_impl(_pjit_call_impl)

# This cache is important for python dispatch performance.
@weakref_lru_cache
def _pjit_lower(
    jaxpr: core.ClosedJaxpr,
    in_shardings,
    out_shardings,
    in_layouts: pxla.MaybeLayout,
    out_layouts: pxla.MaybeLayout,
    donated_invars,
    ctx_mesh,
    name: str,
    keep_unused: bool,
    inline: bool,
    compiler_options_kvs: tuple[tuple[str, Any], ...],
    *,
    lowering_platforms: tuple[str, ...] | None,
    lowering_parameters: mlir.LoweringParameters,
    pgle_profiler: profiler.PGLEProfiler | None) -> pxla.MeshComputation:
  return pxla.lower_sharding_computation(
      jaxpr, 'jit', name, in_shardings, out_shardings,
      in_layouts, out_layouts, tuple(donated_invars),
      keep_unused=keep_unused, context_mesh=ctx_mesh,
      compiler_options_kvs=compiler_options_kvs,
      lowering_platforms=lowering_platforms,
      lowering_parameters=lowering_parameters,
      pgle_profiler=pgle_profiler)


def pjit_staging_rule(trace, source_info, *args, **params):
  if params["compiler_options_kvs"]:
    raise ValueError(
        '`compiler_options` can only be passed to top-level `jax.jit`. Got'
        f' compiler_options={dict(params["compiler_options_kvs"])} specified on'
        f' a nested jit with name: {params["name"]} and source info:'
        f' {source_info_util.summarize(source_info)}')
  # If we're inlining, no need to compute forwarding information; the inlined
  # computation will in effect forward things.
  if (params["inline"] and
      all(isinstance(i, UnspecifiedValue) for i in params["in_shardings"]) and
      all(isinstance(o, UnspecifiedValue) for o in params["out_shardings"]) and
      all(i is None for i in params["in_layouts"]) and
      all(o is None for o in params["out_layouts"])):
    jaxpr = params["jaxpr"]
    if config.dynamic_shapes.value:
      # Inline jaxpr doesn't handle dynamic shapes when inlining. If dynamic
      # shapes are enabled, use eval_jaxpr, which uses the tracing machinery,
      # but redundantly performs abstract evaluation again.
      with core.set_current_trace(trace):
        out = core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args,
                              propagate_source_info=False)
    else:
      out = pe.inline_jaxpr_into_trace(
          trace, source_info, jaxpr.jaxpr, jaxpr.consts, *args)
    return [trace.to_jaxpr_tracer(x, source_info) for x in out]

  jaxpr = params['jaxpr']
  if config.dynamic_shapes.value:
    jaxpr, in_fwd, out_shardings, out_layouts = _pjit_forwarding(
        jaxpr, params['out_shardings'], params['out_layouts'])
    params = dict(params, jaxpr=jaxpr, out_shardings=out_shardings,
                  out_layouts=out_layouts)
    outvars = map(trace.frame.newvar, _out_type(jaxpr))
    eqn = core.new_jaxpr_eqn(
      [arg.var for arg in args], outvars, jit_p, params,
      jaxpr.effects, source_info)
    trace.frame.add_eqn(eqn)
    out_tracers = [pe.DynamicJaxprTracer(trace, v.aval, v, source_info)
                   for v in outvars]
    out_tracers_ = iter(out_tracers)
    out_tracers = [args[f] if type(f) is int else next(out_tracers_)
                   for f in in_fwd]
    assert next(out_tracers_, None) is None
  elif any(isinstance(c, core.MutableArray) for c in jaxpr.consts):
    jaxpr, consts = pxla._move_mutable_consts(jaxpr)
    consts = [trace.new_const(c, source_info) for c in consts]
    in_shardings = (*params['in_shardings'],) + (UNSPECIFIED,) * len(consts)
    in_layouts = (*params['in_layouts'],) + (None,) * len(consts)
    donated_invars = (*params['donated_invars'],) + (False,) * len(consts)
    new_params = dict(params, jaxpr=jaxpr, in_shardings=in_shardings,
                      in_layouts=in_layouts, donated_invars=donated_invars)
    out_tracers = trace.default_process_primitive(
        jit_p, (*args, *consts), new_params, source_info=source_info)
  else:
    out_tracers = trace.default_process_primitive(
        jit_p, args, params, source_info=source_info)
  return out_tracers
pe.custom_staging_rules[jit_p] = pjit_staging_rule


def _pjit_forwarding(jaxpr, out_shardings, out_layouts):
  in_fwd: list[int | None] = pe._jaxpr_forwarding(jaxpr.jaxpr)
  in_fwd = [fwd if isinstance(os, UnspecifiedValue) and ol is None else None
            for fwd, os, ol in zip(in_fwd, out_shardings, out_layouts)]
  keep = [f is None for f in in_fwd]
  jaxpr = pe.prune_closed_jaxpr_outputs(jaxpr, keep)
  out_shardings = tuple(o for o, k in zip(out_shardings, keep) if k)
  out_layouts   = tuple(o for o, k in zip(out_layouts  , keep) if k)
  return jaxpr, in_fwd, out_shardings, out_layouts

def pjit_forwarding_rule(eqn):
  if not config.dynamic_shapes.value:
    return [None] * len(eqn.outvars), eqn
  jaxpr, in_fwd, out_shardings, out_layouts = _pjit_forwarding(
      eqn.params['jaxpr'], eqn.params['out_shardings'], eqn.params['out_layouts'])
  new_outvars = [v for v, f in zip(eqn.outvars, in_fwd) if f is None]
  new_params = dict(eqn.params, jaxpr=jaxpr, out_shardings=out_shardings,
                    out_layouts=out_layouts)
  new_eqn = eqn.replace(params=new_params, outvars=new_outvars)
  return in_fwd, new_eqn
# TODO(mattjj): Remove pjit_forwarding_rule and also in staging rule.
pe.forwarding_rules[jit_p] = pjit_forwarding_rule


# TODO(mattjj): remove/trivialize this when jaxprs have type annotation on them,
# since it's actually not possible in general to infer the type from the term
def _out_type(jaxpr: core.ClosedJaxpr) -> list[core.AbstractValue]:
  out = []
  in_idx = {v: i for i, v in enumerate(jaxpr.jaxpr.invars)}
  out_idx = {x: i for i, x in enumerate(jaxpr.jaxpr.invars)
             if type(x) is core.Var}
  for x in jaxpr.jaxpr.outvars:
    aval = x.aval
    if type(aval) is core.DShapedArray:
      shape = [core.InDBIdx(in_idx[d]) if d in in_idx else
               core.OutDBIdx(out_idx[d]) if d in out_idx else
               d for d in x.aval.shape]
      aval = aval.update(shape=tuple(shape))
    out.append(aval)
  return out


def _pjit_typecheck(ctx_factory, *in_atoms, jaxpr, **params):
  return core._check_call(ctx_factory, jit_p, in_atoms,
                          dict(params, call_jaxpr=jaxpr.jaxpr))
core.custom_typechecks[jit_p] = _pjit_typecheck


def _pjit_abstract_eval(*args, jaxpr, out_shardings, **_):
  return jaxpr.out_avals, jaxpr.effects
jit_p.def_effectful_abstract_eval(_pjit_abstract_eval)


def _pjit_cached_lower_jaxpr_to_fun(ctx: mlir.LoweringRuleContext,
                                    name: str, jaxpr: core.ClosedJaxpr,
                                    num_const_args: int, in_avals,
                                    effects, in_shardings,
                                    out_shardings, in_layouts, out_layouts,
                                    api_name):
  assert len(in_avals) == num_const_args + len(jaxpr.in_avals)
  assert len(in_avals) == len(in_shardings)
  assert len(in_avals) == len(in_layouts)
  mod_ctx = ctx.module_context
  axis_ctx = ctx.module_context.axis_context
  num_devices = None
  if isinstance(axis_ctx, sharding_impls.ShardingContext):
    num_devices = axis_ctx.num_devices
  elif isinstance(axis_ctx, sharding_impls.SPMDAxisContext):
    num_devices = axis_ctx.mesh.size
  key = (jit_p, name, jaxpr, effects, num_devices,
         pxla.SemanticallyEqualShardings(in_shardings, in_avals),  # pytype: disable=wrong-arg-types
         pxla.SemanticallyEqualShardings(out_shardings, jaxpr.out_avals),  # pytype: disable=wrong-arg-types
         in_layouts, out_layouts, api_name)

  func = mod_ctx.cached_primitive_lowerings.get(key, None)
  if func is None:
    arg_shardings = [None if isinstance(i, UnspecifiedValue) else i for i in in_shardings]
    result_shardings = [None if isinstance(o, UnspecifiedValue) else o for o in out_shardings]
    # TODO(b/228598865): non-top-level functions cannot have shardings set
    # directly on the inputs or outputs because they are lost during MLIR->HLO
    # conversion. using_sharding_annotation=False means we add an identity
    # operation instead.
    num_callbacks = len(mod_ctx.host_callbacks)
    func = mlir.lower_jaxpr_to_fun(
        mod_ctx, name, jaxpr, effects,
        num_const_args=num_const_args, in_avals=in_avals,
        arg_shardings=arg_shardings, result_shardings=result_shardings,
        use_sharding_annotations=False,
        arg_layouts=in_layouts, result_layouts=out_layouts)

    # If this Jaxpr includes callbacks, we can't cache the lowering because
    # on TPU every callback must have a globally unique channel, but the
    # channel gets assigned during lowering.
    has_callbacks = len(mod_ctx.host_callbacks) > num_callbacks
    if not has_callbacks or "tpu" not in mod_ctx.platforms:
      mod_ctx.cached_primitive_lowerings[key] = func
  return func


def _pjit_lowering(ctx: mlir.LoweringRuleContext, *args, name: str,
                   jaxpr: core.ClosedJaxpr, in_shardings,
                   out_shardings, in_layouts, out_layouts, donated_invars,
                   ctx_mesh, keep_unused, inline, compiler_options_kvs):
  effects = list(ctx.tokens_in.effects())
  output_types = map(mlir.aval_to_ir_type, ctx.avals_out)
  output_types = [mlir.token_type()] * len(effects) + output_types
  flat_output_types = mlir.flatten_ir_types(output_types)

  const_args = core.jaxpr_const_args(jaxpr.jaxpr)
  const_arg_avals = [core.shaped_abstractify(c) for c in const_args]
  in_avals = const_arg_avals + jaxpr.in_avals
  ca_shardings = const_args_shardings(const_args)
  in_shardings = ca_shardings + in_shardings  # type: ignore
  ca_layouts = const_args_layouts(const_args, const_arg_avals, ca_shardings)
  in_layouts = ca_layouts + in_layouts  # type: ignore

  func = _pjit_cached_lower_jaxpr_to_fun(
      ctx, name, jaxpr, len(const_args), in_avals,
      tuple(effects), in_shardings,
      out_shardings, in_layouts, out_layouts,
      api_name='jit')

  tokens_in = [ctx.tokens_in.get(eff) for eff in effects]
  hoisted_const_values = [
      mlir.ir_constant(c, ctx.const_lowering, canonicalize_dtype=True)
      for c in const_args]
  args = (*ctx.dim_var_values, *tokens_in, *hoisted_const_values, *args)
  with mlir.source_info_to_location(
      ctx.module_context, None,
      ctx.name_stack.extend(util.wrap_name('jit', name)),
      ctx.traceback
  ):
    call = func_dialect.CallOp(flat_output_types,
                              ir.FlatSymbolRefAttr.get(func.name.value),
                              mlir.flatten_ir_values(args))
  mlir.wrap_compute_type_in_place(ctx, call)
  out_nodes = mlir.unflatten_ir_values_like_types(call.results, output_types)
  tokens, out_nodes = split_list(out_nodes, [len(effects)])
  tokens_out = ctx.tokens_in.update_tokens(mlir.TokenSet(zip(effects, tokens)))
  ctx.set_tokens_out(tokens_out)
  return out_nodes

# TODO(phawkins): this is marked uncacheable because it has its own cache and
# because the cache breaks jaxpr metadata like source locations. We should fix
# the metadata problem and consolidate the caches.
mlir.register_lowering(jit_p, _pjit_lowering, cacheable=False)

def const_args_shardings(const_args: Sequence[ArrayLike]) -> Sequence[PjitSharding]:
  return _resolve_in_shardings(
      const_args, (sharding_impls.UNSPECIFIED,) * len(const_args))

def const_args_layouts(
    const_args: Sequence[ArrayLike],
    avals: Sequence[core.AbstractValue],
    shardings: Sequence[PjitSharding]
    ) -> Sequence[Layout | AutoLayout | None]:
  return _resolve_in_layouts(
      const_args, (None,) * len(const_args), shardings, avals)

def _pjit_batcher(axis_data, vals_in,
                  dims_in: tuple[int, ...],
                  jaxpr: core.ClosedJaxpr,
                  in_shardings, out_shardings, in_layouts, out_layouts,
                  donated_invars, ctx_mesh, name, keep_unused, inline,
                  compiler_options_kvs):
  segment_lens, dims_in = batching.indirectify_ragged_axes(dims_in)
  new_jaxpr, axes_out = batching.batch_jaxpr2(jaxpr, axis_data, dims_in)

  # TODO(axch): prepend with Nones (?) to account for new segment_lens inputs
  in_shardings = tuple(
      _pjit_batcher_for_sharding(i, axis_in, axis_data.spmd_name, ctx_mesh,
                                 aval.ndim)
      if axis_in is not None else i
      for axis_in, i, aval in zip(dims_in, in_shardings, new_jaxpr.in_avals))
  out_shardings = tuple(
      _pjit_batcher_for_sharding(o, axis_out, axis_data.spmd_name, ctx_mesh,
                                 aval.ndim)
      if axis_out is not None else o
      for axis_out, o, aval in zip(axes_out, out_shardings, new_jaxpr.out_avals))
  # TODO(yashkatariya): Figure out layouts should change under vmap.
  if not (all(l is None for l in in_layouts) and
          all(l is None for l in out_layouts)):
    raise NotImplementedError(
        'Concrete layouts are not supported for vmap(jit).')

  vals_out = jit_p.bind(
    *vals_in,
    jaxpr=new_jaxpr,
    in_shardings=in_shardings,
    out_shardings=out_shardings,
    in_layouts=in_layouts,
    out_layouts=out_layouts,
    donated_invars=donated_invars,
    ctx_mesh=ctx_mesh,
    name=name,
    keep_unused=keep_unused,
    inline=inline,
    compiler_options_kvs=compiler_options_kvs)

  resolved_axes_out = batching.resolve_ragged_axes_against_inputs_outputs(
      vals_in, vals_out, axes_out)
  return vals_out, resolved_axes_out

batching.fancy_primitive_batchers[jit_p] = _pjit_batcher
batching.ragged_prop_rules[jit_p] = batching.ragged_mask_no_op_rule


def _pjit_batcher_for_sharding(
    s, dim: int | batching.RaggedAxis, spmd_axis_name: tuple[str, ...] | None,
    mesh, ndim: int):
  if isinstance(s, UnspecifiedValue):
    return s
  hlo_s = s._to_xla_hlo_sharding(ndim)
  if spmd_axis_name is None:
    if sharding_impls.is_op_sharding_replicated(hlo_s):
      return s
    if isinstance(s, NamedSharding) and isinstance(s.mesh, AbstractMesh):
      return NamedSharding(
          s.mesh, pxla.batch_spec(s.spec, dim, PartitionSpec.UNCONSTRAINED))
    new_op = hlo_s.to_proto().clone()
    tad = list(new_op.tile_assignment_dimensions)
    tad.insert(dim, 1)  # type: ignore
    new_op.tile_assignment_dimensions = tad
    new_gs = GSPMDSharding(s._internal_device_list, new_op)
    return pxla._get_out_sharding_from_orig_sharding([new_gs], [None], s, None)[0]
  else:
    if isinstance(s, NamedSharding) and isinstance(s.mesh, AbstractMesh):
      return NamedSharding(
          s.mesh, pxla.batch_spec(s.spec, dim, spmd_axis_name))
    if isinstance(s, NamedSharding):
      mesh = s.mesh
    if mesh.empty:
      raise ValueError(
          'If you are using spmd_axis_name parameter of jax.vmap,'
          ' please make sure to run your jitted function inside the mesh'
          ' context manager. Only `jax.lax.with_sharding_constraint` with'
          ' `jax.sharding.NamedSharding` as an input can be transformed with'
          ' spmd_axis_name batching rules outside of an explicit mesh context'
          f' manager scope{s!r}')
    spec = parse_flatten_op_sharding(hlo_s, mesh)[0]
    return NamedSharding(
        mesh, pxla.batch_spec(spec, dim, spmd_axis_name))


def _pjit_jvp(primals_in, tangents_in,
              jaxpr, in_shardings, out_shardings, in_layouts, out_layouts,
              donated_invars, ctx_mesh, name, keep_unused, inline,
              compiler_options_kvs):
  is_nz_tangents_in = [type(t) is not ad.Zero for t in tangents_in]
  jaxpr_jvp, is_nz_tangents_out = ad.jvp_jaxpr(
      jaxpr, is_nz_tangents_in, instantiate=False)

  def _filter_zeros(is_nz_l, l):
    return (x for nz, x in zip(is_nz_l, l) if nz)
  _filter_zeros_in = partial(_filter_zeros, is_nz_tangents_in)
  _filter_zeros_out = partial(_filter_zeros, is_nz_tangents_out)
  outputs = jit_p.bind(
      *primals_in, *_filter_zeros_in(tangents_in),
      jaxpr=jaxpr_jvp,
      in_shardings=(*in_shardings, *_filter_zeros_in(in_shardings)),
      out_shardings=(*out_shardings, *_filter_zeros_out(out_shardings)),
      in_layouts=(*in_layouts, *_filter_zeros_in(in_layouts)),
      out_layouts=(*out_layouts, *_filter_zeros_out(out_layouts)),
      donated_invars=(*donated_invars, *_filter_zeros_in(donated_invars)),
      ctx_mesh=ctx_mesh,
      name=name,
      keep_unused=keep_unused,
      inline=inline,
      compiler_options_kvs=compiler_options_kvs)

  primals_out, tangents_out = split_list(outputs, [len(jaxpr.jaxpr.outvars)])
  assert len(primals_out) == len(jaxpr.jaxpr.outvars)
  tangents_out_it = iter(tangents_out)
  return primals_out, [next(tangents_out_it) if nz else ad.Zero(aval)
                       for nz, aval in zip(is_nz_tangents_out, jaxpr.out_avals)]
ad.primitive_jvps[jit_p] = _pjit_jvp


def _pjit_linearize(nzs, *primals_in, jaxpr, in_shardings, out_shardings,
                    in_layouts, out_layouts, donated_invars, ctx_mesh, name,
                    keep_unused, inline, compiler_options_kvs):
  primal_jaxpr, num_residuals_out, nzs_out, in_fwd_res, tangent_jaxpr = \
      ad.linearize_jaxpr(jaxpr, nzs)
  num_residuals_in = len(in_fwd_res)
  num_primals_out = len(primal_jaxpr.out_avals) - num_residuals_out

  res_shardings_in = (UNSPECIFIED,) * num_residuals_in
  res_layouts_in = (None,) * num_residuals_in
  res_donated = (False,) * num_residuals_in
  primal_out_shardings = tuple(out_shardings) + (UNSPECIFIED,) * num_residuals_out
  primal_out_layouts = tuple(out_layouts) + (None,) * num_residuals_out

  config.enable_checks.value and core.check_jaxpr(primal_jaxpr.jaxpr)
  config.enable_checks.value and core.check_jaxpr(tangent_jaxpr.jaxpr)

  def keep_where(l, should_keep):
    return tuple(x for x, keep in zip(l, should_keep) if keep)

  # Input-to-output forwarding.
  in_fwd = pe._jaxpr_forwarding(primal_jaxpr.jaxpr)
  in_fwd_primal, in_fwd_res_ = split_list(in_fwd, [num_primals_out])
  assert all(f is None for f in in_fwd_res_)
  in_fwd = [
      fwd if isinstance(os, UnspecifiedValue) and ol is None else None
      for os, ol, fwd in zip(out_shardings, out_layouts, in_fwd_primal)
  ] + in_fwd_res_
  del in_fwd_res_, in_fwd_primal
  keep = [f is None for f in in_fwd]
  primal_jaxpr = pe.prune_closed_jaxpr_outputs(primal_jaxpr, keep)
  primal_out_shardings = keep_where(primal_out_shardings, keep)
  primal_out_layouts = keep_where(primal_out_layouts, keep)
  _, kept_res = split_list(keep, [num_primals_out])
  num_kept_residuals = sum(kept_res)
  del keep, kept_res, num_primals_out

  # Output-to-output forwarding.
  num_primals_out = len(primal_jaxpr.out_avals) - num_kept_residuals
  out_vars, res_vars = split_list(primal_jaxpr.jaxpr.outvars, [num_primals_out])
  idx_map = {id(v): i for i, v in enumerate(out_vars)}
  out_fwd = [None] * num_primals_out + [idx_map.get(id(v)) for v in res_vars]
  keep = [f is None for f in out_fwd]
  primal_jaxpr = pe.prune_closed_jaxpr_outputs(primal_jaxpr, keep)
  primal_out_shardings = keep_where(primal_out_shardings, keep)
  primal_out_layouts = keep_where(primal_out_layouts, keep)
  del keep

  def tangent_fun(residuals, *tangents):
    tangents_nz = _filter_zeros(nzs, tangents)
    nz_tangents_out = jit_p.bind(
        *residuals, *tangents_nz, jaxpr=tangent_jaxpr,
        in_shardings=res_shardings_in + _filter_zeros(nzs, in_shardings),
        out_shardings=_filter_zeros(nzs_out, out_shardings),
        in_layouts=res_layouts_in + _filter_zeros(nzs, in_layouts),
        out_layouts=_filter_zeros(nzs_out, out_layouts),
        donated_invars=res_donated + _filter_zeros(nzs, donated_invars),
        ctx_mesh=ctx_mesh,
        name=name,
        keep_unused=keep_unused,
        inline=inline,
        compiler_options_kvs=compiler_options_kvs)
    tangent_avals_out = [v.aval.to_tangent_aval() for v in jaxpr.jaxpr.outvars]
    nz_tangents_out_ = iter(nz_tangents_out)
    tangents_out = [next(nz_tangents_out_) if nz else ad.Zero(aval)
                   for (aval, nz) in zip(tangent_avals_out, nzs_out)]
    return tangents_out

  def _filter_zeros(is_nz_l, l):
    return tuple(x for nz, x in zip(is_nz_l, l) if nz)

  assert len(in_shardings) == len(primal_jaxpr.in_avals)
  ans = jit_p.bind(*primals_in, jaxpr=primal_jaxpr,
                   in_shardings=in_shardings,
                   out_shardings=primal_out_shardings,
                   in_layouts=in_layouts,
                   out_layouts=primal_out_layouts,
                   donated_invars=donated_invars,
                   ctx_mesh=ctx_mesh,
                   name=name,
                   keep_unused=keep_unused,
                   inline=inline,
                   compiler_options_kvs=compiler_options_kvs)
  ans = subs_list(out_fwd, ans, ans)
  ans = subs_list(in_fwd, primals_in, ans)
  primal_ans, residuals_ans = split_list(ans, [len(ans) - num_residuals_out])
  residuals_ans = subs_list(in_fwd_res, [*jaxpr.consts, *primals_in], residuals_ans)

  return primal_ans, nzs_out, residuals_ans, tangent_fun

ad.primitive_linearizations[jit_p] = _pjit_linearize


def _pjit_partial_eval(trace: pe.JaxprTrace,
                       *in_tracers,
                       jaxpr: core.ClosedJaxpr, in_shardings, out_shardings,
                       in_layouts, out_layouts, donated_invars, ctx_mesh,
                       name, keep_unused, inline, compiler_options_kvs):
  in_pvals = [t.pval for t in in_tracers]

  known_ins = tuple(pv.is_known() for pv in in_pvals)
  unknown_ins = tuple(not k for k in known_ins)
  known_jaxpr, unknown_jaxpr, unknown_outs, res_out_avals, in_fwd_res = \
      pe.partial_eval_jaxpr_nounits_fwd(jaxpr, unknown_ins, instantiate=False)
  unknown_outs = tuple(unknown_outs)  # type: ignore[assignment]
  known_outs = tuple(not uk for uk in unknown_outs)

  # out_shardings and out_layouts for residual values output by known_jaxpr
  def keep_where(l, should_keep):
    return tuple(x for x, keep in zip(l, should_keep) if keep)

  known_out_shardings = (keep_where(out_shardings, known_outs)
                         + (UNSPECIFIED,) * len(res_out_avals))
  known_out_layouts = (keep_where(out_layouts, known_outs)
                       + (None,) * len(res_out_avals))

  # Input-to-output forwarding: compute which outputs are just forwarded inputs.
  num_out_primals = len(known_jaxpr.out_avals) - len(res_out_avals)
  in_fwd: list[int | None] = pe._jaxpr_forwarding(known_jaxpr.jaxpr)
  in_fwd_primal, in_fwd_res_ = split_list(in_fwd, [num_out_primals])
  assert all(f is None for f in in_fwd_res_)
  in_fwd = [
      fwd if isinstance(os, UnspecifiedValue) and ol is None else None
      for os, ol, fwd in zip(
          keep_where(out_shardings, known_outs),
          keep_where(out_layouts, known_outs), in_fwd_primal)
  ] + in_fwd_res_
  del in_fwd_primal, in_fwd_res_
  # Prune jaxpr outputs and out_shardings by removing the input-forwards.
  keep = [f is None for f in in_fwd]
  known_jaxpr = pe.prune_closed_jaxpr_outputs(known_jaxpr, keep)
  known_out_shardings = keep_where(known_out_shardings, keep)
  known_out_layouts = keep_where(known_out_layouts, keep)
  # Update num_out_primals to reflect pruning.
  kept_primals, kept_res = split_list(keep, [num_out_primals])
  num_out_primals = sum(kept_primals)
  del keep, kept_primals, kept_res

  # Output-to-output forwarding: compute which residuals are just primal outputs
  out_vars, res_vars = split_list(known_jaxpr.jaxpr.outvars, [num_out_primals])
  idx_map = {id(v): i for i, v in enumerate(out_vars)}
  out_fwd = [None] * num_out_primals + [idx_map.get(id(v)) for v in res_vars]
  # Prune jaxpr outputs and out_shardings by removing forwarded residuals.
  keep = [f is None for f in out_fwd]
  known_jaxpr = pe.prune_closed_jaxpr_outputs(known_jaxpr, keep)
  known_out_shardings = keep_where(known_out_shardings, keep)
  known_out_layouts = keep_where(known_out_layouts, keep)
  del keep

  known_params = dict(
      jaxpr=known_jaxpr, in_shardings=keep_where(in_shardings, known_ins),
      out_shardings=known_out_shardings,
      in_layouts=keep_where(in_layouts, known_ins),
      out_layouts=known_out_layouts,
      donated_invars=keep_where(donated_invars, known_ins),
      ctx_mesh=ctx_mesh,
      name=name, keep_unused=keep_unused, inline=inline,
      compiler_options_kvs=compiler_options_kvs)
  assert len(known_params['out_shardings']) == len(known_params['jaxpr'].out_avals)
  assert len(known_params['out_layouts']) == len(known_params['jaxpr'].out_avals)

  # Bind known things to pjit_p.
  known_inputs = [pv.get_known() for pv in in_pvals if pv.is_known()]
  all_known_outs = jit_p.bind(*known_inputs, **known_params)
  # Add back in the output fwds.
  all_known_outs = subs_list(out_fwd, all_known_outs, all_known_outs)
  # Add back in the input fwds.
  all_known_outs = subs_list(in_fwd, known_inputs, all_known_outs)

  known_out_vals, residual_vals = \
      split_list(all_known_outs, [len(all_known_outs) - len(res_out_avals)])
  residual_vals_ = iter(residual_vals)
  residual_vals = [next(residual_vals_) if f is None
                   else [*jaxpr.consts, *known_inputs][f] for f in in_fwd_res]
  assert next(residual_vals_, None) is None
  residual_tracers = map(trace.new_instantiated_const, residual_vals)

  # The convention of partial_eval_jaxpr_nounits is to place residual binders at
  # the front of the jaxpr produced, so we move them to the back since both the
  # jaxpr equation built below and the pjit transpose rule assume a
  # residual-inputs-last convention.
  unknown_jaxpr = pe.move_binders_to_back(
      unknown_jaxpr, [True] * len(residual_vals) + [False] * sum(unknown_ins))

  # Set up staged-out 'unknown' eqn
  unknown_in_shardings = (keep_where(in_shardings, unknown_ins)
                          + (UNSPECIFIED,) * len(residual_tracers))
  unknown_in_layouts = (keep_where(in_layouts, unknown_ins)
                        + (None,) * len(residual_tracers))
  unknown_donated_invars = (keep_where(donated_invars, unknown_ins)
                            + (False,) * len(residual_tracers))
  unknown_params = dict(
      jaxpr=unknown_jaxpr,
      in_shardings=unknown_in_shardings,
      in_layouts=unknown_in_layouts,
      out_shardings=keep_where(out_shardings, unknown_outs),
      out_layouts=keep_where(out_layouts, unknown_outs),
      donated_invars=unknown_donated_invars,
      ctx_mesh=ctx_mesh,
      name=name,
      keep_unused=keep_unused,
      inline=inline,
      compiler_options_kvs=compiler_options_kvs)
  unknown_tracers_in = [t for t in in_tracers if not t.pval.is_known()]
  unknown_out_avals = unknown_jaxpr.out_avals
  unknown_tracers_out = [
      pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
      for aval in unknown_out_avals
  ]
  unknown_tracers_in = [*unknown_tracers_in, *residual_tracers]
  eqn = pe.new_eqn_recipe(trace, unknown_tracers_in,
                          unknown_tracers_out,
                          jit_p,
                          unknown_params,
                          unknown_jaxpr.effects,
                          source_info_util.current())
  for t in unknown_tracers_out: t.recipe = eqn
  if effects.partial_eval_kept_effects.filter_in(unknown_jaxpr.effects):
    trace.effect_handles.append(pe.EffectHandle(unknown_tracers_in, eqn))  # type: ignore
  return merge_lists(unknown_outs, known_out_vals, unknown_tracers_out)

pe.custom_partial_eval_rules[jit_p] = _pjit_partial_eval


def _pjit_partial_eval_custom_params_updater(
    unks_in: Sequence[bool], inst_in: Sequence[bool],
    kept_outs_known: Sequence[bool], kept_outs_staged: Sequence[bool],
    num_res_out: int, num_res_in: int, params_known: dict, params_staged: dict
  ) -> tuple[dict, dict]:
  # prune inputs to jaxpr_known according to unks_in
  donated_invars_known, _ = pe.partition_list(unks_in, params_known['donated_invars'])
  in_shardings_known, _ = pe.partition_list(unks_in, params_known['in_shardings'])
  _, out_shardings_known = pe.partition_list(kept_outs_known, params_known['out_shardings'])
  in_layouts_known, _ = pe.partition_list(unks_in, params_known['in_layouts'])
  _, out_layouts_known = pe.partition_list(kept_outs_known, params_known['out_layouts'])

  new_params_known = dict(params_known,
                          in_shardings=tuple(in_shardings_known),
                          out_shardings=(*out_shardings_known,
                                         *[UNSPECIFIED] * num_res_out),
                          in_layouts=tuple(in_layouts_known),
                          out_layouts=(*out_layouts_known, *[None] * num_res_out),
                          donated_invars=tuple(donated_invars_known))
  assert len(new_params_known['in_shardings']) == len(params_known['jaxpr'].in_avals)
  assert len(new_params_known['out_shardings']) == len(params_known['jaxpr'].out_avals)
  assert len(new_params_known['in_layouts']) == len(params_known['jaxpr'].in_avals)
  assert len(new_params_known['out_layouts']) == len(params_known['jaxpr'].out_avals)

  # added num_res new inputs to jaxpr_staged, and pruning according to inst_in
  _, donated_invars_staged = pe.partition_list(inst_in, params_staged['donated_invars'])
  donated_invars_staged = [False] * num_res_in + donated_invars_staged
  _, in_shardings_staged = pe.partition_list(inst_in, params_staged['in_shardings'])
  in_shardings_staged = [*[UNSPECIFIED] * num_res_in, *in_shardings_staged]
  _, out_shardings_staged = pe.partition_list(kept_outs_staged, params_staged['out_shardings'])
  _, in_layouts_staged = pe.partition_list(inst_in, params_staged['in_layouts'])
  in_layouts_staged = [*[None] * num_res_in, *in_layouts_staged]
  _, out_layouts_staged = pe.partition_list(kept_outs_staged, params_staged['out_layouts'])

  new_params_staged = dict(params_staged,
                           in_shardings=tuple(in_shardings_staged),
                           out_shardings=tuple(out_shardings_staged),
                           in_layouts=tuple(in_layouts_staged),
                           out_layouts=tuple(out_layouts_staged),
                           donated_invars=tuple(donated_invars_staged))
  assert len(new_params_staged['in_shardings']) == len(params_staged['jaxpr'].in_avals)
  assert len(new_params_staged['out_shardings']) == len(params_staged['jaxpr'].out_avals)
  assert len(new_params_staged['in_layouts']) == len(params_staged['jaxpr'].in_avals)
  assert len(new_params_staged['out_layouts']) == len(params_staged['jaxpr'].out_avals)
  return new_params_known, new_params_staged

pe.partial_eval_jaxpr_custom_rules[jit_p] = \
    partial(pe.closed_call_partial_eval_custom_rule, 'jaxpr',
            _pjit_partial_eval_custom_params_updater)


@lu.cache
def _pjit_transpose_trace(fun: lu.WrappedFun,
                          in_avals: Sequence[core.AbstractValue]):
  transpose_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(fun, in_avals)
  transpose_jaxpr = core.ClosedJaxpr(transpose_jaxpr, consts)
  return transpose_jaxpr


def _pjit_transpose(cts_in, *primals_in,
                    jaxpr: core.ClosedJaxpr,
                    in_shardings, out_shardings, in_layouts, out_layouts,
                    donated_invars, ctx_mesh, name, keep_unused, inline,
                    compiler_options_kvs):
  def prune_type(ty, xs, maybe_zeros):
    return tuple(x for x, mz in zip(xs, maybe_zeros) if type(mz) is not ty)

  body = lu.wrap_init(ad.closed_backward_pass, debug_info=jaxpr.jaxpr.debug_info)
  body = lu.hashable_partial(body, jaxpr, False)
  primals_and_nz_cts_in, in_treedef = tree_flatten((primals_in, cts_in))
  body, cts_out_treedef_thunk = flatten_fun_nokwargs(body, in_treedef)

  transpose_in_shardings = (
    *prune_type(ad.UndefinedPrimal, in_shardings, primals_in),
    *prune_type(ad.Zero, out_shardings, cts_in)
  )
  transpose_in_layouts = (
    *prune_type(ad.UndefinedPrimal, in_layouts, primals_in),
    *prune_type(ad.Zero, out_layouts, cts_in)
  )
  global_cts_in_avals = tuple(
      core.AvalQDD(a, cur_qdd(x)) if (a := typeof(x)).has_qdd else a
      for x in primals_and_nz_cts_in)

  transpose_jaxpr = _pjit_transpose_trace(body, global_cts_in_avals)
  cts_out_treedef = cts_out_treedef_thunk()
  transpose_out_shardings = prune_type(
      ad.Zero,
      in_shardings,
      tree_unflatten(cts_out_treedef, [object()] * cts_out_treedef.num_leaves))
  transpose_out_layouts = prune_type(
      ad.Zero,
      in_layouts,
      tree_unflatten(cts_out_treedef, [object()] * cts_out_treedef.num_leaves))

  try:
    nz_cts_out = jit_p.bind(
        *primals_and_nz_cts_in,
        jaxpr=transpose_jaxpr,
        in_shardings=transpose_in_shardings,
        out_shardings=transpose_out_shardings,
        in_layouts=transpose_in_layouts,
        out_layouts=transpose_out_layouts,
        donated_invars=(False,) * len(primals_and_nz_cts_in),
        ctx_mesh=ctx_mesh,
        name=name,
        keep_unused=keep_unused,
        inline=inline,
        compiler_options_kvs=compiler_options_kvs)
  except api_util.InternalFloatingPointError as e:
    print("Invalid nan value encountered in the backward pass of a jax.jit "
          "function. Calling the de-optimized backward pass.")
    try:
      _ = ad.closed_backward_pass(jaxpr, None, primals_in, cts_in)
    except (FloatingPointError, ZeroDivisionError) as e2:
      raise e2 from None  # great
    else:
      # If control reaches this line, we got a NaN on the output of `compiled`
      # but not `fun.call_wrapped` on the same arguments. Let's tell the user.
      api_util._raise_no_nan_in_deoptimized(e)

  return tree_unflatten(cts_out_treedef, nz_cts_out)
ad.primitive_transposes[jit_p] = _pjit_transpose


def _pjit_transpose_fancy(
    cts_in, *args, jaxpr, in_shardings, out_shardings, in_layouts,
    out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline,
    compiler_options_kvs):
  primals_ctrefs, specs = ad.project_accums(args)
  in_flat, in_tree = tree_flatten((primals_ctrefs, cts_in))
  in_avals = [core.AvalQDD(a, cur_qdd(x)) if (a := typeof(x)).has_qdd  # type: ignore
              else a for x in in_flat]
  trans_jaxpr, out_tree = _transpose_jaxpr_fancy(jaxpr, in_tree, (*in_avals,), specs)

  trans_in_shardings = (
      [s for x, s in zip(args, in_shardings) if not isinstance(x,ad.ValAccum)] +
      [s for x, s in zip(cts_in, out_shardings) if not isinstance(x, ad.Zero)])
  trans_in_layouts = (
      [l for x, l in zip(args, in_layouts) if not isinstance(x, ad.ValAccum)] +
      [l for x, l in zip(cts_in, out_layouts) if not isinstance(x, ad.Zero)])
  cts_out_ = tree_unflatten(out_tree, trans_jaxpr.out_avals)
  trans_out_shardings = tuple(s for x, s in zip(cts_out_, in_shardings) if x)
  trans_out_layouts   = tuple(l for x, l in zip(cts_out_, in_layouts  ) if x)

  try:
    cts_out = jit_p.bind(
        *in_flat, jaxpr=trans_jaxpr, in_shardings=tuple(trans_in_shardings),
        in_layouts=tuple(trans_in_layouts), out_shardings=trans_out_shardings,
        out_layouts=trans_out_layouts, donated_invars=(False,) * len(in_flat),
        ctx_mesh=ctx_mesh, name=name, keep_unused=keep_unused, inline=inline,
        compiler_options_kvs=compiler_options_kvs)
  except api_util.InternalFloatingPointError as e:
    print("Invalid nan value encountered in the backward pass of a jax.jit "
          "function. Calling the de-optimized backward pass.")
    try:
      ad.backward_pass3(jaxpr.jaxpr, False, jaxpr.consts, args, cts_in)
    except (FloatingPointError, ZeroDivisionError) as e2:
      raise e2 from None  # great
    else:
      # If control reaches this line, we got a NaN on the output of `compiled`
      # but not `fun.call_wrapped` on the same arguments. Let's tell the user.
      api_util._raise_no_nan_in_deoptimized(e)

  for x, ct in zip(args, tree_unflatten(out_tree, cts_out)):
    if isinstance(x, ad.ValAccum): x.accum(ct)

@weakref_lru_cache
def _transpose_jaxpr_fancy(jaxpr, in_tree, in_avals, specs):
  cell = lambda: None
  def transposed(*in_flat):
    primals_ctrefs, cts_in = tree_unflatten(in_tree, in_flat)
    args = ad.unproject_accums(specs, primals_ctrefs)
    ad.backward_pass3(jaxpr.jaxpr, False, jaxpr.consts, args, cts_in)
    cts_out = [x.freeze() if isinstance(x, ad.ValAccum) else None for x in args]
    cts_out, cell.out_tree = tree_flatten(cts_out)  # type: ignore
    return cts_out
  dbg = jaxpr.jaxpr.debug_info._replace(arg_names=(), result_paths=())
  trans_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(transposed, debug_info=dbg), in_avals)
  return core.ClosedJaxpr(trans_jaxpr, consts), cell.out_tree  # type: ignore
ad.fancy_transposes[jit_p] = _pjit_transpose_fancy

@weakref_lru_cache
def _dce_jaxpr_pjit(
    jaxpr: core.ClosedJaxpr, used_outputs: tuple[bool, ...]
) -> tuple[core.ClosedJaxpr, list[bool]]:
  new_jaxpr, used_inputs = pe.dce_jaxpr(jaxpr.jaxpr, used_outputs)
  return core.ClosedJaxpr(new_jaxpr, jaxpr.consts), used_inputs


def dce_jaxpr_pjit_rule(used_outputs: list[bool], eqn: core.JaxprEqn
                        ) -> tuple[list[bool], core.JaxprEqn | None]:

  if not any(used_outputs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None

  dced_jaxpr, used_inputs = _dce_jaxpr_pjit(
      eqn.params['jaxpr'], tuple(used_outputs))

  def keep_where(xs, keeps):
    return tuple(x for x, keep in zip(xs, keeps) if keep)

  eqn_params = eqn.params
  new_params = dict(
      eqn_params,
      jaxpr=dced_jaxpr,
      in_shardings=keep_where(eqn_params["in_shardings"], used_inputs),
      out_shardings=keep_where(eqn_params["out_shardings"], used_outputs),
      in_layouts=keep_where(eqn_params["in_layouts"], used_inputs),
      out_layouts=keep_where(eqn_params["out_layouts"], used_outputs),
      donated_invars=keep_where(eqn_params["donated_invars"], used_inputs),
  )
  if not any(used_inputs) and not any(used_outputs) and not dced_jaxpr.effects:
    return used_inputs, None
  else:
    new_eqn = core.new_jaxpr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs) if used],
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, dced_jaxpr.effects, eqn.source_info, eqn.ctx)
    return used_inputs, new_eqn

pe.dce_rules[jit_p] = dce_jaxpr_pjit_rule


def _pjit_pp_rule(eqn: core.JaxprEqn,
                  context: core.JaxprPpContext,
                  settings: core.JaxprPpSettings) -> core.pp.Doc:
  params = dict(eqn.params)
  del params['inline']
  if not any(params['donated_invars']):
    del params['donated_invars']
  if all(isinstance(s, UnspecifiedValue) for s in params['in_shardings']):
    del params['in_shardings']
  if all(isinstance(s, UnspecifiedValue) for s in params['out_shardings']):
    del params['out_shardings']
  if all(l is None for l in params['in_layouts']):
    del params['in_layouts']
  if all(l is None for l in params['out_layouts']):
    del params['out_layouts']
  if not params['keep_unused']:
    del params['keep_unused']
  if params['ctx_mesh'].empty:
    del params['ctx_mesh']
  if not params['compiler_options_kvs']:
    del params['compiler_options_kvs']

  if params['jaxpr'].jaxpr not in context.shared_jaxprs:
    context.suggest_same_var_names(params['jaxpr'].jaxpr.invars, eqn.invars)
    context.suggest_same_var_names(params['jaxpr'].jaxpr.outvars, eqn.outvars)

  # Move name= to the front to make the resulting equation easier to scan.
  del params["name"]
  return core._pp_eqn(eqn, context, settings, params=["name"] + sorted(params))

core.pp_eqn_rules[jit_p] = _pjit_pp_rule


# -------------------- with_sharding_constraint --------------------

def check_shardings_are_auto(s: Sharding) -> None:
  if not isinstance(s, NamedSharding):
    return
  mesh = s.mesh.abstract_mesh
  if not all(mesh._name_to_type[i] == mesh_lib.AxisType.Auto
              for axes in s.spec
              if axes is not PartitionSpec.UNCONSTRAINED and axes is not None
              for i in (axes if isinstance(axes, tuple) else (axes,))):
    raise ValueError(
        'The spec of NamedSharding passed to with_sharding_constraint can'
        f' only refer to Auto axes of the mesh. Got spec={s.spec} and'
        f' mesh={mesh}. You probably meant to use `reshard` API?')

def assert_shardings_equal(x_aval, user_sharding: NamedSharding):
  x_spec = x_aval.sharding.spec
  user_spec = user_sharding.spec._normalized_spec_for_aval(x_aval.ndim)
  for x, s in zip(x_spec, user_spec):
    if s is PartitionSpec.UNCONSTRAINED:
      continue
    else:
      if x != s:
        raise AssertionError(
            '`with_sharding_constraint` acts as an assert when all axes of'
            f' mesh are of type `Explicit`. The array sharding: {x_spec} did'
            f' not match the sharding provided: {user_spec}. Please use'
            ' `jax.sharding.reshard` to shard your input to the sharding you'
            ' want.')


def with_sharding_constraint(x, shardings):
  """Mechanism to constrain the sharding of an Array inside a jitted computation

  This is a strict constraint for the GSPMD partitioner and not a hint. For examples
  of how to use this function, see `Distributed arrays and automatic parallelization`_.

  Inside of a jitted computation, with_sharding_constraint makes it possible to
  constrain intermediate values to an uneven sharding. However, if such an
  unevenly sharded value is output by the jitted computation, it will come out
  as fully replicated, no matter the sharding annotation given.

  Args:
    x: PyTree of jax.Arrays which will have their shardings constrained
    shardings: PyTree of sharding specifications. Valid values are the same as for
      the ``in_shardings`` argument of :func:`jax.experimental.pjit`.
  Returns:
    x_with_shardings: PyTree of jax.Arrays with specified sharding constraints.

  .. _Distributed arrays and automatic parallelization: https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
  """
  x_flat, tree = tree_flatten(x)
  x_avals_flat = [core.shaped_abstractify(x) for x in x_flat]
  layouts, shardings = _split_layout_and_sharding(shardings)

  user_shardings = prepare_axis_resources(
      shardings, "shardings", allow_unconstrained_dims=True)
  del shardings

  user_shardings_flat = tuple(
      flatten_axes("with_sharding_constraint shardings", tree, user_shardings))
  del user_shardings

  user_layouts_flat = tuple(
      flatten_axes("with_sharding_constraint layouts", tree, layouts))
  del layouts

  context_mesh = (
      mesh_lib.get_abstract_mesh() if not mesh_lib.get_concrete_mesh().empty
      else mesh_lib.thread_resources.env.physical_mesh)

  shardings_flat = [_create_sharding_for_array(context_mesh, a, 'shardings',
                                               'with_sharding_constraint')
                    for a in user_shardings_flat]
  for s, u in zip(shardings_flat, user_shardings_flat):
    if isinstance(s, (UnspecifiedValue, AUTO)):
      raise ValueError(
          f'One of with_sharding_constraint arguments got sharding {u} which is'
          ' not allowed. Please only pass `jax.sharding.Sharding` instances.')
  del user_shardings_flat

  # TODO(bartchr): remove `unconstrained_dims` after migrating to Shardy. It's
  # already part of the shardings.
  unconstrained_dims = [get_unconstrained_dims(s)
                        if isinstance(s, NamedSharding) else frozenset()
                        for s in shardings_flat]

  pjit_check_aval_sharding(
      shardings_flat, x_avals_flat, ("",) * len(shardings_flat),
      "with_sharding_constraint arguments",
      allow_uneven_sharding=True)
  check_aval_layout_compatibility(user_layouts_flat, x_avals_flat,
                                  ("",) * len(user_layouts_flat),
                                  "with_sharding_constraint arguments")

  outs = []
  for xf, x_aval, s, l, ud in zip(x_flat, x_avals_flat, shardings_flat,
                                  user_layouts_flat, unconstrained_dims):
    if (mesh_lib.get_abstract_mesh().are_all_axes_explicit and l is None and
        isinstance(s, NamedSharding)):
      assert_shardings_equal(x_aval, s)
      outs.append(xf)
    else:
      check_shardings_are_auto(s)
      outs.append(sharding_constraint_p.bind(
          xf, sharding=s, layout=l, context_mesh=context_mesh,
          unconstrained_dims=ud))
  return tree_unflatten(tree, outs)

def _identity_fn(x): return x

def _sharding_constraint_impl(x, sharding, layout, context_mesh,
                              unconstrained_dims):
  if (isinstance(sharding, NamedSharding) and
      isinstance(sharding.mesh, AbstractMesh)):
    if (not context_mesh.empty and isinstance(context_mesh, AbstractMesh) and
        not hasattr(x, 'sharding')):
      concrete_mesh = mesh_lib.get_concrete_mesh()
      assert not concrete_mesh.empty
      sharding = NamedSharding(concrete_mesh, sharding.spec)
    else:
      aval = core.shaped_abstractify(x)
      if not hasattr(x, 'sharding'):
        raise ValueError(
            'Target sharding contains a `jax.sharding.AbstractMesh` which'
            ' requires the input passed should be a `jax.Array`. Got'
            f' {type(x)} with shape {aval.str_short()}')
      if not isinstance(x.sharding, NamedSharding):
        raise TypeError(
            'The sharding on the input must be a `NamedSharding` since the'
            ' target sharding has an `AbstractMesh` in it. Got sharding type'
            f' {type(x.sharding)} for shape {aval.str_short()}')
      if x.sharding.mesh.shape_tuple != sharding.mesh.shape_tuple:
        raise ValueError(
            f'Mesh shape of the input {x.sharding.mesh.shape_tuple} does not'
            ' match the mesh shape of the target sharding'
            f' {sharding.mesh.shape_tuple} for shape {aval.str_short()}')
      sharding = NamedSharding(x.sharding.mesh, sharding.spec)

  if layout is None:
    if hasattr(x, 'sharding') and x.sharding.is_equivalent_to(sharding, x.ndim):
      return x
    # Run a jit here to raise good errors when device assignment don't match.
    return api.jit(_identity_fn, out_shardings=sharding)(x)
  else:
    if (hasattr(x, 'format') and x.format.layout == layout and
        x.sharding.is_equivalent_to(sharding, x.ndim)):
      return x
    return api.jit(_identity_fn, out_shardings=Format(layout, sharding))(x)


sharding_constraint_p = core.Primitive("sharding_constraint")
sharding_constraint_p.def_impl(_sharding_constraint_impl)
ad.deflinear2(sharding_constraint_p,
              lambda ct, _, **params: (sharding_constraint_p.bind(ct, **params),))

def _sharding_constraint_abstract_eval(
    x_aval, *, sharding, layout, context_mesh, unconstrained_dims):
  if x_aval.sharding.mesh.empty and isinstance(sharding, NamedSharding):
    return x_aval.update(
        sharding=x_aval.sharding.update(mesh=sharding.mesh.abstract_mesh))
  return x_aval
sharding_constraint_p.def_abstract_eval(_sharding_constraint_abstract_eval)

def _sharding_constraint_hlo_lowering(ctx, x_node, *, sharding, layout,
                                      context_mesh, unconstrained_dims):
  in_aval, = ctx.avals_in
  out_aval, = ctx.avals_out
  axis_ctx = ctx.module_context.axis_context

  if (isinstance(sharding, NamedSharding) and
      any(o is not None for o in out_aval.sharding.spec)):
    spec = sharding.spec._normalized_spec_for_aval(in_aval.ndim)
    new_spec = []
    for user_spec, aval_spec in zip(spec, out_aval.sharding.spec):
      if aval_spec is None:
        new_spec.append(user_spec)
      else:
        aval_spec = aval_spec if isinstance(aval_spec, tuple) else (aval_spec,)
        if user_spec is PartitionSpec.UNCONSTRAINED:
          raise NotImplementedError
        if user_spec is None:
          new_spec.append(aval_spec)
        elif isinstance(user_spec, tuple):
          new_spec.append(aval_spec + user_spec)
        else:
          new_spec.append(aval_spec + (user_spec,))
    sharding = sharding.update(spec=new_spec)

  if dtypes.issubdtype(in_aval.dtype, dtypes.extended):
    in_aval = core.physical_aval(in_aval)
  if (isinstance(axis_ctx, sharding_impls.SPMDAxisContext) and
      axis_ctx.manual_axes):
    sharding = mlir.add_manual_axes(axis_ctx, sharding, in_aval.ndim)
  if config.use_shardy_partitioner.value:
    sharding = sharding._to_sdy_sharding(in_aval.ndim)
  else:
    sharding = sharding._to_xla_hlo_sharding(in_aval.ndim).to_proto()
  out = mlir.wrap_with_sharding_op(
      ctx, x_node, out_aval, sharding, unspecified_dims=unconstrained_dims)
  if layout is not None:
    out = mlir.wrap_with_layout_op(ctx, out, out_aval, layout, in_aval)
  return [out]
mlir.register_lowering(sharding_constraint_p,
                       _sharding_constraint_hlo_lowering)


def _sharding_constraint_batcher(
    axis_data, vals_in, dims_in, sharding, layout, context_mesh,
    unconstrained_dims):
  if axis_data.spmd_name is not None and isinstance(sharding, NamedSharding):
    used = {n for ns in sharding.spec
            for n in (ns if isinstance(ns, tuple) else (ns,))}
    if set(axis_data.spmd_name) & used:
      raise ValueError(f"vmap spmd_axis_name {axis_data.spmd_name} cannot appear in "
                       "with_sharding_constraint spec, but got spec "
                       f"{sharding.spec}")
  x, = vals_in
  d, = dims_in
  unconstrained_dims = {ud + (d <= ud) for ud in unconstrained_dims}
  if axis_data.spmd_name is None:
    unconstrained_dims.add(d)

  vmapped_sharding = _pjit_batcher_for_sharding(
      sharding, d, axis_data.spmd_name, context_mesh, x.ndim)
  if unconstrained_dims and isinstance(vmapped_sharding, NamedSharding):
    new_spec = list(vmapped_sharding.spec) + [None] * (x.ndim - len(vmapped_sharding.spec))
    for u in unconstrained_dims:
      new_spec[u] = PartitionSpec.UNCONSTRAINED
    vmapped_sharding = NamedSharding(
        vmapped_sharding.mesh, PartitionSpec(*new_spec))

  vmapped_layout = (get_layout_for_vmap(d, layout) if layout is not None else
                    layout)

  y = sharding_constraint_p.bind(
      x,
      sharding=vmapped_sharding,
      layout=vmapped_layout,
      context_mesh=context_mesh,
      unconstrained_dims=frozenset(unconstrained_dims))
  return y, d
batching.fancy_primitive_batchers[sharding_constraint_p] = _sharding_constraint_batcher
batching.skippable_batchers[sharding_constraint_p] = lambda _: ()

# -------------------- mesh_cast ---------------------------

# TODO(yashkatariya): Make shardings optional.
def mesh_cast(xs, out_shardings):
  x_flat, treedef = tree_flatten(xs)
  shardings_flat = flatten_axis_resources(
      "mesh_cast out_shardings", treedef, out_shardings, tupled_args=True)
  out_flat = [
      mesh_cast_p.bind(
          x, dst_sharding=canonicalize_sharding(
              s, 'mesh_cast', check_mesh_consistency=False))
      for x, s in safe_zip(x_flat, shardings_flat)
  ]
  return tree_unflatten(treedef, out_flat)

mesh_cast_p = core.Primitive('mesh_cast')
mesh_cast_p.skip_canonicalization = True
def _mesh_cast_abstract_eval(aval, dst_sharding):
  src_sharding = aval.sharding
  if src_sharding == dst_sharding:
    return aval
  if src_sharding.mesh.empty or dst_sharding.mesh.empty:
    return aval.update(sharding=dst_sharding)
  if src_sharding.mesh.shape_tuple != dst_sharding.mesh.shape_tuple:
    raise ValueError(
        f'Mesh shape of the input {src_sharding.mesh.shape_tuple} does not'
        ' match the mesh shape of the target sharding'
        f' {dst_sharding.mesh.shape_tuple} for shape {aval.str_short()}')
  if (src_sharding.mesh.axis_types == dst_sharding.mesh.axis_types
      and src_sharding.spec != dst_sharding.spec):
    raise ValueError(
        'mesh_cast should only be used when AxisType changes between the'
        ' input mesh and the target mesh. Got src'
        f' axis_types={src_sharding.mesh.axis_types} and dst'
        f' axis_types={dst_sharding.mesh.axis_types}. To reshard between'
        ' the same mesh, use `jax.sharding.reshard` instead?')
  if src_sharding.mesh._any_axis_explicit and dst_sharding.mesh._any_axis_explicit:
    for s, d in safe_zip(flatten_spec(src_sharding.spec),
                         flatten_spec(dst_sharding.spec)):
      if s is None and d is None:
        continue
      if s is None and d is not None:
        assert (src_sharding.mesh._name_to_type[d] == mesh_lib.AxisType.Auto
                and dst_sharding.mesh._name_to_type[d] == mesh_lib.AxisType.Explicit)
        continue
      if s is not None and d is None:
        assert (src_sharding.mesh._name_to_type[s] == mesh_lib.AxisType.Explicit
                and dst_sharding.mesh._name_to_type[s] == mesh_lib.AxisType.Auto)
        continue
      if d != s:
        raise ValueError(
            'Explicit data movement in mesh_cast is not allowed. Got src spec:'
            f' {s} and dst spec: {d}')
  return aval.update(sharding=dst_sharding)
mesh_cast_p.def_abstract_eval(_mesh_cast_abstract_eval)

def _mesh_cast_impl(x, dst_sharding):
  return dispatch.apply_primitive(mesh_cast_p, x, dst_sharding=dst_sharding)
mesh_cast_p.def_impl(_mesh_cast_impl)

def _mesh_cast_transpose_rule(ct, x, dst_sharding):
  return [mesh_cast_p.bind(ct, dst_sharding=x.aval.sharding)]
ad.deflinear2(mesh_cast_p, _mesh_cast_transpose_rule)

def _mesh_cast_hlo_lowering(ctx, x_node, *, dst_sharding):
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  if dtypes.issubdtype(aval_in.dtype, dtypes.extended):
    aval_in = core.physical_aval(aval_in)
  proto = (dst_sharding._to_sdy_sharding(aval_in.ndim)
           if config.use_shardy_partitioner.value else
           dst_sharding._to_xla_hlo_sharding(aval_in.ndim).to_proto())
  return [mlir.lower_with_sharding_in_types(ctx, x_node, aval_out, proto)]
mlir.register_lowering(mesh_cast_p, _mesh_cast_hlo_lowering)

def _mesh_cast_batcher(axis_data, vals_in, dims_in, dst_sharding):
  x, = vals_in
  d, = dims_in
  vmapped_dst_sharding = batching.get_sharding_for_vmap(
      axis_data, dst_sharding, d)
  y = mesh_cast_p.bind(x, dst_sharding=vmapped_dst_sharding)
  return y, d
batching.fancy_primitive_batchers[mesh_cast_p] = _mesh_cast_batcher
batching.skippable_batchers[mesh_cast_p] = lambda _: ()

# -------------------- reshard ------------------------------------

def reshard(xs, out_shardings):
  x_flat, treedef = tree_flatten(xs)
  shardings_flat = flatten_axis_resources(
      "reshard out_shardings", treedef, out_shardings, tupled_args=True)
  x_avals_flat = [core.shaped_abstractify(x) for x in x_flat]
  out_flat = []
  for x, x_aval, s in safe_zip(x_flat, x_avals_flat, shardings_flat):
    ds = canonicalize_sharding(s, 'reshard')
    if ds is None:
      raise ValueError(
          'Reshard should only be used with out_shardings which are non-None '
          f'and have a nonempty mesh. Got sharding {s}.'
      )
    ds = ds.update(spec=ds.spec._normalized_spec_for_aval(x_aval.ndim))  # pytype: disable=attribute-error
    out_flat.append(reshard_p.bind(x, dst_sharding=ds))
  return tree_unflatten(treedef, out_flat)

reshard_p = core.Primitive('reshard')

def _reshard_abstract_eval(aval, dst_sharding):
  src_sharding = aval.sharding
  if (not src_sharding.mesh.empty and
      src_sharding.mesh.abstract_mesh != dst_sharding.mesh.abstract_mesh):
    raise ValueError(
        f'Mesh of the input {src_sharding.mesh.abstract_mesh} does not'
        ' equal the mesh of the target sharding'
        f' {dst_sharding.mesh.abstract_mesh} for shape {aval.str_short()}')
  return aval.update(sharding=dst_sharding)
reshard_p.def_abstract_eval(_reshard_abstract_eval)

def _reshard_impl(x, dst_sharding):
  cur_concrete_mesh = mesh_lib.get_concrete_mesh()
  if not cur_concrete_mesh.empty and not cur_concrete_mesh.is_multi_process:
    return api.device_put(x, dst_sharding.spec)
  return dispatch.apply_primitive(reshard_p, x, dst_sharding=dst_sharding)
reshard_p.def_impl(_reshard_impl)

def _reshard_transpose_rule(ct, x, dst_sharding):
  return [reshard_p.bind(ct, dst_sharding=x.aval.to_cotangent_aval().sharding)]
ad.deflinear2(reshard_p, _reshard_transpose_rule)

def _reshard_hlo_lowering(ctx, x_node, *, dst_sharding):
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  if dtypes.issubdtype(aval_in.dtype, dtypes.extended):
    aval_in = core.physical_aval(aval_in)
  proto = (dst_sharding._to_sdy_sharding(aval_in.ndim)
           if config.use_shardy_partitioner.value else
           dst_sharding._to_xla_hlo_sharding(aval_in.ndim).to_proto())
  return [mlir.lower_with_sharding_in_types(ctx, x_node, aval_out, proto)]
mlir.register_lowering(reshard_p, _reshard_hlo_lowering)

def _reshard_batcher(axis_data, vals_in, dims_in, dst_sharding):
  x, = vals_in
  d, = dims_in
  vmapped_dst_sharding = batching.get_sharding_for_vmap(
      axis_data, dst_sharding, d)
  y = reshard_p.bind(x, dst_sharding=vmapped_dst_sharding)
  return y, d
batching.fancy_primitive_batchers[reshard_p] = _reshard_batcher
batching.skippable_batchers[reshard_p] = lambda _: ()

# -------------------- auto and user mode -------------------------

def _get_new_mesh(axes: str | tuple[str, ...] | None,
                  axis_type: mesh_lib.AxisType, name: str, shardings=None,
                  error_on_manual_to_auto_explicit=False):
  cur_mesh = mesh_lib.get_abstract_mesh()
  flat_shardings, _ = tree_flatten(shardings)
  sharding_mesh = mesh_lib.empty_abstract_mesh
  for i in flat_shardings:
    if isinstance(i, NamedSharding):
      if not sharding_mesh.empty and sharding_mesh != i.mesh:
        raise ValueError(
            f'Shardings passed to {name} should have the same mesh. Got one'
            f' mesh {sharding_mesh} and another {i.mesh}')
      sharding_mesh = i.mesh.abstract_mesh

  if sharding_mesh.empty and cur_mesh.empty:
    raise ValueError(
        f'Context mesh {cur_mesh} cannot be empty. Please use'
        ' `jax.set_mesh` API to enter into a mesh context when using'
        f' `{name}` API.')
  if not sharding_mesh.empty and not cur_mesh.empty:
    if sharding_mesh != cur_mesh:
      raise ValueError(
          f'Context mesh {cur_mesh} must match the mesh passed to shardings'
          f' {sharding_mesh}. Recommended approach is to use'
          ' `jax.set_mesh` context manager.')
    mesh_to_use = cur_mesh
  elif sharding_mesh.empty and not cur_mesh.empty:
    mesh_to_use = cur_mesh
  else:
    assert not sharding_mesh.empty and cur_mesh.empty
    mesh_to_use = sharding_mesh

  if axes is None:
    axes = mesh_to_use.axis_names
  if not isinstance(axes, tuple):
    axes = (axes,)
  for a in axes:
    if (error_on_manual_to_auto_explicit and
        mesh_to_use._name_to_type[a] == mesh_lib.AxisType.Manual and
        axis_type in {mesh_lib.AxisType.Auto, mesh_lib.AxisType.Explicit}):
      raise NotImplementedError(
          'Going from `Manual` AxisType to `Auto` or `Explicit` AxisType is not'
          ' allowed. Please file a bug at https://github.com/jax-ml/jax/issues'
          ' with your use case')
  return (mesh_to_use.update_axis_types({a: axis_type for a in axes}),
          mesh_to_use, axes)

def auto_axes(f=None, /, *, axes: str | tuple[str, ...] | None = None,
              out_sharding=None):
  kwargs = dict(axes_=axes, out_sharding=out_sharding)
  if f is None:
    return lambda g: _auto_axes(g, **kwargs)
  return _auto_axes(f, **kwargs)

def _auto_axes(fun, *, axes_, out_sharding):
  def decorator(*args, **kwargs):
    if out_sharding is None:
      if "out_sharding" in kwargs:
        _out_sharding = kwargs.pop("out_sharding")
      else:
        raise TypeError("Missing required keyword argument: 'out_sharding'")
    else:
      _out_sharding = out_sharding
    new_mesh, prev_mesh, axes = _get_new_mesh(
        axes_, mesh_lib.AxisType.Auto, 'auto_axes', shardings=_out_sharding,
        error_on_manual_to_auto_explicit=True)
    if set(prev_mesh.auto_axes) == set(axes):
      return fun(*args, **kwargs)
    with mesh_lib.use_abstract_mesh(new_mesh):
      in_specs = tree_map(lambda a: core.modify_spec_for_auto_manual(
          core.get_aval(a).sharding.spec, new_mesh), args)
      args = mesh_cast(args, in_specs)
      out = fun(*args, **kwargs)
    return mesh_cast(out, _out_sharding)
  return decorator


@contextlib.contextmanager
def use_auto_axes(*axes):
  new_mesh, _, _ = _get_new_mesh(axes, mesh_lib.AxisType.Auto, 'use_auto_axes')
  with mesh_lib.use_abstract_mesh(new_mesh):
    yield


def explicit_axes(f=None, /, *, axes: str | tuple[str, ...] | None = None,
                  in_sharding=None):
  kwargs = dict(axes=axes, in_sharding=in_sharding)
  if f is None:
    return lambda g: _explicit_axes(g, **kwargs)
  return _explicit_axes(f, **kwargs)

def _explicit_axes(fun, *, axes, in_sharding):
  def decorator(*args, **kwargs):
    if in_sharding is None:
      if "in_sharding" in kwargs:
        _in_sharding = kwargs.pop("in_sharding")
      else:
        raise TypeError("Missing required keyword argument: 'in_sharding'")
    else:
      _in_sharding = in_sharding
    new_mesh, _, _ = _get_new_mesh(
        axes, mesh_lib.AxisType.Explicit, 'explicit_axes',
        error_on_manual_to_auto_explicit=True)
    with mesh_lib.use_abstract_mesh(new_mesh):
      args = mesh_cast(args, _in_sharding)
      out = fun(*args, **kwargs)
    out_specs = tree_map(lambda o: core.modify_spec_for_auto_manual(
        core.get_aval(o).sharding.spec, mesh_lib.get_abstract_mesh()), out)
    return mesh_cast(out, out_specs)
  return decorator


@contextlib.contextmanager
def use_explicit_axes(*axes):
  new_mesh, _, _ = _get_new_mesh(
      axes, mesh_lib.AxisType.Explicit, 'use_explicit_axes')
  with mesh_lib.use_abstract_mesh(new_mesh):
    yield

# -------------------- with_layout_constraint --------------------

def with_layout_constraint(x, layouts):
  x_flat, tree = tree_flatten(x)
  x_avals_flat = [core.shaped_abstractify(x) for x in x_flat]
  layouts_flat = tuple(flatten_axes("with_layout_constraint layouts", tree,
                                    layouts))
  if any(not isinstance(l, Layout) for l in layouts_flat):
    raise ValueError(
        'layouts passed to `with_layout_constraint` must be of type'
        f' `Layout`. Got {[type(l) for l in layouts_flat]}')
  check_aval_layout_compatibility(
      layouts_flat, x_avals_flat, ("",) * len(layouts_flat),
      "with_layout_constraint arguments")
  outs = [layout_constraint_p.bind(xf, layout=l)
          for xf, l in zip(x_flat, layouts_flat)]
  return tree_unflatten(tree, outs)

layout_constraint_p = core.Primitive('layout_constraint')
layout_constraint_p.def_abstract_eval(lambda x, **_: x)
ad.deflinear2(layout_constraint_p,
              lambda ct, _, **params: (layout_constraint_p.bind(ct, **params),))

def _layout_constraint_impl(x, *, layout):
  if not isinstance(x, xc.ArrayImpl):
    raise ValueError(
        'with_layout_constraint in eager mode can only be applied to'
        f' jax.Arrays. Got {type(x)}')
  if x.format.layout == layout:  # type: ignore
    return x
  return api.jit(_identity_fn, out_shardings=Format(layout, x.sharding))(x)
layout_constraint_p.def_impl(_layout_constraint_impl)

def _layout_constraint_hlo_lowering(ctx, x_node, *, layout):
  aval, = ctx.avals_in
  out_aval, = ctx.avals_out
  return [mlir.wrap_with_layout_op(ctx, x_node, out_aval, layout, aval)]
mlir.register_lowering(layout_constraint_p,
                       _layout_constraint_hlo_lowering)

def _layout_constraint_batcher(axis_data, vals_in, dims_in, layout):
  x, = vals_in
  d, = dims_in
  vmapped_layout = get_layout_for_vmap(d, layout)
  y = layout_constraint_p.bind(x, layout=vmapped_layout)
  return y, d
batching.fancy_primitive_batchers[layout_constraint_p] = _layout_constraint_batcher
batching.skippable_batchers[layout_constraint_p] = lambda _: ()

# -------------------- helpers --------------------

def get_unconstrained_dims(sharding: NamedSharding):
  assert sharding.spec is not None
  return frozenset(i for i, axes in enumerate(sharding.spec)
                   if axes is PartitionSpec.UNCONSTRAINED)
