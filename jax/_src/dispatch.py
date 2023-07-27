# Copyright 2018 The JAX Authors.
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

# Primitive dispatch and jit dispatch.
from __future__ import annotations

import atexit
from collections.abc import Iterator, Sequence
import contextlib
import dataclasses
from functools import partial
import itertools
import time
from typing import (Any, Callable, Optional, NamedTuple)
import logging
import os
import re
import threading
import warnings

import numpy as np

from jax._src import compilation_cache
from jax._src import config as jax_config
from jax._src import core
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import api_util
from jax._src import path
from jax._src import profiler
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util
from jax._src import op_shardings
from jax._src import xla_bridge as xb
from jax._src.config import config
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import xla
from jax._src.interpreters import pxla
from jax._src.lib.mlir import ir
from jax._src.lib import xla_client as xc
from jax._src.monitoring import record_event_duration_secs
from jax._src.partition_spec import PartitionSpec
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (
    PmapSharding, SingleDeviceSharding, NamedSharding, XLACompatibleSharding,
    UNSPECIFIED, GSPMDSharding)


JAXPR_TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"
JAXPR_TO_MLIR_MODULE_EVENT = "/jax/core/compile/jaxpr_to_mlir_module_duration"
BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"

_DUMP_IR_TO = jax_config.DEFINE_string(
    'jax_dump_ir_to', os.getenv('JAX_DUMP_IR_TO', ''),
    help="Path to which the IR that is emitted by JAX as input to the "
         "compiler should be dumped as text files. Optional. If omitted, JAX "
         "will not dump IR.")


traceback_util.register_exclusion(__file__)

MYPY = False  # Are we currently type checking with mypy?

xe = xc._xla

Backend = xe.Client
Device = xc.Device

CompileOptions = xc.CompileOptions

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

logger = logging.getLogger(__name__)

# This flag is set on exit; no logging should be attempted
_on_exit = False

### op-by-op execution

ArgSpec = tuple[core.AbstractValue, Optional[Device]]

def arg_spec(x: Any) -> ArgSpec:
  from jax._src import pjit

  aval = xla.abstractify(x)
  try:
    if isinstance(x.sharding, PmapSharding):
      return aval, None
    return aval, (pjit.to_gspmd_sharding(x.sharding, x.ndim)  # type: ignore
                  if x._committed else None)
  except:
    return aval, None


@dataclasses.dataclass(frozen=True)
class OrigShardings:
  shardings: Sequence[GSPMDSharding | None]

  def __hash__(self):
    return hash(tuple(s for s in self.shardings))

  def __eq__(self, other):
    if not isinstance(other, OrigShardings):
      return False
    return all(getattr(s, "_original_sharding", s) == getattr(o, "_original_sharding", o)
               for s, o in zip(self.shardings, other.shardings))


def apply_primitive(prim, *args, **params):
  """Impl rule that compiles and runs a single primitive 'prim' using XLA."""
  from jax._src import pjit

  try:
    in_avals, in_shardings = util.unzip2([arg_spec(a) for a in args])
    compiled_fun = xla_primitive_callable(
        prim, in_avals, OrigShardings(in_shardings), **params)
  except pxla.DeviceAssignmentMismatchError as e:
    fails, = e.args
    # TODO(yashkatariya): Thread through a signature_fun via every primitive
    # using apply_primitive so that the error message has the right argument
    # name instead of `args[0]`, etc.
    arg_names = api_util._arg_names(prim.impl, args, {}, (), ())
    msg = pjit._device_assignment_mismatch_error(
        prim.name, fails, args, 'jit', arg_names)
    raise ValueError(msg) from None

  return compiled_fun(*args)


def simple_impl(prim):
  prim.def_impl(partial(apply_primitive, prim))

RuntimeToken = Any

class RuntimeTokenSet(threading.local):
  tokens: dict[core.Effect, tuple[RuntimeToken, Device]]
  output_tokens: dict[Device, RuntimeToken]
  output_runtime_tokens: dict[Device, RuntimeToken]

  def __init__(self):
    self.tokens = {}
    # TODO(sharadmv): remove redundant output token dictionary when minimum
    # jaxlib version is bumped to 0.3.16.
    self.output_tokens = {}
    self.output_runtime_tokens = {}

  def get_token(self, eff: core.Effect, device: Device) -> RuntimeToken:
    s = SingleDeviceSharding(device)
    if eff not in self.tokens:
      inp = np.zeros(0, np.bool_)
      indices = tuple(
          s.addressable_devices_indices_map(inp.shape).values())
      out = pxla.shard_args([device], [indices], [s], [inp])
      self.tokens[eff] = out, device
    elif self.tokens[eff][1] != device:
      (old_token,), _ = self.tokens[eff]
      indices = tuple(
          s.addressable_devices_indices_map((0,)).values())
      out = pxla.shard_args([device], [indices], [s], [old_token])
      self.tokens[eff] = out, device
    return self.tokens[eff][0]

  def update_token(self, eff: core.Effect, token: RuntimeToken):
    self.tokens[eff] = token, self.tokens[eff][1]

  def set_output_token(self, device: Device, token: RuntimeToken):
    # We're free to clobber the previous output token because on each
    # device we have a total ordering of computations. Only the token
    # from the latest computation matters. If this weren't the case
    # we'd need to store a set of output tokens.
    self.output_tokens[device] = token

  def set_output_runtime_token(self, device: Device, token: RuntimeToken):
    # TODO(sharadmv): remove this method when minimum jaxlib version is bumped
    self.output_runtime_tokens[device] = token

  def clear(self):
    self.tokens = {}
    self.output_tokens = {}
    self.output_runtime_tokens = {}

  def block_until_ready(self):
    for token, _ in self.tokens.values():
      token[0].block_until_ready()
    for token in self.output_tokens.values():
      token[0].block_until_ready()
    for token in self.output_runtime_tokens.values():
      token.block_until_ready()
    self.clear()

runtime_tokens: RuntimeTokenSet = RuntimeTokenSet()

@atexit.register
def wait_for_tokens():
  runtime_tokens.block_until_ready()

@util.cache()
def xla_primitive_callable(prim, in_avals, orig_in_shardings, **params):
  def prim_fun(*args):
    out = prim.bind(*args, **params)
    if prim.multiple_results:
      return out
    else:
      return out,
  donated_invars = (False,) * len(in_avals)
  compiled = _xla_callable_uncached(
      lu.wrap_init(prim_fun), prim.name, donated_invars, False, in_avals,
      orig_in_shardings)
  if not prim.multiple_results:
    return lambda *args, **kw: compiled(*args, **kw)[0]
  else:
    return compiled


def sharded_lowering(fun, name, donated_invars, keep_unused, inline,
                     in_avals, in_shardings, lowering_platform: str | None):
  if isinstance(in_shardings, OrigShardings):
    in_shardings = in_shardings.shardings

  in_shardings = [UNSPECIFIED if i is None else i for i in in_shardings]  # type: ignore

  # Pass in a singleton `UNSPECIFIED` for out_shardings because we don't know
  # the number of output avals at this stage. lower_sharding_computation will
  # apply it to all out_avals.
  return pxla.lower_sharding_computation(
      fun, 'jit', name, in_shardings, UNSPECIFIED, donated_invars,
      tuple(in_avals), keep_unused=keep_unused, inline=inline, always_lower=False,
      devices_from_context=None, lowering_platform=lowering_platform)


def _xla_callable_uncached(fun: lu.WrappedFun, name, donated_invars,
                           keep_unused, in_avals, orig_in_shardings):
  computation = sharded_lowering(
      fun, name, donated_invars, keep_unused, True, in_avals, orig_in_shardings,
      lowering_platform=None)
  return computation.compile().unsafe_call


def is_single_device_sharding(sharding) -> bool:
  # Special case PmapSharding here because PmapSharding maps away an axis
  # and needs to be handled separately.test_pjit_single_device_sharding_add
  return len(sharding.device_set) == 1 and not isinstance(sharding, PmapSharding)


@contextlib.contextmanager
def log_elapsed_time(fmt: str, fun_name: str, event: str | None = None):
  if _on_exit:
    yield
  else:
    log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    if logger.isEnabledFor(log_priority):
      logger.log(logging.WARNING, fmt.format(
          fun_name=fun_name, elapsed_time=elapsed_time))
    if event is not None:
      record_event_duration_secs(event, elapsed_time)


def should_tuple_args(num_args: int, platform: str):
  # CPU and GPU do not need tuples as they use host-side data structures that
  # do not have small bounds.
  # TPU only needs a tuple for very long lists
  if platform == "tpu":
    return num_args > 2000
  else:
    return False


def raise_warnings_or_errors_for_jit_of_pmap(nreps, backend, name, jaxpr):
  if nreps > 1:
    warnings.warn(
        f"The jitted function {name} includes a pmap. Using "
         "jit-of-pmap can lead to inefficient data movement, as the outer jit "
         "does not preserve sharded data representations and instead collects "
         "input and output arrays onto a single device. "
         "Consider removing the outer jit unless you know what you're doing. "
         "See https://github.com/google/jax/issues/2926.")

  if nreps > xb.device_count(backend):
    raise ValueError(
        f"compiling computation `{name}` that requires {nreps} replicas, but "
        f"only {xb.device_count(backend)} XLA devices are available.")

  if xb.process_count() > 1 and (nreps > 1 or
                                 jaxpr_has_primitive(jaxpr, "xla_pmap")):
    raise NotImplementedError(
        "jit of multi-host pmap not implemented (and jit-of-pmap can cause "
        "extra data movement anyway, so maybe you don't want it after all).")


def jaxpr_has_primitive(jaxpr, prim_name: str):
  """Whether there is a primitive given by user anywhere inside a Jaxpr."""
  for eqn in jaxpr.eqns:
    if prim_name in eqn.primitive.name:
      return True
  for subjaxpr in core.subjaxprs(jaxpr):
    if jaxpr_has_primitive(subjaxpr, prim_name):
      return True
  return False


class SourceInfo(NamedTuple):
  source_info: str
  eqn_name: str


def jaxpr_shardings(
    jaxpr) -> Iterator[tuple[XLACompatibleSharding, SourceInfo]]:
  from jax._src import pjit
  from jax.experimental import shard_map

  for eqn in jaxpr.eqns:
    if eqn.primitive is pjit.sharding_constraint_p:
      source_info = SourceInfo(source_info_util.summarize(eqn.source_info),
                                eqn.primitive.name)
      yield (eqn.params['sharding'], source_info)
    elif eqn.primitive is pjit.pjit_p:
      source_info = SourceInfo(source_info_util.summarize(eqn.source_info),
                                eqn.primitive.name)
      yield from ((i, source_info) for i in eqn.params['in_shardings'])
      yield from ((o, source_info) for o in eqn.params['out_shardings'])
    elif eqn.primitive is shard_map.shard_map_p:
      source_info = SourceInfo(source_info_util.summarize(eqn.source_info),
                                eqn.primitive.name)
      def _names_to_pspec(names):
        ndmin = max(names) + 1 if names else 0
        return PartitionSpec(*(names.get(i) for i in range(ndmin)))
      yield from ((NamedSharding(eqn.params['mesh'], _names_to_pspec(names)), source_info)
                  for names in [*eqn.params['in_names'], *eqn.params['out_names']])
  for subjaxpr in core.subjaxprs(jaxpr):
    yield from jaxpr_shardings(subjaxpr)


def jaxpr_has_bints(jaxpr: core.Jaxpr) -> bool:
  return (any(type(v.aval.dtype) is core.bint for v in jaxpr.invars
              if isinstance(v.aval, core.UnshapedArray)) or
          any(_is_bint_axis_size(d)
              for j in itertools.chain([jaxpr], core.subjaxprs(jaxpr))
              for e in j.eqns for v in e.outvars
              if isinstance(v.aval, core.DShapedArray) for d in v.aval.shape))

def _is_bint_axis_size(d: core.AxisSize) -> bool:
  if isinstance(d, core.DArray):
    assert not d.shape
    return type(d.dtype) is core.bint
  elif isinstance(d, core.Var):
    return (isinstance(d.aval, core.DShapedArray) and
            type(d.aval.dtype) is core.bint)
  return False

def _prune_unused_inputs(
    jaxpr: core.Jaxpr) -> tuple[core.Jaxpr, set[int], set[int]]:
  used_outputs = [True] * len(jaxpr.outvars)
  new_jaxpr, used_consts, used_inputs = pe.dce_jaxpr_consts(jaxpr, used_outputs)
  kept_const_idx = {i for i, b in enumerate(used_consts) if b}
  kept_var_idx = {i for i, b in enumerate(used_inputs) if b}
  return new_jaxpr, kept_const_idx, kept_var_idx


# We can optionally set a Jaxpr rewriter that can be applied just before
# compilation. This mechanism is used for compiling id_tap, we can
# remove it once we bring the id_tap implementation into the core.
outfeed_rewriter: Callable[[core.Jaxpr], core.Jaxpr] | None = None
def apply_outfeed_rewriter(jaxpr: core.Jaxpr) -> core.Jaxpr:
  if outfeed_rewriter is not None:
    return outfeed_rewriter(jaxpr)
  else:
    return jaxpr


# TODO(mattjj,necula): this duplicates code in core.valid_jaxtype, but one
# internal user relies on it for duck-typing. must fix downstream user!
def _valid_jaxtype(arg):
  try:
    xla.abstractify(arg)  # faster than core.get_aval
  except TypeError:
    return core.valid_jaxtype(arg)
  else:
    return True

def check_arg(arg):
  if not (isinstance(arg, core.Tracer) or _valid_jaxtype(arg)):
    raise TypeError(f"Argument '{arg}' of type {type(arg)} is not a valid "
                    "JAX type.")


def jaxpr_replicas(jaxpr) -> int:
  """The number of replicas needed for a jaxpr.

  For a eqn, multiply the `axis_size` with the `jaxpr_replicas` of the
  subjaxprs. For a list of eqns, take the maximum number of replicas.
  """
  if isinstance(jaxpr, core.ClosedJaxpr):
    jaxpr = jaxpr.jaxpr
  return max(unsafe_map(eqn_replicas, jaxpr.eqns), default=1)

# TODO(mattjj): this function assumes that only pmap has a parameter named
# axis_size, and that it corresponds to cross-replica mapping
def eqn_replicas(eqn):
  call_jaxpr = eqn.params.get("call_jaxpr")
  if call_jaxpr:
    return eqn.params.get('axis_size', 1) * jaxpr_replicas(call_jaxpr)
  elif eqn.primitive in xla.initial_style_primitives:
    return initial_style_primitive_replicas(eqn.params)
  else:
    return 1

def initial_style_primitive_replicas(params):
  return max(core.traverse_jaxpr_params(jaxpr_replicas, params).values(), default=1)


def needs_check_special():
  return config.jax_debug_infs or config.jax_debug_nans

def check_special(name, bufs):
  if needs_check_special():
    for buf in bufs:
      _check_special(name, buf.dtype, buf)

def _check_special(name, dtype, buf):
  if dtypes.issubdtype(dtype, np.inexact):
    if config.jax_debug_nans and np.any(np.isnan(np.asarray(buf))):
      raise FloatingPointError(f"invalid value (nan) encountered in {name}")
    if config.jax_debug_infs and np.any(np.isinf(np.asarray(buf))):
      raise FloatingPointError(f"invalid value (inf) encountered in {name}")


@profiler.annotate_function
def backend_compile(backend, module: ir.Module, options, host_callbacks):
  # Convert ir.Module to a string representation, unless the
  # back-end expliclity flags the ability to handle a module directly
  # (avoiding the overhead of back and forth conversions)
  if getattr(backend, "needs_str_ir", True):
    built_c = mlir.module_to_bytecode(module)
  else:
    built_c = module

  # we use a separate function call to ensure that XLA compilation appears
  # separately in Python profiling results
  if host_callbacks:
    return backend.compile(built_c, compile_options=options,
                           host_callbacks=host_callbacks)
  # Some backends don't have `host_callbacks` option yet
  # TODO(sharadmv): remove this fallback when all backends allow `compile`
  # to take in `host_callbacks`
  return backend.compile(built_c, compile_options=options)

_ir_dump_counter = itertools.count()

def _make_string_safe_for_filename(s: str) -> str:
  return re.sub(r'[^\w.)( -]', '', s)

def _dump_ir_to_file(name: str, ir: str):
  id = next(_ir_dump_counter)
  name = f"jax_ir{id}_{_make_string_safe_for_filename(name)}.mlir"
  name = path.Path(_DUMP_IR_TO.value) / name
  name.write_text(ir)


def compile_or_get_cached(backend, computation: ir.Module, devices: np.ndarray,
                          compile_options, host_callbacks):
  sym_name = computation.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value

  if _DUMP_IR_TO.value:
    _dump_ir_to_file(module_name, mlir.module_to_string(computation))

  # Persistent compilation cache only implemented on TPU and GPU.
  # TODO(skye): add warning when initializing cache on unsupported default platform
  supported_platforms = ["tpu", "gpu"]
  # (b/233850967) CPU caching can be enabled if XLA Runtime is enabled.
  if "--xla_cpu_use_xla_runtime=true" in os.environ.get("XLA_FLAGS", ""):
    supported_platforms.append("cpu")
  use_compilation_cache = (compilation_cache.is_initialized() and
                           backend.platform in supported_platforms)

  if not use_compilation_cache:
    return backend_compile(backend, computation, compile_options,
                           host_callbacks)

  cache_key = compilation_cache.get_cache_key(
      computation, devices, compile_options, backend)

  executable, compile_time_retrieved = _cache_read(
      module_name, cache_key, compile_options, backend)
  if executable is not None:
    # TODO(b/289098047): Will instrument a metric which uses the 'compile_time'
    # to measure the savings due to the cache hit.
    logger.info("Persistent compilation cache hit for '%s'", module_name)
    return executable
  else:
    start_time = time.monotonic()
    executable = backend_compile(backend, computation,
                                compile_options, host_callbacks)
    compile_time = time.monotonic() - start_time
    _cache_write(cache_key, compile_time, module_name, backend, executable,
                 host_callbacks)
    return executable


def _cache_read(
    module_name: str, cache_key: str, compile_options, backend
) -> tuple[xc.LoadedExecutable | None, int | None]:
  """Looks up the `computation` and it's compilation time in the persistent
  compilation cache repository.
  """
  try:
    return compilation_cache.get_executable_and_time(
        cache_key, compile_options, backend)
  except Exception as ex:
    if config.jax_raise_persistent_cache_errors:
      raise
    warnings.warn(
        f"Error reading persistent compilation cache entry for "
        f"'{module_name}': {type(ex).__name__}: {ex}")
    return None, None


def _cache_write(cache_key: str,
                 compile_time_secs: float,
                 module_name: str,
                 backend: Backend, executable: xc.LoadedExecutable,
                 host_callbacks: list[Any]):
  """Writes the `serialized_computation` and its compilation time to the
  persistent compilation cache repository.
  """
  if host_callbacks:
    logger.info(
        "Not writing persistent cache entry for '%s' because it uses host "
        "callbacks (e.g. from jax.debug.print or breakpoint)", module_name)
    return

  min_compile_time = config.jax_persistent_cache_min_compile_time_secs
  if min_compile_time:
    if compile_time_secs < min_compile_time:
      logger.info(
          "Not writing persistent cache entry for '%s' because it took < %.2f "
          "seconds to compile (%.2fs)", module_name, min_compile_time,
          compile_time_secs)
      return
    else:
      logger.info(
          "'%s' took at least %.2f seconds to compile (%.2fs), writing "
          "persistent cache entry", module_name, min_compile_time,
          compile_time_secs)

  try:
    compilation_cache.put_executable_and_time(
        cache_key, module_name, executable, backend, int(compile_time_secs))
  except Exception as ex:
    if config.jax_raise_persistent_cache_errors:
      raise
    warnings.warn(
        f"Error writing persistent compilation cache entry for "
        f"'{module_name}': {type(ex).__name__}: {ex}")


# TODO(yashkatariya): Generalize is_compatible_aval (maybe renamed) and use that
# to check if shardings are compatible with the input.
def _check_sharding(aval, s):
  from jax._src import pjit

  if isinstance(s, XLACompatibleSharding) and not isinstance(s, PmapSharding):
    pjit.pjit_check_aval_sharding(
        (s,), (aval,), None, "device_put args", allow_uneven_sharding=False)

  assert isinstance(aval, core.ShapedArray), aval
  s.shard_shape(aval.shape)  # should raise an Error if incompatible


def _put_x(x, s: Sharding, aval: core.AbstractValue, committed: bool):
  result_handler = pxla.global_aval_to_result_handler(aval, s, committed, False)
  map_ = s.devices_indices_map(aval.shape)  # type: ignore
  return result_handler(pxla.shard_arg(x, list(map_), list(map_.values()), s))

def _override_get_device_assignment(sharding, *args, **kwargs):
  da = sharding._device_assignment
  return xb.get_device_backend(da[0]), da

def _identity_fn(x):
  return x

def _mcjax_reshard(x, target_sharding):
  from jax._src import api, array

  inp_sharding = x.sharding

  if inp_sharding._device_assignment == target_sharding._device_assignment:
    return api.jit(_identity_fn, out_shardings=target_sharding)(x)

  if inp_sharding.device_set != target_sharding.device_set:
    inp_ids = [d.id for d in inp_sharding._device_assignment]
    inp_plat = inp_sharding._device_assignment[0].platform.upper()
    target_ids = [d.id for d in target_sharding._device_assignment]
    target_plat = target_sharding._device_assignment[0].platform.upper()
    raise ValueError("Input and target sharding should have the same set of "
                     f"devices. Got input's device set ids: {inp_ids} on "
                     f"platform {inp_plat} and target sharding's device set "
                     f"ids: {target_ids} on platform {target_plat}")

  old_op_sharding = inp_sharding._to_xla_hlo_sharding(x.ndim).to_proto()
  if op_shardings.is_op_sharding_replicated(old_op_sharding):
    new_op_sharding = old_op_sharding
  else:
    permute_order = np.vectorize(target_sharding._device_assignment.index,
                                 otypes=[int])(inp_sharding._device_assignment)
    new_op_sharding = old_op_sharding.clone()
    new_op_sharding.tile_assignment_devices = np.take(
        old_op_sharding.tile_assignment_devices, permute_order)

  new_x = array.make_array_from_single_device_arrays(
      x.shape,
      GSPMDSharding(target_sharding._device_assignment, new_op_sharding),
      x._arrays)

  _orig_get_and_check_device_assignment = pxla._get_and_check_device_assignment
  pxla._get_and_check_device_assignment = partial(
      _override_get_device_assignment, target_sharding)
  try:
    return api.jit(_identity_fn, out_shardings=target_sharding)(new_x)
  finally:
    pxla._get_and_check_device_assignment = _orig_get_and_check_device_assignment


def _device_put_impl(
    x,
    device: Device | Sharding | None = None,
    src: Device | Sharding | None = None):
  from jax._src import array
  try:
    aval = xla.abstractify(x)
  except TypeError as err:
    raise TypeError(
        f"Argument '{x}' of type {type(x)} is not a valid JAX type") from err

  if isinstance(device, Sharding):
    s = device
    _check_sharding(aval, s)
    if getattr(x, 'sharding', None) == s:
      return x
    if (not s.is_fully_addressable and  # type: ignore
        isinstance(x, array.ArrayImpl) and not x.is_fully_addressable):
      # This has to be XLACompatible because _mcjax_reshard will run a
      # XLA computation.
      assert isinstance(s, XLACompatibleSharding)
      return _mcjax_reshard(x, s)
    if not s.is_fully_addressable:  # type: ignore
      raise ValueError(
          "device_put's second argument must be a Device or a Sharding which "
          f"represents addressable devices, but got {s}")
    return _put_x(x, s, aval, True)

  # Only `Device` exists below. `Sharding` instance is handled above.
  if isinstance(x, array.ArrayImpl):
    if not x.is_fully_addressable:
      raise ValueError(
          "device_put's first argument must be a fully addressable array, but "
          f"got value with devices {x.devices()}")
    if device is None:
      return x
    elif is_single_device_sharding(x.sharding):
      return pxla.batched_device_put(aval, SingleDeviceSharding(device), [x],
                                     [device])

  sh = SingleDeviceSharding(pxla._get_default_device()
                            if device is None else device)
  return _put_x(x, sh, aval, device is not None)


device_put_p = core.Primitive('device_put')
device_put_p.def_impl(_device_put_impl)
device_put_p.def_abstract_eval(lambda x, device=None, src=None: x)

def device_put_transpose_rule(ct, _, device, src):
  return [device_put_p.bind(ct, device=src, src=device)]
ad.deflinear2(device_put_p, device_put_transpose_rule)
batching.defvectorized(device_put_p)

def _device_put_lowering(ctx, x, *, device, src):
  return [x]
mlir.register_lowering(device_put_p, _device_put_lowering)
