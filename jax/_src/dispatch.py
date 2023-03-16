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
import contextlib
from functools import partial
import itertools
import time
from typing import (
    Any, Callable, Dict, Iterator, Optional, Protocol,
    Sequence, Set, Tuple, List, Type, Union, NamedTuple)
import logging
import os
import re
import threading
import warnings

import numpy as np

import jax
from jax.monitoring import record_event_duration_secs
import jax.interpreters.mlir as mlir
import jax.interpreters.partial_eval as pe

from jax._src import array
from jax._src import core
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import path
from jax._src import profiler
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.config import config, flags
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import xla
from jax._src.interpreters import pxla
from jax._src.lib.mlir import ir
from jax._src.lib import xla_client as xc
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (
    PmapSharding, SingleDeviceSharding, NamedSharding,
    PartitionSpec, XLACompatibleSharding)
from jax._src.util import flatten


JAXPR_TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"
BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'jax_dump_ir_to', os.getenv('JAX_DUMP_IR_TO', ''),
    help="Path to which the IR that is emitted by JAX as input to the "
         "compiler should be dumped as text files. Optional. If omitted, JAX "
         "will not dump IR.")


traceback_util.register_exclusion(__file__)

MYPY = False  # Are we currently type checking with mypy?

xe = xc._xla

Backend = xe.Client
Device = xc.Device
Buffer = xe.Buffer

CompileOptions = xc.CompileOptions

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

logger = logging.getLogger(__name__)

# This flag is set on exit; no logging should be attempted
_on_exit = False

### op-by-op execution

ArgSpec = Tuple[core.AbstractValue, Optional[Device]]

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


def apply_primitive(prim, *args, **params):
  """Impl rule that compiles and runs a single primitive 'prim' using XLA."""
  compiled_fun = xla_primitive_callable(prim, *unsafe_map(arg_spec, args),
                                        **params)
  return compiled_fun(*args)

def simple_impl(prim):
  prim.def_impl(partial(apply_primitive, prim))

RuntimeToken = Any

class RuntimeTokenSet(threading.local):
  tokens: Dict[core.Effect, Tuple[RuntimeToken, Device]]
  output_tokens: Dict[Device, RuntimeToken]
  output_runtime_tokens: Dict[Device, RuntimeToken]

  def __init__(self):
    self.tokens = {}
    # TODO(sharadmv): remove redundant output token dictionary when minimum
    # jaxlib version is bumped to 0.3.16.
    self.output_tokens = {}
    self.output_runtime_tokens = {}

  def get_token(self, eff: core.Effect, device: Device) -> RuntimeToken:
    s = jax.sharding.SingleDeviceSharding(device)
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
def xla_primitive_callable(prim, *arg_specs: ArgSpec, **params):
  donated_invars = (False,) * len(arg_specs)
  def prim_fun(*args):
    out = prim.bind(*args, **params)
    if prim.multiple_results:
      return out
    else:
      return out,
  compiled = _xla_callable_uncached(lu.wrap_init(prim_fun), None, None,
                                    prim.name, donated_invars, False, *arg_specs)
  if not prim.multiple_results:
    return lambda *args, **kw: compiled(*args, **kw)[0]
  else:
    return compiled


def _device_from_arg_devices(devices: Sequence[Optional[Device]]) -> Optional[Device]:
  """Given devices of inputs, determine where to perform a computation.

  Args:
    devices: list where each element is a either a `Device` instance or `None`.
  Returns:
    A `Device` instance or None.
  Raises:
    ValueError if input devices are inconsistent.
  """
  try:
    device, = {d for d in devices if d is not None} or (None,)
    return device
  except ValueError as err:
    msg = "primitive arguments must be colocated on the same device, got {}"
    raise ValueError(msg.format(", ".join(map(str, devices)))) from err



def sharded_lowering(fun, device, backend, name, donated_invars, always_lower,
                     keep_unused, *arg_specs,
                     lowering_platform: Optional[str]):
  in_avals, in_shardings = util.unzip2(arg_specs)
  in_shardings = [pxla._UNSPECIFIED if i is None else i for i in in_shardings]  # type: ignore

  # Pass in a singleton `_UNSPECIFIED` for out_shardings because we don't know
  # the number of output avals at this stage. lower_sharding_computation will
  # apply it to all out_avals.
  return pxla.lower_sharding_computation(
      fun, 'jit', name, in_shardings, pxla._UNSPECIFIED, donated_invars,
      in_avals, in_is_global=(True,) * len(arg_specs), keep_unused=keep_unused,
      always_lower=always_lower, devices_from_context=None,
      lowering_platform=lowering_platform)


def _xla_callable_uncached(fun: lu.WrappedFun, device, backend, name,
                           donated_invars, keep_unused, *arg_specs):
  computation = sharded_lowering(fun, device, backend, name, donated_invars,
                                  False, keep_unused, *arg_specs,
                                  lowering_platform=None)
  allow_prop = [True] * len(computation.compile_args['global_out_avals'])
  return computation.compile(_allow_propagation_to_outputs=allow_prop).unsafe_call


def is_single_device_sharding(sharding) -> bool:
  # Special case PmapSharding here because PmapSharding maps away an axis
  # and needs to be handled separately.test_pjit_single_device_sharding_add
  return len(sharding.device_set) == 1 and not isinstance(sharding, PmapSharding)


@contextlib.contextmanager
def log_elapsed_time(fmt: str, event: Optional[str] = None):
  if _on_exit:
    yield
  else:
    log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    logger.log(log_priority, fmt.format(elapsed_time=elapsed_time))
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
    jaxpr) -> Iterator[Tuple[jax.sharding.XLACompatibleSharding, SourceInfo]]:
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
                  for names in eqn.params['in_names'])
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
    jaxpr: core.Jaxpr) -> Tuple[core.Jaxpr, Set[int], Set[int]]:
  used_outputs = [True] * len(jaxpr.outvars)
  new_jaxpr, used_consts, used_inputs = pe.dce_jaxpr_consts(jaxpr, used_outputs)
  kept_const_idx = {i for i, b in enumerate(used_consts) if b}
  kept_var_idx = {i for i, b in enumerate(used_inputs) if b}
  return new_jaxpr, kept_const_idx, kept_var_idx


# We can optionally set a Jaxpr rewriter that can be applied just before
# compilation. This mechanism is used for compiling id_tap, we can
# remove it once we bring the id_tap implementation into the core.
outfeed_rewriter: Optional[Callable[[core.Jaxpr], core.Jaxpr]] = None
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


# Argument and result handlers

num_buffers_handlers: Dict[Type[core.AbstractValue],
                           Callable[[core.AbstractValue], int]] = {}

def aval_to_num_buffers(aval: core.AbstractValue) -> int:
  """Returns the number of buffers in the runtime representation of `aval`.

  In general this may differ from the number of buffers in the compiler-IR
  representation of the same value.
  """
  try:
    return num_buffers_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No num_buffers handler for type: {type(aval)}") from err

num_buffers_handlers[core.AbstractToken] = lambda _: 1
num_buffers_handlers[core.ShapedArray] = lambda _: 1
num_buffers_handlers[core.DShapedArray] = lambda _: 1
num_buffers_handlers[core.ConcreteArray] = lambda _: 1


def _input_handler(backend: Backend,
                   in_type: Optional[pe.InputType],
                   out_type: Optional[pe.OutputType],
                   ) -> Optional[Callable]:
  if in_type is None:
    assert out_type is None
    return None
  in_avals, which_explicit = util.unzip2(in_type)
  # Check whether we actually need an input_handler.
  needs_implicit = which_explicit and not all(which_explicit)
  needs_out_handling = any(type(d) is core.InDBIdx for a, _ in out_type or []
                           if type(a) is core.DShapedArray for d in a.shape)

  if not needs_implicit and not needs_out_handling:
    return None
  assert config.jax_dynamic_shapes

  # Precompute how to grab implicit inputs from explicit inputs' axis sizes.
  which_explicit = which_explicit or (True,) * len(in_avals)
  implicit_idxs = {i for i, ex in enumerate(which_explicit) if not ex}
  implicit_args_from_axes: List[Tuple[int, int, int]] = []
  for arg_idx, aval in enumerate(in_avals):
    if isinstance(aval, core.DShapedArray):
      for axis_idx, d in enumerate(aval.shape):
        if isinstance(d, core.DBIdx) and d.val in implicit_idxs:
          implicit_args_from_axes.append((d.val, arg_idx, axis_idx))
  assert {i for i, _, _ in implicit_args_from_axes} == implicit_idxs

  # Precompute which input values are needed for output types.
  inputs_needed_for_out_types = out_type and [
      d.val for aval, _ in out_type if type(aval) is core.DShapedArray  # type: ignore
      for d in aval.shape if type(d) is core.InDBIdx]

  def elaborate(explicit_args: Sequence[Any]) -> Tuple[Tuple, Optional[Tuple]]:
    if needs_implicit:
      # Build full argument list, leaving Nones for implicit arguments.
      explicit_args_ = iter(explicit_args)
      args = [next(explicit_args_) if ex else None for ex in which_explicit]
      assert next(explicit_args_, None) is None
      # Populate implicit arguments.
      for i, j, k in implicit_args_from_axes:
        if args[i] is None:
          args[i] = args[j].shape[k]  # type: ignore
        else:
          if args[i] != args[j].shape[k]:
            raise Exception("inconsistent argument axis sizes for type")
    else:
      args = list(explicit_args)

    if needs_out_handling:
      # Make a list of inputs needed by output types, leaving unneeded as None.
      out_type_env = [None] * len(args)
      for i in inputs_needed_for_out_types or []:
        out_type_env[i] = args[i]
    else:
      out_type_env = None  # type: ignore

    return tuple(args), out_type_env and tuple(out_type_env)  # type: ignore
  return elaborate

def _result_handler(backend: Backend,
                    sticky_device: Optional[Device],
                    out_type: pe.OutputType,
                    ) -> Callable:
  out_avals, kept_outputs = util.unzip2(out_type)
  handlers = map(partial(aval_to_result_handler, sticky_device), out_avals)
  dyn_outs = any(type(aval) is core.DShapedArray and
                 any(type(d) in (core.InDBIdx, core.OutDBIdx) for d in aval.shape)
                 for aval in out_avals)
  if not dyn_outs:
    return SimpleResultHandler(handlers)
  assert config.jax_dynamic_shapes

  def result_handler(input_env, lists_of_bufs):
    results = []
    for handler, bufs in unsafe_zip(handlers, lists_of_bufs):
      results.append(handler((input_env, results), *bufs))
    return [r for r, keep in unsafe_zip(results, kept_outputs) if keep]
  return result_handler

class SimpleResultHandler:
  handlers: Sequence[ResultHandler]
  def __init__(self, handlers): self.handlers = handlers
  def __iter__(self): return iter(self.handlers)
  def __len__(self): return len(self.handlers)
  def __call__(self, env, lists_of_bufs):
    return tuple(h(env, *bs) for h, bs in zip(self.handlers, lists_of_bufs))


def maybe_create_array_from_da(buf, aval, device):
  return array.ArrayImpl(aval, SingleDeviceSharding(buf.device()), [buf],
                          committed=(device is not None), _skip_checks=True)


if MYPY:
  ResultHandler = Any
else:
  class ResultHandler(Protocol):
    def __call__(self, env: Optional[Sequence[Any]], *args: xc.Buffer) -> Any:
      """Boxes raw buffers into their user-facing representation."""

def aval_to_result_handler(sticky_device: Optional[Device],
                           aval: core.AbstractValue) -> ResultHandler:
  try:
    return result_handlers[type(aval)](sticky_device, aval)
  except KeyError as err:
    raise TypeError(f"No result handler for type: {type(aval)}") from err

def array_result_handler(sticky_device: Optional[Device],
                         aval: core.ShapedArray):
  if not core.is_opaque_dtype(aval.dtype) and aval.dtype == dtypes.float0:
    return lambda _, __: np.zeros(aval.shape, dtypes.float0)
  aval = core.raise_to_shaped(aval)
  if core.is_opaque_dtype(aval.dtype):
    return aval.dtype._rules.result_handler(sticky_device, aval)
  handler = lambda _, b: maybe_create_array_from_da(b, aval, sticky_device)
  handler.args = aval, sticky_device  # for C++ dispatch path in api.py
  return handler

def dynamic_array_result_handler(sticky_device: Optional[Device],
                                 aval: core.DShapedArray):
  if not core.is_opaque_dtype(aval.dtype) and aval.dtype == dtypes.float0:
    return lambda _: np.zeros(aval.shape, dtypes.float0)  # type: ignore
  else:
    return partial(_dynamic_array_result_handler, sticky_device, aval)

def _dynamic_array_result_handler(sticky_device, aval, env, buf):
  in_env, out_env = env or (None, None)
  shape = [in_env[d.val] if type(d) is core.InDBIdx else
           out_env[d.val] if type(d) is core.OutDBIdx else d
           for d in aval.shape]
  if all(type(d) is int for d in shape) and type(aval.dtype) is not core.bint:
    aval = core.ShapedArray(tuple(shape), aval.dtype)
    return maybe_create_array_from_da(buf, aval, sticky_device)
  else:
    pad_shape = [d.dtype.bound if _is_bint_axis_size(d) else d for d in shape]
    buf_dtype = (aval.dtype if not core.is_opaque_dtype(aval.dtype) else
                 aval.dtype._rules.physical_avals(aval)[0])
    buf_aval = core.ShapedArray(tuple(pad_shape), buf_dtype, aval.weak_type)
    data = maybe_create_array_from_da(buf, buf_aval, sticky_device)
    return core.DArray(aval.update(shape=tuple(shape)), data)


result_handlers: Dict[
    Type[core.AbstractValue],
    Callable[[Optional[Device], Any], ResultHandler]] = {}
result_handlers[core.AbstractToken] = lambda _, __: lambda _, __: core.token
result_handlers[core.ShapedArray] = array_result_handler
result_handlers[core.DShapedArray] = dynamic_array_result_handler
result_handlers[core.ConcreteArray] = array_result_handler


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

def _add_tokens(has_unordered_effects: bool, ordered_effects: List[core.Effect],
                has_host_callbacks: bool, device: Device, input_bufs):
  tokens = [runtime_tokens.get_token(eff, device) for eff in ordered_effects]
  tokens_flat = flatten(tokens)
  input_bufs = [*tokens_flat, *input_bufs]
  def _remove_tokens(output_bufs, runtime_token):
    num_output_tokens = len(ordered_effects)
    token_bufs, output_bufs = util.split_list(output_bufs, [num_output_tokens])
    if has_unordered_effects or has_host_callbacks:
      runtime_tokens.set_output_runtime_token(device, runtime_token)
    for eff, token_buf in zip(ordered_effects, token_bufs):
      runtime_tokens.update_token(eff, token_buf)
    return output_bufs
  return input_bufs, _remove_tokens


@profiler.annotate_function
def backend_compile(backend, built_c, options, host_callbacks):
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
  name = path.Path(FLAGS.jax_dump_ir_to) / name
  name.write_text(ir)


def compile_or_get_cached(backend, computation: ir.Module, compile_options,
                          host_callbacks):
  # Avoid import cycle between jax and jax.experimental
  from jax.experimental.compilation_cache import compilation_cache as cc

  sym_name = computation.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value

  if FLAGS.jax_dump_ir_to:
    _dump_ir_to_file(module_name, mlir.module_to_string(computation))

  # Convert ir.Module to a string representation, unless the
  # back-end expliclity flags the ability to handle a module directly
  # (avoiding the overhead of back and forth conversions)
  serialized_computation: Union[str, bytes, ir.Module]
  if getattr(backend, "needs_str_ir", True):
    serialized_computation = mlir.module_to_bytecode(computation)
  else:
    serialized_computation = computation

  # Persistent compilation cache only implemented on TPU and GPU.
  # TODO(skye): add warning when initializing cache on unsupported default platform
  supported_platforms = ["tpu", "gpu"]
  # (b/233850967) CPU caching can be enabled if XLA Runtime is enabled.
  if "--xla_cpu_use_xla_runtime=true" in os.environ.get("XLA_FLAGS", ""):
    supported_platforms.append("cpu")
  if cc.is_initialized() and backend.platform in supported_platforms:
    cached_executable = _cache_read(serialized_computation, module_name,
                                    compile_options, backend)
    if cached_executable is not None:
      logger.info("Persistent compilation cache hit for '%s'", module_name)
      return cached_executable
    else:
      start_time = time.monotonic()
      compiled = backend_compile(backend, serialized_computation,
                                 compile_options, host_callbacks)
      compile_time = time.monotonic() - start_time
      _cache_write(serialized_computation, compile_time, module_name,
                   compile_options, backend, compiled, host_callbacks)
      return compiled

  return backend_compile(backend, serialized_computation, compile_options,
                         host_callbacks)


def _cache_read(computation: Union[str, bytes, ir.Module], module_name: str,
                compile_options: CompileOptions,
                backend: Backend) -> Optional[xc.LoadedExecutable]:
  """Looks up `computation` in the persistent compilation cache."""
  # Avoid import cycle between jax and jax.experimental
  from jax.experimental.compilation_cache import compilation_cache as cc

  try:
    return cc.get_executable(computation, compile_options, backend)
  except Exception as ex:
    if config.jax_raise_persistent_cache_errors:
      raise
    warnings.warn(
        f"Error reading persistent compilation cache entry for "
        f"'{module_name}': {type(ex).__name__}: {ex}")
    return None


def _cache_write(serialized_computation: Union[str, bytes, ir.Module],
                 compile_time_secs: float,
                 module_name: str, compile_options: CompileOptions,
                 backend: Backend, compiled: xc.LoadedExecutable,
                 host_callbacks: List[Any]):
  """Writes `serialized_computation` to the persistent compilation cache."""
  # Avoid import cycle between jax and jax.experimental
  from jax.experimental.compilation_cache import compilation_cache as cc

  if host_callbacks:
    logger.info(
        "Not writing persistent cache entry for '%s' because it uses host "
        "callbacks (e.g. from jax.debug.print or breakpoint)")
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
    cc.put_executable(module_name, serialized_computation, compile_options,
                      compiled, backend)
  except Exception as ex:
    if config.jax_raise_persistent_cache_errors:
      raise
    warnings.warn(
        f"Error writing persistent compilation cache entry for "
        f"'{module_name}': {type(ex).__name__}: {ex}")


def _set_aval(val):
  if val.aval is None:
    val.aval = core.ShapedArray(val.shape, val.dtype)
  return val


# TODO(yashkatariya): Generalize is_compatible_aval (maybe renamed) and use that
# to check if shardings are compatible with the input.
def _check_sharding(aval, s):
  from jax._src import pjit

  if isinstance(s, XLACompatibleSharding) and not isinstance(s, PmapSharding):
    pjit.pjit_check_aval_sharding(
        (s,), (aval,), "device_put args", allow_uneven_sharding=False)

  s.shard_shape(aval.shape)  # should raise an Error if incompatible


def _put_x(x, s: Sharding, aval: core.AbstractValue, committed: bool):
  result_handler = pxla.global_aval_to_result_handler(aval, s, committed, False)
  map_ = s.devices_indices_map(aval.shape)  # type: ignore
  return result_handler(pxla.shard_arg(x, list(map_), list(map_.values()), s))


def _device_put_impl(
    x, device: Optional[Union[Device, jax.sharding.Sharding]] = None):
  try:
    aval = xla.abstractify(x)
  except TypeError as err:
    raise TypeError(
        f"Argument '{x}' of type {type(x)} is not a valid JAX type") from err

  if isinstance(device, Sharding):
    s = device
    if not s.is_fully_addressable:  # type: ignore
      raise ValueError(
          "device_put's second argument must be a Device or a Sharding which "
          f"represents addressable devices, but got {s}")

    _check_sharding(aval, s)

    if getattr(x, 'sharding', None) == s:
      return x
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
device_put_p.def_abstract_eval(lambda x, device=None: x)
ad.deflinear2(device_put_p, lambda cotangent, _, **kwargs: [cotangent])
batching.defvectorized(device_put_p)

def _device_put_lowering(ctx, x, *, device):
  return [x]


mlir.register_lowering(device_put_p, _device_put_lowering)
