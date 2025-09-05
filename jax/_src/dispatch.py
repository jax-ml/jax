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
from collections.abc import Sequence
from functools import partial
import itertools
import logging
import threading
import time
from typing import Any

from jax._src import api
from jax._src import basearray
from jax._src import config
from jax._src import core
from jax._src import device_put
from jax._src import dtypes
from jax._src import lib
from jax._src import pjit
from jax._src import traceback_util
from jax._src import util

from jax._src.api_util import InternalFloatingPointError
from jax._src.lib import xla_client as xc
from jax._src.mesh import AbstractMesh, Mesh
from jax._src.monitoring import record_scalar, record_event_duration_secs, record_event_time_span
from jax._src.partition_spec import PartitionSpec
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (
    NamedSharding, GSPMDSharding)
from jax._src.stages import SourceInfo
import numpy as np


JAXPR_TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"
JAXPR_TO_MLIR_MODULE_EVENT = "/jax/core/compile/jaxpr_to_mlir_module_duration"
BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"

traceback_util.register_exclusion(__file__)

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

def apply_primitive(prim, *args, **params):
  """Impl rule that compiles and runs a single primitive 'prim' using XLA."""
  fun = xla_primitive_callable(prim, **params)
  # TODO(yashkatariya): Investigate adding is_primitive to jit and never
  # triggering the disable jit path instead of messing around with it here.
  prev = lib.jax_jit.swap_thread_local_state_disable_jit(False)
  try:
    outs = fun(*args)
  finally:
    lib.jax_jit.swap_thread_local_state_disable_jit(prev)
  return outs

# TODO(necula): this cache will contain strong references to all
# Jaxprs in `params` (for higher-order primitives).
# This is not immediately fixable by using
# util.multi_weakref_lru_cache, because the `params` (including the Jaxpr)
# are closed over in the `prim_fun` lambda. Leaving this fix for a later PR.
@util.cache()
def xla_primitive_callable(prim: core.Primitive, **params):
  util.test_event("xla_primitive_callable_cache_miss")
  def prim_fun(*args):
    with config.eager_constant_folding(False):
      return prim.bind(*args, **params)
  prim_fun.__name__ = prim.name
  prim_fun.__qualname__ = prim.name
  prim_fun._apply_primitive = True
  return api.jit(prim_fun)


def simple_impl(prim):
  prim.def_impl(partial(apply_primitive, prim))

RuntimeToken = Any

class RuntimeTokenSet(threading.local):
  """See docstring for effects.py module for the calling convention for tokens."""

  # For each ordered effect, the token returned by the last dispatched
  # computation, sharded over the devices in that computation.
  current_tokens: dict[core.Effect, core.Token]

  # For each device, the runtime token returned by the last dispatched
  # computation on that device.
  output_runtime_tokens: dict[Device, RuntimeToken]

  def __init__(self):
    self.current_tokens = {}
    self.output_runtime_tokens = {}

  def get_token_input(
      self, eff: core.Effect, devices: list[Device]
  ) -> core.Token:
    tok = self.current_tokens.get(eff, np.zeros(0, np.bool_))

    if isinstance(tok, core.Token):
      # The order of devices may change, so we need to reshard if necessary.
      # TODO(yueshengys): This might still be buggy in a multi-process SPMD
      # scenario. Revise the logic later. A distributed shutdown barrier inside
      # the XLA program may be needed.
      return api.device_put(
          tok, NamedSharding(Mesh(devices, 'x'), PartitionSpec('x')))

    # We only use replicated sharding for the first time when the token for the
    # order effect hasn't been created.
    s = GSPMDSharding.get_replicated(devices)
    sharded_tok = core.Token(
        device_put.shard_args(
            [s], [None], [xc.ArrayCopySemantics.REUSE_INPUT], [tok]
        )[0]
    )
    self.current_tokens[eff] = sharded_tok
    return sharded_tok

  def set_token_result(self, eff: core.Effect, token: core.Token):
    self.current_tokens[eff] = token

  def set_output_runtime_token(self, device: Device, token: RuntimeToken):
    # We're free to clobber the previous output token because on each
    # device we have a total ordering of computations. Only the token
    # from the latest computation matters.
    self.output_runtime_tokens[device] = token

  def clear(self):
    self.current_tokens = {}
    self.output_runtime_tokens = {}

  def block_until_ready(self):
    for token in self.current_tokens.values():
      token.block_until_ready()
    for token in self.output_runtime_tokens.values():
      token.block_until_ready()
    self.clear()

runtime_tokens: RuntimeTokenSet = RuntimeTokenSet()

@atexit.register
def wait_for_tokens():
  runtime_tokens.block_until_ready()


class LogElapsedTimeContextManager:
  __slots__ = ['fmt', 'fun_name', 'event', 'start_time']

  def __init__(self, fmt: str, fun_name: str, event: str | None = None):
    self.fmt = fmt
    self.fun_name = fun_name
    self.event = event

  def __enter__(self):
    self.start_time = time.time()
    if self.event is not None:
      record_scalar(
          self.event, self.start_time, fun_name=self.fun_name
      )

  def __exit__(self, exc_type, exc_value, traceback):
    if _on_exit:
      return

    end_time = time.time()
    elapsed_time = end_time - self.start_time
    log_priority = logging.WARNING if config.log_compiles.value else logging.DEBUG
    if logger.isEnabledFor(log_priority):
      logger.log(log_priority, self.fmt.format(
          fun_name=self.fun_name, elapsed_time=elapsed_time))
    if self.event is not None:
      record_event_duration_secs(
          self.event, elapsed_time, fun_name=self.fun_name
      )
      record_event_time_span(
          self.event, self.start_time, end_time, fun_name=self.fun_name
      )

log_elapsed_time = LogElapsedTimeContextManager


def should_tuple_args(num_args: int, platform: str) -> bool:
  # CPU and GPU do not need tuples as they use host-side data structures that
  # do not have small bounds.
  # TPU only needs a tuple for very long lists
  if platform == "tpu":
    return num_args > 2000
  else:
    return False

def jaxpr_has_primitive(jaxpr: core.Jaxpr, prim_name: str) -> bool:
  """Whether there is a primitive given by user anywhere inside a Jaxpr."""
  for eqn in jaxpr.eqns:
    if prim_name in eqn.primitive.name:
      return True
  for subjaxpr in core.subjaxprs(jaxpr):
    if jaxpr_has_primitive(subjaxpr, prim_name):
      return True
  return False


# Use this registry with caution. It will void the guarantee that lowering to
# stablehlo is oblivious of physical devices.
prim_requires_devices_during_lowering: set[core.Primitive] = set()

@util.weakref_lru_cache
def jaxpr_has_prim_requiring_devices(jaxpr: core.Jaxpr) -> bool:
  for eqn in jaxpr.eqns:
    if eqn.primitive in prim_requires_devices_during_lowering:
      return True
  for subjaxpr in core.subjaxprs(jaxpr):
    if jaxpr_has_prim_requiring_devices(subjaxpr):
      return True
  return False


@util.weakref_lru_cache
def get_intermediate_shardings(
    jaxpr: core.Jaxpr) -> Sequence[tuple[Sharding, SourceInfo]]:
  from jax._src import shard_map  # pytype: disable=import-error

  out = []
  for eqn in jaxpr.eqns:
    if eqn.primitive is pjit.sharding_constraint_p:
      s = eqn.params['sharding']
      if isinstance(s, NamedSharding) and isinstance(s.mesh, AbstractMesh):
        continue
      source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
      out.append((s, source_info))
    elif eqn.primitive is pjit.jit_p:
      source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
      out.extend((i, source_info) for i in eqn.params['in_shardings'])
      out.extend((o, source_info) for o in eqn.params['out_shardings'])
    elif eqn.primitive is shard_map.shard_map_p:
      mesh = eqn.params['mesh']
      if isinstance(mesh, AbstractMesh):
        continue
      source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
      out.extend((NamedSharding(mesh, spec), source_info)
                 for spec in [*eqn.params['in_specs'], *eqn.params['out_specs']])
    elif eqn.primitive is device_put.device_put_p:
      source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
      out.extend((s, source_info) for s in eqn.params['devices']
                 if isinstance(s, Sharding) and s.memory_kind is not None)
  for subjaxpr in core.subjaxprs(jaxpr):
    out.extend(get_intermediate_shardings(subjaxpr))
  return out


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


def check_arg(arg: Any):
  if not (isinstance(arg, core.Tracer) or core.valid_jaxtype(arg)):
    raise TypeError(f"Argument '{arg}' of type {type(arg)} is not a valid "
                    "JAX type.")


def needs_check_special() -> bool:
  return config.debug_infs.value or config.debug_nans.value

def check_special(name: str, bufs: Sequence[basearray.Array]) -> None:
  if needs_check_special():
    for buf in bufs:
      _check_special(name, buf.dtype, buf)

def _check_special(name: str, dtype: np.dtype, buf: basearray.Array) -> None:
  if dtypes.issubdtype(dtype, np.inexact):
    if config.debug_nans.value and np.any(np.isnan(np.asarray(buf))):
      raise InternalFloatingPointError(name, "nan")
    if config.debug_infs.value and np.any(np.isinf(np.asarray(buf))):
      raise InternalFloatingPointError(name, "inf")

