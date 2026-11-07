# Copyright 2023 The JAX Authors.
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

"""Contains registrations for pallas_call on TPU."""

from __future__ import annotations

import array
import collections
from collections.abc import Callable, Hashable, Mapping, Sequence
import dataclasses
import enum
import hashlib
import json
import logging
import threading
from typing import Any, cast
import weakref

import jax
from jax._src import config
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import effects as effects_lib
from jax._src import flattree as ft
from jax._src import frozen_dict
from jax._src import sharding_impls
from jax._src import state
from jax._src import tpu_custom_call
from jax._src import util
from jax._src import xla_metadata
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager
from jax._src.pallas import core as pallas_core
from jax._src.pallas import mpmd
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import helpers
from jax._src.pallas.mosaic import lowering
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax._src.pallas.mosaic import sc_core
from jax._src.pallas.mosaic import sc_lowering  # noqa: F401
from jax._src.pallas.mosaic import tpu_info
from jax._src.state import types as state_types
from jax.experimental import mosaic
from jax.experimental.mosaic.dialects import tpu
import numpy as np

logger = logging.getLogger(__name__)


class _UncacheableKernelError(Exception):
  """A kernel has a component that cannot be keyed structurally."""


# Constants larger than this make a kernel uncacheable instead of being hashed.
_MAX_HASHED_CONST_BYTES = 1 << 20
_MAX_LEAF_DEPTH = 32
_MAX_INTERNED_LEAVES = 1 << 20


def _include_metadata_in_key() -> bool:
  # Same policy as the persistent compilation cache: unless metadata is part of
  # cache keys, kernels that differ only in source locations share a lowering,
  # so the reused kernel can carry stale debug locations.
  return config.compilation_cache_include_metadata_in_key.value


class _KernelFingerprinter:
  """Computes structural fingerprints of kernel jaxprs.

  Pallas traces the kernel body anew at every call site, so identical kernels
  arrive as distinct jaxpr objects, and identity keyed lowering caches miss.
  This class maps each leaf (aval, parameter value, literal, primitive, ...) to
  a process unique integer by exact equality, and fingerprints a jaxpr by
  hashing its structure over those integers. Integers are never reused, so two
  jaxprs with equal fingerprints have equal structure, parameters, literals and
  constants; they differ at most in source information, which only feeds debug
  locations. Evicting a leaf only turns later cache hits into misses.
  """

  def __init__(self):
    self._lock = threading.RLock()
    self._leaf_ids: collections.OrderedDict[Hashable, int] = (
        collections.OrderedDict()
    )
    self._next_leaf_id = 0
    # Jaxprs are immutable, so their fingerprints can be memoized, separately
    # for fingerprints with and without source information.
    self._jaxpr_fingerprints: dict[
        bool, weakref.WeakKeyDictionary[jax_core.Jaxpr, bytes]
    ] = {False: weakref.WeakKeyDictionary(), True: weakref.WeakKeyDictionary()}
    self._include_metadata = False

  def _intern(self, leaf: Hashable) -> int:
    leaf_ids = self._leaf_ids
    try:
      leaf_id = leaf_ids.get(leaf)
    except Exception as e:  # pylint: disable=broad-except
      # Some objects define __eq__ that raises, e.g. elementwise on arrays.
      raise _UncacheableKernelError(f"cannot compare {type(leaf)}") from e
    if leaf_id is None:
      leaf_id = self._next_leaf_id
      self._next_leaf_id += 1
      leaf_ids[leaf] = leaf_id
      if len(leaf_ids) > _MAX_INTERNED_LEAVES:
        leaf_ids.popitem(last=False)
    else:
      leaf_ids.move_to_end(leaf)
    return leaf_id

  def cache_clear(self) -> None:
    # Leaf ids keep counting up, so ids handed out earlier are never reused.
    with self._lock:
      self._leaf_ids.clear()
      for memo in self._jaxpr_fingerprints.values():
        memo.clear()

  def cache_info(self) -> dict[str, int]:
    with self._lock:
      return dict(
          leaves=len(self._leaf_ids),
          jaxprs=sum(map(len, self._jaxpr_fingerprints.values())),
      )

  def cache_keys(self) -> list[Hashable]:
    with self._lock:
      return list(self._leaf_ids)

  def leaf(self, value: Any) -> Hashable:
    """Returns a hashable whose equality is exact equality of `value`."""
    with self._lock:
      self._include_metadata = _include_metadata_in_key()
      return self._leaf(value, 0)

  def jaxpr_fingerprint(self, jaxpr: jax_core.Jaxpr) -> bytes:
    with self._lock:
      self._include_metadata = _include_metadata_in_key()
      return self._jaxpr_fingerprint(jaxpr)

  def _leaf(self, value: Any, depth: int) -> Hashable:
    if depth > _MAX_LEAF_DEPTH:
      raise _UncacheableKernelError("parameter nesting is too deep")
    depth += 1
    t = type(value)
    # Tag every leaf with its type: equal values of different types, such as
    # 1, 1.0 and True, can lower differently.
    if value is None or t in (bool, int, str, bytes):
      return (t, value)
    if t is float:
      # float.hex separates 0.0 from -0.0 and makes NaN equal to itself.
      return (t, value.hex())
    if t is complex:
      return (t, value.real.hex(), value.imag.hex())
    if t is tuple or t is list:
      return (t, tuple(self._leaf(v, depth) for v in value))
    if isinstance(value, jax_core.Jaxpr):
      return (jax_core.Jaxpr, self._jaxpr_fingerprint(value))
    if isinstance(value, (np.ndarray, np.generic)):
      if value.dtype.hasobject:
        raise _UncacheableKernelError("object arrays are not hashable")
      return (
          t,
          value.dtype,
          value.shape,
          value.tobytes(),
          getattr(value, "aval", None),
      )
    if isinstance(value, (int, float, complex)):
      # Subclasses of Python scalars, such as typed literals.
      if isinstance(value, complex):
        raw = (complex(value).real.hex(), complex(value).imag.hex())
      elif isinstance(value, float):
        raw = float(value).hex()
      else:
        raw = int(value)
      return (t, getattr(value, "dtype", None), raw)
    if isinstance(value, enum.Enum):
      return (t, value.name)
    if isinstance(value, tuple):
      return (t, tuple(self._leaf(v, depth) for v in value))
    if isinstance(value, Mapping):
      return (
          t,
          tuple(
              (self._leaf(k, depth), self._leaf(v, depth))
              for k, v in value.items()
          ),
      )
    if isinstance(value, (set, frozenset)):
      return (t, frozenset(self._leaf(v, depth) for v in value))
    if isinstance(value, jax_core.Tracer):
      # Some primitives keep the closed jaxprs of nested bodies in their
      # parameters and pass the constants of those bodies as operands, e.g.
      # emit_pipeline. The constants left in the parameter are tracers of the
      # finished kernel trace, so lowering can only depend on their types.
      return (jax_core.Tracer, value.aval)
    if isinstance(value, jax.Array):
      if (
          not value.is_fully_addressable
          or value.size * value.dtype.itemsize > _MAX_HASHED_CONST_BYTES
      ):
        raise _UncacheableKernelError(
            f"array of type {jax_core.typeof(value)} is too large to hash"
        )
      try:
        host_value = np.asarray(value)
      except Exception as e:  # e.g. extended dtypes
        raise _UncacheableKernelError("array has no host value") from e
      return (t, jax_core.typeof(value), self._leaf(host_value, depth))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
      return (
          t,
          tuple(
              self._leaf(getattr(value, f.name), depth)
              for f in dataclasses.fields(value)
          ),
      )
    try:
      hash(value)
    except TypeError as e:
      raise _UncacheableKernelError(f"unhashable {t}") from e
    # Avals, primitives, meshes, treedefs, interned eqn contexts and callables
    # define exact (or identity) equality.
    return (t, value)

  def _atom(self, atom: jax_core.Atom, var_ids: dict[jax_core.Var, int]) -> int:
    if type(atom) is jax_core.Literal:
      return -1 - self._intern((atom.aval, self._leaf(atom.val, 0)))
    try:
      return var_ids[atom]
    except KeyError:
      raise _UncacheableKernelError(
          f"jaxpr refers to an unbound var of type {atom.aval}"
      ) from None

  def _effects(
      self, effects: jax_core.Effects, var_ids: dict[jax_core.Var, int]
  ) -> int:
    if not effects:
      return -1
    leaves = []
    for effect in effects:
      if isinstance(effect, effects_lib.JaxprInputEffect) and isinstance(
          effect.input, jax_core.Var
      ):
        # The input is a var of this jaxpr, which only matters by position.
        leaves.append((type(effect), self._atom(effect.input, var_ids)))
      else:
        leaves.append(self._leaf(effect, 0))
    return self._intern(frozenset(leaves))

  def _jaxpr_fingerprint(self, jaxpr: jax_core.Jaxpr) -> bytes:
    include_metadata = self._include_metadata
    memo = self._jaxpr_fingerprints[include_metadata]
    fingerprint = memo.get(jaxpr)
    if fingerprint is not None:
      return fingerprint
    intern = self._intern
    var_ids: dict[jax_core.Var, int] = {}
    tokens: list[int] = []
    put = tokens.append
    if include_metadata:
      # Argument names become locations of the kernel's block arguments.
      put(hash(tuple(jaxpr.debug_info.arg_names or ())))
    consts = jaxpr.consts
    put(len(consts))
    for v, c in zip(jaxpr.constvars, consts):
      var_ids[v] = len(var_ids)
      put(intern(v.aval))
      put(intern(self._leaf(c, 0)))
    put(len(jaxpr.invars))
    for v in jaxpr.invars:
      var_ids[v] = len(var_ids)
      put(intern(v.aval))
    put(len(jaxpr.eqns))
    for eqn in jaxpr.eqns:
      put(intern(eqn.primitive))
      put(len(eqn.invars))
      for a in eqn.invars:
        put(self._atom(a, var_ids))
      put(len(eqn.outvars))
      for v in eqn.outvars:
        if type(v) is jax_core.DropVar:
          put(-1 - intern(v.aval))
        else:
          var_ids[v] = len(var_ids)
          put(intern(v.aval))
      params = eqn.params
      ignored = lowering.params_ignored_by_lowering.get(eqn.primitive, ())
      put(
          intern(
              tuple(
                  (k, self._leaf(params[k], 0))
                  for k in sorted(params)
                  if k not in ignored
              )
          )
      )
      # XLA metadata only annotates the custom call in the enclosing HLO
      # module; Mosaic lowering never reads it.
      put(intern(eqn.ctx.configs))
      put(self._effects(eqn.effects, var_ids))
      if include_metadata:
        # Source information only feeds debug locations, so a 64 bit hash is
        # enough: a collision can only make metadata stale.
        put(hash(eqn.source_info.traceback))
        put(hash(eqn.source_info.name_stack))
    put(len(jaxpr.outvars))
    for a in jaxpr.outvars:
      put(self._atom(a, var_ids))
    put(self._effects(jaxpr.effects, var_ids))
    put(int(jaxpr.is_high))
    fingerprint = hashlib.blake2b(
        array.array("q", tokens).tobytes(), digest_size=32
    ).digest()
    memo[jaxpr] = fingerprint
    return fingerprint


class _MosaicLoweringCache:
  """A process wide LRU cache of serialized Mosaic kernels."""

  def __init__(self):
    self._lock = threading.Lock()
    self._entries: collections.OrderedDict[
        Hashable, tpu_custom_call.SerializedMosaicModule
    ] = collections.OrderedDict()
    self._total_bytes = 0
    self.hits = 0
    self.misses = 0

  @staticmethod
  def _size(value: tpu_custom_call.SerializedMosaicModule) -> int:
    return len(value.asm) + len(value.debug_locations or b"")

  def get(self, key: Hashable) -> tpu_custom_call.SerializedMosaicModule | None:
    with self._lock:
      value = self._entries.get(key)
      if value is None:
        self.misses += 1
      else:
        self.hits += 1
        self._entries.move_to_end(key)
      return value

  def put(
      self,
      key: Hashable,
      value: tpu_custom_call.SerializedMosaicModule,
      max_bytes: int,
  ) -> None:
    with self._lock:
      if key in self._entries:
        return
      self._entries[key] = value
      self._total_bytes += self._size(value)
      while self._total_bytes > max_bytes and self._entries:
        _, evicted = self._entries.popitem(last=False)
        self._total_bytes -= self._size(evicted)

  def cache_clear(self) -> None:
    with self._lock:
      self._entries.clear()
      self._total_bytes = 0
      self.hits = 0
      self.misses = 0

  def cache_info(self) -> dict[str, int]:
    with self._lock:
      return dict(
          hits=self.hits,
          misses=self.misses,
          currsize=len(self._entries),
          total_bytes=self._total_bytes,
      )

  def cache_keys(self) -> list[Hashable]:
    with self._lock:
      return list(self._entries)


_kernel_fingerprinter = _KernelFingerprinter()
_mosaic_lowering_cache = _MosaicLoweringCache()
util.register_cache(_kernel_fingerprinter, "pallas_tpu_kernel_fingerprints")
util.register_cache(_mosaic_lowering_cache, "pallas_tpu_mosaic_lowering")


def _lower_and_serialize_mosaic_module(
    ctx: mlir.LoweringRuleContext,
    kernel_key: Callable[[], Hashable],
    lower_module: Callable[[], ir.Module],
    *,
    jax_mesh: Any,
) -> tpu_custom_call.SerializedMosaicModule:
  """Lowers a kernel to a serialized Mosaic module, reusing earlier lowerings.

  `kernel_key` must cover everything `lower_module` depends on besides the
  lowering environment, which is added here. Unless metadata is included in
  compilation cache keys, a kernel that differs from an earlier one only in
  source information reuses its lowering, including its debug locations.
  """
  ir_version = tpu_custom_call.get_ir_version(ctx)
  unzip_debug_locations = tpu_custom_call.should_unzip_debug_locations(ctx)
  max_bytes = config.jax_pallas_tpu_lowering_cache_max_bytes.value
  key = None
  if (
      max_bytes > 0
      and not pallas_core.dynamic_shapes_export_enabled()
      and not ctx.module_context.shape_poly_state.dim_vars
  ):
    try:
      device = (tpu_info.get_device_kind(), tpu_info.get_num_device_cores())
    except Exception:  # pylint: disable=broad-except
      # Kernels that never query the device lower without one.
      device = None
    # The enclosing equation's XLA metadata is in the trace context here, but
    # only annotates the custom call, which is emitted for every call site.
    with xla_metadata.clear_xla_metadata():
      trace_context = config.trace_context()
    try:
      key = (
          kernel_key(),
          tuple(ctx.module_context.platforms),
          ctx.is_forward_compat(),
          ir_version,
          unzip_debug_locations,
          trace_context,
          pallas_core.poison_buffers_enabled(),
          _include_metadata_in_key(),
          config.include_full_tracebacks_in_locations.value,
          config.traceback_in_locations_limit.value,
          device,
          # Lowering only sees the mesh through `MeshInfo`, not its devices,
          # so all stages of an MPMD program share kernels.
          None
          if jax_mesh is None
          else (tuple(jax_mesh.axis_names), tuple(jax_mesh.shape.items())),
      )
    except _UncacheableKernelError as e:
      logger.log(
          logging.WARNING
          if config.explain_cache_misses.value
          else logging.DEBUG,
          "Not caching the Mosaic lowering of a Pallas TPU kernel: %s",
          e,
      )
      key = None
  if key is not None:
    cached = _mosaic_lowering_cache.get(key)
    if cached is not None:
      return cached
  serialized = tpu_custom_call.serialize_mosaic_module(
      lower_module(),
      ir_version=ir_version,
      unzip_debug_locations=unzip_debug_locations,
  )
  if key is not None:
    _mosaic_lowering_cache.put(key, serialized, max_bytes)
  return serialized


def _get_memory_space_from_aval(
    out_aval: jax_core.AbstractValue, kernel_type: tpu_core.CoreType | None
) -> tpu_custom_call.MemorySpace | None:
  if not isinstance(out_aval, jax_core.ShapedArray):
    raise ValueError("Memory spaces not defined for non-ShapedArrays")
  if not isinstance(
      ms := getattr(out_aval, "memory_space", None),
      (
          tpu_core.MemorySpace,
          pallas_core.MemorySpace,
          pallas_core.CoreMemorySpace,
          jax_core.MemorySpace,
      ),
  ):
    return None  # If we are passed a non-TPU memory space, ignore it.
  # If we are passed an aval with an explicit memory space tag, we use it
  # to constrain the memory space.
  match ms:
    case tpu_core.MemorySpace.HBM:
      return tpu_custom_call.MemorySpace.HBM
    case tpu_core.MemorySpace.VMEM:
      match kernel_type:
        case tpu_core.CoreType.TC:
          return tpu_custom_call.MemorySpace.VMEM
        case tpu_core.CoreType.SC_VECTOR_SUBCORE:
          return tpu_custom_call.MemorySpace.SC_VECTOR_VMEM
        case _:
          raise ValueError(f"Invalid kernel type for VMEM: {kernel_type}")
    case tpu_core.MemorySpace.SMEM:
      match kernel_type:
        case tpu_core.CoreType.SC_SCALAR_SUBCORE:
          return tpu_custom_call.MemorySpace.SC_SCALAR_SMEM
        case tpu_core.CoreType.SC_VECTOR_SUBCORE:
          return tpu_custom_call.MemorySpace.SC_VECTOR_SMEM
        case tpu_core.CoreType.TC:
          return tpu_custom_call.MemorySpace.SMEM
        case _:
          raise ValueError(f"Invalid kernel type for SMEM: {kernel_type}")
    case tpu_core.MemorySpace.SEMAPHORE:
      match kernel_type:
        case tpu_core.CoreType.SC_SCALAR_SUBCORE:
          return tpu_custom_call.MemorySpace.SC_SCALAR_SEMAPHORE_MEM
        case tpu_core.CoreType.TC:
          return tpu_custom_call.MemorySpace.SEMAPHORE_MEM
        case _:
          raise ValueError(f"Invalid kernel type for semaphore: {kernel_type}")
    case jax_core.MemorySpace.Host:
      return tpu_custom_call.MemorySpace.HOST
    case pallas_core.CoreMemorySpace(tpu_core.MemorySpace.VMEM, mesh):
      match mesh.core_type:
        case tpu_core.CoreType.TC:
          return tpu_custom_call.MemorySpace.VMEM
        case tpu_core.CoreType.SC_VECTOR_SUBCORE:
          return tpu_custom_call.MemorySpace.SC_VECTOR_VMEM
        case _:
          raise ValueError(f"Invalid core type for VMEM: {mesh.core_type}")
    case pallas_core.CoreMemorySpace(tpu_core.MemorySpace.SMEM, mesh):
      match mesh.core_type:
        case tpu_core.CoreType.SC_SCALAR_SUBCORE:
          return tpu_custom_call.MemorySpace.SC_SCALAR_SMEM
        case tpu_core.CoreType.SC_VECTOR_SUBCORE:
          return tpu_custom_call.MemorySpace.SC_VECTOR_SMEM
        case tpu_core.CoreType.TC:
          return tpu_custom_call.MemorySpace.SMEM
        case _:
          raise ValueError(f"Invalid core type for SMEM: {mesh.core_type}")
    case pallas_core.CoreMemorySpace(tpu_core.MemorySpace.SEMAPHORE, mesh):
      match mesh.core_type:
        case tpu_core.CoreType.SC_SCALAR_SUBCORE:
          return tpu_custom_call.MemorySpace.SC_SCALAR_SEMAPHORE_MEM
        case tpu_core.CoreType.SC_VECTOR_SUBCORE:
          return tpu_custom_call.MemorySpace.SC_VECTOR_SEMAPHORE_MEM
        case tpu_core.CoreType.TC:
          return tpu_custom_call.MemorySpace.SEMAPHORE_MEM
        case _:
          raise ValueError(
              f"Invalid core type for semaphore: {mesh.core_type}"
          )
    case _:
      pass
  return None


def _get_memory_spaces_from_avals(
    avals: Sequence[jax_core.AbstractValue],
    kernel_type: tpu_core.CoreType | None,
) -> tuple[tpu_custom_call.MemorySpace | None, ...] | None:
  memory_spaces = None
  if any(isinstance(aval, jax_core.ShapedArray)
         and (
             not isinstance(aval.memory_space, jax_core.MemorySpace)
             or aval.memory_space is jax_core.MemorySpace.Host
         )
         for aval in avals):
    memory_spaces = tuple(
        _get_memory_space_from_aval(aval, kernel_type=kernel_type)
        for aval in avals)
  return memory_spaces


def _resolve_memory_spaces(
    in_avals: Sequence[jax_core.AbstractValue],
    out_avals: Sequence[jax_core.AbstractValue],
    *,
    input_output_aliases: tuple[tuple[int, int], ...],
    kernel_type: tpu_core.CoreType | None,
) -> tuple[
    tuple[tpu_custom_call.MemorySpace | None, ...] | None,
    tuple[tpu_custom_call.MemorySpace | None, ...] | None,
]:
  output_memory_spaces = _get_memory_spaces_from_avals(
      out_avals, kernel_type=kernel_type
  )
  input_memory_spaces = None
  if any(isinstance(aval, jax_core.ShapedArray)
         and not isinstance(aval.memory_space, jax_core.MemorySpace)
         for aval in in_avals):
    input_memory_spaces = _get_memory_spaces_from_avals(
        in_avals, kernel_type=kernel_type
    )
    # ShapedArrayWithMemorySpace wasn't allowed to escape the Pallas kernel, so
    # it was stripped from the original out_avals. We need to resolve this for
    # outputs aliased to inputs with pltpu.with_memory_space_constraint.
    if input_memory_spaces is not None:
      output_memory_spaces_list: list[tpu_custom_call.MemorySpace | None]
      if output_memory_spaces is None:
        output_memory_spaces_list = [None] * len(out_avals)
      else:
        output_memory_spaces_list = list(output_memory_spaces)
      modified = False
      for in_idx, out_idx in input_output_aliases:
        if (ms := input_memory_spaces[in_idx]) is not None:
          if (out_ms := output_memory_spaces_list[out_idx]) not in (None, ms):
            raise ValueError(
                f"In/out memory space conflict for alias {in_idx}->{out_idx}:"
                f" {ms} != {out_ms}"
            )
          output_memory_spaces_list[out_idx] = ms
          modified = True
      if modified:
        output_memory_spaces = tuple(output_memory_spaces_list)
  if output_memory_spaces is not None:
    input_memory_spaces_list: list[tpu_custom_call.MemorySpace | None]
    if input_memory_spaces is None:
      input_memory_spaces_list = [None] * len(in_avals)
    else:
      input_memory_spaces_list = list(input_memory_spaces)
    modified = False
    for in_idx, out_idx in input_output_aliases:
      if (ms := output_memory_spaces[out_idx]) is not None:
        if (in_ms := input_memory_spaces_list[in_idx]) not in (None, ms):
          raise ValueError(
              f"In/out memory space conflict for alias {in_idx}->{out_idx}:"
              f"{in_ms} != {ms}"
          )
        input_memory_spaces_list[in_idx] = ms
        modified = True
    if modified:
      input_memory_spaces = tuple(input_memory_spaces_list)
  if input_memory_spaces is not None:
    input_memory_spaces = tuple(
        i
        if i
        in {
            tpu_custom_call.MemorySpace.HBM,
            tpu_custom_call.MemorySpace.VMEM,
            tpu_custom_call.MemorySpace.SMEM,
        }
        else None
        for i in input_memory_spaces
    )
  return input_memory_spaces, output_memory_spaces


def _resolve_side_effect_type(
    has_side_effects: bool | tpu_core.SideEffectType,
) -> bool | tpu_custom_call.TpuSideEffectType:
  match has_side_effects:
    case bool():
      return has_side_effects
    case tpu_core.SideEffectType.PURE:
      return tpu_custom_call.TpuSideEffectType.PURE
    case tpu_core.SideEffectType.DATAFLOW_SIDE_EFFECTING:
      return tpu_custom_call.TpuSideEffectType.DATAFLOW_SIDE_EFFECTING
    case tpu_core.SideEffectType.SIDE_EFFECTING:
      return tpu_custom_call.TpuSideEffectType.SIDE_EFFECTING
    case _:
      raise ValueError(f"Invalid side effect type: {has_side_effects}")


def _resolve_tiling(
    mosaic_params: tpu_core.CompilerParams,
    kernel_type: tpu_core.CoreType | None,
) -> tpu_custom_call.Tiling | None:
  if kernel_type is tpu_core.CoreType.TC:
    if mosaic_params.use_tc_tiling_on_sc is None:
      return None
    raise ValueError(
        "use_tc_tiling_on_sc= is not supported for TC kernels"
    )

  return (
      tpu_custom_call.Tiling.COMPACT
      if mosaic_params.use_tc_tiling_on_sc
      or mosaic_params.use_tc_tiling_on_sc is None
      else tpu_custom_call.Tiling.SPARSE_CORE
  )


def _jaxpr_kernel_aval_to_mosaic(
    aval: jax_core.AbstractValue,
) -> jax_core.AbstractValue:
  match aval:
    case state_types.AbstractLinVal():
      if dtypes.issubdtype(aval.dtype, jax.numpy.bool_):
        raise NotImplementedError  # TODO(mattjj,sharadmv)
      return aval
    case jax_core.ShapedArray():
      if dtypes.issubdtype(aval.dtype, jax.numpy.bool_):
        return aval.update(dtype=lowering.BOOL_MEMREF_TYPE)
      return aval
    case _:
      raise ValueError(f"Unsupported JAX aval type: {type(aval)}")


def _jax_value_to_mosaic_value(x: jax.Array) -> jax.Array:
  if dtypes.issubdtype(x.dtype, dtypes.bool_):
    return x.astype(lowering.BOOL_MEMREF_TYPE)
  return x


def _lower_to_custom_call(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    mosaic_module: ir.Module | tpu_custom_call.SerializedMosaicModule,
    mosaic_params: tpu_core.CompilerParams,
    kernel_type: tpu_core.CoreType | None,
    num_dynamic_grid_bounds: int,
    input_output_aliases: tuple[tuple[int, int], ...],
    cost_estimate: pallas_core.CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    effects: jax_core.Effects,
    metadata: frozen_dict.FrozenDict[str, str] | None,
    name: str,
    jax_mesh,
    collective_id_tag: Hashable | None = None,
):
  input_output_aliases = tuple(
      (a[0] + num_dynamic_grid_bounds, a[1]) for a in input_output_aliases
  )

  # Replace in_avals to physical avals.
  # This step is required for mapping logical types to physical types.
  # (e.g. PRNG key -> uint32[2])
  physical_avals = [jax_core.physical_aval(aval) for aval in ctx.avals_in]
  physical_out_avals = [jax_core.physical_aval(aval) for aval in ctx.avals_out]
  ctx = ctx.replace(avals_in=physical_avals, avals_out=physical_out_avals)

  # Booleans are loaded into the kernel as integers.
  def _maybe_cast_inputs(*args):
    args = [_jax_value_to_mosaic_value(x) for x in args]
    return args

  kernel_in_avals = [_jaxpr_kernel_aval_to_mosaic(x) for x in physical_avals]
  kernel_out_avals = [
      _jaxpr_kernel_aval_to_mosaic(x) for x in physical_out_avals
  ]
  cast_ctx = ctx.replace(avals_out=kernel_in_avals)
  in_nodes = mlir.lower_fun(_maybe_cast_inputs)(cast_ctx, *in_nodes)

  # Dynamic grid bounds have to go at the front.
  dynamic_grid_args, args = (
      in_nodes[:num_dynamic_grid_bounds],
      in_nodes[num_dynamic_grid_bounds:],
  )
  kernel_ctx = ctx.replace(avals_in=kernel_in_avals, avals_out=kernel_out_avals)
  input_memory_spaces, output_memory_spaces = _resolve_memory_spaces(
      ctx.avals_in,
      out_avals,
      input_output_aliases=input_output_aliases,
      kernel_type=kernel_type,
  )

  if cost_estimate is not None:
    mosaic_cost_estimate = cast(
        tpu_custom_call.CostEstimate, dataclasses.asdict(cost_estimate)
    )
  else:
    mosaic_cost_estimate = None

  dict_metadata = dict(metadata) if metadata is not None else {}
  del metadata
  if jax_mesh is not None:
    mesh_axes = {
        e.name
        for e in effects
        if isinstance(e, jax_core.NamedAxisEffect)
        # Filter for only device mesh axis name effects
        and e.name in jax_mesh.axis_names
    }
    # Only put mesh axes in metadata if there are any.
    if mesh_axes:
      if "mesh_axes" in dict_metadata:
        raise ValueError("Metadata already contains mesh axes.")
      mesh_axes_list = list(mesh_axes)
      if all(isinstance(a, str) for a in mesh_axes):
        mesh_axes_list = sorted(mesh_axes)  # pyrefly: ignore[bad-specialization]
      dict_metadata["mesh_axes"] = json.dumps(mesh_axes_list)
  out_nodes = mosaic.lower_module_to_custom_call(
      kernel_ctx,
      *dynamic_grid_args,
      *args,
      module=mosaic_module,
      out_type=kernel_out_avals,
      kernel_name=mlir.sanitize_name(name),
      cost_estimate=mosaic_cost_estimate,
      vmem_limit_bytes=mosaic_params.vmem_limit_bytes,
      flags=mosaic_params.flags,
      allow_input_fusion=mosaic_params.allow_input_fusion,
      input_output_aliases=input_output_aliases,
      serialization_format=mosaic_params.serialization_format,
      internal_scratch_in_bytes=mosaic_params.internal_scratch_in_bytes,
      collective_id=mosaic_params.collective_id,
      collective_id_tag=collective_id_tag,
      has_side_effects=_resolve_side_effect_type(
          mosaic_params.has_side_effects
      ),
      output_memory_spaces=output_memory_spaces,
      disable_bounds_checks=mosaic_params.disable_bounds_checks,
      disable_semaphore_checks=mosaic_params.disable_semaphore_checks,
      input_memory_spaces=input_memory_spaces,
      metadata=dict_metadata,
      skip_device_barrier=mosaic_params.skip_device_barrier,
      allow_collective_id_without_custom_barrier=mosaic_params.allow_collective_id_without_custom_barrier,
      shape_invariant_numerics=mosaic_params.shape_invariant_numerics,
      needs_layout_passes=mosaic_params.needs_layout_passes,
      tiling=_resolve_tiling(mosaic_params, kernel_type),
      opt_level=mosaic_params.opt_level,
  )
  _maybe_cast_to_bool = (
      lambda x, aval: x.astype(jax.numpy.bool_)
      if aval.dtype == jax.numpy.bool_
      else x
  )

  def _maybe_cast_outputs(*args):
    args = [_maybe_cast_to_bool(x, aval) for x, aval in zip(args, out_avals)]
    return args

  cast_ctx = ctx.replace(avals_in=kernel_out_avals)
  return mlir.lower_fun(_maybe_cast_outputs)(cast_ctx, *out_nodes)


# Memoized per jaxpr because inner-jit jaxprs are shared within and across
# kernels, and re-walking them at every reference dominates lowering time.
@util.weakref_lru_cache(trace_context_in_key=False)
def _barrier_semaphore_tags(jaxpr: jax_core.Jaxpr) -> frozenset[Hashable]:
  tags = {
      eqn.params["tag"]
      for eqn in jaxpr.eqns
      if eqn.primitive is tpu_primitives.get_barrier_semaphore_p
      and eqn.params.get("tag") is not None
  }
  for subjaxpr in jax_core.subjaxprs(jaxpr):
    tags |= _barrier_semaphore_tags(subjaxpr)
  return frozenset(tags)


def _extract_single_barrier_semaphore_tag(
    jaxpr: jax_core.Jaxpr,
) -> Hashable | None:
  tags = set(_barrier_semaphore_tags(jaxpr))
  if len(tags) > 1:
    raise ValueError(
        "Cannot specify multiple distinct barrier semaphore tags within a"
        f" single kernel: {tags}. XLA only supports one collective_id per"
        " kernel."
    )
  return next(iter(tags)) if tags else None


def pallas_call_tpu_lowering_rule(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    grid_mapping: pallas_core.GridMapping,
    mesh: pallas_core.Mesh | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    debug: bool,
    interpret: bool,
    compiler_params: pallas_core.CompilerParams | None,
    cost_estimate: pallas_core.CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    metadata: frozen_dict.FrozenDict[str, str] | None,
    name: str | None,
):
  """Lowers a pallas_call to a Mosaic TPU custom call."""
  del interpret  # Unused.

  debug_info = jaxpr.debug_info
  if debug:
    print(f"\nThe kernel jaxpr for pallas_call {debug_info.func_src_info}:")
    print(jaxpr)

  if compiler_params is None:
    mosaic_params = tpu_core.CompilerParams()
  else:
    assert isinstance(compiler_params, tpu_core.CompilerParams)
    mosaic_params = compiler_params

  tag = _extract_single_barrier_semaphore_tag(jaxpr)

  jax_mesh = None
  axis_context = ctx.module_context.axis_context
  if axis_context is not None:
    if isinstance(axis_context, sharding_impls.SPMDAxisContext):
      jax_mesh = axis_context.mesh
  mlir_ctx = ctx.module_context.context
  tpu.register_dialect(mlir_ctx)

  def lower_module() -> ir.Module:
    with (
        mlir_ctx,
        ir.Location.unknown(mlir_ctx),
        xla_metadata.clear_xla_metadata(),
    ):
      return lowering.lower_jaxpr_to_pipelined_module(
          ctx,
          grid_mapping,
          jaxpr,
          dimension_semantics=mosaic_params.dimension_semantics,
          kernel_type=tpu_core.CoreType.TC,
          mesh=jax_mesh,
          dynamic_shape_replacement_enabled=pallas_core.dynamic_shapes_export_enabled(),
          fuse_transposed_lhs_in_matmul=mosaic_params.fuse_transposed_lhs_in_matmul,
      )

  mosaic_module: ir.Module | tpu_custom_call.SerializedMosaicModule
  if debug:
    module = lower_module()
    pm = passmanager.PassManager.parse("builtin.module(canonicalize)", mlir_ctx)
    pm.run(module.operation)
    print(f"\nThe Mosaic module for pallas_call {debug_info.func_src_info}:")
    print(module.operation.get_asm(use_name_loc_as_prefix=True))
    mosaic_module = module
  else:
    # Everything `lower_module` reads besides the lowering environment.
    def kernel_key() -> Hashable:
      return (
          "pallas_call",
          _kernel_fingerprinter.jaxpr_fingerprint(jaxpr),
          debug_info.func_name,
          _kernel_fingerprinter.leaf(grid_mapping),
          _kernel_fingerprinter.leaf(mosaic_params.dimension_semantics),
          mosaic_params.fuse_transposed_lhs_in_matmul,
      )

    mosaic_module = _lower_and_serialize_mosaic_module(
        ctx, kernel_key, lower_module, jax_mesh=jax_mesh
    )

  return _lower_to_custom_call(
      ctx,
      *in_nodes,
      mosaic_module=mosaic_module,
      mosaic_params=mosaic_params,
      kernel_type=tpu_core.CoreType.TC,
      num_dynamic_grid_bounds=grid_mapping.num_dynamic_grid_bounds,
      input_output_aliases=input_output_aliases,
      cost_estimate=cost_estimate,
      out_avals=out_avals,
      effects=jaxpr.effects,
      metadata=metadata,
      name=name or debug_info.func_name,
      jax_mesh=jax_mesh,
      collective_id_tag=tag,
  )


def _rewrite_jaxpr_for_lowering(
    jaxpr: jax_core.Jaxpr,
    mesh: pallas_core.Mesh,
    all_meshes: tuple[pallas_core.Mesh, ...],
) -> jax_core.Jaxpr:
  # If the jaxpr has any scalar shaped arrays as inputs, they must have come
  # from closed over scalars. We need to rewrite the jaxpr to actually take in
  # a (1,) shaped Ref as an input. On TC and SC Vector Subcore, this Ref will be
  # in SMEM and on the SC scalar subcore, it will be in SMEM.
  core_type = mesh.core_type
  is_scalar_input = [
      isinstance(v.aval, jax_core.ShapedArray) and not v.aval.shape
      for v in jaxpr.invars
  ]
  if not any(is_scalar_input):
    return jaxpr

  new_in_avals = []
  for is_scalar, v in zip(is_scalar_input, jaxpr.invars):
    if not is_scalar:
      new_in_avals.append(v.aval)
      continue
    if core_type == tpu_core.CoreType.TC:
      # TC compiler supports passing Refs in SMEM directly.
      mem_space = tpu_core.MemorySpace.SMEM
    elif core_type in (
        tpu_core.CoreType.SC_SCALAR_SUBCORE,
        tpu_core.CoreType.SC_VECTOR_SUBCORE,
    ):
      # SC compiler doesn't support pass-via-SMEM so we need to pass the Refs
      # in HBM and then copy them into SMEM inside the kernel.
      mem_space = tpu_core.MemorySpace.HBM
    else:
      raise ValueError(f"Unsupported core type: {core_type}")
    new_in_avals.append(
        state.AbstractRef(
            jax_core.ShapedArray((1,), v.aval.dtype), memory_space=mem_space  # pyrefly: ignore[missing-attribute]
        )
    )

  def new_body(*args):
    sync_copy_srcs = []
    sync_copy_dsts = []
    refs = []
    # First pass: collect the destinations and sources for all scalar inputs
    # so we can perform a single, grouped sync_copy for performance.
    for is_scalar, arg in zip(is_scalar_input, args):
      if not is_scalar:
        refs.append(None)
        continue
      if core_type == tpu_core.CoreType.TC:
        refs.append(arg)
      elif core_type == tpu_core.CoreType.SC_SCALAR_SUBCORE:
        smem_ref = jax.empty_ref(
            jax_core.ShapedArray((1,), arg.dtype),
            memory_space=tpu_core.MemorySpace.SMEM,
        )
        sync_copy_srcs.append(arg)
        sync_copy_dsts.append(smem_ref)
        refs.append(smem_ref)
      elif core_type == tpu_core.CoreType.SC_VECTOR_SUBCORE:
        num_lanes = sc_core.get_sparse_core_info().num_lanes
        vmem_ref = jax.empty_ref(
            jax_core.ShapedArray((num_lanes,), arg.dtype),
            memory_space=tpu_core.MemorySpace.VMEM,
        )
        sync_copy_srcs.append(arg)
        sync_copy_dsts.append(vmem_ref.at[:1])
        refs.append(vmem_ref)

    # Perform bulk sync_copy for all scalars before any reads are emitted.
    if sync_copy_srcs:
      helpers.sync_copy(tuple(sync_copy_srcs), tuple(sync_copy_dsts))

    processed_args = []
    # Second pass: load values from the populated SMEM/VMEM references.
    # Doing this in a second pass ensures the read operations appear after
    # the sync_copy operation in the traced jaxpr.
    for is_scalar, arg, ref in zip(is_scalar_input, args, refs):
      if is_scalar:
        assert ref is not None
        if core_type == tpu_core.CoreType.SC_VECTOR_SUBCORE:
          processed_args.append(ref[...][0])
        else:
          processed_args.append(ref[0])
      else:
        processed_args.append(arg)

    return jax_core.eval_jaxpr(jaxpr, jaxpr.constvars, *processed_args)

  with mpmd.mpmd_map_tracing_context(mesh, all_meshes):
    # TODO(necula): use trace_to_jaxpr_nocache instead
    new_jaxpr, _ = pe.trace_to_jaxpr(
        new_body, ft.flatten_args(*new_in_avals),
        jaxpr.debug_info.with_unknown_names(),
    )
  assert not new_jaxpr.consts
  return new_jaxpr


def mpmd_map_tpu_lowering_rule(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    meshes,
    jaxprs,
    out_avals,
    input_output_aliases,
    compiler_params,
    interpret,
    debug,
    cost_estimate,
    metadata,
    name,
    external_meshes,
    num_scratch,
):
  del interpret  # Unused.

  if debug:
    for idx, jaxpr in enumerate(jaxprs):
      print(
          f"\nThe kernel jaxpr {idx=} for mpmd_map"
          f" {jaxpr.debug_info.func_src_info}:"
      )
      print(jaxpr)

  if compiler_params is None:
    mosaic_params = tpu_core.CompilerParams()
  else:
    assert isinstance(compiler_params, tpu_core.CompilerParams)
    mosaic_params = compiler_params

  all_tags: set[Hashable] = set()
  for j in jaxprs:
    if (tag := _extract_single_barrier_semaphore_tag(j)) is not None:
      all_tags.add(tag)
  if len(all_tags) > 1:
    raise ValueError(
        "Cannot specify multiple distinct barrier semaphore tags across"
        f" kernels in mpmd_map: {all_tags}. XLA only supports one"
        " collective_id per custom call."
    )
  tag = next(iter(all_tags)) if all_tags else None

  # TODO(slebedev): Check kernel type and raise if it is set.
  if mosaic_params.dimension_semantics is not None:
    raise ValueError(
        "mpmd_map does not support dimension_semantics= in compiler_params="
    )

  mpmd_meshes_map = {
      mesh.core_type: mesh for mesh in [*meshes, *external_meshes]
  }
  jax_mesh = None
  axis_context = ctx.module_context.axis_context
  if axis_context is not None:
    if isinstance(axis_context, sharding_impls.SPMDAxisContext):
      jax_mesh = axis_context.mesh
  mlir_ctx = ctx.module_context.context
  tpu.register_dialect(mlir_ctx)

  some_jaxpr = jaxprs[0]
  is_scalar_input = [
      isinstance(v.aval, jax_core.ShapedArray) and not v.aval.shape
      for v in some_jaxpr.invars
  ]

  for mesh, jaxpr in zip(meshes, jaxprs, strict=True):
    if not hasattr(mesh, "core_type") or not hasattr(
        mesh, "dimension_semantics"
    ):
      raise ValueError(
          "mpmd_map requires the mesh to define its ``core_type`` and"
          " ``dimension_semantics``"
      )

    match kernel_type := mesh.core_type:
      case tpu_core.CoreType.TC:
        if mpmd_meshes_map is not None and mpmd_meshes_map.keys() != {
            tpu_core.CoreType.TC
        }:
          raise NotImplementedError("mpmd_map does not support TC kernels yet.")
        if mesh.num_cores > 1:
          actual_invars = (
              jaxpr.invars[:-num_scratch] if num_scratch > 0 else jaxpr.invars
          )
          if any(
              pallas_core.get_memory_space_aval(v.aval)
              == tpu_core.MemorySpace.VMEM
              for v in actual_invars
          ):
            raise NotImplementedError(
                "TensorCoreMesh does not support VMEM inputs/outputs when there"
                " are >1 cores. Use HBM or ANY instead."
            )
      case (
          tpu_core.CoreType.SC_SCALAR_SUBCORE
          | tpu_core.CoreType.SC_VECTOR_SUBCORE
      ):
        if not tpu_info.is_tpu_device():
          raise ValueError(
              "SparseCore kernels are only supported on TPU, but the current"
              f" device is {tpu_info.get_device_kind()}."
          )
        info = tpu_info.get_tpu_info()
        if not info.sparse_core:
          raise ValueError(
              "SparseCore is not available on the current device"
              f" ({info.chip_version}), but the kernel type is set to"
              " SparseCore."
          )
      case _:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

  def lower_module() -> ir.Module:
    with mlir_ctx, ir.Location.unknown(mlir_ctx):
      module = ir.Module.create()
      for mesh, jaxpr in zip(meshes, jaxprs, strict=True):
        if any(is_scalar_input):
          jaxpr = _rewrite_jaxpr_for_lowering(
              jaxpr, mesh, (*meshes, *external_meshes)
          )
        lowering.lower_jaxpr_into_unpipelined_module(
            ctx,
            module,
            jaxpr,
            jax_mesh=jax_mesh,
            pallas_mesh=mesh,
            name=mlir.sanitize_name(jaxpr.debug_info.func_name),
            dynamic_shape_replacement_enabled=pallas_core.dynamic_shapes_export_enabled(),
            num_scratch=num_scratch,
            needs_layout_passes=mosaic_params.needs_layout_passes,
        )
    return module

  mosaic_module: ir.Module | tpu_custom_call.SerializedMosaicModule
  if debug:
    module = lower_module()
    pm = passmanager.PassManager.parse("builtin.module(canonicalize)", mlir_ctx)
    pm.run(module.operation)
    print("\nThe Mosaic module for mpmd_map:")
    print(module.operation.get_asm(use_name_loc_as_prefix=True))
    mosaic_module = module
  else:
    # Everything `lower_module` reads besides the lowering environment.
    def kernel_key() -> Hashable:
      return (
          "mpmd_map",
          tuple(_kernel_fingerprinter.jaxpr_fingerprint(j) for j in jaxprs),
          tuple(j.debug_info.func_name for j in jaxprs),
          _kernel_fingerprinter.leaf(tuple(meshes)),
          _kernel_fingerprinter.leaf(tuple(external_meshes)),
          num_scratch,
          mosaic_params.needs_layout_passes,
      )

    mosaic_module = _lower_and_serialize_mosaic_module(
        ctx, kernel_key, lower_module, jax_mesh=jax_mesh
    )

  if name is None:
    name = "_".join(jaxpr.debug_info.func_name for jaxpr in jaxprs)

  match [*{mesh.core_type for mesh in meshes}]:
    case [kernel_type]:
      pass
    case _:
      # Use a stub ``kernel_type`` if we are lowering multiple kernels.
      # This will hopefully cause a runtime error if ``kernel_type`` is ever
      # accessed.
      kernel_type = None

  def _maybe_expand_scalar_input(is_scalar, in_node, aval):
    if is_scalar:
      expand_ctx = ctx.replace(
          avals_in=[aval],
          avals_out=[aval.update(shape=(1,))]
      )
      return mlir.lower_fun(lambda x: x[None], multiple_results=False)(
          expand_ctx, in_node
      )[0]
    return in_node
  in_nodes = [
      _maybe_expand_scalar_input(is_scalar, in_node, aval)
      for is_scalar, in_node, aval in zip(is_scalar_input, in_nodes, ctx.avals_in)
  ]
  ctx = ctx.replace(
      avals_in=[
          aval.update(shape=(1,))
          if is_scalar
          else aval
          for is_scalar, aval in zip(is_scalar_input, ctx.avals_in)
      ]
  )

  return _lower_to_custom_call(
      ctx,
      *in_nodes,
      mosaic_module=mosaic_module,
      mosaic_params=mosaic_params,
      kernel_type=kernel_type,
      num_dynamic_grid_bounds=0,
      input_output_aliases=tuple(input_output_aliases.items()),
      cost_estimate=cost_estimate,
      out_avals=out_avals,
      effects=set().union(*(jaxpr.effects for jaxpr in jaxprs)),
      metadata=metadata,
      name=name,
      jax_mesh=jax_mesh,
      collective_id_tag=tag,
  )


pallas_core.register_lowering_rule(tpu_core.CompilerParams, pallas_call_tpu_lowering_rule, "tpu")
