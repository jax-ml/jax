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

from __future__ import annotations

from collections import defaultdict
from functools import reduce
from typing import Any, Callable

import jax
from jax import core
from jax import lax
from jax import tree_util
from jax._src import api_util
from jax._src import linear_util as lu
from jax._src import pjit
from jax._src import prng
from jax._src import random
from jax._src import util
from jax._src.debugging import debug_callback_p
from jax._src.interpreters import partial_eval as pe

from jax.experimental.key_reuse._common import (
  consume_p, unconsumed_copy_p, assert_consumed_value_p, KeyReuseError,
  Sink, Source, KeyReuseSignature
)
import numpy as np

# The behavior of most primitives can be described via simple signatures.
key_reuse_signatures: dict[core.Primitive, KeyReuseSignature] = {}

key_reuse_signatures[consume_p] = KeyReuseSignature([Sink(0)], [])
key_reuse_signatures[unconsumed_copy_p] = KeyReuseSignature([], [Source(0)])
key_reuse_signatures[prng.random_bits_p] = KeyReuseSignature([Sink(0)], [])
key_reuse_signatures[prng.random_fold_in_p] = KeyReuseSignature([Sink(0)], [Source(0)])
key_reuse_signatures[prng.random_seed_p] = KeyReuseSignature([], [Source(0)])
key_reuse_signatures[prng.random_split_p] = KeyReuseSignature([Sink(0)], [Source(0)])
key_reuse_signatures[random.random_gamma_p] = KeyReuseSignature([Sink(0)], [])
key_reuse_signatures[lax.broadcast_in_dim_p] = KeyReuseSignature([Sink(0)], [Source(0)])
key_reuse_signatures[lax.copy_p] = KeyReuseSignature([Sink(0)], [Source(0)])
key_reuse_signatures[lax.convert_element_type_p] = KeyReuseSignature([Sink(0)], [Source(0)])
key_reuse_signatures[lax.device_put_p] = KeyReuseSignature([Sink(0)], [Source(0)])
key_reuse_signatures[lax.reshape_p] = KeyReuseSignature([Sink(0)], [Source(0)])
key_reuse_signatures[lax.squeeze_p] = KeyReuseSignature([Sink(0)], [Source(0)])
key_reuse_signatures[prng.random_wrap_p] = KeyReuseSignature([], [Source(0)])
key_reuse_signatures[prng.random_unwrap_p] = KeyReuseSignature([Sink(0)], [])
key_reuse_signatures[debug_callback_p] = KeyReuseSignature([], [])
key_reuse_signatures[lax.dynamic_slice_p] = KeyReuseSignature([Sink(0)], [Source(0)])
key_reuse_signatures[lax.dynamic_update_slice_p] = KeyReuseSignature([], [])

# Rules which require more dynamic logic.
key_reuse_signatures_dynamic: dict[core.Primitive, Callable[..., KeyReuseSignature]] = {}


# The default signature will Sink all key inputs, and not Source any.
def unknown_signature(eqn, args_consumed):
  def is_key(var: core.Atom):
    return hasattr(var.aval, "dtype") and jax.dtypes.issubdtype(var.aval.dtype, jax.dtypes.prng_key)
  return KeyReuseSignature(
    sinks=[Sink(idx, True) for idx, var in enumerate(eqn.invars) if is_key(var)],
    sources=[],
  )


def get_jaxpr_type_signature(
    jaxpr: core.Jaxpr,
    consumed_inputs: list[bool | np.ndarray] | None = None,
    ) -> KeyReuseSignature:
  """Parse the jaxpr to determine key reuse signature"""
  consumed: dict[core.Atom, bool | np.ndarray] = {}

  def is_key(var: core.Atom):
    return hasattr(var.aval, "dtype") and jax.dtypes.issubdtype(var.aval.dtype, jax.dtypes.prng_key)

  def sink(var: core.Atom, mask=True):
    if not is_key(var):
      return
    assert not isinstance(var, core.Literal)
    if np.any(np.logical_and(consumed.get(var, False), mask)):
      return True
    consumed[var] = np.logical_or(consumed.get(var, False), mask)

  def source(var: core.Atom, mask=False):
    if not is_key(var):
      return
    assert not isinstance(var, core.Literal)
    consumed[var] = mask

  def is_consumed(var: core.Atom):
    if isinstance(var, core.Literal):
      return False
    return consumed.get(var, False)

  if consumed_inputs:
    for var, mask in util.safe_zip(jaxpr.invars, consumed_inputs):
      if not isinstance(var, core.Literal):
        source(var, mask)

  for eqn in jaxpr.eqns:
    if eqn.primitive in key_reuse_signatures:
      signature = key_reuse_signatures[eqn.primitive]
    elif eqn.primitive in key_reuse_signatures_dynamic:
      args_consumed = [is_consumed(var) for var in eqn.invars]
      signature = key_reuse_signatures_dynamic[eqn.primitive](eqn, args_consumed)
    else:
      args_consumed = [is_consumed(var) for var in eqn.invars]
      signature = unknown_signature(eqn, args_consumed)

    for snk in signature.sinks:
      if sink(eqn.invars[snk.idx], snk.mask):
        raise KeyReuseError(f"In {eqn.primitive}, key values {eqn.invars[snk.idx]} are already consumed.\n"
                            f"eqn: {eqn}\njaxpr:\n{jaxpr}")
    for var in eqn.outvars:
      if not isinstance(var, core.Literal):
        source(var, True)  # consumed unless in a Source.
    for src in signature.sources:
      source(eqn.outvars[src.idx])

  forwards = [v for v in jaxpr.outvars
              if is_key(v) and v in jaxpr.invars and not np.any(consumed.get(v, False))]
  sinks = [v for v in jaxpr.invars if is_key(v) and np.any(consumed.get(v, False))]
  sources = [v for v in jaxpr.outvars if is_key(v) and not np.any(consumed.get(v, False))]
  return KeyReuseSignature(
    sinks=[
      Sink(i, True if v in forwards else consumed[v])
      for i, v in enumerate(jaxpr.invars)
      if v in forwards or v in sinks
    ],
    sources=[
      Source(i) for i, v in enumerate(jaxpr.outvars)
      if (v in forwards or v in sources)
      and v not in jaxpr.outvars[:i]  # Only source the first of duplicate return values
    ],
  )


def check_key_reuse(fun: Callable[..., Any], /, *args: Any) -> KeyReuseSignature:
  """Function to statically check key reuse."""
  args_flat, in_tree = tree_util.tree_flatten(args)
  in_avals_flat = [core.get_aval(arg) for arg in args_flat]
  wrapped_fun, _ = api_util.flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals_flat)
  return get_jaxpr_type_signature(jaxpr)


#----------------------------------------------------------------------------------
# key reuse rules for particular primitives:

def _slice_signature(eqn, args_consumed):
  del args_consumed  # unused here
  in_aval = eqn.invars[0].aval
  start_indices = eqn.params['start_indices']
  limit_indices = eqn.params['limit_indices']
  strides = eqn.params['strides'] or (1,) * len(start_indices)
  idx = tuple(slice(*tup) for tup in util.safe_zip(start_indices, limit_indices, strides))
  mask = np.zeros(in_aval.shape, dtype=bool)
  mask[idx] = True
  return KeyReuseSignature([Sink(0, mask)], [Source(0)])

key_reuse_signatures_dynamic[lax.slice_p] = _slice_signature

def _pjit_key_type_signature(eqn, args_consumed):
  jaxpr = eqn.params['jaxpr']
  non_literal_invars = [v for v in eqn.invars if not isinstance(v, core.Literal)]
  if len(set(non_literal_invars)) != len(non_literal_invars):
    raise ValueError(f"pjit with duplicate inputs: {eqn.invars=}")
  return get_jaxpr_type_signature(jaxpr.jaxpr, consumed_inputs=args_consumed)

key_reuse_signatures_dynamic[pjit.pjit_p] = _pjit_key_type_signature

def _assert_consumed_value_key_type_signature(eqn, args_consumed):
  actual = args_consumed[0]
  expected =  eqn.params['value']
  if not np.all(actual == expected):
    if np.all(expected):
      raise AssertionError(f"Expected key to be consumed in {eqn}")
    elif not np.any(expected):
      raise AssertionError(f"Expected key to not be consumed in {eqn}")
    else:
      raise AssertionError(f"Expected {expected}, got {actual} in {eqn}")
  return KeyReuseSignature([], [])

key_reuse_signatures_dynamic[assert_consumed_value_p] = _assert_consumed_value_key_type_signature

def _cond_key_type_signature(eqn, args_consumed):
  signatures = [get_jaxpr_type_signature(branch.jaxpr, consumed_inputs=args_consumed[1:])
                for branch in eqn.params['branches']]
  sinks = defaultdict(list)
  sources = defaultdict(list)
  for sig in signatures:
    for sink in sig.sinks:
      sinks[sink.idx].append(sink.mask)
    for source in sig.sources:
      sources[source.idx].append(source.mask)

  combined_sinks = [Sink(i + 1, reduce(np.logical_or, m)) for i, m in sinks.items()]
  combined_sources = [Source(i + 1, reduce(np.logical_and, m)) for i, m in sources.items()]
  return KeyReuseSignature(combined_sinks, combined_sources)

key_reuse_signatures_dynamic[lax.cond_p] = _cond_key_type_signature

def _scan_key_type_signature(eqn, args_consumed):
  jaxpr = eqn.params['jaxpr'].jaxpr
  num_consts = eqn.params['num_consts']
  num_carry = eqn.params['num_carry']
  length = eqn.params['length']
  signature = get_jaxpr_type_signature(jaxpr, args_consumed)

  # scan body should not consume key in constants
  if any(np.any(s.mask) for s in signature.sinks if s.idx < num_consts):
    raise KeyReuseError(f"scan body function leads to key reuse when repeatedly executed: {signature=}")

  # scan carry should only consume keys that are sourced on output.
  carry_sinks = {s.idx - num_consts: s.mask for s in signature.sinks if 0 <= s.idx - num_consts < num_carry}
  carry_sources = {s.idx: s.mask for s in signature.sources if 0 <= s.idx < num_carry}
  if carry_sinks.keys() != carry_sources.keys():  # TODO(jakevdp): check that masks match
    raise KeyReuseError(f"scan body function leads to key reuse when repeatedly executed: {signature=}")
  return signature

key_reuse_signatures_dynamic[jax.lax.scan_p] = _scan_key_type_signature

def _while_key_type_signature(eqn, args_consumed):
  cond_jaxpr = eqn.params['cond_jaxpr'].jaxpr
  cond_nconsts = eqn.params['cond_nconsts']
  body_jaxpr = eqn.params['body_jaxpr'].jaxpr
  body_nconsts = eqn.params['body_nconsts']

  # TODO(jakevdp): pass args_consumed here?
  cond_signature = get_jaxpr_type_signature(cond_jaxpr)
  body_signature = get_jaxpr_type_signature(body_jaxpr)

  # Error if there are sinks among consts.
  if any(np.any(s.mask) for s in cond_signature.sinks if s.idx < cond_nconsts):
    raise KeyReuseError("while_loop cond function leads to key reuse when repeatedly executed: "
                        f"{cond_signature=}")
  if any(np.any(s.mask) for s in body_signature.sinks if s.idx < body_nconsts):
    raise KeyReuseError("while_loop body function leads to key reuse when repeatedly executed: "
                        f"{body_signature=}")

  # carry should only consume keys that are sourced on output.
  body_carry_sinks = {s.idx - body_nconsts: s.mask for s in body_signature.sinks if s.idx >= body_nconsts}
  cond_carry_sinks = {s.idx - cond_nconsts: s.mask for s in cond_signature.sinks if s.idx >= cond_nconsts}
  carry_sources = {s.idx: s.mask for s in body_signature.sources}
  # TODO(jakevdp): check masks at each index?
  if not (cond_carry_sinks.keys() <= carry_sources.keys()):
    raise KeyReuseError("while_loop cond function leads to key reuse when repeatedly executed: "
                        f"{cond_signature=}")
  if not (body_carry_sinks.keys() <= carry_sources.keys()):
    raise KeyReuseError("while_loop body function leads to key reuse when repeatedly executed: "
                        f"{body_signature=}")
  if body_carry_sinks.keys() & cond_carry_sinks.keys():
    raise KeyReuseError("while_loop cond and body functions both use the same key: "
                        f"{cond_signature=} {body_signature=}")
  return body_signature

key_reuse_signatures_dynamic[jax.lax.while_p] = _while_key_type_signature
