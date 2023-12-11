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
from typing import Any, Callable, NamedTuple

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

class Forward(NamedTuple):
  in_idx: int
  out_idx: int


class KeyReuseSignatureWithForwards(NamedTuple):
  sinks: list[Sink]
  sources: list[Source]
  forwards: list[Forward] = []

# The behavior of most primitives can be described via simple signatures.
key_reuse_signatures: dict[core.Primitive, KeyReuseSignatureWithForwards] = {}

key_reuse_signatures[consume_p] = KeyReuseSignatureWithForwards([Sink(0)], [], [Forward(0, 0)])
key_reuse_signatures[unconsumed_copy_p] = KeyReuseSignatureWithForwards([], [Source(0)])
key_reuse_signatures[prng.random_bits_p] = KeyReuseSignatureWithForwards([Sink(0)], [])
key_reuse_signatures[prng.random_fold_in_p] = KeyReuseSignatureWithForwards([Sink(0)], [Source(0)])
key_reuse_signatures[prng.random_seed_p] = KeyReuseSignatureWithForwards([], [Source(0)])
key_reuse_signatures[prng.random_split_p] = KeyReuseSignatureWithForwards([Sink(0)], [Source(0)])
key_reuse_signatures[random.random_gamma_p] = KeyReuseSignatureWithForwards([Sink(0)], [])
key_reuse_signatures[lax.broadcast_in_dim_p] = KeyReuseSignatureWithForwards([], [], [Forward(0, 0)])
key_reuse_signatures[lax.copy_p] = KeyReuseSignatureWithForwards([], [], [Forward(0, 0)])
key_reuse_signatures[lax.convert_element_type_p] = KeyReuseSignatureWithForwards([], [], [Forward(0, 0)])
key_reuse_signatures[lax.device_put_p] = KeyReuseSignatureWithForwards([], [], [Forward(0, 0)])
key_reuse_signatures[lax.reshape_p] = KeyReuseSignatureWithForwards([], [], [Forward(0, 0)])
key_reuse_signatures[lax.squeeze_p] = KeyReuseSignatureWithForwards([], [], [Forward(0, 0)])
key_reuse_signatures[prng.random_wrap_p] = KeyReuseSignatureWithForwards([], [Source(0)], [])
key_reuse_signatures[prng.random_unwrap_p] = KeyReuseSignatureWithForwards([Sink(0)], [], [])
key_reuse_signatures[debug_callback_p] = KeyReuseSignatureWithForwards([], [])
key_reuse_signatures[lax.dynamic_slice_p] = KeyReuseSignatureWithForwards([], [], [Forward(0, 0)])
key_reuse_signatures[lax.dynamic_update_slice_p] = KeyReuseSignatureWithForwards([], [], [])

# Rules which require more dynamic logic.
key_reuse_signatures_dynamic: dict[core.Primitive, Callable[..., KeyReuseSignatureWithForwards]] = {}

# The default signature will Sink all key inputs, and not Source any.
def unknown_signature(eqn, args_consumed):
  def is_key(var: core.Atom):
    return hasattr(var.aval, "dtype") and jax.dtypes.issubdtype(var.aval.dtype, jax.dtypes.prng_key)
  return KeyReuseSignatureWithForwards(
    sinks=[Sink(idx, True) for idx, var in enumerate(eqn.invars) if is_key(var)],
    sources=[],
  )

def get_jaxpr_type_signature(
    jaxpr: core.Jaxpr,
    consumed_inputs: list[bool | np.ndarray] | None = None,
    forwarded_inputs: dict[int, int] | None = None,
    ) -> KeyReuseSignatureWithForwards:
  """Parse the jaxpr to determine key reuse signature"""
  consumed: dict[core.Atom, bool | np.ndarray] = {}
  forwards: dict[core.Atom, core.Atom] = {}  # map forwarded outputs to inputs.

  def resolve_forwards(var: core.Atom) -> core.Atom:
    if not forwards:
      return var
    for _ in range(len(forwards) + 1):
      if isinstance(var, core.Literal):
        return var
      if var in forwards:
        var = forwards[var]
      else:
        return var
    raise ValueError("forwarding cycle detected")

  def is_key(var: core.Atom):
    return hasattr(var.aval, "dtype") and jax.dtypes.issubdtype(var.aval.dtype, jax.dtypes.prng_key)

  def sink(var: core.Atom, mask=True):
    if not is_key(var):
      return
    var = resolve_forwards(var)
    assert not isinstance(var, core.Literal)
    if np.any(np.logical_and(consumed.get(var, False), mask)):
      return True
    consumed[var] = np.logical_or(consumed.get(var, False), mask)


  def source(var: core.Atom, mask=False):
    if not is_key(var):
      return
    var = resolve_forwards(var)
    assert not isinstance(var, core.Literal)
    consumed[var] = mask

  def is_consumed(var: core.Atom):
    var = resolve_forwards(var)
    if isinstance(var, core.Literal):
      return False
    return consumed.get(var, False)

  if forwarded_inputs:
    for i, j in forwarded_inputs.items():
      forwards[jaxpr.invars[i]] = jaxpr.invars[j]

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
    for in_idx, out_idx in signature.forwards:
      forwards[eqn.outvars[out_idx]] = eqn.invars[in_idx]

    for snk in signature.sinks:
      if sink(eqn.invars[snk.idx], snk.mask):
        raise KeyReuseError(f"In {eqn.primitive}, key values {eqn.invars[snk.idx]} are already consumed.\n"
                            f"eqn: {eqn}\njaxpr:\n{jaxpr}")
    for var in eqn.outvars:
      if not isinstance(var, core.Literal) and var not in forwards:
        source(var, True)  # consumed unless in a Source.
    for src in signature.sources:
      source(eqn.outvars[src.idx])

  return KeyReuseSignatureWithForwards(
    sinks=[Sink(i, consumed[v]) for i, v in enumerate(jaxpr.invars)
           if is_key(v) and np.any(consumed.get(v, False))],
    sources=[Source(i) for i, v in enumerate(jaxpr.outvars)
             if is_key(v) and resolve_forwards(v) not in jaxpr.invars and not consumed.get(v, False)],
    forwards=[Forward(jaxpr.invars.index(resolve_forwards(outvar)), idx_out)  # type: ignore[arg-type]
              for idx_out, outvar in enumerate(jaxpr.outvars)
              if is_key(outvar) and resolve_forwards(outvar) in jaxpr.invars]
  )


def check_key_reuse(fun: Callable[..., Any], /, *args: Any) -> KeyReuseSignatureWithForwards:
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
  return KeyReuseSignatureWithForwards([Sink(0, mask)], [Source(0)])

key_reuse_signatures_dynamic[lax.slice_p] = _slice_signature

def _pjit_key_type_signature(eqn, args_consumed):
  jaxpr = eqn.params['jaxpr']
  forwarded_inputs = {i: eqn.invars.index(var) for i, var in enumerate(eqn.invars)
                      if var in eqn.invars[:i]}
  return get_jaxpr_type_signature(jaxpr.jaxpr, consumed_inputs=args_consumed,
                                  forwarded_inputs=forwarded_inputs)

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
  return KeyReuseSignatureWithForwards([], [], [Forward(0, 0)])

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
  combined_forwards = [Forward(f.in_idx + 1, f.out_idx) for f in
                       set.intersection(*(set(sig.forwards) for sig in signatures))]
  return KeyReuseSignatureWithForwards(combined_sinks, combined_sources, combined_forwards)

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
