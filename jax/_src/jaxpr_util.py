# Copyright 2020 The JAX Authors.
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

"""Utilities for the Jaxpr IR."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
import gzip
import itertools
import json
import types
from typing import Any, Iterator, Union

from jax._src import core
from jax._src import util
from jax._src import source_info_util
from jax._src.lib import xla_client

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


def _all_eqns(
    jaxpr: core.Jaxpr, visited: set[core.Jaxpr] | None,
) -> Iterator[tuple[core.Jaxpr, core.JaxprEqn]]:
  for eqn in jaxpr.eqns:
    yield (jaxpr, eqn)
  for subjaxpr in core.subjaxprs(jaxpr):
    if visited is None:
      yield from _all_eqns(subjaxpr, visited)
    elif subjaxpr not in visited:
      visited.add(subjaxpr)
      yield from _all_eqns(subjaxpr, visited)

def all_eqns(
    jaxpr: core.Jaxpr, revisit_inner_jaxprs: bool = True
) -> Iterator[tuple[core.Jaxpr, core.JaxprEqn]]:
  yield from _all_eqns(jaxpr, None if revisit_inner_jaxprs else set())


def collect_eqns(jaxpr: core.Jaxpr, key: Callable):
  d = defaultdict(list)
  for _, eqn in all_eqns(jaxpr):
    d[key(eqn)].append(eqn)
  return dict(d)

def histogram(jaxpr: core.Jaxpr, key: Callable,
              key_fmt: Callable = lambda x: x):
  d = collect_eqns(jaxpr, key)
  return {key_fmt(k): len(v) for k, v in d.items()}

def primitives(jaxpr: core.Jaxpr):
  return histogram(jaxpr, lambda eqn: eqn.primitive.name)

def primitives_by_source(jaxpr: core.Jaxpr):
  def key(eqn):
    src = source_info_util.summarize(eqn.source_info)
    return (eqn.primitive.name, src)
  return histogram(jaxpr, key, ' @ '.join)

def primitives_by_shape(jaxpr: core.Jaxpr):
  def shape_fmt(var):
    return '*' if isinstance(var, core.DropVar) else var.aval.str_short()
  def key(eqn):
    return (eqn.primitive.name, ' '.join(map(shape_fmt, eqn.outvars)))
  return histogram(jaxpr, key, ' :: '.join)

def source_locations(jaxpr: core.Jaxpr):
  def key(eqn):
    return source_info_util.summarize(eqn.source_info)
  return histogram(jaxpr, key)

MaybeEqn = Union[core.JaxprEqn, None]

def var_defs_and_refs(jaxpr: core.Jaxpr):
  defs: dict[core.Var, MaybeEqn] = {}
  refs: dict[core.Var, list[MaybeEqn]] = {}

  def read(a: core.Atom, eqn: MaybeEqn):
    if not isinstance(a, core.Literal):
      assert a in defs, a
      assert a in refs, a
      refs[a].append(eqn)

  def write(v: core.Var, eqn: MaybeEqn):
    assert v not in defs, v
    assert v not in refs, v
    if not isinstance(v, core.DropVar):
      defs[v] = eqn
      refs[v] = []

  for v in jaxpr.constvars:
    write(v, None)
  for v in jaxpr.invars:
    write(v, None)

  for eqn in jaxpr.eqns:
    for a in eqn.invars:
      read(a, eqn)
    for v in eqn.outvars:
      write(v, eqn)

  for a in jaxpr.outvars:
    read(a, None)

  res = [(v, defs[v], refs[v]) for v in defs]
  subs = map(var_defs_and_refs, core.subjaxprs(jaxpr))
  return [(jaxpr, res), *subs] if subs else (jaxpr, res)

def vars_by_fanout(jaxpr: core.Jaxpr):
  def fmt_key(var, eqn):
    if eqn is None:
      return f'{var} <- invar'
    else:
      src = source_info_util.summarize(eqn.source_info)
      return f'{var} <- {eqn.primitive.name} @ {src}'

  def hist(jaxpr, reads):
    return {fmt_key(var, var_def): len(var_refs)
            for var, var_def, var_refs in reads}

  return [(j, hist(j, reads)) for j, reads in var_defs_and_refs(jaxpr)]  # pytype: disable=bad-unpacking

def print_histogram(histogram: dict[Any, int]):
  count_width = max(len(str(v)) for v in histogram.values())
  count_fmt = '{:>' + str(count_width) + 'd}'
  pairs = [(v, k) for k, v in histogram.items()]
  for count, name in sorted(pairs, reverse=True):
    print(count_fmt.format(count), name)


def _pprof_profile(
    profile: dict[tuple[xla_client.Traceback | None, core.Primitive], int]
) -> bytes:
  """Converts a profile into a compressed pprof protocol buffer.

  The input profile is a map from (traceback, primitive) pairs to counts.
  """
  s: defaultdict[str, int]
  func: defaultdict[types.CodeType, int]
  loc: defaultdict[tuple[types.CodeType, int], int]

  s = defaultdict(itertools.count(1).__next__)
  func = defaultdict(itertools.count(1).__next__)
  loc = defaultdict(itertools.count(1).__next__)
  s[""] = 0
  primitive_key = s["primitive"]
  samples = []
  for (tb, primitive), count in profile.items():
    if tb is None:
      frames = []
    else:
      raw_frames = zip(*tb.raw_frames())
      frames = [loc[(code, lasti)] for code, lasti in raw_frames
                if source_info_util.is_user_filename(code.co_filename)]
    samples.append({
       "location_id": frames,
       "value": [count],
       "label": [{
         "key": primitive_key,
         "str": s[primitive.name]
        }]
    })

  locations = [
      {"id": loc_id,
       "line": [{"function_id": func[code],
                 "line": xla_client.Traceback.code_addr2line(code, lasti)}]}
      for (code, lasti), loc_id in loc.items()
  ]
  functions = [
      {"id": func_id,
       "name": s[code.co_name],
       "system_name": s[code.co_name],
       "filename": s[code.co_filename],
       "start_line": code.co_firstlineno}
      for code, func_id in func.items()
  ]
  sample_type = [{"type": s["equations"], "unit": s["count"]}]
  # This is the JSON encoding of a pprof profile protocol buffer. See:
  # https://github.com/google/pprof/blob/master/proto/profile.proto for a
  # description of the format.
  json_profile = json.dumps({
    "string_table": list(s.keys()),
    "location": locations,
    "function": functions,
    "sample_type": sample_type,
    "sample": samples,
  })
  return gzip.compress(xla_client._xla.json_to_pprof_profile(json_profile))


def pprof_equation_profile(jaxpr: core.Jaxpr) -> bytes:
  """Generates a pprof profile that maps jaxpr equations to Python stack traces.

  By visualizing the profile using pprof, one can identify Python code that is
  responsible for yielding large numbers of jaxpr equations.

  Args:
    jaxpr: a Jaxpr.

  Returns:
    A gzip-compressed pprof Profile protocol buffer, suitable for passing to
    pprof tool for visualization.
  """
  d = Counter(
      (eqn.source_info.traceback, eqn.primitive)
      for _, eqn in all_eqns(jaxpr, revisit_inner_jaxprs=False)
  )
  return _pprof_profile(d)

def eqns_using_var_with_invar_index(jaxpr: core.Jaxpr, invar: core.Var) -> Iterator[tuple[core.JaxprEqn, int]]:
  """Find all the equations which use invar and the positional index of its binder"""
  for eqn in jaxpr.eqns:
    for invar_index, eqn_var in enumerate(eqn.invars):
      if eqn_var == invar:
        yield eqn, invar_index
        break # we found the var, no need to keep looking in this eqn

def jaxpr_and_binder_in_params(params, index: int) -> Iterator[tuple[core.Jaxpr, core.Var]]:
  for val in params.values():
    vals = val if isinstance(val, tuple) else (val,)
    for v in vals:
      if isinstance(v, core.Jaxpr):
        if index >= len(v.invars):
          raise RuntimeError(f"Failed to find index {index} in jaxpr.invars while building report")
        yield v, v.invars[index]
      elif isinstance(v, core.ClosedJaxpr):
        if index >= len(v.jaxpr.invars):
          raise RuntimeError(f"Failed to find index {index} in jaxpr.invars while building report")
        yield v.jaxpr, v.jaxpr.invars[index]

def eqns_using_var(jaxpr: core.Jaxpr, invar: core.Var) -> Iterator[core.JaxprEqn]:
  """Find the leaf equations using a variable"""
  # The complexity of this call is because the invar might originate from a nested jaxpr
  for eqn, invar_index in eqns_using_var_with_invar_index(jaxpr, invar):
    if (child_jaxprs_and_vars := tuple(jaxpr_and_binder_in_params(eqn.params, invar_index))):
      for (jaxpr, invar) in child_jaxprs_and_vars:
        yield from eqns_using_var(jaxpr, invar)
    else:
      # if the previous condition fails, there is no deeper jaxpr to explore =(
      yield eqn
