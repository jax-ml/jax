# Copyright 2020 Google LLC
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
"""
Summary statistics for jaxprs
"""

import collections
from typing import Any, Callable, Dict, List, Optional

from jax import core, source_info_util, util

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


def all_eqns(jaxpr: core.Jaxpr):
  for eqn in jaxpr.eqns:
    yield (jaxpr, eqn)
  for subjaxpr in core.subjaxprs(jaxpr):
    yield from all_eqns(subjaxpr)

def collect_eqns(jaxpr: core.Jaxpr, key: Callable):
  d = collections.defaultdict(list)
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
    return '*' if var is core.dropvar else var.aval.str_short()
  def key(eqn):
    return (eqn.primitive.name, ' '.join(map(shape_fmt, eqn.outvars)))
  return histogram(jaxpr, key, ' :: '.join)

def source_locations(jaxpr: core.Jaxpr):
  def key(eqn):
    return source_info_util.summarize(eqn.source_info)
  return histogram(jaxpr, key)

MaybeEqn = Optional[core.JaxprEqn]

def var_defs_and_refs(jaxpr: core.Jaxpr):
  defs: Dict[core.Var, MaybeEqn] = {}
  refs: Dict[core.Var, List[MaybeEqn]] = {}

  def read(a: core.Atom, eqn: MaybeEqn):
    if a is not core.unitvar and not isinstance(a, core.Literal):
      assert a in defs, a
      assert a in refs, a
      refs[a].append(eqn)

  def write(v: core.Var, eqn: MaybeEqn):
    assert v is not core.unitvar
    assert v not in defs, v
    assert v not in refs, v
    if v is not core.dropvar:
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

  return [(j, hist(j, reads)) for j, reads in var_defs_and_refs(jaxpr)]

def print_histogram(histogram: Dict[Any, int]):
  count_width = max(len(str(v)) for v in histogram.values())
  count_fmt = '{:>' + str(count_width) + 'd}'
  pairs = [(v, k) for k, v in histogram.items()]
  for count, name in reversed(sorted(pairs)):
    print(count_fmt.format(count), name)
