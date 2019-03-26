# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import collections
import heapq
import itertools

import six

from .. import core
from ..core import Trace, Tracer, new_master, pack, AbstractTuple, JaxTuple
from ..util import unzip2, partial, safe_map, safe_zip, WrapHashably
from ..linear_util import transformation, transformation_with_aux, wrap_init
from ..ad_util import add_jaxvals_p
from ..tree_util import tree_map, tree_multimap

map = safe_map
zip = safe_zip

if six.PY2:
  def heap_merge(a, b):
    return tuple(sorted(a + b, key=hash))
else:
  def heap_merge(a, b):
    return tuple(heapq.merge(a, b, key=hash))

def heap_insert(heap, item):
  pos = bisect.bisect(map(hash, heap), hash(item))
  return heap[:pos] + (item,) + heap[pos:]


class ID(object):
  __slots__ = ['val', 'hash']
  def __init__(self, val):
    self.val = val
    self.hash = hash(val)

  def __hash__(self):
    return self.hash

  def __eq__(self, other):
    return type(other) is ID and self.val == other.val

  def __repr__(self):
    return 'ID({})'.format(hash(self))
  __str__ = __repr__

class UniqueID(WrapHashably):
  def __repr__(self):
    return 'UniqueID'
  __str__ = __repr__

AssocCommutID = collections.namedtuple('AssocCommutID', ['prim', 'sorted_ids'])


def const_id(val):
  if isinstance(val, tuple):
    return IDTuple(map(const_id, val))
  elif isinstance(val, (int, float)):
    return ID(val)
  else:
    return UniqueID(val)


class IDTuple(tuple): pass


@transformation
def cse(*args):
  with new_master(CSETrace) as master:
    cse_table = {}
    trace = CSETrace(master, core.cur_sublevel())
    make_tracer = partial(CSETracer, trace, cse_table)
    in_tracers = tree_multimap(make_tracer, args, tree_map(const_id, args))
    out = yield in_tracers
    out_val = tree_map(lambda x: trace.full_raise(x).val, out)
    del master, cse_table
  yield out_val


class CSETracer(Tracer):
  __slots__ = ['cse_table', 'val', 'id']

  def __init__(self, trace, cse_table, val, id):
    self.trace = trace
    self.cse_table = cse_table
    self.val = val
    self.id = id

  @property
  def aval(self):
    return core.get_aval(self.val)

  def unpack(self):
    t = type(self.id)
    if t is UniqueID:
      ids = (ID(('unpack', self.id, i)) for i in range(len(self.val)))
      return map(partial(CSETracer, self.trace, self.cse_table), self.val, ids)
    elif t is IDTuple:
      return map(partial(CSETracer, self.trace, self.cse_table), self.val, self.id)
    else:
      raise TypeError(t)

  def full_lower(self):
    return self

class CSETrace(Trace):
  def pure(self, val):
    return CSETracer(self, None, val, const_id(val))

  def lift(self, val):
    return CSETracer(self, None, val, const_id(val))

  def sublift(self, val):
    return CSETracer(self, None, val.val, val.id)

  def process_primitive(self, primitive, tracers, params):
    vals_in, ids_in = unzip2((t.val, t.id) for t in tracers)
    id = idfuns.get(primitive, default_id)(primitive, *ids_in, **params)
    cse_table = next(t.cse_table for t in tracers if t.cse_table is not None)
    if id not in cse_table:
      cse_table[id] = primitive.bind(*vals_in, **params)
    return CSETracer(self, cse_table, cse_table[id], id)

  def process_call(self, call_primitive, f, tracers, params):
    raise NotImplementedError

  def post_process_call(self, _, out_tracer):
    raise NotImplementedError

  def pack(self, tracers):
    val = pack([t.val for t in tracers])
    id = IDTuple([t.id for t in tracers])
    cse_table = next(t.cse_table for t in tracers if t.cse_table is not None)
    return CSETracer(self, cse_table, val, id)


def default_id(primitive, *ids_in, **params):
  return ID(((primitive,) + tuple(ids_in)
             + tuple(sorted((k, const_id(v)) for k, v in params.items()))))

def assoc_commut_binop_id(prim, id1, id2):
  t1, t2 = type(id1), type(id2)
  if t1 is AssocCommutID and t2 is AssocCommutID and id1.prim == id2.prim == prim:
    return AssocCommutID(prim, heap_merge(id1.sorted_ids, id2.sorted_ids))
  elif t1 is AssocCommutID and id1.prim == prim:
    return AssocCommutID(prim, heap_insert(id1.sorted_ids, id2))
  elif t2 is AssocCommutID and id2.prim == prim:
    return AssocCommutID(prim, heap_insert(id2.sorted_ids, id1))
  else:
    return AssocCommutID(prim, tuple(sorted((id1, id2), key=hash)))


idfuns = {}

def defassoccommut(primitive):
  idfuns[primitive] = assoc_commut_binop_id


defassoccommut(add_jaxvals_p)
