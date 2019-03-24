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

from .. import core
from ..core import Trace, Tracer, new_master, pack, AbstractTuple, JaxTuple
from ..linear_util import transformation, transformation_with_aux, wrap_init
from ..util import unzip2, partial, safe_map, safe_zip, WrapHashably

map = safe_map
zip = safe_zip


class ID(WrapHashably): pass
class IDTuple(tuple): pass
class IDAddList(list): pass

def const_id(val):
  try:
    hash(val)
    return val
  except TypeError:
    return ID(val)

@transformation
def cse(*args):
  with new_master(CSETrace) as master:
    cse_table = {}
    arg_ids = map(const_id, args)
    trace = CSETrace(master, core.cur_sublevel())
    in_tracers = map(partial(CSETracer, trace, cse_table), args, arg_ids)
    out = yield in_tracers
    out_tracer = trace.full_raise(out)
    out_val = out_tracer.val
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
    if t is ID:
      raise NotImplementedError  # reachable?
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
    cse_table = next(t.cse_table for t in tracers if t.cse_table is not None)
    # TODO(mattjj); handle add with a primitive registry of id-computing rules
    id = ((primitive,) + tuple(ids_in)
           + tuple(sorted((k, const_id(v)) for k, v in params.items())))
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
