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

import enum
from collections import namedtuple

import jax
from . import core
from . import linear_util as lu
from .interpreters import partial_eval as pe
from .interpreters import batching

class LoopType(enum.Enum):
    vectorized = enum.auto()
    parallel = enum.auto()
    sequential = enum.auto()

Loop = namedtuple('Loop', ['type', 'size'])

def gmap_impl(fun: lu.WrappedFun, *args, axis_name, axis_size, mapped_invars, schedule):
  avals = [core.raise_to_shaped(core.get_aval(arg)) for arg in args]
  scheduled_fun = _apply_schedule(fun, axis_size, mapped_invars, schedule, *avals)
  return scheduled_fun(*args)

def _normalize_schedule(schedule, axis_size):
  # TODO: Check vmap only at the end. Check None appears only once!
  scheduled = 1
  for loop in schedule:
    if loop[1] is not None:
      scheduled *= loop[1]
  unscheduled = axis_size / scheduled
  return [Loop(loop[0], loop[1] or unscheduled) for loop in schedule]

@lu.cache
def _apply_schedule(fun: lu.WrappedFun, axis_size, mapped_invars, schedule, *avals):
  mapped_avals = [core.mapped_aval(axis_size, aval) if m else aval
                  for m, aval in zip(mapped_invars, avals)]
  jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, mapped_avals)

  schedule = _normalize_schedule(schedule, axis_size)
  dim_sizes = tuple(loop.size for loop in schedule)

  sched_fun = lambda *args: core.eval_jaxpr(jaxpr, consts, *args)

  if schedule[-1].type is LoopType.vectorized:
    sched_fun = jax.vmap(sched_fun)
    nonvector_schedule = schedule[:-1]
  else:
    nonvector_schedule = schedule
  for (ltype, size) in nonvector_schedule[::-1]:
    if ltype is LoopType.parallel:
      sched_fun = jax.pmap(sched_fun)
    elif ltype is LoopType.sequential:
      sched_fun = lambda *args, sched_fun=sched_fun: jax.lax.map(lambda xs: sched_fun(*xs), args)

  def sched_fun_wrapper(*args):
    split_args = [arg.reshape(dim_sizes + arg.shape[1:]) for arg in args]
    results = sched_fun(*split_args)
    return [res.reshape((axis_size,) + res.shape[len(dim_sizes):]) for res in results]
  return sched_fun_wrapper

gmap_p = core.MapPrimitive('gmap')
gmap = gmap_p.bind
gmap_p.def_impl(gmap_impl)
