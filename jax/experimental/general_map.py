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

import enum
from collections import namedtuple
from typing import Callable
from warnings import warn
from functools import wraps

import jax
from .. import core
from .. import linear_util as lu
from ..api import _TempAxisName, _mapped_axis_size, _check_callable, _check_arg
from ..tree_util import tree_flatten, tree_unflatten
from ..api_util import flatten_fun
from ..interpreters import partial_eval as pe


def gmap(fun: Callable, schedule, axis_name = None) -> Callable:
  warn("gmap is an experimental feature and probably has bugs!")
  _check_callable(fun)

  if axis_name is not None:
    raise ValueError("gmap doesn't support binding axis names yet")

  axis_name = _TempAxisName(fun) if axis_name is None else axis_name

  @wraps(fun)
  def f_gmapped(*args, **kwargs):
    f = lu.wrap_init(fun)
    args_flat, in_tree = tree_flatten((args, kwargs))
    mapped_invars = (True,) * len(args_flat)
    axis_size = _mapped_axis_size(in_tree, args_flat, (0,) * len(args_flat), "gmap")
    for arg in args_flat: _check_arg(arg)
    flat_fun, out_tree = flatten_fun(f, in_tree)
    outs = gmap_p.bind(
        flat_fun, *args_flat,
        axis_name=axis_name,
        axis_size=axis_size,
        mapped_invars=mapped_invars,
        schedule=tuple(schedule))
    return tree_unflatten(out_tree(), outs)
  return f_gmapped


class LoopType(enum.Enum):
    vectorized = enum.auto()
    parallel = enum.auto()
    sequential = enum.auto()

Loop = namedtuple('Loop', ['type', 'size'])


def gmap_impl(fun: lu.WrappedFun, *args, axis_size, axis_name, mapped_invars, schedule):
  avals = [core.raise_to_shaped(core.get_aval(arg)) for arg in args]
  scheduled_fun = _apply_schedule(fun, axis_size, mapped_invars, schedule, *avals)
  return scheduled_fun(*args)

def _parse_name(name):
  if isinstance(name, LoopType):
    return name
  try:
    return LoopType[name]
  except KeyError as err:
    raise ValueError(f"Unrecognized loop type: {name}") from err

def _normalize_schedule(schedule, axis_size):
  scheduled = 1
  seen_none = False
  for loop in schedule:
    if loop[1] is not None:
      scheduled *= loop[1]
    elif seen_none:
      raise ValueError("gmap schedule can only contain at most a single None size specification")
    else:
      seen_none = True
  unscheduled = axis_size // scheduled

  new_schedule = []
  for i, loop in enumerate(schedule):
    loop_type = _parse_name(loop[0])
    if loop_type is LoopType.vectorized and i < len(schedule) - 1:
      raise ValueError("vectorized loops can only appear as the last component of the schedule")
    new_schedule.append(Loop(loop_type, loop[1] or unscheduled))
  return new_schedule

@lu.cache
def _apply_schedule(fun: lu.WrappedFun, axis_size, mapped_invars, schedule, *avals):
  mapped_avals = [core.mapped_aval(axis_size, aval) if mapped else aval
                  for mapped, aval in zip(mapped_invars, avals)]
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
gmap_p.def_impl(gmap_impl)
