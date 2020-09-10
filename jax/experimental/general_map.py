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
  binds_axis_name = axis_name is not None
  axis_name = _TempAxisName(fun) if axis_name is None else axis_name

  @wraps(fun)
  def f_gmapped(*args, **kwargs):
    f = lu.wrap_init(fun)
    args_flat, in_tree = tree_flatten((args, kwargs))
    mapped_invars = (True,) * len(args_flat)
    axis_size = _mapped_axis_size(in_tree, args_flat, (0,) * len(args_flat), "gmap")
    parsed_schedule = _normalize_schedule(schedule, axis_size, binds_axis_name)
    for arg in args_flat: _check_arg(arg)
    flat_fun, out_tree = flatten_fun(f, in_tree)
    outs = gmap_p.bind(
        flat_fun, *args_flat,
        axis_name=axis_name,
        axis_size=axis_size,
        mapped_invars=mapped_invars,
        schedule=parsed_schedule,
        binds_axis_name=binds_axis_name)
    return tree_unflatten(out_tree(), outs)
  return f_gmapped


class LoopType(enum.Enum):
    vectorized = enum.auto()
    parallel = enum.auto()
    sequential = enum.auto()

Loop = namedtuple('Loop', ['type', 'size'])


def _normalize_schedule(schedule, axis_size, binds_axis_name):
  if not schedule:
    raise ValueError("gmap expects a non-empty schedule")

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
    if loop_type is LoopType.sequential and binds_axis_name:
      raise ValueError("gmaps that bind a new axis name cannot have sequential components in the schedule")
    new_schedule.append(Loop(loop_type, loop[1] or unscheduled))
  return tuple(new_schedule)

def _parse_name(name):
  if isinstance(name, LoopType):
    return name
  try:
    return LoopType[name]
  except KeyError as err:
    raise ValueError(f"Unrecognized loop type: {name}") from err


def gmap_impl(fun: lu.WrappedFun, *args, axis_size, axis_name, binds_axis_name, mapped_invars, schedule):
  avals = [core.raise_to_shaped(core.get_aval(arg)) for arg in args]
  scheduled_fun = _apply_schedule(fun, axis_size, axis_name, binds_axis_name,
                                  mapped_invars, schedule, *avals)
  return scheduled_fun(*args)

class _GMapSubaxis:
  def __init__(self, axis_name, index):
    self.axis_name = axis_name
    self.index = index
  def __repr__(self):
    return f'<subaxis {self.index} of {self.axis_name}>'
  def __hash__(self):
    return hash((self.axis_name, self.index))
  def __eq__(self, other):
    return (isinstance(other, _GMapSubaxis) and
            self.axis_name == other.axis_name and
            self.index == other.index)

@lu.cache
def _apply_schedule(fun: lu.WrappedFun,
                    axis_size, full_axis_name, binds_axis_name,
                    mapped_invars,
                    schedule,
                    *avals):
  assert all(mapped_invars)
  mapped_avals = [core.mapped_aval(axis_size, aval) if mapped else aval
                  for mapped, aval in zip(mapped_invars, avals)]
  with core.extend_axis_env(full_axis_name, axis_size, None):
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, mapped_avals)

  axis_names = tuple(_GMapSubaxis(full_axis_name, i) for i in range(len(schedule)))
  if binds_axis_name:
    jaxpr = subst_axis_names(jaxpr, full_axis_name, axis_names)

  sched_fun = lambda *args: core.eval_jaxpr(jaxpr, consts, *args)
  for (ltype, size), axis_name in list(zip(schedule, axis_names))[::-1]:
    if ltype is LoopType.vectorized:
      sched_fun = jax.vmap(sched_fun, axis_name=axis_name)
    elif ltype is LoopType.parallel:
      sched_fun = jax.pmap(sched_fun, axis_name=axis_name)
    elif ltype is LoopType.sequential:
      if binds_axis_name:
        raise NotImplementedError("gmaps with sequential components of the schedule don't support "
                                  "collectives yet. Please open a feature request!")
      assert not binds_axis_name
      sched_fun = lambda *args, sched_fun=sched_fun: jax.lax.map(lambda xs: sched_fun(*xs), args)

  dim_sizes = tuple(loop.size for loop in schedule)
  def sched_fun_wrapper(*args):
    split_args = [arg.reshape(dim_sizes + arg.shape[1:]) for arg in args]
    results = sched_fun(*split_args)
    return [res.reshape((axis_size,) + res.shape[len(dim_sizes):]) for res in results]
  return sched_fun_wrapper

gmap_p = core.MapPrimitive('gmap')
gmap_p.def_impl(gmap_impl)


def subst_axis_names(jaxpr, replaced_name, axis_names):
  eqns = [subst_eqn_axis_names(eqn, replaced_name, axis_names) for eqn in jaxpr.eqns]
  return core.Jaxpr(jaxpr.constvars, jaxpr.invars, jaxpr.outvars, eqns)

def subst_eqn_axis_names(eqn, replaced_name, axis_names):
  if isinstance(eqn.primitive, (core.CallPrimitive, core.MapPrimitive)):
    if eqn.params.get('axis_name', None) == replaced_name:  # Check for shadowing
      return eqn
    new_call_jaxpr = subst_axis_names(eqn.params['call_jaxpr'], replaced_name, axis_names)
    return eqn._replace(params=dict(eqn.params, call_jaxpr=new_call_jaxpr))
  elif eqn.params.get('axis_name', None) == replaced_name:
    return eqn._replace(params=dict(eqn.params, axis_name=axis_names))
  else:
    return eqn
