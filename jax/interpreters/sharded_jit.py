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

from functools import partial

from absl import logging
import numpy as onp

from .. import core
from ..abstract_arrays import ShapedArray, ConcreteArray, array_types, abstract_token
from . import partial_eval as pe
from . import xla
from .. import linear_util as lu
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..api_util import flatten_fun
from ..tree_util import tree_flatten, tree_unflatten
from ..util import extend_name_stack, wrap_name

"""WIP shared_jit"""

def _map(f, *xs):
  return tuple(map(f, *xs))


### arg handling


def _spatial_partitioned_args(devices, assignments, partitions, args):
  nargs = len(args)
  nrep, npar = assignments.shape
  # buffers = [[[None] * nargs for _ in range(npar)] for _ in range(nrep)] # TODO
  buffers = [[None] * nargs for _ in range(nrep * npar)]
  for a, (arg, partition) in enumerate(zip(args, partitions)):
    bufs = _partition_array(arg, devices, assignments,
                                             partition)
    for r in range(nrep):
      for p in range(npar):
        # buffers[r][p][a] = bufs[r][p]  # TODO update C++
        buffers[r * npar + p][a] = bufs[r][p]
  return buffers


partition_arg_handlers = {}


def _partition_array(x, devices, assignments, partition):
  nrep, npar = assignments.shape
  assert nrep == 1  # TODO generalize beyond single-replica
  shards = [x]
  for i, parts in enumerate(partition):
    shards = _flatten(onp.split(s, parts, i) for s in shards)
    # logging.error("===== shards: %s" % [s.shape for s in shards])
  bufs = [[None] * npar for _ in range(nrep)]
  for (r, p), device in onp.ndenumerate(assignments):
    bufs[r][p] = xla.device_put(shards[p], devices[device])
  return bufs


def _flatten(lst):
  return [elt for sublst in lst for elt in sublst]


for _t in array_types:
  partition_arg_handlers[_t] = _partition_array

### result handling


def _pvals_to_results_handler(nrep, npar, partitions, out_pvals):
  nouts = len(out_pvals)
  handlers = _map(_pval_to_result_handler, partitions, out_pvals)

  def handler(out_bufs):
    buffers = [[[None] * npar for _ in range(nrep)] for _ in range(nouts)]
    for raw_idx, tuple_buf in enumerate(out_bufs):
      r, p = onp.unravel_index(raw_idx, (nrep, npar))
      for i, buf in enumerate(tuple_buf.destructure()):
        buffers[i][r][p] = buf
    return [h(bufs) for h, bufs in zip(handlers, buffers)]

  return handler


def _pval_to_result_handler(partition, pval):
  pv, const = pval
  if pv is None:
    raise NotImplementedError  # TODO handle constant outputs
  else:
    return _aval_to_result_handler(partition, pv)


def _aval_to_result_handler(partition, aval):
  return result_handlers[type(aval)](partition, aval)


result_handlers = {}


def _array_result_handler(partition, aval):

  def handler(bufs):
    bufs, = bufs  # TODO generalize beyond single replica
    shards = [buf.to_py() for buf in bufs]  # TODO device persistence
    partition = (1,) # TODO (wangtao): revisit this hack.
    for i, parts in enumerate(partition):
      shards = [onp.concatenate(cs, axis=i) for cs in _chunk(shards, parts)]
    result = shards
    return result

  return handler


def _chunk(lst, sz):
  assert not len(lst) % sz
  return [lst[i:i + sz] for i in range(0, len(lst), sz)]


result_handlers[ShapedArray] = _array_result_handler
result_handlers[ConcreteArray] = _array_result_handler

### computation building


@lu.cache
def _sharded_callable(fun, partitions, name, *abstract_args):
  nrep = 1  # TODO generalize

  in_pvals = [pe.PartialVal((aval, core.unit)) for aval in abstract_args]
  with core.new_master(pe.JaxprTrace, True) as master:
    jaxpr, (out_pvals, consts,
            env) = pe.trace_to_subjaxpr(fun, master,
                                        False).call_wrapped(in_pvals)
    assert not env  # no subtraces here
    del master, env

  if not jaxpr.eqns and all(outvar is core.unitvar for outvar in jaxpr.outvars):
    return lambda *_: [core.unit] * len(jaxpr.outvars)


  c = xb.make_computation_builder("spjit_{}".format(fun.__name__))
  xla_consts = _map(c.Constant, consts)
  xla_args = _xla_sharded_args(c, abstract_args, partitions[0])
  axis_env = xla.AxisEnv(nrep, [], [])
  out_nodes = xla.jaxpr_subcomp(
      c, jaxpr, None, axis_env, xla_consts, (),
      extend_name_stack(wrap_name(name, "sharded_jit")),
      *xla_args)
  c._builder.SetSharding(_sharding_to_proto(partitions[1]))
  out_tuple = c.Tuple(*out_nodes)
  c._builder.ClearSharding()
  built = c.Build(out_tuple)

  num_partitions = _get_num_partitions(partitions[0])
  devices = xb.local_devices()[:num_partitions]
  assert len(devices) == num_partitions  # TODO generalize beyond single-replica
  device_assignment = onp.array([[d.id for d in devices]])
  device_assignment = onp.reshape(device_assignment, (-1, num_partitions))
  # device_assignment = None  # TODO(skye): replace with default device assignment?

  compiled = built.Compile(
      compile_options=xb.get_compile_options(nrep, num_partitions, device_assignment),
      backend=xb.get_backend(None))

  # logging.error("===== hlo:\n%s" % built.GetHloText())

  handle_args = partial(_spatial_partitioned_args, compiled.local_devices(),
                        device_assignment, partitions[0])
  handle_outs = _pvals_to_results_handler(nrep, num_partitions, partitions[1],
                                          out_pvals)
  return partial(_execute_spatially_partitioned, compiled, handle_args,
                 handle_outs)


def _sharded_jit_translation_rule(c, jaxpr, axis_env, freevar_nodes,
                                  in_nodes, name_stack, partitions, backend, name):
  subc = xb.make_computation_builder("jaxpr_subcomputation")  # TODO(mattjj): name
  freevars = [subc.ParameterWithShape(c.GetShape(n)) for n in freevar_nodes]

  args = []
  for p, a in zip(partitions[0], in_nodes):
    subc._builder.SetSharding(_sharding_to_proto(p))
    args.append(subc.ParameterWithShape(c.GetShape(a)))
    subc._builder.ClearSharding()
  # args = [subc.ParameterWithShape(c.GetShape(n)) for n in in_nodes]

  out_nodes = xla.jaxpr_subcomp(subc, jaxpr, backend, axis_env, (), freevars, name_stack, *args)

  subc._builder.SetSharding(_sharding_to_proto(partitions[1]))
  out_tuple = subc.Tuple(*out_nodes)
  subc._builder.ClearSharding()

  subc = subc.Build(out_tuple)
  return c.Call(subc, list(freevar_nodes) + list(in_nodes))

def _execute_spatially_partitioned(compiled, in_handler, out_handler, *args):
  input_bufs = in_handler(args)
  out_bufs = compiled.ExecuteOnLocalDevices(list(input_bufs))
  return out_handler(out_bufs)


def _xla_sharded_args(c, avals, partitions):
  xla_args = []
  for p, a in zip(partitions, avals):
    c._builder.SetSharding(_sharding_to_proto(p))
    # logging.error("===== aval shape: %s" % str(a.shape))
    shape = xc.Shape.array_shape(a.dtype, (4,8))
    xla_args.append(c.ParameterWithShape(xla.aval_to_xla_shape(a)))
    c._builder.ClearSharding()
  return xla_args


def _sharding_to_proto(sharding):
  proto = xc.OpSharding()
  if isinstance(sharding, tuple):
    if sharding[0] is None or isinstance(sharding[0], tuple):
      sub_protos = [_sharding_to_proto(s) for s in sharding]
      xc.type = xc.OpSharding.Type.TUPLE
      xc.tuple_shardings = sub_protos
      return proto

  if sharding is None:
    proto.type = xc.OpSharding.Type.REPLICATED
  else:
    proto.type = xc.OpSharding.Type.OTHER
    proto.tile_assignment_dimensions = list(sharding)
    proto.tile_assignment_devices = list(range(onp.product(sharding)))
  return proto


def _get_num_partitions(partitions):
  num_partitions = onp.prod(onp.max(partitions, axis=0))
  return num_partitions


def get_num_partitions(partitions):
  num_partitions_set = set(_get_num_partitions(parts) for parts in partitions)
  if len(num_partitions_set) > 1:
    raise ValueError(
        "All partition specs must use the same number of total partitions, "
        "got: %s" % partitions)
  return num_partitions_set.pop()


def jaxpr_partitions(jaxpr):
  for eqn in jaxpr.eqns:
    if eqn.primitive == sharded_call_p:
      # TODO(skye): change API to support different output partitions
      return (eqn.params["partitions"][0], (eqn.params["partitions"][1],))
      # TODO(skye): more error checking
      # return _get_num_partitions(eqn.params["partitions"][0])
  return None, None



### sharded_call


def _sharded_call_impl(fun, *args, **params):
  partitions = params.pop("partitions")
  name = params.pop("name")
  assert not params, params
  compiled_fun = _sharded_callable(fun, partitions, name,
                                   *map(xla.abstractify, args))
  return compiled_fun(*args)


sharded_call_p = core.Primitive("sharded_call")
sharded_call_p.multiple_results = True
sharded_call = partial(core.call_bind, sharded_call_p)
sharded_call_p.def_custom_bind(sharded_call)
sharded_call_p.def_impl(_sharded_call_impl)
xla.call_translations[sharded_call_p] = _sharded_jit_translation_rule


def sharded_jit(fun, partitions):
  if xb.get_backend().platform != "tpu":
    logging.warning("sharded_jit only works on TPU")

  def wrapped(*args, **kwargs):
    f = lu.wrap_init(fun)
    args_flat, in_tree = tree_flatten((args, kwargs))
    flat_fun, out_tree = flatten_fun(f, in_tree)
    out = sharded_call(flat_fun, *args_flat, partitions=partitions,
                       name=flat_fun.__name__)
    return tree_unflatten(out_tree(), out)

  return wrapped
