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

from contextlib import contextmanager
from functools import partial
from functools import wraps
import itertools
import threading

import numpy as onp

from .. import core
from .. import linear_util as lu
from .. import partition_util
from ..abstract_arrays import raise_to_shaped
from ..api_util import flatten_fun
from ..config import flags
from ..interpreters import ad
from ..interpreters import partial_eval as pe
from ..interpreters import xla
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..tree_util import tree_flatten
from ..tree_util import tree_map
from ..tree_util import tree_unflatten
from ..util import cache
from ..util import unzip2


FLAGS = flags.FLAGS


class _ThreadLocalState(threading.local):
  def __init__(self):
    self.partition_ids_stack = []
_thread_local_state = _ThreadLocalState()


def _get_partition_ids(num_partitions, outer_partition_ids):
  """Returns the partitions IDs after remapping the `outer_partition_ids`."""
  partition_ids_stack = _thread_local_state.partition_ids_stack
  if partition_ids_stack:
    if outer_partition_ids is None:
      raise ValueError(
          "`outer_partition_ids` must be specified for nested calls to "
          "`jax.partition`.")
    elif len(outer_partition_ids) != num_partitions:
      raise ValueError(
          "`outer_partition_ids` length ({}) did not match the number of "
          "partitions ({})".format(len(outer_partition_ids), num_partitions))
    else:
      return tuple(partition_ids_stack[-1][i] for i in outer_partition_ids)
  else:
    if outer_partition_ids is not None:
      raise ValueError(
          "`outer_partition_ids` must not be set for outermost `jax.partition` "
          "call")
    else:
      return tuple(range(num_partitions))


def _abstractify(x):
  return raise_to_shaped(core.get_aval(x))


def partition(fun, num_partitions):
  """Partitions the given function across multiple devices.

  Operations will be executed on the same partition as their inputs (throwing an
  error if there is a mismatch). Values can be moved between partitions using
  `partition_put`.

  Example:
    ```
    @partial(partition, num_partitions=2)
    def f(w):
      x = w ** 2
      y = jax.partition_put(x, 1)
      z = 42 * y
      return z

    result = f(1.)
    ```

    `x` will be computed on partition 0, `z` on partition 1.

  Note that if a call to a partitioned function is nested inside another, it
  must specify `outer_partition_ids`. See `partition_put` for more information.

  Args:
    fun: The function to be partitioned.
    num_partitions: The number of partitions.

  Returns:
    A partitioned function, with an additional, keyword-only parameter,
    `outer_partition_ids` that must be used when called within another
    partitioned function.
  """
  @wraps(fun)
  def fun_partitioned(*args, outer_partition_ids=None, **kwargs):
    partition_ids = _get_partition_ids(num_partitions, outer_partition_ids)

    _thread_local_state.partition_ids_stack.append(partition_ids)
    try:
      in_vals, in_tree = tree_flatten((args, kwargs))
      in_avals = _map(_abstractify, in_vals)
      in_pvals = [pe.PartialVal((aval, core.unit)) for aval in in_avals]
      fun1, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
      jaxpr, out_pvals, consts = pe.trace_to_jaxpr(
          fun1, in_pvals, instantiate=True, stage_out_calls=True)
      out_avals = _map(raise_to_shaped, unzip2(out_pvals)[0])
      const_avals = _map(_abstractify, consts)
      jaxpr = core.TypedJaxpr(
          pe.convert_constvars_jaxpr(jaxpr), (), const_avals + in_avals, out_avals)
      outs = partition_p.bind(*itertools.chain(consts, in_vals),
                              jaxpr=jaxpr, partition_ids=partition_ids)
      return tree_unflatten(out_tree(), outs)
    finally:
      _thread_local_state.partition_ids_stack.pop()

  return fun_partitioned


def _partition_impl(*args, **params):
  return _partition_callable(*map(xla.arg_spec, args), **params)(*args)


@cache()
def _partition_callable(*args, jaxpr, partition_ids):
  nparts = max(partition_ids) + 1

  if nparts > xb.device_count():
    msg = ("compiling computation that requires {} devices, but only {} XLA "
           "devices are available")
    raise ValueError(msg.format(nparts, xb.device_count()))

  abstract_args, arg_devices = unzip2(args)
  result_handlers = tuple(map(partial(xla.aval_to_result_handler, None), jaxpr.out_avals))

  var_partition_ids = partition_util.get_var_partition_ids(jaxpr.jaxpr, partition_ids)
  out_partition_ids = tuple(var_partition_ids[v] for v in jaxpr.jaxpr.outvars)

  tuple_args = len(abstract_args) > 100
  built = _partition_computation(
      *abstract_args, tuple_args=tuple_args, jaxpr=jaxpr, partition_ids=partition_ids)

  device_assignment = xb.get_backend().get_default_device_assignment(1, nparts)
  device_assignment = onp.vectorize(lambda d: d.id)(onp.array(device_assignment))

  options = xb.get_compile_options(
      num_replicas=1, num_partitions=nparts, device_assignment=device_assignment)
  compiled = built.Compile(compile_options=options, backend=xb.get_backend())

  return xla.create_execute_fn(
      compiled, device_assignment=device_assignment, backend=None,
      tuple_args=tuple_args, result_handlers=result_handlers,
      out_partition_ids=out_partition_ids)


@cache()
def _partition_computation(*avals, tuple_args, jaxpr, partition_ids):
  c = xb.make_computation_builder('partition_computation')
  backend = xb.get_backend()
  xla_args = xla._xla_callable_args(c, avals, tuple_args)
  out = _partition_translation_rule(
      c, xla.AxisEnv(1), 'partition/', *xla_args,
      jaxpr=jaxpr, partition_ids=partition_ids, backend=backend)
  return c.Build(out)


def _partition_abstract_eval(*args, jaxpr, **other_params):
  del args, other_params  # Unused.
  return _map(raise_to_shaped, jaxpr.out_avals)


@contextmanager
def _op_partition(computation_builder, partition_id):
  if partition_id is None:
    yield
  else:
    sharding = xc.OpSharding()
    sharding.type = xc.OpSharding.Type.MAXIMAL
    sharding.tile_assignment_devices = [partition_id]
    computation_builder.SetSharding(sharding)
    try:
      yield
    finally:
      computation_builder.ClearSharding()


def _map(f, *xs): return tuple(map(f, *xs))


def _partition_translation_rule(
    c, axis_env, name_stack, *in_nodes, jaxpr, partition_ids, backend):
  jaxpr = jaxpr.jaxpr  # The `jaxpr` parameter is a `TypedJaxpr`.
  nodes = {}

  def read_node(v):
    if type(v) is core.Literal:
      return c.Constant(xla.canonicalize_dtype(v.val))
    else:
      return nodes[v]

  def write_node(v, node):
    assert node is not None
    nodes[v] = node

  write_node(core.unitvar, c.Tuple())
  _map(write_node, jaxpr.invars, in_nodes)

  var_partition_ids = partition_util.get_var_partition_ids(jaxpr, partition_ids)

  for eqn in jaxpr.eqns:
    invars = [v for v in eqn.invars if type(v) is not core.Literal]

    if invars:
      # Take partition ID from the first input var. We've already checked, in
      # `_get_var_partition_ids`, that they all match.
      partition_id = var_partition_ids[invars[0]]
    else:
      partition_id = None

    with _op_partition(c, partition_id):
      out_nodes = xla.jaxpr_subcomp_eqn(
          c, eqn, backend, axis_env, name_stack, read_node)
    _map(write_node, eqn.outvars, out_nodes)
  return c.Tuple(*_map(read_node, jaxpr.outvars))


def _partition_jvp(primals, tangents, jaxpr):
  raise NotImplementedError()  # TODO(cjfj)


partition_p = core.Primitive('partition')
partition_p.multiple_results = True
partition_p.def_impl(_partition_impl)
partition_p.def_abstract_eval(_partition_abstract_eval)
ad.primitive_jvps[partition_p] = _partition_jvp
xla.initial_style_translations[partition_p] = _partition_translation_rule


def partition_put(x, partition_id):
  """Places the given value(s) onto the specified partition.

  Note that for nested `partition` calls, the `partition_id` is the index into
  the list of partition IDs used by the innermost partitioned function.

  Example:
    ```
    @partial(partition, num_partitions=2)
    def f(w):
      x = w ** 2
      y = jax.partition_put(x, 1)
      z = 42 * y
      return z

    @partial(partition, num_partitions=4)
    def g(i):
      j = i ** 3
      k = f(i, outer_partition_ids=[1, 2])
      l = 42 * jax.partition_put(k, 3)
      return l

    b = g(a)
    ```

    `g` is the outermost partitioned function here, so partition IDs map
    directly to the logical XLA devices. `f` is nested, with its two partitions
    mapped onto `[1, 2]` using `outer_partition_ids`. `x` is therefore
    calculated on XLA partition 1, whilst `z` is calculated on XLA partition 2.

  Args:
    x: The value(s) to copy. This can be a tree of values.
    partition_id: The partition ID, within the innermost `partition` scope,
      on which to place the value(s).

  Returns:
    The input values, copied onto the specified partition.
  """
  return tree_map(partial(partition_put_p.bind, partition_id=partition_id), x)


def _partition_put_impl(x, *, partition_id):
  del x, partition_id  # Unused.
  raise ValueError("`partition_put` only supported within a `partition` call.")


# TODO(cjfj): Will XLA elide unnecessary copies between partitions?
def _partition_put_translation_rule(c, x, *, partition_id):
  del partition_id  # Unused (already applied by translation rule).
  # There is no identity op in XLA, so we'll use a no-op reshape instead.
  return c.Reshape(x, dimensions=None, new_sizes=c.GetShape(x).dimensions())


partition_put_p = core.Primitive("partition_put")
partition_put_p.def_impl(_partition_put_impl)
partition_put_p.def_abstract_eval(lambda x, partition_id: x)
xla.translations[partition_put_p] = _partition_put_translation_rule
ad.deflinear(partition_put_p, lambda cotangent, **kwargs: [cotangent])
