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
from typing import Callable, Iterable, Optional, Tuple, Union

from absl import logging
import numpy as np

from .. import core
from . import ad
from . import partial_eval as pe
# TODO(skye): separate pmap into it's own module?
from . import pxla
from . import xla
from .. import linear_util as lu
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..api_util import argnums_partial, flatten_axes, flatten_fun, _ensure_index_tuple
from ..tree_util import tree_flatten, tree_unflatten
from .._src.util import (extend_name_stack, wrap_name, wraps, safe_zip,
                         HashableFunction)
from .._src.config import config

xops = xc._xla.ops


def _map(f, *xs):
  return tuple(map(f, *xs))


class ResultToPopulate: pass
result_to_populate = ResultToPopulate()


def _avals_to_results_handler(nrep, npart, partitions, out_avals):
  handlers = [_aval_to_result_handler(npart, parts, out_aval)
              for parts, out_aval in safe_zip(partitions, out_avals)]

  def handler(out_bufs):
    return [h(bufs) for h, bufs in zip(handlers, out_bufs)]

  return handler

def _aval_to_result_handler(npart, parts, aval):
  if aval is not core.abstract_unit:
    spec = pxla.partitioned_sharding_spec(npart, parts, aval)
    indices = pxla.spec_to_indices(aval.shape, spec)
  else:
    spec = indices = None
  return pxla.aval_to_result_handler(spec, indices, aval)


@lu.cache
def _sharded_callable(
    fun: lu.WrappedFun, nparts: Optional[int],
    in_parts: Tuple[pxla.PartitionsOrReplicated, ...],
    out_parts_thunk: Callable[[], Tuple[pxla.PartitionsOrReplicated, ...]],
    local_in_parts: Optional[Tuple[pxla.PartitionsOrReplicated, ...]],
    local_out_parts_thunk: Callable[[], Optional[Tuple[pxla.PartitionsOrReplicated, ...]]],
    local_nparts: Optional[int], name: str, *abstract_args):
  nrep = 1

  if local_in_parts is None:
    local_in_parts = in_parts

  global_abstract_args = [pxla.get_global_aval(arg, parts, lparts)
                          for arg, parts, lparts
                          in safe_zip(abstract_args, in_parts, local_in_parts)]

  logging.vlog(2, "abstract_args: %s", abstract_args)
  logging.vlog(2, "global_abstract_args: %s", global_abstract_args)
  logging.vlog(2, "in_parts: %s", in_parts)
  logging.vlog(2, "local_in_parts: %s", local_in_parts)

  jaxpr, global_out_avals, consts = pe.trace_to_jaxpr_final(fun, global_abstract_args)

  if xb.get_backend().platform not in ["tpu", "gpu"]:
    # TODO(skye): fall back to regular jit?
    raise ValueError("sharded_jit not supported for " +
                     xb.get_backend().platform)

  nparts = pxla.reconcile_num_partitions(jaxpr, nparts)
  assert nparts is not None
  if nparts > xb.device_count():
    raise ValueError(
        f"sharded_jit computation requires {nparts} devices, "
        f"but only {xb.device_count()} devices are available.")
  if xb.local_device_count() < nparts < xb.device_count():
    raise NotImplementedError(
        f"sharded_jit across multiple hosts must use all available devices. "
        f"Got {nparts} out of {xb.device_count()} requested devices "
        f"(local device count: {xb.local_device_count()})")

  if local_nparts is None:
    if nparts > xb.local_device_count():
      raise ValueError(
        "Specify 'local_nparts' when using cross-process sharded_jit "
        "and all inputs and outputs are replicated.")
    else:
      local_nparts = nparts
  if local_nparts > xb.local_device_count():
    raise ValueError(
        f"sharded_jit computation requires {local_nparts} local devices, "
        f"but only {xb.local_device_count()} local devices are available.")

  logging.vlog(2, "nparts: %d  local_nparts: %d", nparts, local_nparts)

  out_parts = out_parts_thunk()

  local_out_parts = local_out_parts_thunk()
  if local_out_parts is None:
    local_out_parts = out_parts

  logging.vlog(2, "out_parts: %s", out_parts)
  logging.vlog(2, "local_out_parts: %s", local_out_parts)

  local_out_avals = [pxla.get_local_aval(out, parts, lparts)
                     for out, parts, lparts
                     in safe_zip(global_out_avals, out_parts, local_out_parts)]

  log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
  logging.log(log_priority,
              f"Compiling {fun.__name__} for {nparts} devices with "
              f"args {global_abstract_args}.")

  c = xb.make_computation_builder("spjit_{}".format(fun.__name__))
  xla_consts = _map(partial(xb.constant, c), consts)
  xla_args = _xla_sharded_args(c, global_abstract_args, in_parts)
  axis_env = xla.AxisEnv(nrep, (), ())
  out_nodes = xla.jaxpr_subcomp(
      c, jaxpr, None, axis_env, xla_consts,
      extend_name_stack(wrap_name(name, "sharded_jit")), *xla_args)
  out_tuple = xb.with_sharding(c, out_parts, xops.Tuple, c, out_nodes)
  built = c.Build(out_tuple)

  if nparts <= xb.local_device_count():
    devices = xb.local_devices()[:nparts]
  else:
    assert nparts == xb.device_count()
    devices = xb.devices()
  device_assignment = np.array([[d.id for d in devices]])
  device_assignment = np.reshape(device_assignment, (-1, nparts))
  # device_assignment = None  # TODO(skye): replace with default device assignment?

  compiled = xla.backend_compile(
      xb.get_backend(), built,
      xb.get_compile_options(nrep, nparts, device_assignment))

  input_specs = [
      pxla.partitioned_sharding_spec(local_nparts, parts, aval)
      for parts, aval in zip(local_in_parts, abstract_args)]
  input_indices = [pxla.spec_to_indices(aval.shape, spec)
                   if spec is not None else None
                   for aval, spec in zip(abstract_args, input_specs)]

  handle_args = partial(pxla.shard_args, compiled.local_devices(),
                        input_indices)
  handle_outs = _avals_to_results_handler(nrep, local_nparts,  # type: ignore
                                          local_out_parts, local_out_avals)
  return partial(_execute_spatially_partitioned, compiled, handle_args,
                 handle_outs)


def _sharded_jit_translation_rule(c, axis_env, in_nodes, name_stack,
                                  in_parts, out_parts_thunk, nparts, backend,
                                  name, call_jaxpr, local_in_parts,
                                  local_out_parts_thunk, local_nparts):
  subc = xc.XlaBuilder(f"sharded_jit_{name}")

  # We assume any extra leading in_nodes are constants and replicate them.
  num_extra_nodes = len(in_nodes) - len(in_parts)
  assert num_extra_nodes >= 0
  in_parts = (None,) * num_extra_nodes + in_parts

  args = []
  for i, (n, sharding) in enumerate(safe_zip(in_nodes, in_parts)):
    # We use xb.set_sharding instead of xb.with_sharding because inlined calls
    # shouldn't have shardings set directly on the inputs or outputs.
    arg = xb.parameter(subc, i, c.GetShape(n))
    args.append(xb.set_sharding(subc, arg, sharding))

  out_nodes = xla.jaxpr_subcomp(
      subc, call_jaxpr, backend, axis_env, (),
      extend_name_stack(name_stack, wrap_name(name, "sharded_jit")), *args)
  out_parts = out_parts_thunk()
  assert len(out_parts) == len(out_nodes)
  out_nodes = [xb.set_sharding(subc, out, sharding)
               for out, sharding in safe_zip(out_nodes, out_parts)]

  subc = subc.build(xops.Tuple(subc, out_nodes))
  return xops.Call(c, subc, list(in_nodes))


def _execute_spatially_partitioned(compiled, in_handler, out_handler, *args):
  input_bufs = in_handler(args)
  out_bufs = compiled.execute_sharded_on_local_devices(input_bufs)
  return out_handler(out_bufs)


def _xla_sharded_args(c, avals, in_parts):
  xla_args = []
  for i, (sharding, aval) in enumerate(safe_zip(in_parts, avals)):
    param = xb.with_sharding(c, sharding, xb.parameter, c, i,
                             *xla.aval_to_xla_shapes(aval))
    xla_args.append(param)
  return xla_args


def _sharded_call_impl(fun, *args, nparts, in_parts, out_parts_thunk,
                       local_in_parts, local_out_parts_thunk, local_nparts,
                       name):
  compiled_fun = _sharded_callable(fun, nparts, in_parts, out_parts_thunk,
                                   local_in_parts, local_out_parts_thunk,
                                   local_nparts, name,
                                   *map(xla.abstractify, args))
  return compiled_fun(*args)


sharded_call_p = core.CallPrimitive("sharded_call")
sharded_call = sharded_call_p.bind
sharded_call_p.def_impl(_sharded_call_impl)
xla.call_translations[sharded_call_p] = _sharded_jit_translation_rule


class PartitionSpec(tuple):
  """Tuple of integer specifying how a value should be partitioned.

  Each integer corresponds to how many ways a dimension is partitioned. We
  create a separate class for this so JAX's pytree utilities can distinguish it
  from a tuple that should be treated as a pytree.
  """
  def __new__(cls, *partitions):
    return tuple.__new__(PartitionSpec, partitions)

  def __repr__(self):
    return "PartitionSpec%s" % tuple.__repr__(self)


def sharded_jit(fun: Callable, in_parts, out_parts, num_partitions: int = None,
                local_in_parts=None, local_out_parts=None,
                local_num_partitions=None,
                static_argnums: Union[int, Iterable[int]] = (),
):
  """Like ``jit``, but partitions ``fun`` across multiple devices.

  WARNING: this feature is still under active development! It may not work well,
  and may change without warning!

  `sharded_jit` sets up ``fun`` for just-in-time compilation with XLA, but
  unlike ``jit``, the compiled function will run across multiple devices
  (e.g. multiple GPUs or multiple TPU cores). This is achieved by spatially
  partitioning the data that flows through the computation, so each operation is
  run across all devices and each device runs only a shard of the full
  data. (Some data can optionally be replicated, which is sometimes more
  efficient for small arrays when combined with larger spatially-partitioned
  arrays.) Communication between devices is automatically inserted as necessary.

  ``sharded_jit`` can be useful if the jitted version of ``fun`` would not fit
  in a single device's memory, or to speed up ``fun`` by running each operation
  in parallel across multiple devices.

  Note: ``sharded_jit`` is currently available on TPU only!

  Args:
    fun: Function to be jitted.
    in_parts: Specifications for how each argument to ``fun`` should be
      partitioned or replicated. This should be a PartitionSpec indicating into
      how many partitions each dimension should be sharded, ``None`` indicating
      replication, or (nested) standard Python containers thereof. For example,
      ``in_parts=PartitionSpec(2,1)`` means all arguments should be partitioned
      over two devices across the first dimension;
      ``in_parts=(PartitionSpec(2,2), PartitionSpec(4,1), None)`` means the
      first argument should be partitioned over four devices by splitting both
      of its dimensions in half, the second argument should be partitioned over
      the four devices across the first dimension, and the third argument is
      replicated across the four devices.

      All PartitionSpecs in a given ``sharded_jit`` call must correspond to the
      same total number of partitions, i.e. the product of all PartitionSpecs
      must be equal, and the number of dimensions in the PartitionSpec
      corresponding to an array ``a`` should equal ``a.ndim``. Arguments marked
      as static using ``static_argnums`` (see below) do not require a
      PartitionSpec.
    out_parts: The output partitions, i.e. how each output of ``fun`` should be
      partitioned or replicated. This follows the same convention as
     ``in_parts``.
    num_partitions: Optional. If set, explicitly specifies the number of devices
      ``fun`` should partitioned across (rather than inferring it from
      ``in_parts``, ``out_parts``, and/or any ``with_sharding_constraint``
      calls).  Setting this should usually be unnecessary, but can be used to
      maintain device persistence across multiple sharded_jit calls when some of
      those calls only involve replicated values.
    local_in_parts: Optional. This should be set when partitioning across
      multiple processes, and says how each process's worth of data should be
      partitioned (vs. in_parts which is the "global" partitioning across all
      processes). This API is likely to change in the future.
    local_out_parts: Optional. This should be set when partitioning across
      multiple processes, and says how each process's worth of data should be
      partitioned (vs. out_parts which is the "global" partitioning across all
      processes). This API is likely to change in the future.
    local_num_partitions: Optional. Explicitly specifies the numbers of local
      devices to partitions across in a multi-process setting. This API is
      likely to change in the future.
    static_argnums: An int or collection of ints specifying which positional
      arguments to treat as static (compile-time constant). Operations that only
      depend on static arguments will be constant-folded. Calling the jitted
      function with different values for these constants will trigger
      recompilation. If the jitted function is called with fewer positional
      arguments than indicated by ``static_argnums`` then an error is raised.
      Each of the static arguments will be broadcasted to all devices, and
      cannot be partitioned - these arguments will be removed from the *args
      list before matching each remaining argument with its corresponding
      PartitionSpec. Arguments that are not arrays or containers thereof must
      be marked as static. Defaults to ``()``.

  Returns:
    A version of ``fun`` that will be distributed across multiple devices.
  """
  if num_partitions is not None:
    nparts = num_partitions
  else:
    nparts = pxla.get_num_partitions(in_parts, out_parts)

  if local_num_partitions is not None:
    local_nparts = local_num_partitions
  else:
    local_nparts = pxla.get_num_partitions(local_in_parts, local_out_parts)

  static_argnums = _ensure_index_tuple(static_argnums)

  @wraps(fun)
  def wrapped(*args, **kwargs):
    if kwargs:
      raise NotImplementedError("sharded_jit over kwargs not yet supported")

    f = lu.wrap_init(fun)
    if static_argnums:
      if max(static_argnums) >= len(args):
        raise ValueError(
            f"jitted function has static_argnums={static_argnums}"
            f" but was called with only {len(args)} positional "
            f"argument{'s' if len(args) > 1 else ''}. "
            "All static broadcasted arguments must be passed positionally.")
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      f, args = argnums_partial(f, dyn_argnums, args)

    args_flat, in_tree = tree_flatten((args, kwargs))
    in_parts_flat = tuple(flatten_axes("sharded_jit in_parts",
                                       in_tree.children()[0], in_parts))
    if local_in_parts is not None:
      local_in_parts_flat = tuple(flatten_axes("sharded_jit local_in_parts",
                                               in_tree.children()[0], local_in_parts))
    else:
      local_in_parts_flat = None

    flat_fun, out_tree = flatten_fun(f, in_tree)
    # TODO(skye): having a function-typed param in a primitive seems dicey, is
    # there a better way?
    out_parts_thunk = HashableFunction(
        lambda: tuple(flatten_axes("sharded_jit out_parts", out_tree(), out_parts)),
        closure=out_parts)
    if local_out_parts:
      local_out_parts_thunk = HashableFunction(
          lambda: tuple(flatten_axes("sharded_jit local_out_parts",
                                     out_tree(), local_out_parts)),
          closure=local_out_parts)
    else:
      local_out_parts_thunk = HashableFunction(lambda: None, closure=None)

    out = sharded_call(
        flat_fun,
        *args_flat,
        nparts=nparts,
        in_parts=in_parts_flat,
        out_parts_thunk=out_parts_thunk,
        local_in_parts=local_in_parts_flat,
        local_out_parts_thunk=local_out_parts_thunk,
        local_nparts=local_nparts,
        name=flat_fun.__name__)
    return tree_unflatten(out_tree(), out)

  return wrapped


def _sharding_constraint_impl(x, partitions):
  # TODO(skye): can we also prevent this from being called in other
  # non-sharded_jit contexts? (e.g. pmap, control flow)
  raise NotImplementedError(
      "with_sharding_constraint() should only be called inside sharded_jit()")

def _sharding_constraint_translation_rule(c, x_node, partitions):
  return xb.set_sharding(c, x_node, partitions)

sharding_constraint_p = core.Primitive("sharding_constraint")
sharding_constraint_p.def_impl(_sharding_constraint_impl)
sharding_constraint_p.def_abstract_eval(lambda x, partitions: x)
ad.deflinear2(sharding_constraint_p,
              lambda ct, _, partitions: (with_sharding_constraint(ct, partitions),))
xla.translations[sharding_constraint_p] = _sharding_constraint_translation_rule

def with_sharding_constraint(x, partitions: Optional[PartitionSpec]):
  """Identity-like function that specifies how ``x`` should be sharded.

  WARNING: this feature is still under active development! It may not work well,
  and may change without warning!

  This should only be called inside a function transformed by ``sharded_jit``.
  It constrains how the function is sharded: regardless of any other specified
  partitions, the compiler will make sure that ``x`` is sharded according to
  ``partitions``.  Note that a ``with_sharding_constraint`` call doesn't
  necessarily correspond to a reshard, since the compiler is free to achieve
  this sharding as long as the constraint is met, e.g. it might insert a reshard
  earlier in the computation. Another way to think of this is that the
  ``with_sharding_constraint`` call may flow "up" the function to preceding
  operations as well as "down" to subsequent ones.

  ``partitions`` must correspond to the same number of total partitions dictated
  by the outer ``sharded_jit`` and any other ``with_sharding_constraint`` calls.
  In the case where only replication has been specified, any ``partitions`` are
  valid.

  Example usage:
    @partial(sharded_jit, in_parts=None, out_parts=None, num_shards=2
    def f(x):
      y = x + 1
      y = with_sharding_constraint(y, PartitionSpec(2,1))
      return y * 2

  In this example, the inputs and outputs of ``f`` will be replicated, but the
  inner value of ``y`` will be partitioned in half. ``f`` will run on two
  devices due to the with_sharding_constraint call.

  Args:
    x: Array value
    partitions: PartitionSpec indicating how ``x`` should be partitioned, or
      None for replication.

  Returns:
    A new version of ``x`` with the specified sharding applied.
  """
  return sharding_constraint_p.bind(x, partitions=partitions)
