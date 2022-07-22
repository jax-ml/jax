# Copyright 2021 Google LLC
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

from enum import IntEnum
import numpy as np
from collections import OrderedDict, Counter
from typing import Callable, Sequence, Tuple, Union, cast, List
import itertools as it
from functools import partial, lru_cache

from jax.experimental import maps
from jax.experimental.global_device_array import GlobalDeviceArray as GDA
from jax.experimental.sharding import MeshPspecSharding, Sharding, XLACompatibleSharding
from jax import core
from jax import linear_util as lu
from jax import stages
from jax._src.api import _check_callable, _check_arg
from jax._src.config import config
from jax._src import dispatch
from jax._src import source_info_util
from jax._src.api_util import (argnums_partial_except, flatten_axes,
                               flatten_fun_nokwargs, _ensure_index_tuple,
                               donation_vector, rebase_donate_argnums,
                               shaped_abstractify)
from jax.errors import JAXTypeError
from jax.interpreters import ad
from jax.interpreters import mlir
from jax.interpreters import pxla
from jax.interpreters import xla
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
from jax.interpreters.pxla import PartitionSpec
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib import xla_client as xc
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           treedef_is_leaf, tree_structure, treedef_tuple)
from jax._src.tree_util import prefix_errors
from jax._src import util
from jax._src.util import (
    HashableFunction, safe_map, safe_zip, wrap_name, wraps,
    distributed_debug_log, split_list, tuple_insert, weakref_lru_cache,
    merge_lists)

class _FromGdaSingleton:
  pass
FROM_GDA = _FromGdaSingleton()

def _is_from_gda(x):
  # It's occasionally possible to end up with two FROM_GDA singletons (e.g. if
  # pickling in_axis_resources and sending to other processes). Make sure this
  # doesn't cause an error to avoid user confusion.
  return isinstance(x, type(FROM_GDA))

_AUTOAxisResource = pxla._AUTOAxisResource
AUTO = pxla.AUTO
_is_auto = pxla._is_auto


class PjitCompiled(stages.Compiled):

  @pxla.maybe_cached_property
  def input_shardings(self) -> Sequence[XLACompatibleSharding]:
    args, kwargs = tree_unflatten(self.in_tree, self._executable._in_shardings)    # pytype: disable=attribute-error
    assert not kwargs
    return args

  @pxla.maybe_cached_property
  def output_shardings(self) -> Sequence[XLACompatibleSharding]:
    return tree_unflatten(self.out_tree, self._executable._out_shardings)  # pytype: disable=attribute-error


class PjitLowered(stages.Lowered):

  def compile(self) -> PjitCompiled:
    return PjitCompiled(self._lowering.compile(), self.args_info, self.out_tree,
                        no_kwargs=self._no_kwargs)


_UnspecifiedValue = pxla._UnspecifiedValue
_UNSPECIFIED = pxla._UNSPECIFIED
_is_unspecified = pxla._is_unspecified

def _is_unspecified_or_from_gda_or_auto(x):
  return _is_from_gda(x) or _is_auto(x) or _is_unspecified(x)


def _check_all_or_none_unspecified(axis_resources, name):
  if not axis_resources:
    return False
  unspecified_count = 0
  unspecified = _is_unspecified(axis_resources[0])
  for resource in axis_resources:
    current_is_unspecified = _is_unspecified(resource)
    if current_is_unspecified:
      unspecified_count += 1
      assert unspecified_count == 1
    if current_is_unspecified != unspecified:
      raise ValueError(f'`pjit._UNSPECIFIED` exists in {name}. '
                       f'Make sure that every entry in {name} is '
                       '`pjit._UNSPECIFIED`.')
  return unspecified


# TODO(yashkatariya): Add pjit microbenchmarks.
# in_axis_resources and out_axis_resources can't be None as the default value
# because `None` means that the input is fully replicated.
def pjit(fun: Callable,
         in_axis_resources=_UNSPECIFIED,
         out_axis_resources=_UNSPECIFIED,
         static_argnums: Union[int, Sequence[int]] = (),
         donate_argnums: Union[int, Sequence[int]] = ()) -> stages.Wrapped:
  """Makes ``fun`` compiled and automatically partitioned across multiple devices.

  The returned function has semantics equivalent to those of ``fun``, but is
  compiled to an XLA computation that runs across multiple devices
  (e.g. multiple GPUs or multiple TPU cores). This can be useful if the jitted
  version of ``fun`` would not fit in a single device's memory, or to speed up
  ``fun`` by running each operation in parallel across multiple devices.

  The partitioning over devices happens automatically based on the
  propagation of the input partitioning specified in ``in_axis_resources`` and
  the output partitioning specified in ``out_axis_resources``. The resources
  specified in those two arguments must refer to mesh axes, as defined by
  the :py:func:`jax.experimental.maps.Mesh` context manager. Note that the mesh
  definition at ``pjit`` application time is ignored, and the returned function
  will use the mesh definition available at each call site.

  Inputs to a pjit'd function will be automatically partitioned across devices
  if they're not already correctly partitioned based on ``in_axis_resources``.
  In some scenarios, ensuring that the inputs are already correctly pre-partitioned
  can increase performance. For example, if passing the output of one pjit'd function
  to another pjit’d function (or the same pjit’d function in a loop), make sure the
  relevant ``out_axis_resources`` match the corresponding ``in_axis_resources``.

  .. note::
    **Multi-process platforms:** On multi-process platforms such as TPU pods,
    ``pjit`` can be used to run computations across all available devices across
    processes. To achieve this, ``pjit`` is designed to be used in SPMD Python
    programs, where every process is running the same Python code such that all
    processes run the same pjit'd function in the same order.

    When running in this configuration, the mesh should contain devices across
    all processes. However, any input argument dimensions partitioned over
    multi-process mesh axes should be of size equal to the corresponding *local*
    mesh axis size, and outputs will be similarly sized according to the local
    mesh. ``fun`` will still be executed across *all* devices in the mesh,
    including those from other processes, and will be given a global view of the
    data spread across multiple processes as a single array. However, outside
    of ``pjit`` every process only "sees" its local piece of the input and output,
    corresponding to its local sub-mesh.

    This means that each process's participating local devices must form a
    _contiguous_ local sub-mesh within the full global mesh. A contiguous
    sub-mesh is one where all of its devices are adjacent within the global
    mesh, and form a rectangular prism.

    The SPMD model also requires that the same multi-process ``pjit``'d
    functions must be run in the same order on all processes, but they can be
    interspersed with arbitrary operations running in a single process.

  Args:
    fun: Function to be compiled. Should be a pure function, as side-effects may
      only be executed once. Its arguments and return value should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      Positional arguments indicated by ``static_argnums`` can be anything at
      all, provided they are hashable and have an equality operation defined.
      Static arguments are included as part of a compilation cache key, which is
      why hash and equality operators must be defined.
    in_axis_resources: Pytree of structure matching that of arguments to ``fun``,
      with all actual arguments replaced by resource assignment specifications.
      It is also valid to specify a pytree prefix (e.g. one value in place of a
      whole subtree), in which case the leaves get broadcast to all values in
      that subtree.

      The valid resource assignment specifications are:
        - :py:obj:`None`, in which case the value will be replicated on all devices
        - :py:class:`PartitionSpec`, a tuple of length at most equal to the rank
          of the partitioned value. Each element can be a :py:obj:`None`, a mesh
          axis or a tuple of mesh axes, and specifies the set of resources assigned
          to partition the value's dimension matching its position in the spec.

      The size of every dimension has to be a multiple of the total number of
      resources assigned to it.
    out_axis_resources: Like ``in_axis_resources``, but specifies resource
      assignment for function outputs.
    static_argnums: An optional int or collection of ints that specify which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded in
      Python (during tracing), and so the corresponding argument values can be
      any Python object.

      Static arguments should be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and immutable. Calling the jitted function
      with different values for these constants will trigger recompilation.
      Arguments that are not arrays or containers thereof must be marked as
      static.

      If ``static_argnums`` is not provided, no arguments are treated as static.
    donate_argnums: Specify which argument buffers are "donated" to the computation.
      It is safe to donate argument buffers if you no longer need them once the
      computation has finished. In some cases XLA can make use of donated
      buffers to reduce the amount of memory needed to perform a computation,
      for example recycling one of your input buffers to store a result. You
      should not reuse buffers that you donate to a computation, JAX will raise
      an error if you try to.

      For more details on buffer donation see the [FAQ](https://jax.readthedocs.io/en/latest/faq.html#buffer-donation).
  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation and
    automaticly partitioned by the mesh available at each call site.

  For example, a convolution operator can be automatically partitioned over
  an arbitrary set of devices by a single ``pjit`` application:

  >>> import jax
  >>> import jax.numpy as jnp
  >>> from jax.experimental.maps import Mesh
  >>> from jax.experimental.pjit import PartitionSpec, pjit
  >>>
  >>> x = jnp.arange(8, dtype=jnp.float32)
  >>> f = pjit(lambda x: jax.numpy.convolve(x, jnp.asarray([0.5, 1.0, 0.5]), 'same'),
  ...         in_axis_resources=None, out_axis_resources=PartitionSpec('devices'))
  >>> with Mesh(jax.devices(), ('devices',)):
  ...   print(f(x))  # doctest: +SKIP
  [ 0.5  2.   4.   6.   8.  10.  12.  10. ]
  """
  _check_callable(fun)

  if not config.jax_array and (_is_unspecified(in_axis_resources) or
                               _is_unspecified(out_axis_resources)):
    raise ValueError('in_axis_resources and out_axis_resouces should not '
                     'be the unspecified singleton value.')

  if isinstance(in_axis_resources, list):
    # To be a tree prefix of the positional args tuple, in_axes can never be a
    # list: if in_axes is not a leaf, it must be a tuple of trees. However,
    # in cases like these users expect tuples and lists to be treated
    # essentially interchangeably, so we canonicalize lists to tuples here
    # rather than raising an error. https://github.com/google/jax/issues/2367
    in_axis_resources = tuple(in_axis_resources)

  in_any_auto: bool
  if not config.jax_array:
    in_axis_resources, _, _, in_any_auto = _prepare_axis_resources(
        in_axis_resources, "in_axis_resources")
    out_axis_resources, _, _, _ = _prepare_axis_resources(
        out_axis_resources, "out_axis_resources")
  else:
    if not _is_unspecified(in_axis_resources):
      # `pjit.AUTO` is allowed partially in `in_axis_resources` i.e. you can
      # put sharding instances and `pjit.AUTO` together.
      if not all(isinstance(s, Sharding) or _is_auto(s)
                 for s in tree_flatten(in_axis_resources)[0]):
        raise ValueError('When `config.jax_array` flag is enabled, '
                         'in_axis_resources should contain instances of '
                         '`Sharding` or `pjit.AUTO`.')

    # `out_axis_resources` should be instances of `Sharding` if it's not
    # unspecified. For `AUTO` sharding, it can only be used with
    # MeshPspecSharding.
    if not _is_unspecified(out_axis_resources):
      if not all(isinstance(s, Sharding) or _is_auto(s)
                 for s in tree_flatten(out_axis_resources)[0]):
        raise ValueError('When `config.jax_array` flag is enabled, '
                         'out_axis_resources should contain instances of '
                         '`Sharding` or `pjit.AUTO`.')

  static_argnums = _ensure_index_tuple(static_argnums)
  donate_argnums = _ensure_index_tuple(donate_argnums)
  donate_argnums = rebase_donate_argnums(donate_argnums, static_argnums)

  def infer_params(*args, _global_avals=False, **kwargs):
    if kwargs:
      raise NotImplementedError("pjit does not support kwargs")
    if max(static_argnums + donate_argnums, default=-1) >= len(args):
      raise ValueError(f"jitted function has static_argnums={static_argnums}, "
                       f"donate_argnums={donate_argnums} but "
                       f"was called with only {len(args)} positional arguments.")

    # Putting this outside of wrapped would make resources lexically scoped
    resource_env = pxla.thread_resources.env
    pjit_mesh = resource_env.physical_mesh
    if pjit_mesh.empty:
      raise RuntimeError("pjit requires a non-empty mesh! Are you sure that "
                         "it's defined at the call site?")

    f = lu.wrap_init(fun)
    f, dyn_args = argnums_partial_except(f, static_argnums, args, allow_invalid=False)
    del args

    args_flat, in_tree = tree_flatten(dyn_args)
    flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
    if donate_argnums:
      donated_invars = donation_vector(donate_argnums, dyn_args, ())
    else:
      donated_invars = (False,) * len(args_flat)

    if config.jax_array:
      in_shardings, out_shardings = _get_and_check_in_and_out_shardings(
          args_flat, in_axis_resources, out_axis_resources, pjit_mesh, in_tree)
    else:
      in_shardings = tree_map(lambda x: _create_mesh_pspec_sharding(pjit_mesh, x),
                              in_axis_resources)
      out_shardings = tree_map(lambda x: x if _is_unspecified(x) else
                               _create_mesh_pspec_sharding(pjit_mesh, x), out_axis_resources)
      # This check fails extrememly rarely and has a huge cost in the dispatch
      # path. So hide it behind the jax_enable_checks flag.
      if config.jax_enable_checks:
        _maybe_check_pjit_gda_mesh(args_flat, pjit_mesh)

    if not config.jax_array and in_any_auto and not _global_avals:
      raise ValueError('Auto sharding is only enabled for global inputs. '
                       'Please set `_global_avals=True` during `.lower()`. '
                       'Use the compiled object to call the inputs.')

    local_in_avals = tuple(shaped_abstractify(a) for a in args_flat)
    # TODO(yashkatariya): This is a hack. This should go away when avals have
    # is_global attribute.
    if _global_avals or config.jax_array:
      in_positional_semantics = (maps._PositionalSemantics.GLOBAL,) * len(args_flat)
    else:
      in_positional_semantics = tuple(tree_map(_get_in_positional_semantics, args_flat))
    out_positional_semantics = (
        maps._PositionalSemantics.GLOBAL
        if config.jax_parallel_functions_output_gda or config.jax_array else
        maps._positional_semantics.val)

    global_in_avals, normalized_in_shardings_flat = _process_in_axis_resources(
        hashable_pytree(in_shardings), local_in_avals, in_tree, in_positional_semantics,
        tuple(isinstance(a, GDA) for a in args_flat))

    jaxpr, normalized_out_shardings_flat = _pjit_jaxpr(
        flat_fun, hashable_pytree(out_shardings), global_in_avals,
        HashableFunction(out_tree, closure=()))

    if not config.jax_array:
      normalized_in_shardings_flat = tree_map(
          _maybe_replace_from_gda_with_pspec, normalized_in_shardings_flat,
          tuple(args_flat))

    params = dict(
        jaxpr=jaxpr,
        in_shardings=normalized_in_shardings_flat,
        out_shardings=normalized_out_shardings_flat,
        resource_env=resource_env,
        donated_invars=donated_invars,
        name=getattr(flat_fun, '__name__', '<unnamed function>'),
        in_positional_semantics=in_positional_semantics,
        out_positional_semantics=out_positional_semantics)
    return (args_flat, local_in_avals, params, in_tree, out_tree(),
            donate_argnums)

  @wraps(fun)
  def wrapped(*args, **kwargs):
    args_flat, _, params, _, out_tree, _ = infer_params(*args, **kwargs)
    for arg in args_flat:
      _check_arg(arg)
    out = pjit_p.bind(*args_flat, **params)
    return tree_unflatten(out_tree, out)

  def lower(*args, _global_avals=False, **kwargs):
    (_, flat_local_in_avals, params, in_tree, out_tree,
     donate_argnums) = infer_params(*args, _global_avals=_global_avals, **kwargs)
    in_is_global = _calc_is_global_sequence(
        params['in_positional_semantics'], params['in_shardings'])
    lowering = _pjit_lower(
        params['jaxpr'], params['in_shardings'],
        params['out_shardings'],  params['resource_env'],
        params['donated_invars'], params['name'],
        in_is_global)

    args_kwargs_in_tree = treedef_tuple([in_tree, tree_flatten({})[1]])
    local_in_avals = args_kwargs_in_tree.unflatten(flat_local_in_avals)
    return PjitLowered.from_flat_info(
        lowering, args_kwargs_in_tree, local_in_avals, donate_argnums, out_tree,
        no_kwargs=True)

  wrapped.lower = lower
  return wrapped

class _ListWithW(list):
  __slots__ = ('__weakref__',)

def hashable_pytree(pytree):
  vals, treedef = tree_flatten(pytree)
  vals = tuple(vals)
  return HashableFunction(lambda: tree_unflatten(treedef, vals),
                          closure=(treedef, vals))


def _get_and_check_in_and_out_shardings(args_flat, pjit_in_shardings, out_shardings,
                                        pjit_mesh, in_tree):
  arg_in_shardings_flat = tuple(a.sharding if hasattr(a, 'sharding') else _UNSPECIFIED
                                for a in args_flat)

  if _is_unspecified(pjit_in_shardings):
    # If pjit_in_shardings is unspecified, then arg_in_shardings cannot have
    # unspecified in them.
    for a in arg_in_shardings_flat:
      if _is_unspecified(a):
        raise ValueError('Please specify sharding either on the arg or on '
                         f'pjit. Found sharding {a} which is invalid.')
    in_shardings_flat = arg_in_shardings_flat
  else:
    # This function is cached.
    in_shardings_flat = _get_and_check_pjit_arg_shardings(
        hashable_pytree(pjit_in_shardings), arg_in_shardings_flat,
        in_tree)

  out_shardings_flat = tuple(tree_flatten(out_shardings)[0])
  # Check if the device assignment is the same across inputs and outputs.
  # This function is cached.
  _check_array_device_assignment(pjit_mesh, in_shardings_flat + out_shardings_flat)

  return tree_unflatten(in_tree, in_shardings_flat), out_shardings


@lru_cache(maxsize=4096)
def _create_mesh_pspec_sharding(mesh, x):
  if _is_unspecified_or_from_gda_or_auto(x):
    return x
  return MeshPspecSharding(mesh, x.user_spec, x)


def flatten_axis_resources(what, tree, shardings, tupled_args):
  try:
    return tuple(flatten_axes(what, tree, shardings, tupled_args=tupled_args))
  except ValueError:
    pass  # Raise a tree prefix error below

  # Tree leaves are always valid prefixes, so if there was a prefix error as
  # assumed here, axis_resources must not be a leaf.
  assert not treedef_is_leaf(tree_structure(shardings))

  # Check the type directly rather than using isinstance because of namedtuples.
  if tupled_args and (type(shardings) is not tuple or
                      len(shardings) != len(tree.children())):
    # We know axis_resources is meant to be a tuple corresponding to the args
    # tuple, but while it is a non-leaf pytree, either it wasn't a tuple or it
    # wasn't the right length.
    msg = (f"{what} specification must be a tree prefix of the positional "
           f"arguments tuple passed to the `pjit`-decorated function. In "
           f"particular, {what} must either be a None, a PartitionSpec, or "
           f"a tuple of length equal to the number of positional arguments.")
    # If `tree` represents an args tuple, then `axis_resources` must be a tuple.
    # TODO(mattjj,apaszke): disable implicit list casts, remove 'or list' below
    if type(shardings) is not tuple:
      msg += f" But {what} is not a tuple: got {type(shardings)} instead."
    elif len(shardings) != len(tree.children()):
      msg += (f" But {what} is the wrong length: got a tuple or list of length "
              f"{len(shardings)} for an args tuple of length "
              f"{len(tree.children())}.")

    # As an extra hint, let's check if the user just forgot to wrap
    # shardings in a singleton tuple.
    if len(tree.children()) == 1:
      try: flatten_axes(what, tree, (shardings,))
      except ValueError: pass  # That's not the issue.
      else:
        msg += (f" Given the corresponding argument being "
                f"passed, it looks like {what} might need to be wrapped in "
                f"a singleton tuple.")

    raise ValueError(msg)

  if config.jax_array:
    axis_tree = shardings
  else:
    # Replace axis_resources with unparsed versions to avoid revealing internal details
    axis_tree = tree_map(lambda parsed: parsed.spec, shardings)

  # Because ecause we only have the `tree` treedef and not the full pytree here,
  # we construct a dummy tree to compare against. Revise this in callers?
  dummy_tree = tree_unflatten(tree, [PytreeLeaf()] * tree.num_leaves)
  errors = prefix_errors(axis_tree, dummy_tree)
  if errors:
    e = errors[0]  # Only show information about the first disagreement found.
    raise e(what)

  # At this point we've failed to find a tree prefix error.
  assert False, "Please open a bug report!"  # This should be unreachable.

class PytreeLeaf:
  def __repr__(self): return "pytree leaf"


@lru_cache(maxsize=4096)
def _process_in_axis_resources(in_shardings_thunk, local_in_avals,
                               in_tree, in_positional_semantics, is_gda):
  in_shardings_flat = flatten_axis_resources(
        "pjit in_axis_resources", in_tree, in_shardings_thunk(), tupled_args=True)

  # Fork here because the `Array` path is very simple and doesn't need all the
  # complexity below.
  if config.jax_array:
    for aval, i in safe_zip(local_in_avals, in_shardings_flat):
      if _is_auto(i): continue
      pjit_check_aval_sharding(i, aval, "pjit arguments",
                               allow_uneven_sharding=False)
    global_in_avals = local_in_avals
    return tuple(global_in_avals), tuple(i if _is_auto(i) else i.normalize()
                                         for i in in_shardings_flat)

  if not local_in_avals:
    assert not in_shardings_flat
    return (), ()

  in_axis_resources_flat = tuple(
      i if _is_from_gda(i) or _is_auto(i) else i._parsed_pspec
      for i in in_shardings_flat)

  # This check should be above local_to_global call below otherwise if
  # `FROM_GDA` is passed to any input other than GDA, a ugly error message
  # will be raised because get_array_mapping (in local_to_global) of a
  # FROM_GDA cannot happen.
  tree_map(_check_resources_mismatch, in_axis_resources_flat, is_gda)
  # If all inputs have global semantics or fully replicated, then the avals are
  # global and the mesh should also be global. This split is because
  # non-contiguous mesh can only be used if all inputs have global semantics or
  # fully replicated.
  # Use canonicalized in_axis_resources here because we want to treat P(None)
  # and None (for example) as equivalent.
  if all(
      (not _is_from_gda(p) and not _is_auto(p) and
       CanonicalizedParsedPartitionSpec(p).partitions == ()) or
      ips == maps._PositionalSemantics.GLOBAL
      for p, ips in safe_zip(in_axis_resources_flat, in_positional_semantics)):
    # Shapes should be checked against non canonicalized in_axis_resources.
    # For example, partitions of () and ((),) are not equivalent, since the
    # first one is a valid spec for a scalar value, while the second is not!
    for aval, i in safe_zip(local_in_avals, in_shardings_flat):
      if _is_from_gda(i) or _is_auto(i): continue
      pjit_check_aval_sharding(i, aval, "pjit arguments",
                               allow_uneven_sharding=False)
  else:
    for aval, i in safe_zip(local_in_avals, in_shardings_flat):
      if _is_from_gda(i) or _is_auto(i): continue
      pjit_check_aval_sharding(i, aval, "pjit arguments",
                               allow_uneven_sharding=False, local=True)

  normalized_in_shardings_flat = tuple(
      i if _is_from_gda(i) or _is_auto(i) else i.normalize() for i in in_shardings_flat)

  global_in_avals = local_to_global(in_positional_semantics,
                                    local_in_avals, normalized_in_shardings_flat)
  return tuple(global_in_avals), normalized_in_shardings_flat


@lu.cache
def _pjit_jaxpr(fun, out_shardings_thunk, global_in_avals, out_tree):
  prev_positional_val = maps._positional_semantics.val
  try:
    maps._positional_semantics.val = maps._PositionalSemantics.GLOBAL
    with dispatch.log_elapsed_time(f"Finished tracing + transforming {fun.__name__} "
                                   "for pjit in {elapsed_time} sec"):
      jaxpr, global_out_avals, consts = pe.trace_to_jaxpr_dynamic(fun, global_in_avals)
  finally:
    maps._positional_semantics.val = prev_positional_val
  jaxpr = core.ClosedJaxpr(jaxpr, consts)

  out_shardings_flat = flatten_axis_resources(
      "pjit out_axis_resources", out_tree(), out_shardings_thunk(), tupled_args=False)

  for aval, o in safe_zip(global_out_avals, out_shardings_flat):
    if _is_unspecified(o) or _is_auto(o): continue
    pjit_check_aval_sharding(o, aval, "pjit outputs", allow_uneven_sharding=False)

  normalized_out_shardings_flat = tuple(
      o if _is_unspecified(o) or _is_auto(o) else o.normalize()
      for o in out_shardings_flat
  )

  # lu.cache needs to be able to create weakrefs to outputs, so we can't return a plain tuple
  return _ListWithW([jaxpr, normalized_out_shardings_flat])


# TODO(yashkatariya): Replace this with shape check against sharding which
# uses OpSharding.tile_assignment_dimension.
def pjit_check_aval_sharding(
    sharding: MeshPspecSharding, aval, what_aval: str,
    allow_uneven_sharding: bool, local: bool = False):
  if local:
    m = sharding.mesh.local_mesh
  else:
    m = sharding.mesh
  _check_shapes_against_resources(
      what_aval, m.is_multi_process, m.shape, [aval],
      [sharding._parsed_pspec], allow_uneven_sharding)


class SpecSync(IntEnum):
  """Encodes how much out of sync the real value of partitions is compared to the user specified one.

  We use this to make sure we don't show garbage modified values while claiming
  that the users have specified them like that.
  """
  OUT_OF_SYNC = 0  # Arbitrary changes, including new axes inserted
  DIM_PERMUTE = 1  # Dimensions permuted, but no new sharding axes
  IN_SYNC = 2  # Entirely in sync

class ParsedPartitionSpec:
  __slots__ = ('unsafe_user_spec', 'partitions', 'sync')

  def __init__(self, user_spec, partitions, sync=SpecSync.IN_SYNC):
    self.unsafe_user_spec = user_spec
    # None in partitions represents unconstrained dim.
    # TODO(yashkatariya): May use a sentinel value.
    self.partitions = tuple(partitions)
    self.sync = sync

  @property
  def user_spec(self):
    return self.unsynced_user_spec(SpecSync.IN_SYNC)

  def unsynced_user_spec(self, min_sync):
    if self.sync < min_sync:
      raise AssertionError(f"Please open a bug report! ({self.sync} >= {min_sync})")
    return self.unsafe_user_spec

  def insert_axis_partitions(self, dim, val):
    parts = self.partitions
    too_short = dim - len(parts)
    if too_short > 0:
      parts += ((),) * too_short
    new_partitions = tuple_insert(parts, dim, val)
    new_sync = SpecSync.DIM_PERMUTE if (val == () or val is None) else SpecSync.OUT_OF_SYNC
    return ParsedPartitionSpec(self.unsafe_user_spec, new_partitions, sync=new_sync)

  @classmethod
  def from_user_input(cls, entry, arg_name, allow_unconstrained_dims=False):
    if entry is None:
      return cls(entry, ())
    if not isinstance(entry, PartitionSpec):
      raise TypeError(f"{arg_name} are expected to be "
                      f"PartitionSpec instances or None, but got {entry}")
    axis_specs = []
    for axis_spec in entry:
      if axis_spec is None:
        axis_spec = ()
      elif isinstance(axis_spec, (list, tuple)):
        axis_spec = tuple(axis_spec)
      elif axis_spec == PartitionSpec.UNCONSTRAINED:
        if not allow_unconstrained_dims:
          raise ValueError(f"Unconstrained dims are not allowed: {entry}")
        axis_spec = None
      else:
        axis_spec = (axis_spec,)
      axis_specs.append(axis_spec)
    return cls(entry, axis_specs)

  def __hash__(self):
    return hash((self.partitions, self.sync))

  def __eq__(self, other):
    return (self.partitions == other.partitions and
            self.sync == other.sync)

  def __len__(self):
    return len(self.partitions)

  def __getitem__(self, i):
    return self.partitions[i]

  def __iter__(self):
    return iter(self.partitions)

  def __repr__(self):
    return (f"ParsedPartitionSpec(partitions={self.partitions}, "
            f"unsafe_user_spec={self.unsafe_user_spec}, "
            f"sync={self.sync})")

class CanonicalizedParsedPartitionSpec(ParsedPartitionSpec):
  """ParsedPartitionSpecs that are canonicalized.

  ParsedPartitionSpecs may contain trailing empty tuples, that make them
  semantically different in general, and yet in some situations we prefer
  to regard them as equivalent. For example, partitions of () and ((),)
  cannot be always considered equivalent, since the first one is a valid
  spec for a scalar value, while the second is not! However, when either of
  those are applied to a 2D array, they both mean that the array is fully
  replicated.

  So CanonicalizedParsedPartitionSpecs removes the trailing empty tuples from
  partitions.
  """

  def __init__(self, parsed_pspec: ParsedPartitionSpec):
    partitions = list(parsed_pspec.partitions)
    while partitions and partitions[-1] == ():
      partitions.pop()

    super().__init__(parsed_pspec.unsafe_user_spec, partitions,
                     parsed_pspec.sync)

  def __repr__(self):
    return (f"CanonicalizedParsedPartitionSpec(partitions={self.partitions}, "
            f"unsafe_user_spec={self.unsafe_user_spec}, "
            f"sync={self.sync})")


REPLICATED = CanonicalizedParsedPartitionSpec(ParsedPartitionSpec(None, ()))


def _prepare_axis_resources(axis_resources,
                            arg_name,
                            allow_unconstrained_dims=False):
  # PyTrees don't treat None values as leaves, so we use an is_leaf function.
  entries, treedef = tree_flatten(axis_resources, is_leaf=lambda x: x is None)
  what = f"{arg_name} leaf specifications"
  # All entries should be specified or if unspecified then there should only
  # be 1 entry for that since _UNSPECIFIED is a private API.
  _check_all_or_none_unspecified(entries, arg_name)
  any_auto = pxla._check_if_any_auto(entries)
  entries = [
      (entry if _is_unspecified_or_from_gda_or_auto(entry)
       else ParsedPartitionSpec.from_user_input(
          entry, what, allow_unconstrained_dims=allow_unconstrained_dims))
      for entry in entries
  ]
  _check_unique_resources(entries, arg_name)
  return tree_unflatten(treedef, entries), entries, treedef, any_auto


def _check_resources_mismatch(in_axis_resources_flat, is_gda):
  if not is_gda and _is_from_gda(in_axis_resources_flat):
    raise ValueError('For a non-GDA input, the corresponding resource in '
                     'in_axis_resources cannot be `pjit.FROM_GDA`.')

def _check_unique_resources(axis_resources, arg_name):
  for arg_axis_resources in axis_resources:
    if not arg_axis_resources: continue
    if _is_unspecified_or_from_gda_or_auto(arg_axis_resources): continue
    constrained_dims = [d for d in arg_axis_resources if d is not None]
    resource_counts = Counter(it.chain.from_iterable(constrained_dims))
    if not resource_counts: continue
    if resource_counts.most_common(1)[0][1] > 1:
      multiple_uses = [r for r, c in resource_counts.items() if c > 1]
      if multiple_uses:
        raise ValueError(f"A single {arg_name} specification can map every mesh axis "
                         f"to at most one positional dimension, but {arg_axis_resources.user_spec} "
                         f"has duplicate entries for {pxla.show_axes(multiple_uses)}")

def _check_shapes_against_resources(what: str, is_global_shape: bool,
                                    mesh_shape, flat_avals, flat_axis_resources,
                                    allow_uneven_sharding: bool):
  global_str = " global" if is_global_shape else ""
  for aval, aval_axis_resources in zip(flat_avals, flat_axis_resources):
    if _is_unspecified_or_from_gda_or_auto(aval_axis_resources):
      continue
    shape = aval.shape
    if len(shape) < len(aval_axis_resources):
      raise ValueError(f"One of {what} was given the resource assignment "
                       f"of {aval_axis_resources.user_spec}, which implies that "
                       f"it has a rank of at least {len(aval_axis_resources)}, "
                       f"but it is {len(shape)}")
    for i, axis_resources in enumerate(aval_axis_resources):
      if axis_resources is None:
        continue
      try:
        size = int(np.prod([mesh_shape[resource] for resource in axis_resources], dtype=np.int64))
      except KeyError as e:
        raise ValueError(f"One of {what} was given the resource assignment "
                         f"of {aval_axis_resources.user_spec}, but resource axis "
                         f"{e.args[0]} is undefined. Did you forget to declare the mesh?") from None
      if not allow_uneven_sharding and shape[i] % size != 0:
        raise ValueError(f"One of {what} was given the resource assignment "
                         f"of {aval_axis_resources.user_spec}, which implies that "
                         f"the{global_str} size of its dimension {i} should be "
                         f"divisible by {size}, but it is equal to {shape[i]}")

# -------------------- pjit rules --------------------

pjit_p = core.Primitive("pjit")
pjit_p.multiple_results = True


def _pjit_call_impl(*args, jaxpr,
                    in_shardings, out_shardings, resource_env,
                    donated_invars, name,
                    in_positional_semantics, out_positional_semantics):
  in_is_global = _calc_is_global_sequence(in_positional_semantics, in_shardings)
  if config.jax_array and all(_is_unspecified(o) for o in out_shardings):
    _allow_propagation_to_outputs = True
  else:
    _allow_propagation_to_outputs = False
  compiled = _pjit_lower(
      jaxpr, in_shardings, out_shardings, resource_env,
      donated_invars, name, in_is_global).compile(
          _allow_propagation_to_outputs=_allow_propagation_to_outputs)
  # This check is expensive so only do it if enable_checks is on.
  if compiled._auto_spmd_lowering and config.jax_enable_checks:
    pxla._check_gda_or_array_xla_sharding_match(args, compiled._in_shardings)
  if config.jax_distributed_debug:
    # Defensively only perform fingerprint logic if debug logging is enabled
    # NOTE(skyewm): I didn't benchmark this
    fingerprint = None
    if hasattr(compiled.runtime_executable(), "fingerprint"):
      fingerprint = compiled.runtime_executable().fingerprint
    if fingerprint is not None:
      fingerprint = fingerprint.hex()
    distributed_debug_log(("Running pjit'd function", name),
                          ("in_shardings", in_shardings),
                          ("out_shardings", out_shardings),
                          ("abstract args", list(map(xla.abstractify, args))),
                          ("fingerprint", fingerprint))
  return compiled.unsafe_call(*args)
pjit_p.def_impl(_pjit_call_impl)

@weakref_lru_cache
def _pjit_lower(
    jaxpr: core.ClosedJaxpr,
    in_shardings,
    out_shardings,
    resource_env,
    donated_invars,
    name: str,
    in_is_global: Sequence[bool]):
  pxla.resource_typecheck(jaxpr, resource_env, {}, lambda: "pjit")
  f = core.jaxpr_as_fun(jaxpr)
  f.__name__ = name
  fun = lu.wrap_init(f)
  # TODO(yashkatariya): UNSPECIFIED should go through lower_sharding_computation.
  # Also the `jaxpr_has_primitive` for xmap is temporary until xmap supports
  # sharding instances.
  # For `pjit(xmap)` cases, it needs to take the `lower_mesh_computation` path
  # because `xmap` only supports SPMDAxisContext right now.
  if (pxla._check_if_any_auto_or_unspecified(in_shardings + out_shardings) or
      dispatch.jaxpr_has_primitive(jaxpr.jaxpr, 'xmap')):
    return pxla.lower_mesh_computation(
      fun, 'pjit', name, resource_env.physical_mesh,
      in_shardings, out_shardings, donated_invars,
      True, jaxpr.in_avals, tiling_method=None, in_is_global=in_is_global)
  else:
    # Pass `in_is_global` here because this path is taken by both host local
    # avals and global avals.
    return pxla.lower_sharding_computation(
        fun, 'pjit', name, in_shardings, out_shardings, donated_invars,
        jaxpr.in_avals, in_is_global=in_is_global)


def _pjit_abstract_eval(*args, jaxpr, out_shardings, resource_env,
                        out_positional_semantics, **_):
  if jaxpr.effects:
    raise NotImplementedError('Effects not supported in `pjit`.')
  return global_to_local(out_positional_semantics, jaxpr.out_avals,
                         out_shardings), jaxpr.effects
pjit_p.def_effectful_abstract_eval(_pjit_abstract_eval)


def _pjit_lowering(ctx, *args, name, jaxpr, in_shardings,
                   out_shardings, resource_env, donated_invars,
                   in_positional_semantics, out_positional_semantics):
  if not isinstance(ctx.module_context.axis_context,
                    (mlir.SPMDAxisContext, mlir.ShardingContext)):
    raise RuntimeError("Nesting pjit() inside jit() is not allowed.")

  output_types = safe_map(mlir.aval_to_ir_types, ctx.avals_out)
  flat_output_types = util.flatten(output_types)

  arg_shardings = [i._to_xla_op_sharding(aval.ndim)
                   for aval, i in safe_zip(ctx.avals_in, in_shardings)]
  result_shardings = [o._to_xla_op_sharding(aval.ndim)
                      for aval, o in safe_zip(ctx.avals_out, out_shardings)]

  sub_ctx = ctx.module_context.replace(
      name_stack=xla.extend_name_stack(ctx.module_context.name_stack,
                                       wrap_name(name, "pjit")))
  # TODO(b/228598865): inlined calls cannot have shardings set directly on the
  # inputs or outputs because they are lost during MHLO->HLO conversion.
  # using_sharding_annotation=False means we add an identity operation instead.
  func = mlir.lower_jaxpr_to_fun(sub_ctx, f"pjit_{name}", jaxpr, (),
                                 arg_shardings=arg_shardings,
                                 result_shardings=result_shardings,
                                 use_sharding_annotations=False)
  call = func_dialect.CallOp(flat_output_types,
                             ir.FlatSymbolRefAttr.get(func.name.value),
                             mlir.flatten_lowering_ir_args(args))
  return util.unflatten(call.results, safe_map(len, output_types))

mlir.register_lowering(pjit_p, _pjit_lowering)


def _pjit_batcher(insert_axis,
                  axis_size, axis_name, main_type,
                  vals_in, dims_in,
                  jaxpr, in_shardings, out_shardings,
                  resource_env, donated_invars, name, in_positional_semantics,
                  out_positional_semantics):
  # batch_jaxpr expects all batching dimensions to be equal to 0
  vals_in = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
             else x for x, d in zip(vals_in, dims_in)]
  is_mapped_in = [d is not batching.not_mapped for d in dims_in]
  new_jaxpr, is_mapped_out = batching.batch_jaxpr(
      jaxpr, axis_size, is_mapped_in,
      instantiate=False, axis_name=axis_name, main_type=main_type)

  new_parts = (axis_name,) if insert_axis else ()
  in_shardings = tuple(
      _pjit_batcher_for_sharding(i, 0, new_parts) if is_mapped else i
      for is_mapped, i in zip(is_mapped_in, in_shardings))
  out_shardings = tuple(
      _pjit_batcher_for_sharding(o, 0, new_parts) if is_mapped else o
      for is_mapped, o in zip(is_mapped_out, out_shardings))
  vals_out = pjit_p.bind(
    *vals_in,
    jaxpr=new_jaxpr,
    in_shardings=in_shardings,
    out_shardings=out_shardings,
    resource_env=resource_env,
    donated_invars=donated_invars,
    name=name,
    in_positional_semantics=in_positional_semantics,
    out_positional_semantics=out_positional_semantics)
  dims_out = [0 if batched else batching.not_mapped for batched in is_mapped_out]
  return vals_out, dims_out
batching.axis_primitive_batchers[pjit_p] = partial(_pjit_batcher, False)
pxla.spmd_primitive_batchers[pjit_p] = partial(_pjit_batcher, True)

# TODO(yashkatariya, apaszke): Remove this and replace this with `VmapSharding`.
def _pjit_batcher_for_sharding(s, dim, val):
  parsed_pspec = s._parsed_pspec.insert_axis_partitions(dim, val)
  # Use `_from_parsed_pspec` because after you `insert_axis_partitions` the
  # `sync` attribute changes. To make sure we preserve that, we need to pass
  # that parsed partition spec when created the sharding instance.
  # Inferring the `PartitiionSpec` from that is easy as done in the classmethod.
  return MeshPspecSharding._from_parsed_pspec(s.mesh, parsed_pspec).normalize()


def _pjit_jvp(primals_in, tangents_in,
              jaxpr, in_shardings, out_shardings,
              resource_env, donated_invars, name, in_positional_semantics,
              out_positional_semantics):
  is_nz_tangents_in = [type(t) is not ad.Zero for t in tangents_in]
  jaxpr_jvp, is_nz_tangents_out = ad.jvp_jaxpr(
      jaxpr, is_nz_tangents_in, instantiate=False)

  def _filter_zeros(is_nz_l, l):
    return (x for nz, x in zip(is_nz_l, l) if nz)
  _filter_zeros_in = partial(_filter_zeros, is_nz_tangents_in)
  _filter_zeros_out = partial(_filter_zeros, is_nz_tangents_out)
  outputs = pjit_p.bind(
      *primals_in, *_filter_zeros_in(tangents_in),
      jaxpr=jaxpr_jvp,
      in_shardings=(*in_shardings, *_filter_zeros_in(in_shardings)),
      out_shardings=(*out_shardings, *_filter_zeros_out(out_shardings)),
      resource_env=resource_env,
      donated_invars=(*donated_invars, *_filter_zeros_in(donated_invars)),
      name=wrap_name(name, 'jvp'),
      in_positional_semantics=(*in_positional_semantics, *_filter_zeros_in(in_positional_semantics)),
      out_positional_semantics=out_positional_semantics)

  primals_out, tangents_out = split_list(outputs, [len(jaxpr.jaxpr.outvars)])
  assert len(primals_out) == len(jaxpr.jaxpr.outvars)
  tangents_out_it = iter(tangents_out)
  return primals_out, [next(tangents_out_it) if nz else ad.Zero(aval)
                       for nz, aval in zip(is_nz_tangents_out, jaxpr.out_avals)]
ad.primitive_jvps[pjit_p] = _pjit_jvp


def _pjit_partial_eval(trace, *in_tracers,
                       jaxpr, in_shardings, out_shardings,
                       resource_env, donated_invars, name, in_positional_semantics,
                       out_positional_semantics):
  mesh = resource_env.physical_mesh
  in_pvals = [t.pval for t in in_tracers]

  known_ins = tuple(pv.is_known() for pv in in_pvals)
  unknown_ins = tuple(not k for k in known_ins)
  known_jaxpr, unknown_jaxpr, unknown_outs, res_avals = pe.partial_eval_jaxpr_nounits(
      jaxpr, unknown_ins, instantiate=False)
  unknown_outs = tuple(unknown_outs)
  known_outs = tuple(not uk for uk in unknown_outs)
  num_residuals = len(res_avals)

  def keep_where(l, should_keep):
    return tuple(x for x, keep in zip(l, should_keep) if keep)

  # Compute the known outputs
  known_params = dict(
      jaxpr=known_jaxpr,
      in_shardings=keep_where(in_shardings, known_ins),
      # TODO(yashkatariya): Remove the `MeshPspecSharding` creation here.
      # This is done like this because all output
      # shardings are that even in the `Array` codepath.
      out_shardings=(
          keep_where(out_shardings, known_outs) +
          (MeshPspecSharding(mesh, pxla.PartitionSpec(None)),) * num_residuals),
      resource_env=resource_env,
      donated_invars=keep_where(donated_invars, known_ins),
      name=name,
      in_positional_semantics=keep_where(in_positional_semantics, known_ins),
      out_positional_semantics=out_positional_semantics)

  if num_residuals:
    in_is_global = _calc_is_global_sequence(
        known_params['in_positional_semantics'], known_params['in_shardings'])
    compiled = _pjit_lower(
        known_params["jaxpr"], known_params["in_shardings"],
        known_params["out_shardings"], known_params["resource_env"],
        known_params["donated_invars"], known_params["name"],
        in_is_global).compile(_allow_propagation_to_outputs=True,
                              _allow_compile_replicated=False)
    _, output_ppspec = _get_ppspec_from_executable(compiled.xla_executable, mesh)
    residual_specs = tuple(output_ppspec[-num_residuals:])
  else:
    residual_specs = ()
  residual_sharding = tuple(MeshPspecSharding._from_parsed_pspec(mesh, r)
                            for r in residual_specs)
  known_params['out_shardings'] = (
      keep_where(out_shardings, known_outs) + residual_sharding)

  all_known_outs = pjit_p.bind(
      *(pv.get_known() for pv in in_pvals if pv.is_known()),
      **known_params)
  if num_residuals:
    known_out_vals, residual_vals = \
        split_list(all_known_outs, [len(all_known_outs) - num_residuals])
  else:
    known_out_vals, residual_vals = all_known_outs, ()
  residual_tracers = [trace.new_instantiated_const(residual) for residual in residual_vals]

  # The convention of partial_eval_jaxpr_nounits is to place residual binders
  # at the front of the jaxpr produced, so we move them to the back since both
  # the jaxpr equation built below and the pjit transpose rule assume a
  # residual-inputs-last convention.
  unknown_jaxpr = pe.move_binders_to_back(
      unknown_jaxpr, [True] * num_residuals + [False] * sum(unknown_ins))
  # Prepare unknown tracers
  unknown_params = dict(
      jaxpr=unknown_jaxpr,
      in_shardings=(keep_where(in_shardings, unknown_ins) + residual_sharding),
      out_shardings=keep_where(out_shardings, unknown_outs),
      resource_env=resource_env,
      donated_invars=(keep_where(donated_invars, unknown_ins) +
                      (False,) * num_residuals),
      name=name,
      in_positional_semantics=(keep_where(
          in_positional_semantics, unknown_ins) + (out_positional_semantics,) * num_residuals),
      out_positional_semantics=out_positional_semantics)
  unknown_tracers_in = [t for t in in_tracers if not t.pval.is_known()]
  unknown_tracers_out = [
      pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
      for aval in global_to_local(unknown_params["out_positional_semantics"],
                                  unknown_jaxpr.out_avals,
                                  unknown_params["out_shardings"])
  ]
  eqn = pe.new_eqn_recipe((*unknown_tracers_in, *residual_tracers),
                          unknown_tracers_out,
                          pjit_p,
                          unknown_params,
                          unknown_jaxpr.effects,
                          source_info_util.current())
  for t in unknown_tracers_out: t.recipe = eqn
  return merge_lists(unknown_outs, known_out_vals, unknown_tracers_out)
pe.custom_partial_eval_rules[pjit_p] = _pjit_partial_eval


def _pjit_transpose(reduce_axes, cts_in, *primals_in,
                    jaxpr, in_shardings, out_shardings,
                    resource_env, donated_invars, name, in_positional_semantics,
                    out_positional_semantics):
  def prune_type(ty, xs, maybe_zeros):
    return tuple(x for x, mz in zip(xs, maybe_zeros) if type(mz) is not ty)

  body = lu.wrap_init(ad.closed_backward_pass)
  body = lu.hashable_partial(body, jaxpr, reduce_axes, False)
  primals_and_nz_cts_in, in_treedef = tree_flatten((primals_in, cts_in))
  body, cts_out_treedef_thunk = flatten_fun_nokwargs(body, in_treedef)

  transpose_in_shardings = (
    *prune_type(ad.UndefinedPrimal, in_shardings, primals_in),
    *prune_type(ad.Zero, out_shardings, cts_in)
  )
  transpose_in_positional_semantics = (
    *prune_type(ad.UndefinedPrimal, in_positional_semantics, primals_in),
    *prune_type(ad.Zero, (out_positional_semantics,) * len(cts_in), cts_in)
  )
  global_cts_in_avals = [core.raise_to_shaped(core.get_aval(ct))
                         for ct in primals_and_nz_cts_in]
  if not config.jax_array:
    global_cts_in_avals = local_to_global(
        transpose_in_positional_semantics, global_cts_in_avals,
        transpose_in_shardings)

  transpose_jaxpr, global_cts_out_avals, consts = pe.trace_to_jaxpr_dynamic(
      body, global_cts_in_avals)
  # TODO(apaszke): Creating ClosedJaxpr by hand will break compilation cache!
  transpose_jaxpr = core.ClosedJaxpr(transpose_jaxpr, consts)
  del consts
  cts_out_treedef = cts_out_treedef_thunk()
  transpose_out_shardings = prune_type(
      ad.Zero,
      in_shardings,
      tree_unflatten(cts_out_treedef, [object()] * cts_out_treedef.num_leaves))

  nz_cts_out = pjit_p.bind(
      *primals_and_nz_cts_in,
      jaxpr=transpose_jaxpr,
      in_shardings=transpose_in_shardings,
      out_shardings=transpose_out_shardings,
      resource_env=resource_env,
      donated_invars=(False,) * len(primals_and_nz_cts_in),
      name=name,
      in_positional_semantics=transpose_in_positional_semantics,
      out_positional_semantics=out_positional_semantics)
  return tree_unflatten(cts_out_treedef, nz_cts_out)
ad.reducing_transposes[pjit_p] = _pjit_transpose


def _check_resources_against_named_axes(what, aval, pos_axis_resources, named_axis_resources):
  pjit_resources = set(
      it.chain.from_iterable([d for d in pos_axis_resources if d is not None]))
  aval_resources = set(it.chain.from_iterable(
    named_axis_resources[a] for a in aval.named_shape))
  overlap = pjit_resources & aval_resources
  if overlap:
    raise JAXTypeError(
        f"{what} has an axis resources specification of "
        f"{pos_axis_resources.unsynced_user_spec(SpecSync.DIM_PERMUTE)} "
        f"that uses one or more mesh axes already used by xmap to partition "
        f"a named axis appearing in its named_shape (both use mesh axes "
        f"{pxla.show_axes(overlap)})")

def _resource_typing_pjit(avals, params, source_info, resource_env, named_axis_resources):
  jaxpr = params["jaxpr"]
  what = "pjit input"
  if resource_env.physical_mesh != params['resource_env'].physical_mesh:
    raise RuntimeError("Changing the physical mesh is not allowed inside pjit.")
  for aval, s in zip(jaxpr.in_avals, params['in_shardings']):
    _check_resources_against_named_axes(what, aval, s._parsed_pspec, named_axis_resources)
  pxla.resource_typecheck(
      jaxpr.jaxpr, resource_env, named_axis_resources,
      lambda: (f"a pjit'ed function {params['name']} "
               f"(pjit called at {source_info_util.summarize(source_info)})"))
  what = "pjit output"
  for aval, s in zip(jaxpr.out_avals, params['out_shardings']):
    _check_resources_against_named_axes(what, aval, s._parsed_pspec, named_axis_resources)
pxla.custom_resource_typing_rules[pjit_p] = _resource_typing_pjit


# -------------------- with_sharding_constraint --------------------

def with_sharding_constraint(x, axis_resources):
  x_flat, tree = tree_flatten(x)
  parsed_axis_resources, _, _, _ = _prepare_axis_resources(
      axis_resources, "axis_resources", allow_unconstrained_dims=True)
  axis_resources_flat = tuple(
      flatten_axes("with_sharding_constraint axis_resources",
                   tree, parsed_axis_resources))
  resource_env = pxla.thread_resources.env
  mesh = resource_env.physical_mesh

  sharding_flat = [MeshPspecSharding._from_parsed_pspec(mesh, a)
                   for a in axis_resources_flat]

  for xf, i in safe_zip(x_flat, sharding_flat):
    pjit_check_aval_sharding(i, xf, "with_sharding_constraint arguments",
                             allow_uneven_sharding=True)

  outs = [sharding_constraint_p.bind(xf, sharding=i.normalize(),
                                     resource_env=resource_env)
          for xf, i in safe_zip(x_flat, sharding_flat)]
  return tree_unflatten(tree, outs)

def _sharding_constraint_impl(x, sharding, resource_env):
  # TODO(skye): can we also prevent this from being called in other
  # non-pjit contexts? (e.g. pmap, control flow)
  raise NotImplementedError(
      "with_sharding_constraint() should only be called inside pjit()")

sharding_constraint_p = core.Primitive("sharding_constraint")
sharding_constraint_p.def_impl(_sharding_constraint_impl)
sharding_constraint_p.def_abstract_eval(lambda x, **_: x)
ad.deflinear2(sharding_constraint_p,
              lambda ct, _, sharding, resource_env: (
                  sharding_constraint_p.bind(
                      ct, sharding=sharding,
                      resource_env=resource_env),))

def _sharding_constraint_mhlo_lowering(ctx, x_node, *, sharding,
                                       resource_env):
  aval, = ctx.avals_in
  assert isinstance(sharding, MeshPspecSharding)
  return [
      mlir.wrap_with_sharding_op(
          x_node,
          sharding._to_xla_op_sharding(
              aval.ndim, axis_ctx=ctx.module_context.axis_context),
          unspecified_dims=get_unconstrained_dims(sharding))
  ]
mlir.register_lowering(sharding_constraint_p,
                       _sharding_constraint_mhlo_lowering)


def _sharding_constraint_batcher(insert_axis, axis_size, axis_name, main_type,
                                 vals_in, dims_in, sharding, resource_env):
  x, = vals_in
  d, = dims_in
  # None means unconstrained in ParsedPartitionSpec
  new_parts = (axis_name,) if insert_axis else None
  y = sharding_constraint_p.bind(
      x,
      sharding=_pjit_batcher_for_sharding(sharding, d, new_parts),
      resource_env=resource_env)
  return y, d
batching.axis_primitive_batchers[sharding_constraint_p] = partial(_sharding_constraint_batcher, False)
pxla.spmd_primitive_batchers[sharding_constraint_p] = partial(_sharding_constraint_batcher, True)

def _resource_typing_sharding_constraint(avals, params, source_info,
                                         resource_env, named_axis_resources):
  aval, = avals
  _check_resources_against_named_axes(
    "with_sharding_constraint input", aval,
    params['sharding']._parsed_pspec, named_axis_resources)
pxla.custom_resource_typing_rules[sharding_constraint_p] = \
    _resource_typing_sharding_constraint

# -------------------- helpers --------------------

def get_array_mapping(
    axis_resources: Union[ParsedPartitionSpec, _AUTOAxisResource, _UnspecifiedValue]
) -> pxla.ArrayMappingOrAutoOrUnspecified:
  # TODO(yashkatariya): Use `TypeGuard` on `_is_auto` when it is supported.
  # Don't use `_is_auto` here to satisfy pytype and mypy.
  if isinstance(axis_resources, (_AUTOAxisResource, _UnspecifiedValue)):
    return axis_resources
  return OrderedDict((axis, i)
                     for i, axes in enumerate(axis_resources)
                     if axes is not None for axis in axes)


def get_unconstrained_dims(sharding: MeshPspecSharding):
  return {i for i, axes in enumerate(sharding._parsed_pspec)
          if axes is None}


def global_to_local(positional_semantics, avals, shardings):
  if config.jax_array:
    return avals
  if isinstance(positional_semantics, maps._PositionalSemantics):
    positional_semantics = [positional_semantics] * len(shardings)
  return [
      aval if ps == maps._PositionalSemantics.GLOBAL or
      s._parsed_pspec.partitions == () else s.mesh._global_to_local(
          get_array_mapping(s._parsed_pspec), aval)
      for aval, s, ps in safe_zip(avals, shardings, positional_semantics)
  ]

def local_to_global(positional_semantics, avals, shardings):
  if config.jax_array:
    return avals
  return [
      aval if ps == maps._PositionalSemantics.GLOBAL or
      s._parsed_pspec.partitions == () else s.mesh._local_to_global(
          get_array_mapping(s._parsed_pspec), aval)
      for aval, s, ps in safe_zip(avals, shardings, positional_semantics)
  ]


def _calc_is_global_sequence(in_positional_semantics, in_shardings):
  if config.jax_array:
    return (True,) * len(in_positional_semantics)
  return tuple(
      ips == maps._PositionalSemantics.GLOBAL or i._parsed_pspec.partitions == ()
      for ips, i in safe_zip(in_positional_semantics, in_shardings))

def _get_in_positional_semantics(arg) -> maps._PositionalSemantics:
  if isinstance(arg, GDA):
    return maps._PositionalSemantics.GLOBAL
  return maps._positional_semantics.val


def _maybe_replace_from_gda_with_pspec(
    in_sharding_flat, arg) -> MeshPspecSharding:
  if isinstance(arg, GDA):
    if _is_auto(in_sharding_flat):
      return in_sharding_flat
    gda_cpspec = CanonicalizedParsedPartitionSpec(
        ParsedPartitionSpec.from_user_input(arg.mesh_axes, arg_name="GDA spec"))
    if (not _is_from_gda(in_sharding_flat) and
        in_sharding_flat._parsed_pspec != gda_cpspec):
      raise ValueError(
          f"Got an input GDA to pjit with different partitioning than specified in "
          "the in_axis_resources argument to pjit. The partitioning must match, or "
          "use `jax.experimental.pjit.FROM_GDA` in `in_axis_resources` for GDA. "
          f"Got GDA spec: {gda_cpspec.user_spec} and "
          f"pjit spec: {in_sharding_flat.spec} for GDA: {arg}")
    # This is an optimization to return the original sharding if its not
    # FROM_GDA.
    if not _is_from_gda(in_sharding_flat):
      return in_sharding_flat
    return MeshPspecSharding(arg.mesh, arg.mesh_axes).normalize()
  return in_sharding_flat


@lru_cache(maxsize=4096)
def _check_array_device_assignment(pjit_mesh, shardings):
  if not shardings:
    return
  first_device_assignment = None
  mesh_devices = list(pjit_mesh.devices.flat)
  for i in shardings:
    if _is_auto(i) or _is_unspecified(i):
      continue
    # Assign `first_device_assignment` after `AUTO` and `UNSPECIFIED` have been
    # skipped.
    if first_device_assignment is None:
      first_device_assignment = i._device_assignment()
    arr_device_assignment = i._device_assignment()
    if pjit_mesh.empty:
      # If mesh is empty, then check if all devices across shardings are
      # equal
      if first_device_assignment != arr_device_assignment:
        raise ValueError("Devices of all `Array` inputs and outputs should be "
                         "the same. "
                         f"Got array devices: {first_device_assignment},\n "
                         f"another array devices: {arr_device_assignment}")
    else:
      # If mesh is not empty, then check devices of all shardings against the
      # mesh devices.
      if mesh_devices != arr_device_assignment:
        raise ValueError("Pjit's devices and Array's devices should be equal. "
                         f"Got Pjit devices: {list(pjit_mesh.devices.flat)},\n "
                         f"Array devices: {arr_device_assignment}")

@lru_cache(maxsize=4096)
def _get_and_check_pjit_arg_shardings(pjit_in_shardings, arg_in_shardings_flat,
                                      in_tree):
  pjit_in_shardings_flat = flatten_axis_resources(
      "pjit in_shardings", in_tree, pjit_in_shardings(), tupled_args=True)

  out = []
  for pjit_sharding, arg_sharding in safe_zip(
      pjit_in_shardings_flat, arg_in_shardings_flat):
    # If the sharding of the arg is not known, replace it with the sharding on
    # pjit.
    if _is_unspecified(arg_sharding):
      out.append(pjit_sharding)
    elif _is_auto(pjit_sharding):
      raise ValueError('Passing sharding on pjit and on args while using the '
                       'auto spmd partitioner is not allowed. Please call the '
                       'compiled object on the inputs.')
    else:
      # TODO(yashkatariya): Use opsharding proto to compare equality after
      # opsharding proto supports equality checks.
      if pjit_sharding.normalize() != arg_sharding.normalize():
        raise ValueError('Sharding passed to pjit does not match the sharding '
                         'on the respective arg. '
                         f'Got pjit sharding: {pjit_sharding},\n'
                         f'arg sharding: {arg_sharding}')
      out.append(pjit_sharding)

  assert not any(_is_unspecified(o) for o in out)
  return tuple(out)


def _maybe_check_pjit_gda_mesh(args, mesh):
  for x in args:
    if isinstance(x, GDA) and x.mesh != mesh:
      raise ValueError("Pjit's mesh and GDA's mesh should be equal. Got Pjit "
                       f"mesh: {mesh},\n GDA mesh: {x.mesh}")

# -------------------- XLA OpSharding to PartitionSpec --------------------
# Note that OpSharding is more expressive than PartitionSpecs, so it's not
# always possible to convert them, but the code below should at least
# support handle all cases when this is possible.

def strides_for_sizes(sizes):
  """Returns an array of strides for major-to-minor sizes."""
  return np.cumprod(sizes[::-1])[::-1] // np.asarray(sizes)

def unflatten_array(named_sizes, assignment):
  """Recovers the ordering of axis names based on a device assignment.

  The device assignments that this function can convert into axis orders
  are of the form::

    np.arange(np.prod(named_sizes.values())).transpose(...).flatten()

  for some transposition ``...``. This is satisfied by all OpSharding assignments
  generated from partition specs.

  Arguments:
    named_sizes: A dictionary mapping axis names to their sizes.
    assignment: A permutation of integers between 0 and the product of all
      named sizes.

  Returns:
    A major-to-minor list of axis names that corresponds to the given assignment.
  """
  named_sizes = {name: size for name, size in named_sizes.items() if size != 1}
  sizes = np.fromiter(named_sizes.values(), dtype=np.int64)
  strides = strides_for_sizes(sizes)
  dims = explode_superdims(sizes, unflatten_superdims(assignment))
  dim_to_name = {(size, stride): name for size, stride, name in zip(sizes, strides, named_sizes)}
  return [dim_to_name[d] for d in dims]

def unflatten_superdims(assignment):
  """Unflatten a list of dimension sizes and their strides that generates assignment.

  If this function succeeds for a given ``assignment``, then the following property
  should be satisfied::

    dims_with_strides = unflatten_superdims(assignment)
    base_array = np.arange(map(fst, sorted(dims_with_strides, key=snd, reverse=True)))
    assignment == base_array.transpose(argsort(dims_with_strides, key=snd, reverse=True)).flatten()

  That is, the returned dimensions list all sizes of the base array (with strides
  indicating their initial order). The order of dimensions in the list corresponds
  to the permutation that applied to the base array generates the assignment.
  """
  def check(cond):
    if cond: return
    raise NotImplementedError("Failed to convert OpSharding into a ShardingSpec. "
                              "Please open a bug report!")
  flat_assignment = np.asarray(assignment, dtype=np.int64)
  check(flat_assignment[0] == 0)
  dims = []
  while flat_assignment.size > 1:
    stride = flat_assignment[1]
    for i in range(len(flat_assignment)):
      if flat_assignment[i] != i * stride: break
    else:
      # After this loop i should point to an "element after the sequence", so
      # we have to increment it if the whole array is a strided sequence.
      i += 1
    size = i
    dims.append((size, stride))
    assert size > 1  # Ensure progress
    flat_assignment = flat_assignment[::size]
  return dims

def explode_superdims(sizes, dims):
  """Explode superdims to fit a known shape.

  The unflattening process might mistakenly generate too few too large dimensions.
  For example, ``unflatten_superdims(np.arange(n))`` always returns ``[(n, 1)]``.
  This function takes a list of such contiguous super-dimensions and splits them
  into smaller dimensions such that::

    set(map(fst, explode_superdims(sizes, dims))) == set(sizes)
  """
  strides_to_sizes = {stride: size for size, stride in zip(sizes, strides_for_sizes(sizes))}
  dims = list(reversed(dims))
  final_dims = []
  for size, stride in dims:
    target_size = strides_to_sizes[stride]
    new_dims = []
    while size > target_size:
      assert target_size > 1  # Ensure progress
      assert size % target_size == 0
      new_dims.append((target_size, stride))
      size //= target_size
      stride *= target_size
      target_size = strides_to_sizes[stride]
    assert size == target_size
    new_dims.append((size, stride))
    final_dims += reversed(new_dims)
  return final_dims

def parse_flatten_op_sharding(op_sharding: xc.OpSharding,
                              mesh: pxla.Mesh) -> Sequence[ParsedPartitionSpec]:
  if op_sharding.type == xc.OpSharding.Type.TUPLE:
    out: List[ParsedPartitionSpec] = []
    for s in op_sharding.tuple_shardings:
      out.extend(parse_flatten_op_sharding(s, mesh))
    return out
  elif op_sharding.type == xc.OpSharding.Type.REPLICATED:
    return [REPLICATED]
  elif op_sharding.type == xc.OpSharding.Type.OTHER:
    mesh_shape = mesh.shape
    mesh_axis_order = unflatten_array(mesh.shape, op_sharding.tile_assignment_devices)
    mesh_axis = iter(mesh_axis_order)
    shape = op_sharding.tile_assignment_dimensions
    partitions = []
    for dim_size in shape:
      dim_partitions = []
      while dim_size > 1:
        axis = next(mesh_axis)
        axis_size = mesh_shape[axis]
        assert dim_size % axis_size == 0
        dim_size //= axis_size
        dim_partitions.append(axis)
      partitions.append(tuple(dim_partitions))
    if op_sharding.last_tile_dims == [xc.OpSharding.Type.REPLICATED]:
      replicate_on_last_tile_dim = True
    else:
      replicate_on_last_tile_dim = op_sharding.replicate_on_last_tile_dim
      if op_sharding.last_tile_dims:
        raise NotImplementedError("Unhandled OpSharding type. Please open a bug report!")
    if replicate_on_last_tile_dim:
      partitions = partitions[:-1]
    return [ParsedPartitionSpec('<internally generated spec>', partitions)]
  else:
    raise AssertionError("Unhandled OpSharding type. Please open a bug report!")

_get_single_pspec = lambda p: pxla.array_mapping_to_axis_resources(
    cast(pxla.ArrayMapping, get_array_mapping(p)))

def _get_partition_spec(ppspec: Sequence[ParsedPartitionSpec]) -> Sequence[PartitionSpec]:
  return [_get_single_pspec(p) for p in ppspec]


def _get_ppspec_from_executable(executable, mesh) -> Tuple[Sequence[ParsedPartitionSpec], Sequence[ParsedPartitionSpec]]:
  input_op_shardings: Sequence[xc.OpSharding] = executable.hlo_modules()[0].spmd_parameters_shardings
  output_op_sharding: xc.OpSharding = executable.hlo_modules()[0].spmd_output_sharding
  in_ppspec: List[ParsedPartitionSpec] = []
  for s in input_op_shardings:
    in_ppspec.extend(parse_flatten_op_sharding(s, mesh))
  out_ppspec = parse_flatten_op_sharding(output_op_sharding, mesh)
  return in_ppspec, out_ppspec


def _get_sharding_from_executable(
    executable, mesh: pxla.Mesh
) -> Tuple[Tuple[PartitionSpec, ...], Tuple[PartitionSpec, ...]]:
  in_ppspec, out_ppspec = _get_ppspec_from_executable(executable, mesh)
  out_partition_spec = _get_partition_spec(out_ppspec)
  in_partition_spec = _get_partition_spec(in_ppspec)
  return tuple(in_partition_spec), tuple(out_partition_spec)
