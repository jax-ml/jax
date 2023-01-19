# Copyright 2018 The JAX Authors.
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

import jax
from jax import core
from jax import tree_util
from jax._src import linear_util as lu
from jax.experimental import pjit

from jax._src.lib.mlir.dialects import hlo
from jax._src.lib.mlir import ir
import jax.interpreters.pxla as pxla
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax._src import custom_api_util
from jax._src.lib import xla_client as xc
from jax._src.api_util import flatten_fun_nokwargs

import weakref


class _ShardingCallbackInfo:

  def __init__(self, propagate_user_sharding, partition, to_mesh_pspec_sharding,
               infer_sharding_from_operands, module_context, mesh):
    self.propagate_user_sharding = propagate_user_sharding
    self.partition = partition
    self.to_mesh_pspec_sharding = to_mesh_pspec_sharding
    self.infer_sharding_from_operands = infer_sharding_from_operands
    self.module_context = module_context
    self.mesh = mesh


_sharding_callbacks = weakref.WeakValueDictionary()  # type: ignore

_CUSTOM_PARTITIONING_CALL_NAME = "CustomSPMDPartitioning"


def _to_jax_shape(s):
  return jax.core.ShapedArray(s.dimensions(), s.numpy_dtype())


def _custom_partitioning_propagate_user_sharding(sharding, shape, backend_string):
  return _sharding_callbacks[backend_string].propagate_user_sharding(sharding, shape)


def _custom_partitioning_partition(arg_shapes, arg_shardings, result_shape,
                                   result_sharding, backend_string):
  info = _sharding_callbacks[backend_string]
  lower_fn, result_sharding, arg_shardings = info.partition(
      [_to_jax_shape(s) for s in arg_shapes],
      [info.to_mesh_pspec_sharding(s.to_proto()) for s in arg_shardings],
      _to_jax_shape(result_shape),
      info.to_mesh_pspec_sharding(result_sharding.to_proto()))
  module_context = info.module_context

  def to_hlo_sharding(sharding, shape):
    return xc.HloSharding.from_proto(
        sharding._to_xla_op_sharding(len(shape.dimensions())))

  result_sharding = to_hlo_sharding(result_sharding, result_shape)
  arg_shardings = [
      to_hlo_sharding(sharding, s)
      for sharding, s in zip(arg_shardings, arg_shapes)
  ]
  tiled_args = [
      _to_jax_shape(sharding.tile(s))
      for sharding, s in zip(arg_shardings, arg_shapes)
  ]
  closed_jaxpr = jax.make_jaxpr(
      lower_fn, axis_env=list(info.mesh.shape.items()))(*tiled_args)
  axis_context = mlir.SPMDAxisContext(info.mesh)
  built = mlir.build_xla_computation_helper(
      closed_jaxpr,
      name="tmp_xla_computation",
      platform=module_context.platform,
      backend_or_name=module_context.backend_or_name,
      axis_context=axis_context.extend_manual(frozenset(info.mesh.axis_names)))
  return built, arg_shardings, result_sharding


def _custom_partitioning_infer_sharding_from_operands(arg_shapes, arg_shardings,
                                                      shape, backend_string):
  info = _sharding_callbacks[backend_string]
  result_shape = _to_jax_shape(shape)
  result = info.infer_sharding_from_operands(
      [_to_jax_shape(s) for s in arg_shapes],
      [info.to_mesh_pspec_sharding(s.to_proto()) for s in arg_shardings],
      result_shape)
  return xc.HloSharding.from_proto(
      result._to_xla_op_sharding(len(result_shape.shape)))


custom_partitioning_p = core.Primitive("custom_partitioning")
custom_partitioning_p.multiple_results = True


def _custom_partitioning_abstract_eval(*avals, call, in_tree, out_tree,
                                      propagate_user_sharding, partition,
                                      infer_sharding_from_operands):
  del in_tree, out_tree, propagate_user_sharding, partition, infer_sharding_from_operands
  return call.out_avals


def _custom_partitioning_impl(*args, call, in_tree, out_tree, propagate_user_sharding,
                             partition, infer_sharding_from_operands):
  del in_tree, out_tree, propagate_user_sharding, partition, infer_sharding_from_operands
  return core.jaxpr_as_fun(call)(*args)


custom_partitioning_p.def_abstract_eval(_custom_partitioning_abstract_eval)
custom_partitioning_p.def_impl(_custom_partitioning_impl)

def _default_propagate_user_shardings(sharding, shape):
  return sharding


@custom_api_util.register_custom_decorator_type
class custom_partitioning:
  """Inserts a CustomCallOp into the XLA graph with custom SPMD lowering rules.

  Usage
  -----

  .. code-block:: python

    @custom_partitioning
    def f(*args):
      return ...

    def propagate_user_sharding(sharding, shape):
      '''Update the sharding of the op from a user's sharding.'''

    def partition(arg_shapes, arg_shardings, result_shape, result_sharding):
      def lower_fn(*args):
        ... builds computation on per-device shapes ...
      # result_sharding and arg_shardings may optionally be modified and the
      # partitioner will insert collectives to reshape.
      return lower_fn, result_sharding, arg_shardings

    def infer_sharding_from_operands(arg_shapes, arg_shardings, shape):
      '''Compute the result sharding from the sharding of the operands.'''

    f.def_partition(partition, propagate_user_sharding, infer_sharding_from_operands)

  The args to ``def_partition`` are as follows:

  * ``propagate_user_sharding``: Callable which takes the sharding of a user (in the dag)
    and returns a suggestion for a new `NamedSharding`. The default
    implementation is just to return the suggested sharding.
  * ``partition``: Callable which takes the SPMD suggested partition shapes and
    partition specs and returns a per-shard lowering function and the final
    input and output sharding specs (the SPMD partitioner will repartition the
    inputs to match).
  * ``infer_sharding_from_operands``: Callable which computes an output ``NamedSharding``
    from the ``NamedSharding`` chosen for each argument.

  Example
  -------

  Assume we want to enhance the existing ``jax.numpy.fft.fft``. This function computes the
  discrete Fourier transform of an N-dimensional input along the last dimension, and is batched
  along the first N-1 dimensions.
  By default, however, it will ignore the sharding of the input and gather the input on all devices.
  However, since ``jax.numpy.fft.fft`` is batched along the first N-1 dimensions,
  this is unnecessary. We will create a new ``my_fft`` op that, instead, does not alter the sharding
  along the first `N-1` dimensions, and only gathers the input along the last dimension if needed.

  .. code-block:: python

    import jax
    from jax._src.sharding import NamedSharding
    from jax.experimental.custom_partitioning import custom_partitioning
    from jax.experimental.pjit import pjit
    from jax.sharding import PartitionSpec as P
    from jax.experimental.maps import Mesh
    from jax.numpy.fft import fft
    import numpy as np

    # For an N-D input, keeps sharding along the first N-1 dimensions
    # but replicate along the last dimension
    def supported_sharding(sharding, shape):
        rank = len(shape.shape)
        max_shared_dims = min(len(sharding.spec), rank-1)
        names = tuple(sharding.spec[:max_shared_dims]) + tuple(None for _ in range(rank - max_shared_dims))
        return NamedSharding(sharding.mesh, P(*names))

    def partition(arg_shapes, arg_shardings, result_shape, result_sharding):
        return fft, \
            supported_sharding(arg_shardings[0], arg_shapes[0]), \
            [supported_sharding(arg_shardings[0], arg_shapes[0])]

    def infer_sharding_from_operands(arg_shapes, arg_shardings, shape):
        return supported_sharding(arg_shardings[0], arg_shapes[0])

    @custom_partitioning
    def my_fft(x):
        return fft(x)

    my_fft.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition)

  Now create a 2D array sharded along the first axis, pass it through ``my_fft``
  and notice how it is still sharded as expected, and identical to the output
  of ``fft``. However, the output of ``fft`` is replicated

  >>> with Mesh(np.array(jax.devices()), ('x',)):
  ...   x = np.asarray(np.random.randn(32*1024, 1024), dtype=np.complex64)
  ...   y = pjit(lambda x: x, in_axis_resources=None, out_axis_resources=P('x'))(x)
  ...   pjit_my_fft = pjit(my_fft, in_axis_resources=P('x'), out_axis_resources=P('x'))
  ...   pjit_fft    = pjit(fft,    in_axis_resources=P('x'), out_axis_resources=P('x'))
  ...   print(pjit_my_fft(y))
  ...   print(pjit_fft(y))

  .. code-block::

    # my_fft
    [[-38.840824   +0.j        -40.649452  +11.845365j
     ...
      -1.6937828  +0.8402481j  15.999859   -4.0156755j]]

    # jax.numpy.fft.fft
    [[-38.840824   +0.j        -40.649452  +11.845365j
      ...
      -1.6937828  +0.8402481j  15.999859   -4.0156755j]]

  If we dump the HLO using ``XLA_FLAGS="--xla_dump_to=$(pwd)"``, we see that ``pjit_fft`` compiles
  to

  .. code-block::

    HloModule pjit_fft, entry_computation_layout={(c64[16384,1024]{1,0})->c64[16384,1024]{1,0}}

    fused_computation {
      ...
      ROOT dynamic-slice.2 = c64[16384,1024]{1,0} dynamic-slice(param_1, multiply.3, constant_7), dynamic_slice_sizes={16384,1024}, metadata={op_name="pjit(fft)/jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(1024,)]" source_file="doc.py" source_line=42}
    }

    ENTRY main.8_spmd {
      param = c64[16384,1024]{1,0} parameter(0), sharding={devices=[2,1]0,1}
      all-gather = c64[32768,1024]{1,0} all-gather(param), channel_id=1, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(fft)/jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(1024,)]" source_file="doc.py" source_line=42}
      fft.1 = c64[32768,1024]{1,0} fft(all-gather), fft_type=FFT, fft_length={1024}, metadata={op_name="pjit(fft)/jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(1024,)]" source_file="doc.py" source_line=42}
      partition-id = u32[] partition-id(), metadata={op_name="pjit(fft)/jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(1024,)]" source_file="doc.py" source_line=42}
      ROOT fusion = c64[16384,1024]{1,0} fusion(fft.1, partition-id), kind=kLoop, calls=fused_computation, metadata={op_name="pjit(fft)/jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(1024,)]" source_file="doc.py" source_line=42}
    }

  Where the ``all-gather`` before the FFT and the dynamic-slice after are both clearly visible. 
  This means that the input is gathered on all devices prior to the FFT, and sliced after.

  ``pjit_my_fft``, on the other hand, simply compiles to

  .. code-block::

    HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(c64[16384,1024]{1,0})->c64[16384,1024]{1,0}}

    ENTRY main.5_spmd {
      param = c64[16384,1024]{1,0} parameter(0), sharding={devices=[2,1]0,1}
      ROOT fft.0 = c64[16384,1024]{1,0} fft(param), fft_type=FFT, fft_length={1024}, metadata={op_name="jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(1024,)]" source_file="doc.py" source_line=41}
    }

  where no unnecessary sharding is taking place.

  Because of the logic in ``supported_sharding``, ``my_fft`` also works on 1-dimensional arrays.

  >>> with Mesh(np.array(jax.devices()), ('x',)):
  ...   x = np.asarray(np.random.randn(32*1024*1024), dtype=np.complex64)
  ...   y = pjit(lambda x: x, in_axis_resources=None, out_axis_resources=P('x'))(x)
  ...   pjit_my_fft = pjit(my_fft, in_axis_resources=P('x'), out_axis_resources=P('x'))
  ...   pjit_fft    = pjit(fft,    in_axis_resources=P('x'), out_axis_resources=P('x'))
  ...   print(pjit_my_fft(y))
  ...   print(pjit_fft(y))

  .. code-block::

    # my_fft
    [    7.217285   +0.j     -3012.4937  +4287.635j   -405.83594 +3042.984j
    ...  1422.4502  +7271.4297j  -405.84033 -3042.983j
    -3012.4963  -4287.6343j]

    # jax.numpy.fft.fft
    [    7.217285   +0.j     -3012.4937  +4287.635j   -405.83594 +3042.984j
    ...  1422.4502  +7271.4297j  -405.84033 -3042.983j
    -3012.4963  -4287.6343j]

  In this case, the HLO of ``my_fft`` does show an all-gather and dynamic-slice, since the last dimension
  is the dimension along which FFTs are calculated.

  .. code-block::

    HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(c64[16777216]{0})->c64[16777216]{0}}

    fused_computation {
     ...
      ROOT dynamic-slice.2 = c64[16777216]{0} dynamic-slice(param_1, multiply.3), dynamic_slice_sizes={16777216}, metadata={op_name="pjit(<unnamed wrapped function>)/jit(main)/custom_partitioning[partition=<function partition at 0x7f73a1fbc820> propagate_user_sharding=<function _default_propagate_user_shardings at 0x7f73a1fbc550> infer_sharding_from_operands=<function infer_sharding_from_operands at 0x7f73a1fbc8b0> in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*)]" source_file="doc.py" source_line=51}
    }

    ENTRY main.5_spmd {
      param = c64[16777216]{0} parameter(0), sharding={devices=[2]0,1}
      all-gather = c64[33554432]{0} all-gather(param), channel_id=1, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(<unnamed wrapped function>)/jit(main)/custom_partitioning[partition=<function partition at 0x7f73a1fbc820> propagate_user_sharding=<function _default_propagate_user_shardings at 0x7f73a1fbc550> infer_sharding_from_operands=<function infer_sharding_from_operands at 0x7f73a1fbc8b0> in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*)]" source_file="doc.py" source_line=51}
      fft.0 = c64[33554432]{0} fft(all-gather), fft_type=FFT, fft_length={33554432}, metadata={op_name="jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(33554432,)]" source_file="doc.py" source_line=51}
      partition-id = u32[] partition-id(), metadata={op_name="pjit(<unnamed wrapped function>)/jit(main)/custom_partitioning[partition=<function partition at 0x7f73a1fbc820> propagate_user_sharding=<function _default_propagate_user_shardings at 0x7f73a1fbc550> infer_sharding_from_operands=<function infer_sharding_from_operands at 0x7f73a1fbc8b0> in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*)]" source_file="doc.py" source_line=51}
      ROOT fusion = c64[16777216]{0} fusion(fft.0, partition-id), kind=kLoop, calls=fused_computation, metadata={op_name="pjit(<unnamed wrapped function>)/jit(main)/custom_partitioning[partition=<function partition at 0x7f73a1fbc820> propagate_user_sharding=<function _default_propagate_user_shardings at 0x7f73a1fbc550> infer_sharding_from_operands=<function infer_sharding_from_operands at 0x7f73a1fbc8b0> in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*)]" source_file="doc.py" source_line=51}
    }

  """

  def __init__(self, fun):
    self.fun = fun
    self.partition = None
    self.propagate_user_sharding = None
    self.infer_sharding_from_operands = None

  __getattr__ = custom_api_util.forward_attr

  def def_partition(self, partition, infer_sharding_from_operands,
                    propagate_user_sharding=_default_propagate_user_shardings):
    self.partition = partition
    self.propagate_user_sharding = propagate_user_sharding
    self.infer_sharding_from_operands = infer_sharding_from_operands
    return partition

  def __call__(self, *args):
    args_flat, in_tree = tree_util.tree_flatten(args)
    flat_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(self.fun), in_tree)
    in_avals = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
    debug = pe.debug_info(self.fun, in_tree, False, "custom_partitioning")
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals, debug)
    assert not len(consts)
    closed_call = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
    out_flat = custom_partitioning_p.bind(
        *consts,
        *args_flat,
        call=closed_call,
        partition=self.partition,
        propagate_user_sharding=self.propagate_user_sharding,
        infer_sharding_from_operands=self.infer_sharding_from_operands,
        in_tree=in_tree,
        out_tree=out_tree())
    return tree_util.tree_unflatten(out_tree(), out_flat)


def _custom_partitioning_lowering_rule(ctx: mlir.LoweringRuleContext, *values,
                                       call, in_tree, out_tree,
                                       propagate_user_sharding, partition,
                                       infer_sharding_from_operands):
  mesh = pxla.thread_resources.env.physical_mesh
  axis_context = ctx.module_context.axis_context

  if isinstance(axis_context, mlir.ShardingContext):
    devices = axis_context.device_assignment
  elif isinstance(axis_context, mlir.SPMDAxisContext):
    devices = list(axis_context.mesh.devices.flat)
  else:
    devices = None

  if not devices or len(devices) == 1:
    return mlir.lower_fun(
        core.jaxpr_as_fun(call), multiple_results=True)(ctx, *values)

  def to_mesh_pspec_sharding(op_sharding: xc.OpSharding):
    if mesh.empty:
      from jax._src.sharding import OpShardingSharding
      return OpShardingSharding(devices, op_sharding)
    pspec = pjit.parse_flatten_op_sharding(op_sharding,
                                           mesh)[0].get_partition_spec()
    return pjit.NamedSharding(mesh, pspec)

  sharding_callback_info = _ShardingCallbackInfo(propagate_user_sharding, partition,
                                                to_mesh_pspec_sharding,
                                                infer_sharding_from_operands,
                                                ctx.module_context, mesh)
  key = str(id(sharding_callback_info))
  _sharding_callbacks[key] = sharding_callback_info
  # We need to make sure `sharding_callback_info` is still alive when the SPMD
  # partitioner runs so we keep it alive by attaching it to the executable.
  ctx.module_context.add_keepalive(sharding_callback_info)

  mlir_shapes = [mlir.aval_to_ir_types(s) for s in call.out_avals]
  if len(mlir_shapes) == 1:
    out_type = mlir_shapes[0]
  else:
    out_type = [ir.TupleType.get_tuple(mlir_shapes)]

  out = hlo.CustomCallOp(
      out_type,
      list(values),
      call_target_name=ir.StringAttr.get(_CUSTOM_PARTITIONING_CALL_NAME),
      has_side_effect=ir.BoolAttr.get(False),
      api_version=mlir.i32_attr(2),
      called_computations=ir.ArrayAttr.get([]),
      backend_config=ir.StringAttr.get(key),
      operand_layouts=None,
      result_layouts=None)
  if len(mlir_shapes) == 1:
    return [out.result]
  else:
    return [
        hlo.GetTupleElementOp(out, mlir.i32_attr(i)).result
        for i in range(len(mlir_shapes))
    ]


mlir.register_lowering(custom_partitioning_p,
                       _custom_partitioning_lowering_rule)

xc.register_custom_call_partitioner(  # pytype: disable=module-attr
    _CUSTOM_PARTITIONING_CALL_NAME,
    _custom_partitioning_propagate_user_sharding,
    _custom_partitioning_partition,
    _custom_partitioning_infer_sharding_from_operands, True)
