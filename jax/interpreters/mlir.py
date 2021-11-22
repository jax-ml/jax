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

# Lowering and execution path that converts jaxprs into the MLIR MHLO/CHLO
# dialects.

import collections
import dataclasses
from functools import partial
import io
import itertools
import typing
from typing import (Any, Callable, Dict, Optional, Sequence, Type, Union, Tuple)
from typing_extensions import Protocol
import warnings

from absl import logging
from jax import core
from jax import linear_util as lu
from jax._src.config import config
from jax._src import ad_util
from jax._src import custom_derivatives
from jax._src import device_array
from jax._src import dispatch
from jax._src import dtypes
from jax._src.lax import lax
from jax._src.lax import control_flow
from jax._src.lax import windowed_reductions as lax_windowed_reductions
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import builtin
from jax._src.lib.mlir.dialects import chlo
from jax._src.lib.mlir.dialects import mhlo
from jax._src.lib.mlir.dialects import std
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
import jax._src.prng as prng
from jax._src import source_info_util
import jax._src.util as util
from jax.errors import UnexpectedTracerError
import jax.interpreters.ad as ad
import jax.interpreters.partial_eval as pe
import jax.interpreters.xla as xla
import numpy as np


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

T = typing.TypeVar("T")

# mypy implicitly sets this variable to true when type checking.
MYPY = False


# IR Helpers
def _dense_int_elements(xs: Sequence[int]) -> ir.DenseIntElementsAttr:
  return ir.DenseIntElementsAttr.get(np.asarray(xs, np.int64))

def _dense_bool_elements(xs: Sequence[bool]) -> ir.DenseElementsAttr:
  return ir.DenseElementsAttr.get(
      np.packbits(np.array(xs, np.bool_), bitorder='little'),
      type=ir.IntegerType.get_signless(1), shape=[len(xs)])

def _i32_attr(i): return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), i)
def _i64_attr(i): return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i)

def _real_dtype(dtype): return np.finfo(dtype).dtype

# IR Types

# Non-canonicalized dtype to IR type mapping.
_dtype_to_ir_type : Dict[np.dtype, Callable[[], ir.Type]] = {
  np.dtype(dtypes.float0): partial(ir.IntegerType.get_signless, 1),
  np.dtype(np.bool_): partial(ir.IntegerType.get_signless, 1),
  np.dtype(np.int8): partial(ir.IntegerType.get_signless, 8),
  np.dtype(np.int16): partial(ir.IntegerType.get_signless, 16),
  np.dtype(np.int32): partial(ir.IntegerType.get_signless, 32),
  np.dtype(np.int64): partial(ir.IntegerType.get_signless, 64),
  np.dtype(np.uint8): partial(ir.IntegerType.get_unsigned, 8),
  np.dtype(np.uint16): partial(ir.IntegerType.get_unsigned, 16),
  np.dtype(np.uint32): partial(ir.IntegerType.get_unsigned, 32),
  np.dtype(np.uint64): partial(ir.IntegerType.get_unsigned, 64),
  np.dtype(dtypes.bfloat16): ir.BF16Type.get,
  np.dtype(np.float16): ir.F16Type.get,
  np.dtype(np.float32): ir.F32Type.get,
  np.dtype(np.float64): ir.F64Type.get,
  np.dtype(np.complex64): lambda: ir.ComplexType.get(ir.F32Type.get()),
  np.dtype(np.complex128): lambda: ir.ComplexType.get(ir.F64Type.get()),
}


def _array_ir_types(aval: core.ShapedArray) -> ir.Type:
  try:
    ir_type_factory = _dtype_to_ir_type[aval.dtype]
  except KeyError as err:
    raise TypeError(
        f"No dtype_to_ir_type handler for dtype: {aval.dtype}") from err
  return (ir.RankedTensorType.get(aval.shape, ir_type_factory()),)

_ir_type_handlers: Dict[Type[core.AbstractValue],
                        Callable[[Any], Sequence[ir.Type]]] = {}

def aval_to_ir_types(aval: core.AbstractValue) -> Sequence[ir.Type]:
  """Converts a JAX aval to zero or more MLIR IR types.

  In general, a JAX value may be represented by multiple IR values, so this
  function returns multiple types."""
  try:
    return _ir_type_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No ir_type_handler for aval type: {type(aval)}") from err

_ir_type_handlers[core.AbstractUnit] = lambda _: ()
_ir_type_handlers[core.ShapedArray] = _array_ir_types
_ir_type_handlers[core.ConcreteArray] = _array_ir_types
_ir_type_handlers[core.AbstractToken] = lambda _: [mhlo.TokenType.get()]

def aval_to_ir_type(aval: core.AbstractValue) -> ir.Type:
  """Convenience wrapper around aval_to_ir_types for single types.

  For some common cases, e.g. dense arrays, we know JAX values are represented
  by a single IR value."""
  types = aval_to_ir_types(aval)
  if len(types) != 1:
    raise TypeError(f"aval_to_ir_type called on {aval} which corresponds to "
                    f"multiple IR types {types}")
  return types[0]


# Constants

class ConstantHandler(Protocol):
  def __call__(self, val: Any, canonicalize_types: bool) -> Sequence[ir.Value]:
    """Builds an IR representation for a constant `val`.

    A JAX value is represented by zero or more IR values."""

_constant_handlers : Dict[type, ConstantHandler] = {}

def register_constant_handler(type_: type, handler_fun: ConstantHandler):
  _constant_handlers[type_] = handler_fun

def ir_constants(val: Any,
                 canonicalize_types: bool = True) -> Sequence[ir.Value]:
  """Translate a Python `val` to an IR constant, canonicalizing its dtype.

  Args:
    val: a Python value to be translated to a constant.

  Returns:
    A representation of the constant as a list of IR values.
  """
  for t in type(val).mro():
    handler = _constant_handlers.get(t)
    if handler: return handler(val, canonicalize_types)
  if hasattr(val, '__jax_array__'):
    return ir_constants(val.__jax_array__(), canonicalize_types)
  raise TypeError("No constant handler for type: {}".format(type(val)))

register_constant_handler(core.Unit, lambda val, canonicalize_types: ())

def _numpy_array_constant(x: np.ndarray, canonicalize_types
                         ) -> Sequence[ir.Value]:
  if canonicalize_types:
    x = np.asarray(x, dtypes.canonicalize_dtype(x.dtype))
  aval = xla.abstractify(x)
  ir_type = aval_to_ir_type(aval)
  if x.dtype == np.bool_:
    x = np.packbits(x, bitorder='little')
  elif x.dtype == dtypes.bfloat16:
    x = x.view(np.uint16)
  x = np.ascontiguousarray(x)
  attr = ir.DenseElementsAttr.get(x, type=ir_type.element_type,
                                  shape=aval.shape)
  return (mhlo.ConstOp(ir_type, attr).result,)


def _ndarray_constant_handler(val: np.ndarray, canonicalize_types
                             ) -> Sequence[ir.Value]:
  """Constant handler for ndarray literals, handling zero-size strides.

  In most cases this function calls _numpy_array_constant(val) except it has
  special handling of arrays with any strides of size zero: for those, it
  generates appropriate calls to NumpyArrayConstant, Broadcast, and Transpose
  to avoid staging in large literals that might arise from np.zeros or np.ones
  or the output of lax.broadcast (which uses np.broadcast_to which in turn
  uses size-zero strides).

  Args:
    val: an ndarray.

  Returns:
    An XLA ComputationDataHandle / XlaOp representing the constant ndarray
    staged into the XLA Computation.
  """
  if dtypes.result_type(val) == dtypes.float0:
    return _numpy_array_constant(np.zeros(val.shape, dtype=np.bool_),
                                 canonicalize_types=False)
  elif np.any(np.equal(0, val.strides)) and val.size > 0:
    zero_stride_axes, = np.where(np.equal(0, val.strides))
    other_axes, = np.where(np.not_equal(0, val.strides))
    collapsed_val = val[tuple(0 if ax in zero_stride_axes else slice(None)
                              for ax in range(val.ndim))]
    out = mhlo.BroadcastInDimOp(
        aval_to_ir_type(xla.abstractify(val)),
        _numpy_array_constant(collapsed_val, canonicalize_types)[0],
        _dense_int_elements(other_axes)).result
    return (out,)
  else:
    return _numpy_array_constant(val, canonicalize_types)

register_constant_handler(np.ndarray, _ndarray_constant_handler)

for _scalar_type in [np.int8, np.int16, np.int32, np.int64,
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.float16, np.float32, np.float64,
                    np.bool_, np.longlong, dtypes.bfloat16]:
  register_constant_handler(_scalar_type, _ndarray_constant_handler)

def _python_scalar_handler(dtype, val, canonicalize_dtypes):
  return _numpy_array_constant(np.array(val, dtype), canonicalize_dtypes)

for ptype, dtype in dtypes.python_scalar_dtypes.items():
  register_constant_handler(ptype, partial(_python_scalar_handler, dtype))

def _device_array_constant_handler(val, canonicalize_types):
  return _ndarray_constant_handler(val.device_buffer.to_py(),
                                   canonicalize_types)
for t in device_array.device_array_types:
  register_constant_handler(t, _device_array_constant_handler)


# Source locations

def _source_info_to_location(
    source_info: source_info_util.SourceInfo) -> ir.Location:
  frame = source_info_util.user_frame(source_info)
  if frame is None:
    return ir.Location.unknown()
  return ir.Location.file(xla._get_canonical_source_file(frame), frame.line_num,
                          1)


# Translation rules

@dataclasses.dataclass
class LoweringContext:
  context: ir.Context
  module: ir.Module
  ip: ir.InsertionPoint
  symbol_table: ir.SymbolTable
  platform: str
  axis_env: xla.AxisEnv
  name_stack: str

  # Should function results be tupled?
  tuple_results: bool

  def __init__(self, platform: str, axis_env: xla.AxisEnv, name_stack: str,
               context: Optional[ir.Context] = None,
               module: Optional[ir.Module] = None,
               ip: Optional[ir.InsertionPoint] = None,
               symbol_table: Optional[ir.SymbolTable] = None,
               tuple_results: bool = True):
    assert platform is not None
    self.context = context or ir.Context()
    self.module = module or ir.Module.create(loc=ir.Location.unknown(self.context))
    self.ip = ip or ir.InsertionPoint(self.module.operation.opview.body)
    self.symbol_table = symbol_table or ir.SymbolTable(self.module.operation)
    self.platform = platform
    self.axis_env = axis_env
    self.name_stack = name_stack
    self.tuple_results = tuple_results
    mhlo.register_mhlo_dialect(self.context)
    chlo.register_chlo_dialect(self.context)

  def replace(self, **kw): return dataclasses.replace(self, **kw)


if not MYPY:
  class TranslationRule(Protocol):
    def __call__(self, ctx: LoweringContext,
                 avals_in: Sequence[core.AbstractValue],
                 avals_out: Sequence[core.AbstractValue],
                 *args: Union[ir.Value, Sequence[ir.Value]],
                 **kw) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
      """Converts a JAX primitive invocation into MLIR."""
else:
  TranslationRule = Any

translations: Dict[core.Primitive, TranslationRule] = {}
platform_specific_translations: Dict[str, Dict[core.Primitive, TranslationRule]]
platform_specific_translations = collections.defaultdict(dict)


def _unwrap_singleton_ir_values(x): return x[0] if len(x) == 1 else x
def _wrap_singleton_ir_values(x: Union[ir.Value, Sequence[ir.Value]]
                             ) -> Sequence[ir.Value]:
  return (x,) if isinstance(x, ir.Value) else tuple(x)

def _flatten_lowering_ir_args(
    xs: Sequence[Union[ir.Value, Sequence[ir.Value]]]
) -> Sequence[Sequence[ir.Value]]:
  return util.flatten(map(_wrap_singleton_ir_values, xs))

def lower_jaxpr_to_fun(ctx: LoweringContext, name: str,
                       jaxpr: core.ClosedJaxpr, *,
                       public: bool = False,
                       prune_tokens: bool = False) -> str:
  """Lowers jaxpr and its callees to an IR function.

  Assumes that an MLIR context, location, and insertion point are set.

  Args:
    ctx: the lowering context.
    name: the function name. The name will be uniquified by the symbol table,
      so it is ok to use the same name multiple times.
    jaxpr: the jaxpr to lower.
    public: if true, the function's visibility is set to "public".
    prune_tokens: if true, tokens are pruned from the arguments and return
      values.
  Returns the name of the function.
  """
  # print(jaxpr.jaxpr)
  if prune_tokens:
    pruned_in_avals = [aval for aval in jaxpr.in_avals
                       if aval is not core.abstract_token]
    pruned_out_avals = [aval for aval in jaxpr.out_avals
                        if aval is not core.abstract_token]
  else:
    pruned_in_avals = jaxpr.in_avals
    pruned_out_avals = jaxpr.out_avals

  input_types = map(aval_to_ir_types, pruned_in_avals)
  output_types = map(aval_to_ir_types, pruned_out_avals)
  flat_input_types = util.flatten(input_types)
  flat_output_types = util.flatten(output_types)
  if ctx.tuple_results:
    output_tuple_type = ir.TupleType.get_tuple(flat_output_types)
    fn_output_types = [output_tuple_type]
  else:
    fn_output_types = flat_output_types
  ftype = ir.FunctionType.get(flat_input_types, fn_output_types)
  func_op = builtin.FuncOp(name, ftype, ip=ctx.ip)
  func_op.attributes["sym_visibility"] = ir.StringAttr.get(
      "public" if public else "private")
  symbol_name = ir.StringAttr(ctx.symbol_table.insert(func_op)).value
  entry_block = func_op.add_entry_block()
  with ir.InsertionPoint(entry_block):
    unflattened_args = util.unflatten(entry_block.arguments,
                                      map(len, input_types))
    # If we pruned tokens out of the parameter list, create a new token and add
    # it here.
    if prune_tokens and len(pruned_in_avals) != len(jaxpr.in_avals):
      token = mhlo.CreateTokenOp(mhlo.TokenType.get()).results
      arg_iter = iter(unflattened_args)
      unflattened_args = [
          token if aval is core.abstract_token else next(arg_iter)
          for aval in jaxpr.in_avals
      ]
      done = object()
      assert next(arg_iter, done) is done

    callee_name_stack = xla.extend_name_stack(ctx.name_stack,
                                              xla.wrap_name(name, 'jit'))
    out_vals = jaxpr_subcomp(ctx.replace(name_stack=callee_name_stack),
                             jaxpr.jaxpr, map(ir_constants, jaxpr.consts),
                             *unflattened_args)
    if prune_tokens:
      out_vals = [v for v, aval in zip(out_vals, jaxpr.out_avals)
                  if aval is not core.abstract_token]
    flat_outputs = util.flatten(out_vals)
    if ctx.tuple_results:
      std.ReturnOp([mhlo.TupleOp(output_tuple_type, flat_outputs).result])
    else:
      std.ReturnOp(flat_outputs)

  return symbol_name


def jaxpr_subcomp(ctx: LoweringContext, jaxpr: core.Jaxpr,
                  consts: Sequence[Sequence[ir.Value]],
                  *args: Sequence[ir.Value]) -> Sequence[Sequence[ir.Value]]:
  """Lowers a jaxpr into mHLO, inlined into an existing function.

  Assumes that an MLIR context, location, and insertion point are set.
  """
  def read(v):
    if type(v) is core.Literal:
      return ir_constants(v.val, canonicalize_types=True)
    else:
      return env[v]

  def aval(v):
    if type(v) is core.Literal:
      return xla.abstractify(v.val)
    else:
      return v.aval

  def write(v, node):
    assert node is not None
    env[v] = tuple(node)


  env: Dict[core.Var, Tuple[ir.Value]] = {}

  assert len(args) == len(jaxpr.invars), (jaxpr, args)
  assert len(consts) == len(jaxpr.constvars), (jaxpr, consts)
  write(core.unitvar, ())
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_nodes = map(read, eqn.invars)
    # TODO(phawkins): attach the primitive name, parameters, and name stack as
    # metadata.
    loc = _source_info_to_location(eqn.source_info)
    with source_info_util.user_context(eqn.source_info.traceback), loc:
      if eqn.primitive in platform_specific_translations[ctx.platform]:
        rule = platform_specific_translations[ctx.platform][eqn.primitive]
      elif eqn.primitive in translations:
        rule = translations[eqn.primitive]
      elif eqn.primitive in xla._translations:
        rule = partial(_fallback_lowering, eqn.primitive)
      else:
        raise NotImplementedError(
            f"MLIR translation rule for primitive '{eqn.primitive.name}' not "
            "found")

      ans = rule(ctx, map(aval, eqn.invars), map(aval, eqn.outvars),
                 *map(_unwrap_singleton_ir_values, in_nodes),
                 **eqn.params)

    try:
      out_nodes = tuple(map(_wrap_singleton_ir_values, ans))
    except TypeError as e:
      raise ValueError("Output of translation rule must be iterable: "
                       f"{eqn}") from e

    assert all(isinstance(v, tuple) for v in out_nodes), (ans, eqn)
    assert all(isinstance(v, ir.Value) for w in out_nodes for v in w), (ans, eqn)
    assert len(ans) == len(eqn.outvars), (ans, eqn)
    map(write, eqn.outvars, out_nodes)
  return map(read, jaxpr.outvars)

def _ir_consts(consts):
  unique_consts = {id(const): const for const in consts}
  ir_consts = {
      id_: ir_constants(const) for id_, const in unique_consts.items()}
  return [ir_consts[id(const)] for const in consts]

def lower_fun(fun: Callable, multiple_results: bool = True) -> Callable:
  """Converts a traceable JAX function `fun` into a lowering rule.

  The returned function does not use `avals_out`, so callers may pass any value
  as `avals_out`."""
  def f_lowered(ctx, avals_in, avals_out, *args, **params):
    if multiple_results:
      f = fun
    else:
      f = lambda *args, **kw: (fun(*args, **kw),)
    wrapped_fun = lu.wrap_init(f, params)
    with core.extend_axis_env_nd(zip(ctx.axis_env.names, ctx.axis_env.sizes)):
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, avals_in)
    return jaxpr_subcomp(ctx, jaxpr, _ir_consts(consts),
                         *map(_wrap_singleton_ir_values, args))

  return f_lowered



def _call_lowering(fn_name, stack_name, call_jaxpr, backend, ctx, avals_in,
                   avals_out, *args):
  xla.check_backend_matches(backend, ctx.platform)
  output_types = map(aval_to_ir_types, avals_out)
  flat_output_types = util.flatten(output_types)
  if ctx.tuple_results:
    output_tuple_type = ir.TupleType.get_tuple(flat_output_types)
    call_output_types = [output_tuple_type]
  else:
    call_output_types = flat_output_types
  sub_ctx = ctx.replace(
      name_stack=xla.extend_name_stack(ctx.name_stack, stack_name))
  symbol_name = lower_jaxpr_to_fun(sub_ctx, fn_name,
                                   core.ClosedJaxpr(call_jaxpr, ()))
  call = std.CallOp(call_output_types,
                    ir.FlatSymbolRefAttr.get(symbol_name),
                    _flatten_lowering_ir_args(args))
  if ctx.tuple_results:
    flat_results = [
        mhlo.GetTupleElementOp(typ, call.result, _i32_attr(i)).result
        for i, typ in enumerate(flat_output_types)]
  else:
    flat_results = call.results
  return util.unflatten(flat_results, map(len, output_types))

def _xla_call_lower(ctx, avals_in, avals_out, *args,
                    backend=None, name, call_jaxpr, donated_invars, inline=None,
                    device=None):
  del device, donated_invars, inline  # Ignored.
  return _call_lowering(f"jit_{name}", xla.wrap_name(name, "jit"), call_jaxpr,
                        backend, ctx, avals_in, avals_out, *args)

translations[xla.xla_call_p] = _xla_call_lower


def _named_call_lowering(ctx, avals_in, avals_out, *args, name, backend=None,
                         call_jaxpr):
  return _call_lowering(name, name, call_jaxpr, backend, ctx, avals_in,
                        avals_out, *args)

translations[core.named_call_p] = _named_call_lowering
translations[core.call_p] = partial(_named_call_lowering, name="core_call")


def _device_put_lowering(ctx, avals_in, avals_out, x, *, device):
  return [x]

translations[dispatch.device_put_p] = _device_put_lowering


def _full_like_aval(value, aval: core.ShapedArray) -> ir.Value:
  """Returns an IR constant shaped full of `value` shaped like `aval`."""
  zero, = ir_constants(np.array(value, aval.dtype))
  return mhlo.BroadcastOp(aval_to_ir_type(aval), zero,
                          _dense_int_elements(aval.shape)).result

def zeros_like_lowering(ctx, avals_in, avals_out, x):
  aval, = avals_in
  assert isinstance(aval, core.ShapedArray), aval
  return [_full_like_aval(0, aval)]
translations[ad_util.zeros_like_p] = zeros_like_lowering

def add_jaxvals_lowering(ctx, avals_in, avals_out, x, y):
  return mhlo.AddOp(x, y).results
translations[ad_util.add_jaxvals_p] = add_jaxvals_lowering

translations[ad_util.stop_gradient_p] = lambda ctx, avals_in, avals_out, x: [x]


# # Computation dispatch

_aval_to_num_buffers: Dict[Type[core.AbstractValue],
                           Callable[[core.AbstractValue], int]] = {}

def aval_to_num_buffers(aval: core.AbstractValue) -> int:
  """Returns the number of buffers in the runtime representation of `aval`.

  Note: the compile-time representation may have more buffers! This is a small
  hack to deal with tokens that have no true runtime representation but have an
  IR type.

  Must match the arity of the result of `aval_to_ir_types`."""
  try:
    return _aval_to_num_buffers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No num_buffers handler for type: {type(aval)}") from err

_aval_to_num_buffers[core.AbstractUnit] = lambda _: 0
_aval_to_num_buffers[core.AbstractToken] = lambda _: 0
_aval_to_num_buffers[core.ShapedArray] = lambda _: 1
_aval_to_num_buffers[core.ConcreteArray] = lambda _: 1


class ArgHandler(Protocol):
  def __call__(self, device: xla.Device, arg: Any) -> Sequence[xla.Buffer]:
    """A argument handler unboxes raw buffers into their Python representation."""

_aval_to_arg_handler: Dict[
    Type[core.AbstractValue], Callable[[Any], ArgHandler]] = {}

def aval_to_arg_handler(aval: core.AbstractValue) -> ArgHandler:
  try:
    return _aval_to_arg_handler[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No arg_handler for type: {type(aval)}") from err

def _array_arg_handler(aval: core.ShapedArray) -> ArgHandler:
  return lambda device, val: xla.device_put(val, device)

_aval_to_arg_handler[core.AbstractUnit] = lambda _: lambda _device, _: ()
_aval_to_arg_handler[core.AbstractToken] = lambda _: lambda _device, _: ()
_aval_to_arg_handler[core.ShapedArray] = _array_arg_handler
_aval_to_arg_handler[core.ConcreteArray] = _array_arg_handler


if not MYPY:
  class ResultHandler(Protocol):
    def __call__(self, device: xla.Device, *args: xla.Buffer) -> Any:
      """A result handler boxes raw buffers into their Python representation.

      Inverse of ArgHandler."""
else:
  ResultHandler = Any

_aval_to_result_handler: Dict[
    Type[core.AbstractValue], Callable[[Any], ResultHandler]] = {}

def aval_to_result_handler(aval: core.AbstractValue) -> ResultHandler:
  try:
    return _aval_to_result_handler[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No result_handler for type: {type(aval)}") from err

def _array_result_handler(aval: core.ShapedArray) -> ResultHandler:
  if aval.dtype is dtypes.float0:
    return lambda _device, _: np.zeros(aval.shape, dtypes.float0)
  aval = core.raise_to_shaped(aval)
  return lambda device, buffer: xla.make_device_array(aval, device, buffer)

_aval_to_result_handler[core.AbstractUnit] = lambda _: lambda _: core.unit
_aval_to_result_handler[core.AbstractToken] = lambda _: lambda _: xla.token
_aval_to_result_handler[core.ShapedArray] = _array_result_handler
_aval_to_result_handler[core.ConcreteArray] = _array_result_handler


def _execute_compiled(name: str, compiled: xla.XlaExecutable,
                      device: xla.Device, buffer_counts,
                      arg_handlers, result_handlers, kept_var_idx, *args):
  input_bufs = util.flatten(
      arg_handler(device, x) for arg_handler, x in
      unsafe_zip(arg_handlers,
                 (x for i, x in enumerate(args) if i in kept_var_idx)))
  out_bufs = compiled.execute(input_bufs)
  dispatch.check_special(name, out_bufs)
  return [handler(device, *bs) for handler, bs in
          zip(result_handlers, xla._partition_outputs(buffer_counts, out_bufs))]


def _execute_replicated(name: str, compiled: xla.XlaExecutable,
                        device: xla.Device,
                        buffer_counts, arg_handlers, result_handlers,
                        kept_var_idx, *args):
  input_bufs = [
      util.flatten(
        arg_handler(device, x) for arg_handler, x in
        unsafe_zip(arg_handlers,
                   (x for i, x in enumerate(args) if i in kept_var_idx)))
      for device in compiled.local_devices()
  ]
  out_bufs = [
      buf[0] for buf in compiled.execute_sharded_on_local_devices(
          list(zip(*input_bufs)))
  ]
  dispatch.check_special(name, out_bufs)
  return [handler(device, *bs) for handler, bs in
          zip(result_handlers, xla._partition_outputs(buffer_counts, out_bufs))]


def _execute_trivial(jaxpr, device: Optional[xla.Device], consts, buffer_counts,
                     result_handlers, kept_var_idx, *args):
  env = {core.unitvar: core.unit}
  pruned_args = (x for i, x in enumerate(args) if i in kept_var_idx)
  map(env.setdefault, jaxpr.invars, pruned_args)
  map(env.setdefault, jaxpr.constvars, consts)
  outs = [xla.canonicalize_dtype(v.val) if type(v) is core.Literal else env[v]
          for v in jaxpr.outvars]
  return [dispatch.device_put_p.bind(x, device=device) for x in outs]


class XlaCompiledComputation:
  def __init__(self, xla_executable, unsafe_call):
    self._xla_executable = xla_executable
    self.unsafe_call = unsafe_call

  @staticmethod
  def from_xla_computation(
      name: str,
      xla_computation,
      nreps: int,
      device,
      backend,
      tuple_args: bool,
      avals_in: Sequence[core.AbstractValue],
      avals_out: Sequence[core.AbstractValue],
      kept_var_idx) -> 'XlaCompiledComputation':
    arg_handlers = map(aval_to_arg_handler, avals_in)
    result_handlers = map(aval_to_result_handler, avals_out)
    options = xb.get_compile_options(
        num_replicas=nreps,
        num_partitions=1,
        device_assignment=(device.id,) if device else None)
    options.parameter_is_tupled_arguments = tuple_args
    compiled = dispatch.compile_or_get_cached(backend, xla_computation, options)
    buffer_counts = [aval_to_num_buffers(aval) for aval in avals_out]
    if nreps == 1:
      return XlaCompiledComputation(compiled, partial(
          _execute_compiled, name, compiled, device, buffer_counts,
          arg_handlers, result_handlers, kept_var_idx))
    else:
      return XlaCompiledComputation(compiled, partial(
          _execute_replicated, name, compiled, device, buffer_counts,
          arg_handlers, result_handlers, kept_var_idx))

  def is_trivial(self):
    return self._xla_executable == None

  def xla_executable(self):
    if self.is_trivial():
      raise ValueError('A trivial compiled computation has no XLA executable')
    return self._xla_executable

  @staticmethod
  def from_trivial_jaxpr(jaxpr, consts, device, avals_in, avals_out,
                         kept_var_idx)  -> 'XlaCompiledComputation':
    result_handlers = map(aval_to_result_handler, avals_out)
    return XlaCompiledComputation(None, partial(
        _execute_trivial, jaxpr, device, consts, avals_out,
        result_handlers, kept_var_idx))

  def call(self, *args):
    # TODO(apaszke,frostig): Check that args are compatible with input avals!
    return self.unsafe_call(*args)

  def __call__(self, *args):
    return self.call(*args)


class XlaComputation:
  name: str
  _is_trivial: bool
  _executable: Optional['XlaCompiledComputation']

  def __init__(self, name: str, hlo, is_trivial: bool, *compile_args):
    self.name = name
    self._hlo = hlo
    self._is_trivial = is_trivial
    self._executable = None
    self.compile_args = compile_args

  def is_trivial(self):
    return self._is_trivial

  def hlo(self):
    if self.is_trivial():
      raise ValueError('A trivial computation has no HLO')
    return self._hlo

  def compile(self) -> 'XlaCompiledComputation':
    if self._executable is None:
      if self.is_trivial():
        self._executable = XlaCompiledComputation.from_trivial_jaxpr(
            *self.compile_args)
      else:
        self._executable = XlaCompiledComputation.from_xla_computation(
            self.name, self.hlo(), *self.compile_args)
    return self._executable


def _xla_callable_uncached(fun: lu.WrappedFun, device, backend, name, donated_invars, *arg_specs):
  return lower_xla_callable(fun, device, backend, name, donated_invars, *arg_specs).compile().unsafe_call

_xla_callable = lu.cache(_xla_callable_uncached)

# TODO(phawkins): refactor this code to share more with the xla.py version.
def lower_xla_callable(fun: lu.WrappedFun, device, backend, name, donated_invars, *arg_specs):
  if device is not None and backend is not None:
    raise ValueError("can't specify both a device and a backend for jit, "
                     "got device={} and backend={}".format(device, backend))

  avals_in, arg_devices = util.unzip2(arg_specs)
  jaxpr, avals_out, consts = pe.trace_to_jaxpr_final(
      fun, avals_in, pe.debug_info_final(fun, "jit"))
  if any(isinstance(c, core.Tracer) for c in consts):
    raise UnexpectedTracerError("Encountered an unexpected tracer.")

  jaxpr, kept_const_idx, kept_var_idx = dispatch._prune_unused_inputs(jaxpr)
  consts = [c for i, c in enumerate(consts) if i in kept_const_idx]
  pruned_arg_specs = (a for i, a in enumerate(arg_specs) if i in kept_var_idx)
  avals_in, arg_devices = util.unzip2(pruned_arg_specs)
  donated_invars = [
      x for i, x in enumerate(donated_invars) if i in kept_var_idx
  ]
  map(dispatch.prefetch, itertools.chain(consts, dispatch.jaxpr_literals(jaxpr)))
  jaxpr = dispatch.apply_outfeed_rewriter(jaxpr)

  nreps = dispatch.jaxpr_replicas(jaxpr)
  device = dispatch._xla_callable_device(nreps, backend, device, arg_devices)
  backend = xb.get_device_backend(device) if device else xb.get_backend(backend)

  # Computations that only produce constants and/or only rearrange their inputs,
  # which are often produced from partial evaluation, don't need compilation,
  # and don't need to evaluate their arguments.
  if not jaxpr.eqns:
    return XlaComputation(name, None, True, jaxpr, consts, device, avals_in,
                          avals_out, kept_var_idx)

  if not dispatch._on_exit:
    log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
    logging.log(log_priority, "Compiling %s (%s) for args %s.",
                fun.__name__, id(fun), avals_in)

  if nreps > 1:
    warnings.warn(f"The jitted function {fun.__name__} includes a pmap. Using "
         "jit-of-pmap can lead to inefficient data movement, as the outer jit "
         "does not preserve sharded data representations and instead collects "
         "input and output arrays onto a single device. "
         "Consider removing the outer jit unless you know what you're doing. "
         "See https://github.com/google/jax/issues/2926.")

  if nreps > xb.device_count(backend):
    raise ValueError(
        f"compiling computation that requires {nreps} replicas, but only "
        f"{xb.device_count(backend)} XLA devices are available")

  if xb.process_count() > 1 and (nreps > 1 or dispatch.jaxpr_has_pmap(jaxpr)):
    raise NotImplementedError(
        "jit of multi-host pmap not implemented (and jit-of-pmap can cause "
        "extra data movement anyway, so maybe you don't want it after all).")

  ctx = LoweringContext(backend.platform, xla.AxisEnv(nreps, (), ()), "")
  if backend.runtime_type == "iree":
    tuple_args = False
    ctx = ctx.replace(tuple_results=False)
  else:
    tuple_args = len(avals_in) > 100  # pass long arg lists as tuple for TPU

  with ctx.context, ir.Location.unknown(ctx.context):
    # XLA doesn't have a runtime representation of tokens, so we prune them out
    # of the arguments and return values of the top-level function. This is fine
    # since the purpose of tokens is to preserve ordering inside compiled
    # functions.
    lower_jaxpr_to_fun(ctx, "main", core.ClosedJaxpr(jaxpr, consts),
                       public=True, prune_tokens=True)

  assert not any(donated_invars), donated_invars
  # TODO(b/203122001): implement buffer donation.
  # if backend.platform in ("gpu", "tpu"):
  #   donated_invars = set_up_aliases(c, xla_args, out_tuple, donated_invars, tuple_args)
  # if any(donated_invars):
  #   # TODO(tomhennigan): At call time we should mark these buffers as deleted.
  #   unused_donations = [str(c.GetShape(a))
  #                       for a, d in zip(xla_args, donated_invars) if d]
  #   warn("Some donated buffers were not usable: {}".format(", ".join(unused_donations)))
  ctx.module.operation.verify()
  output = io.StringIO()
  ctx.module.operation.print(file=output, #enable_debug_info=True,
                             print_generic_op_form=False)
  module = output.getvalue()
  # print("MLIR module to be compiled:")
  # print(module)
  return XlaComputation(
      name, module, False, nreps, device, backend, tuple_args, avals_in,
      avals_out, kept_var_idx)


def _xla_call_impl_mlir(fun: lu.WrappedFun, *args, device, backend, name,
                        donated_invars, inline):
  del inline  # Only used at tracing time
  compiled_fun = _xla_callable(fun, device, backend, name, donated_invars,
                               *unsafe_map(dispatch.arg_spec, args))
  return compiled_fun(*args)


@util.cache()
def _xla_primitive_callable(prim, *arg_specs: dispatch.ArgSpec, **params):
  avals, arg_devices = util.unzip2(arg_specs)
  donated_invars = (False,) * len(arg_specs)
  device = dispatch._device_from_arg_devices(arg_devices)
  def prim_fun(*args):
    out = prim.bind(*args, **params)
    if prim.multiple_results:
      return out
    else:
      return out,
  compiled = _xla_callable_uncached(lu.wrap_init(prim_fun), device, None,
                                    prim.name, donated_invars, *arg_specs)
  if not prim.multiple_results:
    return lambda *args, **kw: compiled(*args, **kw)[0]
  else:
    return compiled

def apply_primitive(prim, *args, **params):
  """Impl rule that compiles and runs a single primitive 'prim' using XLA."""
  compiled_fun = _xla_primitive_callable(prim, *unsafe_map(dispatch.arg_spec, args),
                                         **params)
  return compiled_fun(*args)


# Translation rules for lax primitives

def _broadcast(aval_out: core.ShapedArray, avals: Sequence[core.ShapedArray],
               args: Sequence[ir.Value]) -> Sequence[ir.Value]:
  out = []
  for aval, arg in zip(avals, args):
    if aval.shape != aval_out.shape:
      assert len(aval.shape) <= len(aval_out.shape), (aval, aval_out)
      dims = _dense_int_elements(
          range(len(aval_out.shape) - len(aval.shape), len(aval_out.shape)))
      arg = mhlo.BroadcastInDimOp(
          aval_to_ir_type(aval.update(shape=aval_out.shape)), arg, dims).result
    out.append(arg)
  return out


def _nary_lower(op: Callable, ctx: LoweringContext,
                avals_in: Sequence[core.ShapedArray],
                avals_out: Sequence[core.ShapedArray],
                *args: Union[ir.Value, Sequence[ir.Value]],
                explicit_type=False, **params):
  del params
  aval_out, = avals_out
  broadcasted_args = _broadcast(aval_out, avals_in, args)
  if explicit_type:
    return op(aval_to_ir_type(aval_out), *broadcasted_args).results
  else:
    return op(*broadcasted_args).results

translations[lax.neg_p] = partial(_nary_lower, mhlo.NegOp)
translations[lax.floor_p] = partial(_nary_lower, mhlo.FloorOp)
translations[lax.ceil_p] = partial(_nary_lower, mhlo.CeilOp)
translations[lax.is_finite_p] = partial(_nary_lower, mhlo.IsFiniteOp)
translations[lax.log_p] = partial(_nary_lower, mhlo.LogOp)
translations[lax.exp_p] = partial(_nary_lower, mhlo.ExpOp)
translations[lax.log1p_p] = partial(_nary_lower, mhlo.Log1pOp)
translations[lax.expm1_p] = partial(_nary_lower, mhlo.Expm1Op)
translations[lax.tanh_p] = partial(_nary_lower, mhlo.TanhOp)
translations[lax.sinh_p] = partial(_nary_lower, chlo.SinhOp)
translations[lax.sin_p] = partial(_nary_lower, mhlo.SinOp)
translations[lax.cos_p] = partial(_nary_lower, mhlo.CosOp)
translations[lax.lgamma_p] = partial(_nary_lower, chlo.LgammaOp)
translations[lax.digamma_p] = partial(_nary_lower, chlo.DigammaOp)
translations[lax.real_p] = partial(_nary_lower, mhlo.RealOp)
translations[lax.imag_p] = partial(_nary_lower, mhlo.ImagOp)

def _conj_impl(x, *, input_dtype):
  if dtypes.issubdtype(x.dtype, np.complexfloating):
    return lax.complex(lax.real(x), -lax.imag(x))
  else:
    return lax.complex(x, lax._zeros(x))

translations[lax.conj_p] = lower_fun(_conj_impl, multiple_results=False)
translations[lax.abs_p] = partial(_nary_lower, mhlo.AbsOp)
translations[lax.sqrt_p] = partial(_nary_lower, mhlo.SqrtOp)
translations[lax.rsqrt_p] = partial(_nary_lower, mhlo.RsqrtOp)
translations[lax.cbrt_p] = partial(_nary_lower, mhlo.CbrtOp)
translations[lax.not_p] = partial(_nary_lower, mhlo.NotOp)


translations[lax.add_p] = partial(_nary_lower, mhlo.AddOp)
translations[lax.sub_p] = partial(_nary_lower, mhlo.SubOp)
translations[lax.mul_p] = partial(_nary_lower, mhlo.MulOp)
translations[lax.div_p] = partial(_nary_lower, mhlo.DivOp)
translations[lax.rem_p] = partial(_nary_lower, mhlo.RemOp)

def _minmax(op, cmp, x, y):
  """Min/max that compares complex values lexicographically as pairs."""
  tensor_type = ir.RankedTensorType(x.type)
  if ir.ComplexType.isinstance(tensor_type.element_type):
    rx = mhlo.RealOp(x).result
    ry = mhlo.RealOp(y).result
    dims = [tensor_type.get_dim_size(i) for i in range(tensor_type.rank)]
    bool_shape = ir.RankedTensorType.get(dims, ir.IntegerType.get_signless(1))
    real_eq = mhlo.CompareOp(bool_shape, rx, ry, ir.StringAttr.get("EQ"),
                             ir.StringAttr.get("FLOAT"))
    real_cmp = mhlo.CompareOp(bool_shape, rx, ry,
                              ir.StringAttr.get(cmp),
                              ir.StringAttr.get("FLOAT"))
    imag_cmp = mhlo.CompareOp(bool_shape, mhlo.ImagOp(x).result,
                              mhlo.ImagOp(y).result,
                              ir.StringAttr.get(cmp),
                              ir.StringAttr.get("FLOAT"))
    which = mhlo.SelectOp(real_eq, imag_cmp, real_cmp).result
    return mhlo.SelectOp(which, x, y)
  else:
    return op(x, y)

_min_op = partial(_minmax, mhlo.MinOp, "LT")
_max_op = partial(_minmax, mhlo.MaxOp, "GT")

translations[lax.min_p] = partial(_nary_lower, _min_op)
translations[lax.max_p] = partial(_nary_lower, _max_op)

translations[lax.shift_left_p] = partial(_nary_lower, mhlo.ShiftLeftOp)
translations[lax.shift_right_logical_p] = partial(
    _nary_lower, mhlo.ShiftRightLogicalOp)
translations[lax.shift_right_arithmetic_p] = partial(
    _nary_lower, mhlo.ShiftRightArithmeticOp)
translations[lax.nextafter_p] = partial(_nary_lower, chlo.NextAfterOp)
translations[lax.exp_p] = partial(_nary_lower, mhlo.ExpOp)
translations[lax.atan2_p] = partial(_nary_lower, mhlo.Atan2Op)
translations[lax.complex_p] = partial(_nary_lower, mhlo.ComplexOp)
translations[lax.pow_p] = partial(_nary_lower, mhlo.PowOp)
translations[lax.and_p] = partial(_nary_lower, mhlo.AndOp)
translations[lax.or_p] = partial(_nary_lower, mhlo.OrOp)
translations[lax.xor_p] = partial(_nary_lower, mhlo.XorOp)
translations[lax.population_count_p] = partial(_nary_lower,
                                               mhlo.PopulationCountOp)
translations[lax.clz_p] = partial(_nary_lower, mhlo.ClzOp)

translations[lax.clamp_p] = partial(_nary_lower, mhlo.ClampOp,
                                    explicit_type=True)
translations[lax.select_p] = partial(_nary_lower, mhlo.SelectOp)

def _sign_lower(ctx, avals_in, avals_out, x):
  x_aval, = avals_in
  if dtypes.issubdtype(x_aval.dtype, np.unsignedinteger):
    return mhlo.SelectOp(
        mhlo.CompareOp(aval_to_ir_type(x_aval.update(dtype=np.dtype(np.bool_))),
                       x, _full_like_aval(0, x_aval),
                       ir.StringAttr.get("EQ"),
                       ir.StringAttr.get("UNSIGNED")).result,
        _full_like_aval(0, x_aval),
        _full_like_aval(1, x_aval)).results
  return mhlo.SignOp(x).results

translations[lax.sign_p] = _sign_lower

def _compare_lower(direction: str, ctx, avals_in, avals_out, x, y):
  x_aval, y_aval = avals_in
  aval_out, = avals_out
  x, y = _broadcast(aval_out.update(dtype=x_aval.dtype), avals_in, (x, y))
  if dtypes.issubdtype(x_aval.dtype, np.inexact):
    compare_type = "FLOAT"
  elif dtypes.issubdtype(x_aval.dtype, np.signedinteger):
    compare_type = "SIGNED"
  else:
    compare_type = "UNSIGNED"
  return mhlo.CompareOp(aval_to_ir_type(aval_out), x, y,
                        ir.StringAttr.get(direction),
                        ir.StringAttr.get(compare_type)).results

translations[lax.eq_p] = partial(_compare_lower, "EQ")
translations[lax.ne_p] = partial(_compare_lower, "NE")
translations[lax.ge_p] = partial(_compare_lower, "GE")
translations[lax.gt_p] = partial(_compare_lower, "GT")
translations[lax.le_p] = partial(_compare_lower, "LE")
translations[lax.lt_p] = partial(_compare_lower, "LT")


def _convert_element_type_lower(ctx, avals_in, avals_out, operand, *,
                                new_dtype, weak_type):
  aval_in, = avals_in
  aval_out, = avals_out
  if (dtypes.issubdtype(aval_in.dtype, np.complexfloating) and
      not dtypes.issubdtype(new_dtype, np.complexfloating)):
    operand = mhlo.RealOp(operand).result
  return mhlo.ConvertOp(aval_to_ir_type(aval_out), operand).results

translations[lax.convert_element_type_p] = _convert_element_type_lower


def _bitcast_convert_type_lower(ctx, avals_in, avals_out, operand, *,
                                new_dtype):
  aval_out, = avals_out
  return mhlo.BitcastConvertOp(aval_to_ir_type(aval_out), operand).results

translations[lax.bitcast_convert_type_p] = _bitcast_convert_type_lower


def _reduce_precision_lower(ctx, avals_in, avals_out, operand, *, exponent_bits,
                            mantissa_bits):
  aval_out, = avals_out
  return mhlo.ReducePrecisionOp(aval_to_ir_type(aval_out), operand,
                                _i32_attr(exponent_bits),
                                _i32_attr(mantissa_bits)).results

translations[lax.reduce_precision_p] = _reduce_precision_lower



def _precision_attr(precision: lax.PrecisionType) -> ir.ArrayAttr:
  if precision is None:
    precision = (lax.Precision.DEFAULT, lax.Precision.DEFAULT)
  elif not isinstance(precision, tuple):
    precision = (precision, precision)
  return ir.ArrayAttr.get([ir.StringAttr.get(str(p)) for p in precision])

def _dot_general_lower(ctx, avals_in, avals_out, lhs, rhs, *, dimension_numbers,
                       precision, preferred_element_type: Optional[np.dtype]):
  del preferred_element_type  # Implied by the output aval.
  aval_out, = avals_out
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  dot_dnums = mhlo.DotDimensionNumbers.get(
      lhs_batching_dimensions=list(lhs_batch),
      rhs_batching_dimensions=list(rhs_batch),
      lhs_contracting_dimensions=list(lhs_contracting),
      rhs_contracting_dimensions=list(rhs_contracting))
  return mhlo.DotGeneralOp(aval_to_ir_type(aval_out), lhs, rhs, dot_dnums,
                           _precision_attr(precision)).results

translations[lax.dot_general_p] = _dot_general_lower


def _complex_mul(mul, x, y):
  # We use a trick for complex multiplication sometimes attributed to Gauss
  # which uses three multiplications and five additions; instead of the naive
  # method of four multiplications and two additions.
  # https://en.wikipedia.org/wiki/Multiplication_algorithm#Complex_multiplication_algorithm
  #
  # This performance win comes with a trade-off in accuracy; especially in
  # cases when the real and imaginary differ hugely in magnitude. The relative
  # error bound (e.g. 1p-24 in case of float32) would be relative to the
  # maximum of real and imaginary parts of the result instead of being
  # satisfied by the real and imaginary parts independently of each other.
  x_re, x_im = lax.real(x), lax.imag(x)
  y_re, y_im = lax.real(y), lax.imag(y)
  k1 = mul(lax.add(x_re, x_im), y_re)
  k2 = mul(x_re, lax.sub(y_im, y_re))
  k3 = mul(x_im, lax.add(y_re, y_im))
  return lax.complex(lax.sub(k1, k3), lax.add(k1, k2))

def _conv_general_dilated_lower(
    ctx, avals_in, avals_out, lhs, rhs, *, window_strides, padding,
    lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count,
    batch_group_count, precision, expand_complex_convolutions=False,
    **unused_kwargs):
  lhs_aval, rhs_aval = avals_in
  aval_out, = avals_out
  assert isinstance(dimension_numbers, lax.ConvDimensionNumbers)
  dtype = lhs_aval.dtype
  if expand_complex_convolutions and np.issubdtype(dtype, np.complexfloating):
    complex_conv = lower_fun(
      partial(
        _complex_mul,
        partial(lax.conv_general_dilated, window_strides=window_strides,
                padding=padding, lhs_dilation=lhs_dilation,
                rhs_dilation=rhs_dilation, dimension_numbers=dimension_numbers,
                feature_group_count=feature_group_count,
                batch_group_count=batch_group_count, precision=precision)),
      multiple_results=False)
    return complex_conv(ctx, avals_in, avals_out, lhs, rhs)

  lhs_spec, rhs_spec, out_spec = dimension_numbers
  dnums = mhlo.ConvDimensionNumbers.get(
    input_batch_dimension=lhs_spec[0],
    input_feature_dimension=lhs_spec[1],
    input_spatial_dimensions=list(lhs_spec[2:]),
    kernel_output_feature_dimension=rhs_spec[0],
    kernel_input_feature_dimension=rhs_spec[1],
    kernel_spatial_dimensions=list(rhs_spec[2:]),
    output_batch_dimension=out_spec[0],
    output_feature_dimension=out_spec[1],
    output_spatial_dimensions=list(out_spec[2:]))
  num_spatial_dims = len(rhs_spec) - 2
  window_reversal = _dense_bool_elements([False] * num_spatial_dims)

  return mhlo.ConvOp(aval_to_ir_type(aval_out), lhs, rhs,
                     _dense_int_elements(window_strides),
                     _dense_int_elements(padding),
                     _dense_int_elements(lhs_dilation),
                     _dense_int_elements(rhs_dilation),
                     window_reversal,
                     dnums, _i64_attr(feature_group_count),
                     _i64_attr(batch_group_count),
                     _precision_attr(precision)).results

translations[lax.conv_general_dilated_p] = _conv_general_dilated_lower
platform_specific_translations['cpu'][lax.conv_general_dilated_p] = partial(
    _conv_general_dilated_lower, expand_complex_convolutions=True)
platform_specific_translations['gpu'][lax.conv_general_dilated_p] = partial(
    _conv_general_dilated_lower, expand_complex_convolutions=True)


def _integer_pow(x, *, y):
  # This should be kept in sync with the jax2tf translation rule.
  if y == 0:
    return lax.full_like(x, 1)
  is_reciprocal = y < 0
  if is_reciprocal:
    y = -y
  acc = None
  while y > 0:
    if y & 1:
      acc = x if acc is None else lax.mul(acc, x)
    y >>= 1
    if y > 0:
      # We don't call lax.square because it calls lax.integer_pow.
      x = lax.mul(x, x)
  return lax.div(lax.full_like(acc, 1), acc) if is_reciprocal else acc

translations[lax.integer_pow_p] = lower_fun(
            _integer_pow, multiple_results=False)


def _round_lower(ctx, avals_in, avals_out, x, *, rounding_method):
  if rounding_method is lax.RoundingMethod.AWAY_FROM_ZERO:
    return mhlo.RoundOp(x).results
  else:
    assert rounding_method is lax.RoundingMethod.TO_NEAREST_EVEN
    round_nearest = lower_fun(lax._round_to_nearest_even,
                              multiple_results=False)
    return round_nearest(ctx, avals_in, avals_out, x)
translations[lax.round_p] = _round_lower


# iota_p
def _iota_lower(ctx, avals_in, avals_out, *, dtype, shape, dimension):
  del dtype, shape
  aval_out, = avals_out
  return mhlo.IotaOp(aval_to_ir_type(aval_out), _i64_attr(dimension)).results
translations[lax.iota_p] = _iota_lower

# Array origami

def _broadcast_in_dim_lower(ctx, avals_in, avals_out, x, *, shape,
                            broadcast_dimensions):
  del shape
  aval_out, = avals_out
  return mhlo.BroadcastInDimOp(
      aval_to_ir_type(aval_out), x, _dense_int_elements(broadcast_dimensions)
  ).results
translations[lax.broadcast_in_dim_p] = _broadcast_in_dim_lower


def _concatenate_lower(ctx, avals_in, avals_out, *xs, dimension):
  return mhlo.ConcatenateOp(xs, _i64_attr(dimension)).results
translations[lax.concatenate_p] = _concatenate_lower


def _dynamic_slice_lower(ctx, avals_in, avals_out, x, *start_indices,
                         slice_sizes):
  aval_out, = avals_out
  return mhlo.DynamicSliceOp(aval_to_ir_type(aval_out), x,
                             start_indices,
                             _dense_int_elements(slice_sizes)).results

translations[lax.dynamic_slice_p] = _dynamic_slice_lower


def _dynamic_update_slice_lower(ctx, avals_in, avals_out, x, update,
                                *start_indices):
  aval_out, = avals_out
  return mhlo.DynamicUpdateSliceOp(aval_to_ir_type(aval_out), x, update,
                                   start_indices).results

translations[lax.dynamic_update_slice_p] = _dynamic_update_slice_lower


def _gather_lower(ctx, avals_in, avals_out, operand, indices, *,
                  dimension_numbers, slice_sizes, unique_indices,
                  indices_are_sorted, mode, fill_value):
  aval_out, = avals_out
  if mode == lax.GatherScatterMode.FILL_OR_DROP:
    gather_fill_fn = lower_fun(lax._gather_fill, multiple_results=False)
    return gather_fill_fn(
        ctx, avals_in, avals_out, operand, indices,
        dimension_numbers=dimension_numbers, slice_sizes=slice_sizes,
        unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
        fill_value=fill_value, output_shape=aval_out.shape)

  assert mode in (lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                  lax.GatherScatterMode.CLIP), mode
  dnums = mhlo.GatherDimensionNumbers.get(
    collapsed_slice_dims=list(dimension_numbers.collapsed_slice_dims),
    index_vector_dim=len(avals_in[1].shape) - 1,
    offset_dims=list(dimension_numbers.offset_dims),
    start_index_map=list(dimension_numbers.start_index_map))
  return mhlo.GatherOp(operand, indices, dnums,
                       _dense_int_elements(slice_sizes),
                       ir.BoolAttr.get(indices_are_sorted)).results

translations[lax.gather_p] = _gather_lower



def _scatter_lower(ctx, avals_in, avals_out, operand, indices, updates, *,
                   update_jaxpr, update_consts, dimension_numbers,
                   indices_are_sorted, unique_indices, mode):
  if mode == lax.GatherScatterMode.CLIP:
    clip_fn = lower_fun(lax._clamp_scatter_indices, multiple_results=False)
    (indices,), = clip_fn(ctx, avals_in, None, operand, indices, updates,
                          dnums=dimension_numbers)

  aval_out, = avals_out
  dnums = dimension_numbers
  scatter_dnums = mhlo.ScatterDimensionNumbers.get(
    update_window_dims=list(dnums.update_window_dims),
    inserted_window_dims=list(dnums.inserted_window_dims),
    scattered_dims_to_operand_dims=list(dnums.scatter_dims_to_operand_dims),
    index_vector_dim=len(avals_in[1].shape) - 1)
  op = mhlo.ScatterOp(aval_to_ir_type(aval_out), operand, indices, updates,
                      scatter_dnums, ir.BoolAttr.get(indices_are_sorted),
                      ir.BoolAttr.get(unique_indices))
  scalar_type = aval_to_ir_type(core.ShapedArray((), aval_out.dtype))
  update = op.update_computation.blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(update):
    ctx = ctx.replace(name_stack='')
    out_nodes = jaxpr_subcomp(
        ctx, update_jaxpr, update_consts,
        (update.arguments[0],), (update.arguments[1],))
    mhlo.ReturnOp(util.flatten(out_nodes))
  return op.results

translations[lax.scatter_p] = _scatter_lower
translations[lax.scatter_add_p] = _scatter_lower
translations[lax.scatter_mul_p] = _scatter_lower
translations[lax.scatter_min_p] = _scatter_lower
translations[lax.scatter_max_p] = _scatter_lower


def _scatter_add_lower_gpu(ctx, avals_in, avals_out, operand, indices, updates, *,
                           update_jaxpr, update_consts, dimension_numbers,
                           indices_are_sorted, unique_indices, mode):
  if operand.dtype != np.complex128:
    return _scatter_lower(ctx, avals_in, avals_out, operand, indices, updates,
                          update_jaxpr=update_jaxpr,
                          update_consts=update_consts,
                          dimension_numbers=dimension_numbers,
                          indices_are_sorted=indices_are_sorted,
                          unique_indices=unique_indices, mode=mode)
  assert mode == lax.GatherScatterMode.PROMISE_IN_BOUNDS, mode
  _, _, updates_aval_in = avals_in
  aval_out, = avals_out
  dnums = dimension_numbers
  scatter_dnums = mhlo.ScatterDimensionNumbers.get(
    update_window_dims=list(dnums.update_window_dims),
    inserted_window_dims=list(dnums.inserted_window_dims),
    scattered_dims_to_operand_dims=list(dnums.scatter_dims_to_operand_dims),
    index_vector_dim=len(avals_in[1].shape) - 1)
  real_dtype = _real_dtype(aval_out.dtype)
  operand_type_part = aval_to_ir_type(
      core.ShapedArray(aval_out.shape, real_dtype))
  updates_type_part = aval_to_ir_type(
      core.ShapedArray(updates_aval_in.shape, real_dtype))

  def _scatter(operand_part, updates_part):
    scatter = mhlo.ScatterOp(operand_type_part, operand_part, indices,
                             updates_part, scatter_dnums,
                             ir.BoolAttr.get(indices_are_sorted),
                             ir.BoolAttr.get(unique_indices))
    scalar_type = aval_to_ir_type(core.ShapedArray((), real_dtype))
    reducer = scatter.regions[0].blocks.append(scalar_type, scalar_type)
    with ir.InsertionPoint(reducer):
      add = mhlo.AddOp(scalar_type, *reducer.arguments).result
      mhlo.ReturnOp([add])
    return scatter.result

  real = _scatter(mhlo.RealOp(operand_type_part, operand).result,
                  mhlo.RealOp(updates_type_part, updates).result)
  imag = _scatter(mhlo.ImagOp(operand_type_part, operand).result,
                  mhlo.ImagOp(updates_type_part, updates).result)
  return mhlo.ComplexOp(aval_to_ir_type(aval_out), real, imag).results

platform_specific_translations["gpu"][lax.scatter_add_p] = _scatter_add_lower_gpu


def _pad_lower(ctx, avals_in, avals_out, x, padding_value, *, padding_config):
  aval_out, = avals_out
  low, high, interior = util.unzip3(padding_config)
  return mhlo.PadOp(aval_to_ir_type(aval_out), x, padding_value,
                    _dense_int_elements(low),
                    _dense_int_elements(high),
                    _dense_int_elements(interior)).results
translations[lax.pad_p] = _pad_lower


def _reshape_lower(ctx, avals_in, avals_out, x, *, new_sizes, dimensions):
  aval_in, = avals_in
  aval_out, = avals_out
  if dimensions is not None:
    aval = core.ShapedArray(np.take(aval_in.shape, dimensions), aval_in.dtype)
    x = mhlo.TransposeOp(aval_to_ir_type(aval), x,
                         _dense_int_elements(dimensions)).result
  return mhlo.ReshapeOp(aval_to_ir_type(aval_out), x).results
translations[lax.reshape_p] = _reshape_lower

def _rev_lower(ctx, avals_in, avals_out, x, *, dimensions):
  return mhlo.ReverseOp(x, _dense_int_elements(dimensions)).results
translations[lax.rev_p] = _rev_lower

def _slice_lower(ctx, avals_in, avals_out, x, *, start_indices,
                 limit_indices, strides):
  aval_out, = avals_out
  strides = strides or [1] * len(start_indices)
  return mhlo.SliceOp(x,
                      _dense_int_elements(start_indices),
                      _dense_int_elements(limit_indices),
                      _dense_int_elements(strides)).results

translations[lax.slice_p] = _slice_lower

def _squeeze_lower(ctx, avals_in, avals_out, operand, *, dimensions):
  del dimensions  # Implied by the output aval.
  aval_out, = avals_out
  return mhlo.ReshapeOp(aval_to_ir_type(aval_out), operand).results

translations[lax.squeeze_p] = _squeeze_lower

def _transpose_lower(ctx, avals_in, avals_out, x, *, permutation):
  aval_out, = avals_out
  return mhlo.TransposeOp(aval_to_ir_type(aval_out), x,
                          _dense_int_elements(permutation)).results
translations[lax.transpose_p] = _transpose_lower


# Reductions

def _unary_reduce_lower(reducer, unit_factory, ctx, avals_in, avals_out, x, *,
                        axes):
  aval_out, = avals_out
  dtype = aval_out.dtype
  op = mhlo.ReduceOp([aval_to_ir_type(aval_out)], [x],
                     ir_constants(unit_factory(aval_out.dtype)),
                     _dense_int_elements(axes))
  scalar_type = aval_to_ir_type(core.ShapedArray((), dtype))
  reducer_region = op.regions[0].blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(reducer_region):
    add = reducer(*reducer_region.arguments)
    mhlo.ReturnOp(add.results)
  return op.results

translations[lax.reduce_sum_p] = partial(_unary_reduce_lower, mhlo.AddOp,
                                         lambda dtype: np.array(0, dtype))
translations[lax.reduce_prod_p] = partial(_unary_reduce_lower, mhlo.MulOp,
                                          lambda dtype: np.array(1, dtype))
translations[lax.reduce_or_p] = partial(_unary_reduce_lower, mhlo.OrOp,
                                         lambda dtype: np.array(False, dtype))
translations[lax.reduce_and_p] = partial(_unary_reduce_lower, mhlo.AndOp,
                                          lambda dtype: np.array(True, dtype))
translations[lax.reduce_min_p] = partial(_unary_reduce_lower, _min_op,
                                         lax._get_min_identity)
translations[lax.reduce_max_p] = partial(_unary_reduce_lower, _max_op,
                                         lax._get_max_identity)


def _reduce_lower(ctx, avals_in, avals_out, *values, computation, jaxpr,
                  consts, dimensions):
  assert all(isinstance(x, core.ShapedArray) for x in avals_in), avals_in
  operands, init_values = util.split_list(values, [len(values) // 2])
  init_value_avals = avals_in[len(values) // 2:]
  op = mhlo.ReduceOp([aval_to_ir_type(aval) for aval in avals_out], operands,
                     init_values, _dense_int_elements(dimensions))
  ir_types = [aval_to_ir_type(aval) for aval in init_value_avals]
  reducer = op.regions[0].blocks.append(*(ir_types + ir_types))
  with ir.InsertionPoint(reducer):
    ctx = ctx.replace(name_stack='')
    out_nodes = jaxpr_subcomp(ctx, jaxpr, consts,
                              *([a] for a in reducer.arguments))
    mhlo.ReturnOp(util.flatten(out_nodes))
  return op.results

translations[lax.reduce_p] = _reduce_lower

translations[lax.argmin_p] = lower_fun(
  partial(lax._compute_argminmax, lax.lt, lax._get_min_identity),
  multiple_results=False)

translations[lax.argmax_p] = lower_fun(
  partial(lax._compute_argminmax, lax.gt, lax._get_max_identity),
  multiple_results=False)


def _generic_reduce_window_lower(ctx, avals_in, avals_out, *args, jaxpr, consts,
                                 window_dimensions, window_strides, padding,
                                 base_dilation, window_dilation):
  operands, init_values = util.split_list(args, [len(args) // 2])
  _, init_value_avals = util.split_list(avals_in, [len(operands)])
  scalar_types = [aval_to_ir_type(aval) for aval in init_value_avals]
  rw = mhlo.ReduceWindowOp(
      map(aval_to_ir_type, avals_out), operands, init_values,
      _dense_int_elements(window_dimensions),
      _dense_int_elements(window_strides), _dense_int_elements(base_dilation),
      _dense_int_elements(window_dilation),
      ir.DenseIntElementsAttr.get(np.asarray(padding, np.int64)))
  reducer = rw.regions[0].blocks.append(*(scalar_types + scalar_types))
  with ir.InsertionPoint(reducer):
    out_nodes = jaxpr_subcomp(ctx, jaxpr, consts,
                              *([a] for a in reducer.arguments))
    mhlo.ReturnOp(util.flatten(out_nodes))
  return rw.results

translations[lax_windowed_reductions.reduce_window_p] = _generic_reduce_window_lower


def _reduce_window_lower(
    reduce_op, init_value, ctx, avals_in, avals_out, operand, *,
    window_dimensions, window_strides, padding, base_dilation, window_dilation):
  aval_out, = avals_out
  operand_aval, = avals_in
  scalar_aval = operand_aval.update(shape=())
  scalar_type = aval_to_ir_type(scalar_aval)
  rw = mhlo.ReduceWindowOp(
      aval_to_ir_types(aval_out), [operand],
      [_full_like_aval(init_value(scalar_aval.dtype), scalar_aval)],
      _dense_int_elements(window_dimensions),
      _dense_int_elements(window_strides), _dense_int_elements(base_dilation),
      _dense_int_elements(window_dilation),
      ir.DenseIntElementsAttr.get(np.asarray(padding, np.int64)))
  reducer = rw.regions[0].blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(reducer):
    mhlo.ReturnOp(reduce_op(*reducer.arguments))
  return rw.results

translations[lax_windowed_reductions.reduce_window_sum_p] = partial(
    _reduce_window_lower, mhlo.AddOp, lambda _: 0)
translations[lax_windowed_reductions.reduce_window_min_p] = partial(
    _reduce_window_lower, mhlo.MinOp, lax._get_min_identity)
translations[lax_windowed_reductions.reduce_window_max_p] = partial(
    _reduce_window_lower, mhlo.MaxOp, lax._get_max_identity)


def _select_and_scatter_lower(
    ctx, avals_in, avals_out, operand, source, init_value, *, select_jaxpr,
    select_consts, scatter_jaxpr, scatter_consts, window_dimensions,
    window_strides, padding):
  operand_aval, source_aval, init_value_aval = avals_in
  aval_out, = avals_out
  scalar_aval = operand_aval.update(shape=())
  scalar_type = aval_to_ir_type(scalar_aval)
  op = mhlo.SelectAndScatterOp(
      aval_to_ir_type(aval_out), operand, source,
      init_value, _dense_int_elements(window_dimensions),
      _dense_int_elements(window_strides),
      ir.DenseIntElementsAttr.get(np.asarray(padding, np.int64)))
  select = op.select.blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(select):
    out_nodes = jaxpr_subcomp(ctx, select_jaxpr, select_consts,
                              *([a] for a in select.arguments))
    mhlo.ReturnOp(util.flatten(out_nodes))
  scatter = op.scatter.blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(scatter):
    out_nodes = jaxpr_subcomp(ctx, scatter_jaxpr, scatter_consts,
                              *([a] for a in scatter.arguments))
    mhlo.ReturnOp(util.flatten(out_nodes))
  return op.results

translations[lax_windowed_reductions.select_and_scatter_p] = _select_and_scatter_lower


def _select_and_scatter_add(source, operand, *, select_prim, window_dimensions,
                            window_strides, padding, expand_padding):
  dtype = source.dtype
  select = lambda x, y: select_prim.bind(x, y)
  scatter = lax.bitwise_or if dtype == np.bool_ else lax.add
  if expand_padding:
    operand_shape = operand.shape
    original_padding = padding
    identity = (lax._get_max_identity if select_prim is lax.ge_p
                else lax._get_min_identity)
    pads = [(lo, hi, 0) for (lo, hi) in padding]
    operand = lax.pad(operand, identity(dtype), pads)
    padding = [(0, 0) for _ in padding]
  out = lax._select_and_scatter(operand, select, window_dimensions,
                                window_strides, padding, source,
                                lax._zero(operand), scatter)
  if expand_padding:
    start_indices = [lo for (lo, hi) in original_padding]
    stop_indices = [lo + d for ((lo, hi), d) in zip(original_padding,
                                                    operand_shape)]
    out = lax.slice(out, start_indices, stop_indices)
  return out

translations[lax_windowed_reductions.select_and_scatter_add_p] = lower_fun(
    partial(_select_and_scatter_add, expand_padding=False),
    multiple_results=False)
platform_specific_translations['cpu'][lax_windowed_reductions.select_and_scatter_add_p] = lower_fun(
    partial(_select_and_scatter_add, expand_padding=True),
    multiple_results=False)
platform_specific_translations['gpu'][lax_windowed_reductions.select_and_scatter_add_p] = lower_fun(
    partial(_select_and_scatter_add, expand_padding=True),
    multiple_results=False)


def _sort_lower(ctx, avals_in, avals_out, *operands, dimension, is_stable,
                num_keys):
  assert all(isinstance(x, core.ShapedArray) for x in avals_in), avals_in
  sort = mhlo.SortOp([aval_to_ir_type(aval) for aval in avals_out],
                     _flatten_lowering_ir_args(operands), _i64_attr(dimension),
                     ir.BoolAttr.get(is_stable))
  scalar_avals = [aval.update(shape=()) for aval in avals_in]
  scalar_types = map(aval_to_ir_type, scalar_avals)
  comparator = sort.comparator.blocks.append(
      *util.flatten(zip(scalar_types, scalar_types)))
  with ir.InsertionPoint(comparator):
    lower_comparator = lower_fun(partial(lax._sort_lt_comparator),
                                 multiple_results=False)
    out = lower_comparator(ctx, util.flatten(zip(scalar_avals, scalar_avals)),
                           [core.ShapedArray((), np.bool_)],
                           *[[a] for a in comparator.arguments],
                           num_keys=num_keys)
    mhlo.ReturnOp(util.flatten(out))
  return sort.results

translations[lax.sort_p] = _sort_lower


def _create_token_lowering(ctx, avals_in, avals_out, *operands):
  aval_out, = avals_out
  return mhlo.CreateTokenOp(aval_to_ir_type(aval_out)).results

translations[lax.create_token_p] = _create_token_lowering


def _after_all_lowering(ctx, avals_in, avals_out, *operands):
  aval_out, = avals_out
  return mhlo.AfterAllOp(aval_to_ir_type(aval_out), operands).results

translations[lax.after_all_p] = _after_all_lowering

def _infeed_lowering(ctx, avals_in, avals_out, token, *, shapes, partitions):
  assert partitions is None, partitions  # TODO(phawkins): implement me.
  output_types = map(aval_to_ir_types, avals_out[:-1])
  flat_output_types = util.flatten(output_types)
  output_tuple_type = ir.TupleType.get_tuple(flat_output_types)
  # TODO(phawkins): verify `shapes` have a major-to-minor layout.
  layouts = ir.ArrayAttr.get([
      ir.ArrayAttr.get(
          [ir.ArrayAttr.get(
              [_i64_attr(i) for i in range(len(aval.shape) - 1, -1, -1)])
           for aval in shapes]),
      ir.UnitAttr.get(),
  ])
  output_and_token_tuple_type = ir.TupleType.get_tuple(
      [output_tuple_type, mhlo.TokenType.get()])
  outs_and_token = mhlo.InfeedOp(
      output_and_token_tuple_type, token, ir.StringAttr.get(""),
      layouts).result
  outs_tuple = mhlo.GetTupleElementOp(output_tuple_type, outs_and_token,
                                      _i32_attr(0)).result
  token = mhlo.GetTupleElementOp(mhlo.TokenType.get(), outs_and_token,
                                 _i32_attr(1)).result
  outs = [mhlo.GetTupleElementOp(typ, outs_tuple, _i32_attr(i)).result
          for i, typ in enumerate(flat_output_types)]
  return util.unflatten(outs, map(len, output_types)) + [[token,]]

translations[lax.infeed_p] = _infeed_lowering

def _outfeed_lowering(ctx, avals_in, avals_out, token, *xs, partitions):
  assert partitions is None, partitions  # TODO(phawkins): implement me.
  token_aval = avals_in[0]
  xs_avals = avals_in[1:]
  input_types = map(aval_to_ir_types, xs_avals)
  flat_input_types = util.flatten(input_types)
  input_tuple_type = ir.TupleType.get_tuple(flat_input_types)
  tup = mhlo.TupleOp(input_tuple_type, _flatten_lowering_ir_args(xs)).result
  return mhlo.OutfeedOp(aval_to_ir_type(token_aval), tup, token,
                        ir.StringAttr.get("")).results

translations[lax.outfeed_p] = _outfeed_lowering


def _rng_uniform_lowering(ctx, avals_in, avals_out, a, b, *, shape):
  aval_out, = avals_out
  shape, = ir_constants(np.array(aval_out.shape, np.int64),
                        canonicalize_types=False)
  return mhlo.RngUniformOp(a, b, shape).results

translations[lax.rng_uniform_p] = _rng_uniform_lowering


# def _rng_bit_generator_lower(
#     ctx, avals_in, avals_out, key, *, shape, dtype, algorithm):
#   key_aval, = avals_in
#   c = ctx.builder
#   backend_is_gpu = ctx.platform == "gpu"
#   key_shape, key_dtype = c.get_shape(key).dimensions(), c.get_shape(key).numpy_dtype()
#   # While the RngBitGenerator HLO accepts a u64[2] key on all backends, we
#   # typically represent the key argument to this primitive as a u32[4] so as to
#   # sidestep issues with the jax_enable_x64=False configuration. As a result, we
#   # need to convert u32[4] -> u64[2] here in the translation rule. However, we
#   # also polymorphically allow a u64[2] for backward compatibility.
#   assert ((key_aval.shape == (4,) and key_aval.dtype == np.dtype('uint32')) or
#           (key_aval.shape == (2,) and key_aval.dtype == np.dtype('uint64'))), key_aval.shape
#   xla_shape = xc.Shape.array_shape(np.dtype(dtype), shape)
#   if key_dtype == np.dtype('uint32'):
#     # TODO(mattjj): the BitcastConvertType segfaults on GPU
#     u64_etype = xla.dtype_to_primitive_type(np.dtype('uint64'))
#     key = xops.BitcastConvertType(xops.Reshape(key, (2, 2)), u64_etype)
#   out_key, out_vals = xla.xla_destructure(
#       c, xops.RngBitGenerator(algorithm, key, xla_shape))
#   if key_dtype == np.dtype('uint32'):
#     u32_etype = xla.dtype_to_primitive_type(np.dtype('uint32'))
#     out_key = xops.Reshape(xops.BitcastConvertType(out_key, u32_etype), (4,))
#   return [out_key, out_vals]



translations[prng.threefry2x32_p] = lower_fun(
    partial(prng._threefry2x32_lowering, use_rolled_loops=False),
    multiple_results=True)

# TODO(phawkins): add CPU and GPU specializations of threefry2x32_p



def _cond_lowering(ctx, avals_in, avals_out, index, *args, branches, linear):
  del linear  # Unused.
  arg_avals = avals_in[1:]
  input_types = map(aval_to_ir_types, arg_avals)
  output_types = map(aval_to_ir_types, avals_out)
  flat_input_types = util.flatten(input_types)
  flat_output_types = util.flatten(output_types)
  input_tuple_type = ir.TupleType.get_tuple(flat_input_types)
  output_tuple_type = ir.TupleType.get_tuple(flat_output_types)
  op = mhlo.TupleOp(input_tuple_type, _flatten_lowering_ir_args(args)).result
  # TODO(phawkins): avoid build_generic when CaseOp is fixed.
  case_op = mhlo.CaseOp.build_generic([output_tuple_type],
                                      [index] + [op] * len(branches),
                                      regions=len(branches))
  for i, jaxpr in enumerate(branches):
    branch = case_op.regions[i].blocks.append(input_tuple_type)
    with ir.InsertionPoint(branch):
      args = [mhlo.GetTupleElementOp(input_type, branch.arguments[0],
                                     _i32_attr(i)).result
              for i, input_type in enumerate(flat_input_types)]
      unflattened_args = util.unflatten(args, map(len, input_types))
      out_vals = jaxpr_subcomp(ctx, jaxpr.jaxpr, jaxpr.consts,
                               *unflattened_args)
      out = mhlo.TupleOp(output_tuple_type, util.flatten(out_vals)).results
      mhlo.ReturnOp(out)

  results = [mhlo.GetTupleElementOp(output_type, case_op.result,
                                    _i32_attr(i)).result
             for i, output_type in enumerate(flat_output_types)]
  return util.unflatten(results, map(len, output_types))

translations[control_flow.cond_p] = _cond_lowering

translations[control_flow.scan_p] = lower_fun(
    control_flow._scan_impl, multiple_results=True)

def _pred_bcast_select(pred_aval: core.ShapedArray,
                       pred: ir.Value, xs: Sequence[ir.Value],
                       ys: Sequence[ir.Value],
                       x_y_aval: core.AbstractValue) -> Sequence[ir.Value]:
  if x_y_aval is core.abstract_unit:
    return []
  elif x_y_aval is core.abstract_token:
    x, = xs
    y, = ys
    return [mhlo.AfterAllOp(aval_to_ir_type(x_y_aval), [x, y]).result]
  else:
    assert isinstance(x_y_aval, core.ShapedArray), x_y_aval
    x, = xs
    y, = ys
    assert x.type == y.type, (x.type, y.type)
    assert (pred_aval.shape == x_y_aval.shape[:len(pred_aval.shape)]), (
            pred_aval.shape, x_y_aval)
    bcast_pred = mhlo.BroadcastInDimOp(
        aval_to_ir_type(x_y_aval.update(dtype=np.dtype(np.bool_))),
        pred, _dense_int_elements(list(range(len(pred_aval.shape))))).result
    return mhlo.SelectOp(bcast_pred, x, y).results


def _while_lowering(ctx, avals_in, avals_out, *args, cond_jaxpr,
                    body_jaxpr, cond_nconsts, body_nconsts):
  pred_aval = cond_jaxpr.out_avals[0]
  batched = bool(pred_aval.shape)

  # Since jaxprs don't have tuples and have multiple return values, but we need
  # the HLO While loop to take a single tuple input and output a single boolean
  # (for the cond computation) or a single tuple output (for the body
  # computation), we build XLA computations that handle the tuple munging before
  # generating a Call into the computations formed from the jaxprs.

  loop_carry_types = map(aval_to_ir_types, avals_in)
  flat_loop_carry_types = util.flatten(loop_carry_types)
  loop_carry_tuple_type = ir.TupleType.get_tuple(flat_loop_carry_types)

  flat_args = _flatten_lowering_ir_args(args)
  init_carry = mhlo.TupleOp(loop_carry_tuple_type, flat_args)
  while_op = mhlo.WhileOp([loop_carry_tuple_type], [init_carry.result])

  # Loop condition
  cond_block = while_op.regions[0].blocks.append(loop_carry_tuple_type)
  with ir.InsertionPoint(cond_block):
    flat_cond_args = [
        mhlo.GetTupleElementOp(input_type, cond_block.arguments[0],
                               _i32_attr(i)).result
        for i, input_type in enumerate(flat_loop_carry_types)]
    cond_args = util.unflatten(flat_cond_args, map(len, loop_carry_types))
    x, _, z = util.split_list(cond_args, [cond_nconsts, body_nconsts])
    cond_ctx = ctx.replace(
        name_stack=xla.extend_name_stack(ctx.name_stack, 'cond'))
    (pred,), = jaxpr_subcomp(cond_ctx, cond_jaxpr.jaxpr,
                             map(ir_constants, cond_jaxpr.consts), *(x + z))
    if batched:
      pred, = _unary_reduce_lower(
          mhlo.OrOp, lambda dtype: np.array(False, dtype), ctx, [pred_aval],
          [pred_aval.update(shape=())], pred,
          axes=tuple(range(len(pred_aval.shape))))
    mhlo.ReturnOp([pred])

  # Loop body
  body_block = while_op.regions[1].blocks.append(loop_carry_tuple_type)
  with ir.InsertionPoint(body_block):
    flat_body_args = [
        mhlo.GetTupleElementOp(input_type, body_block.arguments[0],
                               _i32_attr(i)).result
        for i, input_type in enumerate(flat_loop_carry_types)]
    body_args = util.unflatten(flat_body_args, map(len, loop_carry_types))
    x, y, z = util.split_list(body_args, [cond_nconsts, body_nconsts])
    body_ctx = ctx.replace(
        name_stack=xla.extend_name_stack(ctx.name_stack, 'body'))
    new_z = jaxpr_subcomp(body_ctx, body_jaxpr.jaxpr,
                          map(ir_constants, body_jaxpr.consts), *(y + z))
    if batched:
      body_pred_ctx = ctx.replace(
          name_stack=xla.extend_name_stack(ctx.name_stack, 'body_pred'))
      (body_pred,), = jaxpr_subcomp(
          body_pred_ctx, cond_jaxpr.jaxpr, map(ir_constants, cond_jaxpr.consts),
          *(x + z))
      new_z = map(partial(_pred_bcast_select, pred_aval, body_pred), new_z, z,
                   body_jaxpr.out_avals)

    new_carry = mhlo.TupleOp(
        loop_carry_tuple_type,
        [*util.flatten(x), *util.flatten(y), *util.flatten(new_z)])
    mhlo.ReturnOp([new_carry.result])

  outputs = util.unflatten([
    mhlo.GetTupleElementOp(output_type, while_op.result, _i32_attr(i)).result
    for i, output_type in enumerate(flat_loop_carry_types)
  ], map(len, loop_carry_types))
  _,  _, z = util.split_list(outputs, [cond_nconsts, body_nconsts])
  return z

translations[control_flow.while_p] = _while_lowering


def _add_cumulative_reduce(prim, reducer, tpu_reduce_window_fn):
  translations[prim] = lower_fun(
      partial(control_flow.associative_scan, reducer), multiple_results=False)
  platform_specific_translations['tpu'] = lower_fun(
      partial(control_flow._cumred_tpu_translation_rule, tpu_reduce_window_fn),
      multiple_results=False)

_add_cumulative_reduce(control_flow.cumsum_p, lax.add,
                       lax_windowed_reductions._reduce_window_sum)
_add_cumulative_reduce(control_flow.cumprod_p, lax.mul,
                       lax_windowed_reductions._reduce_window_prod)
_add_cumulative_reduce(control_flow.cummin_p, lax.min,
                       lax_windowed_reductions._reduce_window_min)
_add_cumulative_reduce(control_flow.cummax_p, lax.max,
                       lax_windowed_reductions._reduce_window_max)

translations[custom_derivatives.custom_jvp_call_jaxpr_p] = lower_fun(
    custom_derivatives._custom_jvp_call_jaxpr_impl, multiple_results=True)
translations[custom_derivatives.custom_vjp_call_jaxpr_p] = lower_fun(
    custom_derivatives._custom_vjp_call_jaxpr_impl, multiple_results=True)
translations[custom_derivatives.linear_call_p] = lower_fun(
    custom_derivatives._linear_call_impl, multiple_results=True)
translations[ad.custom_lin_p] = ad._raise_custom_vjp_error_on_jvp


def _dummy_like_aval(aval: core.AbstractValue) -> Sequence[ir.Value]:
  if isinstance(aval, core.ShapedArray):
    return [_full_like_aval(0, aval)]
  elif isinstance(aval, core.AbstractToken):
    return mhlo.CreateTokenOp(aval_to_ir_type(aval)).results
  elif isinstance(aval, core.AbstractUnit):
    return ()
  else:
    raise TypeError(f"Unsupported abstract value {aval}")

def _remat_using_while(ctx, avals_in, avals_out, *args, name, call_jaxpr):
  input_types = map(aval_to_ir_types, avals_in)
  output_types = map(aval_to_ir_types, avals_out)
  flat_output_types = util.flatten(output_types)
  int32_scalar_type = aval_to_ir_type(core.ShapedArray((), np.dtype(np.int32)))
  loop_carry_types = [(int32_scalar_type,)] + input_types + output_types
  flat_loop_carry_types = util.flatten(loop_carry_types)
  counter_init = ir_constants(np.array(0, np.int32))
  flat_args = _flatten_lowering_ir_args(
      (counter_init,) + args +
      tuple(_dummy_like_aval(aval) for aval in avals_out))
  loop_carry_tuple_type = ir.TupleType.get_tuple(flat_loop_carry_types)
  init_carry = mhlo.TupleOp(loop_carry_tuple_type, flat_args)

  one, = ir_constants(np.array(1, np.int32))
  while_op = mhlo.WhileOp([loop_carry_tuple_type], [init_carry.result])

  # Loop condition
  cond_block = while_op.regions[0].blocks.append(loop_carry_tuple_type)
  with ir.InsertionPoint(cond_block):
    bool_scalar_type = aval_to_ir_type(core.ShapedArray((), np.dtype(np.bool_)))
    two, = ir_constants(np.array(2, np.int32))
    shape, = ir_constants(np.array((), np.int64), canonicalize_types=False)
    rng = mhlo.RngUniformOp(one, two, shape).result
    i = mhlo.GetTupleElementOp(int32_scalar_type, cond_block.arguments[0],
                               _i32_attr(0))
    cmp = mhlo.CompareOp(bool_scalar_type, i, rng, ir.StringAttr.get("LT"),
                         ir.StringAttr.get("SIGNED")).result
    mhlo.ReturnOp([cmp])

  body_block = while_op.regions[1].blocks.append(loop_carry_tuple_type)
  with ir.InsertionPoint(body_block):
    flat_body_args = [
        mhlo.GetTupleElementOp(input_type, body_block.arguments[0],
                               _i32_attr(i)).result
        for i, input_type in enumerate(flat_loop_carry_types)]
    body_args = util.unflatten(flat_body_args, map(len, loop_carry_types))
    ((i,),), y, _ = util.split_list(body_args, [1, len(avals_in)])
    body_ctx = ctx.replace(
        name_stack=xla.extend_name_stack(ctx.name_stack,
                                         xla.wrap_name(name, 'remat')))
    z = jaxpr_subcomp(body_ctx, call_jaxpr, (), *y)
    i_next = mhlo.AddOp(i, one).result
    new_carry = mhlo.TupleOp(
        loop_carry_tuple_type,
        [i_next, *util.flatten(y), *util.flatten(z)])
    mhlo.ReturnOp([new_carry.result])

  outputs = [mhlo.GetTupleElementOp(output_type, while_op.result,
                                    _i32_attr(1 + len(avals_in) + i)
                                   ).result
             for i, output_type in enumerate(flat_output_types)]
  return util.unflatten(outputs, map(len, output_types))


def _remat_lowering(ctx, avals_in, avals_out, *args,
                    name, call_jaxpr,
                    prevent_cse, differentiated, concrete,
                    policy, device=None):
  del device, concrete, policy  # Unused.
  if differentiated and prevent_cse:
    if True: # ctx.platform == "gpu":
      return _remat_using_while(ctx, avals_in, avals_out, *args, name=name,
                                call_jaxpr=call_jaxpr)
    else:
      assert False
      #return _remat_using_cond(ctx, args, name, call_jaxpr)
  else:
    return jaxpr_subcomp(
        ctx, call_jaxpr, (), *map(_wrap_singleton_ir_values, args))

translations[pe.remat_call_p] = _remat_lowering

def _fallback_lowering(prim: core.Primitive, ctx: LoweringContext,
                       avals_in, avals_out, *args, **params):
  xla_computation = xla.primitive_subcomputation(ctx.platform, prim, *avals_in,
                                                 **params)
  submodule_str = xc._xla.mlir.xla_computation_to_mlir_module(xla_computation)
  submodule = ir.Module.parse(submodule_str)
  callee_name = None
  for op in submodule.body.operations:
    ctx.module.body.append(op)
    if op.name.value == "main":
      callee_name = ir.StringAttr(ctx.symbol_table.insert(op)).value
      op.attributes["sym_visibility"] = ir.StringAttr.get("private")
    else:
      ctx.symbol_table.insert(op)

  output_types = map(aval_to_ir_types, avals_out)
  flat_output_types = util.flatten(output_types)
  output_type = (ir.TupleType.get_tuple(flat_output_types)
                 if prim.multiple_results else flat_output_types[0])

  call = std.CallOp([output_type], ir.FlatSymbolRefAttr.get(callee_name),
                    _flatten_lowering_ir_args(args)).result
  if not prim.multiple_results:
    return [call]
  flat_results = [mhlo.GetTupleElementOp(typ, call, _i32_attr(i)).result
                  for i, typ in enumerate(flat_output_types)]
  return util.unflatten(flat_results, map(len, output_types))

def add_fallback_lowering(prim: core.Primitive):
  translations[prim] = partial(_fallback_lowering, prim)


map(add_fallback_lowering, [
    # TODO(b/203775215): these are missing from the cHLO dialect. Either add
    # them or port them to Python.
    lax.igamma_p,
    lax.igammac_p,
    lax.igamma_grad_a,
    lax.random_gamma_grad_p,
    lax.bessel_i0e_p,
    lax.bessel_i1e_p,
    lax.erf_inv_p,
    lax.regularized_incomplete_beta_p,

    # CHLO doesn't have complex lowerings of these primitives (b/203718937)
    lax.acos_p,
    lax.acosh_p,
    lax.asin_p,
    lax.asinh_p,
    lax.atan_p,
    lax.atanh_p,
    lax.cosh_p,
    lax.tan_p,

    # CHLO doesn't have a legalization for bf16 (b/203774470)
    lax.erf_p,
    lax.erfc_p,

    # Not present in cHLO or mHLO (b/203798239), although we could just emit the
    # lowered pattern ourselves.
    lax.top_k_p,

    # TODO(phawkins): implement these lax ops:
    lax_windowed_reductions.select_and_gather_add_p,
    lax.rng_bit_generator_p,
])
