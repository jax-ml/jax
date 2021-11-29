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
import typing
from typing import (Any, Callable, Dict, List, Optional, Sequence, Type, Union,
                    Tuple)
from typing_extensions import Protocol

from jax import core
from jax import linear_util as lu
from jax._src import ad_util
from jax._src import device_array
from jax._src import dtypes
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import builtin
from jax._src.lib.mlir.dialects import chlo
from jax._src.lib.mlir.dialects import mhlo
from jax._src.lib.mlir.dialects import std
from jax._src.lib import xla_client as xc
from jax._src import source_info_util
import jax._src.util as util
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

def dense_int_elements(xs: Sequence[int]) -> ir.DenseIntElementsAttr:
  return ir.DenseIntElementsAttr.get(np.asarray(xs, np.int64))

def dense_bool_elements(xs: Sequence[bool]) -> ir.DenseElementsAttr:
  return ir.DenseElementsAttr.get(
      np.packbits(np.array(xs, np.bool_), bitorder='little'),
      type=ir.IntegerType.get_signless(1), shape=[len(xs)])

def i32_attr(i): return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), i)
def i64_attr(i): return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i)


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
        dense_int_elements(other_axes)).result
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
  class LoweringRule(Protocol):
    def __call__(self, ctx: LoweringContext,
                 avals_in: Sequence[core.AbstractValue],
                 avals_out: Sequence[core.AbstractValue],
                 *args: Union[ir.Value, Sequence[ir.Value]],
                 **kw) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
      """Converts a JAX primitive invocation into MLIR."""
else:
  LoweringRule = Any

_lowerings: Dict[core.Primitive, LoweringRule] = {}
_platform_specific_lowerings: Dict[str, Dict[core.Primitive, LoweringRule]]
_platform_specific_lowerings = collections.defaultdict(dict)

def register_lowering(prim: core.Primitive, rule: LoweringRule,
                      platform: Optional[str] = None):
  ts = (_lowerings if platform is None
        else _platform_specific_lowerings[platform])
  ts[prim] = rule


def _unwrap_singleton_ir_values(x): return x[0] if len(x) == 1 else x
def _wrap_singleton_ir_values(x: Union[ir.Value, Sequence[ir.Value]]
                             ) -> Sequence[ir.Value]:
  return (x,) if isinstance(x, ir.Value) else tuple(x)

def flatten_lowering_ir_args(
    xs: Sequence[Union[ir.Value, Sequence[ir.Value]]]
) -> Sequence[Sequence[ir.Value]]:
  return util.flatten(map(_wrap_singleton_ir_values, xs))

def lower_jaxpr_to_module(jaxpr: core.ClosedJaxpr, platform: str,
                          axis_env: xla.AxisEnv, name_stack: str) -> str:
  """Lowers a top-level jaxpr to an MHLO module.

  Handles the quirks of the argument/return value passing conventions of the
  runtime."""
  ctx = LoweringContext(platform, axis_env, name_stack)
  if platform == "iree":
    ctx = ctx.replace(tuple_results=False)
  with ctx.context, ir.Location.unknown(ctx.context):
    # TODO(phawkins): represent units with zero buffers at the runtime level.
    lower_jaxpr_to_fun(
        ctx, "main", jaxpr, public=True, replace_units_with_dummy=True,
        replace_tokens_with_dummy=True)

  ctx.module.operation.verify()
  output = io.StringIO()
  ctx.module.operation.print(file=output, #enable_debug_info=True,
                             print_generic_op_form=False)
  return output.getvalue()


def lower_jaxpr_to_fun(ctx: LoweringContext, name: str,
                       jaxpr: core.ClosedJaxpr, *,
                       public: bool = False,
                       replace_units_with_dummy: bool = False,
                       replace_tokens_with_dummy: bool = False) -> str:
  """Lowers jaxpr and its callees to an IR function.

  Assumes that an MLIR context, location, and insertion point are set.

  Args:
    ctx: the lowering context.
    name: the function name. The name will be uniquified by the symbol table,
      so it is ok to use the same name multiple times.
    jaxpr: the jaxpr to lower.
    public: if true, the function's visibility is set to "public".
    replace_units_with_dummy: if true, unit arguments/return values are
      replaced with bool arrays of size [0].
    replace_tokens_with_dummy: if true, token arguments/return values are
      replaced with bool arrays of size [0].
  Returns the name of the function.
  """
  def aval_to_types(aval):
    if replace_units_with_dummy and aval is core.abstract_unit:
      aval = core.ShapedArray((), np.dtype(np.bool_))
    elif replace_tokens_with_dummy and aval is core.abstract_token:
      aval = core.ShapedArray((), np.dtype(np.bool_))
    return aval_to_ir_types(aval)

  input_types = map(aval_to_types, jaxpr.in_avals)
  output_types = map(aval_to_types, jaxpr.out_avals)
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
    args: List[List[ir.Value]] = []
    for aval, arg in zip(jaxpr.in_avals, unflattened_args):
      if replace_units_with_dummy and aval is core.abstract_unit:
        args.append([])
      elif replace_tokens_with_dummy and aval is core.abstract_token:
        args.append(mhlo.CreateTokenOp(mhlo.TokenType.get()).results)
      else:
        args.append(arg)
    callee_name_stack = xla.extend_name_stack(ctx.name_stack,
                                              xla.wrap_name(name, 'jit'))
    out_vals = jaxpr_subcomp(ctx.replace(name_stack=callee_name_stack),
                             jaxpr.jaxpr, map(ir_constants, jaxpr.consts),
                             *args)
    outs = []
    for aval, out in zip(jaxpr.out_avals, out_vals):
      if replace_units_with_dummy and aval is core.abstract_unit:
        outs.append(ir_constants(np.zeros((), np.bool_)))
      elif replace_tokens_with_dummy and aval is core.abstract_token:
        outs.append(ir_constants(np.zeros((), np.bool_)))
      else:
        outs.append(out)
    flat_outputs = util.flatten(outs)
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
      if eqn.primitive in _platform_specific_lowerings[ctx.platform]:
        rule = _platform_specific_lowerings[ctx.platform][eqn.primitive]
      elif eqn.primitive in _lowerings:
        rule = _lowerings[eqn.primitive]
      elif eqn.primitive in xla._translations:
        rule = partial(xla_fallback_lowering, eqn.primitive)
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
                    flatten_lowering_ir_args(args))
  if ctx.tuple_results:
    flat_results = [
        mhlo.GetTupleElementOp(typ, call.result, i32_attr(i)).result
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

register_lowering(xla.xla_call_p, _xla_call_lower)

def _named_call_lowering(ctx, avals_in, avals_out, *args, name, backend=None,
                         call_jaxpr):
  return _call_lowering(name, name, call_jaxpr, backend, ctx, avals_in,
                        avals_out, *args)

register_lowering(core.named_call_p, _named_call_lowering)
register_lowering(core.call_p, partial(_named_call_lowering, name="core_call"))


def full_like_aval(value, aval: core.ShapedArray) -> ir.Value:
  """Returns an IR constant shaped full of `value` shaped like `aval`."""
  zero, = ir_constants(np.array(value, aval.dtype))
  return mhlo.BroadcastOp(aval_to_ir_type(aval), zero,
                          dense_int_elements(aval.shape)).result

def zeros_like_lowering(ctx, avals_in, avals_out, x):
  aval, = avals_in
  assert isinstance(aval, core.ShapedArray), aval
  return [full_like_aval(0, aval)]
register_lowering(ad_util.zeros_like_p, zeros_like_lowering)

def add_jaxvals_lowering(ctx, avals_in, avals_out, x, y):
  return mhlo.AddOp(x, y).results
register_lowering(ad_util.add_jaxvals_p, add_jaxvals_lowering)

register_lowering(ad_util.stop_gradient_p,
                  lambda ctx, avals_in, avals_out, x: [x])


# MLIR lowerings for lax primitives

def xla_fallback_lowering(prim: core.Primitive, ctx: LoweringContext,
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
                    flatten_lowering_ir_args(args)).result
  if not prim.multiple_results:
    return [call]
  flat_results = [mhlo.GetTupleElementOp(typ, call, i32_attr(i)).result
                  for i, typ in enumerate(flat_output_types)]
  return util.unflatten(flat_results, map(len, output_types))


register_lowering(ad.custom_lin_p, ad._raise_custom_vjp_error_on_jvp)


def _dummy_like_aval(aval: core.AbstractValue) -> Sequence[ir.Value]:
  if isinstance(aval, core.ShapedArray):
    return [full_like_aval(0, aval)]
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
  flat_args = flatten_lowering_ir_args(
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
                               i32_attr(0))
    cmp = mhlo.CompareOp(bool_scalar_type, i, rng, ir.StringAttr.get("LT"),
                         ir.StringAttr.get("SIGNED")).result
    mhlo.ReturnOp([cmp])

  body_block = while_op.regions[1].blocks.append(loop_carry_tuple_type)
  with ir.InsertionPoint(body_block):
    flat_body_args = [
        mhlo.GetTupleElementOp(input_type, body_block.arguments[0],
                               i32_attr(i)).result
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
                                    i32_attr(1 + len(avals_in) + i)
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

register_lowering(pe.remat_call_p, _remat_lowering)

# Lax ops missing MLIR lowerings.
# # TODO(b/203775215): these are missing from the cHLO dialect. Either add
# # them or port them to Python.
# lax.igamma_p,
# lax.igammac_p,
# lax.igamma_grad_a,
# lax.random_gamma_grad_p,
# lax.bessel_i0e_p,
# lax.bessel_i1e_p,
# lax.erf_inv_p,
# lax.regularized_incomplete_beta_p,

# # CHLO doesn't have complex lowerings of these primitives (b/203718937)
# lax.acos_p,
# lax.acosh_p,
# lax.asin_p,
# lax.asinh_p,
# lax.atan_p,
# lax.atanh_p,
# lax.cosh_p,
# lax.tan_p,

# # CHLO doesn't have a legalization for bf16 (b/203774470)
# lax.erf_p,
# lax.erfc_p,

# # Not present in cHLO or mHLO (b/203798239), although we could just emit the
# # lowered pattern ourselves.
# lax.top_k_p,

# # TODO(phawkins): implement these lax ops:
# lax.rng_bit_generator_p,
