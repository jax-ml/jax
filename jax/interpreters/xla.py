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

# Lowering of jaxprs into XLA (HLO) computations.

from collections import defaultdict
import dataclasses
import functools
from functools import partial
import itertools as it
import operator
import re
from typing import (Any, Callable, Dict, List, NamedTuple, Optional,
                    Sequence, Set, Type, Tuple, Union)
from typing_extensions import Protocol

import numpy as np

from jax.config import config
from jax import core
from jax._src import device_array
from jax._src import dtypes
from jax._src import source_info_util
from jax._src.abstract_arrays import (make_shaped_array, array_types)
from jax.core import (ConcreteArray, ShapedArray, str_eqn_compact)
import jax._src.pretty_printer as pp
from jax._src.util import (prod, new_name_stack, safe_zip, safe_map,
                           partition_list)

# TODO: update callers to refer to new location.
from jax._src.util import extend_name_stack as extend_name_stack  # noqa: F401
from jax._src.util import wrap_name as wrap_name  # noqa: F401

from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax.interpreters import partial_eval as pe
from jax.interpreters import ad

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

xe = xc._xla
xops = xc._xla.ops

# Types
Backend = xe.Client
Device = xc.Device
Buffer = xe.Buffer

XlaOp = xc.XlaOp
XlaShape = xc.Shape
XlaBuilder = xc.XlaBuilder
XlaExecutable = xc.Executable

# apply_primitive is defined in jax._src.dispatch.
apply_primitive: Callable
backend_compile: Callable
device_put: Callable

# TODO(phawkins): update code to point to new locations.
DeviceArray = device_array.DeviceArray
_DeviceArray = device_array._DeviceArray
_CppDeviceArray = xe.Buffer
make_device_array = device_array.make_device_array


def identity(x): return x

_scalar_types = dtypes.python_scalar_dtypes.keys()

def _make_array_shape(a: ShapedArray) -> Sequence[XlaShape]:
  if a.dtype == dtypes.float0:
    return (xc.Shape.array_shape(np.dtype('bool'), a.shape),)
  else:
    return (xc.Shape.array_shape(a.dtype, a.shape),)

def _get_canonical_source_file(frame: source_info_util.Frame):
  source_file = frame.file_name
  if config.jax_hlo_source_file_canonicalization_regex:
    source_file = re.sub(config.jax_hlo_source_file_canonicalization_regex,
                         '', source_file)
  return source_file

tracebacks = {}
def make_op_metadata(primitive: core.Primitive,
                     params: Dict, *,
                     source_info: source_info_util.SourceInfo,
                     name_stack: Union[str, source_info_util.NameStack] = "",
                     ) -> xc.OpMetadata:
  if config.jax_experimental_name_stack:
    eqn_str = str(source_info.name_stack) + '/' + str_eqn_compact(primitive.name, params)
  else:
    assert isinstance(name_stack, str)
    eqn_str = name_stack + str_eqn_compact(primitive.name, params)
  tracebacks[eqn_str] = source_info.traceback
  frame = source_info_util.user_frame(source_info)
  return xc.OpMetadata(
        op_type=primitive.name,
        op_name=eqn_str,
        source_file=_get_canonical_source_file(frame) if frame else None,
        source_line=frame.line_num if frame else None)

# Utilities

def parameter(builder, num, shape, name=None, replicated=None):
  if name is None:
    name = ''
  if replicated is None:
    replicated = []
  elif isinstance(replicated, bool):
    replicated = [replicated] * shape.leaf_count()

  return xops.Parameter(builder, num,
                        shape.with_major_to_minor_layout_if_absent(), name,
                        replicated)

# HLO instructions optionally can be annotated to say how the output should be
# spatially partitioned (represented in XLA as OpSharding protos, see
# sharding_to_proto). For array outputs, the annotation is either an int per
# dimension specifying the number of ways that dimension divided (i.e. the total
# number of shards is the product), or None to indicate the array should be
# replicated. Tuple outputs are represented as tuples thereof. XLA supports
# arbitrary tuple nesting, but JAX only uses one level of tupling (and our type
# checkers don't support recursive types), so we only represent one level of
# nesting in this type definition.
SpatialSharding = Union[Tuple[int, ...],
                        None,
                        Tuple[Optional[Tuple[int, ...]], ...]]

def sharding_to_proto(sharding: SpatialSharding):
  """Converts a SpatialSharding to an OpSharding.

  See
  https://github.com/tensorflow/tensorflow/blob/main/tensorflow/compiler/xla/xla_data.proto#L601
  for details on the OpSharding proto.
  """
  proto = xc.OpSharding()
  if isinstance(sharding, tuple) and not isinstance(sharding[0], int):
    assert all(s is None or isinstance(s, tuple) for s in sharding)
    return tuple_sharding_proto(list(map(sharding_to_proto, sharding)))  # type: ignore

  if sharding is None:
    proto.type = xc.OpSharding.Type.REPLICATED
  else:
    proto.type = xc.OpSharding.Type.OTHER
    proto.tile_assignment_dimensions = list(sharding)  # type: ignore
    proto.tile_assignment_devices = list(range(np.product(sharding)))  # type: ignore
  return proto

def tuple_sharding_proto(elems):
  proto = xc.OpSharding()
  assert all(isinstance(e, type(proto)) for e in elems)
  proto.type = xc.OpSharding.Type.TUPLE
  proto.tuple_shardings = elems
  return proto


def with_sharding_proto(builder, sharding_proto, op_fn, *args, **kwargs):
  """Builds op_fn(*args, **kwargs) with sharding annotation."""
  builder.set_sharding(sharding_proto)
  try:
    return op_fn(*args, **kwargs)
  finally:
    builder.clear_sharding()

def with_sharding(builder, sharding: SpatialSharding, op_fn, *args, **kwargs):
  """Builds op_fn(*args, **kwargs) with sharding annotation."""
  return with_sharding_proto(builder, sharding_to_proto(sharding), op_fn, *args,
                             **kwargs)


### handlers

# Numpy dtypes -> XLA primitive types

_dtype_to_primitive_type: Dict[np.dtype, xc.PrimitiveType] = {
  np.dtype('bool'): xc.PrimitiveType.PRED,
  np.dtype('int8'): xc.PrimitiveType.S8,
  np.dtype('int16'): xc.PrimitiveType.S16,
  np.dtype('int32'): xc.PrimitiveType.S32,
  np.dtype('int64'): xc.PrimitiveType.S64,
  np.dtype('uint8'): xc.PrimitiveType.U8,
  np.dtype('uint16'): xc.PrimitiveType.U16,
  np.dtype('uint32'): xc.PrimitiveType.U32,
  np.dtype('uint64'): xc.PrimitiveType.U64,
  np.dtype(dtypes.bfloat16): xc.PrimitiveType.BF16,
  np.dtype('float16'): xc.PrimitiveType.F16,
  np.dtype('float32'): xc.PrimitiveType.F32,
  np.dtype('float64'): xc.PrimitiveType.F64,
  np.dtype('complex64'): xc.PrimitiveType.C64,
  np.dtype('complex128'): xc.PrimitiveType.C128,
}

def dtype_to_primitive_type(dtype: np.dtype) -> xc.PrimitiveType:
  """Converts a NumPy dtype into an XLA PrimitiveType."""
  # Many things (e.g., strings, scalar types) can be compared with NumPy dtypes,
  # but may not hash correctly. Make sure we have a true np.dtype.
  assert isinstance(dtype, np.dtype), type(dtype)
  try:
    return _dtype_to_primitive_type[dtype]
  except KeyError as err:
    raise TypeError(f"No XLA lowering for NumPy dtype: {dtype}") from err


# JAX abstract values -> XLA shapes

def aval_to_xla_shapes(aval: core.AbstractValue) -> Sequence[XlaShape]:
  try:
    return xla_shape_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No xla_shape_handler for type: {type(aval)}") from err

xla_shape_handlers: Dict[Type[core.AbstractValue],
                         Callable[[Any], Sequence[XlaShape]]] = {
    ShapedArray: _make_array_shape,
    ConcreteArray: _make_array_shape,
}
xla_shape_handlers[core.AbstractToken] = lambda _: (xc.Shape.token_shape(),)



# IR constants

# TODO(mattjj): try to remove this canonicalize_dtype stuff
def canonicalize_dtype(x):
  typ = type(x)
  handler = canonicalize_dtype_handlers.get(typ)
  if handler: return handler(x)
  for typ in typ.__mro__:
    handler = canonicalize_dtype_handlers.get(typ)
    if handler: return handler(x)
  if hasattr(x, '__jax_array__'):
    return canonicalize_dtype(x.__jax_array__())
  raise TypeError(f"No canonicalize_dtype handler for type: {type(x)}")

def _canonicalize_ndarray_dtype(x):
  return np.asarray(x, dtypes.canonicalize_dtype(dtypes.result_type(x)))

def _canonicalize_python_scalar_dtype(typ, x):
  return np.asarray(
      x, dtypes.canonicalize_dtype(dtypes._scalar_type_to_dtype(typ, x)))

canonicalize_dtype_handlers: Dict[Any, Callable] = {}
for t in device_array.device_array_types:
  canonicalize_dtype_handlers[t] = identity
canonicalize_dtype_handlers.update(
    (t, _canonicalize_ndarray_dtype) for t in array_types)
canonicalize_dtype_handlers.update(
    (t, partial(_canonicalize_python_scalar_dtype, t)) for t in _scalar_types)
canonicalize_dtype_handlers[core.Token] = identity
canonicalize_dtype_handlers[core.PaddedArray] = identity
canonicalize_dtype_handlers[core.BInt] = \
    lambda x: core.BInt(_canonicalize_python_scalar_dtype(int, x.val), x.bound)

def abstractify(x) -> core.AbstractValue:
  typ = type(x)
  aval_fn = pytype_aval_mappings.get(typ)
  if aval_fn: return aval_fn(x)
  for typ in typ.__mro__:
    aval_fn = pytype_aval_mappings.get(typ)
    if aval_fn: return aval_fn(x)
  if hasattr(x, '__jax_array__'):
    return abstractify(x.__jax_array__())
  raise TypeError(f"Argument '{x}' of type '{type(x)}' is not a valid JAX type")

def _make_abstract_python_scalar(typ, val):
  # Note: all python scalar types are weak except bool, because bool only
  # comes in a single width.
  return ShapedArray((), dtypes._scalar_type_to_dtype(typ, val),
                     weak_type=typ is not bool)

pytype_aval_mappings: Dict[Any, Callable[[Any], core.AbstractValue]] = {}
for t in device_array.device_array_types:
  pytype_aval_mappings[t] = operator.attrgetter('aval')
pytype_aval_mappings[core.BInt] = lambda x: core.AbstractBInt(x.bound)
pytype_aval_mappings[core.PaddedArray] = operator.attrgetter('_aval')
pytype_aval_mappings[core.Token] = lambda _: core.abstract_token
pytype_aval_mappings.update((t, make_shaped_array) for t in array_types)
pytype_aval_mappings.update(
    (t, partial(_make_abstract_python_scalar, t)) for t in _scalar_types)


def primitive_subcomputation(platform: str, axis_env: 'AxisEnv',
                             prim: core.Primitive,
                             avals_in: Sequence[core.AbstractValue],
                             avals_out: Sequence[core.AbstractValue],
                             **params):
  c = xc.XlaBuilder(f"primitive_computation_{prim.name}")
  counts = it.count()
  xla_args = [parameter(c, next(counts), xla_shape)
              for a in avals_in for xla_shape in aval_to_xla_shapes(a)]
  if (platform is not None and
      prim in _backend_specific_translations[platform]):
    rule = _backend_specific_translations[platform][prim]
  elif prim in _translations:
    rule = _translations[prim]

  ctx = TranslationContext(builder=c, platform=platform, axis_env=axis_env,
                           name_stack=new_name_stack())
  ans = rule(ctx, avals_in, avals_out, *xla_args, **params)

  if prim.multiple_results:
    return c.build(xops.Tuple(c, ans))
  else:
    x, = ans
    return c.build(x)


### compiling jaxprs


class AxisEnv(NamedTuple):
  """Represents a pmap mesh (only along the replica axes)."""
  nreps: int
  names: Tuple[Any, ...]
  sizes: Tuple[int, ...]

@dataclasses.dataclass
class TranslationContext:
  builder: xc.XlaBuilder
  # TODO(phawkins): make platform non-optional. We should always be translating
  # with a specific platform in mind.
  platform: Optional[str]
  axis_env: AxisEnv
  name_stack: Union[str, source_info_util.NameStack]

  def replace(self, **kw): return dataclasses.replace(self, **kw)



def xla_destructure(c, ans):
  num_elements = len(c.get_shape(ans).tuple_shapes())
  return [xops.GetTupleElement(ans, i) for i in range(num_elements)]

def check_backend_matches(inner_backend, outer_backend):
  # For nested calls, the outermost call sets the backend for all inner calls;
  # it's an error if the inner call has a conflicting explicit backend spec.
  if inner_backend is None:
    return
  if (inner_backend != outer_backend and
      outer_backend not in xb.expand_platform_alias(inner_backend)):
    raise ValueError(
        f"Outer-jit backend specification {outer_backend} must match explicit "
        f"inner-jit backend specification {inner_backend}.")


def extend_axis_env(env: AxisEnv, name, size: int):
  return AxisEnv(env.nreps, env.names + (name,), env.sizes + (size,))

def axis_read(axis_env, axis_name):
  try:
    return max(i for i, name in enumerate(axis_env.names) if name == axis_name)
  except ValueError:
    raise NameError(f"unbound axis name: {axis_name}") from None

def axis_groups(axis_env: AxisEnv, name) -> Tuple[Tuple[int, ...]]:
  if not isinstance(name, (list, tuple)):
    name = (name,)
  mesh_axes = tuple(unsafe_map(partial(axis_read, axis_env), name))
  trailing_size, ragged = divmod(axis_env.nreps, prod(axis_env.sizes))
  assert not ragged
  mesh_spec = axis_env.sizes + (trailing_size,)
  return _axis_groups(mesh_spec, mesh_axes)

def _axis_groups(mesh_spec, mesh_axes):
  """Computes replica group ids for a collective performed over a subset of the mesh.

  Args:
    mesh_spec: A sequence of integers representing the mesh shape.
    mesh_axes: A sequence of integers between 0 and `len(mesh_spec)` (exclusive)
      indicating over which axes the collective is performed.
  Returns:
    A tuple of replica groups (i.e. tuples containing replica ids).
  """
  iota = np.arange(prod(mesh_spec)).reshape(mesh_spec)
  groups = np.reshape(
      np.moveaxis(iota, mesh_axes, np.arange(len(mesh_axes))),
      (prod(np.take(mesh_spec, mesh_axes)), -1))
  return tuple(unsafe_map(tuple, groups.T))


# TODO(mattjj,skyewm): the functions here are utilities for checking if
# not-yet-supported features are used with multi-host programming


def jaxpr_collectives(jaxpr):
  """Generates all the collective primitives anywhere inside a Jaxpr."""
  for eqn in jaxpr.eqns:
    if eqn.primitive in _collective_primitives:
      yield eqn.primitive
  for subjaxpr in core.subjaxprs(jaxpr): yield from jaxpr_collectives(subjaxpr)


### xla_call underlying jit


xla_call_p: core.CallPrimitive = core.CallPrimitive('xla_call')
xla_call = xla_call_p.bind

def _xla_call_partial_eval_update_params(params, kept_inputs, num_new_inputs):
  donated_invars = params['donated_invars']
  if not kept_inputs and donated_invars:
    # JaxprTrace.post_process_call creates a call with no input tracers
    donated_invars = (False,) * num_new_inputs
  else:
    assert len(kept_inputs) == len(donated_invars)
    # JaxprTrace.process_call drops known input tracers
    donated_invars = [d for d, kept in zip(donated_invars, kept_inputs) if kept]
    # Any new inputs are prepended to the left, so mark those as not donated.
    donated_invars = [False] * num_new_inputs + donated_invars
  return dict(params, donated_invars=tuple(donated_invars))
pe.call_param_updaters[xla_call_p] = _xla_call_partial_eval_update_params

def _xla_call_jvp_update_params(params, nz_tangents):
  donated_invars = params['donated_invars']
  donated_tangents = [d for d, nz in zip(donated_invars, nz_tangents) if nz]
  new_donated_invars = (*donated_invars, *donated_tangents)
  return dict(params, donated_invars=new_donated_invars)
ad.call_param_updaters[xla_call_p] = _xla_call_jvp_update_params

def _xla_call_transpose_update_params(params, undef_primals, nonzero_cts):
  donated_invars = params['donated_invars']
  donated_primals = [d for d, u in zip(donated_invars, undef_primals) if not u]
  donated_cotangents = [False for nz in nonzero_cts if nz]
  return dict(params, donated_invars=(*donated_primals, *donated_cotangents))
ad.call_transpose_param_updaters[xla_call_p] = _xla_call_transpose_update_params


ad.primitive_transposes[xla_call_p] = partial(ad.call_transpose, xla_call_p)


def _xla_call_partial_eval_custom_params_updater(
    unks_in: Sequence[bool], inst_in: Sequence[bool],
    kept_outs_known: Sequence[bool], kept_outs_staged: Sequence[bool],
    num_res: int, params_known: dict, params_staged: dict
  ) -> Tuple[dict, dict]:
  # pruned inputs to jaxpr_known according to unks_in, so prune donated_invars
  donated_known, _ = partition_list(unks_in, params_known['donated_invars'])
  new_params_known = dict(params_known, donated_invars=tuple(donated_known))
  # added num_res new inputs to jaxpr_staged, so extend donated_invars
  _, donated_staged_ = partition_list(inst_in, params_staged['donated_invars'])
  donated_staged = [False] * num_res + donated_staged_
  new_params_staged = dict(params_staged, donated_invars=tuple(donated_staged))
  return new_params_known, new_params_staged
pe.partial_eval_jaxpr_custom_rules[xla_call_p] = \
    partial(pe.call_partial_eval_custom_rule, 'call_jaxpr',
            _xla_call_partial_eval_custom_params_updater)
pe.dce_rules[xla_call_p] = pe.dce_jaxpr_call_rule

pe.padding_rules[xla_call_p] = partial(pe.call_padding_rule, xla_call_p)


def _pp_xla_call(eqn: core.JaxprEqn, context: core.JaxprPpContext,
                 settings: core.JaxprPpSettings,
                 ) -> List[pp.Doc]:
  printed_params = {k:v for k, v in eqn.params.items() if
                    k == 'call_jaxpr' or k == 'name' or
                    k == 'backend' and v is not None or
                    k == 'device' and v is not None or
                    k == 'donated_invars' and any(v)}
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  lhs = core.pp_vars(eqn.outvars, context, print_shapes=settings.print_shapes)
  rhs = [pp.text(eqn.primitive.name),
         core.pp_kv_pairs(sorted(printed_params.items()), context, settings),
         pp.text(" ") + core.pp_vars(eqn.invars, context)]
  return [lhs, pp.text(" = ", annotation=annotation), *rhs]
core.pp_eqn_rules[xla_call_p] = _pp_xla_call


### translation tables

MYPY = False
if not MYPY:
  class TranslationRule(Protocol):
    def __call__(self, ctx: TranslationContext,
                 avals_in: Sequence[core.AbstractValue],
                 avals_out: Sequence[core.AbstractValue],
                 *args: XlaOp, **kw
                ) -> Sequence[XlaOp]:
      """A translation rule lowers a primitive invocation into an XLA HLO."""
else:
  TranslationRule = Any

_translations: Dict[core.Primitive, TranslationRule] = {}
_backend_specific_translations: Dict[str, Dict[core.Primitive, TranslationRule]]
_backend_specific_translations = defaultdict(dict)

_collective_primitives: Set[core.Primitive] = set()
_initial_style_primitives: Set[core.Primitive] = set()

def register_initial_style_primitive(prim: core.Primitive):
  _initial_style_primitives.add(prim)

def register_collective_primitive(prim: core.Primitive):
  _collective_primitives.add(prim)

def register_translation(prim: core.Primitive, rule: TranslationRule, *,
                         platform: Optional[str] = None) -> None:
  if platform is None:
    _translations[prim] = rule
  else:
    # For backward compatibility reasons, we allow rules to be registered
    # under "gpu" even though the platforms are now called "cuda" and "rocm".
    # TODO(phawkins): fix up users to specify either "cuda" or "rocm" and remove
    # this expansion.
    for p in xb.expand_platform_alias(platform):
      _backend_specific_translations[p][prim] = rule


# As a temporary backward compatibility measure, we use an adapter class to
# convert from the old styles of translation rules to the newer ones.
# TODO(phawkins): update users of the older translation rule styles and remove
# the adapters.
class _TranslationRuleAdapter:
  def __init__(self, translations,
               wrap_fn: Callable[[core.Primitive, Callable], TranslationRule]):
    self._translations = translations
    self._wrap_fn = wrap_fn

  def __setitem__(self, key: core.Primitive, value: Callable):
    wrapped = self._wrap_fn(key, value)
    for translations in self._translations:
      translations[key] = wrapped


def _wrap_old_translation(prim: core.Primitive, f: Callable) -> TranslationRule:
  @functools.wraps(f)
  def wrapped(ctx: TranslationContext, avals_in: Sequence[core.AbstractValue],
              avals_out: Sequence[core.AbstractValue],
               *args: XlaOp, **kw) -> Sequence[XlaOp]:
    ans = f(ctx.builder, *args, **kw)
    if (prim.multiple_results or
        any(len(aval_to_xla_shapes(aval)) > 1 for aval in avals_out)):
      return xla_destructure(ctx.builder, ans)
    else:
      return [ans]
  return wrapped


translations : _TranslationRuleAdapter
translations = _TranslationRuleAdapter([_translations], _wrap_old_translation)

class _BackendSpecificTranslationsAdapter(defaultdict):
  def __missing__(self, key):
    translation_tables = [_backend_specific_translations[p]
                          for p in xb.expand_platform_alias(key)]
    ret = self[key] = _TranslationRuleAdapter(
        translation_tables, _wrap_old_translation)
    return ret

backend_specific_translations: Dict[str, _TranslationRuleAdapter]
backend_specific_translations = _BackendSpecificTranslationsAdapter()

# TODO(phawkins): remove lower_fun completely after updating users.
def lower_fun(fun: Callable, *, multiple_results: bool, backend=None,
              new_style: bool = False) -> Callable:
  def f(*args, **kw):
    raise RuntimeError("XLA translation rules are deprecated and "
                       "jax.interpreters.xla.lower_fun is no longer supported. "
                       "Add an MLIR (MHLO) lowering via jax.interpreters.mlir "
                       "instead.")
  return f
