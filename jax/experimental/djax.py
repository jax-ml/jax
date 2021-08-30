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

from functools import partial, reduce
import itertools as it
import operator as op
from typing import (Tuple, List, Sequence, Set, Dict, Any, Callable, Union,
                    Optional)

from jax import core
from jax._src import dtypes
from jax.core import Var, Literal, Atom, Tracer
from jax._src.util import (safe_zip, safe_map, curry, unzip2, split_list,
                           tuple_delete)
from jax._src.pprint_util import pp, vcat, PrettyPrint

map = safe_map
zip = safe_zip
def identity(x): return x

DType = Any
NDArray = Any


# Dynamic shape jaxprs

## Element types

class EltTy: pass

class BaseType(EltTy):
  def __init__(self, dtype: DType):
    self._dtype = dtypes.dtype(dtype)

  def __repr__(self):
    return f'BaseType({self._dtype.name})'

  def __hash__(self):
    return hash(self._dtype)

  def __eq__(self, other):
    return isinstance(other, BaseType) and self._dtype == other._dtype

class BoundedIntTy(EltTy):
  def __init__(self, bound: int):
    assert isinstance(bound, int)
    self._bound = bound

  def __repr__(self):
    return f'BIntTy{{≤{self._bound}}}'

  def __eq__(self, other):
    return isinstance(other, BoundedIntTy) and self._bound == other._bound


## Array types

class AbsArray(core.AbstractValue):
  def __init__(self, shape, eltTy):
    assert isinstance(shape, tuple)
    assert isinstance(eltTy, EltTy)
    self.shape = shape
    self._eltTy = eltTy

  def str_short(self):
    shape = f'[{",".join(str(d) for d in self.shape)}]' if self.shape else ''
    if isinstance(self._eltTy, BoundedIntTy):
      return f'BInt{{≤{self._eltTy._bound}}}{shape}'
    elif isinstance(self._eltTy, BaseType):
      dtype = self._eltTy._dtype.name
      return f'{dtype}{shape}'
    else:
      return repr(self)

  def __eq__(self, other):
    if (isinstance(other, AbsArray) and self._eltTy == other._eltTy and
        len(self.shape) == len(other.shape)):
      for a, b in zip(self.shape, other.shape):
        if type(a) is type(b) is int:
          if a != b: return False
        elif type(a) is type(b) is BoundedInt:
          if a is not b: return False
        elif type(a) is type(b) is Var:
          if a is not b: return False
        elif type(a) is type(b) is AbsArray:
          if a != b: return False
        elif type(a) is type(b) is DimIndexingExpr:
          if a.name is not b.name or a.indices != b.indices: return False
        else:
          return False
      else:
        return True
    return False

  # this duck-typing is needed by eg ad.py using dtypes.py
  @property
  def dtype(self):
    if isinstance(self._eltTy, BaseType):
      return self._eltTy._dtype
    else:
      raise Exception

  def at_least_vspace(self):
    return AbsArray(self.shape, self._eltTy)

  def join(self, other):
    if self == other:
      return self
    raise NotImplementedError  # TODO

class DimIndexingExpr:
  def __init__(self, name, indices):
    assert isinstance(name, (Var, Tracer))
    assert (isinstance(indices, tuple) and
            all(isinstance(i, int) for i in indices))
    self.name = name
    self.indices = indices

  def __repr__(self):
    indices = '.'.join(map(str, self.indices))
    return f'{self.name}.{indices}'


## DJaxprs

class DJaxprTy:
  in_dim_binders: List[Var]
  in_types: List[core.AbstractValue]
  out_dim_binders: List[Var]
  out_types: List[core.AbstractValue]

  def __init__(self, in_dim_binders, in_types, out_dim_binders, out_types):
    self.in_dim_binders = in_dim_binders
    self.in_types = in_types
    self.out_dim_binders = out_dim_binders
    self.out_types = out_types

  def __repr__(self):
    in_dim_binders = pp_vars(self.in_dim_binders)
    in_types = ', '.join(aval.str_short() for aval in self.in_types)
    out_dim_binders = pp_vars(self.out_dim_binders)
    out_types = ', '.join(aval.str_short() for aval in self.out_types)
    return f'[{in_dim_binders}] [{in_types}] -> [{out_dim_binders}] [{out_types}]'

class DJaxpr:
  in_dim_binders: List[Var]
  in_binders: List[Var]
  out_dims: List[Atom]
  outs: List[Atom]
  eqns: List[core.JaxprEqn]  # reusing existing eqns, helps reuse some tracing

  def __init__(self, in_dim_binders, in_binders, out_dims, outs, eqns):
    assert all(isinstance(v, Var) and isinstance(v.aval, AbsArray) and
               isinstance(v.aval._eltTy, BoundedIntTy) for v in in_dim_binders)
    assert all(isinstance(v, Var) for v in in_binders)
    assert all(isinstance(x, (Var, Literal)) and isinstance(x.aval, AbsArray) and
               isinstance(x.aval._eltTy, BoundedIntTy) for x in out_dims)
    assert all(isinstance(x, (Var, Literal)) for x in outs)
    assert all(isinstance(e, core.JaxprEqn) for e in eqns)
    self.in_dim_binders = in_dim_binders
    self.in_binders = in_binders
    self.out_dims = out_dims
    self.outs = outs
    self.eqns = eqns

  def __repr__(self):
    return str(pp_djaxpr(self))

def pp_djaxpr(jaxpr: DJaxpr) -> PrettyPrint:
  eqns = map(pp_eqn, jaxpr.eqns)
  in_dim_binders = pp_vars(jaxpr.in_dim_binders)
  in_binders = pp_vars(jaxpr.in_binders)
  out_dims = ', '.join(map(str, jaxpr.out_dims))
  outs = ', '.join(map(str, jaxpr.outs))
  out_dim_types = pp_vars(jaxpr.out_dims)
  outs_type = ', '.join(v.aval.str_short() for v in jaxpr.outs)
  return (pp(f'{{ lambda {in_dim_binders} ; {in_binders} .')
          + (pp('let ') >> vcat(eqns) +
             pp(f'in ( {out_dims} ; {outs} ) '
                f': ( {out_dim_types} ; {outs_type} ) }}')).indent(2))

def pp_vars(vs: Sequence[Atom]) -> str:
  return ', '.join(f'{v}:{v.aval.str_short()}' for v in vs)

def pp_eqn(eqn: core.JaxprEqn) -> PrettyPrint:
  lhs = pp_vars(eqn.outvars)
  pp_lhs = pp(f'{lhs} =')
  pp_rhs = (pp(eqn.primitive.name) >>
            core.pp_kv_pairs(sorted(eqn.params.items())) >> pp(' ') >>
            pp(' '.join(map(str, eqn.invars))))
  return pp_lhs >> pp(' ') >> pp_rhs

# Typechecking DJaxprs

def typecheck_jaxpr(jaxpr: DJaxpr):
  env: Set[Var] = set()  # bound variables

  for v in jaxpr.in_dim_binders:
    if not (isinstance(v.aval, AbsArray) and
            isinstance(v.aval._eltTy, BoundedIntTy)): raise TypeError
    typecheck_type(env, v.aval)
    env.add(v)

  for v in jaxpr.in_binders:
    typecheck_type(env, v.aval)
  for v in jaxpr.in_binders:
    env.add(v)

  for eqn in jaxpr.eqns:
    for x in eqn.invars:
      typecheck_atom(env, x)
    rule = typecheck_rules[eqn.primitive]
    out_types = rule(*eqn.invars, **eqn.params)
    subst: Dict[Var, Var] = {}
    for v, t in zip(eqn.outvars, out_types):
      if isinstance(t, Var):
        aval = substitute(subst, t.aval)
        if v.aval != aval: raise TypeError(f'{v.aval} != {aval}')
        subst[t] = v
      elif isinstance(t, core.AbstractValue):
        aval = substitute(subst, t)
        if v.aval.strip_weak_type() != aval:
          raise TypeError(f'{v.aval} != {aval}')
      else:
        assert False  # typecheck rule produced unexpected type
      typecheck_type(env, v.aval)
      env.add(v)

  in_types = [v.aval for v in jaxpr.in_binders]
  out_types = []
  for x in jaxpr.outs:
    aval = typecheck_atom(env, x)
    out_types.append(aval)

  return DJaxprTy(jaxpr.in_dim_binders, in_types, jaxpr.out_dims, out_types)

def typecheck_type(env, aval):
  if isinstance(aval, (core.AbstractUnit, core.ShapedArray)):
    return aval  # all syntactic forms are valid
  elif isinstance(aval, AbsArray):
    for i, d in enumerate(aval.shape):
      if isinstance(d, int):
        continue
      elif isinstance(d, Var):
        if d not in env: raise TypeError('unbound dim size')
        if not (isinstance(d.aval, AbsArray) and not d.aval.shape and
                isinstance(d.aval._eltTy, BoundedIntTy)):
          raise TypeError(f'dim var of unexpected type: {d.aval}')
      elif isinstance(d, DimIndexingExpr):
        if d.name not in env: raise TypeError('unbound dim size')
        if not (isinstance(d.name.aval, AbsArray) and
                isinstance(d.name.aval._eltTy, BoundedIntTy)):
          raise TypeError(f'dim var of unexpected type: {d.name.aval}')
        d_indices_set = set(d.indices)
        if i in d_indices_set:
          raise TypeError(f"circular dim indexing expression: {d}")
        for j in d.indices:
          d_j = aval.shape[j]
          if (isinstance(d_j, DimIndexingExpr) and
              not d_indices_set.issuperset(d_j.indices)):
            raise TypeError(f"dim indexing not transitively closed: {d}")
        expected_idx_array_shape = tuple(aval.shape[j] for j in d.indices)
        if d.name.aval.shape != expected_idx_array_shape:
          raise TypeError(f'incompatible shapes in dim indexing: {aval}')
      else:
        raise TypeError(f'unexpected type in shape: {type(d)}')
    return aval
  else:
    raise TypeError(f'unknown type: {aval}')

def typecheck_atom(env, x):
  if isinstance(x, Literal):
    return core.raise_to_shaped(core.get_aval(x.val))
  elif isinstance(x, Var):
    return typecheck_type(env, x.aval)
  else:
    raise TypeError(f'atom of unexpected type {x}')

def substitute(subst, aval):
  if isinstance(aval, AbsArray):
    new_shape = []
    for d in aval.shape:
      if isinstance(d, Var):
        new_d = subst.get(d, d)
      elif isinstance(d, DimIndexingExpr):
        new_d = DimIndexingExpr(subst.get(d.name, d.name), d.indices)
      else:
        new_d = d
      new_shape.append(new_d)
    return AbsArray(tuple(new_shape), aval._eltTy)
  else:
    return aval

typecheck_rules: Dict[core.Primitive, Callable] = {}


# Interpreting DJaxprs

def eval_jaxpr(jaxpr, dim_args, args):
  env: Dict[Var, Any] = {}

  def read(v):
    if type(v) is core.Literal:
      return v.val
    else:
      return env[v]

  def write(v, val):
    env[v] = val

  write(core.unitvar, core.unit)
  map(write, jaxpr.in_dim_binders, dim_args)
  map(write, jaxpr.in_binders, args)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    ans = eqn.primitive.bind(*in_vals, **eqn.params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    elif len(eqn.outvars) > 1:
      # TODO a jaxpr unpacks dependent tuples, while Python packages them up
      map(write, eqn.outvars, eqn.primitive.unpack_result(ans))
    else:
      write(eqn.outvars[0], ans)
  return map(read, jaxpr.out_dims), map(read, jaxpr.outs)

@curry
def jaxpr_as_fun(jaxpr, *args):
  shapevars_to_vals: Dict[Var, Any] = dict(
      (d, t) for v, x in zip(jaxpr.in_binders, args) if isinstance(v.aval, AbsArray)
      for d, t in zip(v.aval.shape, x.shape) if isinstance(d, Var)
      and x is not core.unit)  # TODO partial eval assumes we can plug in units?
  dim_args = [shapevars_to_vals[v] for v in jaxpr.in_dim_binders]
  _, out = eval_jaxpr(jaxpr, dim_args, args)
  return out


# Data representations

class BoundedInt:
  val: Union[int, Tracer]
  bound: int

  def __init__(self, val: Union[int, Tracer], bound: int):
    self._val = val
    self._bound = bound

  def __repr__(self):
    return f'{self._val}{{≤{self._bound}}}'

  def __eq__(self, other):
    if isinstance(other, BoundedInt) and self._bound == other._bound:
      return self._val is other._val or self._val == other._val
    elif isinstance(other, int):
      return self._val == other
    else:
      raise Exception

class DimIndexer:
  data: NDArray
  indices: Tuple[int, ...]

  def __init__(self, data, indices):
    self._data = data
    self._indices = indices

  def __repr__(self):
    indices = '.'.join(map(str, self._indices))
    data = f'{self._data._data}'
    return f'{data}.{indices}'

# We want these to duck-type ndarrays when the element type is BaseType.
class Array:
  def __init__(self,
               shape: Tuple[Union[int, BoundedInt, DimIndexer], ...],
               eltTy: EltTy,
               data: NDArray):
    self.shape = shape
    self._eltTy = eltTy
    self._data = data

  @property
  def dtype(self):
    if isinstance(self._eltTy, BaseType):
      return self._eltTy._dtype
    else:
      raise Exception

  def __repr__(self):
    dtypestr = (self._eltTy._dtype.name if isinstance(self._eltTy, BaseType)
                else f'BInt{{≤{self._eltTy._bound}}}')  # type: ignore
    shapestr = ','.join(map(str, self.shape))
    if any(isinstance(d, DimIndexer) for d in self.shape):
      # find the last DimIndexer, as we'll treat chunks below that as
      # rectangular
      last = next(i for i, d in reversed(list(enumerate(self.shape)))
                  if isinstance(d, DimIndexer))
      shape_prefix = tuple(d._val if type(d) is BoundedInt else d
                            for d in self.shape[:last])
      outs = []
      for idx in it.product(*map(range, shape_prefix)):
        slices = [slice(d._data._data[tuple(idx[i] for i in d._indices)])
                  if isinstance(d, DimIndexer) else
                  slice(d._val) if isinstance(d, BoundedInt) else
                  slice(None) for d in self.shape[last:]]
        full_index = (*idx, *slices)
        data = self._data[full_index]
        outs.append(f'{idx}:\n{data}')
      return f'{dtypestr}[{shapestr}] with values:\n' + '\n\n'.join(outs)
    else:
      slices = tuple(slice(d._val) if type(d) is BoundedInt else slice(None)
                     for d in self.shape)
      data = self._data[slices]
      return f'{dtypestr}[{shapestr}] with value:\n{data}'

  def __array__(self):
    if any(isinstance(d, DimIndexer) for d in self.shape):
      raise NotImplementedError  # ragged ndarray
    else:
      slices = tuple(slice(d._val) if type(d) is BoundedInt else slice(None)
                     for d in self.shape)
      return np.array(self._data[slices])


# Tracing to embed DJaxprs in Python

from jax import linear_util as lu
from jax.interpreters import partial_eval as pe

from jax.api_util import flatten_fun
from jax.tree_util import tree_flatten, tree_unflatten

def make_djaxpr(fun, *args, **kwargs):
  args, in_tree = tree_flatten((args, kwargs))
  f, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
  in_avals = [core.raise_to_shaped(core.get_aval(x)) for x in args]
  return trace_to_jaxpr_dynamic(f, in_avals)

def trace_to_jaxpr_dynamic(fun: lu.WrappedFun, in_avals: Sequence[core.AbstractValue]):
  with core.new_main(DJaxprTrace, dynamic=True) as main:
    main.jaxpr_stack = ()  # type: ignore
    outs = trace_to_subjaxpr_dynamic(fun, main, in_avals)
    del main
  return outs

def trace_to_subjaxpr_dynamic(fun: lu.WrappedFun, main: core.MainTrace,
                              in_avals: Sequence[core.AbstractValue]):
  frame = DJaxprStackFrame()
  with pe.extend_jaxpr_stack(main, frame):
    trace = DJaxprTrace(main, core.cur_sublevel())
    in_dim_tracers, in_avals = _place_in_dim_tracers_in_shapes(trace, in_avals)
    in_tracers = map(trace.new_arg, in_avals)
    ans = fun.call_wrapped(*in_tracers)
    out_tracers = map(trace.full_raise, ans)
  out_dim_tracers = _extract_out_dim_tracers_from_shapes(main, in_dim_tracers, out_tracers)
  return frame.to_jaxpr(in_dim_tracers, in_tracers, out_dim_tracers, out_tracers)

def _place_in_dim_tracers_in_shapes(trace, in_avals):
  dim_tracers = {}
  new_in_avals = []
  for aval in in_avals:
    if not isinstance(aval, AbsArray):
      new_in_avals.append(aval)
    else:
      new_shape = []
      for d in aval.shape:
        if isinstance(d, AbsArray):
          assert d.shape == () and isinstance(d._eltTy, BoundedIntTy)
          dim_tracer = dim_tracers.get(id(d))
          if dim_tracer is None:
            dim_tracer = dim_tracers[id(d)] = trace.new_arg(d)
          new_shape.append(dim_tracer)
        elif isinstance(d, (int, BoundedInt)):
          new_shape.append(d)
        else:
          raise NotImplementedError(d)  # TODO
      new_aval = AbsArray(tuple(new_shape), aval._eltTy)
      new_in_avals.append(new_aval)
  return list(dim_tracers.values()), new_in_avals

def _extract_out_dim_tracers_from_shapes(main, in_dim_tracers, out_tracers):
  seen = {id(d) for d in in_dim_tracers}
  def take(d):
    if isinstance(d, Tracer):
      return d._trace.main is main and id(d) not in seen and not seen.add(id(d))
    elif isinstance(d, DimIndexingExpr):
      return take(d.name)
    else:
      return False
  return [d.name if isinstance(d, DimIndexingExpr) else d
          for t in out_tracers if isinstance(t.aval, AbsArray)
          for d in t.aval.shape if take(d)]

class DJaxprTrace(pe.DynamicJaxprTrace):
  def process_primitive(self, primitive, tracers, params):
    rule = custom_staging_rules.get(primitive)
    if rule:
      return rule(self, tracers, params)
    else:
      # If there's no special staging rule, by default do regular Jaxpr staging
      return super().process_primitive(primitive, tracers, params)

  def get_const(self, tracer):
    assert isinstance(tracer, Tracer)
    return self.frame.constvar_to_val.get(self.frame.tracer_to_var.get(id(tracer)))

  def new_const(self, val):
    if isinstance(val, BoundedInt):
      raise NotImplementedError  # TODO
    elif isinstance(val, Array) and val.shape:
      raise NotImplementedError  # TODO
    else:
      return super().new_const(val)

custom_staging_rules: Dict[core.Primitive, Callable] = {}

class DJaxprStackFrame(pe.JaxprStackFrame):
  def to_jaxpr(self, in_dim_tracers, in_tracers, out_dim_tracers, out_tracers):
    t2v = lambda t: self.tracer_to_var[id(t)]
    in_dim_binders, in_binders = map(t2v, in_dim_tracers), map(t2v, in_tracers)
    out_dims, outs = map(t2v, out_dim_tracers), map(t2v, out_tracers)

    # only include constants that are used
    used_vars = ({a for eqn in self.eqns for a in eqn.invars if isinstance(a, Var)} |
                 {a for grp in [out_dims, outs] for a in grp if isinstance(a, Var)})
    constvars, constvals = unzip2(
        (v, c) for v, c in self.constvar_to_val.items() if v in used_vars)
    in_binders = [*constvars, *in_binders]

    # promote some lambda binders to pi binders
    used_shape_vars = ({d for eqn in self.eqns for v in eqn.outvars
                        if isinstance(v.aval, AbsArray)
                        for d in v.aval.shape if isinstance(d, Var)} |
                       {d.name for eqn in self.eqns for v in eqn.outvars
                        if isinstance(v.aval, AbsArray)
                        for d in v.aval.shape if isinstance(d, DimIndexingExpr)})
    lambda_binders = [v not in used_shape_vars for v in in_binders]
    converted_binders, in_binders = partition_list(lambda_binders, in_binders)
    in_dim_binders = in_dim_binders + converted_binders
    out_dims = [v for v in out_dims if v not in in_dim_binders]  # TODO

    jaxpr = DJaxpr(in_dim_binders, in_binders, out_dims, outs, self.eqns)
    typecheck_jaxpr(jaxpr)
    return jaxpr, constvals, lambda_binders

  def newvar(self, aval):
    if isinstance(aval, AbsArray) and aval.shape:
      # replace any tracers in the shape with their corresponding variables
      shape = []
      for d in aval.shape:
        if isinstance(d, Tracer):
          shape.append(self.tracer_to_var[id(d)])
        elif isinstance(d, DimIndexingExpr):
          assert isinstance(d.name, Tracer)
          shape.append(DimIndexingExpr(self.tracer_to_var[id(d.name)], d.indices))
        else:
          shape.append(d)
      aval = AbsArray(tuple(shape), aval._eltTy)
    return self.gensym(aval)

def partition_list(bs, lst):
  lists = lst1, lst2 = [], []
  for b, x in zip(bs, lst):
    lists[b].append(x)
  return lst1, lst2

def _raise_absarray_to_type_level(aval: AbsArray, weak_type: bool):
  assert isinstance(aval, AbsArray)
  unique_avals: Dict[int, AbsArray] = {}
  shape = []
  for d in aval.shape:
    if isinstance(d, BoundedInt):
      shape.append(unique_avals.setdefault(id(d), AbsArray((), BoundedIntTy(d._bound))))
    elif isinstance(d, DimIndexer):
      raise NotImplementedError  # TODO
    else:
      shape.append(d)
  return AbsArray(tuple(shape), aval._eltTy)
core.raise_to_shaped_mappings[AbsArray] = _raise_absarray_to_type_level

def _abstractify_array_for_ad(x: Array):  # TODO misleading name, used in djit
  return AbsArray(x.shape, x._eltTy)
core.pytype_aval_mappings[Array] = _abstractify_array_for_ad

def _abstractify_bdint(x: BoundedInt):
  return AbsArray((), BoundedIntTy(x._bound))
core.pytype_aval_mappings[BoundedInt] = _abstractify_bdint


# XLA lowering

from jax.interpreters import xla
from jax.lib import xla_bridge as xb
from jax.lib import xla_client as xc
xe = xc._xla
xops = xc._xla.ops

def _abstractify_array_to_type_level(x: Array):
  return core.raise_to_shaped(core.get_aval(x))
xla.pytype_aval_mappings[Array] = _abstractify_array_to_type_level

def _array_xla_shape(aval: AbsArray):
  if isinstance(aval._eltTy, BaseType):
    dtype = aval._eltTy._dtype
    shape = [d._eltTy._bound if isinstance(d, AbsArray) and not d.shape
             else d for d in aval.shape]
    return (xla.xc.Shape.array_shape(dtype, shape),)
  elif isinstance(aval._eltTy, BoundedIntTy):
    shape = [d._bound if isinstance(d, BoundedInt) else d for d in aval.shape]
    return (xla.xc.Shape.array_shape(dtypes.dtype('int32'), shape),)
  else:
    raise NotImplementedError
xla.xla_shape_handlers[AbsArray] = _array_xla_shape
xla.canonicalize_dtype_handlers[Array] = identity

def _array_device_put(x, device):
  return xla._device_put_array(x._data, device)
xla.device_put_handlers[Array] = _array_device_put

def _bdint_device_put(x, device):
  return xla._device_put_scalar(x._val, device)
xla.device_put_handlers[BoundedInt] = _bdint_device_put

def _bdint_canoncalize_dtype(x):
  return BoundedInt(xla.canonicalize_dtype(x._val), x._bound)
xla.canonicalize_dtype_handlers[BoundedInt] = _bdint_canoncalize_dtype

def _make_params(c, dim_in_avals, in_avals):
  n = it.count()
  make = lambda a: [xb.parameter(c, next(n), s) for s in xla.aval_to_xla_shapes(a)]
  return map(make, dim_in_avals), map(make, in_avals)

def _xla_consts(c, consts):
  unique_consts = {id(const): const for const in consts}
  xla_consts = {
      id_: [xb.constant(c, const)] for id_, const in unique_consts.items()}
  return [xla_consts[id(const)] for const in consts]

def djaxpr_subcomp(c, jaxpr, dim_args, args):
  env: Dict[Var, Sequence[xe.XlaOp]] = {}

  def aval(v):
    return xla.abstractify(v.val) if type(v) is core.Literal else v.aval

  def read(v):
    if type(v) is core.Literal:
      return [xb.constant(c, xla.canonicalize_dtype(v.val))]
    else:
      return env[v]

  def write(v, nodes):
    env[v] = nodes

  write(core.unitvar, xla._make_unit_constant(c))
  map(write, jaxpr.in_dim_binders, dim_args)
  map(write, jaxpr.in_binders, args)
  for eqn in jaxpr.eqns:
    in_vals, in_avals = map(read, eqn.invars), map(aval, eqn.invars)
    in_dims = {v:read(v) for a in in_avals if isinstance(a, AbsArray)
               for v in a.shape if isinstance(v, Var)}
    rule = translations[eqn.primitive]
    out_vals = rule(c, in_dims, in_avals, in_vals, **eqn.params)
    map(write, eqn.outvars, out_vals)
  return map(read, jaxpr.out_dims), map(read, jaxpr.outs)

def execute_compiled(compiled, partitioner, handlers, dim_vals, args):
  input_bufs = list(it.chain(
      (buf for x in dim_vals for buf in xla.device_put(x, None)),
      (buf for x in args     for buf in xla.device_put(x, None))))
  out_bufs = compiled.execute(input_bufs)
  dims_dict, grouped_bufs = partitioner(out_bufs)
  return [handler(dims_dict, bs) for handler, bs in zip(handlers, grouped_bufs)]

def result_partitioner(in_dim_binders, in_dim_vals, out_dims, out_bufcounts):
  out_dimvars = [v for v in out_dims if isinstance(v, Var)]
  split_sizes = [len(out_dimvars)] + out_bufcounts[:-1]

  def dim_handler(v, buf):
    if not v.aval.shape:
      return BoundedInt(int(buf.to_py()), v.aval._eltTy._bound)
    else:
      return Array(v.aval.shape, v.aval._eltTy, buf.to_py())

  def partitioner(bufs):
    dim_bufs, *grouped_bufs = split_list(bufs, split_sizes)
    dims_dict = dict(it.chain(
        zip(in_dim_binders, in_dim_vals),
        zip(out_dimvars, map(dim_handler, out_dimvars, dim_bufs))))
    return dims_dict, grouped_bufs
  return partitioner

def result_handler(aval):
  if isinstance(aval, AbsArray):
    return array_result_handler(aval)
  else:
    handler = xla.aval_to_result_handler(None, aval)
    return lambda _, bufs: handler(*bufs)

def array_result_handler(aval):
  if not isinstance(aval._eltTy, BaseType): raise NotImplementedError
  padded_shape = []
  for d in aval.shape:
    if isinstance(d, int):
      padded_shape.append(d)
    elif isinstance(d, Var):
      padded_shape.append(d.aval._eltTy._bound)
    elif isinstance(d, DimIndexingExpr):
      padded_shape.append(d.name.aval._eltTy._bound)
    else:
      raise NotImplementedError  # TODO
  padded_aval = core.ShapedArray(tuple(padded_shape), aval._eltTy._dtype)
  array_handler = xla.array_result_handler(None, padded_aval)
  def handler(dims_dict, bufs):
    shape = tuple(dims_dict[d] if isinstance(d, Var) else
                  DimIndexer(dims_dict[d.name], d.indices) if isinstance(d, DimIndexingExpr) else
                  d for d in aval.shape)
    return Array(shape, aval._eltTy, array_handler(*bufs))
  return handler


translations: Dict[core.Primitive, Callable] = {}


dynamic_xla_call_p = core.Primitive('dxla_call')
dynamic_xla_call_p.multiple_results = True

@dynamic_xla_call_p.def_impl
def _dynamic_xla_call_impl(*args, jaxpr, num_consts):
  in_dim_vals, consts, args = split_list(args, [len(jaxpr.in_dim_binders), num_consts])
  dim_in_avals = [v.aval for v in jaxpr.in_dim_binders]
  c = xb.make_computation_builder("dxla_call")
  dim_params, params = _make_params(c, dim_in_avals, map(xla.abstractify, args))
  const_params = _xla_consts(c, consts)
  dim_outs, outs = djaxpr_subcomp(c, jaxpr, dim_params, const_params + params)
  out = xops.Tuple(c, [o for ops in dim_outs + outs for o in ops])
  compiled = xb.get_backend(None).compile(c.build(out))
  result_handlers = map(result_handler, [v.aval for v in jaxpr.outs])
  out_bufcounts = [v.aval._num_buffers for v in jaxpr.outs]
  partitioner = result_partitioner(jaxpr.in_dim_binders, in_dim_vals,
                                   jaxpr.out_dims, out_bufcounts)
  return execute_compiled(compiled, partitioner, result_handlers,
                          in_dim_vals, args)

def djit(fun):
  def f_jitted(*args, **kwargs):
    args, in_tree = tree_flatten((args, kwargs))
    f, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
    # TODO we shouldn't dedup avals one array at a time; need to do it for the
    # full argument list!
    # unique_avals: Dict[int, core.AbstractValue] = {}
    in_avals = [core.raise_to_shaped(core.get_aval(x)) for x in args]
    jaxpr, consts, unconverted_binders = trace_to_jaxpr_dynamic(f, in_avals)
    num_consts = len(consts)
    args = [*consts, *args]
    dim_vals, args = _extract_dim_vals(jaxpr.in_dim_binders, jaxpr.in_binders,
                                       unconverted_binders, args)
    out_flat = dynamic_xla_call_p.bind(*dim_vals, *args, jaxpr=jaxpr,
                                       num_consts=num_consts)
    return tree_unflatten(out_tree(), out_flat)
  return f_jitted

def _extract_dim_vals(in_dim_binders, in_binders, unconverted_binders, args):
  converted_in_dim_vals, args = partition_list(unconverted_binders, args)
  sizes = {var: size for binder, arg in zip(in_binders, args)
           for var, size in zip(binder.aval.shape, np.shape(arg))
           if isinstance(var, Var)}
  num_binders = len(in_dim_binders) - len(converted_in_dim_vals)
  in_dim_vals = [sizes[v] for v in in_dim_binders[:num_binders]] + converted_in_dim_vals
  return in_dim_vals, args


def traceable_to_padded_translation(traceable):
  def translation(c, dims, avals, operands, **params):
    dim_avals = [core.ShapedArray((), np.int32) for _ in dims]
    padded_avals = map(_replace_vars_with_bounds, avals)

    @lu.wrap_init
    def fun(*args):
      dim_sizes, args = split_list(args, [len(dims)])
      logical_sizes = dict(zip(dims, dim_sizes))
      logical_shapes = [tuple([logical_sizes.get(d, d) for d in aval.shape])
                        for aval in avals]  # TODO more cases
      return traceable(logical_shapes, *args, **params)

    in_avals = [*dim_avals, *padded_avals]
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(fun, in_avals)

    operands_ = it.chain.from_iterable([*dims.values(), *operands])
    outs = xla.jaxpr_subcomp(c, jaxpr, None, xla.AxisEnv(1, (), ()),
                             xla._xla_consts(c, consts), '', *operands_)
    return xla._partition_outputs(out_avals, outs)
  return translation

def _replace_vars_with_bounds(aval):
  if not isinstance(aval, AbsArray):
    return aval
  else:
    new_shape = []
    for d in aval.shape:
      if isinstance(d, Var):
        assert d.aval.shape == () and isinstance(d.aval._eltTy, BoundedIntTy)
        new_shape.append(d.aval._eltTy._bound)
      elif isinstance(d, int):
        new_shape.append(d)
      elif isinstance(d, BoundedInt):
        new_shape.append(d._bound)
      else:
        raise NotImplementedError(d)
    return core.ShapedArray(tuple(new_shape), aval._eltTy._dtype)

# AD

from jax.interpreters import ad

def _dynamic_xla_call_jvp(primals, tangents, *, jaxpr, num_consts):
  del num_consts
  in_dim_vals, primals = split_list(primals, [len(jaxpr.in_dim_binders)])
  _, tangents = split_list(tangents, [len(jaxpr.in_dim_binders)])
  new_jaxpr, consts = jvp_jaxpr(jaxpr)
  outs = dynamic_xla_call_p.bind(*in_dim_vals, *consts, *primals, *tangents,
                                 jaxpr=new_jaxpr, num_consts=len(consts))
  primals_out, tangents_out = split_list(outs, [len(outs) // 2])
  return primals_out, tangents_out
ad.primitive_jvps[dynamic_xla_call_p] = _dynamic_xla_call_jvp

def _dynamic_xla_call_transpose(cts_in, *args, jaxpr, num_consts):
  # TODO make this a dynamic_xla_call_p bind
  del num_consts
  vars_to_vals = dict(
      (d, t) for v, x in zip(jaxpr.in_binders, args)
      if isinstance(v.aval, AbsArray) and not ad.is_undefined_primal(x)
      for d, t in zip(v.aval.shape, x.shape) if isinstance(d, Var))
  dim_args = [vars_to_vals[v] for v in jaxpr.in_dim_binders]
  consts_bar, args_bar = backward_pass(jaxpr, dim_args, args, cts_in)  # type: ignore
  return [*consts_bar, *args_bar]
ad.primitive_transposes[dynamic_xla_call_p] = _dynamic_xla_call_transpose

def backward_pass(jaxpr, dim_args, args, cts_in):
  primal_env = {}
  ct_env = {}

  def write_cotangent(v, ct):
    ct_env[v] = ad.add_tangents(ct_env[v], ct) if v in ct_env else ct

  def read_cotangent(v):
    return ct_env.get(v, ad.Zero(v.aval))

  def read_primal(v):
    if type(v) is core.Literal:
      raise NotImplementedError  # TODO
    else:
      return primal_env.get(v, ad.UndefinedPrimal(v.aval))

  def write_primal(v, val):
    if not ad.is_undefined_primal(val):
      primal_env[v] = val

  write_primal(core.unitvar, core.unit)
  map(write_primal, jaxpr.in_dim_binders, dim_args)
  map(write_primal, jaxpr.in_binders, args)

  map(write_cotangent, jaxpr.outs, cts_in)
  raise NotImplementedError  # TODO finish this

def jvp_jaxpr(jaxpr):
  f = lu.wrap_init(jaxpr_as_fun(jaxpr))
  dimvars = dict((v, v.aval) for v in jaxpr.in_dim_binders)
  in_avals = [_replace_vars_with_avals(dimvars, v.aval) for v in jaxpr.in_binders]
  jaxpr, consts, _ = trace_to_jaxpr_dynamic(jvp_traceable(ad.jvp(f)), in_avals * 2)
  return jaxpr, consts

def _replace_vars_with_avals(dimvars, aval):
  if isinstance(aval, AbsArray):
    shape = [dimvars.get(d, d) for d in aval.shape]
    return AbsArray(tuple(shape), aval._eltTy)
  else:
    return aval

@lu.transformation
def jvp_traceable(*primals_and_tangents):
  n = len(primals_and_tangents)
  primals, tangents = split_list(primals_and_tangents, [n // 2])
  primals_out, tangents_out = yield (primals, tangents), {}
  yield (*primals_out, *tangents_out)

def _dynamic_xla_call_pe(trace, *tracers, jaxpr, num_consts):
  in_dim_tracers, tracers = split_list(tracers, [len(jaxpr.in_dim_binders)])
  if any(not t.pval.is_known() for t in in_dim_tracers):
    raise NotImplementedError
  in_unknowns = [not t.pval.is_known() for t in tracers]
  jaxpr1, jaxpr2, out_unknowns, num_res = partial_eval_jaxpr(jaxpr, in_unknowns)

  known_tracers, unknown_tracers = partition_list(in_unknowns, tracers)
  known_vals = [t.pval.get_known() for t in known_tracers]
  in_dim_vals = [t.pval.get_known() for t in in_dim_tracers]
  outs1_res = dynamic_xla_call_p.bind(*in_dim_vals, *known_vals, jaxpr=jaxpr1,
                                      num_consts=num_consts)
  outs1, res = split_list(outs1_res, [len(jaxpr1.outs) - num_res])

  in_dim_tracers = map(trace.new_instantiated_const, in_dim_tracers)
  res_tracers = map(trace.new_instantiated_const, res)
  outs2 = [pe.JaxprTracer(trace, pe.PartialVal.unknown(v.aval), None)
           for v in jaxpr2.outs]
  eqn = pe.new_eqn_recipe(in_dim_tracers + res_tracers + unknown_tracers, outs2,
                          dynamic_xla_call_p, dict(jaxpr=jaxpr2, num_consts=0),
                          None)
  for t in outs2: t.recipe = eqn
  outs1, outs2 = iter(outs1), iter(outs2)
  return [next(outs2) if uk else next(outs1) for uk in out_unknowns]
pe.custom_partial_eval_rules[dynamic_xla_call_p] = _dynamic_xla_call_pe

def partial_eval_jaxpr(jaxpr, in_unknowns):
  env: Dict[Var, bool] = {}
  res = []

  def read(v):
    if type(v) is core.Literal:
      raise NotImplementedError  # TODO
    else:
      return env[v]

  def write(unk, v):
    env[v] = unk

  def new_res(v):
    res.append(v)
    return v

  eqns1, eqns2 = [], []
  map(write, in_unknowns, jaxpr.in_binders)
  for eqn in jaxpr.eqns:
    unks = map(read, eqn.invars)
    if any(unks):
      invars = [v if unk else new_res(v) for unk, v in zip(unks, eqn.invars)]
      eqns2.append(pe.new_jaxpr_eqn(invars, eqn.outvars, eqn.primitive,
                                    eqn.params, None))
      map(partial(write, True), eqn.outvars)
    else:
      eqns1.append(eqn)
      map(partial(write, False), eqn.outvars)
  out_unknowns = map(read, jaxpr.outs)
  out_dim_unknowns = map(read, jaxpr.out_dims)  # when linearizing, all known

  invars1, invars2 = partition_list(in_unknowns, jaxpr.in_binders)
  outvars1, outvars2 = partition_list(out_unknowns, jaxpr.outs)
  out_dims1, out_dims2 = partition_list(out_dim_unknowns, jaxpr.out_dims)

  outvars1 = outvars1 + res
  invars2 = res + invars2

  # TODO forward the correct residuals here (all dimvars used in types)
  in_dimvars2 = out_dims1 + jaxpr.in_dim_binders

  jaxpr1 = DJaxpr(jaxpr.in_dim_binders, invars1, out_dims1, outvars1, eqns1)
  jaxpr2 = DJaxpr(in_dimvars2,          invars2, out_dims2, outvars2, eqns2)

  return jaxpr1, jaxpr2, out_unknowns, len(res)


# batching

from jax.interpreters import batching

def _dynamic_xla_call_vmap(args, in_dims, *, jaxpr, num_consts):
  del num_consts
  in_dim_vals, args = split_list(args, [len(jaxpr.in_dim_binders)])
  in_dim_bdims, arg_bdims = split_list(in_dims, [len(jaxpr.in_dim_binders)])
  assert all(d is batching.not_mapped for d in in_dim_bdims)
  axis_size, = {x.shape[d] for x, d in zip(args, arg_bdims)
                if d is not batching.not_mapped}
  new_jaxpr, consts, out_dims = batch_jaxpr(jaxpr, axis_size, arg_bdims)
  outs = dynamic_xla_call_p.bind(*in_dim_vals, *consts, *args,
                                 jaxpr=new_jaxpr, num_consts=len(consts))
  return outs, out_dims
batching.primitive_batchers[dynamic_xla_call_p] = _dynamic_xla_call_vmap

def batch_jaxpr(jaxpr, axis_size, in_dims):
  dimvars = dict((v, v.aval) for v in jaxpr.in_dim_binders)
  in_avals = [_replace_vars_with_avals(dimvars, v.aval) for v in jaxpr.in_binders]

  in_avals = [core.unmapped_aval(axis_size, d, aval)
              if d is not batching.not_mapped else aval
              for d, aval in zip(in_dims, in_avals)]

  fun, out_dims = batching.batch_subtrace(lu.wrap_init(jaxpr_as_fun(jaxpr)))
  f = _batch_fun(fun, in_dims)
  jaxpr, consts, _ = trace_to_jaxpr_dynamic(f, in_avals)
  return jaxpr, consts, out_dims()

@lu.transformation
def _batch_fun(in_dims, *in_vals, **params):
  with core.new_main(batching.BatchTrace, axis_name=None) as main:
    out_vals = yield (main, in_dims, *in_vals), params
    del main
  yield out_vals

def _map_array(size: int, axis: int, aval: AbsArray) -> AbsArray:
  return AbsArray(tuple_delete(aval.shape, axis), aval._eltTy)

def _unmap_array(size: int, axis: int, aval: AbsArray) -> AbsArray:
  raise NotImplementedError

core.aval_mapping_handlers[AbsArray] = _map_array, _unmap_array


# Primitives

import numpy as np
from jax._src.lax import lax


## sin

def sin(x: Any) -> Any:
  return sin_p.bind(x)
sin_p = core.Primitive('sin_p')

@sin_p.def_abstract_eval
def _sin_abstract_eval(x):
  if isinstance(x, AbsArray):
    return AbsArray(x.shape, x._eltTy)
  else:
    return lax.sin_p.abstract_eval(x)

def _sin_typecheck_rule(invar):
  return [invar.aval]
typecheck_rules[sin_p] = _sin_typecheck_rule

def _sin_translation_rule(c, dims, avals, operands):
  (x,), = operands
  return [[xops.Sin(x)]]
translations[sin_p] = _sin_translation_rule

ad.defjvp(sin_p, lambda g, x: mul(g, cos(x)))


## cos

def cos(x: Any) -> Any:
  return cos_p.bind(x)
cos_p = core.Primitive('cos_p')

@cos_p.def_abstract_eval
def _cos_abstract_eval(x):
  if isinstance(x, AbsArray):
    return AbsArray(x.shape, x._eltTy)
  else:
    return lax.cos_p.abstract_eval(x)

def _cos_typecheck_rule(invar):
  return [invar.aval]
typecheck_rules[cos_p] = _cos_typecheck_rule

def _cos_translation_rule(c, dims, avals, operands):
  (x,), = operands
  return [[xops.Cos(x)]]
translations[cos_p] = _cos_translation_rule


## reduce-sum

def reduce_sum(x: Any, axes: Optional[Sequence[int]] = None) -> Any:
  if axes is None:
    axes = tuple(range(len(x.shape)))
  return reduce_sum_p.bind(x, axes=axes)
reduce_sum_p = core.Primitive('reduce_sum')

@reduce_sum_p.def_abstract_eval
def _sum_abstract_eval(operand, *, axes):
  if isinstance(operand, AbsArray):
    axes = set(axes)
    new_shape = [d for i, d in enumerate(operand.shape) if i not in axes]
    if (all(isinstance(d, int) for d in new_shape) and
        isinstance(operand._eltTy, BaseType)):
      return core.ShapedArray(tuple(new_shape), operand._eltTy._dtype)
    else:
      return AbsArray(tuple(new_shape), operand._eltTy)
  else:
    return lax.reduce_sum_p.reduce_sum_abstract_eval(operand, axes=axes)

def _reduce_sum_typecheck_rule(x, *, axes):
  return [reduce_sum_p.abstract_eval(x.aval, axes=axes)]
typecheck_rules[reduce_sum_p] = _reduce_sum_typecheck_rule

def _reduce_sum_translation_traceable(logical_shapes, x, *, axes):
  shape, = logical_shapes
  x = _replace_masked_values(shape, x, 0, axes=axes)
  return [lax._reduce_sum(x, axes=axes)]
translations[reduce_sum_p] = traceable_to_padded_translation(
    _reduce_sum_translation_traceable)

def _replace_masked_values(logical_shape, x, val, axes=None):
  axes = axes or set(range(len(logical_shape)))
  masks = [lax.broadcasted_iota(np.int32, x.shape, i) < d
           for i, d in enumerate(logical_shape) if d is not None and i in axes]
  if masks:
    x = lax.select(reduce(op.and_, masks), x, lax.full_like(x, val))
  return x

def _reduce_sum_transpose_rule(cotangent, operand, *, axes):
  raise NotImplementedError  # TODO
ad.deflinear2(reduce_sum_p, _reduce_sum_transpose_rule)


### lt

def lt(x, y):
  return lt_p.bind(x, y)
lt_p = core.Primitive('lt')

@lt_p.def_abstract_eval
def _lt_abstract_eval(x, y):
  if isinstance(x, AbsArray) or isinstance(y, AbsArray):
    # TODO check dtypes match
    if not x.shape:
      return AbsArray(y.shape, BaseType(np.dtype('bool')))
    if not y.shape:
      return AbsArray(x.shape, BaseType(np.dtype('bool')))
    map(_dims_must_equal, x.shape, y.shape)
    return AbsArray(x.shape, BaseType(np.dtype('bool')))
  else:
    return lax.lt_p.abstract_eval(x, y)

def _lt_typecheck_rule(x, y):
  return [lt_p.abstract_eval(x.aval, y.aval)]

def _lt_translation_rule(c, dims, avals, operands):
  (x,), (y,) = operands
  return [[xops.Lt(x, y)]]


### dot

def dot(x, y):
  assert len(x.shape) == len(y.shape) == 2
  return dot_general(x, y, ([1], [0]), ([], []))

Dims = Tuple[Sequence[int], Sequence[int]]

def dot_general(x: Any, y: Any, contract: Dims, batch: Dims) -> Any:
  return dot_general_p.bind(x, y, contract=contract, batch=batch)
dot_general_p = core.Primitive('dot_general')

@dot_general_p.def_abstract_eval
def _dot_general_abstract_eval(x, y, *, contract, batch):
  for i, j in zip(*contract): _dims_must_equal(x.shape[i], y.shape[j])
  for i, j in zip(*batch): _dims_must_equal(x.shape[i], y.shape[j])
  shape = lax._dot_general_shape_computation(x.shape, y.shape, (contract, batch))
  return AbsArray(shape, x._eltTy)

def _dot_general_typecheck_rule(x, y, *, contract, batch):
  return [_dot_general_abstract_eval(x.aval, y.aval,
                                     contract=contract, batch=batch)]
typecheck_rules[dot_general_p] = _dot_general_typecheck_rule

def _dot_general_trans(logical_shapes, x, y, *, contract, batch):
  x_shape, _ = logical_shapes
  lhs_contract, _ = contract
  x = _replace_masked_values(x_shape, x, 0, axes=lhs_contract)
  return [lax.dot_general(x, y, dimension_numbers=(contract, batch))]
translations[dot_general_p] = traceable_to_padded_translation(_dot_general_trans)

def _dot_general_transpose_rule(cotangent, x, y, *, contract, batch):
  assert False  # TODO
ad.primitive_transposes[dot_general_p] = _dot_general_transpose_rule


## add

def add(x: Any, y: Any) -> Any:
  return add_p.bind(x, y)
add_p = core.Primitive('add')

@add_p.def_abstract_eval
def _add_abstract_eval(x, y):
  if isinstance(x, AbsArray) and isinstance(y, AbsArray):
    map(_dims_must_equal, x.shape, y.shape)  # TODO broadcasting?
    return AbsArray(x.shape, x._eltTy)
  else:
    return lax.add_p.abstract_eval(x, y)

def _dims_must_equal(d1, d2):
  if isinstance(d1, (Tracer, Var)) and isinstance(d2, (Tracer, Var)):
    if d1.aval is d2.aval: return True
  elif isinstance(d1, int) and isinstance(d2, int):
    return d1 == d2
  raise Exception("can't prove shapes equal (or unequal)!")

def _add_typecheck_rule(x, y):
  return [add_p.abstract_eval(x.aval, y.aval)]
typecheck_rules[add_p] = _add_typecheck_rule

def _add_translation_rule(c, dims, avals, operands):
  (x,), (y,) = operands
  return [[xops.Add(x, y)]]
translations[add_p] = _add_translation_rule


## mul

def mul(x: Any, y: Any) -> Any:
  return mul_p.bind(x, y)
mul_p = core.Primitive('mul')

@mul_p.def_abstract_eval
def _mul_abstract_eval(x, y):
  if isinstance(x, AbsArray) and isinstance(y, AbsArray):
    map(_dims_must_equal, x.shape, y.shape)  # TODO broadcasting?
    return AbsArray(x.shape, x._eltTy)
  else:
    return lax.mul_p.abstract_eval(x, y)

def _mul_typecheck_rule(x, y):
  return [mul_p.abstract_eval(x.aval, y.aval)]
typecheck_rules[mul_p] = _mul_typecheck_rule

def _mul_translation_rule(c, dims, avals, operands):
  (x,), (y,) = operands
  return [[xops.Mul(x, y)]]
translations[mul_p] = _mul_translation_rule


## nonzero

def nonzero(x):
  return nonzero_p.bind(x)
nonzero_p = core.Primitive('nonzero')

def _nonzero_unpack_result(x):
  return [x.shape[-1], x]
nonzero_p.unpack_result = _nonzero_unpack_result  # type: ignore

def _nonzero_staging_rule(trace, tracers, params):
  aval = tracers[0].aval
  if isinstance(aval, AbsArray) and not isinstance(aval._eltTy, BaseType):
    raise NotImplementedError
  bound = aval.shape[-1]
  bound = bound if isinstance(bound, int) else bound._bound
  out_dim_aval = AbsArray(aval.shape[:-1], BoundedIntTy(bound))
  out_dim_tracer = pe.DynamicJaxprTracer(trace, out_dim_aval, None)
  if len(aval.shape) == 1:
    out_val_aval = AbsArray((out_dim_tracer,), BaseType(np.dtype('int32')))
  else:
    indices = tuple(range(len(aval.shape[:-1])))
    expr = DimIndexingExpr(out_dim_tracer, indices)
    out_val_aval = AbsArray((*aval.shape[:-1], expr),
                              BaseType(np.dtype('int32')))
  out_val_tracer = pe.DynamicJaxprTracer(trace, out_val_aval, None)
  invars = map(trace.getvar, tracers)
  outvars = map(trace.makevar, [out_dim_tracer, out_val_tracer])
  eqn = pe.new_jaxpr_eqn(invars, outvars, nonzero_p, {}, None)
  trace.frame.eqns.append(eqn)
  return out_val_tracer
custom_staging_rules[nonzero_p] = _nonzero_staging_rule

def _nonzero_typecheck_rule(invar):
  bound = invar.aval.shape[-1]
  bound = bound if isinstance(bound, int) else bound._bound
  newvar = core.gensym()
  out_dim_var = newvar(AbsArray(invar.aval.shape[:-1], BoundedIntTy(bound)))
  if len(invar.aval.shape) == 1:
    out_val_aval = AbsArray((out_dim_var,), BaseType(np.dtype('int32')))
  else:
    indices = tuple(range(len(out_dim_var.aval.shape)))  # pytype: disable=attribute-error
    expr = DimIndexingExpr(out_dim_var, indices)
    out_val_aval = AbsArray((*invar.aval.shape[:-1], expr),
                              BaseType(np.dtype('int32')))
  return out_dim_var, out_val_aval
typecheck_rules[nonzero_p] = _nonzero_typecheck_rule

def _nonzero_translation_traceable(logical_shapes, x):
  shape, = logical_shapes
  assert shape
  x = _replace_masked_values(shape, x, 0)
  nonzero_indicators = x != 0
  last_axis = len(shape) - 1
  out_sizes = lax._reduce_sum(nonzero_indicators.astype(np.int32), [last_axis])
  iota = lax.broadcasted_iota(np.int32, x.shape, dimension=last_axis)
  _, idx = lax.sort_key_val(~nonzero_indicators, iota, dimension=last_axis)
  return out_sizes, idx
translations[nonzero_p] = traceable_to_padded_translation(
    _nonzero_translation_traceable)

def _nonzero_vmap_rule(args, in_dims):
  (x,), (d,) = args, in_dims
  if d != 0: raise NotImplementedError
  return nonzero_p.bind(x), 0
batching.primitive_batchers[nonzero_p] = _nonzero_vmap_rule


## iota

def iota(n):
  return iota_p.bind(n)
iota_p = core.Primitive('iota')

def _iota_staging_rule(trace, tracers, params):
  tracer, = tracers
  n = trace.get_const(tracer)
  if n is not None:
    if type(n) is not int: raise NotImplementedError  # TODO batched version?
    out_aval = core.ShapedArray((n,), np.dtype('int32'))
    out_tracer = pe.DynamicJaxprTracer(trace, out_aval, None)
    outvar = trace.makevar(out_tracer)
    eqn = pe.new_jaxpr_eqn([], [outvar], iota_p, dict(size=n), None)
  else:
    aval = tracer.aval
    if not isinstance(aval, AbsArray): raise TypeError
    if aval.shape:
      indices = tuple(range(len(aval.shape)))
      out_aval = AbsArray((*aval.shape, DimIndexingExpr(tracer, indices)),
                             BaseType(np.dtype('int32')))
    else:
      out_aval = AbsArray((tracer,), BaseType(np.dtype('int32')))
    out_tracer = pe.DynamicJaxprTracer(trace, out_aval, None)
    outvar = trace.makevar(out_tracer)
    invar = trace.getvar(tracer)
    eqn = pe.new_jaxpr_eqn([invar], [outvar], iota_p, {}, None)
  trace.frame.eqns.append(eqn)
  return out_tracer
custom_staging_rules[iota_p] = _iota_staging_rule

def _iota_typecheck_rule(*invars, size=None):
  if size is not None:
    if invars: raise TypeError
    return [core.ShapedArray((size,), np.dtype('int32'))]
  else:
    invar, = invars
    if not invar.aval.shape:
      return [AbsArray((invar,), BaseType(np.dtype('int32')))]
    else:
      indices = tuple(range(len(invar.aval.shape)))
      return [AbsArray((*invar.aval.shape, DimIndexingExpr(invar, indices)),
                         BaseType(np.dtype('int32')))]
typecheck_rules[iota_p] = _iota_typecheck_rule

def _iota_translation_rule(c, dims, avals, operands, *, size=None):
  if size is None:
    aval, = avals
    size = aval._eltTy._bound
    shape = aval.shape
  else:
    shape = ()
  etype = xc.dtype_to_etype(np.dtype('int32'))
  xla_shape = xc.Shape.array_shape(etype, (*shape, size))
  return [[xops.Iota(c, xla_shape, len(shape))]]
translations[iota_p] = _iota_translation_rule


## broadcast

def broadcast(x, d):
  return broadcast_p.bind(x, d)
broadcast_p = core.Primitive('broadcast')

def _broadcast_staging_rule(trace, tracers, params):
  x, d = tracers
  d_const = trace.get_const(d)
  if d_const is not None:
    raise NotImplementedError  # TODO
  else:
    aval = x.aval
    dtype = aval._eltTy._dtype if isinstance(aval, AbsArray) else aval.dtype
    out_aval = AbsArray((d, *x.shape), BaseType(dtype))
    out_tracer = pe.DynamicJaxprTracer(trace, out_aval, None)
    eqn = pe.new_jaxpr_eqn([trace.getvar(x), trace.getvar(d)],
                           [trace.makevar(out_tracer)], broadcast_p, {}, None)
    trace.frame.eqns.append(eqn)
    return out_tracer
custom_staging_rules[broadcast_p] = _broadcast_staging_rule

def _broadcast_typecheck_rule(x, d):
  aval = x.aval
  dtype = aval._eltTy._dtype if isinstance(aval, AbsArray) else aval.dtype
  return [AbsArray((d, *x.aval.shape), BaseType(dtype))]
typecheck_rules[broadcast_p] = _broadcast_typecheck_rule

def _broadcast_translation_rule(c, dims, avals, operands, *, size=None):
  (x,), (_,) = operands
  if size is None:
    _, aval = avals
    assert not aval.shape
    size = aval._eltTy._bound
  return [[xops.Broadcast(x, (size,))]]
translations[broadcast_p] = _broadcast_translation_rule


# Examples

import jax.numpy as jnp

def bbarray(bound_shape: Tuple[int, ...], x: NDArray):
  sizes: Dict[int, BoundedInt] = {}
  shape = tuple(sizes.setdefault(d, BoundedInt(d, bound))
                for d, bound in zip(x.shape, bound_shape))
  slices = tuple(slice(d) for d in x.shape)
  padded_x = jnp.ones(bound_shape, x.dtype).at[slices].set(x)
  return Array(shape, BaseType(x.dtype), padded_x)

def ones_like(x):
  if isinstance(x, Array):  # doesn't work with tracers
    return Array(x.shape, x._eltTy, jnp.ones_like(x._data))
  else:
    return jnp.ones_like(x)


if __name__ == '__main__':
  import jax
  jax.config.update('jax_platform_name', 'cpu')
  def p(s): print('\n--- ' + str(s))

  ## Staging and typechecking

  p('typecheck identity')
  def f(x):
    return x
  x = jnp.array([0, 1])
  jaxpr, _, _ = make_djaxpr(f, x)
  print(jaxpr)
  print(typecheck_jaxpr(jaxpr))

  p('typecheck sin')
  def f(x):
    return sin(x)
  x = bbarray((5,), jnp.arange(3.))
  jaxpr, _, _ = make_djaxpr(f, x)
  print(jaxpr)
  print(typecheck_jaxpr(jaxpr))

  p('typecheck sin-and-add')
  def f(x):
    y = sin(x)
    z = sin(y)
    return add(y, z)
  x = bbarray((5,), jnp.arange(3.))
  jaxpr, _, _ = make_djaxpr(f, x)
  print(jaxpr)
  print(typecheck_jaxpr(jaxpr))

  p('typecheck iota(3)')
  def f():  # type: ignore
    return iota(3)
  jaxpr, _, _ = make_djaxpr(f)
  print(jaxpr)
  print(typecheck_jaxpr(jaxpr))

  p('typecheck nonzero')
  def f(x):
    return nonzero(x)
  x = jnp.array([1, 0, -2, 0, 3, 0])
  jaxpr, _, _ = make_djaxpr(f, x)
  print(jaxpr)
  print(typecheck_jaxpr(jaxpr))

  p('typecheck sum-of-nonzero')
  def f(x):
    return reduce_sum(nonzero(x), tuple(range(len(x.shape))))
  x = jnp.array([1, 0, -2, 0, 3, 0])
  jaxpr, _, _ = make_djaxpr(f, x)
  print(jaxpr)
  print(typecheck_jaxpr(jaxpr))


  ## XLA lowering and execution

  @djit
  def f(x):
    nonzero_idx = nonzero(x)
    return reduce_sum(nonzero_idx)
  p('execute sum of nonzero indices')
  x = jnp.array([0, 1, 0, 1, 0, 1])
  print(f(x))
  print('should be', np.sum(np.nonzero(x)[0]))

  @djit
  def f(x):
    return nonzero(x)
  p('execute nonzero')
  x = jnp.array([0, 1, 0, 1, 0, 1])
  print(f(x))
  print('should be', np.nonzero(x)[0])

  @djit
  def f(i):
    return iota(i)
  p('execute iota')
  print(f(BoundedInt(3, 5)))
  print('should be', np.arange(3))

  @djit
  def f(x, n):
    y = nonzero(x)
    return broadcast(y, n)
  p('execute broadcast')
  x = np.arange(3)
  n = BoundedInt(4, 5)
  print(f(x, n))  # type: ignore
  print(f'should be\n{np.broadcast_to(np.nonzero(x)[0], (4, 2))}')


  ## ad

  @djit
  def f(x):
    y = sin(x)
    return reduce_sum(y, axes=(0,))
  x = bbarray((5,), jnp.arange(2.))
  p('basic jvp')
  z, z_dot = jax.jvp(f, (x,), (ones_like(x),))
  print(z, z_dot)


  p('basic linearize')
  _, f_lin = jax.linearize(f, x)
  print(f_lin(ones_like(x)))


  ## vmap

  @djit
  def f(x):
    return nonzero(x)
  p('vmap of nonzero')
  xs = jnp.array([[0, 1, 0, 1, 0, 1],
                  [1, 1, 1, 1, 0, 1]])
  print(jax.vmap(f)(xs))


  ## dot

  @djit
  def f(x):
    return dot(x, x)
  p('dot(x, x)')
  x = bbarray((4, 4), np.arange(9., dtype=np.float32).reshape(3, 3))
  print(f(x))
  y = np.arange(9.).reshape(3, 3)
  print(f'should be\n{np.dot(y, y)}')
