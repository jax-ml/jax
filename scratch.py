import pdb, sys, traceback
def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    pdb.pm()
sys.excepthook = info

from functools import partial
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from jax._src import core

# ========= in jax =======

class HiPrimitive(core.Primitive):
  pass

from jax._src.interpreters import partial_eval as pe
from jax._src import linear_util as lu
from jax._src.util import safe_map as map, safe_zip as zip
from jax._src.api_util import flatten_fun_nokwargs
from jax._src.tree_util import tree_flatten, tree_unflatten

def hijax_to_lojax(hi_jaxpr: core.ClosedJaxpr) -> core.ClosedJaxpr:
  # TODO not to_rep... tree flattening would be perfect! but all we have are
  # types... want values...
  lo_in_avals = [x.to_rep() if isinstance(x, JaxType) else x
                  for x in hi_jaxpr.in_avals]
  flat_lo_in_avals, in_tree = tree_flatten(lo_in_avals)
  f = lu.wrap_init(partial(_eval_hi_to_lo, hi_jaxpr.jaxpr, hi_jaxpr.consts))
  f, out_tree = flatten_fun_nokwargs(f, in_tree)
  lo_jaxpr, _, lo_consts, () = pe.trace_to_jaxpr_dynamic(f, flat_lo_in_avals)
  breakpoint()
  return core.ClosedJaxpr(lo_jaxpr, lo_consts)

LoVal = Any  # Tracer | list[Tracer]
LoVal_ = Any  # Tracer | HiType[Tracer]

def _eval_hi_to_lo(jaxpr, consts, *args: LoVal):
  args_ = [

  def read(x: core.Atom) -> LoVal:
    return x.val if isinstance(x, core.Literal) else env[x]

  def write(v: core.Var, val: LoVal) -> None:
    env[v] = val

  env: dict[core.Var, LoVal] = {}
  map(write, jaxpr.invars, args)
  map(write, jaxpr.constvars, consts)
  for e in jaxpr.eqns:
    if isinstance(e.primitive, HiPrimitive):
      breakpoint()
      pass
    else:
      breakpoint()
      pass
  return map(read, jaxpr.outvars)




class JaxVal:
  @classmethod
  def new(cls, *args):
    return cls._from_rep_p.bind(*args)

  def type_of(self):
    assert False, "subclass should implement"

  @staticmethod
  def constructor_bwd(t):
    assert False, "subclass should implement"

class JaxType:
  def tangent_type_of(self):
    assert False, "subclass should implement"

  def __eq__(self):
    assert False, "subclass should implement"

def register_fancy_type(cls, ty, attr_names):
  _from_rep_p = core.Primitive(cls.__name__ + ".new")
  def _from_rep_impl(*args):
    val = cls.__new__(cls)
    val.__init__(*args)
    return val
  _from_rep_p.def_impl(_from_rep_impl)
  _from_rep_p.def_abstract_eval(lambda _: ty())
  cls._from_rep_p = _from_rep_p
  ty._from_rep_impl = _from_rep_impl  # TODO

  _splat_p = core.Primitive(cls.__name__ + ".splat")
  _splat_p.multiple_results = True
  def _splat_impl(val):
    return tuple(getattr(val, attr_name) for attr_name in attr_names)
  _splat_p.def_impl(_splat_impl)
  _splat_p.def_abstract_eval(lambda ty: (ty.to_rep(),) )
  cls._splat_p = _splat_p

  core.pytype_aval_mappings[cls] = lambda x: x.type_of()
  core.raise_to_shaped_mappings[ty] = lambda x, _: x

  for i, attr in enumerate(attr_names):
    setattr(ty, attr, core.aval_property(lambda self: _splat_p.bind(self)[i]))

  ty.str_short = lambda t, short_dtypes=False: ty.__name__

# ============== in library land ========

# wraps a float32
# tangent type is a float16
class FancyValCls(JaxVal):
  def __init__(self, val:float):
    assert jnp.shape(val) == ()
    assert jnp.result_type(val) == np.float32
    self.val = val

  def type_of(self):
    return FancyType()

  def to_rep(self):
    return self._val

  def splat_bwd(t:tuple):
    val_t, = t
    return val_t.astype(np.float16)

  def constructor_bwd(t): # :TangentType[FancyType]):
    return t.astype(np.float32)

class FancyType(JaxType):
  def to_rep(self):
    return core.ShapedArray((), np.dtype("float32"))

  def tangent_type_of(self):
    return core.ShapedArray((), np.dtype("float16"))

register_fancy_type(FancyValCls, FancyType, ["val"])

FancyVal = FancyValCls.new

@jax.custom_vjp
def fancy_sin(x):
  return FancyVal(jnp.sin(x.val))

def fancy_sin_fwd(x):
  assert False

def fancy_sin_bwd(rest):
  assert False

fancy_sin.defvjp(fancy_sin_fwd, fancy_sin_bwd)

# ============== in user land ========

if __name__ == '__main__':
  def my_function(x:float):
    return fancy_sin(FancyVal(x)).val

  # print(my_function(1.0))
  print(jax.jit(my_function)(1.0))
  # print(jax.grad(my_function)(1.0))

