import ipdb, sys, traceback
def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()
sys.excepthook = info

from functools import partial
from typing import TypeVar, Any, Sequence, List, Tuple, Generic, Callable

import jax
import jax.numpy as jnp

from jax import core
from jax import linear_util as lu
from jax.util import safe_map, safe_zip, split_list
from jax.tree_util import (tree_flatten, tree_unflatten, tree_structure,
                           treedef_tuple)
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import mlir
from jax._src import ad_util
from jax._src.api_util import flatten_fun_nokwargs
import jax._src.pretty_printer as pp

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

# State effect
class State: pass
State = State()


## primitives for the state effect

def ref_get(ref, idx):
  return get_p.bind(ref, *idx)

get_p = core.Primitive('get')

@get_p.def_effectful_abstract_eval
def get_abstract_eval(ref, *idx):
  return core.ShapedArray(ref.shape[len(idx):], ref.dtype), {State}

def _get_jvp(primals, tangents):
  primal_ref, *idx = primals
  tangent_ref, *_ = tangents
  return ref_get(primal_ref, idx), ref_get(tangent_ref, idx)
ad.primitive_jvps[get_p] = _get_jvp

def _get_pp_rule(eqn, context, settings):
  y, = eqn.outvars
  x, *idx = eqn.invars
  # pretty-print `y = get x i` as `x[i] := v`
  idx = ','.join(core.pp_var(i, context) for i in idx)
  lhs = core.pp_vars([y], context, print_shapes=settings.print_shapes)
  return pp.concat([lhs, pp.text(' = '),
                    pp.text(core.pp_var(x, context)),
                    pp.text('['), pp.text(idx), pp.text(']')])
core.pp_eqn_rules[get_p] = _get_pp_rule



def ref_set(ref, idx, x):
  _ = ref_swap(ref, idx, x)
def ref_swap(ref, idx, x):
  return swap_p.bind(ref, *idx, x)
swap_p = core.Primitive('swap')

@swap_p.def_effectful_abstract_eval
def swap_abstract_eval(ref, *idx_x):
  *idx, x = idx_x
  return core.raise_to_shaped(x), {State}

def _swap_jvp(primals, tangents):
  primal_ref, *idx, x = primals
  tangent_ref, *_, xdot = tangents
  return ref_swap(primal_ref, idx, x), ref_swap(tangent_ref, idx, xdot)
ad.primitive_jvps[swap_p] = _swap_jvp

def _swap_pp_rule(eqn, context, settings):
  y, = eqn.outvars
  x, *idx, v = eqn.invars
  idx = ','.join(core.pp_var(i, context) for i in idx)
  if type(y) is core.DropVar:
    # pretty-print `_ = swap x i v` as `x[i] := v`
    del y
    return pp.concat([pp.text(core.pp_var(x, context)),
                      pp.text('['), pp.text(idx), pp.text('] := '),
                      pp.text(core.pp_var(v, context))])
  else:
    # pretty-print `y:T = swap x i v` as `y:T, x[i] = x[i], v`
    x_i = pp.concat([pp.text(core.pp_var(x, context)),
                     pp.text('['), pp.text(idx), pp.text(']')])
    y = core.pp_vars([y], context, print_shapes=settings.print_shapes)
    return pp.concat([y, pp.text(', '), x_i, pp.text(' := '),
                      x_i, pp.text(', '), pp.text(core.pp_var(v, context))])
core.pp_eqn_rules[swap_p] = _swap_pp_rule


def ref_addupdate(ref, idx, x):
  addupdate_p.bind(ref, *idx, x)
addupdate_p = core.Primitive('addupdate')
addupdate_p.multiple_results = True

@addupdate_p.def_effectful_abstract_eval
def addupdate_abstract_eval(ref, *idx_x):
  del ref, idx_x  # Unused.
  return [], {State}

def _addupdate_pp_rule(eqn, context, settings):
  () = eqn.outvars
  x, *idx, v = eqn.invars
  # pretty-print ` = addupdate x i v` as `x[i] += v`
  idx = ','.join(core.pp_var(i, context) for i in idx)
  return pp.concat([pp.text(core.pp_var(x, context)),
                    pp.text('['), pp.text(idx), pp.text('] += '),
                    pp.text(core.pp_var(v, context))])
core.pp_eqn_rules[addupdate_p] = _addupdate_pp_rule



## aval for refs

class ShapedArrayRef(core.AbstractValue):
  __slots__ = ['shape', 'dtype']

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  def _getitem(self, tracer, idx):
    if not isinstance(idx, tuple):
      idx = idx,
    return ref_get(tracer, idx)

  def _setitem(self, tracer, idx, val):
    if not isinstance(idx, tuple):
      idx = idx,
    return ref_set(tracer, idx, val)

  def __repr__(self) -> str:
    a = core.ShapedArray(self.shape, self.dtype)
    return f'Ref{{{a.str_short()}}}'

  def at_least_vspace(self):
    return self

core.raise_to_shaped_mappings[ShapedArrayRef] = lambda aval, _: aval

def prnt(jaxpr):
  jaxpr = getattr(jaxpr, 'jaxpr', jaxpr)
  return print(jaxpr.pretty_print(use_color=True))


# @lu.wrap_init
# def f(i, r):
#   x = r[i]
#   r[i] = 2 * x
#   return x + 1,  # flat
# in_avals = [core.ShapedArray((), jnp.dtype('int32')),
#             ShapedArrayRef((4,), jnp.dtype('float32'))]
# jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(f, in_avals)
# prnt(jaxpr)


# AD

def f(r):
  x = r[0]
  r[1] = jnp.cos(x)

in_avals = [ShapedArrayRef((4,), jnp.dtype('float32'))]
jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(lu.wrap_init(lambda r: f(r) or ()), in_avals)
prnt(jaxpr)

print("==> JVP ==>")

@lu.wrap_init
def g(r, rdot):
  jax.jvp(f, (r,), (rdot,))
  return ()

in_avals = [ShapedArrayRef((4,), jnp.dtype('float32')),
            ShapedArrayRef((4,), jnp.dtype('float32'))]
jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(g, in_avals)
prnt(jaxpr)


print("==> PE ==>")
# pe._partial_eval_jaxpr_custom handles effects, at least these ones!
jaxpr_known, jaxpr_staged_, out_unk, out_inst, num_res = \
    pe._partial_eval_jaxpr_custom(jaxpr, [False, True], lambda *_: True)
prnt(jaxpr_known)
jaxpr_staged, _ = pe.dce_jaxpr(jaxpr_staged_,
                               [True] * len(jaxpr_staged_.outvars))
prnt(jaxpr_staged)
# so just call this sucker in the partial eval rule for the loop


def _get_transpose(g, ref, *idx):
  ref_addupdate(ref, idx, g)
  return [None, None]
ad.primitive_transposes[get_p] = _get_transpose

def _swap_transpose(g, ref, *idx_x):
  *idx, x = idx_x
  x_bar = ref_swap(ref, idx, ad_util.instantiate(g))
  return [None, None, x_bar]
ad.primitive_transposes[swap_p] = _swap_transpose

print("==> TRANSPOSE ==>")

avals = [x.aval for x in jaxpr_staged.outvars]
def trans(res, ref):
  ad.backward_pass(jaxpr_staged, (), (), (), (res, ref), ())
  return []
jaxpr_trans, _, _ = pe.trace_to_jaxpr_dynamic(
    lu.wrap_init(trans), [core.ShapedArray((), jnp.dtype('float32')),
                          ShapedArrayRef((4,), jnp.dtype('float32'))])
prnt(jaxpr_trans)


# discharge!

def discharge_state(jaxpr: core.Jaxpr, consts: Sequence[Any]
                    ) -> Tuple[core.Jaxpr, List[Any]]:
  in_avals = [core.ShapedArray(v.aval.shape, v.aval.dtype)
              if type(v.aval) is ShapedArrayRef
              else v.aval for v in jaxpr.invars]
  eval_jaxpr = lu.wrap_init(partial(_eval_jaxpr_discharge_state, jaxpr, consts))
  new_jaxpr, _, new_consts = pe.trace_to_jaxpr_dynamic(eval_jaxpr, in_avals)
  return new_jaxpr, new_consts

def _eval_jaxpr_discharge_state(jaxpr, consts: List[Any], *args: Any):
  env: Dict[core.Var, Any] = {}

  def read(x: core.Atom) -> Any:
    if type(x) is core.Literal:
      return x.val
    return env[x]

  def write(v: core.Var, val: Any) -> None:
    env[v] = val

  write(core.unitvar, core.unit)
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    if eqn.primitive is get_p:
      # `y = x[i]` becomes `y = ds x i`
      x, *idx = in_vals
      write(eqn.outvars[0], dynamic_index(x, idx))
    elif eqn.primitive is swap_p:
      # `x_i = swap(x[i], val)` becomes `x_i = ds x i; updated_x = dus x i val`
      x, *idx, val = in_vals
      write(eqn.outvars[0], dynamic_index(x, idx))
      write(eqn.invars[0], dynamic_update_index(x, idx, val))
    elif eqn.primitive is addupdate_p:
      # `x[i] += val` becomes `x = dus x i (val + (ds x i))
      x, *idx, val = in_vals
      ans = jax.lax.dynamic_update_index(x, idx, val + dynamic_index(x, idx))
      write(eqn.invars[0], ans)
    else:
      # standard eval_jaxpr stuff (NOTE assumes no State effects possible here!)
      subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
      ans = eqn.primitive.bind(*subfuns, *map(read, eqn.invars), **bind_params)
      if eqn.primitive.multiple_results:
        map(write, eqn.outvars, ans)
      else:
        write(eqn.outvars[0], ans)
  # assert not jaxpr.outvars  # TODO remove this
  return [read(v) for v in jaxpr.invars if type(v.aval) is ShapedArrayRef]

def dynamic_index(x, idx):
  if not idx: return x
  ndim = len(x.shape)
  starts = [*idx] + [jax.lax.full_like(idx[0], 0, shape=())] * (ndim - len(idx))
  sizes = (1,) * len(idx) + x.shape[len(idx):]
  out = jax.lax.dynamic_slice(x, starts, sizes)
  return out.reshape(x.shape[len(idx):])

def dynamic_update_index(x, idx, val):
  if not idx: return val
  ndim = len(x.shape)
  starts = [*idx] + [jax.lax.full_like(idx[0], 0, shape=())] * (ndim - len(idx))
  update = val.reshape((1,) * len(idx) + x.shape[len(idx):])
  return jax.lax.dynamic_update_slice(x, update, starts)

# loop

print('loop!!!!')

# Type annotations
S = TypeVar('S')
class Ref(Generic[TypeVar('T')]): pass

def abstractify(x: Any) -> core.AbstractValue:
  return core.raise_to_shaped(core.get_aval(x))

# for: Int -> (Int -> Ref s -> {State s} ()) -> s -> s
def for_loop(nsteps: int, body: Callable[[int, Ref[S]], None],
             init_state: S) -> S:
  init_state, state_tree = tree_flatten(init_state)
  jaxpr, consts = _trace_to_jaxpr(body, state_tree, map(make_ref, init_state))
  out_flat = for_p.bind(*consts, *init_state, jaxpr=jaxpr, nsteps=int(nsteps))
  return tree_unflatten(state_tree, out_flat)
for_p = core.Primitive('for')
for_p.multiple_results = True

def _trace_to_jaxpr(f, state_tree, state_avals):
  f, out_tree = flatten_fun_nokwargs(
      lu.wrap_init(f), treedef_tuple((tree_structure(0), state_tree)))
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      f, [core.ShapedArray((), jnp.dtype('int32')), *state_avals])
  if out_tree() != tree_structure(None): raise Exception
  return pe.convert_constvars_jaxpr(jaxpr), consts

def make_ref(x) -> ShapedArrayRef:
  aval = core.raise_to_shaped(core.get_aval(x))
  if type(aval) is not core.ShapedArray:
    raise Exception(f"can't make ref from {x}")
  if not aval.shape:
    raise Exception(f"can't make ref from value with scalar shape {aval.shape}")
  return ShapedArrayRef(aval.shape, aval.dtype)

@for_p.def_abstract_eval
def _for_abstract_eval(*_, jaxpr, **__):
  return [core.ShapedArray(v.aval.shape, v.aval.dtype) for v in jaxpr.invars
          if type(v.aval) is ShapedArrayRef]

for_p.def_impl(partial(xla.apply_primitive, for_p))

def _for_impl(*args, jaxpr, nsteps):
  lowered_jaxpr, consts = discharge_state(jaxpr, ())
  def cond(carry):
    i, _ = carry
    return i < nsteps
  def body(carry):
    i, state = carry
    new_state = core.eval_jaxpr(lowered_jaxpr, consts, i, *state)
    return i + 1, new_state
  _, state = jax.lax.while_loop(cond, body, (0, [*args]))
  return state
mlir.register_lowering(for_p, mlir.lower_fun(_for_impl, multiple_results=True))

def _for_jvp(primals, tangents, *, jaxpr, nsteps):
  tangents = map(ad.instantiate_zeros, tangents)  # TODO handle symbolic zero
  jaxpr_ = core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(jaxpr_, [False] + [True] * len(tangents), True)
  jvp_jaxpr, jvp_consts = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts
  out_flat = for_p.bind(*jvp_consts, *primals, *tangents, jaxpr=jvp_jaxpr,
                        nsteps=nsteps)
  return split_list(out_flat, [len(out_flat) // 2])
ad.primitive_jvps[for_p] = _for_jvp

#

def f():
  def body(i, ref):
    ref[i] += i
  return for_loop(5, body, jnp.array([3, 1, 4, 1, 5]))

prnt(jax.make_jaxpr(f)())
print(f())


def f(x):
  def body(i, ref):
    x = ref[i]
    ref[i] = x
    ref[i] = (ref[i] + x) / 2.
  return for_loop(1, body, jnp.array([x]))

prnt(jax.make_jaxpr(f)(3.))
print(f(3.)[0])
print(jax.jvp(f, (3.,), (1.,)))
# print(jax.grad(f)(3.))


# TODO loop partial eval
# TODO loop transpose
# TODO loop batching
# TODO fixpoints, need jvp_jaxpr with extra state-is-differentiated input/output
# TODO nested scans leaving something on the table? how could we nest these
# loops? may need 'heap tags'. could statically give each for an id, and mention
# it in the reference. that doesn't work for standalone functions. maybe can be
# python trace time static, i.e. static in jaxprs. so inner loop can have a
# state effect, like
#  for_loop : Int -> (Int -> Ref h s -> {State h s, effs} ()) -> s
#             -> {effs} s
# whereas if we are okay with nesting being inefficient for now
#   for_loop : Int -> (Int -> Ref h s -> {State h s} ()) -> s -> s
# (may need to require out-of-line functions be pure, at least for now, once we
# have out-of-line functions)
# OR maybe not actually leaving anything on the table...

