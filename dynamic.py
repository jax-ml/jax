from functools import partial

from jax import core
from jax.interpreters import partial_eval as pe
from jax.interpreters import ad
from jax import linear_util as lu
from jax.util import safe_map
from jax.abstract_arrays import raise_to_shaped
from jax.api_util import flatten_fun
from jax.tree_util import tree_flatten, tree_unflatten

map = safe_map


dynamic_call_p = core.Primitive('dynamic_call')
dynamic_call = partial(core.call_bind, dynamic_call_p)
dynamic_call_p.def_custom_bind(dynamic_call)
dynamic_call_p.multiple_results = True

ad.primitive_transposes[dynamic_call_p] = partial(ad.call_transpose, dynamic_call_p)

def dynamic_call_impl(fun, *args):
  abstract_args = [raise_to_shaped(core.get_aval(x)) for x in args]
  pvals = [pe.PartialVal((aval, core.unit)) for aval in abstract_args]
  with core.new_master(pe.JaxprTrace, True) as master:
    trace = pe.JaxprTrace(master, core.cur_sublevel())
    in_tracers = map(trace.new_arg, pvals)
    with core.new_dynamic_trace(trace):
      ans = fun.call_wrapped(*in_tracers)
    out_tracers = [trace.instantiate_const(trace.full_raise(core.full_lower(x)))
                   for x in ans]
    jaxpr, consts, env = pe.tracers_to_jaxpr(in_tracers, out_tracers)
    assert not env
    del trace, master, out_tracers, env
  print(jaxpr)
  return core.eval_jaxpr(jaxpr, consts, (), *args)
dynamic_call_p.def_impl(dynamic_call_impl)

def dynamic(f):
  def wrapped(*args, **kwargs):
    fun = lu.wrap_init(f)
    args_flat, in_tree = tree_flatten((args, kwargs))
    flat_fun, out_tree = flatten_fun(fun, in_tree)
    out = dynamic_call(flat_fun, *args_flat)
    return tree_unflatten(out_tree(), out)
  return wrapped

# What to do with the jaxpr? Here are two ideas:
#  1. dynamic_jit: basically act like jit does
#  2. LazyJaxprArray: produce a lazy array that can stage our little jaxpr as a
#     widget into other jaxprs (think e.g. cond)

###

from jax import lax
from jax import jvp, grad


@dynamic
def thunk():
  return lax.add(1, 2)

out = thunk()
print(out)

# { lambda a b ;  ; .
#   let c = add a b
#   in [c] }
#
# 3



def f(x):
  @dynamic
  def g():
    return lax.mul(x, 2.)
  return g()
out = jvp(f, (3.,), (1.,))
print(out)

# { lambda a b d e ;  ; .
#   let c = mul a b
#       f = mul d e
#   in [c, f] }
#
# (DeviceArray(6., dtype=float32), DeviceArray(2., dtype=float32))


out = grad(f)(3.)
print(out)

# { lambda a b ;  ; .
#   let c = mul a b
#   in [c, *] }
#
# { lambda  ;  ; a.
#   let b = mul a 2.0
#   in [b] }
#
# 2.0


@dynamic
def thunk1():
  @dynamic
  def thunk2():
    return lax.add(1, 2)
  return thunk2()

out = thunk1()
print(out)

# { lambda  ;  ; .
#   let
#   in [*, 1, 2] }
#
# { lambda a b ;  ; .
#   let c = dynamic_call
#         { lambda a b ;  ; .
#           let c = add a b
#           in [c] } [ a b ;  ]
#   in [c] }
#
# { lambda a b ;  ; .
#   let c = add a b
#   in [c] }
#
# 3


def f(x):
  @dynamic
  def thunk1():
    @dynamic
    def thunk2():
      return lax.mul(x, 5.)
    return thunk2()
  return thunk1()

out = grad(f)(1.)
print(out)

# { lambda  ;  ; .
#   let
#   in [*, *, 1.0, 5.0] }
# 
# { lambda a b ;  ; .
#   let c d = dynamic_call
#         { lambda a b ;  ; .
#           let c = mul a b
#           in [c, *] } [ a b ;  ]
#   in [c, *] }
# 
# { lambda a b ;  ; .
#   let c = mul a b
#   in [c, *] }
# 
# { lambda  ;  ; a.
#   let
#   in [*] }
# 
# { lambda  ;  ; a.
#   let b = dynamic_call a
#         { lambda  ;  ; a.
#           let b = mul a 5.0
#           in [b] } [  ;  ]
#   in [b] }
# 
# { lambda  ;  ; a.
#   let b = mul a 5.0
#   in [b] }
# 
# 5.0


def f(x):
  @dynamic
  def g():
    return lax.sin(lax.mul(lax.sin(x), 2.))
  return g()
primal, tangent = jvp(f, (3.,), (1.,))
print(tangent)
print(grad(lambda x: lax.sin(lax.mul(lax.sin(x), 2.)))(3.))
