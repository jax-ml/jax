from collections import defaultdict, deque
import itertools as it

from jax.interpreters import partial_eval as pe
from jax.interpreters import ad
from jax.abstract_arrays import raise_to_shaped
from jax.core import get_aval
from jax.api_util import flatten_fun
from jax.util import safe_map, safe_zip, unzip2
from jax import linear_util as lu
from jax.tree_util import tree_flatten, tree_unflatten
from jax import core
from jax import vjp

from jax import lax

map = safe_map
zip = safe_zip


def trace_to_jaxpr(f, *args, **kwargs):
  args, in_tree = tree_flatten((args, kwargs))
  f, out_tree = flatten_fun(lu.wrap_init(f), in_tree)
  in_avals = [raise_to_shaped(get_aval(x)) for x in args]
  in_pvals = [pe.PartialVal((aval, core.unit)) for aval in in_avals]
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr(f, in_pvals, instantiate=True)
  out_avals, _ = unzip2(out_pvals)
  return core.TypedJaxpr(jaxpr, consts, in_avals, out_avals)

class Unknown(object):
  def __repr__(self): return '?'
unknown = Unknown()

class Deleted(object):
  def __repr__(self): return '-'
deleted = Deleted()

def rvjp(typed_jaxpr, out_primals, out_cotangents):
  jaxpr = typed_jaxpr.jaxpr

  def read_primal(v):
    if type(v) is core.Literal:
      return v.val
    else:
      result = primal_env.get(v, unknown)
      assert result is not deleted
      return result

  def write_primal(v, val):
    assert val is not deleted
    if val is not unknown:
      primal_env[v] = val

  def write_cotangent(v, ct):
    assert ct is not deleted
    if ct is not None:
      ct_env[v] = ad.add_tangents(ct_env[v], ct) if v in ct_env else ct

  def read_cotangent(v):
    result = ct_env.get(v, ad.zero)
    assert result is not deleted
    return result

  def neighbors(v):
    return [eqns_by_id[i] for i in neighbors_[v]]

  neighbors_ = defaultdict(set)
  eqns_by_id = {}
  for eqn in jaxpr.eqns:
    eqns_by_id[id(eqn)] = eqn
    for v in it.chain(eqn.invars, eqn.outvars):
      neighbors_[v].add(id(eqn))

  ct_env = {}
  map(write_cotangent, jaxpr.outvars, out_cotangents)

  primal_env = {}
  map(write_primal, jaxpr.constvars, typed_jaxpr.literals)
  map(write_primal, jaxpr.outvars, out_primals)

  out_eqns = it.chain.from_iterable(neighbors(v) for v in jaxpr.outvars)
  queue = deque({id(e): e for e in out_eqns}.values())  # de-duplicate

  done_eqns = set()
  while queue:
    print(map(id, queue), done_eqns)
    eqn = queue.popleft()
    assert id(eqn) not in done_eqns
    invals = map(read_primal, eqn.invars)
    outvals = map(read_primal, eqn.outvars)
    rule = inverses[eqn.primitive]
    new_invals, new_outvals = rule(invals, outvals, **eqn.params)
    map(write_primal, eqn.invars, new_invals)
    map(write_primal, eqn.outvars, new_outvals)

    all_vars = list(it.chain(eqn.invars, eqn.outvars))
    old_vals = list(it.chain(invals, outvals))
    new_vals = list(it.chain(new_invals, new_outvals))
    updated_vars = [var for var, old_val, new_val in zip(all_vars, old_vals, new_vals)
                    if old_val is unknown and new_val is not unknown]
    if updated_vars:
      if not any(val is unknown for val in new_vals):
        cts_in = map(read_cotangent, eqn.outvars)
        cts_out = primitive_vjp(eqn.primitive, cts_in, new_invals, new_outvals)
        map(write_cotangent, eqn.invars, cts_out)
        done_eqns.add(id(eqn))
      else:
        assert id(eqn) not in done_eqns
        queue.append(eqn)
      for v in updated_vars:
        queue.extendleft(e for e in neighbors(v) if e is not eqn
                         and e not in queue and id(e) not in done_eqns)
    else:
      assert id(eqn) not in done_eqns
      queue.append(eqn)

  return map(read_primal, jaxpr.invars), map(read_cotangent, jaxpr.invars)


def primitive_vjp(prim, cts_in, invals, outvals):
  if prim is lax.add_p:
    y_ct, = cts_in
    return [y_ct, y_ct]
  elif prim is lax.exp_p:
    y, = outvals
    y_ct, = cts_in
    return [y * y_ct]
  else:
    assert False


inverses = {}

def add_inverse(invals, outvals):
  x1, x2 = invals
  y, = outvals
  if y is not unknown:
    if x1 is unknown and x2 is not unknown:
      return [y - x2, x2], [y]
    elif x2 is unknown and x1 is not unknown:
      return [x1, y - x1], [y]
    else:
      return invals, outvals
  elif x1 is not unknown and x2 is not unknown:
    return [x1, x2], [x1 + x2]
  else:
    return invals, outvals
inverses[lax.add_p] = add_inverse

def exp_inverse(invals, outvals):
  x, = invals
  y, = outvals
  assert (x is unknown) ^ (y is unknown)
  if x is unknown:
    return [np.log(y)], [y]
  else:
    return [x], [np.exp(x)]
inverses[lax.exp_p] = exp_inverse


###

import jax.numpy as np

def f(x, y):
  return np.exp(x), x + y
print(f(2., 3.))

jaxpr = trace_to_jaxpr(f, 2., 3.)
print(jaxpr)
# { lambda  ;  ; a b.
#   let c = exp a
#       d = add a b
#   in [c, d] }


in_primals, in_cotangents = rvjp(jaxpr, (7.389, 5.0), (1., 1.))
print(in_primals)
print(in_cotangents)

_, f_vjp = vjp(f, 2., 3.)
print(f_vjp((1., 1.)))


def g(x, y):
  x, y = f(x, y)
  x, y = f(x, y)
  return x, y

print(g(2., 3.))

jaxpr = trace_to_jaxpr(g, 2., 3.)
print(jaxpr)

in_primals, in_cotangents = rvjp(jaxpr, (1618.1782, 12.39), (1., 1.))
print(in_primals)
print(in_cotangents)
