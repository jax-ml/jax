from collections import namedtuple
from functools import partial
import numpy.random as npr
import jax.numpy as np
from jax import jit, jvp
import itertools as it
import sys

npr.seed(0)

from jax.util import unzip2, safe_zip, safe_map

map = safe_map
zip = safe_zip

subfun_prob = 0.5
thin_prob = 0.1
size_reduction_factor = 3

Eqn = namedtuple('Eqn', ['in_vars', 'out_vars', 'fun'])
Prim = namedtuple('Prim', ['fun'])
ArrayType = namedtuple('ArrayType', ['shape', 'dtype'])
Var = namedtuple('Var', ['name', 'vartype'])
Fun = namedtuple('Fun', ['in_vars', 'out_vars', 'eqns'])

def gen_fun_and_types(size):
  in_types = [gen_array_type(size) for _ in range(gen_nonneg_int(size))]
  fun, _ = gen_function(size, in_types)
  return fun

def gen_function(size, in_types):
  eqns = []
  in_vars = map(fresh_var, in_types)
  cur_vars = in_vars[:]
  for _ in range(gen_nonneg_int(size)):
    if not cur_vars:
      break
    if npr.rand() < subfun_prob:
      arg_vars = gen_subset(cur_vars)
      arg_types = [v.vartype for v in arg_vars]
      fun, out_types = gen_function(size / size_reduction_factor, arg_types)
      fun = partial(eval_fun, fun)
    else:
      arity = choice(primitive_generators.keys())
      arg_vars = gen_sized_subset(cur_vars, arity)
      arg_types = [v.vartype for v in arg_vars]
      prim_gen = weighted_choice(primitive_generators[arity])
      fun, out_type = prim_gen(size, *arg_types)
      fun = wrap_singleton(fun)
      out_types = [out_type]

    out_vars = map(fresh_var, out_types)
    eqns.append(Eqn(arg_vars, out_vars, fun))
    cur_vars.extend(out_vars)
    cur_vars = thin(cur_vars, thin_prob)

  out_vars = gen_subset(cur_vars)
  return Fun(in_vars, out_vars, eqns), [v.vartype for v in out_vars]

def eval_fun(fun, *args):
  def read(v):
    return env[v]
  def write(v, x):
    env[v] = x

  env = {}
  map(write, fun.in_vars, args)
  for in_vars, out_vars, f in fun.eqns:
    out_vals = f(*map(read, in_vars))
    map(write, out_vars, out_vals)

  return map(read, fun.out_vars)

counter = it.count()
def fresh_var(ty):
  return Var(counter.next(), ty)

def gen_array_type(size):
  # TODO(dougalm): randomize this
  return ArrayType((2,2), np.float32)

def gen_array_val(array_type):
  # TODO(dougalm): different sizes and dtypes
  return npr.randn(*array_type.shape)

def gen_neg(size, t):
  return (lambda x: -x), t

def gen_trig(size, t):
  op = choice([np.sin, np.cos])
  return op, t

def gen_binop(size, t1, t2):
  unifier, t_out = gen_broadcasting_unifier(t1, t2)
  binop = choice([lambda x, y: x + y,
                  lambda x, y: x * y])
  def unify_and_binop(x, y):
    x_, y_ = unifier(x, y)
    return binop(x_, y_)

  return unify_and_binop, t_out

def thin(xs, p):
  return [x for x in xs if npr.rand() > p]

def gen_broadcasting_unifier(t1, t2):
  assert t1.shape == t2.shape
  return lambda x, y: (x,y), t1
  # TODO: generate slices and paddings to match shapes

def wrap_singleton(f):
  return lambda *xs: (f(*xs),)

unary_primitive_generators = [
  (3, gen_trig),
  (1, gen_neg) ]

binary_primitive_generators = [
  (1, gen_binop)]

primitive_generators = { 1: unary_primitive_generators,
                         2: binary_primitive_generators }

def gen_nonneg_int(size):
  return npr.randint(size)

choice = npr.choice

def weighted_choice(weighted_choices):
  weights, choices = unzip2(weighted_choices)
  return npr_choice(choices, weights)

def npr_choice(xs, weights=None):
  # npr.choice isn't actually RS -> [a] -> a
  # because it inspects the components to see if they're array-like
  assert xs
  n = len(xs)
  if weights is None:
    i = npr.randint(n)
  else:
    normalizer = float(sum(weights))
    weights = [w / normalizer for w in weights]
    i = npr.choice(range(n), p=weights)
  return xs[i]

def gen_sized_subset(xs, size):
  return [npr_choice(xs) for _ in range(size)]

def gen_subset(xs):
  if not xs:
    return []

  return gen_sized_subset(xs, npr.randint(len(xs) + 1))

def gen_inputs(fun):
  return [gen_array_val(v.vartype) for v in fun.in_vars]

def jvp_fd(fun, args, directions):
  EPS = 1e-4
  def eval_eps(eps):
    return fun(*[x if d is None else x + eps * d
                 for x, d in zip(args, directions)])

  ys_neg = eval_eps(-EPS)
  ys_pos = eval_eps(EPS)
  ys = eval_eps(0.0)
  deriv = [(y_pos - y_neg) / (2 * EPS) for y_neg, y_pos in zip(ys_neg, ys_pos)]
  return ys, deriv

def check_all_close(xs, ys, tol=1e-3):
  for x, y in zip(xs, ys):
    assert x.shape == y.shape
    # TODO(dougalm): re-enable once we've tackled the less pendantic bugs
    # assert x.dtype == y.dtype
    assert np.allclose(x, y, rtol=tol, atol=tol), \
       "Value mismatch:\n{}\n  vs\n{}\n".format(x, y)

def jit_is_identity(fun):
  vals = gen_inputs(fun)
  fun = partial(eval_fun, fun)
  ans = fun(*vals)
  static_argnums = thin(range(len(vals)), 0.5)
  ans_jitted = jit(fun, static_argnums=static_argnums)(*vals)
  check_all_close(ans, ans_jitted)

def jvp_matches_fd(fun):
  vals = gen_inputs(fun)
  directions = gen_inputs(fun)
  fun = partial(eval_fun, fun)

  # TODO: differentiate wrt some inputs only
  ans1, deriv1 = jvp_fd(fun, vals, directions)
  ans2, deriv2 = jvp(fun, vals, directions)
  check_all_close(ans1, ans2)
  check_all_close(deriv1, deriv2)

properties = [
  jit_is_identity,
  jvp_matches_fd,
]
# vjp_matches_fd,
# vmap_matches_map ]

def run_tests():
  sizes = [3, 10]
  num_examples = 50
  cases = it.product(sizes, range(num_examples), properties)
  for i, (size, _, check_prop) in enumerate(cases):
    sys.stderr.write('\rTested: {}'.format(i))
    check_prop(gen_fun_and_types(size))

if __name__ == "__main__":
  run_tests()
