# Copyright 2020 Google LLC
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


from functools import partial

import numpy as onp

import jax
from jax import core
from jax.util import unzip2
from jax import ad_util
from jax.tree_util import (register_pytree_node, tree_structure,
                           treedef_is_leaf, tree_flatten, tree_unflatten, tree_map)
import jax.linear_util as lu
from jax.interpreters import xla
from jax.lax import lax
from jax.lax import lax_fft

def jet(fun, primals, series):
  try:
    order, = set(map(len, series))
  except ValueError:
    msg = "jet terms have inconsistent lengths for different arguments"
    raise ValueError(msg) from None

  # TODO(mattjj): consider supporting pytree inputs
  for i, (x, terms) in enumerate(zip(primals, series)):
    treedef = tree_structure(x)
    if not treedef_is_leaf(treedef):
      raise ValueError("primal value at position {} is not an array".format(i))
    for j, t in enumerate(terms):
      treedef = tree_structure(t)
      if not treedef_is_leaf(treedef):
        raise ValueError("term {} for argument {} is not an array".format(j, i))

  @lu.transformation_with_aux
  def flatten_fun_output(*args):
    ans = yield args, {}
    yield tree_flatten(ans)

  f, out_tree = flatten_fun_output(lu.wrap_init(fun))
  out_primals, out_terms = jet_fun(jet_subtrace(f), order).call_wrapped(primals, series)
  return tree_unflatten(out_tree(), out_primals), tree_unflatten(out_tree(), out_terms)

@lu.transformation
def jet_fun(order, primals, series):
  with core.new_master(JetTrace) as master:
    master.order = order
    out_primals, out_terms = yield (master, primals, series), {}
    del master
  out_terms = [[onp.zeros_like(p)] * order if s is zero_series else s
               for p, s in zip(out_primals, out_terms)]
  yield out_primals, out_terms

@lu.transformation
def jet_subtrace(master, primals, series):
  trace = JetTrace(master, core.cur_sublevel())
  in_tracers = map(partial(JetTracer, trace), primals, series)
  ans = yield in_tracers, {}
  out_tracers = map(trace.full_raise, ans)
  out_primals, out_terms = unzip2((t.primal, t.terms) for t in out_tracers)
  yield out_primals, out_terms

@lu.transformation_with_aux
def traceable(in_tree_def, *primals_and_series):
  primals_in, series_in = tree_unflatten(in_tree_def, primals_and_series)
  primals_out, series_out = yield (primals_in, series_in), {}
  out_flat, out_tree_def = tree_flatten((primals_out, series_out))
  yield out_flat, out_tree_def


class JetTracer(core.Tracer):
  __slots__ = ["primal", "terms"]

  def __init__(self, trace, primal, terms):
    assert type(terms) in (ZeroSeries, list, tuple)
    self._trace = trace
    self.primal = primal
    self.terms = terms

  @property
  def aval(self):
    return core.get_aval(self.primal)

  def full_lower(self):
    if self.terms is zero_series or all(t is zero_term for t in self.terms):
      return core.full_lower(self.primal)
    else:
      return self

class JetTrace(core.Trace):

  def pure(self, val):
    return JetTracer(self, val, zero_series)

  def lift(self, val):
    return JetTracer(self, val, zero_series)

  def sublift(self, val):
    return JetTracer(self, val.primal, val.terms)

  def process_primitive(self, primitive, tracers, params):
    assert not primitive.multiple_results  # TODO
    order = self.master.order              # pytype: disable=attribute-error
    primals_in, series_in = unzip2((t.primal, t.terms) for t in tracers)
    series_in = [[zero_term] * order if s is zero_series else s
                 for s in series_in]
    # TODO(mattjj): avoid always instantiating zeros
    series_in = [[onp.zeros(onp.shape(x), dtype=onp.result_type(x))
                  if t is zero_term else t for t in series]
                 for x, series in zip(primals_in, series_in)]
    rule = jet_rules[primitive]
    primal_out, terms_out = rule(primals_in, series_in, **params)
    return JetTracer(self, primal_out, terms_out)

  def process_call(self, call_primitive, f, tracers, params):
    primals_in, series_in = unzip2((t.primal, t.terms) for t in tracers)
    primals_and_series, in_tree_def = tree_flatten((primals_in, series_in))
    f_jet, out_tree_def = traceable(jet_subtrace(f, self.master), in_tree_def)
    result = call_primitive.bind(f_jet, *primals_and_series, **params)
    primals_out, series_out = tree_unflatten(out_tree_def(), result)
    return [JetTracer(self, p, ts) for p, ts in zip(primals_out, series_out)]

  def post_process_call(self, call_primitive, out_tracers, params):
    primals, series = unzip2((t.primal, t.terms) for t in out_tracers)
    out, treedef = tree_flatten((primals, series))
    del primals, series
    master = self.master
    def todo(x):
      primals, series = tree_unflatten(treedef, x)
      trace = JetTrace(master, core.cur_sublevel())
      return map(partial(JetTracer, trace), primals, series)
    return out, todo

  def join(self, xt, yt):
    assert False  # TODO?


class ZeroTerm(object): pass
zero_term = ZeroTerm()
register_pytree_node(ZeroTerm, lambda z: ((), None), lambda _, xs: zero_term)

class ZeroSeries(object): pass
zero_series = ZeroSeries()
register_pytree_node(ZeroSeries, lambda z: ((), None), lambda _, xs: zero_series)


### rule definitions

jet_rules = {}

def defzero(prim):
  jet_rules[prim] = partial(zero_prop, prim)

def zero_prop(prim, primals_in, series_in, **params):
  primal_out = prim.bind(*primals_in, **params)
  return primal_out, zero_series

defzero(lax.le_p)
defzero(lax.lt_p)
defzero(lax.gt_p)
defzero(lax.ge_p)
defzero(lax.eq_p)
defzero(lax.ne_p)
defzero(lax.and_p)
defzero(lax.or_p)
defzero(lax.xor_p)
defzero(lax.floor_p)
defzero(lax.ceil_p)
defzero(lax.round_p)
defzero(lax.sign_p)
defzero(ad_util.stop_gradient_p)


def deflinear(prim):
  jet_rules[prim] = partial(linear_prop, prim)

def linear_prop(prim, primals_in, series_in, **params):
  primal_out = prim.bind(*primals_in, **params)
  series_out = [prim.bind(*terms_in, **params) for terms_in in zip(*series_in)]
  return primal_out, series_out

deflinear(lax.neg_p)
deflinear(lax.real_p)
deflinear(lax.complex_p)
deflinear(lax.add_p)
deflinear(lax.sub_p)
deflinear(lax.convert_element_type_p)
deflinear(lax.broadcast_p)
deflinear(lax.broadcast_in_dim_p)
deflinear(lax.concatenate_p)
deflinear(lax.pad_p)
deflinear(lax.reshape_p)
deflinear(lax.rev_p)
deflinear(lax.transpose_p)
deflinear(lax.slice_p)
deflinear(lax.reduce_sum_p)
deflinear(lax.reduce_window_sum_p)
deflinear(lax.tie_in_p)
deflinear(lax_fft.fft_p)
deflinear(xla.device_put_p)

def def_deriv(prim, deriv):
  """
  Define the jet rule for a primitive in terms of its first derivative.
  """
  jet_rules[prim] = partial(deriv_prop, prim, deriv)

def deriv_prop(prim, deriv, primals_in, series_in):
  x, = primals_in
  series, = series_in
  primal_out = prim.bind(x)
  c0, cs = jet(deriv, primals_in, series_in)
  c = [c0] + cs
  u = [x] + series
  v = [primal_out] + [None] * len(series)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum(_scale(k, j) * c[k-j] * u[j] for j in range(1, k + 1))
  primal_out, *series_out = v
  return primal_out, series_out


def_deriv(lax.erf_p, lambda x: lax.mul(lax._const(x, 2. / onp.sqrt(onp.pi)), lax.exp(lax.neg(lax.square(x)))))

def def_comp(prim, comp):
  """
  Define the jet rule for a primitive in terms of a composition of simpler primitives.
  """
  jet_rules[prim] = partial(jet, comp)


def_comp(lax.erfc_p, lambda x: 1 - lax.erf(x))

### More complicated rules

def fact(n):
  return lax.exp(lax.lgamma(n+1.))

def _scale(k, j):
  return 1. / (fact(k - j) * fact(j - 1))

def _scale2(k, j):
  return 1. / (fact(k - j) * fact(j))

def _exp_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  v = [lax.exp(x)] + [None] * len(series)
  for k in range(1,len(v)):
    v[k] = fact(k-1) * sum([_scale(k, j)* v[k-j] * u[j] for j in range(1, k+1)])
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.exp_p] = _exp_taylor

def _expm1_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  v = [lax.exp(x)] + [None] * len(series)
  for k in range(1,len(v)):
    v[k] = fact(k-1) * sum([_scale(k, j)* v[k-j] * u[j] for j in range(1, k+1)])
  primal_out, *series_out = v
  return lax.expm1(x), series_out
jet_rules[lax.expm1_p] = _expm1_taylor

def _pow_taylor(primals_in, series_in):
  u_, r_ = primals_in

  x, series = jet(lambda x, y: lax.mul(y, lax.log(x)), primals_in, series_in)

  u = [x] + series
  v = [u_ ** r_] + [None] * len(series)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum([_scale(k, j)* v[k-j] * u[j] for j in range(1, k+1)])
  primal_out, *series_out = v

  return primal_out, series_out
jet_rules[lax.pow_p] = _pow_taylor

def _integer_pow_taylor(primals_in, series_in, *, y):
  if y == 2:
    fn = lambda x: x * x
  else:
    fn = lambda x: lax.pow(x, onp.array(y, dtype=x.dtype))
  return jet(fn, primals_in, series_in)
jet_rules[lax.integer_pow_p] = _integer_pow_taylor


def _expit_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  v = [jax.scipy.special.expit(x)] + [None] * len(series)
  e = [v[0] * (1 - v[0])] + [None] * len(series)  # terms for sigmoid' = sigmoid * (1 - sigmoid)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum([_scale(k, j) * e[k-j] * u[j] for j in range(1, k+1)])
    e[k] = (1 - v[0]) * v[k] - fact(k) * sum([_scale2(k, j)* v[j] * v[k-j] for j in range(1, k+1)])

  primal_out, *series_out = v
  return primal_out, series_out

def _tanh_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [2*x] + [2 * series_ for series_ in series]
  primals_in, *series_in = u
  primal_out, series_out = _expit_taylor((primals_in, ), (series_in, ))
  series_out = [2 * series_ for series_ in series_out]
  return 2 * primal_out - 1, series_out
jet_rules[lax.tanh_p] = _tanh_taylor

def _log_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  v = [lax.log(x)] + [None] * len(series)
  for k in range(1, len(v)):
    conv = sum([_scale(k, j) * v[j] * u[k-j] for j in range(1, k)])
    v[k] = (u[k] - fact(k - 1) * conv) / u[0]
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.log_p] = _log_taylor

def _sqrt_taylor(primals_in, series_in):
  return jet(lambda x: x ** 0.5, primals_in, series_in)
jet_rules[lax.sqrt_p] = _sqrt_taylor

def _rsqrt_taylor(primals_in, series_in):
  return jet(lambda x: x ** -0.5, primals_in, series_in)
jet_rules[lax.rsqrt_p] = _rsqrt_taylor

def _asinh_taylor(primals_in, series_in):
  return jet(lambda x: lax.log(x + lax.sqrt(lax.square(x) + 1)), primals_in, series_in)
jet_rules[lax.asinh_p] = _asinh_taylor

def _acosh_taylor(primals_in, series_in):
  return jet(lambda x: lax.log(x + lax.sqrt(lax.square(x) - 1)), primals_in, series_in)
jet_rules[lax.acosh_p] = _acosh_taylor

def _atanh_taylor(primals_in, series_in):
  return jet(lambda x: 0.5 * lax.log(lax.div(1 + x, 1 - x)), primals_in, series_in)
jet_rules[lax.atanh_p] = _atanh_taylor

def _atan2_taylor(primals_in, series_in):
  x, y = primals_in
  primal_out = lax.atan2(x, y)

  x, series = jet(lax.div, primals_in, series_in)
  c0, cs = jet(lambda x: lax.div(1, 1 + lax.square(x)), (x, ), (series, ))
  c = [c0] + cs
  u = [x] + series
  v = [primal_out] + [None] * len(series)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum(_scale(k, j) * c[k-j] * u[j] for j in range(1, k + 1))
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.atan2_p] = _atan2_taylor

def _log1p_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x + 1] + series
  v = [lax.log(x + 1)] + [None] * len(series)
  for k in range(1, len(v)):
    conv = sum([_scale(k, j) * v[j] * u[k-j] for j in range(1, k)])
    v[k] = (u[k] - fact(k - 1) * conv) / u[0]
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.log1p_p] = _log1p_taylor

def _div_taylor_rule(primals_in, series_in, **params):
  x, y = primals_in
  x_terms, y_terms = series_in
  u = [x] + x_terms
  w = [y] + y_terms
  v = [None] * len(u)
  def scale(k, j): return 1. / (fact(k - j) * fact(j))
  for k in range(0, len(v)):
    conv = sum([scale(k, j) * v[j] * w[k-j] for j in range(0, k)])
    v[k] = (u[k] - fact(k) * conv) / w[0]
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.div_p] = _div_taylor_rule

def _sinusoidal_rule(sign, prims, primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  s, c = prims
  s = [s(x)] + [None] * len(series)
  c = [c(x)] + [None] * len(series)
  for k in range(1, len(s)):
    s[k] = fact(k-1) * sum(_scale(k, j) * u[j] * c[k-j] for j in range(1, k + 1))
    c[k] = fact(k-1) * sum(_scale(k, j) * u[j] * s[k-j] for j in range(1, k + 1)) * sign
  return (s[0], s[1:]), (c[0], c[1:])

def _get_ind(f, ind):
  return lambda *args: f(*args)[ind]

jet_rules[lax.sin_p] = _get_ind(partial(_sinusoidal_rule, -1, (lax.sin, lax.cos)), 0)
jet_rules[lax.cos_p] = _get_ind(partial(_sinusoidal_rule, -1, (lax.sin, lax.cos)), 1)
jet_rules[lax.sinh_p] = _get_ind(partial(_sinusoidal_rule, 1, (lax.sinh, lax.cosh)), 0)
jet_rules[lax.cosh_p] = _get_ind(partial(_sinusoidal_rule, 1, (lax.sinh, lax.cosh)), 1)

def _bilinear_taylor_rule(prim, primals_in, series_in, **params):
  x, y = primals_in
  x_terms, y_terms = series_in
  u = [x] + x_terms
  w = [y] + y_terms
  v = [None] * len(u)
  op = partial(prim.bind, **params)
  def scale(k, j): return 1. / (fact(k - j) * fact(j))
  for k in range(0, len(v)):
    v[k] = fact(k) * sum([scale(k, j) * op(u[j], w[k-j]) for j in range(0, k+1)])
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.dot_general_p] = partial(_bilinear_taylor_rule, lax.dot_general_p)
jet_rules[lax.mul_p] = partial(_bilinear_taylor_rule, lax.mul_p)
jet_rules[lax.conv_general_dilated_p] = partial(_bilinear_taylor_rule, lax.conv_general_dilated_p)

def _gather_taylor_rule(primals_in, series_in, **params):
  operand, start_indices = primals_in
  gs, _ = series_in
  primal_out = lax.gather_p.bind(operand, start_indices, **params)
  series_out = [lax.gather_p.bind(g, start_indices, **params) for g in gs]
  return primal_out, series_out
jet_rules[lax.gather_p] = _gather_taylor_rule

def _gen_reduce_choose_taylor_rule(chooser_fun):
  def chooser_taylor_rule(primals_in, series_in, **params):
    operand, = primals_in
    gs, = series_in
    primal_out = chooser_fun(operand, **params)
    axes = params.pop("axes", None)
    primal_dtype = gs[0].dtype
    shape = [1 if i in axes else d for i, d in enumerate(operand.shape)]
    location_indicators = lax.convert_element_type(
          lax._eq_meet(operand, lax.reshape(primal_out, shape)), primal_dtype)
    counts = lax._reduce_sum(location_indicators, axes)
    def _reduce_chooser_taylor_rule(g):
      return lax.div(lax._reduce_sum(lax.mul(g, location_indicators), axes), counts)
    series_out = [_reduce_chooser_taylor_rule(g) for g in gs]
    return primal_out, series_out
  return chooser_taylor_rule
jet_rules[lax.reduce_max_p] = _gen_reduce_choose_taylor_rule(lax.reduce_max_p.bind)
jet_rules[lax.reduce_min_p] = _gen_reduce_choose_taylor_rule(lax.reduce_min_p.bind)

def _abs_taylor_rule(x, series_in, **params):
  x, = x
  primal_out = lax.abs_p.bind(x, **params)
  negs = lax.select(lax.lt(x, 0.0), lax.full_like(x, -1), lax.full_like(x, 1.0))
  fix_sign = lambda y: negs * y
  series_out = [fix_sign(*terms_in, **params) for terms_in in zip(*series_in)]
  return primal_out, series_out
jet_rules[lax.abs_p] = _abs_taylor_rule

def _select_taylor_rule(primal_in, series_in, **params):
  b, x, y = primal_in
  primal_out = lax.select_p.bind(b, x, y, **params)
  sel = lambda _, x, y: lax.select(b, x, y)
  series_out = [sel(*terms_in, **params) for terms_in in zip(*series_in)]
  return primal_out, series_out
jet_rules[lax.select_p] = _select_taylor_rule
