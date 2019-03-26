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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import jax.core as core
import jax.linear_util as lu
import jax.numpy as np
import jax.lax as lax

from jax.lax import _abstractify
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
from jax.interpreters import ad

import jax.util as ju

map = ju.safe_map

# scan :: (a -> c -> c) -> c -> [a] -> [c]
# scan_cc :: (d ->a -> c -> c) -> d-> c -> [a] -> [c]
#  pro: fewer types
#  pro: all types are passed as args

# scan :: (a -> c -> (b,c)) -> c -> [a] -> ([b],c)
# scan_cc :: (d -> a -> c -> (b,c)) -> d -> c -> [a] -> ([b],c)
#  pro: fold and for_i are special cases without needing DCE
#  pro: feels cleaner for transposition
#  pro: accumulation without saving intermediates

def scan_reference(f, init, xs):
  carry = init
  ys = []
  for x in xs:
    (y, carry) = f(x, carry)
    ys.append(y)

  ys = core.pack(map(np.stack, zip(*ys)))
  return ys, np.array(carry)

def demote_aval_rank(xs):
  if isinstance(xs, core.AbstractTuple):
    return core.AbstractTuple(map(demote_aval_rank, xs))
  else:
    return ShapedArray(xs.shape[1:], xs.dtype)

def promote_aval_rank(n, xs):
  if isinstance(xs, core.AbstractTuple):
    return core.AbstractTuple(map(partial(promote_aval_rank, n), xs))
  else:
    return ShapedArray((n,) + xs.shape, xs.dtype)

def leading_dim_size(xs):
  if isinstance(xs, core.JaxTuple):
    return leading_dim_size(xs[0])
  else:
    return xs.shape[0]

def empty_arrays(aval):
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(empty_arrays, aval))
  else:
    return lax.full(aval.shape, 0, aval.dtype)

def index_arrays(i, aval, xs):
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(partial(index_arrays, i), aval, xs))
  else:
    return lax.dynamic_index_in_dim(xs, i, keepdims=False)

def update_arrays(i, aval, xs, x):
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(partial(update_arrays, i), aval, xs, x))
  else:
    return lax.dynamic_update_index_in_dim(xs, x[None, ...], i, axis=0)

# scan :: (a -> c -> (b,c)) -> c -> [a] -> ([b],c)
def scan(f, init, xs):
  consts, avals, jaxpr = trace_scan_fun(f, init, xs)
  ys, carry = scan_p.bind(core.pack(consts), init, xs, avals=avals, jaxpr=jaxpr)
  return ys, carry

def trace_scan_fun(f, init, xs):
  f = lu.wrap_init(f)
  carry_pval = carry_aval, _ = _abstractify(init)
  xs_aval, _ = _abstractify(xs)
  x_aval = demote_aval_rank(xs_aval)
  x_pval = pe.PartialVal((x_aval, core.unit))
  jaxpr, pval_out, consts = pe.trace_to_jaxpr(f, (x_pval, carry_pval))
  (y_aval, carry_aval_out), _ = pval_out
  assert carry_aval == carry_aval_out
  avals = (x_aval, y_aval, carry_aval)
  return consts, avals, jaxpr

def _scan_impl(consts, init, xs, avals, jaxpr):
  length = leading_dim_size(xs)
  (x_aval, y_aval, carry_aval) = avals
  ys_aval = promote_aval_rank(length, y_aval)

  def body_fun(i, vals):
    carry, ys = vals
    x = index_arrays(i, x_aval, xs)
    (y, carry_out) = core.eval_jaxpr(jaxpr, consts, (), x, carry)
    ys_out = update_arrays(i, y_aval, ys, y)
    return (carry_out, ys_out)

  ys_init = empty_arrays(ys_aval)
  carry, out = lax.fori_loop(0, length, body_fun, (init, ys_init))
  return core.pack((out, carry))

# scan :: (a -> c -> (b,c)) -> c -> [a] -> ([b],c)
def _scan_jvp(primals, tangents, avals, jaxpr):
  consts, init, xs = primals
  consts_dot, init_dot, xs_dot = tangents
  f = partial(core.eval_jaxpr, jaxpr)

  # TODO: plumb symbolic zeros in and out of jvp transformation so we can test
  # that they're the same as the inputs and re-run if not
  consts_dot = ad.instantiate_zeros(consts, consts_dot)
  init_dot   = ad.instantiate_zeros(init  , init_dot)
  xs_dot     = ad.instantiate_zeros(xs    , xs_dot)

  f_jvp = ad.jvp(lu.wrap_init(f)).call_wrapped

  def f_jvp_c(carry_dual, x_dual):
    init, init_dot = carry_dual
    x, x_dot = x_dual
    ans = f_jvp(core.pack((consts    , core.unit, init    , x)),
                core.pack((consts_dot, core.unit, init_dot, x_dot)))
    (y, carry_out), (y_dot, carry_out_dot) = ans
    return core.pack((core.pack((y, y_dot)),
                      core.pack((carry_out, carry_out_dot))))

  consts_dual = core.pack((consts, consts_dot))
  init_dual   = core.pack((init  , init_dot))
  xs_dual     = core.pack((xs    , xs_dot))
  consts, avals, jvp_jaxpr = trace_scan_fun(f_jvp_c, init_dual, xs_dual)

  ans = scan_p.bind(core.pack(consts), init_dual, xs_dual,
                    avals=avals, jaxpr=jvp_jaxpr)
  (y, y_dot), (carry_out, carry_out_dot) = ans
  return core.pack((y, carry_out)), core.pack((y_dot, carry_out_dot))


scan_p = core.Primitive("scan")
scan_p.def_impl(_scan_impl)
ad.primitive_jvps[scan_p] = _scan_jvp
